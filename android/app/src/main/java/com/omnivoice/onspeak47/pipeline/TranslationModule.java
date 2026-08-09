/*
 * OmniVoice — Translation Module (NLLB-200 via ONNX Runtime)
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.icu.text.BreakIterator;
import android.util.Log;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.extensions.OrtxPackage;
import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.TensorUtils;

public class TranslationModule {

    private static final String TAG = "TranslationModule";

    private static final String ENCODER_FILE = "encoder_model_int8.onnx";
    private static final String DECODER_FILE = "decoder_model_merged_int8.onnx";
    private static final String VOCAB_FILE = "sentencepiece_bpe.model";

    private static final int MAX_OUTPUT_TOKENS = 256;

    private final OrtEnvironment env;
    private final OrtSession encoderSession;
    private final OrtSession decoderSession;
    private final Tokenizer tokenizer;
    private final Context context;

    private static final Map<String, String> NLLB_CODES = new HashMap<>();
    static {
        NLLB_CODES.put("vi", "vie_Latn");
        NLLB_CODES.put("en", "eng_Latn");
        NLLB_CODES.put("zh", "zho_Hans");
    }

    public TranslationModule(Context context) throws OrtException {
        this.context = context;
        env = OrtEnvironment.getEnvironment();

        // Use External Files Dir to avoid C: drive issues
        File baseDir = context.getExternalFilesDir(null);
        if (baseDir == null) {
            baseDir = context.getFilesDir();
        }

        String encoderPath = copyModel(context, baseDir, ENCODER_FILE);
        String decoderPath = copyModel(context, baseDir, DECODER_FILE);
        String vocabPath = copyModel(context, baseDir, VOCAB_FILE);

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        try {
            options.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
        } catch (OrtException e) {
            Log.e(TAG, "Extensions library not found", e);
        }
        
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        // Try to use NNAPI for hardware acceleration (NPU/GPU/DSP)
        try {
            options.addNnapi();
            Log.i(TAG, "NNAPI execution provider enabled");
        } catch (OrtException e) {
            Log.w(TAG, "NNAPI not available, using CPU fallback", e);
        }

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);

        tokenizer = new Tokenizer(context, vocabPath, VOCAB_FILE, Tokenizer.NLLB);
        if (!tokenizer.isLoaded()) {
            Log.e(TAG, "⚠ Tokenizer failed to load — all translations will fail! "
                    + "Make sure '" + VOCAB_FILE + "' is in android/app/src/main/assets/");
        }
        options.close();
    }

    public TranslationResult translate(String text, String srcLang, String tgtLang) {
        long startTime = System.currentTimeMillis();
        if (text == null || text.trim().isEmpty()) {
            return new TranslationResult("", 0);
        }
        String nllbSrc = getNllbCode(srcLang);
        String nllbTgt = getNllbCode(tgtLang);

        ArrayList<String> sentences = splitIntoSentences(text, srcLang);
        StringBuilder result = new StringBuilder();

        for (String sentence : sentences) {
            if (sentence.trim().isEmpty()) continue;
            String translated = translateSentence(sentence, nllbSrc, nllbTgt);
            if (result.length() > 0) result.append(" ");
            result.append(translated);
        }

        long elapsed = System.currentTimeMillis() - startTime;
        return new TranslationResult(result.toString(), elapsed);
    }

    private String translateSentence(String text, String nllbSrc, String nllbTgt) {
        OrtSession.Result encoderResult = null;
        try {
            Tokenizer.TokenizerResult input = tokenizer.tokenize(nllbSrc, nllbTgt, text);
            if (input == null) {
                Log.e(TAG, "Tokenization returned null — tokenizer.isLoaded()=" + tokenizer.isLoaded());
                return "[error: tokenization failed]";
            }

            OnnxTensor inputIds = TensorUtils.intArrayToTensor(env, input.inputIDs);
            OnnxTensor mask = TensorUtils.intArrayToTensor(env, input.attentionMask);
            Map<String, OnnxTensor> encoderInputs = new HashMap<>();
            encoderInputs.put("input_ids", inputIds);
            encoderInputs.put("attention_mask", mask);

            encoderResult = encoderSession.run(encoderInputs);
            inputIds.close();

            OnnxTensor encoderOutput = (OnnxTensor) encoderResult.get("last_hidden_state").get();

            ArrayList<Integer> outputIds = greedyDecode(encoderOutput, mask, nllbTgt);

            int[] outputArray = new int[outputIds.size()];
            for (int i = 0; i < outputIds.size(); i++) {
                outputArray[i] = outputIds.get(i);
            }
            String translation = tokenizer.decode(outputArray);

            mask.close();
            return translation;
        } catch (OrtException e) {
            Log.e(TAG, "Dịch lỗi", e);
            return "[error]";
        } finally {
            if (encoderResult != null) encoderResult.close();
        }
    }

    private ArrayList<Integer> greedyDecode(OnnxTensor encoderOutput, OnnxTensor mask, String nllbTgt) throws OrtException {
        ArrayList<Integer> outputIds = new ArrayList<>();
        int eosId = tokenizer.pieceToId("</s>");
        int targetLangId = tokenizer.getLanguageID(nllbTgt);
        int currentToken = targetLangId; 
        outputIds.add(currentToken);

        Map<String, OnnxTensor> pastKeyValues = new HashMap<>();
        OrtSession.Result result = null;

        try {
            // Step 0: Initialize empty past key values for the first step
            // Xenova NLLB-200 merged models usually have 12 layers, 16 heads, 64 head_dim
            // Some models might also have a 'use_cache_branch' input.
            boolean hasCacheBranch = false;
            for (String inputName : decoderSession.getInputNames()) {
                if (inputName.equals("use_cache_branch")) {
                    hasCacheBranch = true;
                } else if (inputName.startsWith("past_key_values")) {
                    pastKeyValues.put(inputName, TensorUtils.createFloatTensor(env, new long[]{1, 16, 0, 64}));
                }
            }

            for (int i = 0; i < MAX_OUTPUT_TOKENS; i++) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                OnnxTensor inputTokenTensor = TensorUtils.intArrayToTensor(env, new int[]{currentToken});
                inputs.put("input_ids", inputTokenTensor);
                inputs.put("encoder_hidden_states", encoderOutput);
                inputs.put("encoder_attention_mask", mask);
                inputs.putAll(pastKeyValues);
                
                if (hasCacheBranch) {
                    inputs.put("use_cache_branch", TensorUtils.booleanToTensor(env, i > 0));
                }

                try (OrtSession.Result stepResult = decoderSession.run(inputs)) {
                    inputTokenTensor.close();
                    if (hasCacheBranch) {
                        inputs.get("use_cache_branch").close();
                    }

                    float[][][] logits = (float[][][]) stepResult.get("logits").get().getValue();
                    // Pick the last token's logits
                    currentToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);
                    outputIds.add(currentToken);

                    if (currentToken == eosId) {
                        break;
                    }

                    Map<String, OnnxTensor> nextPastKeyValues = new HashMap<>();
                    for (Map.Entry<String, OnnxValue> entry : stepResult) {
                        if (entry.getKey().startsWith("present")) {
                            String pastName = entry.getKey().replace("present", "past_key_values");
                            nextPastKeyValues.put(pastName, (OnnxTensor) entry.getValue());
                        }
                    }

                    // Close old past tensors
                    for (OnnxTensor t : pastKeyValues.values()) {
                        t.close();
                    }

                    pastKeyValues = nextPastKeyValues;
                }
            }

        } finally {
            if (result != null) result.close();
            for (OnnxTensor t : pastKeyValues.values()) {
                t.close();
            }
        }

        return outputIds;
    }

    private ArrayList<String> splitIntoSentences(String text, String langCode) {
        ArrayList<String> sentences = new ArrayList<>();
        Locale locale = langCode.equals("vi") ? new Locale("vi") : Locale.US;
        BreakIterator iterator = BreakIterator.getSentenceInstance(locale);
        iterator.setText(text);
        int start = iterator.first();
        for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
            sentences.add(text.substring(start, end).trim());
        }
        return sentences;
    }

    private String getNllbCode(String langCode) {
        return NLLB_CODES.getOrDefault(langCode, "vie_Latn");
    }

    private String copyModel(Context context, File baseDir, String assetName) {
        File outFile = new File(baseDir, assetName);
        if (!outFile.exists()) {
            FileUtils.copyAssetToDir(context, assetName, baseDir);
        }
        return outFile.getAbsolutePath();
    }

    public static class TranslationResult {
        public final String text;
        public final long processingMs;
        public TranslationResult(String text, long processingMs) {
            this.text = text;
            this.processingMs = processingMs;
        }
    }
}
