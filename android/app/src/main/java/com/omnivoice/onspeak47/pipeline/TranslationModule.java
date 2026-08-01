/*
 * OmniVoice — Translation Module (NLLB-200 via ONNX Runtime)
 *
 * Follows the same encoder-decoder with KV-cache architecture as
 * RTranslator-2.00's Translator.java, but scoped to VN/EN/CN only.
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
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.LanguageConfig;
import com.omnivoice.onspeak47.util.TensorUtils;


/**
 * NLLB-200 Distilled 600M translation module.
 *
 * Uses the same 4-file ONNX architecture as RTranslator-2.00:
 *   - NLLB_encoder.onnx
 *   - NLLB_decoder.onnx
 *   - NLLB_cache_initializer.onnx
 *   - NLLB_embed_and_lm_head.onnx
 *
 * Plus the SentencePiece vocabulary file:
 *   - sentencepiece_bpe.model
 */
public class TranslationModule {

    private static final String TAG = "TranslationModule";

    // ONNX model files (same naming as RTranslator-2.00)
    private static final String ENCODER_FILE = "NLLB_encoder.onnx";
    private static final String DECODER_FILE = "NLLB_decoder.onnx";
    private static final String CACHE_INIT_FILE = "NLLB_cache_initializer.onnx";
    private static final String EMBED_LM_HEAD_FILE = "NLLB_embed_and_lm_head.onnx";
    private static final String VOCAB_FILE = "sentencepiece_bpe.model";

    // NLLB architecture constants
    private static final int N_LAYERS = 12;
    private static final int HIDDEN_SIZE = 64;
    private static final int EMBED_DIM = 1024;
    private static final int MAX_OUTPUT_TOKENS = 256;

    private final OrtEnvironment env;
    private OrtSession encoderSession;
    private OrtSession decoderSession;
    private OrtSession cacheInitSession;
    private OrtSession embedAndLmHeadSession;
    private Tokenizer tokenizer;

    // NLLB FLORES-200 language code mapping (scoped to VN/EN/CN)
    private static final Map<String, String> NLLB_CODES = new HashMap<>();
    static {
        NLLB_CODES.put("vi", "vie_Latn");
        NLLB_CODES.put("en", "eng_Latn");
        NLLB_CODES.put("zh", "zho_Hans");
        NLLB_CODES.put("zh_hans", "zho_Hans");
        NLLB_CODES.put("zh_hant", "zho_Hant");
    }

    /**
     * Initialize the translation module from ONNX assets.
     */
    public TranslationModule(Context context) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        // Copy models from assets to internal storage
        String encoderPath = copyModel(context, ENCODER_FILE);
        String decoderPath = copyModel(context, DECODER_FILE);
        String cacheInitPath = copyModel(context, CACHE_INIT_FILE);
        String embedPath = copyModel(context, EMBED_LM_HEAD_FILE);
        String vocabPath = copyModel(context, VOCAB_FILE);

        // Create ONNX sessions with the same options as RTranslator
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setMemoryPatternOptimization(false);
        options.setCPUArenaAllocator(false);
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT);

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);
        cacheInitSession = env.createSession(cacheInitPath, options);
        embedAndLmHeadSession = env.createSession(embedPath, options);

        // Initialize SentencePiece tokenizer
        tokenizer = new Tokenizer(vocabPath, Tokenizer.NLLB);

        options.close();
        Log.i(TAG, "Translation module initialized (NLLB-200-distilled-600M)");
    }

    /**
     * Translate text from source to target language.
     *
     * @param text           Input text
     * @param srcLang        Source language code ("vi", "en", "zh")
     * @param tgtLang        Target language code
     * @return Translated text
     */
    public TranslationResult translate(String text, String srcLang, String tgtLang) {
        long startTime = System.currentTimeMillis();

        String nllbSrc = getNllbCode(srcLang);
        String nllbTgt = getNllbCode(tgtLang);

        // Split long text into sentences (same approach as RTranslator)
        ArrayList<String> sentences = splitIntoSentences(text, srcLang);
        StringBuilder result = new StringBuilder();

        for (String sentence : sentences) {
            String translated = translateSentence(sentence, nllbSrc, nllbTgt);
            if (result.length() > 0) result.append(" ");
            result.append(translated);
        }

        long elapsed = System.currentTimeMillis() - startTime;
        Log.i(TAG, "Translation done in " + elapsed + "ms");

        return new TranslationResult(result.toString(), elapsed);
    }

    // ----------------------------------------------------------------
    // Core translation (single sentence)
    // ----------------------------------------------------------------

    private String translateSentence(String text, String nllbSrc, String nllbTgt) {
        try {
            // 1. Tokenize
            Tokenizer.TokenizerResult input = tokenizer.tokenize(nllbSrc, nllbTgt, text);

            // 2. Run encoder
            OnnxTensor encoderOutput = runEncoder(input);
            if (encoderOutput == null) return "[error]";

            // 3. Initialize KV-cache
            OrtSession.Result cacheInit = initializeCache(encoderOutput);

            // 4. Run decoder with greedy search
            ArrayList<Integer> outputIds = greedyDecode(
                    input, encoderOutput, cacheInit
            );

            // 5. Detokenize
            int[] outputArray = outputIds.stream().mapToInt(i -> i).toArray();
            String translation = tokenizer.decode(outputArray);

            encoderOutput.close();
            if (cacheInit != null) cacheInit.close();

            return translation;

        } catch (OrtException e) {
            Log.e(TAG, "Translation ONNX error", e);
            return "[error]";
        }
    }

    private OnnxTensor runEncoder(Tokenizer.TokenizerResult input) throws OrtException {
        OnnxTensor inputIds = TensorUtils.intArrayToTensor(env, input.inputIDs);
        OnnxTensor attentionMask = TensorUtils.intArrayToTensor(env, input.attentionMask);

        // Get embeddings
        Map<String, OnnxTensor> embedInput = new HashMap<>();
        embedInput.put("input_ids", inputIds);
        embedInput.put("pre_logits", TensorUtils.createFloatTensor(env, new long[]{1, 1, EMBED_DIM}));
        embedInput.put("use_lm_head", TensorUtils.booleanToTensor(env, false));

        OrtSession.Result embedResult = embedAndLmHeadSession.run(embedInput);

        // Run encoder
        Map<String, OnnxTensor> encoderInput = new HashMap<>();
        encoderInput.put("input_ids", inputIds);
        encoderInput.put("attention_mask", attentionMask);
        encoderInput.put("embed_matrix", (OnnxTensor) embedResult.get(0));

        OrtSession.Result result = encoderSession.run(encoderInput);
        embedResult.close();

        return (OnnxTensor) result.get("last_hidden_state").get();
    }

    private OrtSession.Result initializeCache(OnnxTensor encoderOutput) throws OrtException {
        Map<String, OnnxTensor> initInput = new HashMap<>();
        initInput.put("encoder_hidden_states", encoderOutput);
        return cacheInitSession.run(initInput);
    }

    private ArrayList<Integer> greedyDecode(
            Tokenizer.TokenizerResult input,
            OnnxTensor encoderOutput,
            OrtSession.Result cacheInit
    ) throws OrtException {
        ArrayList<Integer> outputIds = new ArrayList<>();
        int eosId = tokenizer.pieceToId("</s>");

        // Start with BOS token (id=2 for NLLB)
        outputIds.add(2);

        for (int step = 0; step < MAX_OUTPUT_TOKENS; step++) {
            // [Simplified greedy decode — production would use full
            //  KV-cache management as in RTranslator-2.00's
            //  executeCacheDecoderGreedy method]

            // Get next token prediction
            int nextToken = predictNextToken(outputIds, input, encoderOutput, cacheInit);

            if (nextToken == eosId) break;
            outputIds.add(nextToken);
        }

        return outputIds;
    }

    private int predictNextToken(
            ArrayList<Integer> currentOutput,
            Tokenizer.TokenizerResult input,
            OnnxTensor encoderOutput,
            OrtSession.Result cacheInit
    ) throws OrtException {
        // Simplified — returns placeholder
        // Full implementation would mirror RTranslator's decoder loop
        // with embed → decoder → lm_head → argmax
        return tokenizer.pieceToId("</s>");
    }

    // ----------------------------------------------------------------
    // Text splitting (matching RTranslator approach)
    // ----------------------------------------------------------------

    private ArrayList<String> splitIntoSentences(String text, String langCode) {
        ArrayList<String> sentences = new ArrayList<>();
        Locale locale = langCode.equals("zh") ? Locale.CHINESE
                : langCode.equals("vi") ? new Locale("vi")
                : Locale.ENGLISH;

        BreakIterator iterator = BreakIterator.getSentenceInstance(locale);
        iterator.setText(text);

        int start = iterator.first();
        for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
            sentences.add(text.substring(start, end).trim());
        }

        if (sentences.isEmpty()) sentences.add(text);
        return sentences;
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private String getNllbCode(String langCode) {
        String code = NLLB_CODES.get(langCode);
        if (code == null) {
            Log.w(TAG, "Unknown language code: " + langCode + ", defaulting to vie_Latn");
            return "vie_Latn";
        }
        return code;
    }

    private String copyModel(Context context, String assetName) {
        File outFile = new File(context.getFilesDir(), assetName);
        if (!outFile.exists()) {
            FileUtils.copyAssetToInternal(context, assetName);
        }
        return outFile.getAbsolutePath();
    }

    // ----------------------------------------------------------------
    // Result class
    // ----------------------------------------------------------------

    public static class TranslationResult {
        public final String text;
        public final long processingMs;

        public TranslationResult(String text, long processingMs) {
            this.text = text;
            this.processingMs = processingMs;
        }
    }
}
