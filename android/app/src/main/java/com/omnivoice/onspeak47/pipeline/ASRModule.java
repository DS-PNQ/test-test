/*
 * OmniVoice — ASR Module (Whisper Small via ONNX Runtime)
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.util.Log;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.TensorUtils;


/**
 * Whisper Small ASR module using ONNX Runtime.
 *
 * Handles:
 *   - Loading the Whisper encoder/decoder ONNX models from assets
 *   - Audio feature extraction (log-mel spectrogram)
 *   - Autoregressive decoding to produce text transcription
 *   - Language detection for VI, EN, ZH
 */
public class ASRModule {

    private static final String TAG = "ASRModule";

    private static final String ENCODER_FILE = "whisper_encoder.onnx";
    private static final String DECODER_FILE = "whisper_decoder.onnx";
    private static final String PREPROCESS_FILE = "whisper_preprocess.onnx";
    private static final String VOCAB_FILE = "whisper_vocab.json";
    private static final int MAX_TOKENS = 448;

    private final OrtEnvironment env;
    private OrtSession encoderSession;
    private OrtSession decoderSession;
    private OrtSession preprocessSession;
    /** id -> token piece, loaded from whisper_vocab.json for native BPE decoding. */
    private final Map<Integer, String> vocab = new HashMap<>();

    // Whisper special token IDs
    private static final int SOT = 50258;        // <|startoftranscript|>
    private static final int EOT = 50257;        // <|endoftext|>
    private static final int TRANSCRIBE = 50359; // <|transcribe|>
    private static final int NO_TIMESTAMPS = 50363;

    // Language tokens
    private static final Map<String, Integer> LANGUAGE_TOKENS = new HashMap<>();
    static {
        LANGUAGE_TOKENS.put("vi", 50278); // Corrected from 50264 (<|ko|>)
        LANGUAGE_TOKENS.put("en", 50259);
        LANGUAGE_TOKENS.put("zh", 50260);
    }

    /**
     * Initialize the ASR module, loading ONNX models from assets.
     */
    public ASRModule(Context context) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        // Copy models from assets to internal storage if needed
        String encoderPath = copyModelToInternal(context, ENCODER_FILE);
        String decoderPath = copyModelToInternal(context, DECODER_FILE);
        String preprocessPath = copyModelToInternal(context, PREPROCESS_FILE);

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        try {
            options.registerCustomOpLibrary(ai.onnxruntime.extensions.OrtxPackage.getLibraryPath());
        } catch (OrtException e) {
            Log.e(TAG, "Extensions library not found", e);
        }
        // Was BASIC_OPT — ALL_OPT enables every ONNX Runtime graph optimization
        // (op fusion, constant folding). Drop to EXTENDED_OPT if it conflicts
        // with the registered custom-op nodes above.
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        try {
            options.addNnapi();
        } catch (OrtException e) {
            Log.w(TAG, "NNAPI EP unavailable on this device, falling back to CPU", e);
        }

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);
        try {
            preprocessSession = env.createSession(preprocessPath, options);
        } catch (OrtException e) {
            Log.e(TAG, "whisper_preprocess.onnx not found/failed to load — run "
                    + "optimize/01_export_onnx.py to generate it. ASR will not work "
                    + "until this model is present in assets.", e);
        }

        loadVocab(context);

        Log.i(TAG, "ASR module initialized (Whisper Small)");
    }

    private void loadVocab(Context context) {
        try (InputStream is = context.getAssets().open(VOCAB_FILE)) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int n;
            while ((n = is.read(chunk)) != -1) {
                buffer.write(chunk, 0, n);
            }
            JSONObject json = new JSONObject(buffer.toString(StandardCharsets.UTF_8.name()));
            Iterator<String> keys = json.keys();
            while (keys.hasNext()) {
                String piece = keys.next();
                vocab.put(json.getInt(piece), piece);
            }
            Log.i(TAG, "Loaded Whisper vocab: " + vocab.size() + " tokens");
        } catch (Exception e) {
            Log.e(TAG, VOCAB_FILE + " not found/failed to parse — run "
                    + "optimize/01_export_onnx.py to generate it. Decoded text will be "
                    + "empty until this file is present in assets.", e);
        }
    }

    /**
     * Transcribe audio from a WAV file path.
     *
     * @param audioPath Path to 16kHz mono WAV file
     * @param language  Language hint ("vi", "en", "zh") or null for auto-detect
     * @return Transcription result
     */
    public ASRResult transcribe(String audioPath, String language) {
        long startTime = System.currentTimeMillis();

        try {
            File audioFile = new File(audioPath);
            if (!audioFile.exists() || audioFile.length() <= 44) {
                Log.w(TAG, "Audio file is missing or empty: " + audioPath);
                return new ASRResult("[No audio recorded]", language != null ? language : "auto", 0);
            }

            // 1. Extract log-mel spectrogram features — via whisper_preprocess.onnx
            //    (raw audio file bytes in, log-mel out). No manual FFT/mel-filterbank
            //    computation here; that used to just return a zero-filled array.
            OnnxTensor melFeatures = runPreprocess(audioPath);

            // 2. Run encoder
            OnnxTensor encoderOutput = runEncoder(melFeatures);
            if (melFeatures != null) melFeatures.close();

            // 3. Run decoder (autoregressive) -> raw token ids
            List<Integer> tokenIds = runDecoder(encoderOutput, language);
            if (encoderOutput != null) encoderOutput.close();

            // 4. Detokenize -> text. Native Java BPE decode against
            //    whisper_vocab.json rather than guessing whisper_postprocess.onnx's
            //    expected input shape (see comment in 01_export_onnx.py).
            String text = decodeTokens(tokenIds);

            long elapsed = System.currentTimeMillis() - startTime;
            Log.i(TAG, "Transcription done in " + elapsed + "ms: " + text);

            return new ASRResult(text, language != null ? language : "auto", elapsed);

        } catch (Exception e) {
            Log.e(TAG, "Transcription failed", e);
            return new ASRResult("", "error", 0);
        }
    }

    // ----------------------------------------------------------------
    // ONNX inference
    // ----------------------------------------------------------------

    private OnnxTensor runPreprocess(String audioPath) throws OrtException, IOException {
        if (preprocessSession == null) {
            Log.e(TAG, "Preprocess session is null, skipping preprocessing");
            return null;
        }

        byte[] rawWavBytes;
        try (FileInputStream fis = new FileInputStream(audioPath)) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int n;
            while ((n = fis.read(chunk)) != -1) {
                buffer.write(chunk, 0, n);
            }
            rawWavBytes = buffer.toByteArray();
        }

        if (rawWavBytes.length == 0) {
            Log.e(TAG, "Audio file is empty: " + audioPath);
            return null;
        }

        String inputName = preprocessSession.getInputNames().iterator().next();
        long[] shape = new long[]{1, rawWavBytes.length};
        OnnxTensor audioTensor = OnnxTensor.createTensor(
                env, java.nio.ByteBuffer.wrap(rawWavBytes), shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inputName, audioTensor);

        try (OrtSession.Result result = preprocessSession.run(inputs)) {
            audioTensor.close();
            String outputName = preprocessSession.getOutputNames().iterator().next();
            OnnxValue melValue = result.get(outputName).get();
            if (!(melValue instanceof OnnxTensor)) {
                Log.e(TAG, "Preprocess output is not a tensor: " + outputName);
                return null;
            }
            OnnxTensor melTensor = (OnnxTensor) melValue;
            Log.d(TAG, "Preprocess output shape: " + java.util.Arrays.toString(melTensor.getInfo().getShape()));
            
            return OnnxTensor.createTensor(env, melTensor.getFloatBuffer(), melTensor.getInfo().getShape());
        }
    }

    private OnnxTensor runEncoder(OnnxTensor melFeatures) throws OrtException {
        if (encoderSession == null || melFeatures == null) {
            Log.e(TAG, "Encoder session or mel features is null");
            return null;
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_features", melFeatures);

        try (OrtSession.Result result = encoderSession.run(inputs)) {
            OnnxValue outputValue = result.get("last_hidden_state").get();
            if (!(outputValue instanceof OnnxTensor)) {
                Log.e(TAG, "Encoder output is not a tensor");
                return null;
            }
            OnnxTensor encoderOutput = (OnnxTensor) outputValue;
            Log.d(TAG, "Encoder output shape: " + java.util.Arrays.toString(encoderOutput.getInfo().getShape()));
            
            return OnnxTensor.createTensor(env, encoderOutput.getFloatBuffer(), encoderOutput.getInfo().getShape());
        }
    }

    private List<Integer> runDecoder(OnnxTensor encoderOutput, String language) throws OrtException {
        if (decoderSession == null || encoderOutput == null) return new ArrayList<>();

        int langToken = language != null && LANGUAGE_TOKENS.containsKey(language)
                ? LANGUAGE_TOKENS.get(language)
                : LANGUAGE_TOKENS.get("vi");  // default to Vietnamese

        ArrayList<Integer> outputTokens = new ArrayList<>();
        long[] promptTokens = new long[]{SOT, langToken, TRANSCRIBE, NO_TIMESTAMPS};

        // Detect whether this decoder graph exposes a KV-cache (past_key_values /
        // present), same as the merged NLLB decoder already handled correctly in
        // TranslationModule.greedyDecode(). If optimize/01_export_onnx.py hasn't
        // been re-run with use_cache=True yet, this falls back to the original
        // full-resequence-every-step behaviour so nothing breaks in the meantime.
        boolean usesCache = false;
        boolean hasCacheBranch = false;
        for (String inputName : decoderSession.getInputNames()) {
            if (inputName.equals("use_cache_branch")) {
                hasCacheBranch = true;
            } else if (inputName.startsWith("past_key_values")) {
                usesCache = true;
            }
        }

        Map<String, OnnxTensor> pastKeyValues = new HashMap<>();
        try {
            if (usesCache) {
                // Whisper Small: 12 decoder layers / 12 heads / 64 head_dim.
                // Adjust the "12, ..., 64" here if you export a different Whisper size.
                for (String inputName : decoderSession.getInputNames()) {
                    if (inputName.startsWith("past_key_values")) {
                        pastKeyValues.put(inputName, TensorUtils.createFloatTensor(env, new long[]{1, 12, 0, 64}));
                    }
                }
            }

            long[] currentTokens = promptTokens;
            int nextToken = -1;

            for (int i = 0; i < MAX_TOKENS; i++) {
                // With cache: feed the whole prompt once on step 0 (prefill), then
                // just the single newest token on every step after. Without cache:
                // keep resending the whole growing sequence (old, slower behaviour).
                long[] stepInputIds = usesCache && i > 0 ? new long[]{nextToken} : currentTokens;
                long[] shape = new long[]{1, stepInputIds.length};
                OnnxTensor inputIds = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(stepInputIds), shape);

                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputIds);
                inputs.put("encoder_hidden_states", encoderOutput);
                if (usesCache) {
                    inputs.putAll(pastKeyValues);
                    if (hasCacheBranch) {
                        inputs.put("use_cache_branch", TensorUtils.booleanToTensor(env, i > 0));
                    }
                }

                try (OrtSession.Result result = decoderSession.run(inputs)) {
                    inputIds.close();
                    if (usesCache && hasCacheBranch) {
                        inputs.get("use_cache_branch").close();
                    }

                    // Named lookup like TranslationModule uses; if your exported
                    // graph names the output differently, check session.getOutputNames().
                    float[][][] logits = (float[][][]) result.get("logits").get().getValue();
                    nextToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);

                    if (i < 5) {
                        Log.d(TAG, "Step " + i + ", next token: " + nextToken + " (" + vocab.get(nextToken) + ")");
                    }

                    if (nextToken == EOT || i == MAX_TOKENS - 1) {
                        Log.i(TAG, "Decoding stopped at step " + i + (nextToken == EOT ? " (EOT)" : " (MAX_TOKENS)"));
                        break;
                    }
                    outputTokens.add(nextToken);

                    if (usesCache) {
                        Map<String, OnnxTensor> nextPastKeyValues = new HashMap<>();
                        for (Map.Entry<String, OnnxValue> entry : result) {
                            if (entry.getKey().startsWith("present")) {
                                String pastName = entry.getKey().replace("present", "past_key_values");
                                OnnxTensor presentTensor = (OnnxTensor) entry.getValue();
                                // result (and everything it owns) closes the instant
                                // this try-with-resources block exits, so we must copy
                                // the data out into an independent tensor rather than
                                // keep this reference — otherwise next iteration's
                                // run() throws IllegalStateException on a closed tensor.
                                nextPastKeyValues.put(pastName, OnnxTensor.createTensor(
                                        env, presentTensor.getFloatBuffer(), presentTensor.getInfo().getShape()));
                            }
                        }
                        for (OnnxTensor t : pastKeyValues.values()) {
                            t.close();
                        }
                        pastKeyValues = nextPastKeyValues;
                    } else {
                        long[] grown = new long[currentTokens.length + 1];
                        System.arraycopy(currentTokens, 0, grown, 0, currentTokens.length);
                        grown[currentTokens.length] = nextToken;
                        currentTokens = grown;
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "ASR decoder loop error", e);
        } finally {
            for (OnnxTensor t : pastKeyValues.values()) {
                t.close();
            }
        }

        Log.i(TAG, "Decoded " + outputTokens.size() + " tokens");
        return outputTokens;
    }

    // ----------------------------------------------------------------
    // Detokenization — native Java byte-level BPE decode (GPT-2/Whisper style)
    // ----------------------------------------------------------------

    // Standard GPT-2/Whisper byte-level BPE mapping: printable bytes map to
    // themselves as unicode codepoints, non-printable bytes (control chars,
    // space, etc.) map to codepoints starting at 256. Stable/unchanged since
    // GPT-2's original release, so this is safe to hardcode.
    private static final Map<Integer, Character> BYTE_TO_UNICODE = new HashMap<>();
    private static final Map<Character, Integer> UNICODE_TO_BYTE = new HashMap<>();
    static {
        List<Integer> printable = new ArrayList<>();
        for (int b = 33; b <= 126; b++) printable.add(b);   // '!' .. '~'
        for (int b = 161; b <= 172; b++) printable.add(b);  // '¡' .. '¬'
        for (int b = 174; b <= 255; b++) printable.add(b);  // '®' .. 'ÿ'
        for (int b : printable) {
            BYTE_TO_UNICODE.put(b, (char) b);
        }
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!printable.contains(b)) {
                BYTE_TO_UNICODE.put(b, (char) (256 + n));
                n++;
            }
        }
        for (Map.Entry<Integer, Character> e : BYTE_TO_UNICODE.entrySet()) {
            UNICODE_TO_BYTE.put(e.getValue(), e.getKey());
        }
    }

    /**
     * Convert Whisper token ids back to text: look up each id's vocab piece,
     * concatenate, then reverse the byte-level unicode mapping to recover the
     * original UTF-8 bytes. Ids >= EOT (50257) are special/control tokens
     * (language, task, timestamps, ...) and are skipped, not emitted as text.
     */
    private String decodeTokens(List<Integer> tokenIds) {
        if (vocab.isEmpty()) {
            Log.w(TAG, "whisper_vocab.json not loaded — returning empty transcript");
            return "";
        }

        StringBuilder pieces = new StringBuilder();
        for (int id : tokenIds) {
            if (id >= EOT) continue; // skip special/control tokens
            String piece = vocab.get(id);
            if (piece != null) pieces.append(piece);
        }

        ByteArrayOutputStream rawBytes = new ByteArrayOutputStream();
        for (int i = 0; i < pieces.length(); i++) {
            Integer b = UNICODE_TO_BYTE.get(pieces.charAt(i));
            if (b != null) rawBytes.write(b);
        }
        return new String(rawBytes.toByteArray(), StandardCharsets.UTF_8);
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private String copyModelToInternal(Context context, String assetName) {
        File outFile = new File(context.getFilesDir(), assetName);
        if (!outFile.exists()) {
            FileUtils.copyAssetToInternal(context, assetName);
        }
        return outFile.getAbsolutePath();
    }

    // ----------------------------------------------------------------
    // Result class
    // ----------------------------------------------------------------

    public static class ASRResult {
        public final String text;
        public final String language;
        public final long processingMs;

        public ASRResult(String text, String language, long processingMs) {
            this.text = text;
            this.language = language;
            this.processingMs = processingMs;
        }
    }
}
