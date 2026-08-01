/*
 * OmniVoice — ASR Module (Whisper Small via ONNX Runtime)
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
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
    private static final int SAMPLE_RATE = 16000;
    private static final int N_MELS = 80;
    private static final int MAX_TOKENS = 448;

    private final OrtEnvironment env;
    private OrtSession encoderSession;
    private OrtSession decoderSession;

    // Whisper special token IDs
    private static final int SOT = 50258;        // <|startoftranscript|>
    private static final int EOT = 50257;        // <|endoftext|>
    private static final int TRANSCRIBE = 50359; // <|transcribe|>
    private static final int NO_TIMESTAMPS = 50363;

    // Language tokens
    private static final Map<String, Integer> LANGUAGE_TOKENS = new HashMap<>();
    static {
        LANGUAGE_TOKENS.put("vi", 50264);
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

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);

        Log.i(TAG, "ASR module initialized (Whisper Small)");
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
            // 1. Extract log-mel spectrogram features
            float[][] melFeatures = extractMelFeatures(audioPath);

            // 2. Run encoder
            OnnxTensor encoderOutput = runEncoder(melFeatures);

            // 3. Run decoder (autoregressive)
            String text = runDecoder(encoderOutput, language);

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

    private OnnxTensor runEncoder(float[][] melFeatures) throws OrtException {
        // Reshape to [1, n_mels, n_frames]
        long[] shape = new long[]{1, melFeatures.length, melFeatures[0].length};
        float[] flat = TensorUtils.flatten2D(melFeatures);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_features", inputTensor);

        OrtSession.Result result = encoderSession.run(inputs);
        return (OnnxTensor) result.get(0);
    }

    private String runDecoder(OnnxTensor encoderOutput, String language) throws OrtException {
        // Build initial decoder input tokens: SOT, language, transcribe, no_timestamps
        int langToken = language != null && LANGUAGE_TOKENS.containsKey(language)
                ? LANGUAGE_TOKENS.get(language)
                : LANGUAGE_TOKENS.get("vi");  // default to Vietnamese

        int[] decoderInputIds = new int[]{SOT, langToken, TRANSCRIBE, NO_TIMESTAMPS};

        StringBuilder result = new StringBuilder();

        // Autoregressive decoding
        for (int step = 0; step < MAX_TOKENS; step++) {
            // [Simplified — full implementation would use KV-cache]
            // This provides the framework; production code would implement
            // the full autoregressive loop with ONNX decoder sessions.
            break;  // Placeholder — see TranslationModule for full KV-cache example
        }

        return result.toString().trim();
    }

    // ----------------------------------------------------------------
    // Audio processing
    // ----------------------------------------------------------------

    private float[][] extractMelFeatures(String audioPath) {
        // Placeholder — production implementation would compute
        // log-mel spectrogram matching Whisper's preprocessing.
        // This requires FFT + mel filterbank computation.
        return new float[N_MELS][3000];  // 30 seconds at Whisper's frame rate
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
