/*
 * OmniVoice — ASR Module (Whisper Small via ONNX Runtime)
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
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
 *   - Autoregressive decoding with KV-cache to produce text transcription
 *   - Language detection for VI, EN, ZH
 */
public class ASRModule {

    private static final String TAG = "ASRModule";

    private static final String ENCODER_FILE = "whisper_encoder.onnx";
    private static final String DECODER_FILE = "whisper_decoder.onnx";
    private static final int SAMPLE_RATE = 16000;
    private static final int N_MELS = 80;
    private static final int N_FFT = 400;
    private static final int HOP_LENGTH = 160;
    private static final int MAX_AUDIO_FRAMES = 3000; // 30 seconds at Whisper's frame rate
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

    // Whisper vocabulary for decoding token IDs to text.
    // In production, this would be loaded from a vocab file (e.g., vocab.json).
    // For now, we use the multilingual tokenizer from the model assets.
    private static final String WHISPER_VOCAB_FILE = "whisper_vocab.json";

    // Simple token-to-text mapping loaded from vocab file
    private Map<Integer, String> vocabMap;

    /**
     * Initialize the ASR module, loading ONNX models from assets.
     */
    public ASRModule(Context context) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        // Copy models from assets to internal storage if needed
        String encoderPath = copyModelToInternal(context, ENCODER_FILE);
        String decoderPath = copyModelToInternal(context, DECODER_FILE);

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        try {
            options.registerCustomOpLibrary(ai.onnxruntime.extensions.OrtxPackage.getLibraryPath());
        } catch (OrtException e) {
            Log.e(TAG, "Extensions library not found", e);
        }
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        // Try to use NNAPI for hardware acceleration (NPU/GPU/DSP)
        try {
            options.addNnapi();
            Log.i(TAG, "NNAPI execution provider enabled for ASR");
        } catch (OrtException e) {
            Log.w(TAG, "NNAPI not available, using CPU fallback", e);
        }

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);

        // Load vocabulary for token → text decoding
        loadVocabulary(context);

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
            File audioFile = new File(audioPath);
            if (!audioFile.exists() || audioFile.length() <= 44) {
                Log.w(TAG, "Audio file is missing or empty: " + audioPath);
                return new ASRResult("[No audio recorded]", language != null ? language : "auto", 0);
            }

            // 1. Extract log-mel spectrogram features
            float[][] melFeatures = extractMelFeatures(audioPath);

            // 2. Run encoder
            OnnxTensor encoderOutput = runEncoder(melFeatures);

            // 3. Run decoder (autoregressive with KV-cache)
            String text = runDecoder(encoderOutput, language);
            if (encoderOutput != null) encoderOutput.close();

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
        if (encoderSession == null) return null;
        // Reshape to [1, n_mels, n_frames]
        long[] shape = new long[]{1, melFeatures.length, melFeatures[0].length};
        float[] flat = TensorUtils.flatten2D(melFeatures);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_features", inputTensor);

        OrtSession.Result result = encoderSession.run(inputs);
        inputTensor.close();
        return (OnnxTensor) result.get(0);
    }

    private String runDecoder(OnnxTensor encoderOutput, String language) throws OrtException {
        if (decoderSession == null || encoderOutput == null) return "";

        int langToken = language != null && LANGUAGE_TOKENS.containsKey(language)
                ? LANGUAGE_TOKENS.get(language)
                : LANGUAGE_TOKENS.get("vi");  // default to Vietnamese

        ArrayList<Integer> outputTokens = new ArrayList<>();
        int currentToken;

        // Initial prompt tokens
        long[] initialTokens = new long[]{SOT, langToken, TRANSCRIBE, NO_TIMESTAMPS};

        // KV-cache support: detect past_key_values inputs from the decoder model
        Map<String, OnnxTensor> pastKeyValues = new HashMap<>();
        boolean hasCacheBranch = false;
        boolean hasKVCache = false;

        for (String inputName : decoderSession.getInputNames()) {
            if (inputName.equals("use_cache_branch")) {
                hasCacheBranch = true;
            } else if (inputName.startsWith("past_key_values")) {
                hasKVCache = true;
                // Initialize with empty KV-cache tensors
                // Whisper Small: 12 layers, 12 heads, 64 head_dim
                pastKeyValues.put(inputName, TensorUtils.createFloatTensor(env, new long[]{1, 12, 0, 64}));
            }
        }

        try {
            if (hasKVCache) {
                // === KV-cache path (O(n) per token) ===
                // First step: feed all initial tokens
                OnnxTensor inputIds = OnnxTensor.createTensor(env,
                        java.nio.LongBuffer.wrap(initialTokens),
                        new long[]{1, initialTokens.length});

                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputIds);
                inputs.put("encoder_hidden_states", encoderOutput);
                inputs.putAll(pastKeyValues);
                if (hasCacheBranch) {
                    inputs.put("use_cache_branch", TensorUtils.booleanToTensor(env, false));
                }

                try (OrtSession.Result result = decoderSession.run(inputs)) {
                    inputIds.close();
                    if (hasCacheBranch) inputs.get("use_cache_branch").close();

                    float[][][] logits = (float[][][]) result.get(0).getValue();
                    currentToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);

                    if (currentToken != EOT) {
                        outputTokens.add(currentToken);
                    }

                    // Extract present → past_key_values for next step
                    Map<String, OnnxTensor> nextPastKV = new HashMap<>();
                    for (Map.Entry<String, OnnxValue> entry : result) {
                        if (entry.getKey().startsWith("present")) {
                            String pastName = entry.getKey().replace("present", "past_key_values");
                            nextPastKV.put(pastName, (OnnxTensor) entry.getValue());
                        }
                    }
                    for (OnnxTensor t : pastKeyValues.values()) t.close();
                    pastKeyValues = nextPastKV;
                }

                // Subsequent steps: feed one token at a time with KV-cache
                for (int i = 1; i < MAX_TOKENS && currentToken != EOT; i++) {
                    OnnxTensor singleTokenTensor = OnnxTensor.createTensor(env,
                            java.nio.LongBuffer.wrap(new long[]{currentToken}),
                            new long[]{1, 1});

                    inputs = new HashMap<>();
                    inputs.put("input_ids", singleTokenTensor);
                    inputs.put("encoder_hidden_states", encoderOutput);
                    inputs.putAll(pastKeyValues);
                    if (hasCacheBranch) {
                        inputs.put("use_cache_branch", TensorUtils.booleanToTensor(env, true));
                    }

                    try (OrtSession.Result result = decoderSession.run(inputs)) {
                        singleTokenTensor.close();
                        if (hasCacheBranch) inputs.get("use_cache_branch").close();

                        float[][][] logits = (float[][][]) result.get(0).getValue();
                        currentToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);

                        if (currentToken == EOT) break;
                        outputTokens.add(currentToken);

                        Map<String, OnnxTensor> nextPKV = new HashMap<>();
                        for (Map.Entry<String, OnnxValue> entry : result) {
                            if (entry.getKey().startsWith("present")) {
                                String pastName = entry.getKey().replace("present", "past_key_values");
                                nextPKV.put(pastName, (OnnxTensor) entry.getValue());
                            }
                        }
                        for (OnnxTensor t : pastKeyValues.values()) t.close();
                        pastKeyValues = nextPKV;
                    }
                }
            } else {
                // === Fallback: no KV-cache (grows input sequence each step) ===
                long[] currentTokens = initialTokens.clone();

                for (int i = 0; i < MAX_TOKENS; i++) {
                    long[] shape = new long[]{1, currentTokens.length};
                    OnnxTensor inputIds = OnnxTensor.createTensor(env,
                            java.nio.LongBuffer.wrap(currentTokens), shape);

                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put("input_ids", inputIds);
                    inputs.put("encoder_hidden_states", encoderOutput);

                    try (OrtSession.Result result = decoderSession.run(inputs)) {
                        float[][][] logits = (float[][][]) result.get(0).getValue();
                        int nextToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);

                        if (nextToken == EOT) {
                            inputIds.close();
                            break;
                        }
                        outputTokens.add(nextToken);

                        long[] nextTokens = new long[currentTokens.length + 1];
                        System.arraycopy(currentTokens, 0, nextTokens, 0, currentTokens.length);
                        nextTokens[currentTokens.length] = nextToken;
                        currentTokens = nextTokens;
                    }
                    inputIds.close();
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "ASR decoder loop error", e);
        } finally {
            for (OnnxTensor t : pastKeyValues.values()) {
                try { t.close(); } catch (Exception ignored) {}
            }
        }

        // Decode token IDs → text
        return decodeTokens(outputTokens);
    }

    // ----------------------------------------------------------------
    // Token decoding
    // ----------------------------------------------------------------

    /**
     * Load Whisper vocabulary mapping from assets.
     * Falls back to a basic byte-level BPE decode if vocab file is not available.
     */
    private void loadVocabulary(Context context) {
        vocabMap = new HashMap<>();
        try {
            String vocabPath = copyModelToInternal(context, WHISPER_VOCAB_FILE);
            File vocabFile = new File(vocabPath);
            if (vocabFile.exists()) {
                // Parse simple JSON vocab: {"token_string": id, ...}
                String json = new String(java.nio.file.Files.readAllBytes(vocabFile.toPath()), "UTF-8");
                // Simple JSON parsing without external library
                json = json.trim();
                if (json.startsWith("{")) json = json.substring(1);
                if (json.endsWith("}")) json = json.substring(0, json.length() - 1);

                // Build reverse map: id → token_string
                String[] pairs = json.split(",");
                for (String pair : pairs) {
                    int colonIdx = pair.lastIndexOf(':');
                    if (colonIdx < 0) continue;
                    String key = pair.substring(0, colonIdx).trim();
                    String val = pair.substring(colonIdx + 1).trim();
                    // Remove quotes from key
                    if (key.startsWith("\"") && key.endsWith("\"")) {
                        key = key.substring(1, key.length() - 1);
                    }
                    try {
                        int id = Integer.parseInt(val.trim());
                        // Unescape basic sequences
                        key = key.replace("\\n", "\n")
                                 .replace("\\t", "\t")
                                 .replace("\\\"", "\"")
                                 .replace("\\\\", "\\")
                                 .replace("Ġ", " ");  // GPT-2 BPE space marker
                        vocabMap.put(id, key);
                    } catch (NumberFormatException ignored) {}
                }
                Log.i(TAG, "Loaded Whisper vocabulary: " + vocabMap.size() + " tokens");
            } else {
                Log.w(TAG, "Whisper vocab file not found, token decoding will be limited");
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load Whisper vocabulary", e);
        }
    }

    /**
     * Decode a list of Whisper token IDs to text.
     */
    private String decodeTokens(ArrayList<Integer> tokenIds) {
        if (tokenIds.isEmpty()) return "";

        if (vocabMap != null && !vocabMap.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (int id : tokenIds) {
                // Skip special tokens (>= 50257)
                if (id >= 50257) continue;
                String piece = vocabMap.get(id);
                if (piece != null) {
                    sb.append(piece);
                }
            }
            return sb.toString().trim();
        }

        // Fallback: return raw token IDs as debug info
        Log.w(TAG, "No vocab loaded, returning raw token IDs");
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            if (id < 50257) {
                sb.append("[").append(id).append("]");
            }
        }
        return sb.toString();
    }

    // ----------------------------------------------------------------
    // Audio processing — Log-Mel Spectrogram
    // ----------------------------------------------------------------

    /**
     * Extract log-mel spectrogram features from a 16kHz mono WAV file.
     *
     * Implements the same preprocessing as Whisper:
     * 1. Read raw PCM samples from WAV
     * 2. Compute STFT with n_fft=400, hop_length=160
     * 3. Apply 80-band mel filterbank
     * 4. Convert to log scale
     * 5. Pad/trim to exactly 3000 frames (30 seconds)
     */
    private float[][] extractMelFeatures(String audioPath) {
        // Read PCM samples from WAV file
        float[] audioSamples = readWavPCM(audioPath);
        if (audioSamples == null || audioSamples.length == 0) {
            Log.e(TAG, "Failed to read audio samples, returning zero features");
            return new float[N_MELS][MAX_AUDIO_FRAMES];
        }

        // Pad audio to at least 30 seconds if shorter
        int targetSamples = SAMPLE_RATE * 30; // 30 seconds
        if (audioSamples.length < targetSamples) {
            float[] padded = new float[targetSamples];
            System.arraycopy(audioSamples, 0, padded, 0, audioSamples.length);
            audioSamples = padded;
        }

        // Compute STFT magnitudes
        int numFrames = (audioSamples.length - N_FFT) / HOP_LENGTH + 1;
        if (numFrames > MAX_AUDIO_FRAMES) numFrames = MAX_AUDIO_FRAMES;

        float[][] stftMag = new float[N_FFT / 2 + 1][numFrames];
        float[] window = hannWindow(N_FFT);

        for (int frame = 0; frame < numFrames; frame++) {
            int offset = frame * HOP_LENGTH;
            float[] real = new float[N_FFT];
            float[] imag = new float[N_FFT];

            // Apply Hann window
            for (int j = 0; j < N_FFT && (offset + j) < audioSamples.length; j++) {
                real[j] = audioSamples[offset + j] * window[j];
            }

            // FFT
            fft(real, imag, N_FFT);

            // Compute magnitude squared
            for (int k = 0; k <= N_FFT / 2; k++) {
                stftMag[k][frame] = real[k] * real[k] + imag[k] * imag[k];
            }
        }

        // Apply mel filterbank
        float[][] melFilters = melFilterbank(SAMPLE_RATE, N_FFT, N_MELS);
        float[][] melSpec = new float[N_MELS][MAX_AUDIO_FRAMES];

        for (int m = 0; m < N_MELS; m++) {
            for (int frame = 0; frame < numFrames; frame++) {
                float sum = 0;
                for (int k = 0; k <= N_FFT / 2; k++) {
                    sum += melFilters[m][k] * stftMag[k][frame];
                }
                // Log scale (with floor to avoid log(0))
                melSpec[m][frame] = (float) Math.log10(Math.max(sum, 1e-10));
            }
        }

        // Normalize: Whisper uses log10 and then clamps/normalizes
        // Find max value for normalization
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int m = 0; m < N_MELS; m++) {
            for (int f = 0; f < MAX_AUDIO_FRAMES; f++) {
                if (melSpec[m][f] > maxVal) maxVal = melSpec[m][f];
            }
        }

        // Clamp to max - 8.0, then scale to [0, 1] range, then shift to [-1, 1]
        float clampMin = maxVal - 8.0f;
        for (int m = 0; m < N_MELS; m++) {
            for (int f = 0; f < MAX_AUDIO_FRAMES; f++) {
                melSpec[m][f] = Math.max(melSpec[m][f], clampMin);
                melSpec[m][f] = (melSpec[m][f] - clampMin) / 8.0f;  // normalize to ~[0,1]
                melSpec[m][f] = melSpec[m][f] * 2.0f - 1.0f;        // shift to [-1,1]
            }
        }

        return melSpec;
    }

    /**
     * Read raw PCM float samples from a 16-bit mono WAV file.
     */
    private float[] readWavPCM(String wavPath) {
        try (FileInputStream fis = new FileInputStream(wavPath)) {
            // Read WAV header (44 bytes for standard PCM WAV)
            byte[] header = new byte[44];
            if (fis.read(header) < 44) {
                Log.e(TAG, "WAV header too short");
                return null;
            }

            // Verify RIFF header
            if (header[0] != 'R' || header[1] != 'I' || header[2] != 'F' || header[3] != 'F') {
                Log.e(TAG, "Not a valid WAV file");
                return null;
            }

            // Read sample rate from header (bytes 24-27)
            int sampleRate = ByteBuffer.wrap(header, 24, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            int bitsPerSample = ByteBuffer.wrap(header, 34, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
            int numChannels = ByteBuffer.wrap(header, 22, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();

            if (sampleRate != SAMPLE_RATE) {
                Log.w(TAG, "Expected 16kHz, got " + sampleRate + "Hz — results may be degraded");
            }

            // Read PCM data
            int dataSize = (int) (new File(wavPath).length() - 44);
            byte[] audioBytes = new byte[dataSize];
            int bytesRead = fis.read(audioBytes);

            int bytesPerSample = bitsPerSample / 8;
            int numSamples = bytesRead / (bytesPerSample * numChannels);
            float[] samples = new float[numSamples];

            ByteBuffer buffer = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < numSamples; i++) {
                if (bitsPerSample == 16) {
                    short sample = buffer.getShort();
                    // Skip extra channels
                    for (int c = 1; c < numChannels; c++) {
                        if (buffer.hasRemaining()) buffer.getShort();
                    }
                    samples[i] = sample / 32768.0f;  // normalize to [-1, 1]
                } else {
                    samples[i] = 0;
                }
            }

            Log.i(TAG, "Read " + numSamples + " audio samples (" +
                    String.format("%.1f", numSamples / (float) SAMPLE_RATE) + " seconds)");
            return samples;

        } catch (IOException e) {
            Log.e(TAG, "Failed to read WAV file: " + wavPath, e);
            return null;
        }
    }

    /**
     * Generate a Hann window of given size.
     */
    private float[] hannWindow(int size) {
        float[] window = new float[size];
        for (int i = 0; i < size; i++) {
            window[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / size));
        }
        return window;
    }

    /**
     * In-place Cooley-Tukey FFT (radix-2, decimation-in-time).
     * Input arrays are modified in-place.
     */
    private void fft(float[] real, float[] imag, int n) {
        if (n <= 1) return;

        // Bit-reversal permutation
        int logN = Integer.numberOfTrailingZeros(Integer.highestOneBit(n));
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - logN);
            if (j > i) {
                float tempR = real[i]; real[i] = real[j]; real[j] = tempR;
                float tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
            }
        }

        // Butterfly stages
        for (int size = 2; size <= n; size *= 2) {
            int halfSize = size / 2;
            float angle = (float) (-2.0 * Math.PI / size);
            float wR = (float) Math.cos(angle);
            float wI = (float) Math.sin(angle);

            for (int i = 0; i < n; i += size) {
                float curR = 1.0f, curI = 0.0f;
                for (int j = 0; j < halfSize; j++) {
                    int even = i + j;
                    int odd = i + j + halfSize;

                    float tR = curR * real[odd] - curI * imag[odd];
                    float tI = curR * imag[odd] + curI * real[odd];

                    real[odd] = real[even] - tR;
                    imag[odd] = imag[even] - tI;
                    real[even] += tR;
                    imag[even] += tI;

                    float newR = curR * wR - curI * wI;
                    curI = curR * wI + curI * wR;
                    curR = newR;
                }
            }
        }
    }

    /**
     * Create an 80-band mel filterbank matrix.
     *
     * @return float[n_mels][n_fft/2 + 1] filterbank
     */
    private float[][] melFilterbank(int sampleRate, int nFft, int nMels) {
        int numBins = nFft / 2 + 1;
        float[][] filters = new float[nMels][numBins];

        float fMin = 0.0f;
        float fMax = sampleRate / 2.0f;
        float melMin = hzToMel(fMin);
        float melMax = hzToMel(fMax);

        // Equally spaced mel points
        float[] melPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) {
            melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
        }

        // Convert mel points back to Hz, then to FFT bin indices
        float[] binFreqs = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) {
            float hz = melToHz(melPoints[i]);
            binFreqs[i] = hz * (nFft + 1) / sampleRate;
        }

        // Create triangular filters
        for (int m = 0; m < nMels; m++) {
            float left = binFreqs[m];
            float center = binFreqs[m + 1];
            float right = binFreqs[m + 2];

            for (int k = 0; k < numBins; k++) {
                if (k >= left && k <= center && center != left) {
                    filters[m][k] = (k - left) / (center - left);
                } else if (k > center && k <= right && right != center) {
                    filters[m][k] = (right - k) / (right - center);
                }
            }

            // Normalize the filter (slaney normalization)
            float enorm = 2.0f / (melToHz(melPoints[m + 2]) - melToHz(melPoints[m]));
            for (int k = 0; k < numBins; k++) {
                filters[m][k] *= enorm;
            }
        }

        return filters;
    }

    private float hzToMel(float hz) {
        return 2595.0f * (float) Math.log10(1.0 + hz / 700.0);
    }

    private float melToHz(float mel) {
        return 700.0f * ((float) Math.pow(10.0, mel / 2595.0) - 1.0f);
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
