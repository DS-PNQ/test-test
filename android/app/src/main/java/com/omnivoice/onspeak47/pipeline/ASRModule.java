/*
 * OmniVoice — ASR Module (Whisper Small via ONNX Runtime)
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.util.JsonReader;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
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
 */
public class ASRModule {

    private static final String TAG = "ASRModule";

    private static final String ENCODER_FILE = "whisper_encoder.onnx";
    private static final String DECODER_FILE = "whisper_decoder.onnx";
    private static final int SAMPLE_RATE = 16000;
    private static final int N_MELS = 80;
    private static final int N_FFT = 512;        
    private static final int WINDOW_SIZE = 400;  
    private static final int HOP_LENGTH = 160;
    private static final int MAX_AUDIO_FRAMES = 3000; 
    private static final int MAX_TOKENS = 448;

    private final OrtEnvironment env;
    private OrtSession encoderSession;
    private OrtSession decoderSession;

    // Whisper special token IDs
    private static final int SOT = 50258;        
    private static final int EOT = 50257;        
    private static final int TRANSCRIBE = 50359; 
    private static final int NO_TIMESTAMPS = 50363;

    // Language tokens
    private static final Map<String, Integer> LANGUAGE_TOKENS = new HashMap<>();
    static {
        LANGUAGE_TOKENS.put("vi", 50264);
        LANGUAGE_TOKENS.put("en", 50259);
        LANGUAGE_TOKENS.put("zh", 50260);
    }

    private static final String WHISPER_VOCAB_FILE = "whisper_vocab.json";
    private Map<Integer, String> vocabMap;

    public ASRModule(Context context) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        String encoderPath = copyModelToInternal(context, ENCODER_FILE);
        String decoderPath = copyModelToInternal(context, DECODER_FILE);

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        try {
            options.registerCustomOpLibrary(ai.onnxruntime.extensions.OrtxPackage.getLibraryPath());
        } catch (OrtException e) {
            Log.e(TAG, "Extensions library not found", e);
        }
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        try {
            options.addNnapi();
            Log.i(TAG, "NNAPI execution provider enabled for ASR");
        } catch (OrtException e) {
            Log.w(TAG, "NNAPI not available, using CPU fallback", e);
        }

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);

        loadVocabulary(context);
        Log.i(TAG, "ASR module initialized (Whisper Small)");
    }

    public ASRResult transcribe(String audioPath, String language) {
        long startTime = System.currentTimeMillis();
        try {
            File audioFile = new File(audioPath);
            if (!audioFile.exists() || audioFile.length() <= 44) {
                Log.w(TAG, "Audio file is missing or empty: " + audioPath);
                return new ASRResult("[No audio recorded]", language != null ? language : "auto", 0);
            }

            float[][] melFeatures = extractMelFeatures(audioPath);
            OnnxTensor encoderOutput = runEncoder(melFeatures);
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

    private OnnxTensor runEncoder(float[][] melFeatures) throws OrtException {
        if (encoderSession == null) return null;
        long[] shape = new long[]{1, melFeatures.length, melFeatures[0].length};
        float[] flat = TensorUtils.flatten2D(melFeatures);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_features", inputTensor);

        try (OrtSession.Result result = encoderSession.run(inputs)) {
            inputTensor.close();
            return TensorUtils.cloneTensor(env, (OnnxTensor) result.get(0));
        }
    }

    private String runDecoder(OnnxTensor encoderOutput, String language) throws OrtException {
        if (decoderSession == null || encoderOutput == null) return "";

        int langToken = language != null && LANGUAGE_TOKENS.containsKey(language)
                ? LANGUAGE_TOKENS.get(language)
                : LANGUAGE_TOKENS.get("vi");

        ArrayList<Integer> outputTokens = new ArrayList<>();
        int currentToken;
        long[] initialTokens = new long[]{SOT, langToken, TRANSCRIBE, NO_TIMESTAMPS};

        Map<String, OnnxTensor> pastKeyValues = new HashMap<>();
        boolean hasCacheBranch = false;
        boolean hasKVCache = false;

        for (String inputName : decoderSession.getInputNames()) {
            if (inputName.equals("use_cache_branch")) hasCacheBranch = true;
            else if (inputName.startsWith("past_key_values")) {
                hasKVCache = true;
                pastKeyValues.put(inputName, TensorUtils.createFloatTensor(env, new long[]{1, 12, 0, 64}));
            }
        }

        try {
            if (hasKVCache) {
                OnnxTensor inputIds = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(initialTokens), new long[]{1, initialTokens.length});
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputIds);
                inputs.put("encoder_hidden_states", encoderOutput);
                inputs.putAll(pastKeyValues);
                if (hasCacheBranch) inputs.put("use_cache_branch", TensorUtils.booleanToTensor(env, false));

                try (OrtSession.Result result = decoderSession.run(inputs)) {
                    inputIds.close();
                    if (hasCacheBranch) inputs.get("use_cache_branch").close();
                    float[][][] logits = (float[][][]) result.get(0).getValue();
                    currentToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);
                    if (currentToken != EOT) outputTokens.add(currentToken);

                    Map<String, OnnxTensor> nextPastKV = new HashMap<>();
                    for (Map.Entry<String, OnnxValue> entry : result) {
                        if (entry.getKey().startsWith("present")) {
                            String pastName = entry.getKey().replace("present", "past_key_values");
                            nextPastKV.put(pastName, TensorUtils.cloneTensor(env, (OnnxTensor) entry.getValue()));
                        }
                    }
                    for (OnnxTensor t : pastKeyValues.values()) t.close();
                    pastKeyValues = nextPastKV;
                }

                for (int i = 1; i < MAX_TOKENS && currentToken != EOT; i++) {
                    OnnxTensor singleTokenTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(new long[]{currentToken}), new long[]{1, 1});
                    inputs = new HashMap<>();
                    inputs.put("input_ids", singleTokenTensor);
                    inputs.put("encoder_hidden_states", encoderOutput);
                    inputs.putAll(pastKeyValues);
                    if (hasCacheBranch) inputs.put("use_cache_branch", TensorUtils.booleanToTensor(env, true));

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
                                nextPKV.put(pastName, TensorUtils.cloneTensor(env, (OnnxTensor) entry.getValue()));
                            }
                        }
                        for (OnnxTensor t : pastKeyValues.values()) t.close();
                        pastKeyValues = nextPKV;
                    }
                }
            } else {
                long[] currentTokens = initialTokens.clone();
                for (int i = 0; i < MAX_TOKENS; i++) {
                    OnnxTensor inputIds = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(currentTokens), new long[]{1, currentTokens.length});
                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put("input_ids", inputIds);
                    inputs.put("encoder_hidden_states", encoderOutput);

                    try (OrtSession.Result result = decoderSession.run(inputs)) {
                        float[][][] logits = (float[][][]) result.get(0).getValue();
                        int nextToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);
                        if (nextToken == EOT) { inputIds.close(); break; }
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
        return decodeTokens(outputTokens);
    }

    private void loadVocabulary(Context context) {
        vocabMap = new HashMap<>();
        try {
            String vocabPath = copyModelToInternal(context, WHISPER_VOCAB_FILE);
            File vocabFile = new File(vocabPath);
            if (vocabFile.exists()) {
                try (JsonReader reader = new JsonReader(new InputStreamReader(new FileInputStream(vocabFile), "UTF-8"))) {
                    reader.beginObject();
                    while (reader.hasNext()) {
                        String key = reader.nextName();
                        int id = reader.nextInt();
                        vocabMap.put(id, key.replace("Ġ", " "));
                    }
                    reader.endObject();
                }
                Log.i(TAG, "Loaded Whisper vocabulary: " + vocabMap.size() + " tokens");
            } else {
                Log.w(TAG, "Whisper vocab file not found");
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to load Whisper vocabulary", e);
        }
    }

    private String decodeTokens(ArrayList<Integer> tokenIds) {
        if (tokenIds.isEmpty()) return "";
        if (vocabMap != null && !vocabMap.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (int id : tokenIds) {
                if (id >= 50257) continue;
                String piece = vocabMap.get(id);
                if (piece != null) sb.append(piece);
            }
            return sb.toString().trim();
        }
        return "[No vocab]";
    }

    private float[][] extractMelFeatures(String audioPath) {
        float[] audioSamples = readWavPCM(audioPath);
        if (audioSamples == null || audioSamples.length == 0) return new float[N_MELS][MAX_AUDIO_FRAMES];

        float maxAbs = 0;
        for (float s : audioSamples) if (Math.abs(s) > maxAbs) maxAbs = Math.abs(s);
        Log.i(TAG, "Max audio level: " + maxAbs);

        int targetSamples = SAMPLE_RATE * 30;
        if (audioSamples.length < targetSamples) {
            float[] padded = new float[targetSamples];
            System.arraycopy(audioSamples, 0, padded, 0, audioSamples.length);
            audioSamples = padded;
        }

        int numFrames = (audioSamples.length - WINDOW_SIZE) / HOP_LENGTH + 1;
        if (numFrames > MAX_AUDIO_FRAMES) numFrames = MAX_AUDIO_FRAMES;

        float[][] stftMag = new float[N_FFT / 2 + 1][numFrames];
        float[] window = hannWindow(WINDOW_SIZE);

        for (int frame = 0; frame < numFrames; frame++) {
            int offset = frame * HOP_LENGTH;
            float[] real = new float[N_FFT];
            float[] imag = new float[N_FFT];
            for (int j = 0; j < WINDOW_SIZE && (offset + j) < audioSamples.length; j++) {
                real[j] = audioSamples[offset + j] * window[j];
            }
            fft(real, imag, N_FFT);
            for (int k = 0; k <= N_FFT / 2; k++) stftMag[k][frame] = real[k] * real[k] + imag[k] * imag[k];
        }

        float[][] melFilters = melFilterbank(SAMPLE_RATE, N_FFT, N_MELS);
        float[][] melSpec = new float[N_MELS][MAX_AUDIO_FRAMES];

        for (int m = 0; m < N_MELS; m++) {
            for (int frame = 0; frame < numFrames; frame++) {
                float sum = 0;
                for (int k = 0; k <= N_FFT / 2; k++) sum += melFilters[m][k] * stftMag[k][frame];
                melSpec[m][frame] = (float) Math.log10(Math.max(sum, 1e-10));
            }
        }

        float maxVal = -10.0f;
        for (int m = 0; m < N_MELS; m++) {
            for (int f = 0; f < numFrames; f++) if (melSpec[m][f] > maxVal) maxVal = melSpec[m][f];
        }

        float clampMin = maxVal - 8.0f;
        for (int m = 0; m < N_MELS; m++) {
            for (int f = 0; f < MAX_AUDIO_FRAMES; f++) {
                float val = (f < numFrames) ? melSpec[m][f] : clampMin;
                val = Math.max(val, clampMin);
                melSpec[m][f] = (val - clampMin) / 4.0f - 1.0f;
            }
        }
        return melSpec;
    }

    private float[] readWavPCM(String wavPath) {
        try (FileInputStream fis = new FileInputStream(wavPath)) {
            byte[] header = new byte[44];
            if (fis.read(header) < 44) return null;
            int sampleRate = ByteBuffer.wrap(header, 24, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            int bitsPerSample = ByteBuffer.wrap(header, 34, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
            int numChannels = ByteBuffer.wrap(header, 22, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
            int dataSize = (int) (new File(wavPath).length() - 44);
            byte[] audioBytes = new byte[dataSize];
            fis.read(audioBytes);
            int bytesPerSample = bitsPerSample / 8;
            int numSamples = audioBytes.length / (bytesPerSample * numChannels);
            float[] samples = new float[numSamples];
            ByteBuffer buffer = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < numSamples; i++) {
                if (bitsPerSample == 16) {
                    samples[i] = buffer.getShort() / 32768.0f;
                    for (int c = 1; c < numChannels; c++) if (buffer.hasRemaining()) buffer.getShort();
                }
            }
            return samples;
        } catch (IOException e) { return null; }
    }

    private float[] hannWindow(int size) {
        float[] window = new float[size];
        for (int i = 0; i < size; i++) window[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / size));
        return window;
    }

    private void fft(float[] real, float[] imag, int n) {
        if (n <= 1) return;
        int logN = Integer.numberOfTrailingZeros(Integer.highestOneBit(n));
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - logN);
            if (j > i) {
                float tempR = real[i]; real[i] = real[j]; real[j] = tempR;
                float tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
            }
        }
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

    private float[][] melFilterbank(int sampleRate, int nFft, int nMels) {
        int numBins = nFft / 2 + 1;
        float[][] filters = new float[nMels][numBins];
        float melMin = hzToMel(0.0f), melMax = hzToMel(sampleRate / 2.0f);
        float[] melPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
        float[] binFreqs = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) binFreqs[i] = melToHz(melPoints[i]) * (nFft + 1) / sampleRate;
        for (int m = 0; m < nMels; m++) {
            float left = binFreqs[m], center = binFreqs[m + 1], right = binFreqs[m + 2];
            for (int k = 0; k < numBins; k++) {
                if (k >= left && k <= center && center != left) filters[m][k] = (k - left) / (center - left);
                else if (k > center && k <= right && right != center) filters[m][k] = (right - k) / (right - center);
            }
            float enorm = 2.0f / (melToHz(melPoints[m + 2]) - melToHz(melPoints[m]));
            for (int k = 0; k < numBins; k++) filters[m][k] *= enorm;
        }
        return filters;
    }

    private float hzToMel(float hz) { return 2595.0f * (float) Math.log10(1.0 + hz / 700.0); }
    private float melToHz(float mel) { return 700.0f * ((float) Math.pow(10.0, mel / 2595.0) - 1.0f); }

    private String copyModelToInternal(Context context, String assetName) {
        File outFile = new File(context.getFilesDir(), assetName);
        if (!outFile.exists()) FileUtils.copyAssetToInternal(context, assetName);
        return outFile.getAbsolutePath();
    }

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
