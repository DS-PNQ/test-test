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
import com.omnivoice.onspeak47.util.OrtSessionConfig;
import com.omnivoice.onspeak47.util.TensorUtils;


/**
 * Whisper Small ASR module using ONNX Runtime.
 */
public class ASRModule {

    private static final String TAG = "ASRModule";

    private static final String ENCODER_FILE = "whisper_encoder.onnx";
    private static final String DECODER_FILE = "whisper_decoder.onnx";
    private static final String PREPROCESS_FILE = "whisper_preprocess.onnx";
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
    /** Native PCM->log-mel graph (ort-extensions); null -> Java DSP fallback. */
    private OrtSession preprocessSession;

    // Whisper special token IDs
    private static final int SOT = 50258;        
    private static final int EOT = 50257;        
    private static final int TRANSCRIBE = 50359; 
    private static final int NO_TIMESTAMPS = 50363;

    // Language tokens (verified against whisper_vocab.json)
    private static final Map<String, Integer> LANGUAGE_TOKENS = new HashMap<>();
    static {
        LANGUAGE_TOKENS.put("vi", 50278);  // <|vi|> — was incorrectly 50264 (<|ko|>)
        LANGUAGE_TOKENS.put("en", 50259);  // <|en|>
        LANGUAGE_TOKENS.put("zh", 50260);  // <|zh|>
    }

    private static final String WHISPER_VOCAB_FILE = "whisper_vocab.json";
    private Map<Integer, String> vocabMap;

    // ------------------------------------------------------------------
    // GPT-2 / Whisper byte-level BPE alphabet (bytes_to_unicode()).
    //
    // Whisper's vocab.json does NOT contain literal UTF-8 text. Each raw
    // byte (0-255) is remapped to one printable Unicode "placeholder"
    // character so that byte-level BPE merges can operate on any input
    // byte using ordinary text tooling. e.g. the Vietnamese/Chinese bytes
    // of a multi-byte UTF-8 character each get their own placeholder char
    // (space -> 'Ġ', and CJK bytes -> characters like 'ãĢĤ' for "。").
    //
    // Decoding therefore requires: for every character in a decoded token
    // string, map it back to its original byte via this reverse table,
    // concatenate ALL bytes across the whole output, and only then decode
    // the byte sequence as UTF-8 — never per-token, since a single UTF-8
    // character's bytes can be split across adjacent tokens.
    // ------------------------------------------------------------------
    private static final Map<Integer, Character> BYTE_TO_UNICODE = new HashMap<>();
    private static final Map<Character, Integer> UNICODE_TO_BYTE = new HashMap<>();
    static {
        boolean[] inBaseSet = new boolean[256];
        ArrayList<Integer> bytesList = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) { bytesList.add(b); inBaseSet[b] = true; }
        for (int b = 0xA1; b <= 0xAC; b++) { bytesList.add(b); inBaseSet[b] = true; }
        for (int b = 0xAE; b <= 0xFF; b++) { bytesList.add(b); inBaseSet[b] = true; }
        ArrayList<Integer> codepointsList = new ArrayList<>(bytesList);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!inBaseSet[b]) {
                bytesList.add(b);
                codepointsList.add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bytesList.size(); i++) {
            char c = (char) (int) codepointsList.get(i);
            BYTE_TO_UNICODE.put(bytesList.get(i), c);
            UNICODE_TO_BYTE.put(c, bytesList.get(i));
        }
    }

    public ASRModule(Context context) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        String encoderPath = copyModelToInternal(context, ENCODER_FILE);
        String decoderPath = copyModelToInternal(context, DECODER_FILE);

        // Mobile-tuned options (arena/memory-pattern off on low-RAM devices,
        // NNAPI off by default) — see OrtSessionConfig.
        OrtSession.SessionOptions options = OrtSessionConfig.create(context, true);
        try {
            encoderSession = env.createSession(encoderPath, options);
            decoderSession = env.createSession(decoderPath, options);

            // Optional native pre-processing graph: raw PCM -> log-mel via
            // ort-extensions custom ops (same approach as RTranslator's
            // Whisper initializer). Missing/broken asset -> Java DSP fallback.
            String preprocessPath = copyModelToInternalOptional(context, PREPROCESS_FILE);
            if (preprocessPath != null) {
                try {
                    preprocessSession = env.createSession(preprocessPath, options);
                    Log.i(TAG, "Whisper ONNX preprocessing enabled");
                } catch (OrtException e) {
                    preprocessSession = null;
                    Log.w(TAG, "Failed to load whisper_preprocess.onnx — Java DSP fallback: " + e.getMessage());
                }
            } else {
                Log.i(TAG, "whisper_preprocess.onnx not bundled — using Java DSP fallback");
            }
        } finally {
            options.close();
        }

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

            float[] audioSamples = readWavPCM(audioPath);

            // Prefer the native ONNX preprocessing graph; fall back to the
            // Java DSP path when the asset or the run is unavailable.
            OrtSession.Result preprocessResult = runPreprocess(audioSamples);

            OrtSession.Result encoderResult = null;
            String text;
            try {
                if (preprocessResult != null) {
                    encoderResult = runEncoder((OnnxTensor) preprocessResult.get(0));
                } else {
                    encoderResult = runEncoder(extractMelFeatures(audioSamples));
                }
                text = runDecoder(encoderResult, language);
            } finally {
                if (encoderResult != null) encoderResult.close();
                if (preprocessResult != null) preprocessResult.close();
            }

            long elapsed = System.currentTimeMillis() - startTime;
            Log.i(TAG, "Transcription done in " + elapsed + "ms: " + text);
            return new ASRResult(text, language != null ? language : "auto", elapsed);
        } catch (Exception e) {
            Log.e(TAG, "Transcription failed", e);
            return new ASRResult("", "error", 0);
        }
    }

    /**
     * Runs the exported whisper_preprocess.onnx graph (raw 16 kHz PCM ->
     * normalized log-mel) natively via the ort-extensions custom ops, the
     * same approach as RTranslator's Whisper initializer. The graph pads to
     * the fixed 3000-frame window internally, so the PCM is passed at its
     * REAL length — no 30 s pre-padding.
     *
     * @return the open Result owning the `input_features` tensor, or null to
     *         fall back to the Java DSP path (no session / bad input / error).
     */
    private OrtSession.Result runPreprocess(float[] audioSamples) {
        if (preprocessSession == null || audioSamples == null || audioSamples.length == 0) return null;
        try {
            String inputName = preprocessSession.getInputNames().iterator().next();
            // The declared input rank decides between (N) and (1, N) — the
            // exact contract varies across onnxruntime-extensions versions.
            ai.onnxruntime.TensorInfo inputInfo =
                    (ai.onnxruntime.TensorInfo) preprocessSession.getInputInfo()
                            .get(inputName).getInfo();
            long[] declared = inputInfo.getShape();
            long[] shape = declared.length == 1
                    ? new long[]{audioSamples.length}
                    : new long[]{1, audioSamples.length};
            OnnxTensor audioTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(audioSamples), shape);
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, audioTensor);
            try {
                return preprocessSession.run(inputs);
            } finally {
                audioTensor.close();
            }
        } catch (Exception e) {
            Log.w(TAG, "ONNX preprocessing failed, using Java DSP fallback: " + e.getMessage());
            return null;
        }
    }

    /**
     * Runs the encoder and returns the OPEN Result that owns the output
     * tensor. Callers must keep it alive while decoding (the decoder consumes
     * Result-owned tensors directly) and close it when done — no Java-side
     * clone of the encoder output.
     */
    private OrtSession.Result runEncoder(OnnxTensor inputFeatures) throws OrtException {
        if (encoderSession == null) return null;
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_features", inputFeatures);
        return encoderSession.run(inputs);
    }

    private OrtSession.Result runEncoder(float[][] melFeatures) throws OrtException {
        if (encoderSession == null || melFeatures == null) return null;
        long[] shape = new long[]{1, melFeatures.length, melFeatures[0].length};
        float[] flat = TensorUtils.flatten2D(melFeatures);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape);
        try {
            return runEncoder(inputTensor);
        } finally {
            inputTensor.close();
        }
    }

    private String runDecoder(OrtSession.Result encoderResult, String language) throws OrtException {
        if (decoderSession == null || encoderResult == null) return "";
        // Owned by encoderResult — valid until the caller closes it.
        OnnxTensor encoderOutput = (OnnxTensor) encoderResult.get(0);

        int langToken = language != null && LANGUAGE_TOKENS.containsKey(language)
                ? LANGUAGE_TOKENS.get(language)
                : LANGUAGE_TOKENS.get("vi");

        ArrayList<Integer> outputTokens = new ArrayList<>();
        long[] initialTokens = new long[]{SOT, langToken, TRANSCRIBE, NO_TIMESTAMPS};

        // Detect the decoder graph's cache contract.
        boolean hasCacheBranch = false;
        ArrayList<String> pastNames = new ArrayList<>();   // decoder.* and encoder.* pasts together
        for (String inputName : decoderSession.getInputNames()) {
            if (inputName.equals("use_cache_branch")) hasCacheBranch = true;
            else if (inputName.startsWith("past_key_values")) pastNames.add(inputName);
        }

        if (!pastNames.isEmpty()) {
            try {
                decodeWithCache(encoderOutput, initialTokens, outputTokens, hasCacheBranch, pastNames);
            } catch (Exception e) {
                // The cache path is validated, but stay consistent with
                // TranslationModule: on any cache-path failure, redo the
                // decode with the whole-sequence fallback instead of
                // returning partial output.
                Log.e(TAG, "KV-cache decode failed — retrying with whole-sequence "
                        + "decode: " + e.getMessage(), e);
                outputTokens.clear();
                decodeNoCache(encoderOutput, initialTokens, outputTokens, hasCacheBranch, pastNames);
            }
        } else {
            decodeNoCache(encoderOutput, initialTokens, outputTokens, hasCacheBranch, pastNames);
        }
        return decodeTokens(outputTokens);
    }

    /**
     * KV-cached greedy decode using the zero-copy pattern proven by
     * RTranslator: the decoder self-attention `present.*` tensors owned by
     * the previous step's Result are fed straight back in as
     * `past_key_values.*`, and the previous Result is closed only AFTER the
     * next run has consumed them. No Java-side cloning of the KV tensors
     * (the old loop deep-copied all of them through the Java heap on every
     * generated token).
     *
     * Cross-attention K/V are constant after prefill, so — like RTranslator's
     * cache initializer — the prefill Result stays open and its
     * `present.*.encoder.*` tensors are re-fed on EVERY step instead of
     * cycling the (potentially malformed on quantized graphs) encoder
     * presents of later steps.
     */
    private void decodeWithCache(OnnxTensor encoderOutput, long[] initialTokens,
                                 ArrayList<Integer> outputTokens,
                                 boolean hasCacheBranch, ArrayList<String> pastNames) throws OrtException {
        OrtSession.Result prefillResult = null;
        OrtSession.Result result = null;
        try {
            // Prefill: the 4 fixed prompt tokens in one pass, cache branch off.
            Map<String, OnnxTensor> inputs = new HashMap<>();
            ArrayList<OnnxTensor> created = new ArrayList<>();   // tensors this step owns (must close)
            addDecoderStepInputs(inputs, created, initialTokens, encoderOutput,
                    hasCacheBranch, false, pastNames, null, null);

            int currentToken = -1;
            while (true) {
                OrtSession.Result newResult;
                try {
                    newResult = decoderSession.run(inputs);
                } finally {
                    // The run consumed our small per-step tensors; past tensors
                    // inside `inputs` are owned by open Results, not by us.
                    for (OnnxTensor t : created) t.close();
                }
                // Free the previous step's Result only now. Never close the
                // prefill Result here: after the first iteration it is
                // aliased by `result`, and its encoder presents must stay
                // alive for every remaining step.
                if (result != null && result != prefillResult) result.close();
                result = newResult;
                if (prefillResult == null) prefillResult = newResult;

                float[][][] logits = (float[][][]) result.get(0).getValue();
                currentToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);
                if (currentToken == EOT) break;
                outputTokens.add(currentToken);
                if (outputTokens.size() >= MAX_TOKENS) break;

                // Next step: exactly one token, cache branch on. Decoder pasts
                // come from the previous step; encoder pasts from prefill.
                inputs = new HashMap<>();
                created = new ArrayList<>();
                addDecoderStepInputs(inputs, created, new long[]{currentToken}, encoderOutput,
                        hasCacheBranch, true, pastNames, result, prefillResult);
            }
        } finally {
            if (result != null) result.close();
            if (prefillResult != null && prefillResult != result) prefillResult.close();
        }
    }

    /**
     * Assembles one decoder step's input map. Tensors created here are
     * recorded in {@code created} for closing right after the run;
     * {@code encoderOutput} is owned by the encoder Result and Result-owned
     * past tensors are not closed here. {@code pastSource} null means
     * prefill (empty past); on decode steps, decoder pasts cycle from
     * {@code pastSource} while encoder pasts are re-fed from
     * {@code prefillSource}.
     */
    private void addDecoderStepInputs(Map<String, OnnxTensor> inputs, ArrayList<OnnxTensor> created,
                                      long[] tokenIds, OnnxTensor encoderOutput,
                                      boolean hasCacheBranch, boolean useCache,
                                      ArrayList<String> pastNames,
                                      OrtSession.Result pastSource, OrtSession.Result prefillSource) throws OrtException {
        OnnxTensor ids = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(tokenIds),
                new long[]{1, tokenIds.length});
        inputs.put("input_ids", ids);
        created.add(ids);
        inputs.put("encoder_hidden_states", encoderOutput);
        if (hasCacheBranch) {
            OnnxTensor flag = TensorUtils.booleanToTensor(env, useCache);
            inputs.put("use_cache_branch", flag);
            created.add(flag);
        }
        for (String n : pastNames) {
            // Encoder pasts re-feed from prefill once decoding has started.
            OrtSession.Result source =
                    (pastSource != null && n.contains(".encoder.")) ? prefillSource : pastSource;
            OnnxTensor t = null;
            if (source != null) {
                String presentName = n.replaceFirst("^past_key_values\\.", "present.");
                java.util.Optional<OnnxValue> v = source.get(presentName);
                if (v.isPresent()) t = (OnnxTensor) v.get();
                // A missing `present` output mid-decode violates the merged-export
                // contract — fail loudly instead of feeding an empty past (which
                // corrupts the cross-attention cache).
                else throw new OrtException("decoder did not return '" + presentName + "'");
            }
            if (t == null) {
                // Whisper-small: 12 heads x 64 head_dim (batch=1, seq=0).
                t = TensorUtils.createFloatTensor(env, new long[]{1, 12, 0, 64});
                created.add(t);
            }
            inputs.put(n, t);
        }
    }

    /**
     * Whole-sequence fallback (O(n^2)): re-feeds the accumulated tokens with
     * use_cache_branch=false and empty past tensors each step. Builds inputs
     * via addDecoderStepInputs so the FULL input contract is fed — a merged
     * graph still requires use_cache_branch and every past_key_values input
     * on each run ("Missing Input" otherwise).
     */
    private void decodeNoCache(OnnxTensor encoderOutput, long[] initialTokens,
                               ArrayList<Integer> outputTokens,
                               boolean hasCacheBranch, ArrayList<String> pastNames) throws OrtException {
        long[] currentTokens = initialTokens.clone();
        for (int i = 0; i < MAX_TOKENS; i++) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            ArrayList<OnnxTensor> created = new ArrayList<>();
            addDecoderStepInputs(inputs, created, currentTokens, encoderOutput,
                    hasCacheBranch, false, pastNames, null, null);
            try (OrtSession.Result result = decoderSession.run(inputs)) {
                float[][][] logits = (float[][][]) result.get(0).getValue();
                int nextToken = TensorUtils.argmax(logits[0][logits[0].length - 1]);
                if (nextToken == EOT) break;
                outputTokens.add(nextToken);
                long[] nextTokens = new long[currentTokens.length + 1];
                System.arraycopy(currentTokens, 0, nextTokens, 0, currentTokens.length);
                nextTokens[currentTokens.length] = nextToken;
                currentTokens = nextTokens;
            } finally {
                for (OnnxTensor t : created) t.close();
            }
        }
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
                        // Store the raw byte-level-BPE token string as-is;
                        // decodeTokens() reverses the byte mapping and
                        // re-decodes as UTF-8 (see BYTE_TO_UNICODE above).
                        vocabMap.put(id, key);
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
        if (vocabMap == null || vocabMap.isEmpty()) return "[No vocab]";

        // Reassemble the raw byte sequence first: every character in every
        // token is a byte-level-BPE placeholder that maps back to exactly
        // one byte via UNICODE_TO_BYTE. A single UTF-8 character's bytes
        // may be split across adjacent tokens, so we must concatenate all
        // bytes and decode as UTF-8 only at the very end.
        byte[] bytes = new byte[tokenIds.size() * 16];
        int len = 0;
        for (int id : tokenIds) {
            if (id >= 50257) continue;  // skip special tokens
            String piece = vocabMap.get(id);
            if (piece == null) continue;
            for (int i = 0; i < piece.length(); i++) {
                Integer b = UNICODE_TO_BYTE.get(piece.charAt(i));
                if (b == null) continue;  // unexpected character, skip
                if (len == bytes.length) {
                    byte[] grown = new byte[bytes.length * 2];
                    System.arraycopy(bytes, 0, grown, 0, len);
                    bytes = grown;
                }
                bytes[len++] = (byte) (int) b;
            }
        }

        try {
            return new String(bytes, 0, len, "UTF-8").trim();
        } catch (java.io.UnsupportedEncodingException e) {
            return "";
        }
    }

    /**
     * Java DSP fallback when whisper_preprocess.onnx is unavailable. The
     * log-mel is computed for the audio's REAL length only; the mel matrix is
     * padded to the fixed 3000-frame window afterwards (the normalization
     * loop below fills beyond numFrames with the clamp floor). The old code
     * padded the PCM to 30 s FIRST, making a 3 s clip pay for all 3000 FFT
     * frames (~10x the necessary DSP work).
     */
    private float[][] extractMelFeatures(float[] audioSamples) {
        if (audioSamples == null || audioSamples.length == 0) return new float[N_MELS][MAX_AUDIO_FRAMES];

        float maxAbs = 0;
        for (float s : audioSamples) if (Math.abs(s) > maxAbs) maxAbs = Math.abs(s);
        Log.i(TAG, "Max audio level: " + maxAbs);

        int numFrames = audioSamples.length / HOP_LENGTH;
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

        // Whisper normalization: log-mel is clamped to [max-8, max], then
        // normalized as (val + 4.0) / 4.0  (the +4 accounts for the typical
        // log-mel floor, matching the Whisper processor's fixed offset).
        float clampMin = maxVal - 8.0f;
        for (int m = 0; m < N_MELS; m++) {
            for (int f = 0; f < MAX_AUDIO_FRAMES; f++) {
                float val = (f < numFrames) ? melSpec[m][f] : clampMin;
                val = Math.max(val, clampMin);
                melSpec[m][f] = (val + 4.0f) / 4.0f;
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
            if (dataSize <= 0) return null;

            byte[] audioBytes = new byte[dataSize];
            int totalRead = 0;
            while (totalRead < dataSize) {
                int n = fis.read(audioBytes, totalRead, dataSize - totalRead);
                if (n < 0) break;
                totalRead += n;
            }
            if (totalRead <= 0) return null;

            int bytesPerSample = bitsPerSample / 8;
            int numSamples = totalRead / (bytesPerSample * numChannels);
            float[] samples = new float[numSamples];
            ByteBuffer buffer = ByteBuffer.wrap(audioBytes, 0, totalRead).order(ByteOrder.LITTLE_ENDIAN);
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

    /** Like {@link #copyModelToInternal}, but returns null when the asset is
     *  not bundled in the APK (used for the optional preprocess graph). */
    private String copyModelToInternalOptional(Context context, String assetName) {
        try {
            context.getAssets().open(assetName).close();
        } catch (IOException e) {
            return null;
        }
        return copyModelToInternal(context, assetName);
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
