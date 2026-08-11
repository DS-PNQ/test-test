/*
 * OmniVoice — TTS Module
 *
 * Primary backend: MMS-TTS (VITS) via ONNX Runtime.
 * Fallback backend: Android system TextToSpeech (used when an MMS-TTS ONNX
 * model for a language isn't bundled in assets).
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.TensorUtils;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.extensions.OrtxPackage;


/**
 * Text-to-Speech module with dual backend:
 * <ol>
 *   <li><b>MMS-TTS ONNX</b> — higher quality, loaded per-language from assets
 *       (exported by {@code optimize/export_mms_tts.py}).</li>
 *   <li><b>Android system TTS</b> — fallback for languages without a bundled
 *       ONNX model.</li>
 * </ol>
 *
 * <p>Follows the ONNX-Runtime + NNAPI pattern used by {@link TranslationModule},
 * and writes PCM WAV files compatible with
 * {@link com.omnivoice.onspeak47.audio.AudioPlayer}.</p>
 */
public class TTSModule {

    private static final String TAG = "TTSModule";
    private static final String[] SUPPORTED = {"vi", "en", "zh"};

    private final Context context;

    // ---- Android system TTS (fallback) --------------------------------
    private TextToSpeech androidTTS;
    private volatile boolean systemTtsReady = false;

    /** Human-readable reason for the most recent synthesis failure (surfaced in the UI). */
    private volatile String lastError = null;

    /** Reason the last {@link #synthesize} call failed, or null if it succeeded. */
    public String getLastError() { return lastError; }

    // ---- MMS-TTS ONNX (primary) ---------------------------------------
    private final Map<String, OrtSession> ortSessions = new HashMap<>();
    private final Map<String, MmsTtsTokenizer> tokenizers = new HashMap<>();
    private final Map<String, Integer> sampleRates = new HashMap<>();
    private OrtEnvironment env;
    private File baseDir;

    /** VITS default sample rate; overridden by each model's config.json. */
    private static final int DEFAULT_SAMPLE_RATE = 16000;

    // Language locale mapping for the system-TTS fallback
    private static final Map<String, Locale> LOCALES = new HashMap<>();
    static {
        LOCALES.put("vi", new Locale("vi"));
        LOCALES.put("en", Locale.ENGLISH);
        LOCALES.put("zh", Locale.CHINESE);
        LOCALES.put("zh_hans", Locale.SIMPLIFIED_CHINESE);
        LOCALES.put("zh_hant", Locale.TRADITIONAL_CHINESE);
    }

    /**
     * Initialize the TTS module.
     *
     * Sets up the Android system TTS and probes for bundled MMS-TTS ONNX
     * assets for all supported languages.
     */
    public TTSModule(Context context) {
        this.context = context;
        initSystemTts();
        try {
            initMmsOnnx();
        } catch (Exception e) {
            Log.e(TAG, "MMS-TTS ONNX init failed — system TTS will be used", e);
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Synthesize text to a WAV file.
     *
     * @param text       Text to synthesize
     * @param language   Language code ("vi", "en", "zh")
     * @param outputPath Path for the output WAV file
     * @return Path to the generated audio file, or null on failure
     */
    public String synthesize(String text, String language, String outputPath) {
        long startTime = System.currentTimeMillis();
        String lang = normalizeLang(language);
        lastError = null;

        // Primary path: MMS-TTS ONNX
        if (hasOnnxModel(lang) && runOnnxSynthesis(text, lang, outputPath)) {
            Log.i(TAG, "TTS synthesis (MMS-TTS ONNX) done in "
                    + (System.currentTimeMillis() - startTime) + "ms");
            return outputPath;
        }

        // Fallback: Android system TTS
        if (synthesizeWithSystemTts(text, lang, outputPath)) {
            Log.i(TAG, "TTS synthesis (system TTS fallback) done in "
                    + (System.currentTimeMillis() - startTime) + "ms");
            return outputPath;
        }

        if (lastError == null) {
            lastError = "All TTS backends failed for [" + lang + "] (no ONNX model, system TTS failed)";
        }
        Log.e(TAG, "TTS synthesis failed for language: " + lang + " — " + lastError);
        return null;
    }

    /**
     * Speak text immediately (without saving to file).
     *
     * Uses system TTS directly for instant feedback.
     */
    public void speak(String text, String language) {
        if (!systemTtsReady || androidTTS == null) return;
        Locale locale = LOCALES.getOrDefault(normalizeLang(language), Locale.ENGLISH);
        androidTTS.setLanguage(locale);
        androidTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null, "omnivoice_speak");
    }

    /** Whether a bundled MMS-TTS ONNX model is available for the language. */
    public boolean hasOnnxSupport(String language) {
        return hasOnnxModel(normalizeLang(language));
    }

    /**
     * Release TTS resources.
     */
    public void release() {
        if (androidTTS != null) {
            androidTTS.stop();
            androidTTS.shutdown();
            androidTTS = null;
        }
        for (OrtSession s : ortSessions.values()) {
            try { s.close(); } catch (Exception ignored) {}
        }
        ortSessions.clear();
        tokenizers.clear();
        sampleRates.clear();
    }

    // ------------------------------------------------------------------
    // System TTS (fallback backend)
    // ------------------------------------------------------------------

    private void initSystemTts() {
        CountDownLatch latch = new CountDownLatch(1);
        androidTTS = new TextToSpeech(context.getApplicationContext(), status -> {
            systemTtsReady = (status == TextToSpeech.SUCCESS);
            Log.i(TAG, systemTtsReady ? "Android system TTS initialized"
                                      : "Android system TTS init failed");
            latch.countDown();
        });
        if (Looper.myLooper() != Looper.getMainLooper()) {
            try { latch.await(10, TimeUnit.SECONDS); }
            catch (InterruptedException e) { Log.e(TAG, "TTS init interrupted", e); }
        }
    }

    private boolean synthesizeWithSystemTts(String text, String language, String outputPath) {
        if (!systemTtsReady || androidTTS == null) {
            lastError = "System TTS engine not ready";
            return false;
        }

        Locale locale = LOCALES.getOrDefault(language, Locale.ENGLISH);
        int lr = androidTTS.setLanguage(locale);
        if (lr == TextToSpeech.LANG_MISSING_DATA || lr == TextToSpeech.LANG_NOT_SUPPORTED) {
            Log.w(TAG, "System TTS language not supported: " + language);
        }

        File outputFile = new File(outputPath);
        outputFile.getParentFile().mkdirs();

        CountDownLatch latch = new CountDownLatch(1);
        final boolean[] success = {false};
        androidTTS.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override public void onStart(String id) {}
            @Override public void onDone(String id) { success[0] = true; latch.countDown(); }
            @Override public void onError(String id) {
                lastError = "System TTS utterance error";
                latch.countDown();
            }
        });

        if (androidTTS.synthesizeToFile(text, null, outputFile, "omnivoice_tts")
                != TextToSpeech.SUCCESS) {
            lastError = "System TTS synthesizeToFile rejected the request";
            return false;
        }
        try { latch.await(30, TimeUnit.SECONDS); }
        catch (InterruptedException e) { Log.e(TAG, "System TTS interrupted", e); }
        return success[0];
    }

    // ------------------------------------------------------------------
    // MMS-TTS ONNX backend
    // ------------------------------------------------------------------

    private void initMmsOnnx() throws OrtException {
        env = OrtEnvironment.getEnvironment();

        baseDir = context.getExternalFilesDir(null);
        if (baseDir == null) baseDir = context.getFilesDir();

        // Probe + load a session for every supported language whose ONNX
        // asset is actually bundled in the APK.
        for (String lang : SUPPORTED) {
            String onnxName = "mms_tts_" + lang + ".onnx";
            String vocabName = "mms_tts_" + lang + "_vocab.json";
            try {
                // Only copy if present in assets
                context.getAssets().open(onnxName).close();
                context.getAssets().open(vocabName).close();

                String onnxPath = FileUtils.copyAssetToDir(context, onnxName, baseDir);
                String vocabPath = FileUtils.copyAssetToDir(context, vocabName, baseDir);

                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                try {
                    opts.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                } catch (OrtException e) { /* extensions optional */ }
                // NOTE: NNAPI is intentionally NOT enabled for MMS-TTS (VITS).
                // Unlike Whisper/NLLB, the VITS decoder graph is built from ops
                // NNAPI does not support: 1D Conv (NNAPI is 2D-only),
                // RandomNormalLike (the noise sampling node), 1D Resize/Upsample
                // and dynamic-shape Scatter/CumSum duration expansion. Registering
                // NNAPI makes ORT partition the graph at createSession and then
                // throw inside session.run() on the dynamic-sequence Conv1d /
                // RandomNormalLike boundary — which the catch in runOnnxSynthesis
                // swallows, producing the silent "TTS: 0ms" failure. VITS runs
                // fast enough on the CPU EP for these short utterances.
                Log.i(TAG, "TTS [" + lang + "] using CPU EP (NNAPI disabled — unsupported VITS ops)");

                ortSessions.put(lang, env.createSession(onnxPath, opts));
                tokenizers.put(lang, new MmsTtsTokenizer(vocabPath));
                sampleRates.put(lang, loadSampleRate(lang));
                opts.close();
                Log.i(TAG, "Loaded MMS-TTS ONNX for [" + lang + "] @ "
                        + sampleRates.get(lang) + "Hz");
                Log.i(TAG, "MMS-TTS [" + lang + "] model inputs: "
                        + ortSessions.get(lang).getInputNames());
            } catch (OrtException e) {
                ortSessions.remove(lang);
                tokenizers.remove(lang);
                sampleRates.remove(lang);
                Log.e(TAG, "Failed to load MMS-TTS ONNX for [" + lang
                        + "] — system TTS fallback will be used", e);
            } catch (IOException e) {
                Log.i(TAG, "No bundled MMS-TTS asset for [" + lang + "] (" + onnxName + ")");
            }
        }
    }

    private boolean hasOnnxModel(String language) {
        return ortSessions.containsKey(language) && tokenizers.containsKey(language);
    }

    /**
     * Run VITS inference with ONNX Runtime.
     *
     * Expected graph I/O (Optimum `ORTModelForTextToWaveform` export):
     *   input  "input_ids"   : int64 [batch, sequence]
     *   output "waveform"    : float32 [batch, 1, time]
     */
    private boolean runOnnxSynthesis(String text, String language, String outputPath) {
        OrtSession session = ortSessions.get(language);
        MmsTtsTokenizer tokenizer = tokenizers.get(language);
        if (session == null || tokenizer == null) return false;

        try {
            List<String> chunks = chunkText(text, MAX_CHARS);
            List<float[]> waveforms = new ArrayList<>();
            for (String chunk : chunks) {
                float[] w = synthesizeOnnxChunk(chunk, language, session, tokenizer);
                if (w == null) return false; // error already logged
                if (w.length > 0) waveforms.add(w);
            }
            if (waveforms.isEmpty()) {
                lastError = "ONNX synthesis produced no audio";
                Log.e(TAG, "ONNX synthesis produced no audio for: " + text);
                return false;
            }

            int total = 0;
            for (float[] w : waveforms) total += w.length;
            float[] waveform = new float[total];
            int pos = 0;
            for (float[] w : waveforms) {
                System.arraycopy(w, 0, waveform, pos, w.length);
                pos += w.length;
            }

            int sampleRate = sampleRates.getOrDefault(language, DEFAULT_SAMPLE_RATE);
            writeWav(outputPath, waveform, sampleRate);
            Log.i(TAG, "ONNX synthesis OK [" + language + "]: " + chunks.size()
                    + " chunk(s), " + waveform.length + " samples @ " + sampleRate + "Hz");
            return true;
        } catch (Exception e) {
            lastError = "ONNX inference failed [" + language + "]: "
                    + e.getClass().getSimpleName() + ": " + e.getMessage();
            Log.e(TAG, "ONNX synthesis failed for [" + language + "]: " + lastError, e);
            return false;
        }
    }

    /**
     * Synthesize one text chunk through ONNX. Returns null on failure,
     * or an empty float array when the tokenizer produced no tokens
     * (punctuation-only chunk) so other chunks still synthesize.
     */
    private float[] synthesizeOnnxChunk(String text, String language,
                                        OrtSession session, MmsTtsTokenizer tokenizer)
            throws OrtException {
        int[] ids = tokenizer.encode(text, language);
        if (ids.length == 0) {
            Log.w(TAG, "Tokenizer produced no tokens for: " + text);
            return new float[0];
        }

        OnnxTensor input = null;
        OrtSession.Result result = null;
        try {
            // Prefer "input_ids" (Optimum MMS-TTS export), but fall back to the
            // model's actual first input name — a wrong hardcoded name throws
            // instantly inside run() and shows up as a silent "TTS: 0ms".
            java.util.Set<String> inputNames = session.getInputNames();
            String inputName = inputNames.contains("input_ids")
                    ? "input_ids"
                    : inputNames.iterator().next();

            Map<String, OnnxTensor> feeds = new HashMap<>();
            input = TensorUtils.intArrayToTensor(env, ids);
            feeds.put(inputName, input);

            // Some MMS-TTS ONNX exports declare extra inputs besides
            // "input_ids" (e.g. "attention_mask" from the HF Optimum export,
            // or explicit VITS hyper-params such as "speaker_id" /
            // "noise_scale"). ORT throws "Missing Input: <name>" inside run()
            // when any declared input isn't provided — that failure happens
            // before real work, producing the "TTS: 0-1ms" symptom. Feed
            // sensible defaults for every declared input the model asks for.
            List<OnnxTensor> auxTensors = new ArrayList<>();
            for (String name : inputNames) {
                if (name.equals(inputName) || feeds.containsKey(name)) continue;
                OnnxTensor t = createDefaultInputTensor(name, ids.length);
                if (t == null) {
                    Log.e(TAG, "Unsupported extra ONNX input '" + name
                            + "' — cannot synthesize [" + language + "]");
                    return null;
                }
                auxTensors.add(t);
                feeds.put(name, t);
                Log.i(TAG, "Feeding default for ONNX input '" + name + "'");
            }

            result = session.run(feeds);

            // Close the auxiliary default tensors once inference has consumed them.
            for (OnnxTensor t : auxTensors) {
                try { t.close(); } catch (Exception ignored) {}
            }

            OnnxValue waveformVal = null;
            for (String name : session.getOutputNames()) {
                String ln = name.toLowerCase();
                if (ln.contains("waveform") || ln.contains("audio") || ln.contains("wav")) {
                    waveformVal = result.get(name).get();
                    break;
                }
            }
            if (waveformVal == null) {
                waveformVal = result.get(0);
            }
            if (waveformVal == null) {
                Log.e(TAG, "ONNX output is missing for [" + language + "]");
                return null;
            }

            float[] waveform = flattenWaveform(waveformVal);
            if (waveform.length == 0) {
                Log.w(TAG, "ONNX returned empty waveform for: " + text);
            }
            return trimSilence(waveform);
        } finally {
            if (result != null) try { result.close(); } catch (Exception ignored) {}
            if (input != null) try { input.close(); } catch (Exception ignored) {}
        }
    }

    /**
     * Build a default tensor for a non-token ONNX input declared by the
     * MMS-TTS model. Returns null when the input's element type is
     * unsupported by this app.
     *
     * <p>Defaults follow the HF MMS/Optimum conventions: an all-ones
     * [1, seqLen] attention mask, single-speaker id 0, and VITS
     * noise/length scales used by the exported MMS models.</p>
     */
    private OnnxTensor createDefaultInputTensor(String name, int seqLen)
            throws OrtException {
        TensorInfo info = getTensorInfo(name);
        if (info == null) return null;

        float[] fvals = null;
        long[] shape;

        String lower = name.toLowerCase(Locale.ROOT);

        switch (info.type) {
            case FLOAT:
                if (lower.contains("noise_scale_duration") || lower.equals("duration_noise_scale")) {
                    fvals = new float[]{0.667f};
                    shape = new long[]{1};
                } else if (lower.contains("noise_scale")) {
                    fvals = new float[]{0.6f};
                    shape = new long[]{1};
                } else if (lower.contains("length_scale") || lower.contains("speaking_rate")) {
                    fvals = new float[]{1.0f};
                    shape = new long[]{1};
                } else if (lower.contains("attention_mask") || lower.contains("mask")) {
                    fvals = allOnesFloat(seqLen);
                    shape = new long[]{1, seqLen};
                } else if (lower.contains("speaker")) {
                    fvals = new float[]{0f};
                    shape = new long[]{1};
                } else {
                    fvals = new float[]{1.0f};
                    shape = new long[]{1};
                }
                break;

            case DOUBLE:
                if (lower.contains("noise_scale_duration") || lower.equals("duration_noise_scale")) {
                    double[] d = new double[]{0.667};
                    return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1});
                } else if (lower.contains("noise_scale")) {
                    double[] d = new double[]{0.6};
                    return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1});
                } else if (lower.contains("attention_mask") || lower.contains("mask")) {
                    double[] d = new double[seqLen];
                    java.util.Arrays.fill(d, 1.0);
                    return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1, seqLen});
                }
                double[] d = new double[]{1.0};
                return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1});

            case INT64:
            case INT32:
            case INT16:
            case INT8:
            case UINT8:
                long[] lvals;
                if (lower.contains("attention_mask") || lower.contains("mask")) {
                    lvals = allOnesLong(seqLen);
                    shape = new long[]{1, seqLen};
                } else if (lower.contains("speaker")) {
                    lvals = new long[]{0L};
                    shape = new long[]{1};
                } else {
                    lvals = new long[]{1L};
                    shape = new long[]{1};
                }

                if (info.type == OnnxJavaType.INT64) {
                    return OnnxTensor.createTensor(env, LongBuffer.wrap(lvals), shape);
                } else if (info.type == OnnxJavaType.INT32) {
                    int[] iv = toInt(lvals);
                    return OnnxTensor.createTensor(env, java.nio.IntBuffer.wrap(iv), shape);
                } else if (info.type == OnnxJavaType.INT16) {
                    short[] sv = toShort(lvals);
                    return OnnxTensor.createTensor(env, java.nio.ShortBuffer.wrap(sv), shape);
                } else { // INT8 / UINT8
                    byte[] bv = toByte(lvals);
                    return OnnxTensor.createTensor(env, ByteBuffer.wrap(bv), shape);
                }

            case BOOL:
                return OnnxTensor.createTensor(env, new boolean[]{false});

            default:
                return null;
        }

        if (fvals != null) {
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(fvals), shape);
        }
        return null;
    }

    private static float[] allOnesFloat(int n) {
        float[] a = new float[n];
        Arrays.fill(a, 1.0f);
        return a;
    }

    private static long[] allOnesLong(int n) {
        long[] a = new long[n];
        Arrays.fill(a, 1L);
        return a;
    }

    private static int[] toInt(long[] src) {
        int[] out = new int[src.length];
        for (int i = 0; i < src.length; i++) out[i] = (int) src[i];
        return out;
    }

    private static short[] toShort(long[] src) {
        short[] out = new short[src.length];
        for (int i = 0; i < src.length; i++) out[i] = (short) src[i];
        return out;
    }

    private static byte[] toByte(long[] src) {
        byte[] out = new byte[src.length];
        for (int i = 0; i < src.length; i++) out[i] = (byte) src[i];
        return out;
    }

    /** Fetch TensorInfo for the named input from the active session. */
    private TensorInfo getTensorInfo(String inputName) {
        for (OrtSession session : ortSessions.values()) {
            try {
                Object infoObj = session.getInputInfo().get(inputName).getInfo();
                if (infoObj instanceof TensorInfo) return (TensorInfo) infoObj;
            } catch (Exception ignored) {}
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Waveform → WAV helpers
    // ------------------------------------------------------------------

    /**
     * Flatten an OnnxValue (any of [time], [1,time], [1,1,time]) to float[].
     */
    private float[] flattenWaveform(OnnxValue value) throws OrtException {
        Object raw = value.getValue();
        if (raw instanceof float[]) {
            return (float[]) raw;
        } else if (raw instanceof float[][]) {
            return ((float[][]) raw)[0];
        } else if (raw instanceof float[][][]) {
            return ((float[][][]) raw)[0][0];
        }
        return new float[0];
    }

    /**
     * Trim near-silent samples from both ends of a waveform so concatenated
     * chunks don't carry long leading/trailing pauses.
     */
    private static float[] trimSilence(float[] w) {
        if (w == null || w.length == 0) return w;
        final float threshold = 5e-4f;
        final int margin = 400; // keep a short pad to avoid clipped phonemes
        int start = 0;
        while (start < w.length && Math.abs(w[start]) < threshold) start++;
        int end = w.length - 1;
        while (end > start && Math.abs(w[end]) < threshold) end--;
        if (start >= end) return new float[0];
        start = Math.max(0, start - margin);
        end = Math.min(w.length - 1, end + margin);
        return Arrays.copyOfRange(w, start, end + 1);
    }

    /**
     * Split text into chunks of at most {@code maxChars} characters,
     * preferring sentence boundaries (". " / "! " / "? " / newlines) so a
     * chunk never starts mid-word when a sentence split is available.
     */
    private static List<String> chunkText(String text, int maxChars) {
        List<String> out = new ArrayList<>();
        if (text == null) return out;
        String remaining = text.trim();
        while (remaining.length() > maxChars) {
            int cut = -1;
            for (int i = maxChars; i > maxChars / 2; i--) {
                char c = remaining.charAt(i - 1);
                if (c == '.' || c == '!' || c == '?' || c == '\n') { cut = i; break; }
            }
            if (cut < 0) {
                cut = remaining.lastIndexOf(' ', maxChars);
                if (cut < maxChars / 2) cut = maxChars;
            }
            String chunk = remaining.substring(0, cut).trim();
            if (!chunk.isEmpty()) out.add(chunk);
            remaining = remaining.substring(cut).trim();
        }
        if (!remaining.isEmpty()) out.add(remaining);
        if (out.isEmpty()) out.add(text.trim());
        return out;
    }

    /**
     * Read the model's sampling rate from the sibling
     * {@code mms_tts_<lang>_config.json} asset (falls back to
     * {@link #DEFAULT_SAMPLE_RATE}).
     */
    private int loadSampleRate(String lang) {
        String name = "mms_tts_" + lang + "_config.json";
        try (java.io.InputStream in = context.getAssets().open(name)) {
            byte[] data = new byte[in.available()];
            int n = in.read(data);
            String json = n > 0 ? new String(data, 0, n, "UTF-8") : new String(data, "UTF-8");
            JSONObject root = new JSONObject(json);
            int sr = root.optInt("sampling_rate", DEFAULT_SAMPLE_RATE);
            if (sr > 0) return sr;
        } catch (Exception e) {
            Log.d(TAG, "No/invalid " + name + " — using default sample rate");
        }
        return DEFAULT_SAMPLE_RATE;
    }

    /**
     * Maximum character count per ONNX inference call. VITS degrades / OOMs
     * on very long inputs, so long translations are chunked sentence-wise
     * and concatenated.
     */
    private static final int MAX_CHARS = 220;

    /**
     * Write a mono 16-bit PCM WAV file.
     */
    private void writeWav(String path, float[] samples, int sampleRate) throws IOException {
        File f = new File(path);
        f.getParentFile().mkdirs();

        int dataSize = samples.length * 2;
        ByteBuffer buf = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN);

        // RIFF header
        buf.put("RIFF".getBytes());
        buf.putInt(36 + dataSize);          // total chunk size
        buf.put("WAVE".getBytes());

        // fmt sub-chunk
        buf.put("fmt ".getBytes());
        buf.putInt(16);                     // PCM sub-chunk size
        buf.putShort((short) 1);            // PCM format
        buf.putShort((short) 1);            // mono
        buf.putInt(sampleRate);
        buf.putInt(sampleRate * 2);         // byte rate
        buf.putShort((short) 2);            // block align
        buf.putShort((short) 16);           // bits per sample

        // data sub-chunk
        buf.put("data".getBytes());
        buf.putInt(dataSize);
        for (float s : samples) {
            float clamped = Math.max(-1.0f, Math.min(1.0f, s));
            buf.putShort((short) (clamped * 32767f));
        }

        try (RandomAccessFile raf = new RandomAccessFile(f, "rw")) {
            raf.setLength(0);
            raf.write(buf.array());
        }
    }

    /** Normalize language codes like "zh_hans" / "zh_hant" to "zh". */
    private static String normalizeLang(String language) {
        if (language == null) return "vi";
        if (language.startsWith("zh")) return "zh";
        if (language.equals("vi") || language.equals("en")) return language;
        return language;
    }
}


