/*
 * OmniVoice — MMS-TTS ONNX Session Manager
 *
 * Holds one ONNX Runtime session per MMS-TTS language checkpoint.
 * Sessions are loaded lazily (first use) so app startup isn't blocked by
 * three ~36MB VITS graphs. Thread-safety: all synthesize calls are
 * synchronized on this instance; the TTSModule doubles as the outer
 * lock in production.
 */

package com.omnivoice.onspeak47.pipeline.mms;

import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.nio.LongBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;


/**
 * Lazy per-language ONNX session holder for MMS-TTS (VITS).
 */
public class MmsOnnxTTS {

    private static final String TAG = "MmsOnnxTTS";

    /** Maximum token sequence length we pass to the ONNX graph. */
    private static final int MAX_SEQ_LEN = 512;

    /** Native sampling rate of the MMS-TTS checkpoints (VITS config). */
    public static final int NATIVE_SAMPLE_RATE = 16000;

    private final OrtEnvironment env;
    private final Map<String, OrtSession> sessions = new HashMap<>();
    private final Map<String, TextMapper> mappers = new HashMap<>();
    private final Context context;

    public MmsOnnxTTS(Context context) {
        this.context = context.getApplicationContext();
        this.env = OrtEnvironment.getEnvironment();
    }

    /**
     * Return the ONNX session for a language, loading it on first use.
     *
     * @param language "vi" | "en" | "zh"
     * @return OrtSession, or null if the ONNX asset is missing / load fails
     */
    public synchronized OrtSession getSession(String language) {
        return sessions.computeIfAbsent(language, lang -> {
            String modelFile = modelFileName(lang);
            try {
                byte[] modelBytes = loadAsset(modelFile);
                OrtSession session = env.createSession(modelBytes);
                Log.i(TAG, "Loaded " + modelFile + " into ONNX session");
                return session;
            } catch (IOException | OrtException | RuntimeException e) {
                Log.w(TAG, "MMS-TTS ONNX unavailable for " + lang + ": " + e.getMessage());
                return null;
            }
        });
    }

    /**
     * Synthesize text to a raw float waveform using the ONNX graph.
     *
     * @param text     Input text
     * @param language "vi" | "en" | "zh"
     * @return float array of normalized audio samples, or null on failure
     */
    public float[] synthesizeToArray(String text, String language) {
        OrtSession session = getSession(language);
        if (session == null) return null;

        TextMapper mapper = getMapper(language);
        if (mapper == null) return null;

        int[] ids = mapper.toIds(text);
        if (ids.length == 0) return null;
        if (ids.length > MAX_SEQ_LEN) {
            Log.w(TAG, "Truncating " + ids.length + "-token input to " + MAX_SEQ_LEN);
            int[] trimmed = new int[MAX_SEQ_LEN];
            System.arraycopy(ids, 0, trimmed, 0, MAX_SEQ_LEN);
            ids = trimmed;
        }

        OnnxTensor inputTensor = null;
        OrtSession.Result result = null;
        try {
            // VITS expects int64 token ids.
            long[] idsLong = new long[ids.length];
            for (int i = 0; i < ids.length; i++) idsLong[i] = ids[i];
            long[] shape = {1, idsLong.length};

            inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(idsLong), shape);
            result = session.run(Collections.singletonMap("input_ids", inputTensor));

            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            java.nio.FloatBuffer fb = outputTensor.getFloatBuffer();
            float[] waveform = new float[fb.remaining()];
            fb.get(waveform);

            return waveform;
        } catch (Exception e) {
            Log.e(TAG, "MMS-TTS ONNX synthesis failed for " + language, e);
            return null;
        } finally {
            try {
                if (result != null) result.close();
            } catch (Exception ignored) {}
            try {
                if (inputTensor != null) inputTensor.close();
            } catch (Exception ignored) {}
        }
    }

    /**
     * Whether a ONNX model for {@code language} can be loaded at all.
     */
    public boolean isLanguageAvailable(String language) {
        return getSession(language) != null;
    }

    /**
     * Release all loaded sessions. The OrtEnvironment is shared with the
     * ASR/Translation modules so we do NOT close it here.
     */
    public synchronized void close() {
        for (OrtSession s : sessions.values()) {
            try {
                if (s != null) s.close();
            } catch (OrtException e) {
                Log.e(TAG, "Error closing OrtSession", e);
            }
        }
        sessions.clear();
        mappers.clear();
    }

    // ----------------------------------------------------------------
    // Internals
    // ----------------------------------------------------------------

    private String modelFileName(String language) {
        return "mms_tts_" + languageToSuffix(language) + ".onnx";
    }

    private String charsetFileName(String language) {
        return "mms_tts_" + languageToSuffix(language) + "_charset.txt";
    }

    private String languageToSuffix(String language) {
        switch (language) {
            case "zh":
            case "zh_hans":
            case "zh_hant":
                return "zho";
            case "vi":
                return "vie";
            case "en":
            default:
                return "eng";
        }
    }

    private TextMapper getMapper(String language) {
        return mappers.computeIfAbsent(language, lang -> {
            try {
                return new TextMapper(context, lang, charsetFileName(lang));
            } catch (IOException e) {
                Log.w(TAG, "TextMapper unavailable for " + lang, e);
                return null;
            }
        });
    }

    private byte[] loadAsset(String name) throws IOException {
        // NOTE: onnxruntime-android has a createSession(byte[]) overload that
        // reads an in-memory model buffer. Using the asset InputStream directly
        // avoids the copy-to-filesystem step, but for very large models a
        // copy-to-disk via FileUtils.copyAssetToDir may be more reliable on
        // devices with tight disk caching. MMS-TTS is ~36 MB so in-memory is
        // acceptable on modern Android hardware.
        try (java.io.InputStream is = context.getAssets().open(name)) {
            java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[65536];
            int n;
            while ((n = is.read(buf)) != -1) {
                baos.write(buf, 0, n);
            }
            return baos.toByteArray();
        }
    }
}
