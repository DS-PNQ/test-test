/*
 * OmniVoice — TTS Module
 *
 * Uses Android's built-in TextToSpeech engine for output synthesis.
 * MMS-TTS ONNX inference can be added later when models are available.
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import com.omnivoice.onspeak47.pipeline.mms.MmsOnnxTTS;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;


/**
 * Text-to-Speech module with dual backend:
 *   1. MMS-TTS via ONNX Runtime (on-device, higher quality, preferred when
 *      the exported VITS graphs are present in assets)
 *   2. Android system TTS (fallback — works for all 3 languages with no
 *      model files at all)
 *
 * Falls back to Android TTS if ONNX models are not present.
 */
public class TTSModule {

    private static final String TAG = "TTSModule";

    private TextToSpeech androidTTS;
    private volatile boolean ttsReady = false;
    private final Context context;
    private final MmsOnnxTTS mmsOnnxTTS;

    // Language locale mapping
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
     */
    public TTSModule(Context context) {
        this.context = context;
        this.mmsOnnxTTS = new MmsOnnxTTS(context);
        CountDownLatch latch = new CountDownLatch(1);

        androidTTS = new TextToSpeech(context.getApplicationContext(), status -> {
            if (status == TextToSpeech.SUCCESS) {
                ttsReady = true;
                Log.i(TAG, "Android TTS initialized");
            } else {
                Log.e(TAG, "Android TTS initialization failed");
            }
            latch.countDown();
        });

        if (Looper.myLooper() != Looper.getMainLooper()) {
            try {
                latch.await(10, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Log.e(TAG, "TTS init interrupted", e);
            }
        }
    }

    /**
     * Synthesize text to a WAV file.
     *
     * @param text       Text to synthesize
     * @param language   Language code ("vi", "en", "zh")
     * @param outputPath Path for the output WAV file
     * @return Path to the generated audio file, or null on failure
     */
    public String synthesize(String text, String language, String outputPath) {
        // Backend A (preferred): MMS-TTS ONNX — on-device, no engine dependency.
        String viaOnnx = synthesizeWithOnnx(text, language, outputPath);
        if (viaOnnx != null) return viaOnnx;

        // Backend B (fallback): Android system TTS.
        return synthesizeWithAndroidTts(text, language, outputPath);
    }

    /**
     * MMS-TTS (VITS) ONNX path. Returns null when the model asset is absent
     * or inference fails, so the caller can degrade to Android TTS.
     */
    private String synthesizeWithOnnx(String text, String language, String outputPath) {
        if (!mmsOnnxTTS.isLanguageAvailable(language)) {
            return null;
        }

        long startTime = System.currentTimeMillis();
        float[] waveform = mmsOnnxTTS.synthesizeToArray(text, language);
        if (waveform == null || waveform.length == 0) {
            Log.w(TAG, "MMS-TTS ONNX returned no audio, falling back to Android TTS");
            return null;
        }

        try {
            writeWav16(outputPath, waveform, MmsOnnxTTS.NATIVE_SAMPLE_RATE);
        } catch (IOException e) {
            Log.e(TAG, "Failed writing MMS-TTS WAV", e);
            return null;
        }

        Log.i(TAG, "MMS-TTS ONNX synthesized " + waveform.length + " samples in "
                + (System.currentTimeMillis() - startTime) + "ms");
        return outputPath;
    }

    /**
     * Android system TTS fallback — original implementation.
     */
    private String synthesizeWithAndroidTts(String text, String language, String outputPath) {
        if (!ttsReady || androidTTS == null) {
            Log.e(TAG, "TTS not ready");
            return null;
        }

        long startTime = System.currentTimeMillis();

        // Set language
        Locale locale = LOCALES.getOrDefault(language, Locale.ENGLISH);
        int result = androidTTS.setLanguage(locale);
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            Log.w(TAG, "Language not supported for TTS: " + language + ", using default");
        }

        // Synthesize to file
        File outputFile = new File(outputPath);
        outputFile.getParentFile().mkdirs();

        CountDownLatch latch = new CountDownLatch(1);
        final boolean[] success = {false};

        androidTTS.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override
            public void onStart(String utteranceId) {
                Log.d(TAG, "TTS synthesis started");
            }

            @Override
            public void onDone(String utteranceId) {
                success[0] = true;
                latch.countDown();
            }

            @Override
            public void onError(String utteranceId) {
                Log.e(TAG, "TTS synthesis error");
                latch.countDown();
            }
        });

        int synthResult = androidTTS.synthesizeToFile(text, null, outputFile, "omnivoice_tts");

        if (synthResult != TextToSpeech.SUCCESS) {
            Log.e(TAG, "synthesizeToFile returned error: " + synthResult);
            return null;
        }

        try {
            latch.await(30, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "TTS synthesis interrupted", e);
        }

        long elapsed = System.currentTimeMillis() - startTime;
        Log.i(TAG, "TTS synthesis done in " + elapsed + "ms");

        return success[0] ? outputPath : null;
    }

    /**
     * Speak text immediately (without saving to file).
     */
    public void speak(String text, String language) {
        if (!ttsReady || androidTTS == null) return;

        Locale locale = LOCALES.getOrDefault(language, Locale.ENGLISH);
        androidTTS.setLanguage(locale);
        androidTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null, "omnivoice_speak");
    }

    /**
     * Write a normalized float waveform ([-1.0, 1.0]) to a 16-bit PCM
     * mono WAV file.
     */
    private static void writeWav16(String path, float[] samples, int sampleRate)
            throws IOException {
        File outFile = new File(path);
        if (outFile.getParentFile() != null) outFile.getParentFile().mkdirs();

        int dataSize = samples.length * 2;              // 16-bit = 2 bytes/sample
        int byteRate = sampleRate * 2;                  // mono 16-bit

        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(outFile))) {
            // RIFF header (little-endian fields written byte-swapped)
            out.writeBytes("RIFF");
            writeIntLE(out, 36 + dataSize);
            out.writeBytes("WAVE");
            out.writeBytes("fmt ");
            writeIntLE(out, 16);                        // PCM chunk size
            writeShortLE(out, (short) 1);               // PCM format
            writeShortLE(out, (short) 1);               // mono
            writeIntLE(out, sampleRate);
            writeIntLE(out, byteRate);
            writeShortLE(out, (short) 2);               // block align
            writeShortLE(out, (short) 16);              // bits per sample
            out.writeBytes("data");
            writeIntLE(out, dataSize);

            for (float s : samples) {
                float clamped = Math.max(-1.0f, Math.min(1.0f, s));
                writeShortLE(out, (short) (clamped * 32767.0f));
            }
        }
    }

    private static void writeIntLE(DataOutputStream out, int v) throws IOException {
        out.writeByte(v & 0xFF);
        out.writeByte((v >> 8) & 0xFF);
        out.writeByte((v >> 16) & 0xFF);
        out.writeByte((v >> 24) & 0xFF);
    }

    private static void writeShortLE(DataOutputStream out, short v) throws IOException {
        out.writeByte(v & 0xFF);
        out.writeByte((v >> 8) & 0xFF);
    }

    /**
     * Release TTS resources.
     */
    public void release() {
        mmsOnnxTTS.close();
        if (androidTTS != null) {
            androidTTS.stop();
            androidTTS.shutdown();
            androidTTS = null;
        }
    }
}
