/*
 * OmniVoice — TTS Module
 *
 * Uses Android's built-in TextToSpeech engine for output synthesis.
 * MMS-TTS ONNX inference can be added later when models are available.
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.media.MediaPlayer;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import java.io.File;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;


/**
 * Text-to-Speech module with dual backend:
 *   1. Android system TTS (works immediately for all 3 languages)
 *   2. MMS-TTS via ONNX Runtime (higher quality, loaded when models available)
 *
 * Falls back to Android TTS if ONNX models are not present.
 */
public class TTSModule {

    private static final String TAG = "TTSModule";

    private TextToSpeech androidTTS;
    private volatile boolean ttsReady = false;
    private final Context context;

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
     * Release TTS resources.
     */
    public void release() {
        if (androidTTS != null) {
            androidTTS.stop();
            androidTTS.shutdown();
            androidTTS = null;
        }
    }
}
