/*
 * OmniVoice — On-Device Speech Translation
 * Application class.
 */

package com.omnivoice.onspeak47;

import android.app.Application;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.omnivoice.onspeak47.pipeline.ASRModule;
import com.omnivoice.onspeak47.pipeline.PipelineOrchestrator;
import com.omnivoice.onspeak47.pipeline.TTSModule;
import com.omnivoice.onspeak47.pipeline.TranslationModule;
import com.omnivoice.onspeak47.util.LanguageConfig;


/**
 * Global Application class — manages the lifecycle of the three
 * pipeline models (Whisper, NLLB, MMS-TTS).
 *
 * Modeled after RTranslator-2.00's {@code Global.java} but scoped
 * to the VN↔EN/VN↔CN language pairs only.
 */
public class OmniVoiceApp extends Application {

    private static final String TAG = "OmniVoiceApp";

    @Nullable private ASRModule asrModule;
    @Nullable private TranslationModule translationModule;
    @Nullable private TTSModule ttsModule;
    @Nullable private PipelineOrchestrator orchestrator;

    private Handler mainHandler;

    // ----------------------------------------------------------------
    // Lifecycle
    // ----------------------------------------------------------------

    @Override
    public void onCreate() {
        super.onCreate();
        mainHandler = new Handler(Looper.getMainLooper());
        Log.i(TAG, "OmniVoice application created");
    }

    // ----------------------------------------------------------------
    // Module initializers (lazy, called from LoadingActivity)
    // ----------------------------------------------------------------

    /**
     * Initialize the ASR (Whisper) module.
     */
    public void initializeASR(@NonNull InitListener listener) {
        if (asrModule != null) {
            listener.onInitialized();
            return;
        }
        new Thread(() -> {
            try {
                asrModule = new ASRModule(this);
                mainHandler.post(listener::onInitialized);
            } catch (Exception e) {
                Log.e(TAG, "ASR init failed", e);
                mainHandler.post(() -> listener.onError("ASR initialization failed: " + e.getMessage()));
            }
        }).start();
    }

    /**
     * Initialize the Translation (NLLB) module.
     */
    public void initializeTranslation(@NonNull InitListener listener) {
        if (translationModule != null) {
            listener.onInitialized();
            return;
        }
        new Thread(() -> {
            try {
                translationModule = new TranslationModule(this);
                mainHandler.post(listener::onInitialized);
            } catch (Exception e) {
                Log.e(TAG, "Translation init failed", e);
                mainHandler.post(() -> listener.onError("Translation initialization failed: " + e.getMessage()));
            }
        }).start();
    }

    /**
     * Initialize the TTS (MMS-TTS / Android TTS fallback) module.
     */
    public void initializeTTS(@NonNull InitListener listener) {
        if (ttsModule != null) {
            listener.onInitialized();
            return;
        }
        new Thread(() -> {
            try {
                ttsModule = new TTSModule(this);
                mainHandler.post(listener::onInitialized);
            } catch (Exception e) {
                Log.e(TAG, "TTS init failed", e);
                mainHandler.post(() -> listener.onError("TTS initialization failed: " + e.getMessage()));
            }
        }).start();
    }

    /**
     * Initialize all three modules and create the pipeline orchestrator.
     */
    public void initializeAll(@NonNull InitListener listener) {
        initializeASR(new InitListener() {
            @Override
            public void onInitialized() {
                initializeTranslation(new InitListener() {
                    @Override
                    public void onInitialized() {
                        initializeTTS(new InitListener() {
                            @Override
                            public void onInitialized() {
                                orchestrator = new PipelineOrchestrator(asrModule, translationModule, ttsModule);
                                listener.onInitialized();
                            }
                            @Override
                            public void onError(String message) { listener.onError(message); }
                        });
                    }
                    @Override
                    public void onError(String message) { listener.onError(message); }
                });
            }
            @Override
            public void onError(String message) { listener.onError(message); }
        });
    }

    // ----------------------------------------------------------------
    // Getters
    // ----------------------------------------------------------------

    @Nullable public ASRModule getASRModule() { return asrModule; }
    @Nullable public TranslationModule getTranslationModule() { return translationModule; }
    @Nullable public TTSModule getTTSModule() { return ttsModule; }
    @Nullable public PipelineOrchestrator getOrchestrator() { return orchestrator; }

    // ----------------------------------------------------------------
    // Listener interface
    // ----------------------------------------------------------------

    public interface InitListener {
        void onInitialized();
        void onError(String message);
    }
}
