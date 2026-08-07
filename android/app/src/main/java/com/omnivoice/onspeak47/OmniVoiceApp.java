/*
 * OmniVoice — On-Device Speech Translation
 * Application class.
 */

package com.omnivoice.onspeak47;

import android.app.Application;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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
     * Initialize all three modules in parallel and create the pipeline
     * orchestrator once all are ready.
     *
     * Previously this method (and the LoadingActivity call site) chained the
     * three inits sequentially — ASR done, then start Translation, then start
     * TTS — even though the three modules are fully independent. That made
     * startup take asrMs + translationMs + ttsMs instead of roughly
     * max(asrMs, translationMs, ttsMs). Kicking all three off at once fixes that.
     */
    public void initializeAll(@NonNull AllInitListener listener) {
        if (asrModule != null && translationModule != null && ttsModule != null) {
            createOrchestrator();
            listener.onAllInitialized();
            return;
        }

        final int total = 3;
        AtomicInteger completed = new AtomicInteger(0);
        AtomicBoolean failed = new AtomicBoolean(false);

        initializeASR(trackingListener("ASR (Whisper Small)", completed, total, failed, listener));
        initializeTranslation(trackingListener("Translation (NLLB-200)", completed, total, failed, listener));
        initializeTTS(trackingListener("TTS (MMS-TTS)", completed, total, failed, listener));
    }

    /** Wraps a per-module InitListener that reports progress into the shared AllInitListener. */
    private InitListener trackingListener(String moduleName, AtomicInteger completed, int total,
                                           AtomicBoolean failed, AllInitListener listener) {
        return new InitListener() {
            @Override
            public void onInitialized() {
                int done = completed.incrementAndGet();
                listener.onModuleReady(moduleName, done, total);
                if (done == total && !failed.get()) {
                    createOrchestrator();
                    listener.onAllInitialized();
                }
            }
            @Override
            public void onError(String message) {
                // First failure wins; ignore further errors/successes so the
                // caller doesn't get onError() called more than once.
                if (failed.compareAndSet(false, true)) {
                    listener.onError(message);
                }
            }
        };
    }

    /**
     * Create orchestrator instance if modules are initialized.
     */
    public void createOrchestrator() {
        if (asrModule != null && translationModule != null && ttsModule != null) {
            orchestrator = new PipelineOrchestrator(asrModule, translationModule, ttsModule);
        }
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

    /** Progress-aware listener for initializeAll() (three modules loading in parallel). */
    public interface AllInitListener {
        /** Called on the main thread each time one of the 3 modules finishes loading. */
        void onModuleReady(String moduleName, int completedCount, int totalCount);
        void onAllInitialized();
        void onError(String message);
    }
}
