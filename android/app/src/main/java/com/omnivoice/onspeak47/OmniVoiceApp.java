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
 * pipeline models (Whisper, HY-MT translation, MMS-TTS).
 *
 * Translation is treated as optional: when the HY-MT model is not
 * yet installed, the app starts normally with ASR and TTS working;
 * translation returns a "not available" placeholder.
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
     * Initialize the Translation (HY-MT) module.
     *
     * <p>Always succeeds — if the HY-MT model is not installed the
     * module initializes in "unavailable" mode and translate() calls
     * return a placeholder.  This keeps the app usable for ASR + TTS
     * even before the translation model is set up.</p>
     */
    public void initializeTranslation(@NonNull InitListener listener) {
        if (translationModule != null) {
            listener.onInitialized();
            return;
        }
        new Thread(() -> {
            try {
                translationModule = new TranslationModule(this);
                if (!translationModule.isReady()) {
                    Log.w(TAG, "Translation module loaded but model unavailable "
                            + "— translate() will return placeholders");
                }
                mainHandler.post(listener::onInitialized);
            } catch (Exception e) {
                Log.e(TAG, "Translation init failed", e);
                // Non-fatal: let the app continue without translation
                translationModule = new TranslationModule(this);
                mainHandler.post(listener::onInitialized);
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
     * Initialize all three modules in parallel and create the pipeline orchestrator.
     * This significantly reduces startup time compared to sequential loading.
     */
    public void initializeAll(@NonNull InitListener listener) {
        new Thread(() -> {
            java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(3);
            final String[] errors = new String[3];

            // Load ASR in parallel
            new Thread(() -> {
                try {
                    asrModule = new ASRModule(OmniVoiceApp.this);
                    Log.i(TAG, "ASR module loaded");
                } catch (Exception e) {
                    Log.e(TAG, "ASR init failed", e);
                    errors[0] = "ASR initialization failed: " + e.getMessage();
                } finally {
                    latch.countDown();
                }
            }).start();

            // Load Translation in parallel (non-fatal if model unavailable)
            new Thread(() -> {
                try {
                    translationModule = new TranslationModule(OmniVoiceApp.this);
                    if (translationModule.isReady()) {
                        Log.i(TAG, "Translation module loaded (HY-MT)");
                    } else {
                        Log.w(TAG, "Translation module loaded but model unavailable");
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Translation init failed (non-fatal)", e);
                    // Create a stub so the rest of the pipeline can proceed
                    translationModule = new TranslationModule(OmniVoiceApp.this);
                } finally {
                    latch.countDown();
                }
            }).start();

            // Load TTS in parallel
            new Thread(() -> {
                try {
                    ttsModule = new TTSModule(OmniVoiceApp.this);
                    Log.i(TAG, "TTS module loaded");
                } catch (Exception e) {
                    Log.e(TAG, "TTS init failed", e);
                    errors[2] = "TTS initialization failed: " + e.getMessage();
                } finally {
                    latch.countDown();
                }
            }).start();

            try {
                latch.await(); // Wait for all three to complete
            } catch (InterruptedException e) {
                Log.e(TAG, "Model loading interrupted", e);
                mainHandler.post(() -> listener.onError("Model loading interrupted"));
                return;
            }

            // Check for fatal errors (ASR and TTS are required; translation is optional)
            if (errors[0] != null) {
                final String err = errors[0];
                mainHandler.post(() -> listener.onError(err));
                return;
            }
            if (errors[2] != null) {
                final String err = errors[2];
                mainHandler.post(() -> listener.onError(err));
                return;
            }

            // Create orchestrator (translation may be in stub/unavailable mode)
            orchestrator = new PipelineOrchestrator(asrModule, translationModule, ttsModule);
            mainHandler.post(listener::onInitialized);
        }).start();
    }

    /**
     * Create orchestrator instance if modules are initialized.
     */
    public void createOrchestrator() {
        if (asrModule != null && ttsModule != null) {
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
}
