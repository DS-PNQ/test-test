/*
 * OmniVoice — Splash / Model Loading Activity
 */

package com.omnivoice.onspeak47;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;


/**
 * Loading screen shown at launch while the three ONNX models
 * (Whisper, NLLB, MMS-TTS) are initialized.
 */
public class LoadingActivity extends AppCompatActivity {

    private static final String TAG = "LoadingActivity";

    private ProgressBar progressBar;
    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_loading);

        progressBar = findViewById(R.id.loading_progress);
        statusText = findViewById(R.id.loading_status);

        startModelLoading();
    }

    private void startModelLoading() {
        OmniVoiceApp app = (OmniVoiceApp) getApplication();

        updateStatus("Loading ASR, Translation and TTS models...", 0);

        // The three modules are independent, so they now load in parallel
        // (previously chained ASR → Translation → TTS, which took the sum of
        // all three load times instead of roughly the slowest one).
        app.initializeAll(new OmniVoiceApp.AllInitListener() {
            @Override
            public void onModuleReady(String moduleName, int completedCount, int totalCount) {
                updateStatus(
                        "Loaded " + moduleName + " (" + completedCount + "/" + totalCount + ")",
                        completedCount * 100 / totalCount
                );
            }

            @Override
            public void onAllInitialized() {
                updateStatus("Ready!", 100);
                navigateToTranslation();
            }

            @Override
            public void onError(String message) {
                showError(message);
            }
        });
    }

    private void updateStatus(String message, int progress) {
        Log.i(TAG, message);
        statusText.setText(message);
        progressBar.setProgress(progress);
    }

    private void showError(String message) {
        Log.e(TAG, "Loading error: " + message);
        statusText.setText("Error: " + message);
    }

    private void navigateToTranslation() {
        Intent intent = new Intent(this, TranslationActivity.class);
        startActivity(intent);
        finish();
    }
}
