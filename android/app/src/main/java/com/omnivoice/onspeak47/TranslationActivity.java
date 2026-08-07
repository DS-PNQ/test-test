/*
 * OmniVoice — Main Translation Activity
 *
 * Provides the walkie-talkie style UI: user presses a button to speak,
 * the pipeline transcribes → translates → speaks the result.
 */

package com.omnivoice.onspeak47;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.omnivoice.onspeak47.audio.AudioPlayer;
import com.omnivoice.onspeak47.audio.AudioRecorder;
import com.omnivoice.onspeak47.pipeline.PipelineOrchestrator;
import com.omnivoice.onspeak47.util.LanguageConfig;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


/**
 * Main translation screen with push-to-talk UI.
 *
 * Language directions supported:
 *   - Vietnamese ↔ English
 *   - Vietnamese ↔ Chinese (Simplified)
 */
public class TranslationActivity extends AppCompatActivity {

    private static final String TAG = "TranslationActivity";
    private static final int PERMISSION_REQUEST_AUDIO = 100;

    // UI elements
    private Button talkButton;
    private TextView transcriptView;
    private TextView translationView;
    private TextView timingView;
    private Spinner srcLangSpinner;
    private Spinner tgtLangSpinner;

    // Pipeline
    private PipelineOrchestrator orchestrator;
    private AudioRecorder recorder;
    private AudioPlayer player;
    // Reused across requests instead of a `new Thread()` per talk-button press;
    // single-thread so a new request naturally queues behind an in-flight one
    // rather than racing it.
    private final ExecutorService pipelineExecutor = Executors.newSingleThreadExecutor();

    // State
    private String srcLang = "vi";
    private String tgtLang = "en";
    private boolean isRecording = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_translation);

        // Get pipeline from Application
        OmniVoiceApp app = (OmniVoiceApp) getApplication();
        orchestrator = app.getOrchestrator();
        recorder = new AudioRecorder();
        player = new AudioPlayer();

        // Bind UI
        talkButton = findViewById(R.id.btn_talk);
        transcriptView = findViewById(R.id.txt_transcript);
        translationView = findViewById(R.id.txt_translation);
        timingView = findViewById(R.id.txt_timing);
        srcLangSpinner = findViewById(R.id.spinner_src_lang);
        tgtLangSpinner = findViewById(R.id.spinner_tgt_lang);

        setupLanguageSpinners();
        setupTalkButton();
        requestAudioPermission();
    }

    // ----------------------------------------------------------------
    // Language selection
    // ----------------------------------------------------------------

    private void setupLanguageSpinners() {
        String[] languages = LanguageConfig.getDisplayNames();

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_dropdown_item, languages
        );

        srcLangSpinner.setAdapter(adapter);
        tgtLangSpinner.setAdapter(adapter);

        // Defaults: vi → en
        srcLangSpinner.setSelection(LanguageConfig.indexOf("vi"));
        tgtLangSpinner.setSelection(LanguageConfig.indexOf("en"));

        srcLangSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                srcLang = LanguageConfig.getCodeAtIndex(pos);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        tgtLangSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                tgtLang = LanguageConfig.getCodeAtIndex(pos);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    // ----------------------------------------------------------------
    // Push-to-talk button
    // ----------------------------------------------------------------

    private void setupTalkButton() {
        talkButton.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    startRecording();
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    stopRecordingAndTranslate();
                    return true;
            }
            return false;
        });
    }

    private void startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestAudioPermission();
            return;
        }
        if (isRecording) return;
        isRecording = true;
        talkButton.setText(R.string.btn_recording);
        transcriptView.setText("Listening...");
        translationView.setText("");
        timingView.setText("");

        File audioFile = new File(getCacheDir(), "recording.wav");
        recorder.startRecording(audioFile.getAbsolutePath());
    }

    private void stopRecordingAndTranslate() {
        if (!isRecording) return;
        isRecording = false;
        talkButton.setText(R.string.btn_talk);
        transcriptView.setText("Processing...");

        if (orchestrator == null) {
            OmniVoiceApp app = (OmniVoiceApp) getApplication();
            orchestrator = app.getOrchestrator();
        }

        String audioPath = recorder.stopRecording();
        if (audioPath == null || orchestrator == null) {
            transcriptView.setText("Error: no audio or pipeline not ready");
            return;
        }

        // Run pipeline in background
        pipelineExecutor.submit(() -> {
            try {
                PipelineOrchestrator.PipelineResult result =
                        orchestrator.process(audioPath, srcLang, tgtLang);

                runOnUiThread(() -> {
                    transcriptView.setText(result.transcript);
                    translationView.setText(result.translation);
                    timingView.setText(String.format(
                            "ASR: %dms | Translation: %dms | TTS: %dms | Total: %dms",
                            result.asrMs, result.translationMs, result.ttsMs, result.totalMs
                    ));
                });

                // Play the translated audio
                if (result.audioPath != null) {
                    player.play(result.audioPath);
                }

            } catch (Exception e) {
                Log.e(TAG, "Pipeline error", e);
                runOnUiThread(() -> transcriptView.setText("Error: " + e.getMessage()));
            }
        });
    }

    // ----------------------------------------------------------------
    // Permissions
    // ----------------------------------------------------------------

    private void requestAudioPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    PERMISSION_REQUEST_AUDIO
            );
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "Audio permission granted");
            } else {
                Log.w(TAG, "Audio permission denied — recording won't work");
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (player != null) player.release();
        pipelineExecutor.shutdownNow();
    }
}
