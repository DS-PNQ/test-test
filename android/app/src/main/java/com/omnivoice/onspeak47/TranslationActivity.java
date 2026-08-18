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
import java.util.concurrent.atomic.AtomicInteger;


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

    // Silence gate: an RMS below ~-46 dBFS means the press caught no speech
    // (pocket, table, button fumble). Whisper hallucinates fluent text on
    // such input, so skip the whole pipeline instead of feeding it. Quiet
    // but real speech sits well above this; the measured RMS is logged for
    // per-device tuning.
    private static final double SILENCE_RMS_THRESHOLD = 0.005;

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

    // State
    private String srcLang = "vi";
    private String tgtLang = "en";
    private boolean isRecording = false;

    // Single-thread pipeline executor: ASR/NLLB/VITS are CPU-bound and each
    // request holds decoder KV caches, so two concurrent pipelines spike RAM
    // (critical on a ≤4 GB wearable) and thrash caches for no throughput gain.
    // Rapid re-taps are latest-wins: a stale request exits at the next stage
    // boundary via its generation counter.
    private ExecutorService pipelineExecutor;
    private final AtomicInteger requestGeneration = new AtomicInteger();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_translation);

        // Get pipeline from Application
        OmniVoiceApp app = (OmniVoiceApp) getApplication();
        orchestrator = app.getOrchestrator();
        recorder = new AudioRecorder();
        player = new AudioPlayer();
        pipelineExecutor = Executors.newSingleThreadExecutor();

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

        File audioFile = new File(audioPath);
        Log.i(TAG, "Audio recorded: " + audioFile.length() + " bytes at " + audioPath);

        if (audioFile.length() <= 44) {
            transcriptView.setText("Error: recording too short or silent");
            return;
        }

        // Energy gate — see SILENCE_RMS_THRESHOLD.
        double rms = computePcmRms(audioPath);
        Log.i(TAG, "Recorded audio RMS: "
                + String.format(java.util.Locale.US, "%.4f", rms));
        if (rms >= 0 && rms < SILENCE_RMS_THRESHOLD) {
            transcriptView.setText(R.string.no_speech);
            translationView.setText("");
            timingView.setText("");
            return;
        }

        // Run pipeline on the single-thread executor (latest-wins cancel)
        final int generation = requestGeneration.incrementAndGet();
        pipelineExecutor.execute(() -> {
            try {
                PipelineOrchestrator.PipelineResult result =
                        orchestrator.process(audioPath, srcLang, tgtLang,
                                () -> generation != requestGeneration.get());

                if (result == null                 // superseded between stages
                        || generation != requestGeneration.get()) {
                    Log.i(TAG, "Pipeline request #" + generation + " superseded — dropped");
                    return;
                }

                runOnUiThread(() -> {
                    transcriptView.setText(result.transcript.isEmpty()
                            ? getText(R.string.no_speech)
                            : result.transcript);
                    translationView.setText(result.translation);
                    timingView.setText(String.format(
                            "ASR: %dms | Translation: %dms | TTS: %dms | Total: %dms",
                            result.asrMs, result.translationMs, result.ttsMs, result.totalMs
                    ));
                    // Surface the real TTS failure on screen instead of a silent 0ms.
                    if (result.ttsError != null) {
                        android.widget.Toast.makeText(
                                TranslationActivity.this,
                                "TTS failed: " + result.ttsError,
                                android.widget.Toast.LENGTH_LONG).show();
                    }
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
    // Silence gate
    // ----------------------------------------------------------------

    /**
     * RMS of the recorded 16-bit little-endian mono PCM, normalized to
     * 0..1 — or -1 when the file cannot be read, so the gate is skipped
     * rather than misfiring on real audio. The header is the 44 bytes
     * AudioRecorder writes before the data chunk.
     */
    private static double computePcmRms(String wavPath) {
        try (java.io.FileInputStream fis = new java.io.FileInputStream(wavPath)) {
            byte[] header = new byte[44];
            if (fis.read(header) < 44) return -1;
            byte[] buf = new byte[4096];
            long sumSq = 0;
            long count = 0;
            int read;
            while ((read = fis.read(buf)) > 0) {
                for (int i = 1; i < read; i += 2) {
                    short s = (short) (((buf[i] & 0xFF) << 8) | (buf[i - 1] & 0xFF));
                    sumSq += (long) s * s;
                    count++;
                }
            }
            return count == 0 ? 0.0 : Math.sqrt((double) sumSq / count) / 32768.0;
        } catch (java.io.IOException e) {
            Log.w(TAG, "RMS read failed: " + e.getMessage());
            return -1;
        }
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
        if (pipelineExecutor != null) {
            pipelineExecutor.shutdownNow();
            pipelineExecutor = null;
        }
    }
}
