/*
 * OmniVoice — Pipeline Orchestrator
 *
 * Strings the three models together: ASR → Translation → TTS.
 * NO language-branching or routing logic — every input goes through
 * the same three stages regardless of direction (per pipeline_overview.md).
 */

package com.omnivoice.onspeak47.pipeline;

import android.util.Log;

import java.io.File;


/**
 * Chains ASR → Translation → TTS with no language branching.
 * Modular design: each stage has clear I/O contracts so a future
 * wearable device can swap hardware backends without redesign.
 */
public class PipelineOrchestrator {

    private static final String TAG = "PipelineOrchestrator";

    private final ASRModule asr;
    private final TranslationModule translator;
    private final TTSModule tts;

    public PipelineOrchestrator(ASRModule asr, TranslationModule translator, TTSModule tts) {
        this.asr = asr;
        this.translator = translator;
        this.tts = tts;
    }

    /**
     * Run the full pipeline: audio → text → translated text → audio.
     *
     * @param audioPath Path to input audio (16kHz WAV)
     * @param srcLang   Source language ("vi", "en", "zh")
     * @param tgtLang   Target language
     * @return Pipeline result with transcript, translation, and output audio path
     */
    public PipelineResult process(String audioPath, String srcLang, String tgtLang) {
        long totalStart = System.currentTimeMillis();

        // --- Stage 1: ASR (Speech → Text) ---
        long t0 = System.currentTimeMillis();
        ASRModule.ASRResult asrResult = asr.transcribe(audioPath, srcLang);
        long asrMs = System.currentTimeMillis() - t0;
        Log.i(TAG, "ASR: \"" + asrResult.text + "\" (" + asrMs + "ms)");

        // --- Stage 2: Translation (Text → Translated Text) ---
        t0 = System.currentTimeMillis();
        TranslationModule.TranslationResult translationResult;
        if (translator != null) {
            translationResult = translator.translate(asrResult.text, srcLang, tgtLang);
        } else {
            translationResult = new TranslationModule.TranslationResult(
                    "[translation unavailable — translator not initialized]", 0);
        }
        long translationMs = System.currentTimeMillis() - t0;
        Log.i(TAG, "Translation: \"" + translationResult.text + "\" (" + translationMs + "ms)");

        // --- Stage 3: TTS (Translated Text → Speech) ---
        t0 = System.currentTimeMillis();
        File outputDir = new File(audioPath).getParentFile();
        String outputPath = new File(outputDir, "translated_" + srcLang + "_to_" + tgtLang + ".wav")
                .getAbsolutePath();
        String ttsPath = tts.synthesize(translationResult.text, tgtLang, outputPath);
        long ttsMs = System.currentTimeMillis() - t0;
        String ttsError = (ttsPath == null) ? tts.getLastError() : null;
        Log.i(TAG, "TTS: " + (ttsPath != null ? "success" : "failed (" + ttsError + ")")
                + " (" + ttsMs + "ms)");

        long totalMs = System.currentTimeMillis() - totalStart;
        Log.i(TAG, "Pipeline total: " + totalMs + "ms");

        return new PipelineResult(
                asrResult.text,
                translationResult.text,
                ttsPath,
                asrMs, translationMs, ttsMs, totalMs,
                ttsError
        );
    }

    /**
     * Translation-only shortcut — skips ASR and TTS.
     */
    public PipelineResult translateText(String text, String srcLang, String tgtLang) {
        long t0 = System.currentTimeMillis();
        TranslationModule.TranslationResult result;
        if (translator != null) {
            result = translator.translate(text, srcLang, tgtLang);
        } else {
            result = new TranslationModule.TranslationResult(
                    "[translation unavailable — translator not initialized]", 0);
        }
        long elapsed = System.currentTimeMillis() - t0;

        return new PipelineResult(text, result.text, null, 0, elapsed, 0, elapsed, null);
    }

    // ----------------------------------------------------------------
    // Result
    // ----------------------------------------------------------------

    public static class PipelineResult {
        public final String transcript;
        public final String translation;
        public final String audioPath;
        public final long asrMs;
        public final long translationMs;
        public final long ttsMs;
        public final long totalMs;
        /** Reason TTS failed, or null if TTS succeeded. */
        public final String ttsError;

        public PipelineResult(String transcript, String translation, String audioPath,
                              long asrMs, long translationMs, long ttsMs, long totalMs) {
            this(transcript, translation, audioPath, asrMs, translationMs, ttsMs, totalMs, null);
        }

        public PipelineResult(String transcript, String translation, String audioPath,
                              long asrMs, long translationMs, long ttsMs, long totalMs,
                              String ttsError) {
            this.transcript = transcript;
            this.translation = translation;
            this.audioPath = audioPath;
            this.asrMs = asrMs;
            this.translationMs = translationMs;
            this.ttsMs = ttsMs;
            this.totalMs = totalMs;
            this.ttsError = ttsError;
        }
    }
}
