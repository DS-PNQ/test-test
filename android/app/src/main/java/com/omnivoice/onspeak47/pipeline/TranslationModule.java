/*
 * OmniVoice — Translation Module (HY-MT1.5-1.8B via GGUF / llama.cpp engine)
 *
 * HY-MT1.5-1.8B is a causal language model distributed in GGUF format
 * (e.g. Hy-MT1.5-1.8B-1.25bit.gguf) rather than ONNX encoder-decoder graphs.
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.icu.text.BreakIterator;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;


public class TranslationModule {

    private static final String TAG = "TranslationModule";

    public static final String DEFAULT_GGUF_MODEL = "Hy-MT1.5-1.8B-1.25bit.gguf";
    public static final String ALT_GGUF_MODEL = "hymt.gguf";

    private static final int MAX_OUTPUT_TOKENS = 256;

    /** Whether this module actually has a working backend. */
    private final boolean available;
    private final String unavailableReason;
    private final String modelPath;

    private static final Map<String, String> LANG_NAMES = new HashMap<>();
    static {
        LANG_NAMES.put("vi", "Vietnamese");
        LANG_NAMES.put("en", "English");
        LANG_NAMES.put("zh", "Chinese");
    }

    /**
     * Check whether on-device HY-MT GGUF translation model is present.
     */
    public static boolean isAvailable(Context context) {
        return findModelPath(context) != null;
    }

    /**
     * Find the GGUF model file if present in external/internal storage or assets.
     */
    public static String findModelPath(Context context) {
        if (context == null) return null;

        // Check external files dir
        File extDir = context.getExternalFilesDir(null);
        if (extDir != null) {
            File f1 = new File(extDir, DEFAULT_GGUF_MODEL);
            if (f1.exists() && f1.length() > 0) return f1.getAbsolutePath();
            File f2 = new File(extDir, ALT_GGUF_MODEL);
            if (f2.exists() && f2.length() > 0) return f2.getAbsolutePath();
        }

        // Check internal files dir
        File intDir = context.getFilesDir();
        if (intDir != null) {
            File f1 = new File(intDir, DEFAULT_GGUF_MODEL);
            if (f1.exists() && f1.length() > 0) return f1.getAbsolutePath();
            File f2 = new File(intDir, ALT_GGUF_MODEL);
            if (f2.exists() && f2.length() > 0) return f2.getAbsolutePath();
        }

        // Check assets
        try {
            String[] list = context.getAssets().list("");
            if (list != null) {
                for (String name : list) {
                    if (name.endsWith(".gguf") || name.equals(DEFAULT_GGUF_MODEL) || name.equals(ALT_GGUF_MODEL)) {
                        return name;
                    }
                }
            }
        } catch (IOException ignored) {
        }

        return null;
    }

    public TranslationModule(Context context) {
        String path = findModelPath(context);
        if (path != null) {
            this.modelPath = path;
            this.available = true;
            this.unavailableReason = null;
            Log.i(TAG, "Translation module initialized with HY-MT GGUF model: " + path);
        } else {
            this.modelPath = null;
            this.available = false;
            this.unavailableReason = "HY-MT GGUF model not found (" + DEFAULT_GGUF_MODEL + ").";
            Log.w(TAG, "Translation module unavailable: " + unavailableReason);
        }
    }

    /**
     * Returns true if this module can actually translate.
     */
    public boolean isReady() {
        return available;
    }

    public TranslationResult translate(String text, String srcLang, String tgtLang) {
        long startTime = System.currentTimeMillis();
        if (text == null || text.trim().isEmpty()) {
            return new TranslationResult("", 0);
        }

        if (!available) {
            long elapsed = System.currentTimeMillis() - startTime;
            return new TranslationResult(
                    "[translation unavailable — " + unavailableReason + "]",
                    elapsed);
        }

        // Phase 2: HY-MT inference via onnxruntime-genai
        // For each sentence, build the instruction prompt:
        //   "Translate the following segment into <tgtLang>,
        //    without additional explanation.\n\n<text>"
        // and run autoregressive generation.

        ArrayList<String> sentences = splitIntoSentences(text, srcLang);
        StringBuilder result = new StringBuilder();

        for (String sentence : sentences) {
            if (sentence.trim().isEmpty()) continue;
            String translated = translateSentence(sentence, srcLang, tgtLang);
            if (result.length() > 0) result.append(" ");
            result.append(translated);
        }

        long elapsed = System.currentTimeMillis() - startTime;
        return new TranslationResult(result.toString(), elapsed);
    }

    private String translateSentence(String text, String srcLang, String tgtLang) {
        // Phase 2: run HY-MT inference here
        return "[translation unavailable]";
    }

    private ArrayList<String> splitIntoSentences(String text, String langCode) {
        ArrayList<String> sentences = new ArrayList<>();
        Locale locale = langCode.equals("vi") ? new Locale("vi") : Locale.US;
        BreakIterator iterator = BreakIterator.getSentenceInstance(locale);
        iterator.setText(text);
        int start = iterator.first();
        for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
            sentences.add(text.substring(start, end).trim());
        }
        return sentences;
    }

    public static class TranslationResult {
        public final String text;
        public final long processingMs;
        public TranslationResult(String text, long processingMs) {
            this.text = text;
            this.processingMs = processingMs;
        }
    }
}
