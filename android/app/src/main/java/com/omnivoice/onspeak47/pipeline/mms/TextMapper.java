/*
 * OmniVoice — MMS-TTS Text Mapper
 *
 * Converts UTF-8 text into the token-id arrays MMS-TTS (VITS) expects.
 * Mirrors the character-ordinal tokenization path used by MMS-TTS:
 * each char is looked up by raw char value in the per-language charset,
 * and the model itself interleaves blank tokens (we do not replicate that).
 *
 * The charset is loaded from a plain-text asset produced by
 * optimize/05_export_mms_tts_onnx.py (one charset entry per line, index =
 * line number). This keeps the Android asset small (~50 lines) instead of
 * shipping the full HF tokenizer config.
 */

package com.omnivoice.onspeak47.pipeline.mms;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;


/**
 * MFCU (Mapped-From-Charset Unicode) tokenizer for MMS-TTS/VITS models.
 */
public class TextMapper {

    private static final String TAG = "TextMapper";

    /** Raw char-value -> vocab id for the currently loaded language. */
    private final Map<Integer, Integer> charToId = new HashMap<>();

    /** Vocabulary id of the blank/pad token (usually 0). Inserted between every
     *  character if {@code addBlank} is true, matching the HF VitsTokenizer. */
    private int blankId = 0;

    /** Whether the tokenizer interleaves blank tokens between every character. */
    private boolean addBlank = false;

    /**
     * Build a TextMapper for the given language.
     *
     * @param context    Android context (for asset access)
     * @param language   "vi" | "en" | "zh" (must match asset suffix)
     * @param charsetPath Asset path for the charset file, e.g. {@code "mms_tts_vie_charset.txt"}
     * @throws IOException if the asset cannot be read
     */
    public TextMapper(Context context, String language, String charsetPath) throws IOException {
        AssetManager assets = context.getAssets();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(assets.open(charsetPath), StandardCharsets.UTF_8))) {

            String line;
            int idx = 0;
            while ((line = reader.readLine()) != null) {
                if (!line.isEmpty()) {
                    // The charset file is one entry per line. An entry that is
                    // exactly one Unicode code point is a real symbol —
                    // anything longer is a special token (e.g. "<unk>") which
                    // VITS tokenization never emits through the char path.
                    if (line.codePointCount(0, line.length()) == 1) {
                        charToId.put(line.codePointAt(0), idx);
                    }
                }
                idx++;
            }
        }

        if (charToId.isEmpty()) {
            throw new IOException("No usable symbols found in charset asset: " + charsetPath);
        }

        // Load tokenizer_config.json for add_blank flag + blank/pad token id
        try {
            String configPath = charsetPath.replace("_charset.txt", "_tokenizer_config.json");
            loadTokenizerConfig(assets, configPath);
        } catch (IOException e) {
            Log.w(TAG, "No tokenizer_config.json for " + language +
                    " — defaulting to no blank tokens (add_blank=false)");
        }

        Log.i(TAG, "Loaded " + charToId.size() + " char symbols for language=" + language +
                " from " + charsetPath +
                (addBlank ? " (add_blank=" + blankId + ")" : ""));
    }

    /**
     * Parse a minimal tokenizer_config.json (extract add_blank + pad_token).
     * Runs against the actual tokenizer used by the ONNX-exported model.
     */
    private void loadTokenizerConfig(AssetManager assets, String configPath) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(assets.open(configPath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        String json = sb.toString();
        // Extract "add_blank" and "pad_token" / "unk_token" fields
        addBlank = json.contains("\"add_blank\": true");

        // Blank token id = vocab id of pad_token (or unk_token)
        String padToken = extractString(json, "pad_token", null);
        String unkToken = extractString(json, "unk_token", null);
        String blankToken = padToken != null ? padToken : unkToken;
        if (blankToken != null && blankToken.length() == 1) {
            Integer id = charToId.get((int) blankToken.charAt(0));
            if (id != null) {
                blankId = id;
            }
        }
    }

    private static String extractString(String json, String key, String defaultVal) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return defaultVal;
        int start = json.indexOf('"', idx + pattern.length() + 1);
        if (start < 0) return defaultVal;
        int end = json.indexOf('"', start + 1);
        if (end < 0) return defaultVal;
        return json.substring(start + 1, end);
    }

    /**
     * Convert text to an MMS-TTS input sequence.
     *
     * The model's forward pass interleaves blank ids itself, so caller does
     * NOT need to add blank padding around the sequence. Unmappable
     * characters are skipped with a warning (matching the HF reference
     * behavior) rather than crashing the pipeline.
     *
     * @param text Input text (normalized before calling if desired)
     * @return int array of vocabulary ids ready for the ONNX graph
     */
    public int[] toIds(String text) {
        if (text == null || text.isEmpty()) {
            return new int[0];
        }

        // VITS/MMS-TTS does not perform casefolding in the char path,
        // but the MMS-TTS checkpoints tolerate lowercase better than
        // mixed-case input from the translator.
        String normalized = text.toLowerCase(Locale.ROOT);

        int charCount = normalized.codePointCount(0, normalized.length());
        // With add_blank every character gets a blank id before and after,
        // so output length is 2*N+1 (or just N when no blanks).
        int maxIds = addBlank ? 2 * charCount + 1 : charCount;
        int[] ids = new int[maxIds];
        int out = 0;
        int skipped = 0;

        int len = normalized.length();
        for (int i = 0; i < len; ) {
            int cp = normalized.codePointAt(i);
            i += Character.charCount(cp);
            Integer id = charToId.get(cp);
            if (id != null) {
                if (addBlank) {
                    ids[out++] = blankId;   // blank before each char
                }
                ids[out++] = id;             // character itself
            } else {
                skipped++;
            }
        }
        if (addBlank) {
            ids[out++] = blankId;            // trailing blank
        }

        if (skipped > 0) {
            Log.w(TAG, "Skipped " + skipped + " unmappable characters in TTS input");
        }

        if (out == 0) {
            Log.w(TAG, "All characters unmappable — returning empty token array");
        }

        if (out == ids.length) {
            return ids;
        }
        int[] trimmed = new int[out];
        System.arraycopy(ids, 0, trimmed, 0, out);
        return trimmed;
    }

    /**
     * Number of distinct char-level symbols loaded from the charset.
     */
    public int getSymbolCount() {
        return charToId.size();
    }
}
