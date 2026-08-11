/*
 * OmniVoice — MMS-TTS Tokenizer (pure Java, no native dependencies)
 *
 * Character-level tokenizer for on-device MMS-TTS. Languages that use a
 * phoneme-based vocab (e.g. mms-tts-vie) fall back to the built-in
 * normalization/punctuation handling — a full phonemizer is intentionally
 * avoided to keep no native dependencies.
 *
 * Follows the same pure-Java, no-JNI approach as SentencePieceProcessor.
 */

package com.omnivoice.onspeak47.pipeline;

import android.util.Log;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;


/**
 * Character-level tokenizer for MMS-TTS (VITS).
 *
 * <p>Loads a {@code mms_tts_<lang>_vocab.json} file (exported by
 * {@code optimize/export_mms_tts.py}) and maps normalized input text to
 * token IDs, inserting a blank (0) token between symbols as VITS expects.</p>
 *
 * <p>Design mirrors {@link SentencePieceProcessor}: pure Java, no JNI,
 * reads the asset file directly.</p>
 */
public class MmsTtsTokenizer {

    private static final String TAG = "MmsTtsTokenizer";

    /** Number of blank tokens inserted between symbols (VITS convention). */
    private static final int PAD_BETWEEN = 1;

    private Map<String, Integer> vocab;
    private int padId = 0;

    /**
     * Load a character-level vocab JSON file.
     *
     * @param vocabPath absolute path to {@code mms_tts_<lang>_vocab.json}
     * @throws IOException if the file cannot be read or parsed
     */
    public MmsTtsTokenizer(String vocabPath) throws IOException {
        File file = new File(vocabPath);
        String json = new String(readFileBytes(file), StandardCharsets.UTF_8);
        parseVocab(json);
    }

    private void parseVocab(String json) throws IOException {
        try {
            JSONObject root = new JSONObject(json);
            vocab = new HashMap<>();
            Iterator<String> keys = root.keys();
            while (keys.hasNext()) {
                String symbol = keys.next();
                vocab.put(symbol, root.getInt(symbol));
            }
            if (vocab.containsKey("<pad>")) {
                padId = vocab.get("<pad>");
            }
            Log.i(TAG, "Loaded vocab: " + vocab.size() + " symbols, padId=" + padId);
        } catch (Exception e) {
            throw new IOException("Failed to parse TTS vocab JSON: " + e.getMessage(), e);
        }
    }

    /**
     * Tokenize text into VITS input IDs.
     *
     * @param text     raw input text
     * @param language ISO code ("vi", "en", "zh") used for normalization
     * @return array of token IDs with blank padding between symbols
     */
    public int[] encode(String text, String language) {
        if (text == null || text.isEmpty()) return new int[0];

        String normalized = normalize(text, language);
        String[] symbols = splitSymbols(normalized, language);

        // First pass: map each symbol to its token ID (skip OOV)
        ArrayList<Integer> ids = new ArrayList<>(symbols.length);
        int unknownLogged = 0;
        for (String sym : symbols) {
            Integer id = vocab.get(sym);
            if (id != null) {
                ids.add(id);
            } else {
                // Try lowercase for Latin scripts (MMS vocabs are lowercase-centric)
                Integer lower = vocab.get(sym.toLowerCase(Locale.ROOT));
                if (lower != null) {
                    ids.add(lower);
                } else if (unknownLogged < 10) {
                    Log.w(TAG, "OOV symbol '" + sym + "' (U+" +
                            Integer.toHexString(sym.codePointAt(0)) + ") — skipped");
                    unknownLogged++;
                }
            }
        }


        // Second pass: interleave blank (pad) tokens between symbols
        ArrayList<Integer> out = new ArrayList<>(ids.size() * (PAD_BETWEEN + 1) + PAD_BETWEEN);
        for (int i = 0; i < ids.size(); i++) {
            out.add(ids.get(i));
            if (PAD_BETWEEN > 0) {
                out.add(padId);
            }
        }

        int[] arr = new int[out.size()];
        for (int i = 0; i < out.size(); i++) arr[i] = out.get(i);
        return arr;
    }

    // ------------------------------------------------------------------
    // Text normalization
    // ------------------------------------------------------------------

    private String normalize(String text, String language) {
        // NFC normalization keeps Vietnamese diacritics composed so they
        // match the single-character vocab entries.
        String n = Normalizer.normalize(text.trim(), Normalizer.Form.NFC);
        if (!"zh".equals(language)) {
            // MMS-TTS Latin vocabs (vie/eng) are lowercase
            n = n.toLowerCase(Locale.ROOT);
        }
        return n;
    }

    /**
     * Split normalized text into vocabulary symbols.
     *
     * Chinese text is segmented character-by-character; other languages are
     * split by character, keeping symbols that exist in the vocab and
     * skipping whitespace characters the vocab has no entry for.
     */
    private String[] splitSymbols(String text, String language) {
        ArrayList<String> out = new ArrayList<>();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            String ch = String.valueOf(c);
            if (vocab.containsKey(ch)) {
                out.add(ch);
            } else if (Character.isLetterOrDigit(c) || isCjk(c)) {
                // Keep letters/digits/CJK — encode() will lowercase or drop OOV
                out.add(ch);
            }
            // whitespace & unknown punctuation are dropped (not in vocab)
        }
        return out.toArray(new String[0]);
    }

    private static boolean isCjk(char c) {
        Character.UnicodeBlock b = Character.UnicodeBlock.of(c);
        return b == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS
                || b == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A
                || b == Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static byte[] readFileBytes(File file) throws IOException {
        byte[] data = new byte[(int) file.length()];
        try (FileInputStream fis = new FileInputStream(file)) {
            int offset = 0;
            while (offset < data.length) {
                int n = fis.read(data, offset, data.length - offset);
                if (n < 0) break;
                offset += n;
            }
        }
        return data;
    }

    /** Number of symbols in the loaded vocab. */
    public int size() {
        return vocab != null ? vocab.size() : 0;
    }
}
