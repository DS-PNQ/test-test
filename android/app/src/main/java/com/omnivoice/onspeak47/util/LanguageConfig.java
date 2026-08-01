/*
 * OmniVoice — Language Configuration
 *
 * Scoped to VN/EN/CN only (per pipeline_overview.md).
 */

package com.omnivoice.onspeak47.util;


/**
 * Language configuration for the three in-scope languages.
 *
 * Provides display names, ISO codes, NLLB FLORES-200 codes,
 * and Whisper language tokens — all in one place.
 */
public final class LanguageConfig {

    // Ordered list of supported languages
    private static final Language[] LANGUAGES = {
            new Language("vi", "Tiếng Việt", "vie_Latn", "<|vi|>"),
            new Language("en", "English", "eng_Latn", "<|en|>"),
            new Language("zh", "中文 (简体)", "zho_Hans", "<|zh|>"),
    };

    private LanguageConfig() {}  // static utility class

    /**
     * Get display names for UI spinners.
     */
    public static String[] getDisplayNames() {
        String[] names = new String[LANGUAGES.length];
        for (int i = 0; i < LANGUAGES.length; i++) {
            names[i] = LANGUAGES[i].displayName;
        }
        return names;
    }

    /**
     * Get the ISO code at a given index.
     */
    public static String getCodeAtIndex(int index) {
        if (index < 0 || index >= LANGUAGES.length) return "vi";
        return LANGUAGES[index].code;
    }

    /**
     * Get the index for a given ISO code.
     */
    public static int indexOf(String code) {
        for (int i = 0; i < LANGUAGES.length; i++) {
            if (LANGUAGES[i].code.equals(code)) return i;
        }
        return 0;
    }

    /**
     * Get the NLLB FLORES-200 code for an ISO code.
     */
    public static String getNllbCode(String code) {
        for (Language lang : LANGUAGES) {
            if (lang.code.equals(code)) return lang.nllbCode;
        }
        return "vie_Latn";
    }

    /**
     * Get the Whisper language token for an ISO code.
     */
    public static String getWhisperToken(String code) {
        for (Language lang : LANGUAGES) {
            if (lang.code.equals(code)) return lang.whisperToken;
        }
        return "<|vi|>";
    }

    /**
     * Get the number of supported languages.
     */
    public static int count() {
        return LANGUAGES.length;
    }

    // ----------------------------------------------------------------
    // Language data class
    // ----------------------------------------------------------------

    private static class Language {
        final String code;         // ISO 639-1 ("vi", "en", "zh")
        final String displayName;  // Human-readable name
        final String nllbCode;     // NLLB FLORES-200 code
        final String whisperToken; // Whisper language token

        Language(String code, String displayName, String nllbCode, String whisperToken) {
            this.code = code;
            this.displayName = displayName;
            this.nllbCode = nllbCode;
            this.whisperToken = whisperToken;
        }
    }
}
