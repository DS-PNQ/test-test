/*
 * OmniVoice — NLLB Tokenizer (pure Java, no native dependencies)
 *
 * Uses SentencePieceProcessor to read the .model protobuf directly.
 * Supports both pruned and unpruned NLLB models via an optional
 * language_token_map.json configuration file.
 */

package com.omnivoice.onspeak47.pipeline;

import android.util.Log;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;


/**
 * Tokenizer for NLLB-200 using a pure-Java SentencePiece implementation.
 *
 * <h3>ID spaces</h3>
 * <p>SentencePiece IDs (SP) and HuggingFace IDs (HF) differ by an
 * offset-and-swap for the first three special tokens:</p>
 * <pre>
 *   SP 0 (unk)  → HF 3
 *   SP 1 (bos)  → HF 0
 *   SP 2 (eos)  → HF 2
 *   SP N (N≥3)  → HF N+1
 * </pre>
 * <p>The ONNX model uses HF IDs. This class handles the conversion.</p>
 *
 * <h3>Language token IDs</h3>
 * <p>For the <b>unpruned</b> model, language token IDs are computed as
 * {@code spVocabSize + 1 + languageIndex} (where {@code languageIndex}
 * is the position in the full 200-language NLLB list).</p>
 * <p>For the <b>pruned</b> model, the token IDs change because removed
 * language tokens shift the remaining ones. The correct IDs are read
 * from {@code language_token_map.json} (exported by
 * {@code 02_prune_vocab.py}).</p>
 */
public class Tokenizer {

    private static final String TAG = "Tokenizer";
    public static final int NLLB = 0;

    private static final String LANG_TOKEN_MAP_FILE = "language_token_map.json";

    // HuggingFace special token IDs (constant across pruned/unpruned)
    private static final int HF_EOS_ID = 2;   // </s>

    private SentencePieceProcessor sp;

    /**
     * Non-null if the SentencePiece model failed to load.
     * Contains a human-readable reason.
     */
    private String initError;

    /** SP vocabulary size, used for unpruned language-token offset. */
    private int spVocabSize;

    /**
     * Language code → HF token ID.
     * Populated from {@code language_token_map.json} (pruned model)
     * or computed from the LANGUAGES_NLLB array (unpruned model).
     */
    private final Map<String, Integer> langTokenIds = new HashMap<>();

    // NLLB language codes in their canonical embedding-ID order.
    // Used as fallback when language_token_map.json is not present.
    private static final String[] LANGUAGES_NLLB = {
            "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn",
            "ajp_Arab", "aka_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab",
            "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn",
            "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl",
            "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt",
            "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn",
            "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn",
            "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn",
            "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn",
            "fin_Latn", "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn",
            "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn",
            "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
            "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn",
            "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab",
            "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn", "kaz_Cyrl", "kbp_Latn",
            "kea_Latn", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn",
            "kon_Latn", "kor_Hang", "kmr_Latn", "lao_Laoo", "lvs_Latn", "lij_Latn",
            "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn",
            "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "mag_Deva", "mai_Deva",
            "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn", "mlt_Latn",
            "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr",
            "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn",
            "nya_Latn", "oci_Latn", "gaz_Latn", "ory_Orya", "pag_Latn", "pan_Guru",
            "pap_Latn", "pol_Latn", "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn",
            "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng",
            "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn",
            "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "als_Latn",
            "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn",
            "szl_Latn", "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn",
            "tha_Thai", "tir_Ethi", "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn",
            "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng",
            "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn",
            "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn",
            "yue_Hant", "zho_Hans", "zho_Hant", "zul_Latn"
    };

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /**
     * @param vocabFile path to {@code sentencepiece_bpe.model}
     * @param configDir directory that may contain
     *                  {@code language_token_map.json}
     * @param mode      currently only {@link #NLLB} is supported
     */
    public Tokenizer(String vocabFile, String configDir, int mode) {
        File file = new File(vocabFile);
        if (!file.exists()) {
            initError = "SentencePiece model file not found: " + vocabFile;
            Log.e(TAG, initError);
            return;
        }
        if (file.length() == 0) {
            initError = "SentencePiece model file is empty (0 bytes): " + vocabFile;
            Log.e(TAG, initError);
            return;
        }

        try {
            sp = new SentencePieceProcessor();
            sp.load(vocabFile);
            spVocabSize = sp.getVocabSize();
            Log.i(TAG, "SentencePiece loaded: " + spVocabSize + " pieces");
        } catch (Exception e) {
            initError = "Failed to load SentencePiece model ("
                    + e.getClass().getSimpleName() + "): " + e.getMessage();
            Log.e(TAG, initError, e);
            return;
        }

        // Try to load pruned-model language token map
        loadLanguageTokenMap(configDir);
    }

    /**
     * Read {@code language_token_map.json} if it exists.
     * Falls back to the standard formula for unpruned models.
     */
    private void loadLanguageTokenMap(String configDir) {
        if (configDir == null) {
            buildDefaultLanguageTokenIds();
            return;
        }

        File mapFile = new File(configDir, LANG_TOKEN_MAP_FILE);
        if (!mapFile.exists()) {
            Log.i(TAG, "No " + LANG_TOKEN_MAP_FILE + " found in "
                    + configDir + " — using default (unpruned) language token IDs");
            buildDefaultLanguageTokenIds();
            return;
        }

        try {
            byte[] data = new byte[(int) mapFile.length()];
            try (FileInputStream fis = new FileInputStream(mapFile)) {
                int off = 0;
                while (off < data.length) {
                    int n = fis.read(data, off, data.length - off);
                    if (n < 0) break;
                    off += n;
                }
            }
            JSONObject json = new JSONObject(new String(data, StandardCharsets.UTF_8));
            Iterator<String> keys = json.keys();
            while (keys.hasNext()) {
                String langCode = keys.next();
                int tokenId = json.getInt(langCode);
                langTokenIds.put(langCode, tokenId);
            }
            Log.i(TAG, "Loaded pruned language token map (" + langTokenIds.size()
                    + " languages): " + langTokenIds);
        } catch (Exception e) {
            Log.w(TAG, "Failed to read " + LANG_TOKEN_MAP_FILE
                    + ", falling back to defaults", e);
            buildDefaultLanguageTokenIds();
        }
    }

    /**
     * Populate langTokenIds with the exact HuggingFace language-token ids for
     * the app's supported languages. Values verified against the real
     * NLLB-200 distilled-600M vocab (HF tokenizer.json), where language ids
     * begin right after the 256000 sentence-piece entries + 4 special tokens.
     *
     * Replaces the old `spVocabSize + 1 + index` formula, which was wrong
     * twice: an off-by-one (+1) and a hardcoded LANGUAGES_NLLB list missing
     * ~50 NLLB codes, so vie_Latn/zho_Hans computed to out-of-range ids
     * (256206 > max 256205) that crashed the ONNX decoder embedding Gather
     * and surfaced in the app as "[error]".
     */
    private void buildDefaultLanguageTokenIds() {
        langTokenIds.clear();
        langTokenIds.put("eng_Latn", 256047);
        langTokenIds.put("vie_Latn", 256193);
        langTokenIds.put("zho_Hans", 256200);
        langTokenIds.put("zho_Hant", 256201);
        Log.i(TAG, "Default language token IDs: " + langTokenIds);
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /** Returns the initialization error message, or null if loaded OK. */
    public String getInitError() {
        return initError;
    }

    /** Returns true if the SentencePiece model loaded successfully. */
    public boolean isReady() {
        return sp != null;
    }

    /** Returns the SP vocabulary size (for debugging). */
    public int getDictionaryLength() {
        return spVocabSize;
    }

    /**
     * Tokenize input text for the NLLB encoder.
     *
     * <p>Produces HF IDs: [subword tokens...] [EOS] [src_lang_token]</p>
     *
     * @param srcLanguage NLLB FLORES code (e.g. {@code "vie_Latn"})
     * @param tgtLanguage NLLB FLORES code (unused for encoding, kept for API compat)
     * @param text        input text
     * @return tokenizer result with input IDs + attention mask, or null
     */
    public TokenizerResult tokenize(String srcLanguage, String tgtLanguage, String text) {
        if (sp == null) return null;

        // Encode text → SP IDs
        int[] spIds = sp.encode(text);

        // Convert SP IDs → HF IDs
        int[] hfIds = new int[spIds.length];
        for (int i = 0; i < spIds.length; i++) {
            hfIds[i] = spToHf(spIds[i]);
        }

        // Build final sequence: [tokens...] [EOS] [src_lang_token]
        int srcLangId = getLanguageID(srcLanguage);

        int[] idsExtended = new int[hfIds.length + 2];
        System.arraycopy(hfIds, 0, idsExtended, 0, hfIds.length);
        idsExtended[hfIds.length] = HF_EOS_ID;
        idsExtended[hfIds.length + 1] = srcLangId;

        int[] attentionMask = new int[idsExtended.length];
        Arrays.fill(attentionMask, 1);

        return new TokenizerResult(idsExtended, attentionMask);
    }

    /**
     * Decode HF token IDs back to text.
     *
     * @param ids HF token IDs from the ONNX decoder output
     * @return decoded text
     */
    public String decode(int[] ids) {
        if (sp == null) return "";

        // Convert HF IDs → SP IDs
        int[] spIds = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            spIds[i] = hfToSp(ids[i]);
        }

        return sp.decode(spIds);
    }

    /**
     * Get the HF token ID for an NLLB language code.
     *
     * @param languageCode NLLB FLORES code (e.g. {@code "vie_Latn"})
     * @return HF token ID
     */
    public int getLanguageID(String languageCode) {
        Integer id = langTokenIds.get(languageCode);
        if (id != null) return id;

        // Unknown language: default to English. Never compute a formula-based
        // id -- that produced out-of-range ids that crashed the decoder.
        Log.w(TAG, "Unknown NLLB language code: " + languageCode
                + " — defaulting to eng_Latn (256047)");
        return 256047;
    }

    /**
     * Lookup a single piece's HF token ID.
     * Kept for backward compatibility.
     */
    public int pieceToId(String piece) {
        // Special-case the tokens used in the pipeline
        if ("</s>".equals(piece)) return HF_EOS_ID;
        if ("<s>".equals(piece)) return 0;  // HF BOS/PAD
        if ("<unk>".equals(piece)) return 3; // HF UNK

        // Otherwise encode the piece text and take the first token
        if (sp == null) return 0;
        int[] spIds = sp.encode(piece);
        if (spIds.length == 0) return 0;
        return spToHf(spIds[0]);
    }

    // ------------------------------------------------------------------
    // SP ↔ HF ID conversion
    // ------------------------------------------------------------------

    /** Convert a SentencePiece ID to a HuggingFace ID. */
    private static int spToHf(int spId) {
        int id = spId + 1;
        switch (id) {
            case 1: return 3;   // SP 0 (unk) → HF 3
            case 2: return 0;   // SP 1 (bos) → HF 0
            case 3: return 2;   // SP 2 (eos) → HF 2
            default: return id; // SP N (≥3) → HF N+1
        }
    }

    /** Convert a HuggingFace ID to a SentencePiece ID. */
    private static int hfToSp(int hfId) {
        switch (hfId) {
            case 0: return 1;  // HF 0 (bos/pad) → SP 1
            case 2: return 2;  // HF 2 (eos) → SP 2
            case 3: return 0;  // HF 3 (unk) → SP 0
            default:
                // HF 1 (pad) → SP 0, HF N (≥4) → SP N-1
                return Math.max(0, hfId - 1);
        }
    }

    // ------------------------------------------------------------------
    // Result
    // ------------------------------------------------------------------

    public static class TokenizerResult {
        public final int[] inputIDs;
        public final int[] attentionMask;

        public TokenizerResult(int[] inputIDs, int[] attentionMask) {
            this.inputIDs = inputIDs;
            this.attentionMask = attentionMask;
        }
    }
}
