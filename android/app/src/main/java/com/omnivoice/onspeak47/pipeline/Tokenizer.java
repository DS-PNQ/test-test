/*
 * OmniVoice — Tokenizer (NLLB-200 via Native SentencePiece)
 *
 * Uses SentencePiece C++ native library via JNI (same implementation as RTranslator).
 */

package com.omnivoice.onspeak47.pipeline;

import android.util.Log;
import java.util.Arrays;

/**
 * Tokenizer for NLLB-200 using native SentencePiece library via JNI.
 */
public class Tokenizer {

    private static final String TAG = "Tokenizer";
    public static final int NLLB = 0;

    // NLLB language codes sorted by their embedding ID order
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

    private static final int DICTIONARY_LENGTH = 256000;

    private SentencePieceProcessor spProcessor;

    public Tokenizer(String vocabFile, int mode) {
        try {
            spProcessor = new SentencePieceProcessor();
            spProcessor.Load(vocabFile);
            Log.i(TAG, "Loaded SentencePiece model from " + vocabFile);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load SentencePiece model: " + vocabFile, e);
        }
    }

    public TokenizerResult tokenize(String srcLanguage, String tgtLanguage, String text) {
        if (spProcessor == null) return null;

        int[] ids = spProcessor.encode(text);

        for (int i = 0; i < ids.length; i++) {
            ids[i] = ids[i] + 1;
            switch (ids[i]) {
                case 1: {
                    ids[i] = 3;
                    break;
                }
                case 2: {
                    ids[i] = 0;
                    break;
                }
                case 3: {
                    ids[i] = 2;
                    break;
                }
            }
        }

        int eos = pieceToId("</s>");
        int srcLanguageID = getLanguageID(srcLanguage);

        int[] idsExtended = new int[ids.length + 2];
        System.arraycopy(ids, 0, idsExtended, 1, ids.length);
        idsExtended[0] = srcLanguageID;
        idsExtended[idsExtended.length - 1] = eos;

        int[] attentionMask = new int[idsExtended.length];
        Arrays.fill(attentionMask, 1);

        return new TokenizerResult(idsExtended, attentionMask);
    }

    public String decode(int[] ids) {
        if (spProcessor == null) return "";

        String output = "";
        for (int id : ids) {
            if (id < DICTIONARY_LENGTH && id > 3) {
                output = output.concat(spProcessor.IDToPiece(id - 1));
            }
        }

        if (!output.isEmpty() && output.charAt(0) == '\u2581') {
            output = output.substring(1);
        }
        return output.replace('\u2581', ' ');
    }

    public int pieceToId(String token) {
        if (spProcessor == null) return 0;
        return spProcessor.PieceToID(token);
    }

    public int getLanguageID(String language) {
        for (int i = 0; i < LANGUAGES_NLLB.length; i++) {
            if (LANGUAGES_NLLB[i].equals(language)) {
                return DICTIONARY_LENGTH + i + 1;
            }
        }
        return DICTIONARY_LENGTH + 1;
    }

    public static class TokenizerResult {
        public final int[] inputIDs;
        public final int[] attentionMask;

        public TokenizerResult(int[] inputIDs, int[] attentionMask) {
            this.inputIDs = inputIDs;
            this.attentionMask = attentionMask;
        }
    }
}
