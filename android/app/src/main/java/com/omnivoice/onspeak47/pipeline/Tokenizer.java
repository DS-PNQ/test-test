/*
 * OmniVoice — Tokenizer (pure-Java SentencePiece BPE, NLLB-200 vocab)
 *
 * This intentionally does NOT use ai.djl.sentencepiece. That module only ships
 * native (.so) binaries for desktop/server JVM targets — linux-x86_64, osx,
 * win (see extensions/sentencepiece in https://github.com/deepjavalibrary/djl).
 * There is no ai.djl.android:sentencepiece-native artifact, unlike the
 * separate ai.djl.huggingface:tokenizers module, which does ship one
 * (ai.djl.android:tokenizer-native). Concretely, on an Android device
 * SpTokenizer's constructor fails while resolving/loading the native
 * library — either it can't find a classpath resource for the device's
 * ABI, or (if it did) System.load() on a .so just extracted to an
 * app-writable directory is blocked by Android 10+'s W^X policy. Either
 * way spTokenizer never initializes, tokenize() returns null on every
 * call, and TranslationModule surfaces "[error: tokenization failed]"
 * immediately (0 ms) — exactly what was observed on-device.
 *
 * This class instead parses the SentencePiece `ModelProto` protobuf
 * directly (a small hand-rolled reader — we only need the repeated
 * `pieces` field) and re-implements the standard SentencePiece BPE merge
 * algorithm in Java. No native library, no classpath extraction, same
 * behavior on every ABI.
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.icu.text.Normalizer2;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Tokenizer for NLLB-200 using a pure-Java SentencePiece BPE implementation.
 */
public class Tokenizer {

    private static final String TAG = "Tokenizer";
    public static final int NLLB = 0;

    private static final String SPIECE_UNDERLINE = "\u2581"; // ▁

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

    private String[] idToPiece;
    private Map<String, Integer> pieceToIdMap;
    private float[] idToScore;
    private String[] byteToken;          // index = byte value 0..255, or null
    private boolean byteFallbackAvailable;
    private String unkPiece = "<unk>";
    private boolean loaded = false;

    /**
     * @param context   Android context — used to read the model from assets as
     *                  a fallback when the file-system copy doesn't exist yet.
     * @param vocabFile Absolute path to the SentencePiece .model file on the
     *                  file system (e.g. in externalFilesDir).
     * @param assetName Original asset name (e.g. "sentencepiece_bpe.model")
     *                  so we can fall back to AssetManager if vocabFile is missing.
     * @param mode      Tokenizer mode (currently only {@link #NLLB}).
     */
    public Tokenizer(Context context, String vocabFile, String assetName, int mode) {
        try {
            loadModel(context, vocabFile, assetName);
            loaded = true;
            Log.i(TAG, "Loaded " + idToPiece.length + " SentencePiece pieces"
                    + " (byte fallback " + (byteFallbackAvailable ? "available" : "unavailable") + ")");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load SentencePiece model: " + vocabFile
                    + " (asset=" + assetName + ")", e);
        }
    }

    /** Returns {@code true} if the model was loaded successfully. */
    public boolean isLoaded() {
        return loaded;
    }

    // ---------------------------------------------------------------------
    // Model loading: minimal protobuf reader for sentencepiece's ModelProto
    // ---------------------------------------------------------------------

    private void loadModel(Context context, String vocabFile, String assetName) throws IOException {
        byte[] data = readModelBytes(context, vocabFile, assetName);

        List<String> pieces = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        ProtoReader top = new ProtoReader(data);
        while (top.hasNext()) {
            int tag = top.readVarint32();
            int field = tag >>> 3;
            int wireType = tag & 7;
            if (field == 1 && wireType == 2) { // ModelProto.pieces (SentencePiece message)
                ProtoReader sp = new ProtoReader(top.readBytes());
                String piece = "";
                float score = 0f;
                while (sp.hasNext()) {
                    int t2 = sp.readVarint32();
                    int f2 = t2 >>> 3;
                    int w2 = t2 & 7;
                    if (f2 == 1 && w2 == 2) {
                        piece = new String(sp.readBytes(), StandardCharsets.UTF_8);
                    } else if (f2 == 2 && w2 == 5) {
                        score = Float.intBitsToFloat(sp.readFixed32());
                    } else {
                        sp.skip(w2);
                    }
                }
                pieces.add(piece);
                scores.add(score);
            } else {
                top.skip(wireType);
            }
        }

        if (pieces.isEmpty()) {
            throw new IOException("No SentencePiece pieces found in " + vocabFile);
        }

        idToPiece = pieces.toArray(new String[0]);
        idToScore = new float[idToPiece.length];
        pieceToIdMap = new HashMap<>(idToPiece.length * 2);
        for (int i = 0; i < idToPiece.length; i++) {
            idToScore[i] = scores.get(i);
            pieceToIdMap.putIfAbsent(idToPiece[i], i); // first occurrence wins on duplicate
        }

        if (pieceToIdMap.containsKey("<unk>")) {
            unkPiece = "<unk>";
        } else if (idToPiece.length > 0) {
            unkPiece = idToPiece[0];
        }

        byteToken = new String[256];
        boolean allBytesPresent = true;
        for (int b = 0; b < 256; b++) {
            String t = String.format("<0x%02X>", b);
            byteToken[b] = t;
            if (!pieceToIdMap.containsKey(t)) {
                allBytesPresent = false;
            }
        }
        byteFallbackAvailable = allBytesPresent;
    }

    /**
     * Read the raw bytes of the SentencePiece model, trying the filesystem
     * first and falling back to the Android AssetManager.
     */
    private byte[] readModelBytes(Context context, String vocabFile, String assetName) throws IOException {
        // 1. Try the file-system path (the copy that TranslationModule creates)
        File file = new File(vocabFile);
        if (file.exists() && file.length() > 0) {
            Log.d(TAG, "Reading SP model from filesystem: " + vocabFile
                    + " (" + (file.length() / 1024) + " KB)");
            return readAllBytes(new FileInputStream(file));
        }
        Log.w(TAG, "SP model not found on filesystem (" + vocabFile
                + "), falling back to AssetManager (" + assetName + ")");

        // 2. Fall back to reading directly from the APK's assets
        if (context != null && assetName != null) {
            try (InputStream in = context.getAssets().open(assetName)) {
                Log.d(TAG, "Reading SP model from assets: " + assetName);
                return readAllBytes(in);
            }
        }

        throw new IOException("SentencePiece model not found at " + vocabFile
                + " and AssetManager fallback unavailable (asset=" + assetName + ")");
    }

    /** Read an entire InputStream into a byte[]. */
    private static byte[] readAllBytes(InputStream in) throws IOException {
        try (InputStream is = in) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream(1024 * 1024); // 1 MB initial
            byte[] buf = new byte[8192];
            int n;
            while ((n = is.read(buf)) != -1) {
                bos.write(buf, 0, n);
            }
            return bos.toByteArray();
        }
    }

    /** Tiny sequential protobuf wire-format reader (decode-only, no dependency). */
    private static final class ProtoReader {
        private final byte[] buf;
        private int pos;

        ProtoReader(byte[] buf) {
            this.buf = buf;
        }

        boolean hasNext() {
            return pos < buf.length;
        }

        int readVarint32() {
            int result = 0;
            int shift = 0;
            while (true) {
                byte b = buf[pos++];
                result |= (b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            return result;
        }

        long readVarint64() {
            long result = 0;
            int shift = 0;
            while (true) {
                byte b = buf[pos++];
                result |= (long) (b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            return result;
        }

        int readFixed32() {
            int v = (buf[pos] & 0xFF) | ((buf[pos + 1] & 0xFF) << 8)
                    | ((buf[pos + 2] & 0xFF) << 16) | ((buf[pos + 3] & 0xFF) << 24);
            pos += 4;
            return v;
        }

        byte[] readBytes() {
            int len = readVarint32();
            byte[] out = new byte[len];
            System.arraycopy(buf, pos, out, 0, len);
            pos += len;
            return out;
        }

        void skip(int wireType) {
            switch (wireType) {
                case 0: readVarint64(); break;          // varint
                case 1: pos += 8; break;                 // fixed64
                case 2: int len = readVarint32(); pos += len; break; // length-delimited
                case 5: pos += 4; break;                 // fixed32
                default: throw new IllegalStateException("Unsupported protobuf wire type: " + wireType);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Public API (unchanged signatures — TranslationModule needs no changes)
    // ---------------------------------------------------------------------

    public TokenizerResult tokenize(String srcLanguage, String tgtLanguage, String text) {
        if (!loaded) return null;

        int[] originalIds = encodeToIds(text);
        int[] ids = new int[originalIds.length];

        for (int i = 0; i < originalIds.length; i++) {
            int id = originalIds[i] + 1;
            switch (id) {
                case 1: id = 3; break;
                case 2: id = 0; break;
                case 3: id = 2; break;
            }
            ids[i] = id;
        }

        int eos = pieceToId("</s>");
        int srcLangId = getLanguageID(srcLanguage);

        int[] idsExtended = new int[ids.length + 2];
        System.arraycopy(ids, 0, idsExtended, 0, ids.length);
        idsExtended[ids.length] = eos;
        idsExtended[ids.length + 1] = srcLangId;

        int[] attentionMask = new int[idsExtended.length];
        Arrays.fill(attentionMask, 1);

        return new TokenizerResult(idsExtended, attentionMask);
    }

    public String decode(int[] ids) {
        if (!loaded) return "";

        StringBuilder text = new StringBuilder();
        ByteArrayOutputStream pendingBytes = new ByteArrayOutputStream();

        for (int rawId : ids) {
            int id = rawId;
            switch (id) {
                case 3: id = 1; break;
                case 0: id = 2; break;
                case 2: id = 3; break;
            }
            id = id - 1;
            if (id < 0 || id >= idToPiece.length) {
                continue; // language-id token or out-of-vocab control id: not renderable text
            }

            String piece = idToPiece[id];
            int byteVal = pieceToByteValue(piece);
            if (byteVal >= 0) {
                pendingBytes.write(byteVal);
                continue;
            }
            if (pendingBytes.size() > 0) {
                text.append(new String(pendingBytes.toByteArray(), StandardCharsets.UTF_8));
                pendingBytes.reset();
            }
            if (isControlPiece(piece)) continue;
            text.append(piece.replace(SPIECE_UNDERLINE, " "));
        }
        if (pendingBytes.size() > 0) {
            text.append(new String(pendingBytes.toByteArray(), StandardCharsets.UTF_8));
        }

        String out = text.toString();
        if (out.startsWith(" ")) out = out.substring(1); // drop dummy-prefix leading space
        return out;
    }

    public int pieceToId(String piece) {
        if (!loaded) return 0;
        Integer rawId = pieceToIdMap.get(piece);
        int id = (rawId != null ? rawId : 0) + 1;
        switch (id) {
            case 1: return 3;
            case 2: return 0;
            case 3: return 2;
            default: return id;
        }
    }

    public int getLanguageID(String languageCode) {
        for (int i = 0; i < LANGUAGES_NLLB.length; i++) {
            if (LANGUAGES_NLLB[i].equals(languageCode)) {
                return DICTIONARY_LENGTH + 1 + i;
            }
        }
        return DICTIONARY_LENGTH + 1;
    }

    // ---------------------------------------------------------------------
    // SentencePiece BPE encode
    // ---------------------------------------------------------------------

    private int[] encodeToIds(String text) {
        List<String> pieces = bpeEncode(normalize(text));
        int[] ids = new int[pieces.size()];
        for (int i = 0; i < pieces.size(); i++) {
            Integer id = pieceToIdMap.get(pieces.get(i));
            ids[i] = (id != null) ? id : pieceToIdMap.getOrDefault(unkPiece, 0);
        }
        return ids;
    }

    private String normalize(String text) {
        String nfkc = Normalizer2.getNFKCInstance().normalize(text);
        String collapsed = nfkc.trim().replaceAll("\\s+", " ");
        return SPIECE_UNDERLINE + collapsed.replace(" ", SPIECE_UNDERLINE);
    }

    private List<String> bpeEncode(String normalized) {
        List<Sym> symbols = new ArrayList<>();
        int i = 0;
        int idx = 0;
        while (i < normalized.length()) {
            int cp = normalized.codePointAt(i);
            int cc = Character.charCount(cp);
            String s = normalized.substring(i, i + cc);
            if (pieceToIdMap.containsKey(s)) {
                symbols.add(new Sym(s, idx++));
            } else if (byteFallbackAvailable) {
                for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
                    symbols.add(new Sym(byteToken[b & 0xFF], idx++));
                }
            } else {
                symbols.add(new Sym(unkPiece, idx++));
            }
            i += cc;
        }

        List<String> result = new ArrayList<>();
        if (symbols.isEmpty()) return result;

        for (int k = 0; k < symbols.size(); k++) {
            symbols.get(k).prev = (k > 0) ? symbols.get(k - 1) : null;
            symbols.get(k).next = (k < symbols.size() - 1) ? symbols.get(k + 1) : null;
        }

        PriorityQueue<Bigram> queue = new PriorityQueue<>();
        for (Sym s : symbols) {
            offerBigram(queue, s);
        }

        while (!queue.isEmpty()) {
            Bigram bg = queue.poll();
            if (bg.isStale()) continue;

            Sym left = bg.left;
            Sym right = bg.right;

            left.piece = left.piece + right.piece;
            left.version++;
            right.valid = false;
            left.next = right.next;
            if (right.next != null) right.next.prev = left;

            offerBigram(queue, left);
            if (left.prev != null) offerBigram(queue, left.prev);
        }

        for (Sym s : symbols) {
            if (s.valid) result.add(s.piece);
        }
        return result;
    }

    private void offerBigram(PriorityQueue<Bigram> queue, Sym left) {
        if (left.next == null) return;
        String merged = left.piece + left.next.piece;
        Integer id = pieceToIdMap.get(merged);
        if (id != null) {
            queue.offer(new Bigram(left, left.next, idToScore[id]));
        }
    }

    private int pieceToByteValue(String piece) {
        if (piece != null && piece.length() == 6 && piece.startsWith("<0x") && piece.endsWith(">")) {
            try {
                return Integer.parseInt(piece.substring(3, 5), 16);
            } catch (NumberFormatException e) {
                return -1;
            }
        }
        return -1;
    }

    private boolean isControlPiece(String piece) {
        return "<s>".equals(piece) || "</s>".equals(piece) || "<pad>".equals(piece) || "<unk>".equals(piece);
    }

    private static final class Sym {
        String piece;
        Sym prev;
        Sym next;
        boolean valid = true;
        int version = 0;
        final int index;

        Sym(String piece, int index) {
            this.piece = piece;
            this.index = index;
        }
    }

    private static final class Bigram implements Comparable<Bigram> {
        final Sym left;
        final Sym right;
        final float score;
        final int leftVersion;
        final int rightVersion;

        Bigram(Sym left, Sym right, float score) {
            this.left = left;
            this.right = right;
            this.score = score;
            this.leftVersion = left.version;
            this.rightVersion = right.version;
        }

        boolean isStale() {
            return !left.valid || !right.valid
                    || left.version != leftVersion
                    || right.version != rightVersion
                    || left.next != right;
        }

        @Override
        public int compareTo(Bigram o) {
            int c = Float.compare(o.score, score); // higher score merges first
            if (c != 0) return c;
            return Integer.compare(left.index, o.left.index); // leftmost wins ties
        }
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
