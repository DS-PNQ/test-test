/*
 * OmniVoice — Pure-Java SentencePiece BPE Processor
 *
 * Reads a SentencePiece .model protobuf file directly and implements
 * BPE encoding/decoding in pure Java — no native JNI library required.
 *
 * This replaces the DJL SpTokenizer which fails on Android because its
 * Maven artifact does not ship arm64-v8a native binaries.
 */

package com.omnivoice.onspeak47.pipeline;

import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;


/**
 * Pure-Java SentencePiece BPE processor.
 *
 * <p>Parses a SentencePiece {@code .model} protobuf file (only the fields
 * needed for encoding/decoding), then implements iterative BPE merge with
 * byte-level fallback.</p>
 *
 * <p>Designed for NLLB-200's SentencePiece BPE model (~256 000 pieces).</p>
 */
public class SentencePieceProcessor {

    private static final String TAG = "SentencePieceProcessor";

    // SentencePiece piece types (from sentencepiece_model.proto)
    public static final int TYPE_NORMAL = 1;
    public static final int TYPE_UNKNOWN = 2;
    public static final int TYPE_CONTROL = 3;
    public static final int TYPE_USER_DEFINED = 4;
    public static final int TYPE_UNUSED = 5;
    public static final int TYPE_BYTE = 6;

    /** SentencePiece "meta space" that replaces whitespace. */
    private static final String SPACE_SYMBOL = "\u2581";  // ▁

    private String[] pieces;
    private float[] scores;
    private int[] types;
    private HashMap<String, Integer> pieceToId;
    private int vocabSize;

    // Fast byte-piece lookup: byte value (0-255) → piece ID, or -1
    private int[] bytePieceIds;

    // ------------------------------------------------------------------
    // Loading
    // ------------------------------------------------------------------

    /**
     * Load a SentencePiece {@code .model} file.
     *
     * @param modelPath absolute path to the protobuf model file
     * @throws IOException if the file cannot be read or parsed
     */
    public void load(String modelPath) throws IOException {
        File file = new File(modelPath);
        byte[] data = readFileBytes(file);

        ArrayList<String> pieceList = new ArrayList<>(260_000);
        ArrayList<Float> scoreList = new ArrayList<>(260_000);
        ArrayList<Integer> typeList = new ArrayList<>(260_000);

        int[] pos = {0};
        while (pos[0] < data.length) {
            long tag = readVarint(data, pos);
            int fieldNumber = (int) (tag >>> 3);
            int wireType = (int) (tag & 0x7);

            if (fieldNumber == 1 && wireType == 2) {
                // ModelProto.pieces (field 1, length-delimited sub-message)
                int len = (int) readVarint(data, pos);
                int end = pos[0] + len;

                String piece = "";
                float score = 0.0f;
                int type = TYPE_NORMAL;

                while (pos[0] < end) {
                    long subTag = readVarint(data, pos);
                    int subField = (int) (subTag >>> 3);
                    int subWire = (int) (subTag & 0x7);

                    if (subField == 1 && subWire == 2) {
                        // SentencePiece.piece (string)
                        int sLen = (int) readVarint(data, pos);
                        piece = new String(data, pos[0], sLen, StandardCharsets.UTF_8);
                        pos[0] += sLen;
                    } else if (subField == 2 && subWire == 5) {
                        // SentencePiece.score (float32)
                        score = ByteBuffer.wrap(data, pos[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getFloat();
                        pos[0] += 4;
                    } else if (subField == 3 && subWire == 0) {
                        // SentencePiece.type (enum/varint)
                        type = (int) readVarint(data, pos);
                    } else {
                        skipField(data, pos, subWire);
                    }
                }

                pieceList.add(piece);
                scoreList.add(score);
                typeList.add(type);
            } else {
                // Skip trainer_spec, normalizer_spec, and other fields
                skipField(data, pos, wireType);
            }
        }

        vocabSize = pieceList.size();
        pieces = pieceList.toArray(new String[0]);
        scores = new float[vocabSize];
        types = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            scores[i] = scoreList.get(i);
            types[i] = typeList.get(i);
        }

        // Build reverse lookup: piece string → SP ID
        pieceToId = new HashMap<>(vocabSize * 2);
        bytePieceIds = new int[256];
        Arrays.fill(bytePieceIds, -1);

        for (int i = 0; i < vocabSize; i++) {
            pieceToId.put(pieces[i], i);

            // Index byte pieces (<0x00> through <0xFF>) for fast fallback
            if (types[i] == TYPE_BYTE
                    && pieces[i].length() == 6
                    && pieces[i].startsWith("<0x")
                    && pieces[i].charAt(5) == '>') {
                try {
                    int byteVal = Integer.parseInt(
                            pieces[i].substring(3, 5), 16);
                    bytePieceIds[byteVal] = i;
                } catch (NumberFormatException ignored) { }
            }
        }

        Log.i(TAG, "Loaded SentencePiece model: " + vocabSize + " pieces from "
                + modelPath + " (" + file.length() + " bytes)");
    }

    /** Number of pieces in the loaded vocabulary. */
    public int getVocabSize() {
        return vocabSize;
    }

    // ------------------------------------------------------------------
    // Encoding (text → SP IDs)
    // ------------------------------------------------------------------

    /**
     * Encode a text string into SentencePiece IDs.
     *
     * <p>Returns IDs in the SentencePiece model's own ID space (0-based).
     * The caller is responsible for any additional remapping (e.g. the
     * NLLB HuggingFace offset-by-one-and-swap scheme).</p>
     *
     * @param text input text (may contain Vietnamese diacritics, Chinese
     *             characters, etc.)
     * @return array of SP piece IDs
     */
    public int[] encode(String text) {
        if (text == null || text.isEmpty()) return new int[0];

        // NFKC normalization (matches SentencePiece default)
        text = Normalizer.normalize(text, Normalizer.Form.NFKC);

        // Prepend ▁ and replace spaces with ▁
        text = SPACE_SYMBOL + text.replace(" ", SPACE_SYMBOL);

        // ----- Build initial symbol list -----
        // For each character: use the single-char piece if it exists in the
        // vocabulary, otherwise fall back to UTF-8 byte pieces.
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
        ArrayList<String> symbols = new ArrayList<>();

        int i = 0;
        while (i < utf8.length) {
            int charLen = utf8CharLen(utf8[i]);
            if (i + charLen > utf8.length) charLen = utf8.length - i;

            String ch = new String(utf8, i, charLen, StandardCharsets.UTF_8);

            if (pieceToId.containsKey(ch)) {
                symbols.add(ch);
            } else {
                // Byte-level fallback: each UTF-8 byte → byte piece
                for (int j = 0; j < charLen; j++) {
                    int b = utf8[i + j] & 0xFF;
                    String bytePiece = String.format("<0x%02X>", b);
                    symbols.add(bytePiece);
                }
            }
            i += charLen;
        }

        // ----- Iterative BPE merging -----
        // At each step, find the adjacent pair whose merged string exists in
        // the vocabulary with the highest score, then merge ALL instances of
        // that pair.  Repeat until no more merges are possible.
        boolean merged = true;
        while (merged && symbols.size() >= 2) {
            merged = false;

            // Find best pair (highest score)
            float bestScore = Float.NEGATIVE_INFINITY;
            String bestMerged = null;

            for (int j = 0; j < symbols.size() - 1; j++) {
                String candidate = symbols.get(j) + symbols.get(j + 1);
                Integer id = pieceToId.get(candidate);
                if (id != null && scores[id] > bestScore) {
                    bestScore = scores[id];
                    bestMerged = candidate;
                }
            }

            if (bestMerged == null) break;

            // Merge ALL occurrences of the best pair
            ArrayList<String> next = new ArrayList<>(symbols.size());
            int j = 0;
            while (j < symbols.size()) {
                if (j < symbols.size() - 1) {
                    String candidate = symbols.get(j) + symbols.get(j + 1);
                    if (candidate.equals(bestMerged)) {
                        next.add(candidate);
                        j += 2;
                        merged = true;
                        continue;
                    }
                }
                next.add(symbols.get(j));
                j++;
            }
            symbols = next;
        }

        // ----- Convert symbols to IDs -----
        int[] ids = new int[symbols.size()];
        for (int j = 0; j < symbols.size(); j++) {
            Integer id = pieceToId.get(symbols.get(j));
            if (id != null) {
                ids[j] = id;
            } else {
                // Shouldn't happen if byte fallback works, but safeguard
                ids[j] = 0;  // <unk>
                Log.w(TAG, "Unknown piece during encode: " + symbols.get(j));
            }
        }

        return ids;
    }

    // ------------------------------------------------------------------
    // Decoding (SP IDs → text)
    // ------------------------------------------------------------------

    /**
     * Decode SentencePiece IDs back to a text string.
     *
     * <p>Handles byte-piece sequences correctly: consecutive byte pieces
     * are buffered and decoded as UTF-8 only when a non-byte piece is
     * encountered (a single multi-byte character's bytes may be split
     * across adjacent tokens).</p>
     *
     * @param ids array of SP piece IDs (in SP model's own ID space)
     * @return decoded text with ▁ replaced by spaces, trimmed
     */
    public String decode(int[] ids) {
        if (ids == null || ids.length == 0) return "";

        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
        StringBuilder result = new StringBuilder();

        for (int id : ids) {
            if (id < 0 || id >= vocabSize) continue;
            int type = types[id];
            if (type == TYPE_CONTROL || type == TYPE_UNKNOWN) continue;

            if (type == TYPE_BYTE) {
                // Buffer the raw byte — it may be part of a multi-byte char
                String piece = pieces[id];
                if (piece.length() == 6
                        && piece.startsWith("<0x")
                        && piece.charAt(5) == '>') {
                    try {
                        int b = Integer.parseInt(piece.substring(3, 5), 16);
                        byteBuffer.write(b);
                    } catch (NumberFormatException ignored) { }
                }
            } else {
                // Flush any buffered bytes as UTF-8 before appending text
                flushByteBuffer(byteBuffer, result);
                result.append(pieces[id]);
            }
        }

        // Flush remaining bytes
        flushByteBuffer(byteBuffer, result);

        // Replace ▁ with space and trim leading space
        return result.toString().replace(SPACE_SYMBOL, " ").trim();
    }

    // ------------------------------------------------------------------
    // Protobuf wire-format helpers
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

    private static long readVarint(byte[] data, int[] pos) {
        long result = 0;
        int shift = 0;
        while (pos[0] < data.length) {
            byte b = data[pos[0]++];
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }

    private static void skipField(byte[] data, int[] pos, int wireType) {
        switch (wireType) {
            case 0: // varint
                while (pos[0] < data.length && (data[pos[0]++] & 0x80) != 0) { }
                break;
            case 1: // 64-bit fixed
                pos[0] += 8;
                break;
            case 2: // length-delimited
                int len = (int) readVarint(data, pos);
                pos[0] += len;
                break;
            case 5: // 32-bit fixed
                pos[0] += 4;
                break;
            default:
                // Unknown wire type — skip one byte to avoid infinite loop
                pos[0]++;
                break;
        }
    }

    // ------------------------------------------------------------------
    // UTF-8 helpers
    // ------------------------------------------------------------------

    /** Number of bytes in the UTF-8 character starting with the given byte. */
    private static int utf8CharLen(byte b) {
        int unsigned = b & 0xFF;
        if (unsigned < 0x80) return 1;
        if (unsigned < 0xC0) return 1;  // continuation byte (invalid start)
        if (unsigned < 0xE0) return 2;
        if (unsigned < 0xF0) return 3;
        return 4;
    }

    /** Flush buffered bytes as UTF-8 text into the StringBuilder. */
    private static void flushByteBuffer(ByteArrayOutputStream buf, StringBuilder sb) {
        if (buf.size() > 0) {
            sb.append(new String(buf.toByteArray(), StandardCharsets.UTF_8));
            buf.reset();
        }
    }
}
