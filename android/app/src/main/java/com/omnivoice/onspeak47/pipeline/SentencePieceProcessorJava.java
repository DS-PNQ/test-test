/*
 * OmniVoice — SentencePiece JNI wrapper
 *
 * Mirrors RTranslator-2.00's SentencePieceProcessorJava.java
 */

package com.omnivoice.onspeak47.pipeline;


/**
 * Thin Java wrapper around the SentencePiece native library.
 *
 * The native methods are implemented in C++ (loaded via System.loadLibrary).
 * The .so file must be included in the APK's jniLibs directory.
 */
public class SentencePieceProcessorJava {

    static {
        System.loadLibrary("sentencepiece_jni");
    }

    private long nativeHandle = 0;

    /**
     * Load a SentencePiece model file.
     *
     * @param modelPath Absolute path to the .model file in internal storage
     */
    public native void Load(String modelPath);

    /**
     * Encode text to token IDs.
     *
     * @param text Input text
     * @return Array of SentencePiece token IDs
     */
    public native int[] encode(String text);

    /**
     * Decode token IDs back to text.
     *
     * @param ids Token ID array
     * @return Decoded text string
     */
    public native String decode(int[] ids);

    /**
     * Get the token ID for a specific piece string (e.g., "</s>").
     *
     * @param piece The piece string
     * @return Token ID
     */
    public native int PieceToID(String piece);

    /**
     * Get the piece string for a specific token ID.
     *
     * @param id Token ID
     * @return Piece string
     */
    public native String IDToPiece(int id);

    /**
     * Get the total vocabulary size.
     *
     * @return Number of pieces in the model
     */
    public native int GetPieceSize();
}
