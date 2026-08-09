/*
 * OmniVoice — SentencePiece JNI Wrapper
 * Adapted from RTranslator (Apache 2.0) by Luca Martino.
 */

package com.omnivoice.onspeak47.pipeline;

public class SentencePieceProcessor {
    private final String[] specialTokens = {"<s>", "<pad>", "</s>", "<unk>"};

    static {
        System.loadLibrary("sentencepiece");
    }

    private final long spProcessorPointer;

    public SentencePieceProcessor() {
        spProcessorPointer = SentencePieceProcessorNative();
    }

    public void Load(String vocab_file) {
        LoadNative(spProcessorPointer, vocab_file);
    }

    public int[] encode(String text) {
        return encodeNative(spProcessorPointer, text);
    }

    public int PieceToID(String token) {
        for (int i = 0; i < specialTokens.length; i++) {
            if (token.equals(specialTokens[i])) {
                return i;
            }
        }
        return PieceToIDNative(spProcessorPointer, token) + 1;
    }

    public String IDToPiece(int id) {
        return IDToPieceNative(spProcessorPointer, id);
    }

    private native long SentencePieceProcessorNative();
    private native void LoadNative(long processor, String vocab_file);
    private native int[] encodeNative(long processor, String text);
    private native int PieceToIDNative(long processor, String token);
    public native String IDToPieceNative(long processor, int id);
    private native String decodeNative(long processor, int[] ids);
}
