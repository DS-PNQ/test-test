# Add project specific ProGuard rules here.
# ONNX Runtime must not be obfuscated.
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# Keep DJL SentencePiece tokenizer (used by Tokenizer.java)
-keep class ai.djl.sentencepiece.** { *; }
