# Add project specific ProGuard rules here.
# ONNX Runtime must not be obfuscated.
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# Keep DJL SentencePiece tokenizer
-keep class ai.djl.sentencepiece.** { *; }
