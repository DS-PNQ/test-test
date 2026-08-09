# Add project specific ProGuard rules here.
# ONNX Runtime must not be obfuscated.
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# SentencePiece: uses pure-Java implementation (no JNI/native library)
