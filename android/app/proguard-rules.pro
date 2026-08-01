# Add project specific ProGuard rules here.
# ONNX Runtime must not be obfuscated.
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# Keep SentencePiece JNI native methods
-keepclasseswithmembernames class com.omnivoice.onspeak47.pipeline.SentencePieceProcessorJava {
    native <methods>;
}

# Keep ML Kit language identification
-keep class com.google.mlkit.** { *; }
