/*
 * OmniVoice — JNI bridge for SentencePiece native library.
 * Adapted from RTranslator (Apache 2.0) by Luca Martino.
 */

#include <jni.h>
#include <stdio.h>
#include <string>
#include "sentencepiece_processor.h"

using namespace sentencepiece;

std::string jstringToString(JNIEnv* env, jstring jstr);
jintArray intVectorTojintArray(JNIEnv* env, std::vector<int> vector);
std::vector<int> jintArrayTointVector(JNIEnv* env, jintArray jarray);
jstring stringToJstring(JNIEnv* env, std::string str);

extern "C" jlong
Java_com_omnivoice_onspeak47_pipeline_SentencePieceProcessor_SentencePieceProcessorNative(JNIEnv* env, jobject) {
    return (long)(new SentencePieceProcessor());
}

extern "C" void
Java_com_omnivoice_onspeak47_pipeline_SentencePieceProcessor_LoadNative(JNIEnv* env, jobject, jlong processor, jstring vocab_file) {
    SentencePieceProcessor *proc = (SentencePieceProcessor *)processor;
    std::string vocab_file_string = jstringToString(env, vocab_file);
    (*proc).Load(vocab_file_string);
}

extern "C" jintArray
Java_com_omnivoice_onspeak47_pipeline_SentencePieceProcessor_encodeNative(JNIEnv* env, jobject, jlong processor, jstring text) {
    SentencePieceProcessor *proc = (SentencePieceProcessor *)processor;
    std::vector<int> ids(1024, 0);
    std::string string = jstringToString(env, text);
    (*proc).Encode(string, &ids);
    return intVectorTojintArray(env, ids);
}

extern "C" jint
Java_com_omnivoice_onspeak47_pipeline_SentencePieceProcessor_PieceToIDNative(JNIEnv* env, jobject, jlong processor, jstring token) {
    SentencePieceProcessor *proc = (SentencePieceProcessor *)processor;
    return (*proc).PieceToId(jstringToString(env, token));
}

extern "C" jstring
Java_com_omnivoice_onspeak47_pipeline_SentencePieceProcessor_IDToPieceNative(JNIEnv* env, jobject, jlong processor, jint id) {
    SentencePieceProcessor *proc = (SentencePieceProcessor *)processor;
    int idConverted = (int) id;
    std::string outputString = (*proc).IdToPiece(idConverted);
    return stringToJstring(env, outputString);
}

extern "C" jstring
Java_com_omnivoice_onspeak47_pipeline_SentencePieceProcessor_decodeNative(JNIEnv* env, jobject, jlong processor, jintArray ids) {
    SentencePieceProcessor *proc = (SentencePieceProcessor *)processor;
    std::vector<int> idsConverted = jintArrayTointVector(env, ids);
    std::string outputString;
    (*proc).Decode(idsConverted, &outputString);
    return stringToJstring(env, outputString);
}


std::string jstringToString(JNIEnv* env, jstring jstr) {
    jboolean isCopy;
    const char *convertedValue = (env)->GetStringUTFChars(jstr, &isCopy);
    std::string string = std::string(convertedValue);
    env->ReleaseStringUTFChars(jstr, convertedValue);
    return string;
}

jintArray intVectorTojintArray(JNIEnv* env, std::vector<int> vector) {
    jintArray jarray = env->NewIntArray(vector.size());
    env->SetIntArrayRegion(jarray, 0, vector.size(), reinterpret_cast<jint*>(vector.data()));
    return jarray;
}

std::vector<int> jintArrayTointVector(JNIEnv* env, jintArray jarray) {
    jsize size = env->GetArrayLength(jarray);
    std::vector<int> vector(size);
    env->GetIntArrayRegion(jarray, jsize{0}, size, &vector[0]);
    std::vector<int> vectorFinal(vector.begin(), vector.end());
    return vectorFinal;
}

jstring stringToJstring(JNIEnv* env, std::string str) {
    const char* chars = str.data();
    return env->NewStringUTF(chars);
}