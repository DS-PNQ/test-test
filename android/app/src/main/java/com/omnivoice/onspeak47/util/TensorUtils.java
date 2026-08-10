/*
 * OmniVoice — Tensor Utilities for ONNX Runtime
 */

package com.omnivoice.onspeak47.util;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;


/**
 * Helper methods for creating ONNX Runtime tensors.
 */
public final class TensorUtils {

    private TensorUtils() {}

    public static OnnxTensor intArrayToTensor(OrtEnvironment env, int[] data)
            throws OrtException {
        long[] shape = new long[]{1, data.length};
        long[] longData = new long[data.length];
        for (int i = 0; i < data.length; i++) longData[i] = data[i];
        return OnnxTensor.createTensor(env, LongBuffer.wrap(longData), shape);
    }

    public static OnnxTensor createFloatTensor(OrtEnvironment env, long[] shape)
            throws OrtException {
        int totalSize = 1;
        for (long dim : shape) totalSize *= (int) dim;
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[totalSize]), shape);
    }

    public static OnnxTensor booleanToTensor(OrtEnvironment env, boolean value)
            throws OrtException {
        return OnnxTensor.createTensor(env, new boolean[]{value});
    }

    /**
     * Clones an OnnxTensor to keep data alive after OrtSession.Result is closed.
     */
    public static OnnxTensor cloneTensor(OrtEnvironment env, OnnxTensor tensor) throws OrtException {
        if (tensor == null) return null;
        long[] shape = tensor.getInfo().getShape();

        // Android ONNX Runtime 1.13+ uses specific buffer getters
        try {
            FloatBuffer floatBuffer = tensor.getFloatBuffer();
            float[] copy = new float[floatBuffer.remaining()];
            floatBuffer.get(copy);
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(copy), shape);
        } catch (IllegalStateException e) {
            // Not a float tensor, try long
            try {
                LongBuffer longBuffer = tensor.getLongBuffer();
                long[] copy = new long[longBuffer.remaining()];
                longBuffer.get(copy);
                return OnnxTensor.createTensor(env, LongBuffer.wrap(copy), shape);
            } catch (IllegalStateException e2) {
                throw new OrtException("Unsupported tensor type for cloning");
            }
        }
    }

    public static float[] flatten2D(float[][] data) {
        int rows = data.length, cols = data[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) System.arraycopy(data[i], 0, flat, i * cols, cols);
        return flat;
    }

    public static int argmax(float[] values) {
        if (values == null || values.length == 0) return 0;
        int maxIdx = 0; float maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) { maxVal = values[i]; maxIdx = i; }
        }
        return maxIdx;
    }

    public static void softmax(float[] values) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : values) if (v > max) max = v;
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.exp(values[i] - max);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) values[i] /= sum;
    }
}
