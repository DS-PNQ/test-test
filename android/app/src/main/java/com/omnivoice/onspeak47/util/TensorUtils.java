/*
 * OmniVoice — Tensor Utilities for ONNX Runtime
 */

package com.omnivoice.onspeak47.util;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;


/**
 * Helper methods for creating ONNX Runtime tensors.
 *
 * Mirrors the utility methods used in RTranslator-2.00's TensorUtils.
 */
public final class TensorUtils {

    private TensorUtils() {}

    /**
     * Convert an int[] to a 2D ONNX tensor with shape [1, length].
     */
    public static OnnxTensor intArrayToTensor(OrtEnvironment env, int[] data)
            throws OrtException {
        long[] shape = new long[]{1, data.length};
        // ONNX Runtime expects int64 for input_ids
        long[] longData = new long[data.length];
        for (int i = 0; i < data.length; i++) {
            longData[i] = data[i];
        }
        return OnnxTensor.createTensor(env, LongBuffer.wrap(longData), shape);
    }

    /**
     * Create a float tensor filled with zeros.
     */
    public static OnnxTensor createFloatTensor(OrtEnvironment env, long[] shape)
            throws OrtException {
        int totalSize = 1;
        for (long dim : shape) totalSize *= (int) dim;
        float[] data = new float[totalSize];
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
    }

    /**
     * Create a boolean tensor with a single value.
     */
    public static OnnxTensor booleanToTensor(OrtEnvironment env, boolean value)
            throws OrtException {
        byte[] data = new byte[]{(byte) (value ? 1 : 0)};
        return OnnxTensor.createTensor(env, java.nio.ByteBuffer.wrap(data), new long[]{1});
    }

    /**
     * Flatten a 2D float array to 1D for tensor creation.
     */
    public static float[] flatten2D(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        }
        return flat;
    }

    /**
     * Get the index of the largest value in an array.
     */
    public static int argmax(float[] values) {
        if (values == null || values.length == 0) return 0;
        int maxIdx = 0;
        float maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) {
                maxVal = values[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Apply softmax to a float array in-place.
     */
    public static void softmax(float[] values) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : values) if (v > max) max = v;

        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.exp(values[i] - max);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }
}
