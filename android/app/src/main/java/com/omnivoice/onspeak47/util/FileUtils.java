/*
 * OmniVoice — File Utilities
 */

package com.omnivoice.onspeak47.util;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;


/**
 * File helper methods for copying assets to internal storage.
 */
public final class FileUtils {

    private static final String TAG = "FileUtils";
    private static final int BUFFER_SIZE = 8192;

    private FileUtils() {}

    /**
     * Copy a file from the app's assets directory to the specified directory.
     * Also automatically copies a matching .onnx_data file if it exists.
     *
     * @param context   Application context
     * @param assetName Filename in the assets directory
     * @param targetDir Target directory
     * @return Path to the copied file
     */
    public static String copyAssetToDir(Context context, String assetName, File targetDir) {
        copySingleAssetToDir(context, assetName, targetDir);

        // If it's an ONNX file, check for external data (.onnx_data)
        if (assetName.endsWith(".onnx")) {
            String dataFile = assetName + "_data";
            try {
                // Check if the data file exists in assets
                context.getAssets().open(dataFile).close();
                copySingleAssetToDir(context, dataFile, targetDir);
            } catch (IOException ignored) {
                // .onnx_data doesn't exist for this model, which is fine
            }
        }

        return new File(targetDir, assetName).getAbsolutePath();
    }

    private static void copySingleAssetToDir(Context context, String assetName, File targetDir) {
        File outFile = new File(targetDir, assetName);

        // FAST PATH: reuse an existing copy — but only if it's complete. An
        // earlier interrupted write leaves a truncated file on disk whose
        // createSession() would then fail (or silently corrupt) forever.
        if (outFile.exists()) {
            long expected = assetSize(context, assetName);
            if (expected >= 0 && outFile.length() != expected) {
                Log.w(TAG, "Discarding incomplete copy of " + assetName
                        + " (on disk " + outFile.length() + " B vs asset " + expected + " B)");
                outFile.delete();
            } else {
                Log.d(TAG, "Asset already exists: " + outFile.getAbsolutePath());
                return;
            }
        }

        // Ensure parent directories exist
        if (!targetDir.exists()) {
            targetDir.mkdirs();
        }

        try (InputStream in = context.getAssets().open(assetName);
             FileOutputStream out = new FileOutputStream(outFile)) {

            byte[] buffer = new byte[BUFFER_SIZE];
            int bytesRead;
            long totalBytes = 0;

            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                totalBytes += bytesRead;
            }

            out.flush();
            Log.i(TAG, String.format("Copied asset '%s' → %s (%.1f MB)",
                    assetName, outFile.getAbsolutePath(), totalBytes / 1e6));

        } catch (IOException e) {
            Log.e(TAG, "Failed to copy asset: " + assetName, e);
            if (outFile.exists()) outFile.delete();
            // Propagate — a caller that then passes the returned path into
            // OrtEnvironment.createSession must not be handed a missing file.
            throw new RuntimeException("Failed to copy asset to storage: " + assetName, e);
        }
    }

    /**
     * Size in bytes of an asset (may exceed 2 GiB; NOT limited by
     * AssetFileDescriptor). Returns -1 if the asset cannot be opened.
     */
    private static long assetSize(Context context, String assetName) {
        try (android.content.res.AssetFileDescriptor afd =
                     context.getAssets().openFd(assetName)) {
            return afd.getLength();
        } catch (IOException e) {
            // openFd requires the asset to be stored uncompressed (it is:
            // build.gradle noCompress covers onnx/json). If it's compressed,
            // fall back to the stream length.
            try (InputStream in = context.getAssets().open(assetName)) {
                return in.available();
            } catch (IOException e2) {
                return -1;
            }
        }
    }

    /**
     * Copy a file from the app's assets directory to internal storage.
     * Deprecated: Use copyAssetToDir instead to avoid filling up internal storage.
     */
    public static String copyAssetToInternal(Context context, String assetName) {
        return copyAssetToDir(context, assetName, context.getFilesDir());
    }

    /**
     * Check if a model file exists in internal storage.
     */
    public static boolean modelExists(Context context, String modelName) {
        return new File(context.getFilesDir(), modelName).exists();
    }

    /**
     * Get the size of a file in megabytes.
     */
    public static float getFileSizeMB(File file) {
        if (!file.exists()) return 0;
        return file.length() / (1024f * 1024f);
    }

    /**
     * Delete a file from internal storage.
     */
    public static boolean deleteInternalFile(Context context, String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        if (file.exists()) {
            return file.delete();
        }
        return false;
    }
}
