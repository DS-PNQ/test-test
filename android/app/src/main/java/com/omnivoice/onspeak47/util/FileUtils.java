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
     * Copy a file from the app's assets directory to internal storage.
     *
     * @param context   Application context
     * @param assetName Filename in the assets directory
     * @return Path to the copied file in internal storage
     */
    public static String copyAssetToInternal(Context context, String assetName) {
        File outFile = new File(context.getFilesDir(), assetName);

        if (outFile.exists()) {
            Log.d(TAG, "Asset already exists: " + outFile.getAbsolutePath());
            return outFile.getAbsolutePath();
        }

        // Ensure parent directories exist
        outFile.getParentFile().mkdirs();

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
        }

        return outFile.getAbsolutePath();
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
