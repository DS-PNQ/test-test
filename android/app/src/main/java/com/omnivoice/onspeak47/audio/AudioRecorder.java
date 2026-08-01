/*
 * OmniVoice — Audio Recorder
 *
 * Records audio from the device microphone to a WAV file
 * at 16kHz mono (Whisper requirement).
 */

package com.omnivoice.onspeak47.audio;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


/**
 * Records 16kHz mono 16-bit PCM audio to a WAV file.
 */
public class AudioRecorder {

    private static final String TAG = "AudioRecorder";

    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    private AudioRecord audioRecord;
    private Thread recordingThread;
    private boolean isRecording = false;
    private String outputPath;
    private int bufferSize;

    public AudioRecorder() {
        bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (bufferSize < 0) {
            bufferSize = SAMPLE_RATE * 2;  // 1 second of 16-bit audio
        }
    }

    /**
     * Start recording to a WAV file.
     *
     * @param outputPath Path for the output WAV file
     */
    public void startRecording(String outputPath) {
        if (isRecording) {
            Log.w(TAG, "Already recording");
            return;
        }

        this.outputPath = outputPath;

        try {
            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    bufferSize
            );

            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize");
                return;
            }

            audioRecord.startRecording();
            isRecording = true;

            recordingThread = new Thread(this::writeAudioToFile, "AudioRecorder");
            recordingThread.start();

            Log.i(TAG, "Recording started → " + outputPath);

        } catch (SecurityException e) {
            Log.e(TAG, "Audio permission not granted", e);
        }
    }

    /**
     * Stop recording and return the path to the WAV file.
     *
     * @return Path to the recorded WAV file, or null on failure
     */
    public String stopRecording() {
        if (!isRecording) return null;

        isRecording = false;

        try {
            if (recordingThread != null) {
                recordingThread.join(3000);
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Stop interrupted", e);
        }

        if (audioRecord != null) {
            try {
                audioRecord.stop();
            } catch (IllegalStateException e) {
                Log.e(TAG, "AudioRecord stop error", e);
            }
            audioRecord.release();
            audioRecord = null;
        }

        Log.i(TAG, "Recording stopped");
        return outputPath;
    }

    /**
     * Check if currently recording.
     */
    public boolean isRecording() {
        return isRecording;
    }

    // ----------------------------------------------------------------
    // WAV writing
    // ----------------------------------------------------------------

    private void writeAudioToFile() {
        File file = new File(outputPath);
        file.getParentFile().mkdirs();

        try (FileOutputStream fos = new FileOutputStream(file)) {
            // Write WAV header placeholder (will be updated at the end)
            byte[] header = new byte[44];
            fos.write(header);

            byte[] buffer = new byte[bufferSize];
            long totalBytesWritten = 0;

            while (isRecording) {
                int bytesRead = audioRecord.read(buffer, 0, bufferSize);
                if (bytesRead > 0) {
                    fos.write(buffer, 0, bytesRead);
                    totalBytesWritten += bytesRead;
                }
            }

            fos.flush();

            // Update WAV header with actual sizes
            updateWavHeader(file, totalBytesWritten);

        } catch (IOException e) {
            Log.e(TAG, "Error writing audio file", e);
        }
    }

    private void updateWavHeader(File file, long totalAudioBytes) {
        try (RandomAccessFile raf = new RandomAccessFile(file, "rw")) {
            long totalSize = totalAudioBytes + 36;
            int channels = 1;
            int bitsPerSample = 16;
            int byteRate = SAMPLE_RATE * channels * bitsPerSample / 8;
            int blockAlign = channels * bitsPerSample / 8;

            raf.seek(0);

            // RIFF header
            raf.writeBytes("RIFF");
            raf.write(intToLittleEndian((int) totalSize));
            raf.writeBytes("WAVE");

            // fmt sub-chunk
            raf.writeBytes("fmt ");
            raf.write(intToLittleEndian(16));           // Sub-chunk size
            raf.write(shortToLittleEndian((short) 1));  // PCM format
            raf.write(shortToLittleEndian((short) channels));
            raf.write(intToLittleEndian(SAMPLE_RATE));
            raf.write(intToLittleEndian(byteRate));
            raf.write(shortToLittleEndian((short) blockAlign));
            raf.write(shortToLittleEndian((short) bitsPerSample));

            // data sub-chunk
            raf.writeBytes("data");
            raf.write(intToLittleEndian((int) totalAudioBytes));

        } catch (IOException e) {
            Log.e(TAG, "Error updating WAV header", e);
        }
    }

    private byte[] intToLittleEndian(int value) {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array();
    }

    private byte[] shortToLittleEndian(short value) {
        return ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value).array();
    }
}
