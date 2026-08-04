/*
 * OmniVoice — Audio Player
 *
 * Plays translated speech output WAV files.
 */

package com.omnivoice.onspeak47.audio;

import android.media.MediaPlayer;
import android.util.Log;

import java.io.IOException;


/**
 * Simple audio player for translated speech output.
 */
public class AudioPlayer {

    private static final String TAG = "AudioPlayer";

    private MediaPlayer mediaPlayer;

    /**
     * Play a WAV file.
     *
     * @param audioPath Path to the audio file
     */
    public void play(String audioPath) {
        stop();

        if (audioPath == null) return;
        java.io.File audioFile = new java.io.File(audioPath);
        if (!audioFile.exists() || audioFile.length() == 0) {
            Log.w(TAG, "Audio file does not exist or is empty: " + audioPath);
            return;
        }

        try {
            mediaPlayer = new MediaPlayer();
            mediaPlayer.setDataSource(audioPath);
            mediaPlayer.setOnCompletionListener(mp -> {
                Log.d(TAG, "Playback complete");
                mp.release();
                mediaPlayer = null;
            });
            mediaPlayer.setOnErrorListener((mp, what, extra) -> {
                Log.e(TAG, "Playback error: what=" + what + " extra=" + extra);
                mp.release();
                mediaPlayer = null;
                return true;
            });
            mediaPlayer.prepare();
            mediaPlayer.start();
            Log.i(TAG, "Playing: " + audioPath);

        } catch (IOException e) {
            Log.e(TAG, "Error playing audio", e);
        }
    }

    /**
     * Stop any currently playing audio.
     */
    public void stop() {
        if (mediaPlayer != null) {
            try {
                if (mediaPlayer.isPlaying()) {
                    mediaPlayer.stop();
                }
                mediaPlayer.release();
            } catch (IllegalStateException e) {
                Log.w(TAG, "MediaPlayer stop error", e);
            }
            mediaPlayer = null;
        }
    }

    /**
     * Release all resources.
     */
    public void release() {
        stop();
    }

    /**
     * Check if currently playing.
     */
    public boolean isPlaying() {
        return mediaPlayer != null && mediaPlayer.isPlaying();
    }
}
