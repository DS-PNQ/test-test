# Accuracy tests for Whisper transcription

import math
from pathlib import Path

import pytest
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class TestWhisperASR:
    """Whisper Small ASR quality & sanity tests.

    These tests verify that the WhisperASR wrapper loads correctly and
    produces non-empty transcriptions.  Full WER/CER tests require
    reference audio+transcript pairs in ``tests_local/data/``.
    """

    def test_model_loads(self, asr_model):
        """Verify the model and processor loaded without errors."""
        assert asr_model is not None
        assert asr_model.model is not None
        assert asr_model.processor is not None

    def test_transcribe_synthetic_silence(self, asr_model, tmp_path):
        """Feed a short silent WAV — should return empty or minimal text."""
        import scipy.io.wavfile

        sr = 16000
        duration_s = 2
        silence = np.zeros(sr * duration_s, dtype=np.float32)
        wav_path = tmp_path / "silence.wav"
        scipy.io.wavfile.write(str(wav_path), sr, silence)

        result = asr_model.transcribe(str(wav_path), language="en")
        # Silence may decode to empty or a short noise artifact — either is fine
        assert isinstance(result.text, str)
        assert result.language == "en"

    def test_transcribe_array(self, asr_model):
        """Verify transcribe_array accepts a numpy array and returns a result."""
        sr = 16000
        duration_s = 1
        tone = np.sin(2 * np.pi * 440 * np.arange(sr * duration_s) / sr).astype(np.float32)

        result = asr_model.transcribe_array(tone, language="en")
        assert isinstance(result.text, str)
        assert result.language == "en"

    @pytest.mark.skipif(
        not (DATA_DIR / "audio_vi_sample.wav").exists(),
        reason="No Vietnamese audio sample available",
    )
    def test_transcribe_vietnamese(self, asr_model):
        """Transcribe a Vietnamese audio sample and check basic output."""
        result = asr_model.transcribe(
            str(DATA_DIR / "audio_vi_sample.wav"), language="vi"
        )
        assert result.language == "vi"
        assert len(result.text) > 0

    @pytest.mark.skipif(
        not (DATA_DIR / "audio_en_sample.wav").exists(),
        reason="No English audio sample available",
    )
    def test_transcribe_english(self, asr_model):
        """Transcribe an English audio sample and check basic output."""
        result = asr_model.transcribe(
            str(DATA_DIR / "audio_en_sample.wav"), language="en"
        )
        assert result.language == "en"
        assert len(result.text) > 0

    @pytest.mark.skipif(
        not (DATA_DIR / "audio_en_sample.wav").exists(),
        reason="No English audio sample with reference transcript",
    )
    def test_wer_english(self, asr_model):
        """Compute WER for English sample against reference transcript."""
        import jiwer

        ref_path = DATA_DIR / "audio_en_sample.txt"
        if not ref_path.exists():
            pytest.skip("No reference transcript for WER scoring")

        reference = ref_path.read_text(encoding="utf-8").strip()
        result = asr_model.transcribe(
            str(DATA_DIR / "audio_en_sample.wav"), language="en"
        )

        wer = jiwer.wer(reference, result.text)
        assert math.isfinite(wer)
        # Log the score; we don't assert a threshold in Phase 2
        print(f"English WER: {wer:.4f}")
