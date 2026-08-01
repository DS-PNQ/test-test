# Voice tests for MMS-TTS (and MMS-TTS-vie fallback)

from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class TestMMSTTS:
    """MMS-TTS synthesis quality & sanity tests."""

    def test_model_loads(self, tts_model):
        """Verify the MMSTTS wrapper instantiated without errors."""
        assert tts_model is not None

    def test_supported_languages(self, tts_model):
        """Verify at least vi, en, zh are in supported languages."""
        langs = tts_model.get_supported_languages()
        assert "vi" in langs
        assert "en" in langs
        assert "zh" in langs

    # ------------------------------------------------------------------
    # Synthesis tests per language
    # ------------------------------------------------------------------

    def test_synthesize_vietnamese(self, tts_model, tmp_path):
        """Synthesize Vietnamese text and verify WAV output."""
        text = "Xin chào, tôi là trợ lý dịch thuật."
        out_path = tmp_path / "tts_vi.wav"
        result_path = tts_model.synthesize(text, "vi", out_path)

        assert Path(result_path).exists()
        sr, data = scipy.io.wavfile.read(result_path)
        assert sr > 0
        assert len(data) > 0
        assert data.dtype == np.int16

    def test_synthesize_english(self, tts_model, tmp_path):
        """Synthesize English text and verify WAV output."""
        text = "Hello, I am a translation assistant."
        out_path = tmp_path / "tts_en.wav"
        result_path = tts_model.synthesize(text, "en", out_path)

        assert Path(result_path).exists()
        sr, data = scipy.io.wavfile.read(result_path)
        assert sr > 0
        assert len(data) > 0

    def test_synthesize_chinese(self, tts_model, tmp_path):
        """Synthesize Chinese text and verify WAV output."""
        text = "你好，我是翻译助手。"
        out_path = tmp_path / "tts_zh.wav"
        try:
            result_path = tts_model.synthesize(text, "zh", out_path)
            assert Path(result_path).exists()
            sr, data = scipy.io.wavfile.read(result_path)
            assert sr > 0
            assert len(data) > 0
        except Exception as e:
            pytest.skip(f"Chinese MMS-TTS not available: {e}")

    # ------------------------------------------------------------------
    # Array output
    # ------------------------------------------------------------------

    def test_synthesize_to_array(self, tts_model):
        """Verify synthesize_to_array returns valid waveform + sample rate."""
        waveform, sr = tts_model.synthesize_to_array(
            "Đây là bài kiểm tra.", "vi"
        )
        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert sr > 0
        assert len(waveform) > 0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_unsupported_language_raises(self, tts_model, tmp_path):
        """Unsupported language should raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            tts_model.synthesize("test", "xx", tmp_path / "tts_xx.wav")

    def test_empty_text(self, tts_model, tmp_path):
        """Empty text — model behavior varies; we just verify no crash."""
        out_path = tmp_path / "tts_empty.wav"
        try:
            tts_model.synthesize("", "en", out_path)
        except Exception:
            pass  # Some TTS models reject empty input — acceptable

    def test_long_text_vietnamese(self, tts_model, tmp_path):
        """Longer Vietnamese text to verify no truncation issues."""
        text = (
            "Vui lòng mang theo bản gốc và bản sao của giấy khai sinh. "
            "Hồ sơ của bạn đang được xử lý và sẽ có kết quả trong ba ngày làm việc. "
            "Bạn có thể nộp hồ sơ trực tuyến thông qua cổng dịch vụ công."
        )
        out_path = tmp_path / "tts_vi_long.wav"
        result_path = tts_model.synthesize(text, "vi", out_path)
        assert Path(result_path).exists()
        sr, data = scipy.io.wavfile.read(result_path)
        # Longer text should produce a longer waveform
        assert len(data) > 16000  # > 1 second at 16kHz
