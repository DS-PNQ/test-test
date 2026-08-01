# End-to-end pipeline tests
#
# Verifies the full OmniVoicePipeline: ASR → Translation → TTS.

from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class TestPipeline:
    """End-to-end pipeline integration tests."""

    def test_text_only_pipeline_vi_en(self, pipeline):
        """Text-only shortcut: vi → en translation."""
        result = pipeline.translate_text(
            "Tôi muốn gia hạn căn cước công dân.", "vi", "en"
        )
        assert result.transcript == "Tôi muốn gia hạn căn cước công dân."
        assert result.src_language == "vi"
        assert result.tgt_language == "en"
        assert len(result.translation) > 0
        assert result.timings["total_ms"] > 0
        print(f"\n  vi→en: {result.translation}")

    def test_text_only_pipeline_vi_zh(self, pipeline):
        """Text-only shortcut: vi → zh translation."""
        result = pipeline.translate_text(
            "Vui lòng mang theo bản gốc và bản sao của giấy khai sinh.",
            "vi",
            "zh_hans",
        )
        assert result.tgt_language == "zh_hans"
        assert len(result.translation) > 0
        print(f"\n  vi→zh: {result.translation}")

    def test_text_only_pipeline_zh_vi(self, pipeline):
        """Text-only shortcut: zh → vi translation."""
        result = pipeline.translate_text(
            "请携带出生证明原件和复印件。", "zh_hans", "vi"
        )
        assert result.tgt_language == "vi"
        assert len(result.translation) > 0
        print(f"\n  zh→vi: {result.translation}")

    def test_text_only_pipeline_en_vi(self, pipeline):
        """Text-only shortcut: en → vi translation."""
        result = pipeline.translate_text(
            "Please bring both the original and a copy of your birth certificate.",
            "en",
            "vi",
        )
        assert result.tgt_language == "vi"
        assert len(result.translation) > 0
        print(f"\n  en→vi: {result.translation}")

    def test_full_pipeline_with_synthetic_audio(self, pipeline, tmp_path):
        """Full ASR → Translation → TTS with a synthetic tone.

        This test exercises all three stages but uses a synthetic tone
        (not speech), so the ASR output will be noise/empty.  The goal
        is to verify the pipeline doesn't crash end-to-end.
        """
        sr = 16000
        duration_s = 2
        tone = np.sin(2 * np.pi * 440 * np.arange(sr * duration_s) / sr).astype(np.float32)
        wav_path = tmp_path / "input_tone.wav"
        tone_int16 = (tone * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(wav_path), sr, tone_int16)

        result = pipeline.process(
            str(wav_path),
            src_lang="en",
            tgt_lang="vi",
            output_dir=str(tmp_path / "output"),
        )

        assert isinstance(result.transcript, str)
        assert isinstance(result.translation, str)
        assert result.audio_path is not None
        assert Path(result.audio_path).exists()
        assert result.timings["total_ms"] > 0

        print(f"\n  Timings: {result.timings}")

    def test_pipeline_result_timings(self, pipeline):
        """Verify timings dict has expected keys."""
        result = pipeline.translate_text("Hello world.", "en", "vi")
        assert "translation_ms" in result.timings
        assert "total_ms" in result.timings
        assert result.timings["total_ms"] >= result.timings["translation_ms"]
