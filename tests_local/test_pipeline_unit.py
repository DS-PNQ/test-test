# Unit tests for the full OmniVoicePipeline — runs locally without models.
#
# Mocks all three stages (WhisperASR, NLLBTranslator, MMSTTS) so the test
# suite can verify orchestrator logic, timing capture, argument routing,
# and result assembly without downloading models or needing a GPU.

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import scipy.io.wavfile

# Ensure the project root is on sys.path (matches conftest.py convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.asr_whisper import ASRResult
from backend.orchestrator import OmniVoicePipeline, PipelineResult


# ──────────────────────────────────────────────────────────────────────
# Helpers — mock stage builders
# ──────────────────────────────────────────────────────────────────────

def _make_mock_asr(transcript: str = "Xin chào", language: str = "vi"):
    """Return a mock WhisperASR that returns a fixed ASRResult."""
    asr = MagicMock()
    asr.transcribe.return_value = ASRResult(
        text=transcript, language=language, segments=[]
    )
    return asr


def _make_mock_translator(translation: str = "Hello"):
    """Return a mock NLLBTranslator that returns a fixed translation."""
    translator = MagicMock()
    translator.translate.return_value = translation
    return translator


def _make_mock_tts():
    """Return a mock MMSTTS that writes a real (tiny) WAV file."""
    tts = MagicMock()

    def _fake_synthesize(text, language, output_path, **kwargs):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a minimal valid WAV so Path.exists() succeeds
        sr = 16000
        silence = np.zeros(sr // 10, dtype=np.int16)  # 0.1s
        scipy.io.wavfile.write(str(output_path), sr, silence)
        return str(output_path.resolve())

    tts.synthesize.side_effect = _fake_synthesize
    return tts


def _make_wav(tmp_path: Path, name: str = "input.wav") -> Path:
    """Create a minimal valid WAV file for pipeline input."""
    sr = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    tone_int16 = (tone * 32767).astype(np.int16)
    wav_path = tmp_path / name
    scipy.io.wavfile.write(str(wav_path), sr, tone_int16)
    return wav_path


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_pipeline():
    """Pipeline wired up with all three mocked stages."""
    return OmniVoicePipeline(
        asr=_make_mock_asr(),
        translator=_make_mock_translator(),
        tts=_make_mock_tts(),
    )


# ──────────────────────────────────────────────────────────────────────
# PipelineResult dataclass
# ──────────────────────────────────────────────────────────────────────

class TestPipelineResult:
    """Verify the PipelineResult dataclass contract."""

    def test_basic_construction(self):
        r = PipelineResult(
            transcript="hello",
            src_language="en",
            translation="xin chào",
            tgt_language="vi",
            audio_path=None,
        )
        assert r.transcript == "hello"
        assert r.src_language == "en"
        assert r.translation == "xin chào"
        assert r.tgt_language == "vi"
        assert r.audio_path is None
        assert r.timings == {}

    def test_timings_default_is_independent(self):
        r1 = PipelineResult("a", "en", "b", "vi", None)
        r2 = PipelineResult("c", "vi", "d", "en", None)
        r1.timings["asr_ms"] = 100
        assert "asr_ms" not in r2.timings


# ──────────────────────────────────────────────────────────────────────
# OmniVoicePipeline.__init__
# ──────────────────────────────────────────────────────────────────────

class TestPipelineInit:
    """Verify constructor wiring."""

    def test_accepts_injected_stages(self):
        asr = _make_mock_asr()
        translator = _make_mock_translator()
        tts = _make_mock_tts()

        pipe = OmniVoicePipeline(asr=asr, translator=translator, tts=tts)

        assert pipe.asr is asr
        assert pipe.translator is translator
        assert pipe.tts is tts


# ──────────────────────────────────────────────────────────────────────
# Full pipeline: process() — audio → audio
# ──────────────────────────────────────────────────────────────────────

class TestProcessFullPipeline:
    """Tests for the full ASR → Translation → TTS pipeline."""

    def test_returns_pipeline_result(self, mock_pipeline, tmp_path):
        wav = _make_wav(tmp_path)
        result = mock_pipeline.process(
            str(wav), src_lang="vi", tgt_lang="en", output_dir=str(tmp_path / "out")
        )
        assert isinstance(result, PipelineResult)

    def test_asr_called_with_correct_args(self, mock_pipeline, tmp_path):
        wav = _make_wav(tmp_path)
        mock_pipeline.process(
            str(wav), src_lang="vi", tgt_lang="en", output_dir=str(tmp_path / "out")
        )
        mock_pipeline.asr.transcribe.assert_called_once_with(
            str(wav), language="vi"
        )

    def test_translator_receives_asr_output(self, tmp_path):
        """The translator should receive the exact text that ASR produced."""
        asr = _make_mock_asr(transcript="Tôi cần giúp đỡ", language="vi")
        translator = _make_mock_translator(translation="I need help")
        tts = _make_mock_tts()
        pipe = OmniVoicePipeline(asr=asr, translator=translator, tts=tts)

        wav = _make_wav(tmp_path)
        pipe.process(str(wav), src_lang="vi", tgt_lang="en",
                     output_dir=str(tmp_path / "out"))

        translator.translate.assert_called_once_with(
            "Tôi cần giúp đỡ", "vi", "en"
        )

    def test_tts_receives_translated_text(self, tmp_path):
        """TTS should synthesize the translated text in the target language."""
        asr = _make_mock_asr()
        translator = _make_mock_translator(translation="Good morning")
        tts = _make_mock_tts()
        pipe = OmniVoicePipeline(asr=asr, translator=translator, tts=tts)

        wav = _make_wav(tmp_path)
        pipe.process(str(wav), src_lang="vi", tgt_lang="en",
                     output_dir=str(tmp_path / "out"))

        # TTS should be called with the translated text and target language
        tts_call = tts.synthesize.call_args
        assert tts_call.args[0] == "Good morning"
        assert tts_call.args[1] == "en"

    def test_result_fields_match_stage_outputs(self, tmp_path):
        asr = _make_mock_asr(transcript="Xin chào", language="vi")
        translator = _make_mock_translator(translation="Hello")
        tts = _make_mock_tts()
        pipe = OmniVoicePipeline(asr=asr, translator=translator, tts=tts)

        wav = _make_wav(tmp_path)
        result = pipe.process(str(wav), src_lang="vi", tgt_lang="en",
                              output_dir=str(tmp_path / "out"))

        assert result.transcript == "Xin chào"
        assert result.src_language == "vi"
        assert result.translation == "Hello"
        assert result.tgt_language == "en"

    def test_output_audio_file_created(self, mock_pipeline, tmp_path):
        wav = _make_wav(tmp_path)
        result = mock_pipeline.process(
            str(wav), src_lang="vi", tgt_lang="en", output_dir=str(tmp_path / "out")
        )
        assert result.audio_path is not None
        assert Path(result.audio_path).exists()

    def test_timings_present(self, mock_pipeline, tmp_path):
        wav = _make_wav(tmp_path)
        result = mock_pipeline.process(
            str(wav), src_lang="vi", tgt_lang="en", output_dir=str(tmp_path / "out")
        )
        assert "asr_ms" in result.timings
        assert "translation_ms" in result.timings
        assert "tts_ms" in result.timings
        assert "total_ms" in result.timings

    def test_total_ms_is_sum_of_stages(self, mock_pipeline, tmp_path):
        wav = _make_wav(tmp_path)
        result = mock_pipeline.process(
            str(wav), src_lang="vi", tgt_lang="en", output_dir=str(tmp_path / "out")
        )
        expected = (
            result.timings["asr_ms"]
            + result.timings["translation_ms"]
            + result.timings["tts_ms"]
        )
        assert result.timings["total_ms"] == pytest.approx(expected, abs=0.2)

    def test_output_dir_created_if_missing(self, mock_pipeline, tmp_path):
        wav = _make_wav(tmp_path)
        nested_out = tmp_path / "deeply" / "nested" / "output"
        assert not nested_out.exists()

        mock_pipeline.process(
            str(wav), src_lang="en", tgt_lang="vi", output_dir=str(nested_out)
        )
        assert nested_out.exists()


# ──────────────────────────────────────────────────────────────────────
# Text-only pipeline: translate_text()
# ──────────────────────────────────────────────────────────────────────

class TestTranslateText:
    """Tests for the text-only shortcut (no ASR, no TTS)."""

    def test_returns_pipeline_result(self, mock_pipeline):
        result = mock_pipeline.translate_text("Hello", "en", "vi")
        assert isinstance(result, PipelineResult)

    def test_transcript_echoes_input(self, mock_pipeline):
        """translate_text should store the input text as the 'transcript'."""
        result = mock_pipeline.translate_text("Good morning", "en", "vi")
        assert result.transcript == "Good morning"

    def test_language_fields_set(self, mock_pipeline):
        result = mock_pipeline.translate_text("Hello", "en", "vi")
        assert result.src_language == "en"
        assert result.tgt_language == "vi"

    def test_translation_from_translator(self):
        translator = _make_mock_translator(translation="Xin chào")
        pipe = OmniVoicePipeline(
            asr=_make_mock_asr(), translator=translator, tts=_make_mock_tts()
        )
        result = pipe.translate_text("Hello", "en", "vi")
        assert result.translation == "Xin chào"

    def test_no_audio_path(self, mock_pipeline):
        """Text-only pipeline should not produce an audio file."""
        result = mock_pipeline.translate_text("Hello", "en", "vi")
        assert result.audio_path is None

    def test_asr_not_called(self, mock_pipeline):
        """Text-only pipeline should bypass ASR entirely."""
        mock_pipeline.translate_text("Hello", "en", "vi")
        mock_pipeline.asr.transcribe.assert_not_called()

    def test_tts_not_called(self, mock_pipeline):
        """Text-only pipeline should bypass TTS entirely."""
        mock_pipeline.translate_text("Hello", "en", "vi")
        mock_pipeline.tts.synthesize.assert_not_called()

    def test_timings_contain_translation_and_total(self, mock_pipeline):
        result = mock_pipeline.translate_text("Hello", "en", "vi")
        assert "translation_ms" in result.timings
        assert "total_ms" in result.timings

    def test_total_equals_translation_time(self, mock_pipeline):
        result = mock_pipeline.translate_text("Hello", "en", "vi")
        assert result.timings["total_ms"] == result.timings["translation_ms"]

    def test_translator_called_with_correct_args(self):
        translator = _make_mock_translator()
        pipe = OmniVoicePipeline(
            asr=_make_mock_asr(), translator=translator, tts=_make_mock_tts()
        )
        pipe.translate_text("请带上证件。", "zh_hans", "vi")
        translator.translate.assert_called_once_with("请带上证件。", "zh_hans", "vi")


# ──────────────────────────────────────────────────────────────────────
# All language directions
# ──────────────────────────────────────────────────────────────────────

class TestAllDirections:
    """Verify the pipeline handles every supported direction without error."""

    @pytest.mark.parametrize("src,tgt,text", [
        ("vi", "en", "Xin chào"),
        ("en", "vi", "Hello"),
        ("vi", "zh_hans", "Xin chào"),
        ("zh_hans", "vi", "你好"),
        ("en", "zh_hans", "Hello"),
        ("zh_hans", "en", "你好"),
    ])
    def test_translate_text_all_directions(self, src, tgt, text):
        pipe = OmniVoicePipeline(
            asr=_make_mock_asr(),
            translator=_make_mock_translator(translation="translated"),
            tts=_make_mock_tts(),
        )
        result = pipe.translate_text(text, src, tgt)
        assert result.src_language == src
        assert result.tgt_language == tgt
        assert result.translation == "translated"

    @pytest.mark.parametrize("src,tgt", [
        ("vi", "en"),
        ("en", "vi"),
        ("vi", "zh_hans"),
        ("zh_hans", "vi"),
    ])
    def test_full_pipeline_all_directions(self, src, tgt, tmp_path):
        pipe = OmniVoicePipeline(
            asr=_make_mock_asr(transcript="test input", language=src),
            translator=_make_mock_translator(translation="test output"),
            tts=_make_mock_tts(),
        )
        wav = _make_wav(tmp_path)
        result = pipe.process(
            str(wav), src_lang=src, tgt_lang=tgt,
            output_dir=str(tmp_path / f"out_{src}_{tgt}"),
        )
        assert result.src_language == src
        assert result.tgt_language == tgt
        assert Path(result.audio_path).exists()


# ──────────────────────────────────────────────────────────────────────
# Stage execution order
# ──────────────────────────────────────────────────────────────────────

class TestStageExecutionOrder:
    """Verify stages execute in the correct ASR → Translation → TTS order."""

    def test_stages_called_in_order(self, tmp_path):
        call_order = []

        asr = _make_mock_asr()
        asr.transcribe.side_effect = lambda *a, **kw: (
            call_order.append("asr"),
            ASRResult(text="text", language="vi", segments=[]),
        )[1]

        translator = _make_mock_translator()
        translator.translate.side_effect = lambda *a, **kw: (
            call_order.append("translation"),
            "translated",
        )[1]

        tts = _make_mock_tts()
        original_side_effect = tts.synthesize.side_effect
        def tts_side_effect(*args, **kwargs):
            call_order.append("tts")
            return original_side_effect(*args, **kwargs)
        tts.synthesize.side_effect = tts_side_effect

        pipe = OmniVoicePipeline(asr=asr, translator=translator, tts=tts)
        wav = _make_wav(tmp_path)
        pipe.process(str(wav), src_lang="vi", tgt_lang="en",
                     output_dir=str(tmp_path / "out"))

        assert call_order == ["asr", "translation", "tts"]


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge-case scenarios for the pipeline."""

    def test_empty_asr_output(self, tmp_path):
        """Pipeline should handle empty ASR transcription gracefully."""
        asr = _make_mock_asr(transcript="", language="en")
        translator = _make_mock_translator(translation="")
        tts = _make_mock_tts()
        pipe = OmniVoicePipeline(asr=asr, translator=translator, tts=tts)

        wav = _make_wav(tmp_path)
        result = pipe.process(str(wav), src_lang="en", tgt_lang="vi",
                              output_dir=str(tmp_path / "out"))

        assert result.transcript == ""
        assert result.translation == ""
        assert isinstance(result, PipelineResult)

    def test_empty_text_translate_text(self):
        """translate_text with empty string should still return a result."""
        translator = _make_mock_translator(translation="")
        pipe = OmniVoicePipeline(
            asr=_make_mock_asr(), translator=translator, tts=_make_mock_tts()
        )
        result = pipe.translate_text("", "en", "vi")
        assert result.transcript == ""
        assert isinstance(result, PipelineResult)

    def test_long_text_passthrough(self, tmp_path):
        """Verify long text flows through all stages without truncation."""
        long_text = "Xin chào. " * 200
        asr = _make_mock_asr(transcript=long_text.strip(), language="vi")
        translator = _make_mock_translator(translation="Hello. " * 200)
        tts = _make_mock_tts()
        pipe = OmniVoicePipeline(asr=asr, translator=translator, tts=tts)

        wav = _make_wav(tmp_path)
        result = pipe.process(str(wav), src_lang="vi", tgt_lang="en",
                              output_dir=str(tmp_path / "out"))

        assert result.transcript == long_text.strip()
        assert "Hello." in result.translation
