# Unit tests for asr_whisper — runs locally without downloading the model.
#
# These tests mock the Whisper model and processor so that the test suite
# can verify wrapper logic (argument routing, language detection, stereo→mono
# downmix, return types) without needing a GPU or network access.

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the project root is on sys.path (matches conftest.py convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.asr_whisper import ASRResult, WHISPER_LANG_TOKENS, WhisperASR

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_fake_processor():
    """Return a mock WhisperProcessor with plausible behaviour."""
    proc = MagicMock()
    # __call__ → feature extractor: returns an object with .input_features
    features = MagicMock()
    features.input_features = MagicMock()  # tensor-like
    proc.return_value = features
    # batch_decode → return a list with one transcription string
    proc.batch_decode.return_value = [" Hello world "]
    return proc


def _make_fake_model():
    """Return a mock WhisperForConditionalGeneration."""
    model = MagicMock()
    # generate → returns fake token ids (a MagicMock that acts like a tensor)
    model.generate.return_value = MagicMock()
    return model


@pytest.fixture()
def mock_asr():
    """Build a WhisperASR instance with mocked internals."""
    with patch("backend.asr_whisper.WhisperProcessor") as MockProc, \
         patch("backend.asr_whisper.WhisperForConditionalGeneration") as MockModel:
        MockProc.from_pretrained.return_value = _make_fake_processor()
        MockModel.from_pretrained.return_value = _make_fake_model()
        asr = WhisperASR()
    return asr


# ──────────────────────────────────────────────────────────────────────
# ASRResult dataclass
# ──────────────────────────────────────────────────────────────────────

class TestASRResult:
    """Verify the ASRResult dataclass contract."""

    def test_basic_construction(self):
        r = ASRResult(text="hello", language="en")
        assert r.text == "hello"
        assert r.language == "en"
        assert r.segments == []  # default

    def test_segments_default_is_list(self):
        r1 = ASRResult(text="a", language="vi")
        r2 = ASRResult(text="b", language="zh")
        # Each instance should have its own list (no shared mutable default).
        r1.segments.append("x")
        assert r2.segments == []

    def test_custom_segments(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        r = ASRResult(text="hi", language="en", segments=segs)
        assert r.segments is segs


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

class TestConstants:
    """Sanity-check module-level constants."""

    def test_lang_tokens_keys(self):
        assert set(WHISPER_LANG_TOKENS.keys()) == {"vi", "en", "zh"}

    def test_lang_tokens_format(self):
        for code, token in WHISPER_LANG_TOKENS.items():
            assert token.startswith("<|") and token.endswith("|>"), (
                f"Token for '{code}' has unexpected format: {token}"
            )


# ──────────────────────────────────────────────────────────────────────
# WhisperASR.__init__
# ──────────────────────────────────────────────────────────────────────

class TestWhisperASRInit:
    """Verify __init__ wires up model + processor correctly."""

    def test_model_and_processor_set(self, mock_asr):
        assert mock_asr.model is not None
        assert mock_asr.processor is not None

    def test_model_set_to_eval(self, mock_asr):
        mock_asr.model.eval.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# WhisperASR.transcribe (file-based)
# ──────────────────────────────────────────────────────────────────────

class TestTranscribeFile:
    """Tests for the file-based transcribe() method."""

    def test_returns_asr_result(self, mock_asr, tmp_path):
        """transcribe() should return an ASRResult instance."""
        import scipy.io.wavfile

        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        wav = tmp_path / "test.wav"
        scipy.io.wavfile.write(str(wav), sr, silence)

        result = mock_asr.transcribe(str(wav), language="en")
        assert isinstance(result, ASRResult)

    def test_language_passthrough(self, mock_asr, tmp_path):
        """When a language is specified, it should appear in the result."""
        import scipy.io.wavfile

        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        wav = tmp_path / "test.wav"
        scipy.io.wavfile.write(str(wav), sr, silence)

        result = mock_asr.transcribe(str(wav), language="vi")
        assert result.language == "vi"

    def test_text_is_stripped(self, mock_asr, tmp_path):
        """The decoded text should be stripped of leading/trailing whitespace."""
        import scipy.io.wavfile

        # The mock processor returns " Hello world " — verify it gets stripped.
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        wav = tmp_path / "test.wav"
        scipy.io.wavfile.write(str(wav), sr, silence)

        result = mock_asr.transcribe(str(wav), language="en")
        assert result.text == "Hello world"
        assert not result.text.startswith(" ")
        assert not result.text.endswith(" ")

    def test_generate_called_with_language_kwargs(self, mock_asr, tmp_path):
        """When language is given, generate() should receive language + task."""
        import scipy.io.wavfile

        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        wav = tmp_path / "test.wav"
        scipy.io.wavfile.write(str(wav), sr, silence)

        mock_asr.transcribe(str(wav), language="zh")

        call_kwargs = mock_asr.model.generate.call_args
        assert call_kwargs.kwargs.get("language") == "zh" or \
               (len(call_kwargs.args) > 1 and False), \
               "Expected 'language' in generate kwargs"
        assert call_kwargs.kwargs.get("task") == "transcribe"

    def test_generate_without_language(self, mock_asr, tmp_path):
        """When language is None, generate() should NOT include language/task."""
        import scipy.io.wavfile

        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        wav = tmp_path / "test.wav"
        scipy.io.wavfile.write(str(wav), sr, silence)

        # Set up _detect_language to return "und" so we don't crash.
        mock_asr._detect_language = MagicMock(return_value="und")

        mock_asr.transcribe(str(wav), language=None)

        call_kwargs = mock_asr.model.generate.call_args.kwargs
        assert "language" not in call_kwargs
        assert "task" not in call_kwargs


# ──────────────────────────────────────────────────────────────────────
# WhisperASR.transcribe_array (numpy-based)
# ──────────────────────────────────────────────────────────────────────

class TestTranscribeArray:
    """Tests for the numpy-array-based transcribe_array() method."""

    def test_mono_input(self, mock_asr):
        """1-D array should be accepted and produce an ASRResult."""
        mono = np.zeros(16000, dtype=np.float32)
        result = mock_asr.transcribe_array(mono, language="en")
        assert isinstance(result, ASRResult)
        assert result.language == "en"

    def test_stereo_input_downmixed(self, mock_asr):
        """2-D (stereo) array should be mean-downmixed to mono."""
        sr = 16000
        stereo = np.stack([
            np.ones(sr, dtype=np.float32),
            np.zeros(sr, dtype=np.float32),
        ], axis=0)  # shape: (2, 16000)

        result = mock_asr.transcribe_array(stereo, language="en")
        assert isinstance(result, ASRResult)

        # Verify the processor received a 1-D array (mono).
        call_args = mock_asr.processor.call_args
        waveform_arg = call_args.args[0]
        assert waveform_arg.ndim == 1

    def test_language_none_triggers_detection(self, mock_asr):
        """When language is None, _detect_language should be called."""
        mock_asr._detect_language = MagicMock(return_value="vi")
        mono = np.zeros(16000, dtype=np.float32)

        result = mock_asr.transcribe_array(mono, language=None)

        mock_asr._detect_language.assert_called_once()
        assert result.language == "vi"


# ──────────────────────────────────────────────────────────────────────
# WhisperASR._detect_language
# ──────────────────────────────────────────────────────────────────────

class TestDetectLanguage:
    """Tests for the internal language-detection helper."""

    @pytest.mark.parametrize("lang_code,token", [
        ("vi", "<|vi|>"),
        ("en", "<|en|>"),
        ("zh", "<|zh|>"),
    ])
    def test_detects_known_languages(self, mock_asr, lang_code, token):
        """Should return the correct code when the token is in the decoded output."""
        mock_asr.processor.batch_decode.return_value = [
            f"<|startoftranscript|>{token}<|notimestamps|>Hello"
        ]
        fake_ids = MagicMock()

        detected = mock_asr._detect_language(fake_ids)
        assert detected == lang_code

    def test_undetermined_when_no_token(self, mock_asr):
        """Should return 'und' when no known language token is found."""
        mock_asr.processor.batch_decode.return_value = [
            "<|startoftranscript|><|notimestamps|>Unknown"
        ]
        fake_ids = MagicMock()

        detected = mock_asr._detect_language(fake_ids)
        assert detected == "und"

    def test_batch_decode_called_with_special_tokens(self, mock_asr):
        """_detect_language must call batch_decode with skip_special_tokens=False."""
        mock_asr.processor.batch_decode.return_value = [""]
        fake_ids = MagicMock()

        mock_asr._detect_language(fake_ids)

        mock_asr.processor.batch_decode.assert_called_with(
            fake_ids, skip_special_tokens=False
        )
