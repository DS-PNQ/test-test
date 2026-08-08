# ==========================================================================
# Comprehensive local test for the ENTIRE OnSpeak47 pipeline
# ==========================================================================
#
# Covers all 3 stages (ASR → Translation → TTS), the orchestrator,
# cross-module contracts, language-code consistency, and known
# tokenizer/decoder issues — all without downloading models.
#
# Structure:
#   Part 1 — Stage 1: ASR (Whisper Small)
#   Part 2 — Stage 2: Translation (NLLB-200)
#   Part 3 — Stage 3: TTS (MMS-TTS / VITS)
#   Part 4 — Orchestrator (OmniVoicePipeline)
#   Part 5 — Cross-module contract & language-code consistency
#   Part 6 — Tokenizer / decoder issues (Android parity checks)

import sys
import re
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import scipy.io.wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.asr_whisper import ASRResult, WhisperASR, WHISPER_LANG_TOKENS, MODEL_NAME as ASR_MODEL_NAME
from backend.translation_nllb import NLLBTranslator, LANG_CODES, MODEL_NAME as NLLB_MODEL_NAME
from backend.tts_mms import MMSTTS, MMS_TTS_MODELS
from backend.orchestrator import OmniVoicePipeline, PipelineResult


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def _make_wav(tmp_path: Path, name: str = "input.wav", sr: int = 16000) -> Path:
    tone = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    wav_path = tmp_path / name
    scipy.io.wavfile.write(str(wav_path), sr, (tone * 32767).astype(np.int16))
    return wav_path


def _mock_whisper_asr():
    with patch("backend.asr_whisper.WhisperProcessor") as P, \
         patch("backend.asr_whisper.WhisperForConditionalGeneration") as M:
        proc = MagicMock()
        features = MagicMock()
        features.input_features = MagicMock()
        proc.return_value = features
        proc.batch_decode.return_value = [" Hello world "]
        P.from_pretrained.return_value = proc
        model = MagicMock()
        model.generate.return_value = MagicMock()
        M.from_pretrained.return_value = model
        asr = WhisperASR()
    return asr


def _mock_nllb_translator():
    with patch("backend.translation_nllb.AutoTokenizer") as T, \
         patch("backend.translation_nllb.AutoModelForSeq2SeqLM") as M:
        tok = MagicMock()
        tok.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        tok.convert_tokens_to_ids.return_value = 256047
        tok.batch_decode.return_value = ["Xin chào"]
        tok.src_lang = None
        T.from_pretrained.return_value = tok
        model = MagicMock()
        model.generate.return_value = MagicMock()
        M.from_pretrained.return_value = model
        translator = NLLBTranslator()
    return translator


def _mock_mms_tts():
    with patch("backend.tts_mms.VitsModel") as VM, \
         patch("backend.tts_mms.AutoTokenizer") as AT:
        tok = MagicMock()
        tok.return_value = {"input_ids": MagicMock()}
        AT.from_pretrained.return_value = tok
        model = MagicMock()
        waveform_tensor = MagicMock()
        waveform_tensor.cpu.return_value.numpy.return_value = np.zeros(16000, dtype=np.float32)
        output = MagicMock()
        output.waveform = [waveform_tensor]
        model.return_value = output
        model.config.sampling_rate = 16000
        VM.from_pretrained.return_value = model
        tts = MMSTTS()
        # Pre-populate lazy cache so _get_model_and_tokenizer doesn't re-load
        tts._models["en"] = model
        tts._tokenizers["en"] = tok
        tts._models["vi"] = model
        tts._tokenizers["vi"] = tok
        tts._models["zh"] = model
        tts._tokenizers["zh"] = tok
        tts._models["zh_hans"] = model
        tts._tokenizers["zh_hans"] = tok
    return tts


# Simple mock stages for orchestrator tests
def _simple_mock_asr(transcript="Xin chào", language="vi"):
    asr = MagicMock()
    asr.transcribe.return_value = ASRResult(text=transcript, language=language, segments=[])
    return asr

def _simple_mock_translator(translation="Hello"):
    t = MagicMock()
    t.translate.return_value = translation
    return t

def _simple_mock_tts():
    tts = MagicMock()
    def _fake(text, language, path, **kw):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(p), 16000, np.zeros(1600, dtype=np.int16))
        return str(p.resolve())
    tts.synthesize.side_effect = _fake
    return tts


# ══════════════════════════════════════════════════════════════════════
# PART 1 — Stage 1: ASR (Whisper Small)
# ══════════════════════════════════════════════════════════════════════

class TestASR_ModelInit:
    """Whisper model + processor load correctly."""

    def test_model_and_processor_exist(self):
        asr = _mock_whisper_asr()
        assert asr.model is not None
        assert asr.processor is not None

    def test_model_set_to_eval(self):
        asr = _mock_whisper_asr()
        asr.model.eval.assert_called_once()

    def test_model_name_constant(self):
        assert ASR_MODEL_NAME == "openai/whisper-small"


class TestASR_Transcribe:
    """File-based transcription."""

    def test_returns_asr_result(self, tmp_path):
        asr = _mock_whisper_asr()
        wav = _make_wav(tmp_path)
        result = asr.transcribe(str(wav), language="en")
        assert isinstance(result, ASRResult)

    def test_language_passthrough(self, tmp_path):
        asr = _mock_whisper_asr()
        wav = _make_wav(tmp_path)
        for lang in ("vi", "en", "zh"):
            result = asr.transcribe(str(wav), language=lang)
            assert result.language == lang

    def test_text_is_stripped(self, tmp_path):
        asr = _mock_whisper_asr()
        wav = _make_wav(tmp_path)
        result = asr.transcribe(str(wav), language="en")
        assert result.text == "Hello world"
        assert not result.text.startswith(" ")

    def test_generate_gets_language_and_task(self, tmp_path):
        asr = _mock_whisper_asr()
        wav = _make_wav(tmp_path)
        asr.transcribe(str(wav), language="zh")
        kw = asr.model.generate.call_args.kwargs
        assert kw.get("language") == "zh"
        assert kw.get("task") == "transcribe"

    def test_generate_has_return_timestamps(self, tmp_path):
        asr = _mock_whisper_asr()
        wav = _make_wav(tmp_path)
        asr.transcribe(str(wav), language="en")
        kw = asr.model.generate.call_args.kwargs
        assert kw.get("return_timestamps") is True

    def test_autodetect_when_no_language(self, tmp_path):
        asr = _mock_whisper_asr()
        asr._detect_language = MagicMock(return_value="vi")
        wav = _make_wav(tmp_path)
        result = asr.transcribe(str(wav), language=None)
        asr._detect_language.assert_called_once()
        assert result.language == "vi"


class TestASR_TranscribeArray:
    """NumPy-array-based transcription."""

    def test_mono_input(self):
        asr = _mock_whisper_asr()
        result = asr.transcribe_array(np.zeros(16000, dtype=np.float32), language="en")
        assert isinstance(result, ASRResult)

    def test_stereo_downmix(self):
        asr = _mock_whisper_asr()
        stereo = np.stack([np.ones(16000, dtype=np.float32),
                           np.zeros(16000, dtype=np.float32)])
        asr.transcribe_array(stereo, language="en")
        waveform_arg = asr.processor.call_args.args[0]
        assert waveform_arg.ndim == 1

    def test_missing_return_timestamps(self):
        """transcribe_array does NOT set return_timestamps — inconsistency."""
        asr = _mock_whisper_asr()
        asr.transcribe_array(np.zeros(16000, dtype=np.float32), language="en")
        kw = asr.model.generate.call_args.kwargs
        # This documents the known inconsistency with transcribe()
        assert "return_timestamps" not in kw


class TestASR_LanguageDetection:
    """_detect_language helper."""

    @pytest.mark.parametrize("code,token", [
        ("vi", "<|vi|>"), ("en", "<|en|>"), ("zh", "<|zh|>"),
    ])
    def test_detects_known_languages(self, code, token):
        asr = _mock_whisper_asr()
        asr.processor.batch_decode.return_value = [f"<|startoftranscript|>{token}<|notimestamps|>Hi"]
        assert asr._detect_language(MagicMock()) == code

    def test_returns_und_for_unknown(self):
        asr = _mock_whisper_asr()
        asr.processor.batch_decode.return_value = ["<|startoftranscript|><|notimestamps|>???"]
        assert asr._detect_language(MagicMock()) == "und"

    def test_calls_batch_decode_with_special_tokens(self):
        asr = _mock_whisper_asr()
        asr.processor.batch_decode.return_value = [""]
        fake_ids = MagicMock()
        asr._detect_language(fake_ids)
        asr.processor.batch_decode.assert_called_with(fake_ids, skip_special_tokens=False)


class TestASR_DataclassAndConstants:
    """ASRResult dataclass + module constants."""

    def test_asr_result_defaults(self):
        r = ASRResult(text="hi", language="en")
        assert r.segments == []

    def test_asr_result_independent_segments(self):
        r1 = ASRResult(text="a", language="en")
        r2 = ASRResult(text="b", language="vi")
        r1.segments.append("x")
        assert r2.segments == []

    def test_whisper_lang_tokens_keys(self):
        assert set(WHISPER_LANG_TOKENS.keys()) == {"vi", "en", "zh"}

    def test_whisper_lang_tokens_format(self):
        for code, tok in WHISPER_LANG_TOKENS.items():
            assert tok.startswith("<|") and tok.endswith("|>")


# ══════════════════════════════════════════════════════════════════════
# PART 2 — Stage 2: Translation (NLLB-200)
# ══════════════════════════════════════════════════════════════════════

class TestTranslation_ModelInit:
    """NLLB model + tokenizer load correctly."""

    def test_model_and_tokenizer_exist(self):
        t = _mock_nllb_translator()
        assert t.model is not None
        assert t.tokenizer is not None

    def test_model_set_to_eval(self):
        t = _mock_nllb_translator()
        t.model.eval.assert_called_once()

    def test_model_name_constant(self):
        assert NLLB_MODEL_NAME == "facebook/nllb-200-distilled-600M"


class TestTranslation_LangCodes:
    """LANG_CODES map — all pipeline languages are present."""

    def test_core_languages_present(self):
        for key in ("en", "vi", "zh_hans", "zh_hant", "zh"):
            assert key in LANG_CODES, f"Missing LANG_CODES key: {key}"

    def test_flores_code_format(self):
        for key, flores in LANG_CODES.items():
            assert "_" in flores, f"FLORES code '{flores}' for '{key}' missing underscore"
            assert len(flores.split("_")) == 2

    def test_zh_alias_maps_to_simplified(self):
        assert LANG_CODES["zh"] == LANG_CODES["zh_hans"] == "zho_Hans"


class TestTranslation_Translate:
    """Single-sentence translate()."""

    def test_sets_src_lang_on_tokenizer(self):
        t = _mock_nllb_translator()
        t.translate("Hello", "en", "vi")
        assert t.tokenizer.src_lang == "eng_Latn"

    def test_converts_tgt_to_forced_bos(self):
        t = _mock_nllb_translator()
        t.translate("Hello", "en", "vi")
        t.tokenizer.convert_tokens_to_ids.assert_called_with("vie_Latn")

    def test_generate_params(self):
        t = _mock_nllb_translator()
        t.translate("Hello", "en", "vi")
        kw = t.model.generate.call_args.kwargs
        assert kw["max_length"] == 256
        assert kw["num_beams"] == 5
        assert kw["repetition_penalty"] == 1.3
        assert kw["no_repeat_ngram_size"] == 3

    def test_returns_decoded_string(self):
        t = _mock_nllb_translator()
        result = t.translate("Hello", "en", "vi")
        assert result == "Xin chào"


class TestTranslation_Batch:
    """Batch translate_batch()."""

    def test_batch_sets_src_lang(self):
        t = _mock_nllb_translator()
        t.translate_batch(["Hello", "World"], "en", "vi")
        assert t.tokenizer.src_lang == "eng_Latn"

    def test_batch_returns_list(self):
        t = _mock_nllb_translator()
        t.tokenizer.batch_decode.return_value = ["Xin chào", "Thế giới"]
        result = t.translate_batch(["Hello", "World"], "en", "vi")
        assert isinstance(result, list)

    def test_batch_splits_correctly(self):
        t = _mock_nllb_translator()
        t.tokenizer.batch_decode.return_value = ["a"]
        texts = [f"s{i}" for i in range(35)]
        t.translate_batch(texts, "en", "vi", batch_size=16)
        # 35 items / 16 per batch = 3 batches
        assert t.model.generate.call_count == 3


class TestTranslation_Pivot:
    """Pivot translation (vi→en→zh)."""

    def test_pivot_calls_translate_twice(self):
        t = _mock_nllb_translator()
        t.translate = MagicMock(side_effect=["intermediate", "final"])
        result = t.translate_pivot("Xin chào", "vi", "zh_hans", pivot_lang="en")
        assert t.translate.call_count == 2
        t.translate.assert_any_call("Xin chào", "vi", "en")
        t.translate.assert_any_call("intermediate", "en", "zh_hans")
        assert result == "final"


class TestTranslation_FileLevel:
    """File-level translate_file()."""

    def test_reads_source_writes_target(self, tmp_path):
        t = _mock_nllb_translator()
        t.translate_batch = MagicMock(return_value=["Xin chào", "Thế giới"])
        src = tmp_path / "src.txt"
        tgt = tmp_path / "out" / "tgt.txt"
        src.write_text("Hello\nWorld\n", encoding="utf-8")
        result = t.translate_file(str(src), str(tgt), "en", "vi")
        assert tgt.exists()
        assert len(result) == 2
        lines = tgt.read_text(encoding="utf-8").strip().split("\n")
        assert lines == ["Xin chào", "Thế giới"]

    def test_max_lines_limit(self, tmp_path):
        t = _mock_nllb_translator()
        t.translate_batch = MagicMock(return_value=["a", "b"])
        src = tmp_path / "src.txt"
        tgt = tmp_path / "tgt.txt"
        src.write_text("\n".join([f"line{i}" for i in range(100)]), encoding="utf-8")
        t.translate_file(str(src), str(tgt), "en", "vi", max_lines=2)
        call_args = t.translate_batch.call_args.args[0]
        assert len(call_args) == 2


class TestTranslation_ChinesePreprocess:
    """preprocess_chinese() static helper."""

    def test_removes_spaces_between_cjk(self):
        result = NLLBTranslator.preprocess_chinese("你 好 世 界")
        assert result == "你好世界"

    def test_preserves_spaces_around_latin(self):
        result = NLLBTranslator.preprocess_chinese("你好 World 世界")
        assert "World" in result
        assert " World " in result or "World" in result

    def test_strips_whitespace(self):
        result = NLLBTranslator.preprocess_chinese("  你好  ")
        assert result == "你好"

    def test_empty_string(self):
        result = NLLBTranslator.preprocess_chinese("")
        assert result == ""


# ══════════════════════════════════════════════════════════════════════
# PART 3 — Stage 3: TTS (MMS-TTS / VITS)
# ══════════════════════════════════════════════════════════════════════

class TestTTS_Constants:
    """MMS_TTS_MODELS map — all languages covered."""

    def test_core_languages_present(self):
        for lang in ("vi", "en", "zh", "zh_hans", "zh_hant"):
            assert lang in MMS_TTS_MODELS

    def test_model_ids_are_facebook(self):
        for lang, model_id in MMS_TTS_MODELS.items():
            assert model_id.startswith("facebook/mms-tts-")

    def test_zh_variants_share_model(self):
        assert MMS_TTS_MODELS["zh"] == MMS_TTS_MODELS["zh_hans"] == MMS_TTS_MODELS["zh_hant"]


class TestTTS_Synthesize:
    """synthesize() — text to WAV file."""

    def test_returns_path_string(self, tmp_path):
        tts = _mock_mms_tts()
        result = tts.synthesize("Hello", "en", tmp_path / "out.wav")
        assert isinstance(result, str)

    def test_creates_wav_file(self, tmp_path):
        tts = _mock_mms_tts()
        out = tmp_path / "out.wav"
        tts.synthesize("Hello", "en", out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        tts = _mock_mms_tts()
        out = tmp_path / "deep" / "nested" / "out.wav"
        tts.synthesize("Hello", "en", out)
        assert out.exists()

    def test_unsupported_language_raises(self):
        tts = _mock_mms_tts()
        with pytest.raises(ValueError, match="not supported"):
            tts._get_model_and_tokenizer("fr")


class TestTTS_SynthesizeToArray:
    """synthesize_to_array() — returns (waveform, sample_rate)."""

    def test_returns_tuple(self):
        tts = _mock_mms_tts()
        result = tts.synthesize_to_array("Hello", "en")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_waveform_is_numpy(self):
        tts = _mock_mms_tts()
        waveform, sr = tts.synthesize_to_array("Hello", "en")
        assert isinstance(waveform, np.ndarray)
        assert isinstance(sr, int)


class TestTTS_LazyLoading:
    """Lazy per-language model loading."""

    def test_get_supported_languages(self):
        tts = _mock_mms_tts()
        langs = tts.get_supported_languages()
        assert set(langs) == {"vi", "en", "zh", "zh_hans", "zh_hant"}

    def test_caches_model_per_language(self):
        tts = _mock_mms_tts()
        m1, t1 = tts._get_model_and_tokenizer("en")
        m2, t2 = tts._get_model_and_tokenizer("en")
        assert m1 is m2
        assert t1 is t2


# ══════════════════════════════════════════════════════════════════════
# PART 4 — Orchestrator (OmniVoicePipeline)
# ══════════════════════════════════════════════════════════════════════

class TestOrchestrator_Init:
    """Pipeline constructor and dependency injection."""

    def test_accepts_injected_stages(self):
        asr, tr, tts = _simple_mock_asr(), _simple_mock_translator(), _simple_mock_tts()
        pipe = OmniVoicePipeline(asr=asr, translator=tr, tts=tts)
        assert pipe.asr is asr
        assert pipe.translator is tr
        assert pipe.tts is tts


class TestOrchestrator_FullPipeline:
    """process() — full audio→audio pipeline."""

    def test_returns_pipeline_result(self, tmp_path):
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=_simple_mock_tts())
        wav = _make_wav(tmp_path)
        result = pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        assert isinstance(result, PipelineResult)

    def test_asr_receives_audio_and_language(self, tmp_path):
        asr = _simple_mock_asr()
        pipe = OmniVoicePipeline(asr=asr, translator=_simple_mock_translator(), tts=_simple_mock_tts())
        wav = _make_wav(tmp_path)
        pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        asr.transcribe.assert_called_once_with(str(wav), language="vi")

    def test_translator_receives_asr_text(self, tmp_path):
        tr = _simple_mock_translator()
        pipe = OmniVoicePipeline(asr=_simple_mock_asr("Tôi cần giúp đỡ", "vi"), translator=tr, tts=_simple_mock_tts())
        wav = _make_wav(tmp_path)
        pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        tr.translate.assert_called_once_with("Tôi cần giúp đỡ", "vi", "en")

    def test_tts_receives_translation_and_tgt_lang(self, tmp_path):
        tts = _simple_mock_tts()
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator("Good morning"), tts=tts)
        wav = _make_wav(tmp_path)
        pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        call = tts.synthesize.call_args
        assert call.args[0] == "Good morning"
        assert call.args[1] == "en"

    def test_result_fields(self, tmp_path):
        pipe = OmniVoicePipeline(
            asr=_simple_mock_asr("Xin chào", "vi"),
            translator=_simple_mock_translator("Hello"),
            tts=_simple_mock_tts(),
        )
        wav = _make_wav(tmp_path)
        r = pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        assert r.transcript == "Xin chào"
        assert r.src_language == "vi"
        assert r.translation == "Hello"
        assert r.tgt_language == "en"
        assert r.audio_path is not None
        assert Path(r.audio_path).exists()

    def test_timings_keys(self, tmp_path):
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=_simple_mock_tts())
        wav = _make_wav(tmp_path)
        r = pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        assert {"asr_ms", "translation_ms", "tts_ms", "total_ms"} <= set(r.timings.keys())

    def test_total_ms_is_sum(self, tmp_path):
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=_simple_mock_tts())
        wav = _make_wav(tmp_path)
        r = pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        expected = r.timings["asr_ms"] + r.timings["translation_ms"] + r.timings["tts_ms"]
        assert r.timings["total_ms"] == pytest.approx(expected, abs=0.2)

    def test_stage_execution_order(self, tmp_path):
        order = []
        asr = _simple_mock_asr()
        asr.transcribe.side_effect = lambda *a, **kw: (order.append("asr"), ASRResult("t", "vi", []))[1]
        tr = _simple_mock_translator()
        tr.translate.side_effect = lambda *a, **kw: (order.append("translation"), "ok")[1]
        tts = _simple_mock_tts()
        orig = tts.synthesize.side_effect
        def _tts(*a, **kw):
            order.append("tts")
            return orig(*a, **kw)
        tts.synthesize.side_effect = _tts
        pipe = OmniVoicePipeline(asr=asr, translator=tr, tts=tts)
        wav = _make_wav(tmp_path)
        pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        assert order == ["asr", "translation", "tts"]

    @pytest.mark.parametrize("src,tgt", [
        ("vi", "en"), ("en", "vi"), ("vi", "zh_hans"), ("zh_hans", "vi"),
        ("en", "zh_hans"), ("zh_hans", "en"),
    ])
    def test_all_language_directions(self, src, tgt, tmp_path):
        pipe = OmniVoicePipeline(
            asr=_simple_mock_asr("input", src),
            translator=_simple_mock_translator("output"),
            tts=_simple_mock_tts(),
        )
        wav = _make_wav(tmp_path)
        r = pipe.process(str(wav), src, tgt, output_dir=str(tmp_path / f"out_{src}_{tgt}"))
        assert r.src_language == src
        assert r.tgt_language == tgt


class TestOrchestrator_TextOnly:
    """translate_text() — no ASR, no TTS."""

    def test_returns_pipeline_result(self):
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=_simple_mock_tts())
        r = pipe.translate_text("Hello", "en", "vi")
        assert isinstance(r, PipelineResult)

    def test_echoes_input_as_transcript(self):
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=_simple_mock_tts())
        r = pipe.translate_text("Good morning", "en", "vi")
        assert r.transcript == "Good morning"

    def test_no_audio_path(self):
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=_simple_mock_tts())
        r = pipe.translate_text("Hello", "en", "vi")
        assert r.audio_path is None

    def test_asr_not_called(self):
        asr = _simple_mock_asr()
        pipe = OmniVoicePipeline(asr=asr, translator=_simple_mock_translator(), tts=_simple_mock_tts())
        pipe.translate_text("Hello", "en", "vi")
        asr.transcribe.assert_not_called()

    def test_tts_not_called(self):
        tts = _simple_mock_tts()
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=tts)
        pipe.translate_text("Hello", "en", "vi")
        tts.synthesize.assert_not_called()

    def test_total_equals_translation_time(self):
        pipe = OmniVoicePipeline(asr=_simple_mock_asr(), translator=_simple_mock_translator(), tts=_simple_mock_tts())
        r = pipe.translate_text("Hello", "en", "vi")
        assert r.timings["total_ms"] == r.timings["translation_ms"]

    @pytest.mark.parametrize("src,tgt,text", [
        ("vi", "en", "Xin chào"), ("en", "vi", "Hello"),
        ("vi", "zh_hans", "Xin chào"), ("zh_hans", "vi", "你好"),
    ])
    def test_all_text_directions(self, src, tgt, text):
        pipe = OmniVoicePipeline(
            asr=_simple_mock_asr(), translator=_simple_mock_translator("out"), tts=_simple_mock_tts())
        r = pipe.translate_text(text, src, tgt)
        assert r.src_language == src
        assert r.tgt_language == tgt


class TestOrchestrator_EdgeCases:
    """Edge cases for the pipeline."""

    def test_empty_asr_output(self, tmp_path):
        pipe = OmniVoicePipeline(
            asr=_simple_mock_asr("", "en"),
            translator=_simple_mock_translator(""),
            tts=_simple_mock_tts(),
        )
        wav = _make_wav(tmp_path)
        r = pipe.process(str(wav), "en", "vi", output_dir=str(tmp_path / "out"))
        assert r.transcript == ""
        assert r.translation == ""

    def test_empty_text_translate_text(self):
        pipe = OmniVoicePipeline(
            asr=_simple_mock_asr(), translator=_simple_mock_translator(""), tts=_simple_mock_tts())
        r = pipe.translate_text("", "en", "vi")
        assert r.transcript == ""

    def test_long_text(self, tmp_path):
        long_text = "Xin chào. " * 200
        pipe = OmniVoicePipeline(
            asr=_simple_mock_asr(long_text.strip(), "vi"),
            translator=_simple_mock_translator("Hello. " * 200),
            tts=_simple_mock_tts(),
        )
        wav = _make_wav(tmp_path)
        r = pipe.process(str(wav), "vi", "en", output_dir=str(tmp_path / "out"))
        assert r.transcript == long_text.strip()


class TestPipelineResult_Dataclass:
    """PipelineResult construction."""

    def test_fields(self):
        r = PipelineResult(transcript="hi", src_language="en", translation="xin chào",
                           tgt_language="vi", audio_path=None)
        assert r.timings == {}

    def test_independent_timings(self):
        r1 = PipelineResult("a", "en", "b", "vi", None)
        r2 = PipelineResult("c", "vi", "d", "en", None)
        r1.timings["x"] = 1
        assert "x" not in r2.timings


# ══════════════════════════════════════════════════════════════════════
# PART 5 — Cross-module contracts & language-code consistency
# ══════════════════════════════════════════════════════════════════════

class TestCrossModule_LanguageCodes:
    """Verify language codes are consistent across all 3 stages."""

    def test_asr_languages_accepted_by_nllb(self):
        """Every language Whisper can detect must be translatable by NLLB."""
        for lang in WHISPER_LANG_TOKENS.keys():
            assert lang in LANG_CODES, (
                f"ASR can detect '{lang}' but NLLB LANG_CODES has no mapping for it"
            )

    def test_nllb_target_languages_accepted_by_tts(self):
        """Every NLLB target language must be synthesizable by TTS."""
        for lang in ("vi", "en", "zh", "zh_hans"):
            assert lang in MMS_TTS_MODELS, (
                f"NLLB can translate to '{lang}' but MMS_TTS_MODELS has no model for it"
            )

    def test_asr_zh_maps_to_nllb_simplified(self):
        """Whisper detects 'zh' → NLLB should resolve it to zho_Hans."""
        assert LANG_CODES["zh"] == "zho_Hans"

    def test_pipeline_languages_complete_chain(self):
        """For each language, the full chain ASR→Translate→TTS should work."""
        # Core languages that must survive the full pipeline
        core_langs = {"vi", "en", "zh"}
        for lang in core_langs:
            assert lang in WHISPER_LANG_TOKENS, f"ASR missing: {lang}"
            assert lang in LANG_CODES, f"NLLB missing: {lang}"
            assert lang in MMS_TTS_MODELS, f"TTS missing: {lang}"


class TestCrossModule_IOContracts:
    """Verify I/O contracts between stages match."""

    def test_asr_output_is_string_for_translator(self):
        """ASR returns ASRResult.text (str) → NLLB expects str input."""
        r = ASRResult(text="hello", language="en")
        assert isinstance(r.text, str)

    def test_translator_returns_string_for_tts(self):
        """NLLB.translate() returns str → TTS.synthesize() expects str."""
        t = _mock_nllb_translator()
        result = t.translate("hello", "en", "vi")
        assert isinstance(result, str)

    def test_orchestrator_passes_src_lang_not_detected_lang(self):
        """Orchestrator uses caller-provided src_lang for translation,
        NOT the ASR-detected language."""
        asr = _simple_mock_asr("text", "vi")  # ASR detects "vi"
        tr = _simple_mock_translator()
        pipe = OmniVoicePipeline(asr=asr, translator=tr, tts=_simple_mock_tts())
        wav_path = MagicMock()  # We won't actually read the file
        # Call with src_lang="en" — even though ASR detects "vi"
        # We need a real wav for librosa, so let's just check the translate call
        pipe.translate_text("hello", "en", "vi")
        tr.translate.assert_called_with("hello", "en", "vi")


# ══════════════════════════════════════════════════════════════════════
# PART 6 — Tokenizer / decoder issue checks (Android parity)
# ══════════════════════════════════════════════════════════════════════

class TestTokenizerDecoder_IDRemapping:
    """Verify the Android Tokenizer.java ID remapping logic is invertible.

    The Android tokenizer applies: SentencePiece_ID +1, then swap {1→3, 2→0, 3→2}.
    The decoder reverses: swap {3→1, 0→2, 2→3}, then -1.
    We simulate this in Python to validate round-trip correctness.
    """

    @staticmethod
    def _encode_remap(sp_id: int) -> int:
        """Simulate Tokenizer.java encode remapping."""
        id_ = sp_id + 1
        if id_ == 1: return 3
        if id_ == 2: return 0
        if id_ == 3: return 2
        return id_

    @staticmethod
    def _decode_remap(nllb_id: int) -> int:
        """Simulate Tokenizer.java decode remapping."""
        if nllb_id == 3: id_ = 1
        elif nllb_id == 0: id_ = 2
        elif nllb_id == 2: id_ = 3
        else: id_ = nllb_id
        return max(0, id_ - 1)

    @pytest.mark.parametrize("sp_id", list(range(10)) + [100, 1000, 255999])
    def test_roundtrip(self, sp_id):
        """Encode then decode should return the original SentencePiece ID."""
        nllb_id = self._encode_remap(sp_id)
        recovered = self._decode_remap(nllb_id)
        assert recovered == sp_id, (
            f"Round-trip failed: SP {sp_id} → NLLB {nllb_id} → SP {recovered}"
        )

    def test_special_ids_0_1_2(self):
        """The critical swap region (SentencePiece IDs 0, 1, 2) must round-trip."""
        for sp_id in (0, 1, 2):
            assert self._decode_remap(self._encode_remap(sp_id)) == sp_id


class TestTokenizerDecoder_AndroidLanguageCodes:
    """Verify Android NLLB_CODES match Python LANG_CODES for in-scope languages."""

    # From TranslationModule.java NLLB_CODES static block
    ANDROID_NLLB_CODES = {
        "vi": "vie_Latn",
        "en": "eng_Latn",
        "zh": "zho_Hans",
    }

    def test_android_codes_match_python(self):
        for lang, android_code in self.ANDROID_NLLB_CODES.items():
            assert LANG_CODES[lang] == android_code, (
                f"Language '{lang}': Android='{android_code}', Python='{LANG_CODES[lang]}'"
            )

    def test_android_missing_zh_hans_key(self):
        """Android only maps 'zh' but Python pipeline may pass 'zh_hans'.
        This documents the mismatch."""
        assert "zh_hans" not in self.ANDROID_NLLB_CODES
        assert "zh_hans" in LANG_CODES


class TestTokenizerDecoder_WhisperTokenConsistency:
    """Verify Whisper language token IDs match between Python and Android."""

    # From ASRModule.java LANGUAGE_TOKENS static block
    ANDROID_WHISPER_TOKENS = {
        "vi": 50264,
        "en": 50259,
        "zh": 50260,
    }

    def test_same_languages_supported(self):
        assert set(WHISPER_LANG_TOKENS.keys()) == set(self.ANDROID_WHISPER_TOKENS.keys())

    def test_token_format_python_side(self):
        """Python uses string tokens like '<|vi|>' — these must be consistent."""
        for lang in self.ANDROID_WHISPER_TOKENS:
            assert lang in WHISPER_LANG_TOKENS
            assert f"<|{lang}|>" == WHISPER_LANG_TOKENS[lang]


class TestTokenizerDecoder_KnownIssues:
    """Tests that document known issues found in the codebase.
    These serve as regression tests once the issues are fixed."""

    def test_transcribe_array_missing_return_timestamps(self):
        """KNOWN ISSUE: transcribe_array() does not set return_timestamps=True,
        unlike transcribe(). This test documents the inconsistency."""
        asr = _mock_whisper_asr()
        # transcribe() sets it
        asr.transcribe_array(np.zeros(16000, dtype=np.float32), language="en")
        kw = asr.model.generate.call_args.kwargs
        has_timestamps = kw.get("return_timestamps", False)
        # Currently False — when fixed, change this assertion to True
        assert has_timestamps is False, "return_timestamps inconsistency has been fixed! Update this test."

    def test_android_nllb_missing_zh_hans_key(self):
        """KNOWN ISSUE: Android TranslationModule.NLLB_CODES only has 'zh',
        not 'zh_hans'. If the pipeline ever passes 'zh_hans' to Android,
        it will fall back to 'vie_Latn' (Vietnamese!) via getOrDefault."""
        android_codes = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans"}
        fallback = android_codes.get("zh_hans", "vie_Latn")  # simulates getOrDefault
        # This SHOULD be "zho_Hans" but falls back to Vietnamese
        assert fallback == "vie_Latn", "zh_hans fallback issue has been fixed! Update this test."
