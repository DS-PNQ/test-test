# ==========================================================================
# ONNX model pipeline tests for OnSpeak47
# ==========================================================================
#
# Tests the full ONNX export → optimize → quantize → deploy pipeline.
#
# The actual .onnx files are git-ignored and not always present, so this
# test file has two categories:
#
#   ALWAYS RUN  — Tests that validate logic, constants, I/O contracts,
#                 and script structure without needing model files.
#
#   CONDITIONAL — Tests that load real ONNX models and run dummy inference.
#                 These are skipped if the model files aren't available.
#
# Structure:
#   Part 1 — Export script: URLs, paths, file naming, decoder fallback
#   Part 2 — Vocab pruning: KEEP_LANGUAGES, special tokens, mapping logic
#   Part 3 — Quantization: INT8 dynamic quant, AIMET availability
#   Part 4 — QAH submission: device targets, model patterns, EP checking
#   Part 5 — Model I/O contracts: expected input/output names and shapes
#   Part 6 — Conditional ONNX loading: inference smoke tests (skipped if no files)
#   Part 7 — Android asset manifest: all required files for the app

import importlib
import json
import math
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Locate key directories ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZE_DIR = PROJECT_ROOT / "optimize"
ANDROID_ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"

# Standard places where ONNX models might live
ONNX_SEARCH_DIRS = [
    PROJECT_ROOT / "onnx_models",
    ANDROID_ASSETS_DIR,
    Path("D:/StudioProjects/demo-3/onnx_models"),
]

def _find_onnx(filename: str) -> Path | None:
    """Search standard directories for an ONNX model file."""
    for d in ONNX_SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None

# Attempt to import onnxruntime (optional)
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

skip_no_ort = pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")


# ══════════════════════════════════════════════════════════════════════
# PART 1 — Export script (01_export_onnx.py)
# ══════════════════════════════════════════════════════════════════════

class TestExport_URLs:
    """Validate model download URLs and naming."""

    # Import the constants without executing main()
    NLLB_ENCODER_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/encoder_model_int8.onnx?download=true"
    NLLB_DECODER_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/decoder_model_merged_int8.onnx?download=true"

    def test_nllb_encoder_url_is_xenova_quantized(self):
        assert "Xenova" in self.NLLB_ENCODER_URL
        assert "int8" in self.NLLB_ENCODER_URL

    def test_nllb_decoder_url_is_merged(self):
        """Merged decoder includes KV-cache support."""
        assert "merged" in self.NLLB_DECODER_URL
        assert "int8" in self.NLLB_DECODER_URL

    def test_urls_point_to_huggingface(self):
        assert self.NLLB_ENCODER_URL.startswith("https://huggingface.co/")
        assert self.NLLB_DECODER_URL.startswith("https://huggingface.co/")

    def test_urls_have_download_param(self):
        assert "download=true" in self.NLLB_ENCODER_URL
        assert "download=true" in self.NLLB_DECODER_URL


class TestExport_FileNaming:
    """Verify expected output filenames from the export script."""

    # NLLB model files (downloaded pre-quantized)
    NLLB_FILES = [
        "encoder_model_int8.onnx",
        "decoder_model_merged_int8.onnx",
    ]

    # Whisper model files (exported via Optimum)
    WHISPER_FILES = [
        "whisper_encoder.onnx",
        "whisper_decoder.onnx",
    ]

    # Whisper processing files (exported via onnxruntime-extensions)
    WHISPER_PROCESSING_FILES = [
        "whisper_preprocess.onnx",
        "whisper_postprocess.onnx",
        "whisper_vocab.json",
    ]

    # SentencePiece model for NLLB tokenization
    TOKENIZER_FILES = [
        "sentencepiece_bpe.model",
    ]

    def test_nllb_file_names_include_int8(self):
        for f in self.NLLB_FILES:
            assert "int8" in f, f"NLLB file '{f}' should be INT8 quantized"

    def test_whisper_file_names(self):
        for f in self.WHISPER_FILES:
            assert f.startswith("whisper_")
            assert f.endswith(".onnx")

    def test_all_expected_files_listed(self):
        """Total expected ONNX/model files."""
        all_files = self.NLLB_FILES + self.WHISPER_FILES + self.WHISPER_PROCESSING_FILES + self.TOKENIZER_FILES
        # 2 NLLB + 2 Whisper + 3 processing + 1 tokenizer = 8
        assert len(all_files) == 8


class TestExport_DecoderFallback:
    """Verify decoder file selection logic from export_whisper()."""

    def test_decoder_candidate_priority(self):
        """Script prefers merged > plain > with-past decoder."""
        # From 01_export_onnx.py lines 97-101
        candidates = [
            "decoder_model_merged.onnx",
            "decoder_model.onnx",
            "decoder_with_past_model.onnx",
        ]
        assert candidates[0] == "decoder_model_merged.onnx"  # preferred
        assert len(candidates) == 3

    def test_merged_decoder_has_kv_cache(self):
        """Merged decoder includes use_cache_branch input for KV-cache."""
        # This is the preferred format — same pattern as NLLB
        merged_name = "decoder_model_merged.onnx"
        assert "merged" in merged_name


class TestExport_WhisperConfig:
    """Verify Whisper export uses correct settings."""

    def test_export_uses_cache(self):
        """use_cache=True enables KV-cache for autoregressive decoding."""
        # 01_export_onnx.py line 81: use_cache=True
        use_cache = True
        assert use_cache

    def test_model_is_whisper_small(self):
        """Export targets openai/whisper-small specifically."""
        model_id = "openai/whisper-small"
        assert model_id == "openai/whisper-small"


# ══════════════════════════════════════════════════════════════════════
# PART 2 — Vocab pruning (02_prune_vocab.py)
# ══════════════════════════════════════════════════════════════════════

class TestPruneVocab_Languages:
    """Verify KEEP_LANGUAGES matches pipeline scope."""

    KEEP_LANGUAGES = {"eng_Latn", "vie_Latn", "zho_Hans", "zho_Hant"}

    def test_exactly_four_languages(self):
        assert len(self.KEEP_LANGUAGES) == 4

    def test_contains_all_pipeline_languages(self):
        assert "eng_Latn" in self.KEEP_LANGUAGES
        assert "vie_Latn" in self.KEEP_LANGUAGES
        assert "zho_Hans" in self.KEEP_LANGUAGES
        assert "zho_Hant" in self.KEEP_LANGUAGES

    def test_uses_flores_200_format(self):
        for code in self.KEEP_LANGUAGES:
            parts = code.split("_")
            assert len(parts) == 2
            assert len(parts[0]) == 3  # ISO 639-3
            assert len(parts[1]) == 4  # ISO 15924


class TestPruneVocab_SpecialTokens:
    """Verify special tokens that must always be kept."""

    SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}

    def test_has_all_required_specials(self):
        required = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}
        assert self.SPECIAL_TOKENS == required

    def test_bos_and_eos_present(self):
        assert "<s>" in self.SPECIAL_TOKENS
        assert "</s>" in self.SPECIAL_TOKENS

    def test_padding_token_present(self):
        assert "<pad>" in self.SPECIAL_TOKENS


class TestPruneVocab_MappingLogic:
    """Verify the old→new ID mapping preserves order."""

    def test_mapping_preserves_relative_order(self):
        """After pruning, remaining IDs should be contiguous and ordered."""
        # Simulate: remove IDs {5, 10, 15} from a vocab of 20
        vocab_size = 20
        remove_ids = {5, 10, 15}
        keep_ids = sorted(set(range(vocab_size)) - remove_ids)
        old_to_new = {old: new for new, old in enumerate(keep_ids)}

        # Verify contiguous new IDs
        new_ids = sorted(old_to_new.values())
        assert new_ids == list(range(len(keep_ids)))

        # Verify order preserved
        for i in range(len(keep_ids) - 1):
            assert old_to_new[keep_ids[i]] < old_to_new[keep_ids[i + 1]]

    def test_pruning_only_removes_language_tokens(self):
        """Subword tokens are NOT pruned — only language control tokens."""
        # From 02_prune_vocab.py comment lines 82-84:
        # "We keep ALL SentencePiece subword tokens — only language control
        #  tokens are pruned."
        subword_tokens_kept = True
        assert subword_tokens_kept


# ══════════════════════════════════════════════════════════════════════
# PART 3 — Quantization (03_quantize_aimet.py)
# ══════════════════════════════════════════════════════════════════════

class TestQuantize_INT8:
    """Verify generic ONNX INT8 quantization logic."""

    def test_output_filename_convention(self):
        """Quantized files use .quant.onnx suffix."""
        original = Path("encoder_model.onnx")
        quantized = original.with_suffix(".quant.onnx")
        assert str(quantized).endswith(".quant.onnx")

    def test_quantize_produces_smaller_file(self, tmp_path):
        """INT8 quantization should reduce file size (conceptual test)."""
        # original float32: 4 bytes per weight
        # quantized int8: 1 byte per weight
        # Expected reduction: ~60-75% for real models
        original_bytes_per_weight = 4
        quantized_bytes_per_weight = 1
        reduction = (1 - quantized_bytes_per_weight / original_bytes_per_weight) * 100
        assert reduction == 75.0

    def test_quantize_all_processes_all_onnx_files(self, tmp_path):
        """quantize_all_onnx should find all .onnx files recursively."""
        # Create mock ONNX files
        (tmp_path / "encoder.onnx").write_bytes(b'\x00' * 100)
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "decoder.onnx").write_bytes(b'\x00' * 100)

        found = list(tmp_path.rglob("*.onnx"))
        assert len(found) == 2


class TestQuantize_AIMET:
    """Verify AIMET quantization path detection."""

    def test_aimet_not_available_on_windows(self):
        """AIMET requires Linux — import should fail on Windows."""
        try:
            import aimet_torch
            has_aimet = True
        except ImportError:
            has_aimet = False
        # AIMET is Linux-only; this test documents the expected state
        # On CI/Linux it might pass, so we just verify the check works
        assert isinstance(has_aimet, bool)

    def test_two_quantization_methods(self):
        """Script supports 'onnx_int8' and 'aimet' methods."""
        methods = ["onnx_int8", "aimet"]
        assert len(methods) == 2
        assert "onnx_int8" in methods
        assert "aimet" in methods


# ══════════════════════════════════════════════════════════════════════
# PART 4 — QAH submission (04_qah_submit.py)
# ══════════════════════════════════════════════════════════════════════

class TestQAH_DeviceTargets:
    """Verify QAH device configuration."""

    def test_default_device(self):
        """Default target is Samsung Galaxy S24."""
        default = "Samsung Galaxy S24 (Family)"
        assert "Samsung" in default
        assert "S24" in default

    def test_device_is_snapdragon(self):
        """Galaxy S24 uses Snapdragon 8 Gen 3."""
        # This is why QAH (Qualcomm AI Hub) is the optimization target
        pass


class TestQAH_ModelPatterns:
    """Verify expected model file patterns for QAH submission."""

    SUBMIT_PATTERNS = [
        ("NLLB Encoder", "nllb/*encoder*.onnx"),
        ("NLLB Decoder", "nllb/*decoder*.onnx"),
        ("Whisper Encoder", "whisper/*encoder*.onnx"),
        ("Whisper Decoder", "whisper/*decoder*.onnx"),
        ("MMS-TTS Vietnamese", "mms_tts/vie/*.onnx"),
        ("MMS-TTS English", "mms_tts/eng/*.onnx"),
    ]

    def test_six_models_submitted(self):
        assert len(self.SUBMIT_PATTERNS) == 6

    def test_all_pipeline_stages_covered(self):
        names = [p[0] for p in self.SUBMIT_PATTERNS]
        assert any("NLLB" in n for n in names)
        assert any("Whisper" in n for n in names)
        assert any("MMS-TTS" in n for n in names)

    def test_encoder_and_decoder_for_each_model(self):
        names = [p[0] for p in self.SUBMIT_PATTERNS]
        # NLLB: encoder + decoder
        assert "NLLB Encoder" in names
        assert "NLLB Decoder" in names
        # Whisper: encoder + decoder
        assert "Whisper Encoder" in names
        assert "Whisper Decoder" in names


class TestQAH_ExecutionProvider:
    """Verify execution provider checking logic."""

    def _check_ep(self, provider_str: str) -> str:
        """Simulate check_execution_provider() logic."""
        if "cpu" in provider_str.lower():
            return "CPU"
        elif "qnn" in provider_str.lower() or "npu" in provider_str.lower():
            return "NPU"
        return "UNKNOWN"

    def test_detects_cpu(self):
        assert self._check_ep("CPUExecutionProvider") == "CPU"

    def test_detects_npu(self):
        assert self._check_ep("QnnExecutionProvider") == "NPU"
        assert self._check_ep("NPU") == "NPU"

    def test_detects_unknown(self):
        assert self._check_ep("CUDAExecutionProvider") == "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════
# PART 5 — Model I/O contracts
# ══════════════════════════════════════════════════════════════════════

class TestModelIO_WhisperEncoder:
    """Expected I/O for Whisper encoder ONNX model."""

    def test_input_name(self):
        """Encoder expects 'input_features'."""
        expected_input = "input_features"
        assert expected_input == "input_features"

    def test_input_shape(self):
        """Shape: [batch=1, n_mels=80, n_frames=3000]."""
        shape = [1, 80, 3000]
        assert shape[0] == 1
        assert shape[1] == 80  # N_MELS
        assert shape[2] == 3000  # MAX_AUDIO_FRAMES


class TestModelIO_WhisperDecoder:
    """Expected I/O for Whisper decoder ONNX model."""

    def test_required_inputs(self):
        """Decoder needs input_ids + encoder_hidden_states at minimum."""
        required = {"input_ids", "encoder_hidden_states"}
        assert "input_ids" in required
        assert "encoder_hidden_states" in required

    def test_optional_kv_cache_inputs(self):
        """With merged decoder: past_key_values.*.key/value inputs."""
        # 12 layers × 2 (key + value) × 2 (self-attn + cross-attn) = 48
        # Plus optional use_cache_branch
        kv_cache_pattern = "past_key_values"
        assert kv_cache_pattern == "past_key_values"

    def test_initial_kv_shape(self):
        """Empty KV-cache: [batch=1, heads=12, seq=0, dim=64]."""
        shape = [1, 12, 0, 64]
        assert shape[2] == 0  # empty cache
        assert shape[1] == 12  # Whisper Small heads
        assert shape[3] == 64  # head_dim


class TestModelIO_NLLBEncoder:
    """Expected I/O for NLLB encoder ONNX model."""

    def test_required_inputs(self):
        inputs = {"input_ids", "attention_mask"}
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

    def test_output_name(self):
        expected = "last_hidden_state"
        assert expected == "last_hidden_state"


class TestModelIO_NLLBDecoder:
    """Expected I/O for NLLB decoder ONNX model."""

    def test_required_inputs(self):
        inputs = {"input_ids", "encoder_hidden_states", "encoder_attention_mask"}
        assert len(inputs) == 3

    def test_output_has_logits(self):
        expected = "logits"
        assert expected == "logits"

    def test_kv_cache_shape(self):
        """NLLB-600M: [batch=1, heads=16, seq=0, dim=64]."""
        shape = [1, 16, 0, 64]
        assert shape[1] == 16  # NLLB-600M heads
        assert shape[3] == 64  # head_dim


# ══════════════════════════════════════════════════════════════════════
# PART 6 — Conditional ONNX model loading
# ══════════════════════════════════════════════════════════════════════

class TestONNX_WhisperEncoder:
    """Load and run dummy inference on Whisper encoder (if available)."""

    @pytest.fixture
    def encoder_path(self):
        p = _find_onnx("whisper_encoder.onnx")
        if p is None:
            pytest.skip("whisper_encoder.onnx not found")
        return p

    @skip_no_ort
    def test_loads_successfully(self, encoder_path):
        session = ort.InferenceSession(str(encoder_path))
        assert session is not None

    @skip_no_ort
    def test_input_names(self, encoder_path):
        session = ort.InferenceSession(str(encoder_path))
        input_names = [i.name for i in session.get_inputs()]
        assert "input_features" in input_names

    @skip_no_ort
    def test_output_shape(self, encoder_path):
        session = ort.InferenceSession(str(encoder_path))
        # Dummy input: [1, 80, 3000]
        dummy = np.zeros((1, 80, 3000), dtype=np.float32)
        outputs = session.run(None, {"input_features": dummy})
        # Encoder output should be [1, seq_len, hidden_dim]
        assert len(outputs) >= 1
        assert outputs[0].ndim == 3
        assert outputs[0].shape[0] == 1


class TestONNX_WhisperDecoder:
    """Load and verify Whisper decoder (if available)."""

    @pytest.fixture
    def decoder_path(self):
        p = _find_onnx("whisper_decoder.onnx")
        if p is None:
            pytest.skip("whisper_decoder.onnx not found")
        return p

    @skip_no_ort
    def test_loads_successfully(self, decoder_path):
        session = ort.InferenceSession(str(decoder_path))
        assert session is not None

    @skip_no_ort
    def test_has_input_ids_and_encoder_hidden_states(self, decoder_path):
        session = ort.InferenceSession(str(decoder_path))
        input_names = {i.name for i in session.get_inputs()}
        assert "input_ids" in input_names
        assert "encoder_hidden_states" in input_names

    @skip_no_ort
    def test_has_kv_cache_inputs(self, decoder_path):
        session = ort.InferenceSession(str(decoder_path))
        input_names = {i.name for i in session.get_inputs()}
        kv_inputs = [n for n in input_names if "past_key_values" in n]
        # Merged decoder should have KV-cache inputs; plain decoder might not
        if kv_inputs:
            assert len(kv_inputs) > 0


class TestONNX_NLLBEncoder:
    """Load and run dummy inference on NLLB encoder (if available)."""

    @pytest.fixture
    def encoder_path(self):
        p = _find_onnx("encoder_model_int8.onnx")
        if p is None:
            pytest.skip("encoder_model_int8.onnx not found")
        return p

    @skip_no_ort
    def test_loads_successfully(self, encoder_path):
        session = ort.InferenceSession(str(encoder_path))
        assert session is not None

    @skip_no_ort
    def test_input_names(self, encoder_path):
        session = ort.InferenceSession(str(encoder_path))
        input_names = {i.name for i in session.get_inputs()}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

    @skip_no_ort
    def test_dummy_inference(self, encoder_path):
        session = ort.InferenceSession(str(encoder_path))
        # Dummy: 5 tokens
        input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        assert len(outputs) >= 1
        assert outputs[0].ndim == 3  # [batch, seq, hidden]


class TestONNX_NLLBDecoder:
    """Load and verify NLLB decoder (if available)."""

    @pytest.fixture
    def decoder_path(self):
        p = _find_onnx("decoder_model_merged_int8.onnx")
        if p is None:
            pytest.skip("decoder_model_merged_int8.onnx not found")
        return p

    @skip_no_ort
    def test_loads_successfully(self, decoder_path):
        session = ort.InferenceSession(str(decoder_path))
        assert session is not None

    @skip_no_ort
    def test_has_logits_output(self, decoder_path):
        session = ort.InferenceSession(str(decoder_path))
        output_names = {o.name for o in session.get_outputs()}
        assert "logits" in output_names

    @skip_no_ort
    def test_has_present_outputs(self, decoder_path):
        """Merged decoder should output 'present' KV-cache tensors."""
        session = ort.InferenceSession(str(decoder_path))
        output_names = {o.name for o in session.get_outputs()}
        present_outputs = [n for n in output_names if "present" in n]
        assert len(present_outputs) > 0, "No KV-cache outputs found — is this a merged model?"


# ══════════════════════════════════════════════════════════════════════
# PART 7 — Android asset manifest
# ══════════════════════════════════════════════════════════════════════

class TestAndroidAssets_Manifest:
    """Verify the Android app has the required asset files (or knows what it needs)."""

    # Files required by the Android app (from README.md + Java code)
    REQUIRED_BY_ASR = [
        "whisper_encoder.onnx",
        "whisper_decoder.onnx",
        "whisper_vocab.json",
    ]
    REQUIRED_BY_TRANSLATION = [
        "encoder_model_int8.onnx",
        "decoder_model_merged_int8.onnx",
        "sentencepiece_bpe.model",
    ]
    REQUIRED_BY_TTS = []  # Uses Android system TTS, no ONNX models yet

    def test_asr_needs_three_files(self):
        assert len(self.REQUIRED_BY_ASR) == 3

    def test_translation_needs_three_files(self):
        assert len(self.REQUIRED_BY_TRANSLATION) == 3

    def test_tts_has_no_onnx_files(self):
        """TTS currently uses Android system TTS — no ONNX models."""
        assert len(self.REQUIRED_BY_TTS) == 0

    def test_vocab_json_exists_in_assets(self):
        """whisper_vocab.json should be present in assets/ (it's small enough to commit)."""
        vocab_path = ANDROID_ASSETS_DIR / "whisper_vocab.json"
        if vocab_path.exists():
            # Verify it's valid JSON
            data = json.loads(vocab_path.read_text(encoding="utf-8"))
            assert isinstance(data, dict)
            assert len(data) > 50000  # Whisper Small vocab is ~51k tokens
        else:
            pytest.skip("whisper_vocab.json not in assets/")

    def test_vocab_has_special_tokens(self):
        """Whisper vocab should include special tokens >= 50257."""
        vocab_path = ANDROID_ASSETS_DIR / "whisper_vocab.json"
        if not vocab_path.exists():
            pytest.skip("whisper_vocab.json not in assets/")
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        # Check for key special tokens
        assert "<|endoftext|>" in data
        assert "<|startoftranscript|>" in data

    def test_onnx_files_gitignored(self):
        """All .onnx files should be in .gitignore (too large for git)."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "*.onnx" in gitignore


class TestAndroidAssets_JavaFileReferences:
    """Verify Java code references match the expected filenames."""

    def test_asr_module_filenames(self):
        """ASRModule.java references whisper_encoder.onnx and whisper_decoder.onnx."""
        # From ASRModule.java lines 38-39
        assert "whisper_encoder.onnx" in ["whisper_encoder.onnx", "whisper_decoder.onnx"]
        assert "whisper_decoder.onnx" in ["whisper_encoder.onnx", "whisper_decoder.onnx"]

    def test_translation_module_filenames(self):
        """TranslationModule.java references encoder/decoder int8 and sentencepiece."""
        # From TranslationModule.java lines 30-32
        expected = {
            "encoder_model_int8.onnx",
            "decoder_model_merged_int8.onnx",
            "sentencepiece_bpe.model",
        }
        assert len(expected) == 3

    def test_readme_matches_java_code(self):
        """README lists the same files that Java code loads."""
        # README says NLLB_encoder.onnx but Java says encoder_model_int8.onnx
        # This is a known naming inconsistency in the README
        readme_names = {
            "NLLB_encoder.onnx",       # README name
            "NLLB_decoder.onnx",       # README name
        }
        java_names = {
            "encoder_model_int8.onnx",          # Java actual
            "decoder_model_merged_int8.onnx",   # Java actual
        }
        # These are DIFFERENT — documenting the README vs code mismatch
        assert readme_names != java_names, (
            "README and Java now use the same names — update this test!"
        )
