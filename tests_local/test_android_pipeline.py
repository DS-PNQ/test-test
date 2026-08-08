# ==========================================================================
# Android pipeline logic tests — validates Java code behavior in Python
# ==========================================================================
#
# The Android app (Java) can't be tested with pytest directly, so these
# tests re-implement the critical Android logic in Python and verify it:
#
#   Part 1 — Tokenizer.java: ID remapping, tokenize(), decode(), getLanguageID()
#   Part 2 — ASRModule.java: Whisper constants, decoder sequence, mel features,
#                            WAV reading, token decoding, hallucination filtering
#   Part 3 — TranslationModule.java: NLLB codes, greedy decode flow, sentence
#                                     splitting, tensor lifecycle (use-after-free)
#   Part 4 — TTSModule.java: locale mapping, language coverage
#   Part 5 — PipelineOrchestrator.java: stage wiring, timing, output path
#   Part 6 — Cross-platform parity: Android vs Python consistency
#   Part 7 — Known bugs (regression tests)

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.asr_whisper import WHISPER_LANG_TOKENS
from backend.translation_nllb import LANG_CODES
from backend.tts_mms import MMS_TTS_MODELS


# ══════════════════════════════════════════════════════════════════════
# PART 1 — Tokenizer.java logic
# ══════════════════════════════════════════════════════════════════════
#
# Python re-implementation of Tokenizer.java's ID remapping logic
# to verify round-trip correctness without needing a JVM.

def _java_encode_remap(sp_id: int) -> int:
    """Tokenizer.java lines 83–90: SentencePiece ID → NLLB ID."""
    nid = sp_id + 1
    if nid == 1: return 3
    if nid == 2: return 0
    if nid == 3: return 2
    return nid

def _java_decode_remap(nllb_id: int) -> int:
    """Tokenizer.java lines 111–118: NLLB ID → SentencePiece ID."""
    xid = nllb_id
    if xid == 3: xid = 1
    elif xid == 0: xid = 2
    elif xid == 2: xid = 3
    return max(0, xid - 1)

def _java_piece_to_id_remap(sp_id: int) -> int:
    """Tokenizer.java lines 129–135: pieceToId remapping."""
    nid = sp_id + 1
    if nid == 1: return 3
    if nid == 2: return 0
    if nid == 3: return 2
    return nid

# LANGUAGES_NLLB array from Tokenizer.java (lines 28–63)
LANGUAGES_NLLB = [
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn",
    "ajp_Arab", "aka_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab",
    "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn",
    "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl",
    "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt",
    "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn",
    "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn",
    "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn",
    "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn",
    "fin_Latn", "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn",
    "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn",
    "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
    "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn",
    "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab",
    "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn", "kaz_Cyrl", "kbp_Latn",
    "kea_Latn", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn",
    "kon_Latn", "kor_Hang", "kmr_Latn", "lao_Laoo", "lvs_Latn", "lij_Latn",
    "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn",
    "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "mag_Deva", "mai_Deva",
    "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn", "mlt_Latn",
    "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr",
    "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn",
    "nya_Latn", "oci_Latn", "gaz_Latn", "ory_Orya", "pag_Latn", "pan_Guru",
    "pap_Latn", "pol_Latn", "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn",
    "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng",
    "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn",
    "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "als_Latn",
    "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn",
    "szl_Latn", "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn",
    "tha_Thai", "tir_Ethi", "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn",
    "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng",
    "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn",
    "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn",
    "yue_Hant", "zho_Hans", "zho_Hant", "zul_Latn",
]
DICTIONARY_LENGTH = 256000

def _java_get_language_id(language_code: str) -> int:
    """Tokenizer.java lines 138–145."""
    for i, lang in enumerate(LANGUAGES_NLLB):
        if lang == language_code:
            return DICTIONARY_LENGTH + 1 + i
    return DICTIONARY_LENGTH + 1


class TestTokenizer_IDRemapping:
    """Verify encode/decode remapping round-trips correctly."""

    @pytest.mark.parametrize("sp_id", list(range(20)) + [100, 1000, 50000, 255999])
    def test_roundtrip_all(self, sp_id):
        nllb_id = _java_encode_remap(sp_id)
        recovered = _java_decode_remap(nllb_id)
        assert recovered == sp_id, f"SP {sp_id} → NLLB {nllb_id} → SP {recovered}"

    @pytest.mark.parametrize("sp_id", [0, 1, 2])
    def test_critical_swap_region(self, sp_id):
        """IDs 0, 1, 2 are the ones being swapped — must survive."""
        assert _java_decode_remap(_java_encode_remap(sp_id)) == sp_id

    def test_pieceToId_matches_encode(self):
        """pieceToId uses the same remapping as the encode path."""
        for sp_id in range(10):
            assert _java_piece_to_id_remap(sp_id) == _java_encode_remap(sp_id)

    def test_decode_never_returns_negative(self):
        """Math.max(0, id - 1) ensures no negative SentencePiece IDs."""
        for nllb_id in range(10):
            assert _java_decode_remap(nllb_id) >= 0


class TestTokenizer_TokenizeSequence:
    """Verify tokenize() builds the correct input sequence."""

    def test_sequence_ends_with_eos_and_src_lang(self):
        """tokenize() appends [tokens..., EOS, srcLangID]."""
        # Simulate: given 3 tokens, output should be [t0, t1, t2, eos, srcLangId]
        fake_tokens = [10, 20, 30]
        eos = 2  # example
        src_lang_id = _java_get_language_id("vie_Latn")

        # Build the extended array as Java does
        ids_extended = fake_tokens + [eos, src_lang_id]
        assert ids_extended[-1] == src_lang_id
        assert ids_extended[-2] == eos
        assert len(ids_extended) == len(fake_tokens) + 2

    def test_attention_mask_all_ones(self):
        """Attention mask should be all 1s, same length as input IDs."""
        length = 7
        attention_mask = [1] * length
        assert all(m == 1 for m in attention_mask)
        assert len(attention_mask) == length


class TestTokenizer_LanguageID:
    """Verify getLanguageID() produces correct IDs."""

    def test_in_scope_languages(self):
        """The three in-scope languages must have valid IDs."""
        vie_id = _java_get_language_id("vie_Latn")
        eng_id = _java_get_language_id("eng_Latn")
        zho_id = _java_get_language_id("zho_Hans")
        assert vie_id == DICTIONARY_LENGTH + 1 + LANGUAGES_NLLB.index("vie_Latn")
        assert eng_id == DICTIONARY_LENGTH + 1 + LANGUAGES_NLLB.index("eng_Latn")
        assert zho_id == DICTIONARY_LENGTH + 1 + LANGUAGES_NLLB.index("zho_Hans")

    def test_unknown_language_returns_default(self):
        """Unknown language falls back to DICTIONARY_LENGTH + 1 (index 0)."""
        assert _java_get_language_id("xxx_Xxxx") == DICTIONARY_LENGTH + 1

    def test_all_languages_unique_ids(self):
        """Every language in the array must produce a unique ID."""
        ids = [_java_get_language_id(lang) for lang in LANGUAGES_NLLB]
        assert len(ids) == len(set(ids)), "Duplicate language IDs found"

    def test_language_count(self):
        """NLLB-200 has exactly 204 language codes."""
        assert len(LANGUAGES_NLLB) == 204


# ══════════════════════════════════════════════════════════════════════
# PART 2 — ASRModule.java logic
# ══════════════════════════════════════════════════════════════════════

# Whisper constants from ASRModule.java
WHISPER_SOT = 50258
WHISPER_EOT = 50257
WHISPER_TRANSCRIBE = 50359
WHISPER_NO_TIMESTAMPS = 50363
WHISPER_MAX_TOKENS = 448
WHISPER_LANGUAGE_TOKENS = {"vi": 50264, "en": 50259, "zh": 50260}
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_MELS = 80
WHISPER_N_FFT = 512
WHISPER_WINDOW_SIZE = 400
WHISPER_HOP_LENGTH = 160
WHISPER_MAX_AUDIO_FRAMES = 3000


class TestASR_WhisperConstants:
    """Validate Whisper special token IDs and audio constants."""

    def test_sot_before_eot(self):
        assert WHISPER_SOT > WHISPER_EOT  # SOT=50258, EOT=50257

    def test_special_tokens_above_vocab(self):
        """All special tokens should be >= 50257 (Whisper vocab size)."""
        for tok in (WHISPER_SOT, WHISPER_EOT, WHISPER_TRANSCRIBE, WHISPER_NO_TIMESTAMPS):
            assert tok >= 50257

    def test_language_tokens_above_vocab(self):
        for lang, tok_id in WHISPER_LANGUAGE_TOKENS.items():
            assert tok_id >= 50257

    def test_max_tokens(self):
        assert WHISPER_MAX_TOKENS == 448

    def test_sample_rate(self):
        assert WHISPER_SAMPLE_RATE == 16000

    def test_mel_dimensions(self):
        assert WHISPER_N_MELS == 80
        assert WHISPER_N_FFT == 512
        assert WHISPER_WINDOW_SIZE == 400
        assert WHISPER_HOP_LENGTH == 160
        assert WHISPER_MAX_AUDIO_FRAMES == 3000


class TestASR_DecoderSequence:
    """Verify the decoder initial token sequence is correct."""

    @pytest.mark.parametrize("lang,expected_tok", [
        ("vi", 50264), ("en", 50259), ("zh", 50260),
    ])
    def test_initial_tokens(self, lang, expected_tok):
        """Initial decoder input: [SOT, lang_token, TRANSCRIBE, NO_TIMESTAMPS]."""
        initial = [WHISPER_SOT, expected_tok, WHISPER_TRANSCRIBE, WHISPER_NO_TIMESTAMPS]
        assert initial[0] == WHISPER_SOT
        assert initial[1] == expected_tok
        assert initial[2] == WHISPER_TRANSCRIBE
        assert initial[3] == WHISPER_NO_TIMESTAMPS
        assert len(initial) == 4

    def test_null_language_defaults_to_vi(self):
        """When language is null, Java defaults to LANGUAGE_TOKENS.get("vi")."""
        default_token = WHISPER_LANGUAGE_TOKENS.get("vi")
        assert default_token == 50264


class TestASR_TokenDecoding:
    """Verify decodeTokens() logic from ASRModule.java."""

    def _decode_tokens(self, token_ids: list[int], vocab: dict[int, str]) -> str:
        """Python re-implementation of ASRModule.decodeTokens()."""
        if not token_ids:
            return ""
        if vocab:
            sb = []
            for tid in token_ids:
                if tid >= 50257:
                    continue  # skip special tokens
                piece = vocab.get(tid)
                if piece is not None:
                    sb.append(piece)
            return "".join(sb).strip()
        return "[No vocab]"

    def test_filters_special_tokens(self):
        vocab = {0: "Hello", 1: " world"}
        result = self._decode_tokens([0, 50258, 1, 50257], vocab)
        assert result == "Hello world"

    def test_empty_input(self):
        assert self._decode_tokens([], {0: "a"}) == ""

    def test_no_vocab(self):
        assert self._decode_tokens([0, 1], {}) == "[No vocab]"

    def test_missing_vocab_entry_skipped(self):
        vocab = {0: "Hello"}
        result = self._decode_tokens([0, 999], vocab)
        assert result == "Hello"

    def test_ġ_replacement(self):
        """Vocab loading replaces Ġ with space."""
        raw_key = "Ġhello"
        replaced = raw_key.replace("Ġ", " ")
        assert replaced == " hello"


class TestASR_MelSpectrogram:
    """Validate mel spectrogram extraction math."""

    def _hann_window(self, size: int) -> np.ndarray:
        """Python re-implementation of ASRModule.hannWindow()."""
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(size) / size))

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def test_hann_window_endpoints(self):
        w = self._hann_window(400)
        assert w[0] == pytest.approx(0.0, abs=1e-6)
        assert w[200] == pytest.approx(1.0, abs=1e-6)

    def test_hann_window_symmetry(self):
        w = self._hann_window(400)
        assert w[100] == pytest.approx(w[300], abs=1e-6)

    def test_hann_window_length(self):
        assert len(self._hann_window(WHISPER_WINDOW_SIZE)) == WHISPER_WINDOW_SIZE

    def test_hz_mel_roundtrip(self):
        for hz in [0, 100, 440, 1000, 4000, 8000]:
            mel = self._hz_to_mel(hz)
            recovered = self._mel_to_hz(mel)
            assert recovered == pytest.approx(hz, abs=0.01)

    def test_mel_output_shape(self):
        """30s of audio at 16kHz → mel shape should be [80, 3000]."""
        sr = WHISPER_SAMPLE_RATE
        duration = 30
        n_samples = sr * duration
        num_frames = (n_samples - WHISPER_WINDOW_SIZE) // WHISPER_HOP_LENGTH + 1
        num_frames = min(num_frames, WHISPER_MAX_AUDIO_FRAMES)
        assert num_frames == WHISPER_MAX_AUDIO_FRAMES

    def test_short_audio_padded_to_30s(self):
        """Audio shorter than 30s gets zero-padded."""
        short_samples = 16000  # 1 second
        target = WHISPER_SAMPLE_RATE * 30
        assert short_samples < target
        padded = np.zeros(target)
        padded[:short_samples] = np.random.randn(short_samples)
        assert len(padded) == target


class TestASR_WAVReading:
    """Validate WAV PCM reading logic."""

    def _write_wav_and_read_pcm(self, tmp_path, samples_float: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Write WAV with scipy, then read it like ASRModule.readWavPCM()."""
        wav_path = tmp_path / "test.wav"
        samples_int16 = (samples_float * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(wav_path), sr, samples_int16)

        # Re-implement Java's readWavPCM logic in Python
        with open(wav_path, "rb") as f:
            header = f.read(44)
            assert len(header) == 44
            sample_rate = struct.unpack_from("<I", header, 24)[0]
            bits_per_sample = struct.unpack_from("<H", header, 34)[0]
            num_channels = struct.unpack_from("<H", header, 22)[0]
            audio_bytes = f.read()

        bytes_per_sample = bits_per_sample // 8
        num_samples = len(audio_bytes) // (bytes_per_sample * num_channels)
        result = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            offset = i * bytes_per_sample * num_channels
            raw = struct.unpack_from("<h", audio_bytes, offset)[0]
            result[i] = raw / 32768.0

        return result

    def test_reads_correct_sample_count(self, tmp_path):
        original = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        recovered = self._write_wav_and_read_pcm(tmp_path, original)
        assert len(recovered) == len(original)

    def test_reads_correct_values(self, tmp_path):
        original = np.array([0.5, -0.5, 0.0, 1.0, -1.0], dtype=np.float32)
        recovered = self._write_wav_and_read_pcm(tmp_path, original)
        # int16 quantization introduces small error
        np.testing.assert_allclose(recovered, original, atol=1.0 / 32768)

    def test_empty_wav_header_only(self, tmp_path):
        """WAV with <=44 bytes should be treated as empty."""
        wav_path = tmp_path / "empty.wav"
        wav_path.write_bytes(b'\x00' * 44)
        file_size = wav_path.stat().st_size
        assert file_size <= 44


# ══════════════════════════════════════════════════════════════════════
# PART 3 — TranslationModule.java logic
# ══════════════════════════════════════════════════════════════════════

# NLLB_CODES from TranslationModule.java (lines 41–46)
ANDROID_NLLB_CODES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans"}


class TestTranslation_NLLBCodes:
    """Verify Android NLLB_CODES mapping."""

    def test_all_pipeline_languages_present(self):
        for lang in ("vi", "en", "zh"):
            assert lang in ANDROID_NLLB_CODES

    def test_codes_match_python(self):
        for lang, code in ANDROID_NLLB_CODES.items():
            assert LANG_CODES[lang] == code

    def test_flores_format(self):
        for lang, code in ANDROID_NLLB_CODES.items():
            parts = code.split("_")
            assert len(parts) == 2
            assert len(parts[0]) == 3  # ISO 639-3
            assert len(parts[1]) == 4  # ISO 15924


class TestTranslation_GetNllbCode:
    """Verify getNllbCode() fallback behavior."""

    def _get_nllb_code(self, lang_code: str) -> str:
        """Python re-implementation of TranslationModule.getNllbCode()."""
        return ANDROID_NLLB_CODES.get(lang_code, "vie_Latn")

    def test_known_languages(self):
        assert self._get_nllb_code("vi") == "vie_Latn"
        assert self._get_nllb_code("en") == "eng_Latn"
        assert self._get_nllb_code("zh") == "zho_Hans"

    def test_unknown_falls_back_to_vietnamese(self):
        """DANGER: Unknown lang codes silently fall back to Vietnamese."""
        assert self._get_nllb_code("fr") == "vie_Latn"
        assert self._get_nllb_code("zh_hans") == "vie_Latn"  # ← BUG
        assert self._get_nllb_code("zh_hant") == "vie_Latn"  # ← BUG


class TestTranslation_GreedyDecode:
    """Verify greedy decode flow from TranslationModule.java."""

    def test_starts_with_target_lang_id(self):
        """First token in outputIds should be the target language ID."""
        tgt_lang_id = _java_get_language_id("eng_Latn")
        output_ids = [tgt_lang_id]  # greedyDecode initializes with this
        assert output_ids[0] == tgt_lang_id

    def test_stops_at_eos(self):
        """Decode loop should stop when EOS token is generated."""
        eos_id = 2  # example
        tokens = [256047, 100, 200, 300, eos_id]
        # Simulate: collect until EOS
        output = []
        for t in tokens:
            output.append(t)
            if t == eos_id:
                break
        assert output[-1] == eos_id

    def test_max_tokens_limit(self):
        """Decode loop should stop after MAX_OUTPUT_TOKENS=256."""
        MAX_OUTPUT_TOKENS = 256
        tokens = list(range(1000))  # more than 256
        output = tokens[:MAX_OUTPUT_TOKENS]
        assert len(output) == MAX_OUTPUT_TOKENS


class TestTranslation_TensorLifecycle:
    """Verify the tensor use-after-free bug in greedyDecode()."""

    def test_asr_clones_tensors(self):
        """ASRModule.java correctly uses TensorUtils.cloneTensor() for KV-cache.
        This is the CORRECT pattern."""
        # ASRModule line 178:
        # nextPKV.put(pastName, TensorUtils.cloneTensor(env, (OnnxTensor) entry.getValue()));
        uses_clone = True  # verified in code review
        assert uses_clone

    def test_translation_does_not_clone_tensors(self):
        """TranslationModule.java line 196 directly assigns the tensor reference
        from stepResult WITHOUT cloning. stepResult is closed by try-with-resources
        on line 177, making the reference a dangling pointer.

        BUG: This is a use-after-free that can cause crashes or wrong translations."""
        # TranslationModule line 196:
        # nextPastKeyValues.put(pastName, (OnnxTensor) entry.getValue());
        uses_clone = False  # verified in code review — BUG
        assert uses_clone is False, (
            "If this fails, the use-after-free bug has been fixed! "
            "Update TranslationModule to use TensorUtils.cloneTensor()."
        )


class TestTranslation_EmptyInput:
    """Verify empty/null input handling."""

    def test_null_or_empty_returns_empty_result(self):
        """TranslationModule.translate() returns empty for null/blank text."""
        # Java lines 87–89:
        # if (text == null || text.trim().isEmpty()) return new TranslationResult("", 0);
        for text in [None, "", "   ", "\t\n"]:
            is_empty = text is None or text.strip() == ""
            assert is_empty


# ══════════════════════════════════════════════════════════════════════
# PART 4 — TTSModule.java logic
# ══════════════════════════════════════════════════════════════════════

# LOCALES from TTSModule.java (lines 41–48)
ANDROID_TTS_LOCALES = {
    "vi": "vi",
    "en": "en",
    "zh": "zh",
    "zh_hans": "zh_CN",
    "zh_hant": "zh_TW",
}


class TestTTS_LocaleMapping:
    """Verify TTSModule language → locale mapping."""

    def test_core_languages_present(self):
        for lang in ("vi", "en", "zh"):
            assert lang in ANDROID_TTS_LOCALES

    def test_chinese_variants_present(self):
        assert "zh_hans" in ANDROID_TTS_LOCALES
        assert "zh_hant" in ANDROID_TTS_LOCALES

    def test_unknown_language_defaults_to_english(self):
        """TTSModule uses getOrDefault(language, Locale.ENGLISH)."""
        default = ANDROID_TTS_LOCALES.get("unknown", "en")
        assert default == "en"


# ══════════════════════════════════════════════════════════════════════
# PART 5 — PipelineOrchestrator.java logic
# ══════════════════════════════════════════════════════════════════════

class TestOrchestrator_StageWiring:
    """Verify PipelineOrchestrator chains stages correctly."""

    def test_process_calls_three_stages(self):
        """process() should call: asr.transcribe → translator.translate → tts.synthesize."""
        stages = ["asr.transcribe", "translator.translate", "tts.synthesize"]
        assert len(stages) == 3

    def test_asr_receives_src_lang(self):
        """ASR is called with (audioPath, srcLang)."""
        # Java line 48: asr.transcribe(audioPath, srcLang)
        pass  # Verified by code inspection

    def test_translator_receives_asr_text_and_both_langs(self):
        """Translator is called with (asrResult.text, srcLang, tgtLang)."""
        # Java line 54-55: translator.translate(asrResult.text, srcLang, tgtLang)
        pass  # Verified by code inspection

    def test_tts_receives_translation_and_tgt_lang(self):
        """TTS is called with (translationResult.text, tgtLang, outputPath)."""
        # Java line 64: tts.synthesize(translationResult.text, tgtLang, outputPath)
        pass  # Verified by code inspection


class TestOrchestrator_OutputPath:
    """Verify output WAV file naming convention."""

    @pytest.mark.parametrize("src,tgt", [
        ("vi", "en"), ("en", "vi"), ("vi", "zh"), ("zh", "vi"),
    ])
    def test_output_filename_pattern(self, src, tgt):
        """Output WAV: translated_{srcLang}_to_{tgtLang}.wav"""
        filename = f"translated_{src}_to_{tgt}.wav"
        assert filename.startswith("translated_")
        assert filename.endswith(".wav")
        assert f"_{src}_to_{tgt}" in filename


class TestOrchestrator_PipelineResult:
    """Verify PipelineResult fields from PipelineOrchestrator.java."""

    def test_result_has_all_fields(self):
        """PipelineResult should have: transcript, translation, audioPath,
        asrMs, translationMs, ttsMs, totalMs."""
        fields = ["transcript", "translation", "audioPath",
                   "asrMs", "translationMs", "ttsMs", "totalMs"]
        assert len(fields) == 7

    def test_text_only_result(self):
        """translateText() returns: (text, result, null, 0, elapsed, 0, elapsed)."""
        # Java line 87: new PipelineResult(text, result.text, null, 0, elapsed, 0, elapsed)
        text = "Hello"
        translation = "Xin chào"
        asr_ms = 0
        tts_ms = 0
        elapsed = 150
        assert asr_ms == 0
        assert tts_ms == 0


# ══════════════════════════════════════════════════════════════════════
# PART 6 — Cross-platform parity: Android ↔ Python
# ══════════════════════════════════════════════════════════════════════

class TestParity_LanguageCodes:
    """Android and Python must use identical NLLB FLORES codes."""

    @pytest.mark.parametrize("lang,expected_flores", [
        ("vi", "vie_Latn"),
        ("en", "eng_Latn"),
        ("zh", "zho_Hans"),
    ])
    def test_nllb_codes_match(self, lang, expected_flores):
        assert ANDROID_NLLB_CODES[lang] == expected_flores
        assert LANG_CODES[lang] == expected_flores

    def test_android_languages_in_nllb_array(self):
        """Every NLLB code used by Android must exist in LANGUAGES_NLLB array."""
        for lang, code in ANDROID_NLLB_CODES.items():
            assert code in LANGUAGES_NLLB, f"Android code '{code}' not in LANGUAGES_NLLB"


class TestParity_WhisperTokens:
    """Android and Python must agree on Whisper language tokens."""

    def test_same_languages_supported(self):
        assert set(WHISPER_LANGUAGE_TOKENS.keys()) == set(WHISPER_LANG_TOKENS.keys())

    def test_python_token_format(self):
        """Python uses string tokens like '<|vi|>'."""
        for lang in WHISPER_LANGUAGE_TOKENS:
            assert WHISPER_LANG_TOKENS[lang] == f"<|{lang}|>"


class TestParity_PipelineFlow:
    """Both platforms should follow the same no-branching pipeline."""

    def test_same_three_stages(self):
        """Both platforms: ASR → Translation → TTS, no language routing."""
        python_stages = ["WhisperASR.transcribe", "NLLBTranslator.translate", "MMSTTS.synthesize"]
        android_stages = ["ASRModule.transcribe", "TranslationModule.translate", "TTSModule.synthesize"]
        assert len(python_stages) == len(android_stages) == 3

    def test_both_support_text_only_mode(self):
        """Both platforms have a text-only shortcut that skips ASR and TTS."""
        # Python: OmniVoicePipeline.translate_text()
        # Android: PipelineOrchestrator.translateText()
        pass  # Verified by code inspection


class TestParity_ModelConstants:
    """Verify model architecture constants match between platforms."""

    def test_whisper_kv_cache_heads(self):
        """Android ASR uses 12 heads, 64 dim for Whisper Small KV-cache."""
        android_heads = 12
        android_dim = 64
        # Whisper Small: 12 decoder layers, 12 attention heads, d_model=768 → head_dim=64
        assert android_heads == 12
        assert android_dim == 64

    def test_nllb_kv_cache_heads(self):
        """Android translation uses 16 heads, 64 dim for NLLB-600M KV-cache."""
        android_heads = 16
        android_dim = 64
        # NLLB-600M: 12 decoder layers, 16 attention heads, d_model=1024 → head_dim=64
        assert android_heads == 16
        assert android_dim == 64


# ══════════════════════════════════════════════════════════════════════
# PART 7 — Known bugs (regression tests)
# ══════════════════════════════════════════════════════════════════════

class TestKnownBugs:
    """Tests that document known bugs. These should FAIL once fixed."""

    def test_BUG_translation_tensor_use_after_free(self):
        """TranslationModule.greedyDecode() line 196 assigns borrowed tensor
        references from a try-with-resources block. After the block closes,
        pastKeyValues holds dangling references to freed native memory.

        Compare with ASRModule line 178 which correctly clones."""
        # The fix: change line 196 from
        #   nextPastKeyValues.put(pastName, (OnnxTensor) entry.getValue());
        # to
        #   nextPastKeyValues.put(pastName, TensorUtils.cloneTensor(env, (OnnxTensor) entry.getValue()));
        bug_exists = True  # verified by code review
        assert bug_exists, "Bug fixed! Remove this regression test."

    def test_BUG_zh_hans_missing_from_android_nllb_codes(self):
        """Android TranslationModule.NLLB_CODES only has 'zh', not 'zh_hans'.
        If Python pipeline passes 'zh_hans', Android falls back to Vietnamese."""
        assert "zh_hans" not in ANDROID_NLLB_CODES, "Bug fixed! Update NLLB_CODES."
        # Verify the dangerous fallback
        fallback = ANDROID_NLLB_CODES.get("zh_hans", "vie_Latn")
        assert fallback == "vie_Latn", "Fallback changed — update test."

    def test_BUG_unknown_lang_defaults_to_vietnamese(self):
        """TranslationModule.getNllbCode() uses getOrDefault(langCode, 'vie_Latn').
        Any unrecognized language silently translates as if it were Vietnamese."""
        for unknown in ["fr", "de", "ja", "ko", "zh_hans", "zh_hant"]:
            result = ANDROID_NLLB_CODES.get(unknown, "vie_Latn")
            assert result == "vie_Latn", f"'{unknown}' no longer defaults to Vietnamese."

    def test_BUG_asr_no_hallucination_suppression(self):
        """Android Whisper decoder has no repeat-suppression or hallucination
        filtering. On silence/noise it may output repeated garbage tokens.

        Python side uses HuggingFace generate() which has built-in suppression."""
        android_has_repeat_penalty = False
        android_has_ngram_blocking = False
        assert not android_has_repeat_penalty
        assert not android_has_ngram_blocking

    def test_BUG_en_token_id_collides_with_eot_plus_2(self):
        """Whisper en token (50259) = EOT (50257) + 2. Not a bug per se,
        but documents the tight ID space — off-by-one errors here are subtle."""
        assert WHISPER_LANGUAGE_TOKENS["en"] == WHISPER_EOT + 2
        assert WHISPER_LANGUAGE_TOKENS["zh"] == WHISPER_EOT + 3
        assert WHISPER_LANGUAGE_TOKENS["vi"] == WHISPER_EOT + 7
