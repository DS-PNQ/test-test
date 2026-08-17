# OmniVoice — ONNX parity gate (Phase-0 quality gate).
#
# Runs the EXACT on-device inference scheme — ONNX Runtime with KV-cached
# greedy decoding and prefill-held encoder presents, mirroring
# ASRModule/TranslationModule — against the model files bundled in
# android/app/src/main/assets/, then compares quality against the PyTorch
# greedy reference produced by tests_local/gen_parity_reference.py.
#
# Run with a python whose onnxruntime version matches the app (1.22.0):
#   <venv>/Scripts/python -m pytest tests_local/test_06_onnx_parity.py -v
#
# Gates (per plan):
#   NLLB:    BLEU(onnx) >= BLEU(torch greedy) - 1.0   per direction
#   Whisper: WER(onnx)  <= WER(torch greedy) + 1.0    per language (skips
#            without the synthesized fixtures)
from __future__ import annotations

import json
import math
import os
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(os.environ.get("OMNIVOICE_ASSETS", ROOT / "android/app/src/main/assets"))
REF_JSON = Path(os.environ.get("OMNIVOICE_PARITY_REF", ROOT / "tests_local/output/parity_reference.json"))
OUT_JSON = ROOT / "tests_local/output/onnx_parity_results.json"
BASELINE_JSON = ROOT / "tests_local/baselines/onnx_parity_baseline.json"
# Regression tolerance: any asset/model change (pre-optimize, quantization)
# must keep ONNX quality within this band of the recorded baseline.
TOL_BLEU = 0.5
TOL_WER = 0.5
SAMPLES = ROOT / "tests_local/data/audio_samples"

DIRECTIONS = [("vi", "en"), ("en", "vi"),
              ("vi", "zh_hans"), ("zh_hans", "vi"),
              ("vi", "zh_hant"), ("zh_hant", "vi")]

NLLB_FLORES = {"vi": "vie_Latn", "en": "eng_Latn", "zh_hans": "zho_Hans", "zh_hant": "zho_Hant"}
NLLB_LANG_IDS = {"eng_Latn": 256047, "vie_Latn": 256193, "zho_Hans": 256200, "zho_Hant": 256201}
HF_EOS = 2

# Whisper constants (ASRModule.java)
W_SOT, W_EOT, W_TRANSCRIBE, W_NO_TIMESTAMPS = 50258, 50257, 50359, 50363
W_LANG = {"vi": 50278, "en": 50259, "zh": 50260}


# ----------------------------------------------------------------------
# Shared helpers (Tokenizer.java)
# ----------------------------------------------------------------------
def sp_to_hf(sp_id: int) -> int:
    id_ = sp_id + 1
    return {1: 3, 2: 0, 3: 2}.get(id_, id_)


def hf_to_sp(hf_id: int) -> int:
    return {0: 1, 2: 2, 3: 0}.get(hf_id, max(0, hf_id - 1))


def run_greedy_kv(dec, enc_out: np.ndarray, mask: np.ndarray,
                  prefill_ids: list[int], past_shape: tuple, max_tokens: int,
                  eos: int) -> list[int]:
    """KV-cached greedy decode with prefill-held encoder presents — a faithful
    port of TranslationModule.greedyDecodeWithCache / ASRModule.decodeWithCache."""
    inp_names = [i.name for i in dec.get_inputs()]
    past = sorted(n for n in inp_names if n.startswith("past_key_values"))
    dec_past = [n for n in past if ".encoder." not in n]
    enc_past = [n for n in past if ".encoder." in n]
    has_branch = "use_cache_branch" in inp_names
    out_names = [o.name for o in dec.get_outputs()]

    def step(feed):
        return dict(zip(out_names, dec.run(None, feed)))

    # Whisper's merged decoder has no encoder_attention_mask input (the Java
    # ASRModule doesn't feed one either) — only feed what the graph declares.
    feed_mask = "encoder_attention_mask" in inp_names

    feed = {"input_ids": np.array([prefill_ids], dtype=np.int64),
            "encoder_hidden_states": enc_out}
    if feed_mask:
        feed["encoder_attention_mask"] = mask
    if has_branch:
        feed["use_cache_branch"] = np.array([False])
    for n in past:
        feed[n] = np.zeros(past_shape, dtype=np.float32)

    res = step(feed)
    enc_presents = {n: res[n.replace("past_key_values.", "present.", 1)] for n in enc_past}

    # The Whisper merged export emits INCREMENTAL decoder presents (seq grows
    # by exactly the new tokens, e.g. (1,12,1,64) per step) while NLLB's
    # export emits the FULL accumulated cache — detect per step and
    # accumulate caller-side when needed.
    dec_accum = {n: res[n.replace("past_key_values.", "present.", 1)] for n in dec_past}
    cache_len = len(prefill_ids)

    out: list[int] = []
    while True:
        tok = int(np.argmax(res["logits"][0, -1]))
        if tok == eos:
            break
        out.append(tok)
        if len(out) >= max_tokens:
            break
        feed = {"input_ids": np.array([[tok]], dtype=np.int64),
                "encoder_hidden_states": enc_out}
        if feed_mask:
            feed["encoder_attention_mask"] = mask
        if has_branch:
            feed["use_cache_branch"] = np.array([True])
        for n in dec_past:
            feed[n] = dec_accum[n]
        for n in enc_past:
            feed[n] = enc_presents[n]
        res = step(feed)
        cache_len += 1
        pn0 = dec_past[0].replace("past_key_values.", "present.", 1)
        if res[pn0].shape[2] == cache_len:
            dec_accum = {n: res[n.replace("past_key_values.", "present.", 1)]
                         for n in dec_past}          # full-cache export (NLLB)
        else:
            dec_accum = {n: np.concatenate([dec_accum[n],
                                            res[n.replace("past_key_values.", "present.", 1)]],
                                           axis=2)
                         for n in dec_past}          # incremental export (Whisper)
    return out


# ----------------------------------------------------------------------
# NLLB (TranslationModule mirror)
# ----------------------------------------------------------------------
def nllb_translate(nllb, text: str, src: str, tgt: str) -> str:
    import sentencepiece as spm  # noqa: F401  (loaded in fixture)
    sp, enc, dec = nllb
    sp_ids = sp.encode(text)
    ids = [sp_to_hf(i) for i in sp_ids] + [HF_EOS, NLLB_LANG_IDS[NLLB_FLORES[src]]]
    mask = np.ones((1, len(ids)), dtype=np.int64)
    enc_out = enc.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                             "attention_mask": mask})[0]
    tgt_id = NLLB_LANG_IDS[NLLB_FLORES[tgt]]
    out = run_greedy_kv(dec, enc_out, mask, [HF_EOS, tgt_id], (1, 16, 0, 64),
                        max_tokens=256, eos=HF_EOS)
    return sp.decode([hf_to_sp(t) for t in out])


@pytest.fixture(scope="module")
def nllb():
    if not REF_JSON.exists():
        pytest.skip("parity reference missing — run tests_local/gen_parity_reference.py first")
    import sentencepiece as spm
    import onnxruntime as ort
    sp = spm.SentencePieceProcessor(model_file=str(ASSETS / "sentencepiece_bpe.model"))
    enc = ort.InferenceSession(asset("encoder_model_int8.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(asset("decoder_model_merged_int8.onnx"), providers=["CPUExecutionProvider"])
    return sp, enc, dec


# ----------------------------------------------------------------------
# Whisper (ASRModule mirror)
# ----------------------------------------------------------------------
N_MELS, N_FFT, WINDOW_SIZE, HOP_LENGTH, MAX_FRAMES, SAMPLE_RATE = 80, 512, 400, 160, 3000, 16000


def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    num_bins = N_FFT // 2 + 1
    filters = np.zeros((N_MELS, num_bins), dtype=np.float32)
    mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2.0)
    mel_points = [mel_min + (mel_max - mel_min) * i / (N_MELS + 1) for i in range(N_MELS + 2)]
    bin_freqs = [mel_to_hz(m) * (N_FFT + 1) / SAMPLE_RATE for m in mel_points]
    for m in range(N_MELS):
        left, center, right = bin_freqs[m], bin_freqs[m + 1], bin_freqs[m + 2]
        for k in range(num_bins):
            if left <= k <= center and center != left:
                filters[m, k] = (k - left) / (center - left)
            elif center < k <= right and right != center:
                filters[m, k] = (right - k) / (right - center)
        enorm = 2.0 / (mel_to_hz(mel_points[m + 2]) - mel_to_hz(mel_points[m]))
        filters[m] *= enorm
    return filters


def read_wav_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, "expected 16-bit PCM fixture"
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def extract_mel(samples: np.ndarray) -> np.ndarray:
    """Port of ASRModule.extractMelFeatures (post-fix): computes STFT only for
    the audio's real length, pads the mel matrix to 3000 frames afterwards."""
    num_frames = min(len(samples) // HOP_LENGTH, MAX_FRAMES)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(WINDOW_SIZE) / WINDOW_SIZE)).astype(np.float32)

    stft_mag = np.zeros((N_FFT // 2 + 1, num_frames), dtype=np.float32)
    for frame in range(num_frames):
        seg = samples[frame * HOP_LENGTH: frame * HOP_LENGTH + WINDOW_SIZE]
        buf = np.zeros(N_FFT, dtype=np.float32)
        buf[: len(seg)] = seg * window[: len(seg)]
        spec = np.fft.rfft(buf)
        stft_mag[:, frame] = (spec.real ** 2 + spec.imag ** 2)[: N_FFT // 2 + 1]

    filters = mel_filterbank()
    mel_spec = np.zeros((N_MELS, MAX_FRAMES), dtype=np.float32)
    if num_frames > 0:
        mel_spec[:, :num_frames] = np.log10(np.maximum(filters @ stft_mag, 1e-10))

    max_val = float(mel_spec[:, :max(num_frames, 1)].max()) if num_frames > 0 else -10.0
    clamp_min = max(max_val - 8.0, -1e9)
    out = np.where(np.arange(MAX_FRAMES)[None, :] < num_frames, mel_spec, clamp_min)
    out = np.maximum(out, clamp_min)
    return ((out + 4.0) / 4.0).astype(np.float32)


# GPT-2 byte-level BPE tables (ASRModule.BYTE_TO_UNICODE)
def _byte_to_unicode() -> tuple[dict, dict]:
    base = ([b for b in range(0x21, 0x7F)] + [b for b in range(0xA1, 0xAD)]
            + [b for b in range(0xAE, 0x100)])
    in_base = set(base)
    extra = [b for b in range(256) if b not in in_base]
    all_bytes = base + extra
    codes = base + list(range(256, 256 + len(extra)))
    return dict(zip(all_bytes, codes)), dict(zip(codes, all_bytes))


_BYTE_TO_CP, _CP_TO_BYTE = _byte_to_unicode()


def decode_whisper_tokens(vocab: dict, token_ids: list[int]) -> str:
    raw = bytearray()
    for tid in token_ids:
        if tid >= 50257:
            continue
        piece = vocab.get(tid)   # inverted vocab: id -> token string
        if piece is None:
            continue
        for ch in piece:
            b = _CP_TO_BYTE.get(ord(ch))
            if b is not None:
                raw.append(b)
    return raw.decode("utf-8", errors="ignore").strip()


@pytest.fixture(scope="module")
def whisper():
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))
    if not ref.get("asr"):
        pytest.skip("ASR fixtures not generated — gen_parity_reference.py skipped the ASR part")
    for lang in ("en", "vi"):
        if not (SAMPLES / f"parity_{lang}.wav").exists():
            pytest.skip(f"missing fixture {SAMPLES}/parity_{lang}.wav")
    import onnxruntime as ort
    enc = ort.InferenceSession(asset("whisper_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(asset("whisper_decoder.onnx"), providers=["CPUExecutionProvider"])
    raw = json.loads((ASSETS / "whisper_vocab.json").read_text(encoding="utf-8"))
    vocab = {v: k for k, v in raw.items()}   # invert: the file maps token -> id
    # The app prefers the native mel graph (ASRModule.runPreprocess); mirror it,
    # falling back to the Java DSP port exactly like the app does.
    pre = None
    try:
        pre = ort.InferenceSession(str(ASSETS / "whisper_preprocess.onnx"),
                                   providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"whisper_preprocess.onnx unavailable ({e}) — Java DSP mel in harness")
    return enc, dec, vocab, pre


def whisper_kv_greedy(dec, enc_out: np.ndarray, lang: str, max_tokens: int) -> list[int]:
    """KV-cached greedy decode — mirrors ASRModule.decodeWithCache for the
    09-export graph: prefill-held encoder presents, cumulative decoder
    presents cycled zero-copy, empty encoder_hidden_states on steps."""
    inp_names = [i.name for i in dec.get_inputs()]
    out_names = [o.name for o in dec.get_outputs()]
    dec_past = sorted(n for n in inp_names if n.startswith("past_key_values")
                      and ".encoder." not in n)
    enc_past = sorted(n for n in inp_names if n.startswith("past_key_values")
                      and ".encoder." in n)

    empty = np.zeros((1, 12, 0, 64), np.float32)
    empty_ehs = np.zeros((1, 0, enc_out.shape[2]), np.float32)
    seq = [W_SOT, W_LANG[lang], W_TRANSCRIBE, W_NO_TIMESTAMPS]

    feed = {"input_ids": np.array([seq], dtype=np.int64), "encoder_hidden_states": enc_out}
    for n in dec_past + enc_past:
        feed[n] = empty
    enc_presents: dict = {}
    out: list[int] = []
    for step in range(max_tokens):
        res = dict(zip(out_names, dec.run(None, feed)))
        tok = int(np.argmax(res["logits"][0, -1]))
        if step == 0:
            enc_presents = {n: res[n.replace("past_key_values", "present", 1)]
                            for n in enc_past}
        if tok == W_EOT:
            break
        out.append(tok)
        feed = {"input_ids": np.array([[tok]], dtype=np.int64),
                "encoder_hidden_states": empty_ehs, **enc_presents}
        for n in dec_past:
            feed[n] = res[n.replace("past_key_values", "present", 1)]
    return out


def whisper_transcribe(whisper, wav_path: Path, lang: str) -> str:
    """Mirrors ASRModule.runDecoder: the KV-cached fast path when the decoder
    graph declares past_key_values inputs (optimize/09 export), falling back
    to the whole-sequence path for cache-less graphs."""
    enc, dec, vocab, pre = whisper
    samples = read_wav_pcm(wav_path)
    if pre is not None:
        mel = pre.run(None, {"audio_pcm": samples.reshape(1, -1).astype(np.float32)})[0][0]
    else:
        mel = extract_mel(samples)
    # Mirror ASRModule.sliceShortWindow: with a dynamic-length encoder
    # (optimize/10 export) only the audio's real frames are fed.
    enc_dim = enc.get_inputs()[0].shape[2]
    dyn = not isinstance(enc_dim, int) or enc_dim < 0
    frames = min(3000, max(8, len(samples) // 160))
    if dyn and frames < 3000:
        mel = mel[:, :frames]
    enc_out = enc.run(None, {"input_features": mel[None, :, :].astype(np.float32)})[0]

    inp_names = [i.name for i in dec.get_inputs()]
    past = sorted(n for n in inp_names if n.startswith("past_key_values"))
    out_names = [o.name for o in dec.get_outputs()]
    max_tokens = min(len(samples) // SAMPLE_RATE * 30, 448)

    out: list[int] = []
    if past:
        try:
            out = whisper_kv_greedy(dec, enc_out, lang, max_tokens)
            return decode_whisper_tokens(vocab, out)
        except Exception as e:  # noqa: BLE001 — mirror the Java fallback
            print(f"KV-cache decode failed ({e}) — whole-sequence fallback")

    seq = [W_SOT, W_LANG[lang], W_TRANSCRIBE, W_NO_TIMESTAMPS]
    for _ in range(max_tokens):
        feed = {"input_ids": np.array([seq], dtype=np.int64),
                "encoder_hidden_states": enc_out}
        if "use_cache_branch" in inp_names:
            feed["use_cache_branch"] = np.array([False])
        for n in past:
            feed[n] = np.zeros((1, 12, 0, 64), np.float32)
        res = dict(zip(out_names, dec.run(None, feed)))
        tok = int(np.argmax(res["logits"][0, -1]))
        if tok == W_EOT:
            break
        out.append(tok)
        seq.append(tok)
    return decode_whisper_tokens(vocab, out)


# ----------------------------------------------------------------------
# Tests + result recording
# ----------------------------------------------------------------------
RESULTS: dict = {"nllb": {}, "asr": {}}


def _reference() -> dict:
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def asset(name: str) -> str:
    """Mirrors the app's asset resolution: prefer the offline pre-optimized
    sibling (*.opt.onnx from optimize/07_preoptimize.py) when bundled."""
    opt = ASSETS / name.replace(".onnx", ".opt.onnx")
    return str(opt if opt.exists() else ASSETS / name)


@pytest.fixture(scope="module", autouse=True)
def _record_results():
    yield
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in _reference().get("meta", {}).items()}
    OUT_JSON.write_text(json.dumps({"meta": meta, **RESULTS}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    # First successful full run DEFINES the regression baseline. Re-create it
    # deliberately (delete the file) only when an accepted quality change
    # shifts the numbers (e.g. adopting a new model after gate review).
    if not BASELINE_JSON.exists():
        BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_JSON.write_text(
            json.dumps({"note": "ONNX parity regression baseline (auto-created)",
                        **RESULTS}, ensure_ascii=False, indent=1),
            encoding="utf-8")


def _baseline():
    if not BASELINE_JSON.exists():
        return None
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


@pytest.mark.parametrize("src,tgt", DIRECTIONS)
def test_nllb_parity(nllb, src, tgt):
    import sacrebleu
    ref = _reference()["nllb"]
    sentences = json.loads((ROOT / "tests_local/data/parallel_sentences.json")
                           .read_text(encoding="utf-8"))
    hyps = [nllb_translate(nllb, s[src], src, tgt) for s in sentences]
    refs = [s[tgt] for s in sentences]
    tok = "zh" if tgt.startswith("zh") else "13a"
    score = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok).score
    torch_score = ref["bleu"][f"{src}->{tgt}"]
    RESULTS["nllb"][f"{src}->{tgt}"] = {"onnx_bleu": round(score, 2),
                                        "torch_bleu": torch_score,
                                        "delta_vs_torch": round(score - torch_score, 2)}
    base = _baseline()
    if base is None or "nllb" not in base or f"{src}->{tgt}" not in base["nllb"]:
        return  # this run helps define the baseline
    base_score = base["nllb"][f"{src}->{tgt}"]["onnx_bleu"]
    assert score >= base_score - TOL_BLEU, (
        f"ONNX BLEU regression for {src}->{tgt}: {score:.2f} vs baseline "
        f"{base_score:.2f} (tolerance {TOL_BLEU})")


@pytest.mark.parametrize("lang", ["en", "vi"])
def test_whisper_parity(whisper, lang):
    import jiwer
    ref = _reference()["asr"]
    wav = SAMPLES / f"parity_{lang}.wav"
    ref_text = wav.with_suffix(".txt").read_text(encoding="utf-8")
    text = whisper_transcribe(whisper, wav, lang)
    wer = float(jiwer.wer(ref_text, text))
    ref_wer = ref["wer"][lang]
    RESULTS["asr"][lang] = {"onnx_wer": round(wer, 3), "torch_wer": ref_wer,
                            "transcript": text, "torch_transcript": ref["transcripts"][lang]}
    base = _baseline()
    if base is None or "asr" not in base or lang not in base["asr"]:
        return  # this run helps define the baseline
    base_wer = base["asr"][lang]["onnx_wer"]
    assert wer <= base_wer + TOL_WER, (
        f"ONNX WER regression for {lang}: {wer:.3f} vs baseline "
        f"{base_wer:.3f} (tolerance {TOL_WER})")
