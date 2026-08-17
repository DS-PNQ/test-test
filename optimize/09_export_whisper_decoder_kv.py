# OmniVoice — Export a KV-cached Whisper-small decoder (streaming-fast ASR).
#
# The bundled whisper_decoder.onnx is a cache-less export: it accepts no
# past_key_values inputs, so ASRModule.decodeNoCache must re-feed the whole
# sequence every step (O(n^2): step i costs ~i tokens of full decoder
# compute). This script exports a decoder whose per-step cost is CONSTANT
# (one token through 12 layers), which is what makes ASR feel like a
# streaming model:
#
#   inputs : input_ids (1,S) int64
#            encoder_hidden_states (1,E,768) float32   (E=1500 at prefill,
#              E=0 on decode steps — cross K/V come from the encoder pasts)
#            past_key_values.{i}.decoder.key/value   (1,12,L,64)
#            past_key_values.{i}.encoder.key/value   (1,12,1500,64)
#            NO use_cache_branch — a single uniform graph handles prefill
#            (L=0, empty pasts) and steps (L>0) via concat, so no merged
#            If branches (which double the file when pre-optimized).
#   outputs: logits (1,S,51865)
#            present.{i}.decoder.key/value  — CUMULATIVE (past+new), so the
#              app cycles the previous step's Result zero-copy.
#            present.{i}.encoder.key/value   — only the FRESHLY projected
#              cross K/V (1500 at prefill, 0 afterwards). The app keeps the
#              prefill's encoder presents alive and re-feeds them each step
#              (RTranslator's cache-initializer pattern, same as NLLB here).
#
# Cross-attention is computed in split form (scores = q@past_ek^T ++ q@k_new^T;
# out = p_past@ev + p_new@v_new) so a decode step never copies the ~55 MB
# encoder KV block — only small score tensors are concatenated.
#
# Validation: fp32 logits must match HuggingFace eager greedy logits
# (prefill AND a cached step) to <1e-3, then the graph is quantized with the
# exact recipe of 06_quantize_whisper.py (dynamic int8, per-channel, QInt8)
# and installed as assets/whisper_decoder.onnx. The stale cache-less
# whisper_decoder.opt.onnx is removed (the app prefers .opt.onnx when
# present — leaving it would keep the old slow graph!). Re-run
# 07_preoptimize.py afterwards to regenerate the .opt sibling.
#
# Usage:
#   python optimize/09_export_whisper_decoder_kv.py            # export + validate + install
#   python optimize/09_export_whisper_decoder_kv.py --keep-fp32  # also keep fp32 copy
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"
ONNX_DIR = ROOT / "onnx_models"
HF_CACHE = ASSETS / "hf_cache"

N_LAYER, N_HEAD, HEAD_DIM, D_MODEL = 12, 12, 64, 768
SCALE = HEAD_DIM ** -0.5

W_SOT, W_EOT, W_TRANSCRIBE, W_NO_TIMESTAMPS = 50258, 50257, 50359, 50363
W_LANG = {"vi": 50278, "en": 50259, "zh": 50260}


def load_model():
    import torch
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", cache_dir=str(HF_CACHE), local_files_only=True)
    model.eval()
    return model


def build_wrapper(model):
    """Wraps HF whisper-small decoder submodules with explicit KV inputs."""
    import torch
    import torch.nn.functional as F

    dec = model.model.decoder

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = dec.embed_tokens
            self.pos_weight = dec.embed_positions.weight
            self.layers = dec.layers
            self.layer_norm = dec.layer_norm
            self.proj_out = model.proj_out

        def _shape(self, x):
            b, s, _ = x.shape
            return x.view(b, s, N_HEAD, HEAD_DIM).transpose(1, 2)

        def _unshape(self, x):
            b, _, s, _ = x.shape
            return x.transpose(1, 2).reshape(b, s, D_MODEL)

        def _self_attn(self, x, layer, dk, dv):
            sa = layer.self_attn
            q = self._shape(sa.q_proj(x))
            k_new = self._shape(sa.k_proj(x))
            v_new = self._shape(sa.v_proj(x))
            k = torch.cat([dk, k_new], dim=2)
            v = torch.cat([dv, v_new], dim=2)
            past_len, seq_len = dk.size(2), x.size(1)
            # Causal mask over [past + new] keys, built from shapes only so
            # the export stays dynamic.
            q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)
            k_pos = torch.arange(past_len + seq_len, device=x.device)
            blocked = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1)).to(x.dtype) * -1e9
            scores = torch.matmul(q, k.transpose(-1, -2)) * SCALE + blocked
            probs = torch.softmax(scores, dim=-1)
            out = self._unshape(torch.matmul(probs, v))
            return sa.out_proj(out), k, v

        def _cross_attn(self, x, layer, ek, ev, ehs):
            ea = layer.encoder_attn
            q = self._shape(ea.q_proj(x))
            k_new = self._shape(ea.k_proj(ehs))
            v_new = self._shape(ea.v_proj(ehs))
            # Split form: never materializes past+new encoder KV together.
            s_past = torch.matmul(q, ek.transpose(-1, -2))
            s_new = torch.matmul(q, k_new.transpose(-1, -2))
            scores = torch.cat([s_past, s_new], dim=-1) * SCALE
            probs = torch.softmax(scores, dim=-1)
            p_past = probs[..., : ek.size(2)]
            p_new = probs[..., ek.size(2):]
            out = self._unshape(torch.matmul(p_past, ev) + torch.matmul(p_new, v_new))
            return ea.out_proj(out), k_new, v_new

        def forward(self, input_ids, encoder_hidden_states, *pasts):
            # Flat positional pasts in layer order: dk_i, dv_i, ek_i, ev_i —
            # the legacy torch.onnx exporter only supports positional args.
            dks = [pasts[4 * i + 0] for i in range(N_LAYER)]
            dvs = [pasts[4 * i + 1] for i in range(N_LAYER)]
            eks = [pasts[4 * i + 2] for i in range(N_LAYER)]
            evs = [pasts[4 * i + 3] for i in range(N_LAYER)]

            seq_len = input_ids.size(1)
            past_len = dks[0].size(2)

            x = self.embed_tokens(input_ids)
            pos = self.pos_weight[past_len: past_len + seq_len]
            x = x + pos.unsqueeze(0)

            presents = []
            for i, layer in enumerate(self.layers):
                residual = x
                h = layer.self_attn_layer_norm(x)
                h, dk, dv = self._self_attn(h, layer, dks[i], dvs[i])
                x = residual + h

                residual = x
                h = layer.encoder_attn_layer_norm(x)
                h, ek, ev = self._cross_attn(h, layer, eks[i], evs[i], encoder_hidden_states)
                x = residual + h

                residual = x
                h = layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(x))))
                x = residual + h

                presents += [dk, dv, ek, ev]

            x = self.layer_norm(x)
            logits = self.proj_out(x)
            presents_out = []
            for i in range(N_LAYER):
                presents_out += [presents[4 * i], presents[4 * i + 1],
                                 presents[4 * i + 2], presents[4 * i + 3]]
            return (logits, *presents_out)

    return Wrapper()


def export_fp32(model, wrapper, out_path: Path):
    import torch

    dev = torch.device("cpu")
    B, S, L, E = 1, 2, 3, 5
    args = [torch.zeros(B, S, dtype=torch.int64, device=dev),
            torch.randn(B, E, D_MODEL, device=dev)]
    input_names = ["input_ids", "encoder_hidden_states"]
    for i in range(N_LAYER):
        for pref in ("dk", "dv", "ek", "ev"):
            shape = (B, N_HEAD, (L if pref.startswith("d") else E), HEAD_DIM)
            args.append(torch.randn(*shape, device=dev))
            input_names.append(f"{pref}{i}")

    output_names = ["logits"]
    for i in range(N_LAYER):
        for role in ("decoder_key", "decoder_value", "encoder_key", "encoder_value"):
            output_names.append(f"present_{i}_{role}")

    dyn = {"input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
           "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"}}
    out_dyn = {"logits": {0: "batch_size", 1: "decoder_sequence_length"}}
    for i in range(N_LAYER):
        dyn[f"dk{i}"] = {0: "batch_size", 2: "past_decoder_sequence_length"}
        dyn[f"dv{i}"] = {0: "batch_size", 2: "past_decoder_sequence_length"}
        dyn[f"ek{i}"] = {0: "batch_size", 2: "past_encoder_sequence_length"}
        dyn[f"ev{i}"] = {0: "batch_size", 2: "past_encoder_sequence_length"}
        out_dyn[f"present_{i}_decoder_key"] = {0: "batch_size", 2: "past_decoder_sequence_length + decoder_sequence_length"}
        out_dyn[f"present_{i}_decoder_value"] = {0: "batch_size", 2: "past_decoder_sequence_length + decoder_sequence_length"}
        out_dyn[f"present_{i}_encoder_key"] = {0: "batch_size", 2: "encoder_sequence_length"}
        out_dyn[f"present_{i}_encoder_value"] = {0: "batch_size", 2: "encoder_sequence_length"}

    torch.onnx.export(
        wrapper, tuple(args), str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={**dyn, **out_dyn},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    rename_io_convention(out_path)


def rename_io_convention(path: Path):
    """Renames graph inputs/outputs to the dotted past_key_values./present.
    convention the app and the parity harness already detect:
    dk0 -> past_key_values.0.decoder.key, present_0_encoder_key -> present.0.encoder.key
    """
    import onnx

    m = onnx.load(str(path))

    def inp_map(n: str) -> str:
        # trace input names are dk0 / dv3 / ek11 / ev0 — the suffix is the
        # full layer index (n[2] alone would collide dk10 with dk1).
        if n in ("input_ids", "encoder_hidden_states"):
            return n
        who = {"dk": "decoder.key", "dv": "decoder.value",
               "ek": "encoder.key", "ev": "encoder.value"}[n[:2]]
        return f"past_key_values.{int(n[2:])}.{who}"

    def out_map(n: str) -> str:
        if n == "logits":
            return n
        # present_{i}_{decoder|encoder}_{key|value}
        parts = n.split("_")  # ['present', '0', 'decoder', 'key']
        return f"present.{parts[1]}.{parts[2]}.{parts[3]}"

    imap = {i.name: inp_map(i.name) for i in m.graph.input}
    omap = {o.name: out_map(o.name) for o in m.graph.output}

    for i in m.graph.input:
        i.name = imap[i.name]
    for o in m.graph.output:
        o.name = omap[o.name]
    for node in m.graph.node:
        for k, v in enumerate(node.input):
            # internal consumers may reference either namespace
            if v in imap:
                node.input[k] = imap[v]
            elif v in omap:
                node.input[k] = omap[v]
        for k, v in enumerate(node.output):
            if v in omap:
                node.output[k] = omap[v]
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def role_of(n: str) -> str:
    return "decoder" if n.startswith("d") else "encoder"


def validate_fp32(model, fp32_path: Path):
    """fp32 graph vs eager HF: prefill logits and one cached step must match."""
    import numpy as np
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    B, E, S0 = 1, 7, 4
    ids0 = np.array([[W_SOT, W_LANG["en"], W_TRANSCRIBE, W_NO_TIMESTAMPS]], dtype=np.int64)
    ehs = rng.standard_normal((B, E, D_MODEL)).astype(np.float32)
    pasts = {f"past_key_values.{i}.{w}": np.zeros((B, N_HEAD, 0, HEAD_DIM), np.float32)
             for i in range(N_LAYER) for w in ("decoder.key", "decoder.value",
                                               "encoder.key", "encoder.value")}
    feed = {"input_ids": ids0, "encoder_hidden_states": ehs, **pasts}
    res = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]
    res = dict(zip(out_names, res))

    def eager_logits(ids: np.ndarray) -> np.ndarray:
        h = model.model.decoder(input_ids=torch.from_numpy(ids),
                                encoder_hidden_states=torch.from_numpy(ehs)).last_hidden_state
        return model.proj_out(h).detach().numpy()

    ref = eager_logits(ids0)
    err0 = np.abs(res["logits"] - ref).max()
    log.info(f"prefill  max|Δlogits| = {err0:.2e}")
    assert err0 < 1e-3, f"prefill mismatch: {err0}"

    # One cached step: past = prefill presents, single new token.
    tok = int(np.argmax(res["logits"][0, -1]))
    step_ids = np.array([[tok]], dtype=np.int64)
    enc_k = {f"past_key_values.{i}.encoder.key": res[f"present.{i}.encoder.key"] for i in range(N_LAYER)}
    enc_v = {f"past_key_values.{i}.encoder.value": res[f"present.{i}.encoder.value"] for i in range(N_LAYER)}
    dec_k = {f"past_key_values.{i}.decoder.key": res[f"present.{i}.decoder.key"] for i in range(N_LAYER)}
    dec_v = {f"past_key_values.{i}.decoder.value": res[f"present.{i}.decoder.value"] for i in range(N_LAYER)}
    empty_ehs = np.zeros((B, 0, D_MODEL), np.float32)
    feed2 = {"input_ids": step_ids, "encoder_hidden_states": empty_ehs,
             **dec_k, **dec_v, **enc_k, **enc_v}
    res2 = dict(zip(out_names, sess.run(None, feed2)))

    ids_full = np.concatenate([ids0, [[tok]]], axis=1)
    ref2 = eager_logits(ids_full)
    err1 = np.abs(res2["logits"][0, -1] - ref2[0, -1]).max()
    log.info(f"kv-step  max|Δlogits| = {err1:.2e}")
    assert err1 < 1e-3, f"cached-step mismatch: {err1}"

    # The cumulative-presents contract the Java decode relies on.
    assert res2[f"present.0.decoder.key"].shape[2] == S0 + 1, "decoder presents must be cumulative"
    assert res2[f"present.0.encoder.key"].shape[2] == 0, "step encoder presents must be empty"
    log.info("fp32 validation OK (prefill + cached step vs eager HF)")


def mel_from_fixture(wav_path: Path):
    """DSP log-mel port (same as the parity harness fallback)."""
    import numpy as np

    n_fft, win, hop, n_mels, max_frames, sr = 512, 400, 160, 80, 3000, 16000
    with wave.open(str(wav_path), "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    num_frames = min(len(samples) // hop, max_frames)
    window = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(win) / win)).astype(np.float32)
    stft = np.zeros((n_fft // 2 + 1, num_frames), np.float32)
    for f in range(num_frames):
        seg = samples[f * hop: f * hop + win]
        buf = np.zeros(n_fft, np.float32)
        buf[: len(seg)] = seg * window[: len(seg)]
        spec = np.fft.rfft(buf)
        stft[:, f] = (spec.real ** 2 + spec.imag ** 2)[: n_fft // 2 + 1]
    filters = _mel_filterbank(sr, n_fft, n_mels)
    mel = np.zeros((n_mels, max_frames), np.float32)
    if num_frames:
        mel[:, :num_frames] = np.log10(np.maximum(filters @ stft, 1e-10))
    max_val = float(mel[:, : max(num_frames, 1)].max()) if num_frames else -10.0
    clamp = max(max_val - 8.0, -1e9)
    out = np.where(np.arange(max_frames)[None, :] < num_frames, mel, clamp)
    return ((np.maximum(out, clamp) + 4.0) / 4.0).astype(np.float32)


def _mel_filterbank(sr, n_fft, n_mels):
    import math

    import numpy as np

    def h2m(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def m2h(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    bins = n_fft // 2 + 1
    filters = np.zeros((n_mels, bins), np.float32)
    mel_min, mel_max = h2m(0.0), h2m(sr / 2.0)
    pts = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
    freqs = [m2h(m) * (n_fft + 1) / sr for m in pts]
    for m in range(n_mels):
        left, center, right = freqs[m], freqs[m + 1], freqs[m + 2]
        for k in range(bins):
            if left <= k <= center and center != left:
                filters[m, k] = (k - left) / (center - left)
            elif center < k <= right and right != center:
                filters[m, k] = (right - k) / (right - center)
        filters[m] *= 2.0 / (m2h(pts[m + 2]) - m2h(pts[m]))
    return filters


def kv_greedy(sess, enc_out: "np.ndarray", lang: str, max_tokens: int):
    """Reference KV decode used to smoke-test the exported graph (mirrors the
    Java fast path: prefill-held encoder presents, cumulative decoder presents)."""
    import numpy as np

    out_names = [o.name for o in sess.get_outputs()]
    seq = [W_SOT, W_LANG[lang], W_TRANSCRIBE, W_NO_TIMESTAMPS]
    empty = np.zeros((1, N_HEAD, 0, HEAD_DIM), np.float32)
    feed = {"input_ids": np.array([seq], np.int64), "encoder_hidden_states": enc_out}
    for i in range(N_LAYER):
        feed[f"past_key_values.{i}.decoder.key"] = empty
        feed[f"past_key_values.{i}.decoder.value"] = empty
        feed[f"past_key_values.{i}.encoder.key"] = np.zeros((1, N_HEAD, 0, HEAD_DIM), np.float32)
        feed[f"past_key_values.{i}.encoder.value"] = np.zeros((1, N_HEAD, 0, HEAD_DIM), np.float32)
    empty_ehs = np.zeros((1, 0, D_MODEL), np.float32)
    enc_presents: dict = {}
    out = []
    for step in range(max_tokens):
        res = dict(zip(out_names, sess.run(None, feed)))
        tok = int(np.argmax(res["logits"][0, -1]))
        if step == 0:
            enc_presents = {f"past_key_values.{i}.{w}": res[f"present.{i}.{w}"]
                            for i in range(N_LAYER) for w in ("encoder.key", "encoder.value")}
        if tok == W_EOT:
            break
        out.append(tok)
        feed = {"input_ids": np.array([[tok]], np.int64), "encoder_hidden_states": empty_ehs,
                **enc_presents}
        for i in range(N_LAYER):
            feed[f"past_key_values.{i}.decoder.key"] = res[f"present.{i}.decoder.key"]
            feed[f"past_key_values.{i}.decoder.value"] = res[f"present.{i}.decoder.value"]
    return out


def smoke_test_int8(int8_path: Path):
    """End-to-end greedy decode on the parity fixtures with the int8 graph;
    compares WER against the recorded baseline transcripts."""
    import jiwer
    import numpy as np
    import onnxruntime as ort

    enc_sess = ort.InferenceSession(str(ASSETS / "whisper_encoder.opt.onnx"),
                                    providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    vocab_raw = json.loads((ASSETS / "whisper_vocab.json").read_text(encoding="utf-8"))
    vocab = {v: k for k, v in vocab_raw.items()}

    base = json.loads((ROOT / "tests_local/baselines/onnx_parity_baseline.json")
                      .read_text(encoding="utf-8"))["asr"]

    for lang in ("en", "vi"):
        wav = ROOT / "tests_local/data/audio_samples" / f"parity_{lang}.wav"
        mel = mel_from_fixture(wav)
        enc_out = enc_sess.run(None, {"input_features": mel[None]})[0]
        toks = kv_greedy(dec_sess, enc_out, lang, max_tokens=120)
        text = _decode_tokens(vocab, toks)
        ref = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        wer = float(jiwer.wer(ref, text))
        log.info(f"[{lang}] kv-int8 transcript: {text}")
        log.info(f"[{lang}] WER={wer:.3f} (baseline onnx WER={base[lang]['onnx_wer']}, "
                 f"baseline transcript: {base[lang]['transcript']})")


def _decode_tokens(vocab, ids):
    base = ([b for b in range(0x21, 0x7F)] + [b for b in range(0xA1, 0xAD)]
            + [b for b in range(0xAE, 0x100)])
    in_base = set(base)
    extra = [b for b in range(256) if b not in in_base]
    cp2b = dict(zip(base + list(range(256, 256 + len(extra))), base + extra))
    raw = bytearray()
    for t in ids:
        if t >= 50257:
            continue
        for ch in vocab.get(t, ""):
            b = cp2b.get(ord(ch))
            if b is not None:
                raw.append(b)
    return raw.decode("utf-8", errors="ignore").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fp32", action="store_true",
                    help="keep the fp32 export at onnx_models/whisper_decoder_kv_fp32.onnx")
    ap.add_argument("--skip-install", action="store_true",
                    help="export + validate only; do not touch assets/")
    args = ap.parse_args()

    fp32_out = ONNX_DIR / "whisper_decoder_kv_fp32.onnx"
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading openai/whisper-small from local HF cache ...")
    model = load_model()

    log.info("Exporting KV-cached decoder (fp32) ...")
    wrapper = build_wrapper(model)
    export_fp32(model, wrapper, fp32_out)
    log.info(f"Saved {fp32_out} ({fp32_out.stat().st_size/1e6:.1f} MB)")

    log.info("Validating fp32 graph against eager HuggingFace ...")
    validate_fp32(model, fp32_out)

    log.info("Quantizing to dynamic int8 (per-channel, QInt8 — recipe of 06) ...")
    import onnxruntime.quantization as q
    from onnxruntime.quantization import QuantType

    int8_tmp = ONNX_DIR / "whisper_decoder_kv_int8.onnx"
    if int8_tmp.exists():
        int8_tmp.unlink()
    q.quantize_dynamic(str(fp32_out), str(int8_tmp),
                       weight_type=QuantType.QInt8, per_channel=True)
    log.info(f"Saved {int8_tmp} ({int8_tmp.stat().st_size/1e6:.1f} MB)")

    log.info("Smoke test: KV greedy on the parity fixtures ...")
    smoke_test_int8(int8_tmp)

    if args.skip_install:
        log.info("--skip-install: assets untouched.")
        return
    if not args.keep_fp32:
        fp32_out.unlink()

    # Install: replace the in-APK decoder and DROP the stale pre-optimized
    # sibling (the app would otherwise keep loading the old cache-less graph).
    stale_opt = ASSETS / "whisper_decoder.opt.onnx"
    if stale_opt.exists():
        bak = ONNX_DIR / "preopt_backup" / "whisper_decoder.opt.cacheless.onnx"
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stale_opt), str(bak))
        log.info(f"moved stale {stale_opt.name} -> {bak}")
    shutil.copyfile(str(int8_tmp), str(ASSETS / "whisper_decoder.onnx"))
    log.info(f"installed -> {ASSETS/'whisper_decoder.onnx'}")
    log.info("NEXT: re-run optimize/07_preoptimize.py (whisper_decoder.onnx) in the "
             "ort==1.22 venv, then the parity gate "
             "(pytest tests_local/test_06_onnx_parity.py -k whisper -v).")


if __name__ == "__main__":
    main()
