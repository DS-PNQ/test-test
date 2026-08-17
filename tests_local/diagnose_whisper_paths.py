# OmniVoice — bisect the Whisper garbage-output problem.
# Combos: {int8, fp32} decoder x {native ONNX mel, Java-DSP-port mel} x
# {KV-cache path, whole-sequence path}. Reference = torch transcript.
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests_local"))
from test_06_onnx_parity import (ASSETS, W_SOT, W_EOT, W_TRANSCRIBE, W_NO_TIMESTAMPS,  # noqa: E402
                                 W_LANG, extract_mel, read_wav_pcm,
                                 decode_whisper_tokens)

FP32_DEC = ROOT / "onnx_models/whisper_decoder_fp32.onnx"
wav = ROOT / "tests_local/data/audio_samples/parity_en.wav"
ref_text = wav.with_suffix(".txt").read_text(encoding="utf-8")
print(f"ref: {ref_text!r}")

samples = read_wav_pcm(wav)
pre = ort.InferenceSession(str(ASSETS / "whisper_preprocess.onnx"), providers=["CPUExecutionProvider"])
mel_native = pre.run(None, {"audio_pcm": samples.reshape(1, -1).astype(np.float32)})[0][0]
mel_java = extract_mel(samples)
enc = ort.InferenceSession(str(ASSETS / "whisper_encoder.onnx"), providers=["CPUExecutionProvider"])
raw = json_loads = __import__("json").loads((ASSETS / "whisper_vocab.json").read_text(encoding="utf-8"))
vocab = {v: k for k, v in raw.items()}


def kv_greedy(dec, enc_out, max_tokens=90, cycle_encoder=False, verbose=False):
    inp_names = [i.name for i in dec.get_inputs()]
    past = sorted(n for n in inp_names if n.startswith("past_key_values"))
    dec_past = [n for n in past if ".encoder." not in n]
    enc_past = [n for n in past if ".encoder." in n]
    out_names = [o.name for o in dec.get_outputs()]

    def step(feed):
        return dict(zip(out_names, dec.run(None, feed)))

    prefill = [W_SOT, W_LANG["en"], W_TRANSCRIBE, W_NO_TIMESTAMPS]
    feed = {"input_ids": np.array([prefill], dtype=np.int64), "encoder_hidden_states": enc_out}
    if "use_cache_branch" in inp_names:
        feed["use_cache_branch"] = np.array([False])
    for n in past:
        feed[n] = np.zeros((1, 12, 0, 64), np.float32)
    res = step(feed)
    enc_pres = {n: res[n.replace("past_key_values.", "present.", 1)] for n in enc_past}
    dec_accum = {n: res[n.replace("past_key_values.", "present.", 1)] for n in dec_past}
    cache_len = 4
    if verbose:
        print("    after prefill: dec present", res["present.0.decoder.key"].shape,
              "enc present", res["present.0.encoder.key"].shape)
    out = []
    while True:
        tok = int(np.argmax(res["logits"][0, -1]))
        if tok == W_EOT or len(out) >= max_tokens:
            break
        out.append(tok)
        feed = {"input_ids": np.array([[tok]], dtype=np.int64), "encoder_hidden_states": enc_out}
        if "use_cache_branch" in inp_names:
            feed["use_cache_branch"] = np.array([True])
        for n in dec_past:
            feed[n] = dec_accum[n]
        for n in enc_past:
            feed[n] = (res[n.replace("past_key_values.", "present.", 1)]
                       if cycle_encoder else enc_pres[n])
        res = step(feed)
        cache_len += 1
        if res["present.0.decoder.key"].shape[2] == cache_len:
            dec_accum = {n: res[n.replace("past_key_values.", "present.", 1)]
                         for n in dec_past}
        else:
            dec_accum = {n: np.concatenate(
                [dec_accum[n], res[n.replace("past_key_values.", "present.", 1)]], axis=2)
                for n in dec_past}
        if verbose and len(out) <= 2:
            print(f"    after step {len(out)}: dec present",
                  res["present.0.decoder.key"].shape, "accum",
                  dec_accum["past_key_values.0.decoder.key"].shape)
    return out


def whole_seq_greedy(dec, enc_out, max_tokens=90):
    inp_names = [i.name for i in dec.get_inputs()]
    past = sorted(n for n in inp_names if n.startswith("past_key_values"))
    out_names = [o.name for o in dec.get_outputs()]
    seq = [W_SOT, W_LANG["en"], W_TRANSCRIBE, W_NO_TIMESTAMPS]
    out = []
    for _ in range(max_tokens):
        feed = {"input_ids": np.array([seq], dtype=np.int64), "encoder_hidden_states": enc_out}
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
    return out


for mel_name, mel in (("native", mel_native), ("javaport", mel_java)):
    enc_out = enc.run(None, {"input_features": mel[None, :, :].astype(np.float32)})[0]
    for dec_name, dec_path in (("int8", ASSETS / "whisper_decoder.onnx"),
                               ("fp32", FP32_DEC if FP32_DEC.exists() else None)):
        if dec_path is None:
            print(f"mel={mel_name} decoder=fp32: (backup missing, skipped)")
            continue
        dec = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
        for path_name, fn in (("kv", kv_greedy), ("noseq", whole_seq_greedy)):
            verbose = (path_name == "kv" and mel_name == "native")
            try:
                ids = fn(dec, enc_out, verbose=verbose) if path_name == "kv" else fn(dec, enc_out)
                text = decode_whisper_tokens(vocab, ids)
                print(f"mel={mel_name:8s} dec={dec_name} path={path_name:5s}: "
                      f"({len(ids):3d} tok) {text!r}")
            except Exception as e:
                print(f"mel={mel_name:8s} dec={dec_name} path={path_name:5s}: "
                      f"ERROR {type(e).__name__}: {str(e)[:100]}")
        # cycling variant: encoder presents taken from the previous step's
        # outputs (the ORIGINAL pre-prefill-hold scheme that worked on device)
        try:
            ids = kv_greedy(dec, enc_out, cycle_encoder=True)
            text = decode_whisper_tokens(vocab, ids)
            print(f"mel={mel_name:8s} dec={dec_name} path=kv-cyc: "
                  f"({len(ids):3d} tok) {text!r}")
        except Exception as e:
            print(f"mel={mel_name:8s} dec={dec_name} path=kv-cyc: "
                  f"ERROR {type(e).__name__}: {str(e)[:100]}")
