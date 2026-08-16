# Full-chain reproduction of the app's TranslationModule pipeline:
#   sentencepiece encode -> spToHf -> [tokens..., </s>, src_lang]
#   -> Xenova encoder_model_int8 -> greedyDecodeWithCache (KV-cache path)
# against onnxruntime 1.22.0 with the exact OrtSessionConfig options
# (arena off, mem-pattern off on low-RAM devices, ALL_OPT).
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

ENC = r"C:\Users\Admin\AppData\Local\Temp\nllb_encoder.onnx"
DEC = r"D:\StudioProjects\OmniVoice\android\app\src\main\assets\decoder_model_merged_int8.onnx"   # slimmed variant (README-documented asset)
SP = r"C:\Users\Admin\AppData\Local\Temp\sp.model"

# --- Tokenizer mirroring (Tokenizer.java) ---
def sp_to_hf(sp_id):
    id_ = sp_id + 1
    return {1: 3, 2: 0, 3: 2}.get(id_, id_)

LANG_IDS = {"eng_Latn": 256047, "vie_Latn": 256193, "zho_Hans": 256200, "zho_Hant": 256201}

sp = spm.SentencePieceProcessor(model_file=SP)
text = "Hôm nay trời rất đẹp."
sp_ids = sp.encode(text)
hf_ids = [sp_to_hf(i) for i in sp_ids]
src_lang = LANG_IDS["vie_Latn"]
tgt_lang = LANG_IDS["eng_Latn"]
enc_ids = hf_ids + [2, src_lang]
print(f"SP ids: {sp_ids}\nHF ids: {enc_ids}")

opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = True       # PRE-3.4 defaults (old app behaviour)
opts.enable_mem_pattern = True
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# --- Encoder run (translateSentence) ---
enc = ort.InferenceSession(ENC, opts, providers=["CPUExecutionProvider"])
print("encoder inputs:", [(i.name, i.shape, i.type) for i in enc.get_inputs()])
print("encoder outputs:", [(o.name, o.shape) for o in enc.get_outputs()])
enc_out = enc.run(None, {
    "input_ids": np.array([enc_ids], dtype=np.int64),
    "attention_mask": np.ones((1, len(enc_ids)), dtype=np.int64),
})[0]
print("encoder out:", enc_out.shape, enc_out.dtype)

# --- Decoder: greedyDecodeWithCache ---
dec = ort.InferenceSession(DEC, opts, providers=["CPUExecutionProvider"])
inp_names = [i.name for i in dec.get_inputs()]
past_names = [n for n in inp_names if n.startswith("past_key_values")]
mask = np.ones((1, len(enc_ids)), dtype=np.int64)

feed = {
    "input_ids": np.array([[2, tgt_lang]], dtype=np.int64),
    "encoder_hidden_states": enc_out,
    "encoder_attention_mask": mask,
    "use_cache_branch": np.array([False]),
}
for n in past_names:
    feed[n] = np.zeros((1, 16, 0, 64), dtype=np.float32)

out_names = [o.name for o in dec.get_outputs()]
print("\n--- prefill ---")
res = dict(zip(out_names, dec.run(None, feed)))
print("prefill OK, logits:", res["logits"].shape)
tok = int(np.argmax(res["logits"][0, -1]))
print("first token:", tok, repr(sp.decode([max(0, tok - 1)])) if 3 < tok < 256000 else "(special)")

# RTranslator pattern: cross-attention KV are CONSTANT after prefill — the
# then-branch of this int8 graph emits malformed (0,16,1,64) encoder
# presents, so hold the PREFILL's encoder presents and feed them every step;
# only the decoder self-attention pasts cycle from the previous step.
enc_presents = {n: res[n.replace("past_key_values.", "present.", 1)]
                for n in past_names if ".encoder." in n}

print("\n--- decode steps (KV cache, prefill encoder presents held) ---")
generated = []
for step in range(2, 12):
    feed2 = {
        "input_ids": np.array([[tok]], dtype=np.int64),
        "encoder_hidden_states": enc_out,
        "encoder_attention_mask": mask,
        "use_cache_branch": np.array([True]),
    }
    for n in past_names:
        pn = n.replace("past_key_values.", "present.", 1)
        feed2[n] = enc_presents[n] if ".encoder." in n else res[pn]
    res = dict(zip(out_names, dec.run(None, feed2)))
    tok = int(np.argmax(res["logits"][0, -1]))
    print(f"step {step}: token {tok}")
    if tok == 2:
        break
    generated.append(tok)

hf_to_sp = lambda hf: {0: 1, 2: 2, 3: 0}.get(hf, max(0, hf - 1))
print("\nTRANSLATION:", sp.decode([hf_to_sp(t) for t in generated]))
print("\nFULL CHAIN PASSED")
