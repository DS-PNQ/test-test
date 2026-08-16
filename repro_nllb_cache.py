# Reproduce the exact decode sequence of TranslationModule.greedyDecode()
# against the real Xenova decoder_model_merged_int8.onnx, to see the actual
# ONNX Runtime error behind the "[error]" translation result.
import numpy as np
import onnxruntime as ort

MODEL = r"C:\Users\Admin\AppData\Local\Temp\nllb_decoder_slim.onnx"

sess_opts = ort.SessionOptions()
# Mimic OrtSessionConfig on a low-RAM device (my 3.4 change):
sess_opts.enable_cpu_mem_arena = False
sess_opts.enable_mem_pattern = False
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(MODEL, sess_opts, providers=["CPUExecutionProvider"])
inp = {i.name: i for i in sess.get_inputs()}
out_names = [o.name for o in sess.get_outputs()]

print("=== INPUTS ===")
for i in sess.get_inputs():
    print(f"  {i.name}: shape={i.shape} type={i.type}")
print("=== OUTPUTS ===")
for o in sess.get_outputs():
    print(f"  {o.name}: shape={o.shape} type={o.type}")

past_names = [n for n in inp if n.startswith("past_key_values")]
dec_past = [n for n in past_names if ".encoder." not in n]
enc_past = [n for n in past_names if ".encoder." in n]
has_branch = "use_cache_branch" in inp
print(f"\npast inputs: {len(past_names)} (decoder={len(dec_past)}, encoder={len(enc_past)})")
print(f"use_cache_branch input: {has_branch}")

# Fake inputs — shapes/dtypes are what matter, values just need to be valid ids.
NP = {np.int64}
enc_len = 7
feed = {
    "input_ids": np.array([[2, 5]], dtype=np.int64),          # [</s>, <some lang id>]
    "encoder_hidden_states": np.random.randn(1, enc_len, 1024).astype(np.float32),
    "encoder_attention_mask": np.ones((1, enc_len), dtype=np.int64),
}
if has_branch:
    feed["use_cache_branch"] = np.array([False])   # rank-1 [1], like Java new boolean[]{false}
for n in past_names:
    feed[n] = np.zeros((1, 16, 0, 64), dtype=np.float32)

print("\n--- prefill (use_cache_branch=false, empty past) ---")
try:
    out = sess.run(None, feed)
    res1 = dict(zip(out_names, out))
    print("prefill OK; logits shape:", res1["logits"].shape)
    next_tok = int(np.argmax(res1["logits"][0, -1]))
except Exception as e:
    raise SystemExit(f"PREFILL FAILED: {type(e).__name__}: {e}")

print("\n--- step 2 (use_cache_branch=true, present fed back) ---")
feed2 = {
    "input_ids": np.array([[next_tok]], dtype=np.int64),
    "encoder_hidden_states": feed["encoder_hidden_states"],
    "encoder_attention_mask": feed["encoder_attention_mask"],
}
if has_branch:
    feed2["use_cache_branch"] = np.array([True])   # rank-1 [1], like Java new boolean[]{true}
missing = []
for n in past_names:
    present_name = n.replace("past_key_values.", "present.", 1)
    if present_name in res1:
        feed2[n] = res1[present_name]
    else:
        missing.append(present_name)
if missing:
    print("!!! MISSING present outputs:", missing[:4], "…")
    print("    available outputs:", out_names)
try:
    out2 = sess.run(None, feed2)
    res2 = dict(zip(out_names, out2))
    print("step2 OK; logits shape:", res2["logits"].shape)
except Exception as e:
    print(f"STEP2 FAILED: {type(e).__name__}: {e}")

print("\n--- baseline: old whole-sequence path (branch=false, empty past) ---")
try:
    out3 = sess.run(None, feed)
    print("baseline OK; logits shape:", dict(zip(out_names, out3))["logits"].shape)
except Exception as e:
    print(f"BASELINE FAILED: {type(e).__name__}: {e}")
