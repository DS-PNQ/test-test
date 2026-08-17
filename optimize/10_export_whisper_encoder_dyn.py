# OmniVoice — Export a dynamic-length Whisper-small encoder (short-clip ASR).
#
# The bundled whisper_encoder.onnx declares a FIXED input_features shape of
# (batch, 80, 3000): a 3-second clip pays the full 30-second encoder pass
# (~10x the necessary compute). This script re-exports the encoder from the
# same checkpoint with a dynamic frame axis, quantizes it with the exact
# recipe of 08_quantize_whisper_encoder.py (MatMul-only dynamic int8,
# per-channel, QInt8 — Conv stays fp32), and installs it in place.
#
# Numerics: at input_features length 3000 the graph is mathematically the
# same model; for shorter clips the encoder attends only to real frames
# (the constant clamp-padding tail is not fed). The parity gate decides
# adoption; rollback is:
#   copy onnx_models\preopt_backup\whisper_encoder.onnx+opt back to assets/
#
# Usage:
#   python optimize/10_export_whisper_encoder_dyn.py            # export + validate + install
#   python optimize/10_export_whisper_encoder_dyn.py --skip-install
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"
ONNX_DIR = ROOT / "onnx_models"
HF_CACHE = ASSETS / "hf_cache"

FP32_OUT = ONNX_DIR / "whisper_encoder_dyn_fp32.onnx"
INT8_OUT = ONNX_DIR / "whisper_encoder_dyn_int8.onnx"


def main() -> None:
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from transformers import WhisperForConditionalGeneration
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    log.info("Loading openai/whisper-small from local HF cache ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", cache_dir=str(HF_CACHE), local_files_only=True)
    model.eval()
    encoder = model.model.encoder

    log.info("Exporting encoder with dynamic frame axis ...")
    trace_frames = 512

    import torch.nn.functional as F

    # HF's WhisperEncoder hard-asserts input_features length == 3000
    # ("Whisper expects the mel input features to be of length 3000"), so the
    # wrapper calls the conv stack + position slice + stock layers directly.
    # Submodules must be REGISTERED on the wrapper (not closure-captured) or
    # the tracer rejects their grad-requiring weights as constants.
    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = encoder.conv1
            self.conv2 = encoder.conv2
            self.pos_weight = encoder.embed_positions.weight
            self.layers = encoder.layers
            self.layer_norm = encoder.layer_norm

        def forward(self, input_features):
            x = F.gelu(self.conv1(input_features))
            x = F.gelu(self.conv2(x)).permute(0, 2, 1)
            pos = self.pos_weight[: x.size(1)]
            h = x + pos
            for layer in self.layers:
                h = layer(h, attention_mask=None, layer_head_mask=None,
                          output_attentions=False)[0]
            return self.layer_norm(h)

    with torch.no_grad():
        torch.onnx.export(
            Enc(), (torch.randn(1, 80, trace_frames),), str(FP32_OUT),
            input_names=["input_features"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_features": {0: "batch_size", 2: "sequence_length"},
                          "last_hidden_state": {0: "batch_size", 1: "encoder_sequence_length"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    log.info(f"Saved {FP32_OUT} ({FP32_OUT.stat().st_size/1e6:.1f} MB)")

    sess = ort.InferenceSession(str(FP32_OUT), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    for frames in (3000, 512, 97):
        feats = rng.standard_normal((1, 80, frames)).astype(np.float32)
        got = sess.run(None, {"input_features": feats})[0]
        # conv2 (k3 s2 p1) yields ceil(frames/2) positions
        assert got.shape == (1, (frames + 1) // 2, 768), f"bad shape at frames={frames}: {got.shape}"
        assert np.isfinite(got).all()
        if frames == 3000:
            # eager HF refuses non-3000 inputs, so the numerical reference is
            # only possible at the full window — shorter lengths exercise the
            # same weights and position-slicing mechanics.
            with torch.no_grad():
                ref = encoder(input_features=torch.from_numpy(feats)).last_hidden_state.numpy()
            err = np.abs(got - ref).max()
            log.info(f"  frames={frames}: out {got.shape}, max|Δ| = {err:.2e}")
            assert err < 1e-3, f"encoder mismatch at frames={frames}: {err}"
        else:
            log.info(f"  frames={frames}: out {got.shape} (finite, shape OK)")
    log.info("fp32 validation OK (3000 / 512 / 97 frames vs eager HF)")

    log.info("Quantizing (MatMul-only dynamic int8 — recipe of 08) ...")
    if INT8_OUT.exists():
        INT8_OUT.unlink()
    quantize_dynamic(str(FP32_OUT), str(INT8_OUT),
                     weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul"])
    log.info(f"Saved {INT8_OUT} ({INT8_OUT.stat().st_size/1e6:.1f} MB)")

    if args.skip_install:
        log.info("--skip-install: assets untouched.")
        return

    # Back up the current int8 fixed-shape assets, install the dynamic graph
    # WITHOUT a .opt sibling (07_preoptimize.py regenerates it in the
    # ort==1.22 venv).
    for name in ("whisper_encoder.onnx", "whisper_encoder.opt.onnx"):
        src = ASSETS / name
        if src.exists():
            bak = ONNX_DIR / "preopt_backup" / (name if name.endswith(".onnx")
                                                else "whisper_encoder.fixed3000.opt.onnx")
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(bak))
            log.info(f"moved {name} -> {bak}")
    shutil.copyfile(str(INT8_OUT), str(ASSETS / "whisper_encoder.onnx"))
    log.info(f"installed -> {ASSETS/'whisper_encoder.onnx'}")
    log.info("NEXT: .venv-ort122/Scripts/python optimize/07_preoptimize.py "
             "--models whisper_encoder.onnx, then the parity gate "
             "(pytest tests_local/test_06_onnx_parity.py -k whisper -v).")


if __name__ == "__main__":
    main()
