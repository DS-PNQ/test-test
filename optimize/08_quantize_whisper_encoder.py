# OmniVoice — Quantize the Whisper ENCODER to dynamic int8 (Phase 2.3).
#
# The encoder is the most quantization-sensitive ASR stage (which is why it
# was originally left fp32), but on a ≤4 GB wearable target it is also the
# single biggest fp32 block left (~353 MB -> ~95 MB). This applies the exact
# recipe already validated for the decoder (dynamic int8, per-channel,
# QInt8 — see 06_quantize_whisper.py) and relies on the parity gate to
# decide adoption:
#
#   <venv>/Scripts/python optimize/08_quantize_whisper_encoder.py
#   <venv>/Scripts/python -m pytest tests_local/test_06_onnx_parity.py -k whisper -v
#
# Gate: ONNX WER must stay within tests_local/baselines tolerance. If it
# fails, roll back with:
#   copy onnx_models\whisper_encoder_fp32.onnx android\app\src\main\assets\whisper_encoder.onnx
#
# NOTE: run this BEFORE 07_preoptimize.py (quantize the fp32 original, then
# pre-optimize the quantized graph).
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ASSETS_DIR = (
    Path(__file__).resolve().parent.parent / "android" / "app" / "src" / "main" / "assets"
)
ONNX_DIR = Path(__file__).resolve().parent.parent / "onnx_models"

ENCODER_SRC = ASSETS_DIR / "whisper_encoder.onnx"
ENCODER_BACKUP = ONNX_DIR / "whisper_encoder_fp32.onnx"


def main() -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not ENCODER_SRC.exists():
        log.error(f"whisper_encoder.onnx not found: {ENCODER_SRC}")
        sys.exit(1)
    if ENCODER_SRC.stat().st_size < 200 * 1e6:
        log.error("encoder is already quantized (~95 MB expected) — nothing to do, "
                  f"size is {ENCODER_SRC.stat().st_size/1e6:.1f} MB")
        sys.exit(1)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    if not ENCODER_BACKUP.exists():
        shutil.copyfile(str(ENCODER_SRC), str(ENCODER_BACKUP))
        log.info(f"Backed up fp32 encoder -> {ENCODER_BACKUP}")

    log.info(f"Quantizing {ENCODER_SRC} ({ENCODER_SRC.stat().st_size/1e6:.1f} MB) ...")
    tmp = ENCODER_SRC.with_name(ENCODER_SRC.name + ".int8.tmp")
    if tmp.exists():
        tmp.unlink()
    # MatMul-only int8: the full-graph recipe produces ConvInteger nodes whose
    # CPU kernel is not available in ORT 1.22 (verified on x86; Android
    # unverified) — quantizing only MatMul keeps every op on universally
    # available kernels while still capturing ~99% of the weight mass
    # (attention/MLP projections dominate; the two Conv layers stay fp32).
    quantize_dynamic(
        str(ENCODER_SRC),
        str(tmp),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
    )
    ENCODER_SRC.unlink()
    tmp.rename(str(ENCODER_SRC))
    log.info(f"Saved int8 encoder -> {ENCODER_SRC} "
             f"({ENCODER_SRC.stat().st_size/1e6:.1f} MB)")

    # Structural verification: loads and keeps the I/O contract.
    sess = ort.InferenceSession(str(ENCODER_SRC), providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    log.info(f"  inputs: {ins}; outputs: {[o.name for o in sess.get_outputs()]}")
    assert "input_features" in ins, "expected 'input_features' input"
    log.info("Verification OK — run the parity gate before shipping "
             "(tests_local/test_06_onnx_parity.py -k whisper).")


if __name__ == "__main__":
    main()
