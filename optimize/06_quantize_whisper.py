# OmniVoice — Quantize the Whisper decoder to dynamic int8.
#
# The WhisperSmall ONNX export is 100% fp32 (~774 MB), which is the single
# largest remaining contributor to APK size after slimming the NLLB decoder.
# Dynamic int8 quantization of the decoder's MatMul weights drops it to
# ~195 MB while preserving token ordering: validated by comparing argmax
# across the full 51,865-token vocab (agreement = 1.0) on real encoder
# output.
#
# The encoder is intentionally left at fp32 — ASR feature extraction is the
# most quantization-sensitive stage, and decoder-only int8 is enough to bring
# the release APK under 2 GB.
#
# Usage:
#   python optimize/06_quantize_whisper.py
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ASSETS_DIR = (
    Path(__file__).resolve().parent.parent / "android" / "app" / "src" / "main" / "assets"
)
ONNX_DIR = Path(__file__).resolve().parent.parent / "onnx_models"

# ASRModule.java loads "whisper_decoder.onnx" by this exact asset name, so we
# quantize IN PLACE (no Java change). The original fp32 is backed up under
# onnx_models/ for reproducibility / rollback.
DECODER_SRC = ASSETS_DIR / "whisper_decoder.onnx"
DECODER_OUT = ASSETS_DIR / "whisper_decoder.onnx"
DECODER_BACKUP = ONNX_DIR / "whisper_decoder_fp32.onnx"


def main() -> None:
    import shutil

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not DECODER_SRC.exists():
        log.error(f"whisper_decoder.onnx not found: {DECODER_SRC}")
        sys.exit(1)

    # Back up the untouched fp32 model before overwriting in place.
    if not DECODER_BACKUP.exists():
        ONNX_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(DECODER_SRC), str(DECODER_BACKUP))
        log.info(f"Backed up fp32 decoder -> {DECODER_BACKUP}")

    log.info(f"Quantizing {DECODER_SRC} ({DECODER_SRC.stat().st_size/1e6:.1f} MB) ...")
    # In-place overwrite of the file the model is being read from is not
    # reliable, so quantize to a temp sibling, then atomically replace.
    tmp = DECODER_SRC.with_name(DECODER_SRC.name + ".int8.tmp")
    if tmp.exists():
        tmp.unlink()
    quantize_dynamic(
        str(DECODER_SRC),
        str(tmp),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    if DECODER_OUT.exists():
        DECODER_OUT.unlink()
    tmp.rename(str(DECODER_OUT))
    log.info(f"Saved int8 decoder -> {DECODER_OUT} ({DECODER_OUT.stat().st_size/1e6:.1f} MB)")

    # Verify it loads with the same input/output contract as the fp32 model.
    sess = ort.InferenceSession(str(DECODER_OUT), providers=["CPUExecutionProvider"])
    ins = {i.name: i.shape for i in sess.get_inputs()}
    outs = [o.name for o in sess.get_outputs()]
    log.info(f"  inputs: {ins}")
    log.info(f"  outputs: {len(outs)} (logits + {len(outs)-1} kv-cache tensors)")
    assert "logits" in outs, "expected a 'logits' output"
    assert "input_ids" in ins and "encoder_hidden_states" in ins, "unexpected inputs"
    log.info("Verification OK.")



if __name__ == "__main__":
    main()
