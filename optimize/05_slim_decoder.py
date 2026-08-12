# OmniVoice — Slim the NLLB merged decoder by removing the redundant fp32
# embedding initializer. See module docstring below for details.
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_IN = Path("android/app/src/main/assets/decoder_model_merged_int8.onnx")
DEFAULT_OUT = Path("android/app/src/main/assets/decoder_model_merged_int8.onnx")

FP32_EMBED = "model.shared.weight_merged_0"
Q_EMBED = "model.shared.weight_merged_0_quantized"
Q_SCALE = "model.shared.weight_merged_0_scale"
Q_ZERO = "model.shared.weight_merged_0_zero_point"
FP32_TRANSPOSED = "model.shared.weight_transposed"
Q_TRANSPOSED = "model.shared.weight_transposed_quantized"
ELSE_TRANSPOSE_NODE = "Transpose_2662"


def find_if_else_branch(graph):
    for node in graph.node:
        if node.op_type != "If":
            continue
        for attr in node.attribute:
            if attr.name == "else_branch":
                return attr.g
    return None


def quantize_embed(m):
    """Rewire the else_branch to use the quantized embedding, then drop the
    fp32 one. The decoder model ships BOTH:
      model.shared.weight_merged_0            fp32  [256206,1024] ~1049 MB
      model.shared.weight_merged_0_quantized  uint8 [256206,1024]  ~262 MB
    The fp32 copy only feeds Transpose_2662 in the If/else_branch; rewire it
    to transpose the quantized array then DequantizeLinear, like the
    then_branch already does."""
    import onnx
    from onnx import helper

    names_main = {i.name for i in m.graph.initializer}
    if FP32_EMBED not in names_main:
        log.info(f"{FP32_EMBED} not present — already slimmed, nothing to do.")
        return False
    for need in (Q_EMBED, Q_SCALE, Q_ZERO):
        if need not in names_main:
            raise RuntimeError(f"{need} missing — cannot safely rewire.")

    else_g = find_if_else_branch(m.graph)
    if else_g is None:
        raise RuntimeError("No If else_branch found — unexpected structure.")

    replaced = False
    for idx, node in enumerate(else_g.node):
        if node.name == ELSE_TRANSPOSE_NODE and node.op_type == "Transpose":
            out_name = node.output[0]
            if out_name != FP32_TRANSPOSED:
                raise RuntimeError(
                    f"Unexpected output {out_name!r}; expected {FP32_TRANSPOSED!r}."
                )
            tq = helper.make_node(
                "Transpose",
                inputs=[Q_EMBED],
                outputs=[Q_TRANSPOSED],
                name=ELSE_TRANSPOSE_NODE + "_quantized",
            )
            dq = helper.make_node(
                "DequantizeLinear",
                inputs=[Q_TRANSPOSED, Q_SCALE, Q_ZERO],
                outputs=[FP32_TRANSPOSED],
                name=ELSE_TRANSPOSE_NODE + "_dequant",
            )
            del else_g.node[idx]
            else_g.node.insert(idx, dq)
            else_g.node.insert(idx, tq)
            replaced = True
            break
    if not replaced:
        raise RuntimeError(
            f"{ELSE_TRANSPOSE_NODE} not found in else_branch — model drifted."
        )

    # Drop the fp32 initializer from the top-level graph.
    for idx, init in enumerate(m.graph.initializer):
        if init.name == FP32_EMBED:
            del m.graph.initializer[idx]
            log.info(
                f"Removed {FP32_EMBED} ({len(init.raw_data)/1e6:.1f} MB fp32 embedding)."
            )
            break
    else:
        raise RuntimeError(f"{FP32_EMBED} initializer not found for deletion.")

    # Sanity: the fp32 name must now be unreferenced anywhere in the graph.
    def assert_unreferenced(g):
        for node in g.node:
            for inp in node.input:
                if inp == FP32_EMBED:
                    raise RuntimeError(f"{FP32_EMBED} still referenced by {node.name}")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    assert_unreferenced(attr.g)
    assert_unreferenced(m.graph)
    return True


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Slim decoder_model_merged_int8.onnx by removing the redundant fp32 "
            "embedding initializer (replaced by the already-present quantized twin)."
        )
    )
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the onnxruntime load check after saving")
    args = ap.parse_args()

    import onnx
    src = Path(args.inp)
    dst = Path(args.out)
    if not src.exists():
        log.error(f"Input not found: {src}")
        sys.exit(1)

    log.info(f"Loading {src} ({src.stat().st_size/1e6:.1f} MB)...")
    m = onnx.load(str(src), load_external_data=False)

    if not quantize_embed(m):
        if src != dst:
            log.info(f"Copying unchanged to {dst}")
            onnx.save(m, str(dst), save_as_external_data=False)
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving slimmed model -> {dst}")
    onnx.save(m, str(dst), save_as_external_data=False)
    log.info(f"  src size: {src.stat().st_size/1e6:9.1f} MB")
    log.info(f"  dst size: {dst.stat().st_size/1e6:9.1f} MB")

    if not args.no_verify:
        log.info("Verifying slimmed model loads in ONNX Runtime...")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
            log.info(f"  OK — {len(sess.get_inputs())} inputs, "
                     f"{len(sess.get_outputs())} outputs")
        except Exception as e:
            log.error(f"  Verification FAILED: {e}")
            log.error(f"  The slimmed file is at {dst} — inspect before using.")
            sys.exit(2)

    log.info("Done.")


if __name__ == "__main__":
    main()

