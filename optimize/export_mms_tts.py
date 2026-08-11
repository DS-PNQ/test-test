# OmniVoice — MMS-TTS ONNX Export Script
#
# Downloads each MMS-TTS checkpoint (vi/en/zh), exports it to ONNX, and
# copies the model + tokenizer vocab into the Android assets directory.
#
# Usage:
#   pip install optimum onnxruntime transformers
#   python optimize/export_mms_tts.py
#   python optimize/export_mms_tts.py --langs vi en      # subset
#
# The resulting assets are consumed by
# android/app/src/main/java/com/omnivoice/onspeak47/pipeline/TTSModule.java

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Android assets directory (mirrors 01_export_onnx.py structure)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "android" / "app" / "src" / "main" / "assets"

# MMS-TTS checkpoints keyed by simple language code.
# Keep in sync with backend/tts_mms.py::MMS_TTS_MODELS.
#
# NOTE: Meta never released an MMS-TTS Mandarin checkpoint. The sibling Chinese
# releases are Hakka (-hak) and Min-Nan (-nan), NOT Mandarin, so "zh" is
# intentionally NOT mapped here. TTSModule.java falls back to the Android system
# TTS engine for languages without a bundled ONNX model (i.e. zh). If a
# MMS-compatible Mandarin VITS checkpoint becomes available, add it here and it
# will plug into the existing pipeline with no Java changes.
MMS_TTS_MODELS = {
    "vi": "facebook/mms-tts-vie",
    "en": "facebook/mms-tts-eng",
}


def export_language(lang: str, model_id: str, assets_dir: Path, verify: bool) -> Path:
    """Export one MMS-TTS checkpoint to ONNX and copy outputs into assets."""
    # Show the REAL missing module if imports fail instead of a generic hint,
    # so an unhelpful "install dependencies" message never masks the cause.
    try:
        from optimum.exporters.onnx import main_export
        from optimum.utils.save_utils import maybe_save_preprocessors
        from transformers import AutoTokenizer
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        log.error("Install with: pip install --upgrade \"optimum[onnxruntime]\" transformers onnxruntime")
        sys.exit(1)

    log.info("=" * 60)
    log.info(f"Exporting MMS-TTS [{lang}] from {model_id}")
    assets_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = assets_dir / f"temp_mms_tts_{lang}"
    hf_cache = str(assets_dir / "hf_cache")

    # Export to ONNX via Optimum's stable exporter API.
    # (ORTModelForTextToWaveform was removed from optimum.onnxruntime in recent
    #  releases, so we drive the exporter directly. This produces the same
    #  single-graph VITS model the Android TTSModule expects.)
    # monolith=True forces a single model.onnx instead of encoder/decoder
    # submodels, matching the one-file layout the app loads.
    log.info("  Converting to ONNX (this may take a few minutes)...")
    main_export(
        model_id,
        output=temp_dir,
        task="text-to-audio",
        do_validation=False,   # avoids needing two ORT sessions side-by-side
        monolith=True,         # single .onnx graph
        cache_dir=hf_cache,
    )
    # Save the vocab/tokenizer files alongside the exported graph.
    maybe_save_preprocessors(model_id, temp_dir, trust_remote_code=False)
    # Ensure a HF cache download exists so AutoTokenizer resolves offline.
    AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache)

    # Optimum may emit several .onnx files (some pipelines export submodules
    # separately). MMS-TTS VITS is exported as a single model — pick the
    # largest, which is always the main synthesis graph.
    onnx_files = list(temp_dir.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"No .onnx file produced for {lang} in {temp_dir}")
    main_onnx = max(onnx_files, key=lambda f: f.stat().st_size)
    log.info(f"  Using ONNX graph: {main_onnx.name} "
             f"({main_onnx.stat().st_size / 1e6:.1f} MB)")

    model_out = assets_dir / f"mms_tts_{lang}.onnx"
    shutil.copy(main_onnx, model_out)

    # Copy the character vocab so the app can map text -> token ids on-device.
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache)
    vocab_out = assets_dir / f"mms_tts_{lang}_vocab.json"
    with open(vocab_out, "w", encoding="utf-8") as f:
        json.dump(tok.get_vocab(), f, ensure_ascii=False)
    log.info(f"  Wrote {vocab_out.name} ({len(tok.get_vocab())} tokens)")

    # Copy config.json so the app reads the model's true sampling_rate for the
    # WAV header (defaults to 16 kHz if absent). Prefer the HF cache snapshot
    # (guaranteed to exist) over the export temp dir.
    src_config = None
    cache_cfg = list(Path(hf_cache).glob(
        f"models--{model_id.replace('/', '--')}/snapshots/*/config.json"))
    if cache_cfg:
        src_config = cache_cfg[0]
    elif (temp_dir / "config.json").exists():
        src_config = temp_dir / "config.json"
    if src_config:
        shutil.copy(src_config, assets_dir / f"mms_tts_{lang}_config.json")
        log.info(f"  Wrote mms_tts_{lang}_config.json (from {src_config.name})")
    else:
        log.warning(f"  No config.json found for {lang}; WAV header defaults to 16 kHz")

    shutil.rmtree(temp_dir, ignore_errors=True)

    if verify:
        _verify_onnx(model_out)

    log.info(f"[Done] {lang} -> {model_out.name}")
    return model_out


def _verify_onnx(onnx_path: Path) -> None:
    """Smoke-test the exported model with ONNX Runtime."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        log.warning("--verify requires numpy and onnxruntime; skipping verification")
        return

    log.info(f"  Verifying {onnx_path.name} ...")
    sess = ort.InferenceSession(str(onnx_path))
    inputs = sess.get_inputs()
    log.info(f"    inputs:  {[(i.name, i.shape, i.type) for i in inputs]}")
    log.info(f"    outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

    # Feed a minimal dummy input. VITS ONNX exports accept `input_ids`
    # (int64) and, depending on the exporter, `attention_mask`.
    feed = {}
    for i in inputs:
        if i.name == "input_ids":
            feed[i.name] = np.array([[1, 2, 3]], dtype=np.int64)
        elif i.name == "attention_mask":
            feed[i.name] = np.ones((1, 3), dtype=np.int64)
    out = sess.run(None, feed)[0]
    log.info(f"    output[0] shape: {out.shape}  dtype: {out.dtype}")
    if out.size == 0:
        raise RuntimeError(f"Verification failed for {onnx_path.name}: empty output")
    log.info("    Verification OK")


def main():
    parser = argparse.ArgumentParser(
        description="Export MMS-TTS models to ONNX for the Android app")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR,
                        help="Android assets directory (default: android/app/src/main/assets)")
    parser.add_argument("--langs", nargs="+", choices=sorted(MMS_TTS_MODELS),
                        default=sorted(MMS_TTS_MODELS),
                        help="Languages to export (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List supported languages and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Run a quick ONNX Runtime smoke test after export")
    args = parser.parse_args()

    if args.list:
        for code, mid in sorted(MMS_TTS_MODELS.items()):
            print(f"  {code:5s} -> {mid}")
        return

    results = {}
    for lang in args.langs:
        try:
            results[lang] = export_language(lang, MMS_TTS_MODELS[lang],
                                            args.assets_dir, args.verify)
        except Exception as e:
            log.error(f"FAILED to export {lang}: {e}")

    print("\n=== Summary ===")
    for lang, path in results.items():
        print(f"  {lang}: {path}")
    failed = set(args.langs) - set(results)
    if failed:
        print(f"  FAILED: {sorted(failed)}")
        sys.exit(1)

    log.info(f"All done. Models are in {args.assets_dir}")
    log.info("Re-build the Android app so the assets are bundled "
             "(Build > Make Project).")


if __name__ == "__main__":
    main()
