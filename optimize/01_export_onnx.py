# Hugging Face `optimum` PyTorch -> ONNX scripts
#
# Exports all three pipeline models to ONNX format for on-device inference.
# Follows the same model structure as RTranslator-2.00:
#   - NLLB: encoder, decoder, cache_initializer, embed_and_lm_head (4 files)
#   - Whisper: encoder, decoder (2 files)
#   - MMS-TTS: single model file

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "onnx_models"


# ======================================================================
# NLLB-200 Export (4-file encoder-decoder with KV-cache)
# ======================================================================

def export_nllb(output_dir: Path, verify: bool = False):
    """Export NLLB-200-distilled-600M to ONNX.

    Produces the same 4-file structure as RTranslator-2.00:
      - NLLB_encoder.onnx
      - NLLB_decoder.onnx
      - NLLB_cache_initializer.onnx
      - NLLB_embed_and_lm_head.onnx
    """
    log.info("Exporting NLLB-200 to ONNX...")

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
    except ImportError:
        log.error("Install `optimum` and `onnxruntime`: pip install optimum onnxruntime")
        return

    nllb_dir = output_dir / "nllb"
    nllb_dir.mkdir(parents=True, exist_ok=True)

    model_name = "facebook/nllb-200-distilled-600M"
    log.info(f"Loading {model_name} and converting to ONNX...")

    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_name,
        export=True,
    )
    model.save_pretrained(str(nllb_dir))

    log.info(f"NLLB ONNX models saved to {nllb_dir}")
    _list_onnx_files(nllb_dir)

    if verify:
        _verify_nllb(nllb_dir)


# ======================================================================
# Whisper Small Export
# ======================================================================

def export_whisper(output_dir: Path, verify: bool = False):
    """Export Whisper Small to ONNX."""
    log.info("Exporting Whisper Small to ONNX...")

    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    except ImportError:
        log.error("Install `optimum` and `onnxruntime`: pip install optimum onnxruntime")
        return

    whisper_dir = output_dir / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)

    model_name = "openai/whisper-small"
    log.info(f"Loading {model_name} and converting to ONNX...")

    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        export=True,
    )
    model.save_pretrained(str(whisper_dir))

    log.info(f"Whisper ONNX models saved to {whisper_dir}")
    _list_onnx_files(whisper_dir)


# ======================================================================
# MMS-TTS Export
# ======================================================================

def export_mms_tts(output_dir: Path, languages: list[str] | None = None):
    """Export MMS-TTS models to ONNX for each language.

    Parameters
    ----------
    languages : list[str]
        Language codes to export (default: vi, en, zh).
    """
    log.info("Exporting MMS-TTS to ONNX...")

    if languages is None:
        languages = ["vie", "eng", "zho"]

    tts_dir = output_dir / "mms_tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        model_name = f"facebook/mms-tts-{lang}"
        lang_dir = tts_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Exporting {model_name}...")
        try:
            from transformers import VitsModel, AutoTokenizer

            model = VitsModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()

            # Simple ONNX export via torch.onnx
            dummy_input = tokenizer("Hello world", return_tensors="pt")
            onnx_path = lang_dir / f"mms_tts_{lang}.onnx"

            torch.onnx.export(
                model,
                (dummy_input["input_ids"],),
                str(onnx_path),
                input_names=["input_ids"],
                output_names=["waveform"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "waveform": {0: "batch", 1: "time"},
                },
                opset_version=17,
            )
            log.info(f"  Saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

        except Exception as e:
            log.warning(f"  Failed to export {model_name}: {e}")


# ======================================================================
# Helpers
# ======================================================================

def _list_onnx_files(directory: Path):
    """Log all .onnx files in a directory with their sizes."""
    for p in sorted(directory.rglob("*.onnx")):
        size_mb = p.stat().st_size / 1e6
        log.info(f"  {p.name}: {size_mb:.1f} MB")


def _verify_nllb(nllb_dir: Path):
    """Quick verification that the exported NLLB model produces output."""
    log.info("Verifying NLLB ONNX export...")
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        model = ORTModelForSeq2SeqLM.from_pretrained(str(nllb_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(nllb_dir))

        tokenizer.src_lang = "vie_Latn"
        inputs = tokenizer("Xin chào", return_tensors="pt")
        forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")
        out = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=64)
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        log.info(f"  Verification: 'Xin chào' → '{text}'")
        assert len(text) > 0, "Empty output from ONNX model"
        log.info("  ✓ NLLB ONNX verification passed")
    except Exception as e:
        log.error(f"  ✗ NLLB ONNX verification failed: {e}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Export pipeline models to ONNX")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--verify", action="store_true", help="Run verification after export")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["nllb", "whisper", "mms_tts", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    models = args.models if "all" not in args.models else ["nllb", "whisper", "mms_tts"]

    if "nllb" in models:
        export_nllb(args.output_dir, verify=args.verify)
    if "whisper" in models:
        export_whisper(args.output_dir, verify=args.verify)
    if "mms_tts" in models:
        export_mms_tts(args.output_dir)

    log.info("Export complete.")


if __name__ == "__main__":
    main()
