# OmniVoice — Model Export & Download Script
#
# Automates the setup of ONNX models for the Android app.
# 1. Downloads NLLB-200 quantized models from Hugging Face.
# 2. Exports Whisper Small to ONNX using Optimum.
# 3. Places all files directly into the Android assets directory.

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Target directory: Android Assets
ASSETS_DIR = Path(__file__).resolve().parent.parent / "android" / "app" / "src" / "main" / "assets"

# URLs for NLLB-200 (Xenova's quantized versions)
NLLB_ENCODER_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/encoder_model_int8.onnx?download=true"
NLLB_DECODER_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/decoder_model_merged_int8.onnx?download=true"
NLLB_VOCAB_URL = "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece_bpe.model?download=true"


def download_file(url: str, output_path: Path):
    """Download a file with progress logging."""
    if output_path.exists():
        log.info(f"  [Skip] {output_path.name} already exists.")
        return

    log.info(f"  [Download] {url} -> {output_path.name}")
    try:
        def progress(count, block_size, total_size):
            if total_size <= 0: return
            percent = int(count * block_size * 100 / total_size)
            if percent % 25 == 0:  # Log every 25%
                sys.stdout.write(f"\r    Progress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(output_path), reporthook=progress)
        sys.stdout.write("\n")
        log.info(f"  [Done] Saved {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log.error(f"  [Error] Failed to download {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()


def setup_nllb(assets_dir: Path):
    """Download pre-quantized NLLB-200 models."""
    log.info("Setting up NLLB-200 models...")
    assets_dir.mkdir(parents=True, exist_ok=True)

    download_file(NLLB_ENCODER_URL, assets_dir / "encoder_model_int8.onnx")
    download_file(NLLB_DECODER_URL, assets_dir / "decoder_model_merged_int8.onnx")
    download_file(NLLB_VOCAB_URL, assets_dir / "sentencepiece_bpe.model")


def export_whisper(assets_dir: Path):
    """Export Whisper Small using Optimum and move to assets."""
    log.info("Exporting Whisper Small to ONNX...")
    
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    except ImportError:
        log.error("Install `optimum` and `onnxruntime`: pip install optimum onnxruntime")
        return

    temp_dir = Path("temp_whisper_export")
    temp_dir.mkdir(exist_ok=True)

    log.info("  Converting Whisper Small via Optimum (this may take a while)...")
    model = ORTModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small", export=True)
    model.save_pretrained(str(temp_dir))

    # Move and rename specific files
    log.info("  Moving Whisper models to assets...")
    shutil.move(str(temp_dir / "encoder_model.onnx"), str(assets_dir / "whisper_encoder.onnx"))
    shutil.move(str(temp_dir / "decoder_model.onnx"), str(assets_dir / "whisper_decoder.onnx"))

    # Cleanup
    shutil.rmtree(temp_dir)
    log.info(f"  [Done] Whisper models saved to {assets_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download or Export ONNX models to Android assets")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["nllb", "whisper", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    models = args.models if "all" not in args.models else ["nllb", "whisper"]

    if "nllb" in models:
        setup_nllb(args.assets_dir)
    if "whisper" in models:
        export_whisper(args.assets_dir)

    log.info("Workflow complete. Large models are now in assets/.")
    log.info("IMPORTANT: Add *.onnx and *.model to your .gitignore before pushing!")


if __name__ == "__main__":
    main()
