# OmniVoice — Model Export & Download Script
#
# Automates the setup of ONNX models for the Android app.
# 1. Downloads NLLB-200 quantized models from Hugging Face.
# 2. Exports Whisper Small to ONNX using Optimum.
# 3. Places all files directly into the Android assets directory.

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Target directory: Output to a dedicated folder on D:
ASSETS_DIR = Path("D:/StudioProjects/OmniVoice/onnx_models")

# URLs for NLLB-200 (Xenova's quantized versions)
NLLB_ENCODER_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/encoder_model_int8.onnx?download=true"
NLLB_DECODER_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/decoder_model_merged_int8.onnx?download=true"

# SentencePiece tokenizer from the official facebook/nllb-200-distilled-600M checkpoint.
# Xenova's ONNX export does not bundle the .model file, so we pull it from the
# original repo. The app expects the asset name `sentencepiece_bpe.model`
# (see TranslationModule.VOCAB_FILE), while HF names it `sentencepiece.bpe.model`
# (dots) — always write it under the underscore name.
NLLB_TOKENIZER_URL = "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model"
NLLB_TOKENIZER_ASSET_NAME = "sentencepiece_bpe.model"

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


def setup_nllb_tokenizer(assets_dir: Path):
    """Download the NLLB-200 SentencePiece model and save it under the
    asset name the app expects.

    ``02_prune_vocab.py`` only prunes embedding rows for language control
    tokens — it never regenerates or modifies the SentencePiece ``.model``
    file itself. The tokenizer is therefore identical between the original
    facebook/nllb-200-distilled-600M checkpoint and any pruned variant, so
    it can be exported verbatim here.
    """
    target = assets_dir / NLLB_TOKENIZER_ASSET_NAME
    if target.exists() and target.stat().st_size > 0:
        log.info(f"  [Skip] {target.name} already exists.")
        return

    tmp_target = assets_dir / (NLLB_TOKENIZER_ASSET_NAME + ".tmp")
    try:
        download_file(NLLB_TOKENIZER_URL, tmp_target)
        if not tmp_target.exists():
            log.error(f"  [Error] Failed to download {NLLB_TOKENIZER_ASSET_NAME}")
            return
        # Atomic-ish rename: tmp_target -> final name (handles 'dots' vs
        # 'underscores' naming mismatch and removes the .tmp suffix).
        if target.exists():
            target.unlink()
        tmp_target.rename(target)
        log.info(f"  [Done] Exported NLLB tokenizer -> {target.name} "
                 f"({target.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log.error(f"  [Error] Failed to export NLLB tokenizer: {e}")
        if tmp_target.exists():
            tmp_target.unlink()


def setup_nllb(assets_dir: Path):
    """Download pre-quantized NLLB-200 models."""
    log.info("Setting up NLLB-200 models...")
    assets_dir.mkdir(parents=True, exist_ok=True)

    download_file(NLLB_ENCODER_URL, assets_dir / "encoder_model_int8.onnx")
    download_file(NLLB_DECODER_URL, assets_dir / "decoder_model_merged_int8.onnx")

    # The ONNX weights are useless without the SentencePiece tokenizer that
    # produced their inputs.  Export it alongside the weights so the app has
    # everything it needs after a single run of this script.
    setup_nllb_tokenizer(assets_dir)

def export_whisper(assets_dir: Path):
    """Export Whisper Small using Optimum and move to assets."""
    log.info("Exporting Whisper Small to ONNX...")
    
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    except ImportError:
        log.error("Install `optimum` and `onnxruntime`: pip install optimum onnxruntime")
        return

    temp_dir = assets_dir / "temp_whisper_export"
    temp_dir.mkdir(parents=True, exist_ok=True)

    log.info("  Converting Whisper Small via Optimum (this may take a while)...")
    # use_cache=True so the exported decoder carries past_key_values/present
    # (KV-cache) inputs/outputs. Without this, autoregressive decoding on-device
    # has to reprocess the entire growing token sequence every step (O(n^2))
    # instead of just the newest token — see ASRModule.runDecoder() on the Java side.
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-small",
        export=True,
        use_cache=True,
        cache_dir=str(assets_dir / "hf_cache"),
    )
    model.save_pretrained(str(temp_dir))

    # Move and rename specific files
    log.info("  Moving Whisper models to assets...")
    shutil.move(str(temp_dir / "encoder_model.onnx"), str(assets_dir / "whisper_encoder.onnx"))

    # Depending on the installed Optimum version, the cache-enabled decoder can
    # come out as a single merged graph (preferred — has a use_cache_branch input,
    # same pattern as decoder_model_merged_int8.onnx for NLLB) or as a separate
    # "with past" file alongside the cache-less decoder_model.onnx. Prefer the
    # merged file, then with-past, and only fall back to the plain decoder (no
    # cache — ASRModule will auto-detect this and use the slower loop) if
    # neither cache-enabled file was produced.
    decoder_candidates = [
        "decoder_model_merged.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    ]
    for name in decoder_candidates:
        candidate = temp_dir / name
        if candidate.exists():
            if name == "decoder_model.onnx":
                log.warning(
                    "  No cache-enabled decoder file found (checked %s) — "
                    "falling back to decoder_model.onnx (no KV-cache, slower on-device).",
                    decoder_candidates[:-1],
                )
            shutil.move(str(candidate), str(assets_dir / "whisper_decoder.onnx"))
            break
    else:
        log.error(f"  [Error] No decoder ONNX file found among {decoder_candidates} in {temp_dir}")

    # Cleanup
    shutil.rmtree(temp_dir)
    log.info(f"  [Done] Whisper models saved to {assets_dir}")


def export_whisper_processing(assets_dir: Path):
    """Export Whisper's pre-processing (raw audio bytes -> log-mel features)
    and post-processing (token ids -> text) as their own ONNX graphs, using
    onnxruntime-extensions. This replaces two things that previously had to
    be hand-implemented in Java and were left as stubs:
      - ASRModule.extractMelFeatures() (was returning an all-zero array —
        FFT + mel-filterbank is exactly what USE_ONNX_STFT does here)
      - a Whisper-specific BPE detokenizer (was returning a token-count
        placeholder string instead of real text)
    Both stubs can be replaced by just running these two extra ONNX sessions
    from Java instead of hand-writing DSP/BPE code that's hard to verify.
    """
    log.info("Exporting Whisper pre/post-processing graphs (onnxruntime-extensions)...")
    try:
        import onnx
        from transformers import WhisperProcessor
        from onnxruntime_extensions.cvt import gen_processing_models

        # HACK: Patch onnx.compose.merge_models to ignore IR version mismatch.
        # This is required on Python 3.12/Windows where Torch exports IR 10 
        # but extensions internally use IR 8.
        _original_merge = onnx.compose.merge_models
        def _patched_merge(m1, m2, *args, **kwargs):
            m1.ir_version = 8
            m2.ir_version = 8
            return _original_merge(m1, m2, *args, **kwargs)
        onnx.compose.merge_models = _patched_merge

    except ImportError:
        log.error(
            "Install `onnxruntime-extensions` and `transformers`: "
            "pip install onnxruntime-extensions transformers"
        )
        return

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        cache_dir=str(assets_dir / "hf_cache")
    )

    # USE_AUDIO_DECODER=False: The graph expects raw float samples (16kHz).
    # We decode the WAV file manually in Java to avoid AudioDecoder op errors on Android.
    # USE_ONNX_STFT=False: Tắt để tránh lỗi "Cannot find STFTNorm" trên môi trường
    # Windows/Python 3.12, vẫn đảm bảo tính đúng đắn trên Android.
    pre_m, post_m = gen_processing_models(
        processor,
        pre_kwargs={"USE_AUDIO_DECODER": False, "USE_ONNX_STFT": False},
        post_kwargs={},
        opset=17,
    )

    # Restore original merge function
    onnx.compose.merge_models = _original_merge

    pre_path = assets_dir / "whisper_preprocess.onnx"
    post_path = assets_dir / "whisper_postprocess.onnx"
    onnx.save(pre_m, str(pre_path))
    onnx.save(post_m, str(post_path))

    # post_m expects token ids shaped the way the fused WhisperBeamSearch
    # contrib op emits them, which a custom greedy-decode loop (like
    # ASRModule.runDecoder()) doesn't produce. Detokenizing (id -> text) is a
    # simple, stable, well-documented algorithm — unlike encoding, it doesn't
    # need the merge-rank tables — so it's safer to hand-implement it natively
    # in Java against a plain vocab.json than to guess post_m's exact expected
    # input shape. Dump that vocab here; post_m is still saved above in case
    # you want it for the fused pipeline later.
    vocab_path = assets_dir / "whisper_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(processor.tokenizer.get_vocab(), f, ensure_ascii=False)

    # Print the actual graph I/O names rather than assuming them — the exact
    # names can vary by onnxruntime-extensions version, and ASRModule.java
    # looks these up dynamically via session.getInputNames()/getOutputNames()
    # rather than hardcoding a guess, but it's worth confirming they look sane.
    log.info(f"  [Done] {pre_path.name}: inputs={[i.name for i in pre_m.graph.input]} "
             f"outputs={[o.name for o in pre_m.graph.output]}")
    log.info(f"  [Done] {vocab_path.name}: {len(processor.tokenizer.get_vocab())} tokens "
             "(used by ASRModule's native Java BPE decoder)")
    log.info(f"  [Info] {post_path.name} saved but not currently used by ASRModule.java — "
             "it expects WhisperBeamSearch-shaped input, see comment above.")


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
        export_whisper_processing(args.assets_dir)

    log.info("Workflow complete. Large models are now in assets/.")
    log.info("IMPORTANT: Add *.onnx and *.model to your .gitignore before pushing!")


if __name__ == "__main__":
    main()
