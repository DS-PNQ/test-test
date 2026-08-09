# Script to prune NLLB-200 vocabulary to VN/EN/CN
#
# NLLB-200 covers 200 languages.  For the Omni Voice pipeline we only need
# VN, EN, and CN (Hans + Hant).  Pruning the embedding matrix and
# SentencePiece model reduces:
#   - Model size on disk (important for mobile)
#   - Memory at runtime
#   - Potentially improves CN translation quality by removing conflicting tokens

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# NLLB FLORES-200 language codes we want to KEEP.
KEEP_LANGUAGES = {
    "eng_Latn",
    "vie_Latn",
    "zho_Hans",
    "zho_Hant",
}

# Special tokens that must always be kept regardless of language.
SPECIAL_TOKENS = {
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "<mask>",
}


def prune_vocab(
    model_dir: Path,
    output_dir: Path,
    keep_languages: set[str] | None = None,
):
    """Prune NLLB-200 vocabulary to only the target languages.

    Parameters
    ----------
    model_dir : Path
        Directory containing the HuggingFace NLLB model (or ONNX export).
    output_dir : Path
        Where to write the pruned model.
    keep_languages : set[str]
        FLORES-200 codes to keep.  Defaults to VN/EN/CN.
    """
    if keep_languages is None:
        keep_languages = KEEP_LANGUAGES

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Pruning vocabulary to languages: {keep_languages}")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = str(model_dir) if model_dir.exists() else "facebook/nllb-200-distilled-600M"
    cache_dir = output_dir.parent / "hf_cache"

    log.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))

    # Identify which language tokens to keep vs remove
    all_lang_tokens = [t for t in tokenizer.additional_special_tokens if t not in SPECIAL_TOKENS]
    keep_lang_tokens = {t for t in all_lang_tokens if t in keep_languages}
    remove_lang_tokens = set(all_lang_tokens) - keep_lang_tokens

    log.info(f"  Total language tokens: {len(all_lang_tokens)}")
    log.info(f"  Keeping: {len(keep_lang_tokens)} → {keep_lang_tokens}")
    log.info(f"  Removing: {len(remove_lang_tokens)}")

    # Identify token IDs to keep
    # Note: We keep ALL SentencePiece subword tokens — only language control
    # tokens are pruned.  This is the safest approach since Chinese uses
    # many of the same subword tokens as other CJK languages.
    remove_ids = set()
    for token in remove_lang_tokens:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id:
            remove_ids.add(tid)

    log.info(f"  Token IDs to remove: {len(remove_ids)}")

    # Load model and prune embeddings
    log.info("Loading model for embedding pruning...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(cache_dir))

    vocab_size_before = model.config.vocab_size
    embed_dim = model.config.d_model

    # Get current embedding weights
    shared_embed = model.model.shared.weight.data  # [vocab_size, embed_dim]

    # Build mapping: old_id → new_id (skipping removed IDs)
    keep_ids = sorted(set(range(vocab_size_before)) - remove_ids)
    old_to_new = {old: new for new, old in enumerate(keep_ids)}

    # Create new embedding matrix
    new_embed = shared_embed[keep_ids].clone()
    vocab_size_after = len(keep_ids)

    log.info(f"  Vocabulary: {vocab_size_before} → {vocab_size_after} "
             f"({vocab_size_before - vocab_size_after} tokens removed)")
    log.info(f"  Embedding size: {shared_embed.shape} → {new_embed.shape}")

    # ------------------------------------------------------------------
    # Apply pruned embeddings to the model
    # ------------------------------------------------------------------
    import torch.nn as nn

    # Replace shared embedding
    new_shared = nn.Embedding(vocab_size_after, embed_dim)
    new_shared.weight.data = new_embed
    model.model.shared = new_shared

    # Encoder and decoder input embeddings point to shared
    model.model.encoder.embed_tokens = new_shared
    model.model.decoder.embed_tokens = new_shared

    # Replace LM head (output projection)
    old_lm_head = model.lm_head.weight.data  # [vocab_size, embed_dim]
    new_lm_head_weight = old_lm_head[keep_ids].clone()
    new_lm_head = nn.Linear(embed_dim, vocab_size_after, bias=False)
    new_lm_head.weight.data = new_lm_head_weight
    model.lm_head = new_lm_head

    # Update config
    model.config.vocab_size = vocab_size_after

    log.info("  Replaced shared embedding, encoder/decoder embed_tokens, and lm_head")

    # ------------------------------------------------------------------
    # Save pruned model and tokenizer
    # ------------------------------------------------------------------
    model.save_pretrained(str(output_dir))
    log.info(f"  Pruned model saved to {output_dir}")

    # Save pruning metadata
    pruning_meta = {
        "original_model": model_name,
        "keep_languages": sorted(keep_languages),
        "vocab_size_before": vocab_size_before,
        "vocab_size_after": vocab_size_after,
        "tokens_removed": vocab_size_before - vocab_size_after,
        "old_to_new_mapping_sample": {str(k): v for k, v in list(old_to_new.items())[:20]},
    }
    meta_path = output_dir / "pruning_metadata.json"
    meta_path.write_text(json.dumps(pruning_meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Export language_token_map.json for Android Tokenizer
    # ------------------------------------------------------------------
    # The Android Tokenizer needs to know the NEW HF token IDs for each
    # kept language after pruning (the old_to_new mapping shifts them).
    lang_token_map = {}
    for lang_code in sorted(keep_languages):
        old_id = tokenizer.convert_tokens_to_ids(lang_code)
        if old_id in old_to_new:
            lang_token_map[lang_code] = old_to_new[old_id]
            log.info(f"  {lang_code}: {old_id} → {old_to_new[old_id]}")
        else:
            log.warning(f"  {lang_code}: old ID {old_id} not found in kept IDs!")

    lang_map_path = output_dir / "language_token_map.json"
    lang_map_path.write_text(json.dumps(lang_token_map, indent=2), encoding="utf-8")
    log.info(f"Language token map saved to {lang_map_path}")

    # Save the tokenizer with updated special tokens
    new_special_tokens = sorted(keep_lang_tokens) + list(SPECIAL_TOKENS)
    tokenizer.additional_special_tokens = sorted(keep_lang_tokens)
    tokenizer.save_pretrained(str(output_dir))

    log.info(f"Pruned model metadata saved to {meta_path}")
    log.info(f"Tokenizer saved to {output_dir}")

    # Size estimation
    original_mb = shared_embed.numel() * 4 / 1e6  # float32
    pruned_mb = new_embed.numel() * 4 / 1e6
    log.info(f"  Embedding size: {original_mb:.1f} MB → {pruned_mb:.1f} MB "
             f"(saved {original_mb - pruned_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Prune NLLB-200 vocabulary to VN/EN/CN")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("facebook/nllb-200-distilled-600M"),
        help="Path to HuggingFace model directory or model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "onnx_models" / "nllb_pruned",
    )
    args = parser.parse_args()

    prune_vocab(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
