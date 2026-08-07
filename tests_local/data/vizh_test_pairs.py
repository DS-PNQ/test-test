"""
Convert raw line-aligned VI↔ZH test/train files into evaluation-ready format.

Source files (one sentence per line, aligned by line number):
  - test vi-zh/train2022.vi   (Vietnamese)
  - test vi-zh/train2022.zh   (Simplified Chinese)

Usage:
    python tests_local/data/vizh_test_pairs.py --max-pairs 1000 --output pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_TEST_DIR = Path(os.environ.get("VIZH_TEST_DIR", "test vi-zh"))
DEFAULT_TRAIN_DIR = Path(os.environ.get("VIZH_TRAIN_DIR", "train vi-zh"))


def load_parallel_corpus(
    vi_path: Path,
    zh_path: Path,
    max_pairs: int | None = None,
) -> list[dict]:
    """Load line-aligned VI↔ZH parallel corpus.

    Parameters
    ----------
    vi_path : Path
        Vietnamese sentences (one per line, UTF-8).
    zh_path : Path
        Chinese sentences (aligned with Vietnamese by line number).
    max_pairs : int | None
        Maximum number of pairs to load.  None = all.

    Returns
    -------
    list[dict]
        List of {"id": "vizh-001", "vi": "...", "zh_hans": "..."} dicts.
    """
    with open(vi_path, encoding="utf-8") as f:
        vi_lines = [line.strip() for line in f if line.strip()]
    with open(zh_path, encoding="utf-8") as f:
        zh_lines = [line.strip() for line in f if line.strip()]

    n = min(len(vi_lines), len(zh_lines))
    if max_pairs is not None:
        n = min(n, max_pairs)

    pairs = []
    for i in range(n):
        pairs.append({
            "id": f"vizh-{i + 1:04d}",
            "vi": vi_lines[i],
            "zh_hans": zh_lines[i],
        })

    return pairs


def corpus_stats(pairs: list[dict]) -> dict:
    """Compute basic statistics about the corpus."""
    vi_lengths = [len(p["vi"]) for p in pairs]
    zh_lengths = [len(p["zh_hans"]) for p in pairs]
    vi_words = [len(p["vi"].split()) for p in pairs]

    return {
        "num_pairs": len(pairs),
        "vi_avg_chars": round(sum(vi_lengths) / len(vi_lengths), 1),
        "zh_avg_chars": round(sum(zh_lengths) / len(zh_lengths), 1),
        "vi_avg_words": round(sum(vi_words) / len(vi_words), 1),
        "vi_max_chars": max(vi_lengths),
        "zh_max_chars": max(zh_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Load VI↔ZH parallel corpus")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source", choices=["test", "train"], default="test")
    args = parser.parse_args()

    data_dir = args.test_dir if args.source == "test" else args.train_dir
    vi_path = data_dir / "train2022.vi"
    zh_path = data_dir / "train2022.zh"

    if not vi_path.exists():
        print(f"ERROR: Vietnamese file not found: {vi_path}")
        return
    if not zh_path.exists():
        print(f"ERROR: Chinese file not found: {zh_path}")
        return

    pairs = load_parallel_corpus(vi_path, zh_path, args.max_pairs)
    stats = corpus_stats(pairs)

    print(f"Loaded {stats['num_pairs']} VI↔ZH pairs from {data_dir.name}")
    print(f"  VI: avg {stats['vi_avg_chars']} chars, {stats['vi_avg_words']} words, max {stats['vi_max_chars']} chars")
    print(f"  ZH: avg {stats['zh_avg_chars']} chars, max {stats['zh_max_chars']} chars")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(pairs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved to {args.output}")

    # Print first 3 samples
    print("\nSample pairs:")
    for p in pairs[:3]:
        print(f"  [{p['id']}]")
        print(f"    VI: {p['vi'][:80]}...")
        print(f"    ZH: {p['zh_hans'][:40]}...")


if __name__ == "__main__":
    main()
