# Standardized input/output for NLLB-200

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Simple language keys used across the pipeline, mapped to NLLB's FLORES-200 codes.
# Scope per pipeline_overview.md: VN<->EN and VN<->CN only.
LANG_CODES = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "zh_hans": "zho_Hans",
    "zh_hant": "zho_Hant",
    # Convenience aliases so callers can pass "zh" without worrying about script
    "zh": "zho_Hans",
}


class NLLBTranslator:
    """NLLB-200-distilled-600M translation wrapper.

    Handles single-sentence and batch translation for VN↔EN and VN↔CN,
    plus file-level (corpus) translation for BLEU evaluation.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    # ------------------------------------------------------------------
    # Core: single sentence
    # ------------------------------------------------------------------

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate a single string.

        Parameters
        ----------
        text : str
            Source text.
        src_lang, tgt_lang : str
            Keys in ``LANG_CODES`` (e.g. ``"vi"``, ``"en"``, ``"zh_hans"``).
        """
        self.tokenizer.src_lang = LANG_CODES[src_lang]
        inputs = self.tokenizer(text, return_tensors="pt")
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt_lang])
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=5,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    # ------------------------------------------------------------------
    # Batch translation
    # ------------------------------------------------------------------

    def translate_batch(
        self,
        texts: Sequence[str],
        src_lang: str,
        tgt_lang: str,
        *,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> list[str]:
        """Translate a list of strings in mini-batches.

        Uses the same generation parameters as ``translate()`` but processes
        multiple sentences at once for throughput on GPU or multi-core CPU.
        """
        self.tokenizer.src_lang = LANG_CODES[src_lang]
        forced_bos = self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt_lang])

        results: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=5,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend(decoded)
        return results

    # ------------------------------------------------------------------
    # File / corpus-level translation (for BLEU evaluation)
    # ------------------------------------------------------------------

    def translate_file(
        self,
        src_path: str | Path,
        tgt_path: str | Path,
        src_lang: str,
        tgt_lang: str,
        *,
        max_lines: int | None = None,
        batch_size: int = 16,
    ) -> list[str]:
        """Read a line-aligned source file, translate, write to *tgt_path*.

        Parameters
        ----------
        src_path : str | Path
            One sentence per line (UTF-8).
        tgt_path : str | Path
            Output file (one hypothesis per line).
        max_lines : int | None
            Limit for quick experiments.  ``None`` → full file.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        list[str]
            The translated hypotheses.
        """
        src_path = Path(src_path)
        tgt_path = Path(tgt_path)

        with open(src_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if max_lines is not None:
            lines = lines[:max_lines]

        hypotheses = self.translate_batch(lines, src_lang, tgt_lang, batch_size=batch_size)

        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tgt_path, "w", encoding="utf-8") as f:
            for h in hypotheses:
                f.write(h + "\n")

        return hypotheses

    # ------------------------------------------------------------------
    # Pivot translation (vi → en → zh or zh → en → vi)
    # ------------------------------------------------------------------

    def translate_pivot(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        pivot_lang: str = "en",
    ) -> str:
        """Two-hop translation via a pivot language.

        Useful as a quality comparison baseline for vi↔zh where direct NLLB
        quality may be lower than going through English.
        """
        intermediate = self.translate(text, src_lang, pivot_lang)
        return self.translate(intermediate, pivot_lang, tgt_lang)

    # ------------------------------------------------------------------
    # Chinese-specific pre-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_chinese(text: str) -> str:
        """Light pre-processing for Chinese input text.

        Ensures consistent whitespace handling — Chinese text fed to NLLB
        should not have random spaces between characters, which can confuse
        SentencePiece tokenization.
        """
        # Remove spaces between CJK characters while preserving spaces
        # around Latin/ASCII words.
        import re
        # Remove spaces between two CJK characters
        text = re.sub(
            r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])",
            r"\1\2",
            text,
        )
        return text.strip()
