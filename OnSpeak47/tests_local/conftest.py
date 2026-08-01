import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

_results: list[dict] = []


@pytest.fixture(scope="session")
def translator():
    from backend.translation_nllb import NLLBTranslator

    return NLLBTranslator()


@pytest.fixture(scope="session")
def parallel_data():
    path = DATA_DIR / "parallel_sentences.json"
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    required = {"en", "vi", "zh_hans"}
    has_hant = ["zh_hant" in row for row in rows]
    if any(has_hant) and not all(has_hant):
        raise ValueError(
            "parallel_sentences.json: 'zh_hant' must be present on every row or none — "
            "found a mix, which is ambiguous for scoring."
        )

    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"parallel_sentences.json row {row.get('id', '?')} missing fields: {missing}")

    return rows


@pytest.fixture(scope="session")
def results_recorder():
    def _record(direction: str, n: int, bleu: float, tokenizer: str):
        _results.append({"direction": direction, "n": n, "bleu": bleu, "tokenizer": tokenizer})

    return _record


@pytest.fixture(scope="session")
def translations_dumper():
    def _dump(direction: str, rows: list[dict], src_key: str, tgt_key: str, hypotheses: list[str]):
        OUTPUT_DIR.mkdir(exist_ok=True)
        records = [
            {
                "id": row.get("id"),
                "src": row[src_key],
                "hypothesis": hyp,
                "reference": row[tgt_key],
            }
            for row, hyp in zip(rows, hypotheses)
        ]
        path = OUTPUT_DIR / f"translations_{direction.replace('->', '_to_')}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    return _dump


def pytest_sessionfinish(session, exitstatus):
    if not _results:
        return

    DOCS_DIR.mkdir(exist_ok=True)
    out_path = DOCS_DIR / "translation_quality_results.md"

    lines = [
        "# NLLB-200 Translation Quality Results",
        "",
        f"Model: `facebook/nllb-200-distilled-600M`",
        f"Run date: {datetime.date.today().isoformat()}",
        "Test sentences: `tests_local/data/parallel_sentences.json`",
        "",
        "| Direction | N | BLEU | Tokenizer |",
        "|---|---|---|---|",
    ]
    for r in _results:
        lines.append(f"| {r['direction']} | {r['n']} | {r['bleu']:.2f} | {r['tokenizer']} |")

    by_direction = {r["direction"]: r["bleu"] for r in _results}
    vn_en = [v for k, v in by_direction.items() if {"vi", "en"} == set(k.split("->"))]
    vn_cn = [v for k, v in by_direction.items() if "vi" in k and ("zh_hans" in k or "zh_hant" in k)]

    lines += ["", "## Risk #1 check — VN↔CN vs VN↔EN quality", ""]
    if vn_en and vn_cn:
        lines.append(f"- Average VN↔EN BLEU: {sum(vn_en) / len(vn_en):.2f}")
        lines.append(f"- Average VN↔CN BLEU: {sum(vn_cn) / len(vn_cn):.2f}")
        gap = sum(vn_en) / len(vn_en) - sum(vn_cn) / len(vn_cn)
        lines.append(f"- Gap (EN better by): {gap:.2f} BLEU")
    else:
        lines.append("- Not enough directions scored this run to compare.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
