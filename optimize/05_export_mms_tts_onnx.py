# OmniVoice — Export MMS-TTS (VITS) checkpoints to ONNX
#
# Exports the MMS-TTS checkpoints (vi/en) to ONNX graphs so the Android
# app can run neural TTS on-device with ONNX Runtime instead of the Android
# system TTS fallback (Chinese has no published MMS checkpoint — see below).
#
# Why manual torch.onnx.export instead of Optimum: Optimum's VITS export
# exposes an extra "attention_mask" graph input (the Android side feeds only
# "input_ids") and fails its own output-shape validation for MMS-TTS
# checkpoints. Traced export with dynamo=False (TorchScript path) produces a
# graph with exactly one input ("input_ids", int64, [1, seq]) and one output
# ("waveform", float32, [1, n_samples]) — exactly what MmsOnnxTTS.java feeds.
#
# Usage (from the repository root):
#   python optimize/05_export_mms_tts_onnx.py                    # export all
#   python optimize/05_export_mms_tts_onnx.py --language vi      # single language
#
# Output files (per language, <iso3> = vie | eng | zho):
#   mms_tts_<iso3>.onnx                  — VITS graph
#   mms_tts_<iso3>_charset.txt           — token charset for the Java TextMapper
#   mms_tts_<iso3>_tokenizer_config.json — add_blank / pad_token flags
#   mms_tts_<iso3>_vocab.json            — token -> vocab-id map (debug / tests)

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Language code -> (HuggingFace model ID, ISO-639-3 suffix used in filenames.
# The Java side maps vi->vie / en->eng / zh->zho via languageToSuffix(), so
# the asset filenames must use the iso3 suffix to match.)
#
# No Mandarin MMS-TTS checkpoint was ever published — the upstream MMS corpus
# had no usable Mandarin recording, so neither facebook/mms-tts-zho nor a
# community re-upload exists on the Hub. Chinese stays on the system-TTS
# fallback and is handled as a documented skip below.
TTS_MODELS = {
    "vi": ("facebook/mms-tts-vie", "vie"),
    "en": ("facebook/mms-tts-eng", "eng"),
}

# Language codes Java knows about but for which no public checkpoint exists.
UNSUPPORTED_LANGUAGES = {
    "zh": "no Mandarin MMS-TTS checkpoint was ever published "
          "(the MMS corpus has no cmn/zho recording); "
          "the app falls back to the Android system TTS for Chinese",
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "onnx_models" / "mms_tts"

# Fixed token sequence length used for tracing. The graph must be exported
# with dynamic_axes on this dimension or it will only accept exactly this
# length at runtime. VITS conditions entirely on token count, so any small
# length traces the same graph.
TRACE_SEQ_LEN = 12

# Sample text per language, used to smoke-test the exported graph. The test
# tokenizes with the real HF tokenizer (including blank interleaving) and runs
# onnxruntime so we know the graph loads and produces sane audio *before*
# anyone copies it onto a phone.
SMOKE_TEST_TEXTS = {
    "vi": "xin chào",
    "en": "hello",
    "zh": "你好",
}


class _VitsOnnxWrapper(torch.nn.Module):
    """Exposes VitsModel as a pure function input_ids -> waveform.

    VitsModel.forward returns a VitsModelOutput dataclass (waveform plus
    spectrogram/hidden-states/attentions); tracing it directly would give the
    ONNX graph extra outputs. It also requires an attention_mask argument we
    don't want to ship — VITS treats absent masks as all-ones, so calling it
    without a mask is safe.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self._model(input_ids=input_ids)
        return out.waveform


def export_language(language: str, output_dir: Path) -> None:
    from transformers import VitsModel, AutoTokenizer

    model_id, iso3 = TTS_MODELS[language]
    log.info(f"Exporting {language} ({model_id}) ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VitsModel.from_pretrained(model_id)
    model.eval()

    onnx_path = output_dir / f"mms_tts_{iso3}.onnx"
    _export_with_torch(model, onnx_path)

    size_mb = onnx_path.stat().st_size / 1e6
    log.info(f"  Saved {onnx_path.name} ({size_mb:.1f} MB)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    _dump_tokenizer_data(tokenizer, model_id, iso3, output_dir)
    _smoke_test(onnx_path, tokenizer, language)


def _export_with_torch(model, onnx_path: Path) -> None:
    """torch.onnx.export with TorchScript tracing (dynamo=False).

    dynamo=False is required: the dynamo exporter chokes on VITS' data
    dependent control flow (TorchExportError), while the legacy tracer bakes
    those branches in, matching what ORT-Mobile can run.
    """
    # Keep dummy ids inside [1, vocab_size) — sampling from a wider range
    # than the checkpoint's vocab overflows the embedding table (e.g. the
    # 96-symbol vie vocab tolerated ids up to 60, but eng only has 38).
    vocab_size = int(model.config.vocab_size)
    dummy_input = torch.randint(1, vocab_size, (1, TRACE_SEQ_LEN), dtype=torch.long)
    wrapped = _VitsOnnxWrapper(model)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy_input,),
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["waveform"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "waveform": {0: "batch", 1: "n_samples"},
            },
            opset_version=17,
            dynamo=False,
        )


def _smoke_test(onnx_path: Path, tokenizer, language: str) -> None:
    """Load the exported graph with onnxruntime and synthesize one phrase.

    Catches opset/IR incompatibilities at export time instead of on-device.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    assert len(inputs) == 1 and inputs[0].name == "input_ids", (
        f"Unexpected graph inputs: {[(i.name, i.type) for i in inputs]}"
    )
    assert inputs[0].type == "tensor(int64)", inputs[0].type
    assert len(outputs) == 1 and outputs[0].name == "waveform", (
        f"Unexpected graph outputs: {[o.name for o in outputs]}"
    )

    text = SMOKE_TEST_TEXTS.get(language, SMOKE_TEST_TEXTS["en"])
    ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
    waveform = sess.run(["waveform"], {"input_ids": ids})[0]
    n = waveform.shape[-1]
    peak = float(np.abs(waveform).max())
    log.info(f"  Smoke test OK: '{text}' -> {n} samples ({n / 16000:.2f}s), peak={peak:.3f}")
    assert n > 1600, f"Waveform implausibly short ({n} samples)"
    assert peak > 1e-3, "Waveform is (near-)silent — check tokenization/export"



def _dump_tokenizer_data(tokenizer, model_id: str, iso3: str, output_dir: Path) -> None:
    """Write the charset + token-id map the Java TextMapper needs.

    VITS tokenizes characters by raw ordinal and interleaves blank tokens
    itself, so the Android side only needs the charset order and each
    token's vocabulary id — the full HF tokenizer config is overkill and
    adds a large asset we otherwise don't need. The one exception is
    tokenizer_config.json: TextMapper parses it for the ``add_blank`` flag
    and the pad/unk token so it can interleave blank ids exactly like the
    HF VitsTokenizer. It is a few hundred bytes, so we ship it verbatim.
    """
    from huggingface_hub import hf_hub_download

    charset_path = output_dir / f"mms_tts_{iso3}_charset.txt"
    vocab_path = output_dir / f"mms_tts_{iso3}_vocab.json"
    tok_config_path = output_dir / f"mms_tts_{iso3}_tokenizer_config.json"

    vocab = tokenizer.get_vocab()
    charset = [None] * len(vocab)
    for token, idx in vocab.items():
        if 0 <= idx < len(charset):
            charset[idx] = token

    with open(charset_path, "w", encoding="utf-8") as f:
        for entry in charset:
            f.write((entry if entry is not None else "") + "\n")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Copy the checkpoint's tokenizer_config.json verbatim. The "add_blank"
    # and "pad_token"/"unk_token" fields are what TextMapper parses.
    config_src = hf_hub_download(model_id, "tokenizer_config.json")
    shutil.copy(config_src, tok_config_path)

    log.info(
        f"  Saved {charset_path.name}, {vocab_path.name} and "
        f"{tok_config_path.name} ({len(charset)} tokens)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MMS-TTS checkpoints to ONNX for Android inference"
    )
    parser.add_argument(
        "--language",
        choices=list(TTS_MODELS.keys()) + list(UNSUPPORTED_LANGUAGES.keys()) + ["all"],
        default="all",
        help="Which language checkpoint to export (default: all with a published checkpoint).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write ONNX + tokenizer assets (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if args.language in UNSUPPORTED_LANGUAGES:
        log.warning(
            f"Skipping {args.language}: {UNSUPPORTED_LANGUAGES[args.language]}"
        )
        return

    languages = list(TTS_MODELS.keys()) if args.language == "all" else [args.language]
    for lang in languages:
        export_language(lang, args.output_dir)

    log.info("Export complete.")
    log.info(
        "Next: copy the .onnx + _charset.txt + _tokenizer_config.json files "
        "into android/app/src/main/assets/ so TTSModule picks up the ONNX backend."
    )


if __name__ == "__main__":
    main()
