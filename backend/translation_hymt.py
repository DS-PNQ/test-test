# Standardized input/output for HY-MT1.5-1.8B (on-device, drop-in replacement
# for facebook/nllb-200-distilled-600M).
#
#                             HY-MT
#                              |
#                +-------------+-------------+
#                |                           |
#          Low-RAM profile             High-end profile
#                |                           |
#          1.25-bit STQ (GGUF)         INT4 ONNX
#                |                           |
#          Mobile CPU                ORT GenAI (QNN / CPU)
#                |
#          4-6 GB RAM
#
# Sources:
#   Low-RAM  : tencent/Hy-MT1.5-1.8B-1.25bit-GGUF
#              (Hy-MT1.5-1.8B-1.25bit.gguf, needs the STQ kernel from
#              llama.cpp PR #22836)
#   High-end : tencent/HY-MT1.5-1.8B-GPTQ-Int4 exported to INT4 ONNX and
#              served by onnxruntime-genai on QNN (Snapdragon) or CPU.

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

LANG_CODES = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh_hans": "Chinese",
    "zh_hant": "Traditional Chinese",
}

GGUF_REPO = "tencent/Hy-MT1.5-1.8B-1.25bit-GGUF"
GGUF_FILE = "Hy-MT1.5-1.8B-1.25bit.gguf"
INT4_REPO = "tencent/HY-MT1.5-1.8B-GPTQ-Int4"

CACHE_DIR = Path(
    os.environ.get("OMNIVOICE_MODEL_CACHE", str(Path.home() / ".cache" / "omnivoice"))
)


def _instruction(text: str, tgt_lang: str) -> str:
    """Prompt format prescribed by the HY-MT1.5 model card."""
    return (
        f"Translate the following segment into {LANG_CODES[tgt_lang]}, "
        f"without additional explanation.\n\n{text}"
    )


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return 8.0


def detect_profile() -> str:
    forced = os.environ.get("OMNIVOICE_MT_PROFILE", "").strip().lower()
    if forced in ("low_ram", "high_end"):
        return forced
    return "low_ram" if _total_ram_gb() <= 6.0 else "high_end"


def _select_execution_provider() -> str:
    forced = os.environ.get("OMNIVOICE_MT_EP", "").strip().lower()
    if forced:
        return forced
    import platform
    return "qnn" if platform.machine() in ("aarch64", "arm64") else "cpu"


class HYMTTranslator:
    """HY-MT1.5-1.8B translation wrapper (low-RAM GGUF / high-end ONNX).

    The public API (translate / translate_batch / translate_file /
    translate_pivot / preprocess_chinese / LANG_CODES) is identical to the
    old NLLBTranslator, so orchestrator.py and tests_local/ keep working
    unchanged.
    """

    def __init__(
        self,
        profile: str | None = None,
        model_cache: str | Path | None = None,
        model_name: str = INT4_REPO,  # kept for call-site compatibility
    ):
        self.profile = (profile or detect_profile()).lower()
        if self.profile not in ("low_ram", "high_end"):
            raise ValueError(f"unknown profile {self.profile!r}")
        self.cache_dir = Path(model_cache or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._backend: str | None = None
        self._llama = None
        self._llama_cli: tuple[str, str] | None = None
        self._og = None
        self._og_model = None
        self._og_tokenizer = None
        self._hf = None

        if self.profile == "low_ram":
            self._load_low_ram()
        else:
            self._load_high_end()

    # ------------------------------------------------------------------
    # Low-RAM profile: 1.25-bit STQ GGUF on the mobile CPU (4-6 GB)
    # ------------------------------------------------------------------
    def _gguf_path(self) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=str(self.cache_dir / "hymt-gguf"),
        )

    @staticmethod
    def _find_llama_cli() -> str | None:
        roots = []
        if os.environ.get("LLAMA_CPP_DIR"):
            roots.append(Path(os.environ["LLAMA_CPP_DIR"]))
        roots.append(Path.home() / "llama.cpp")
        for root in roots:
            for name in ("llama-cli", "llama-completion"):
                for cand in (root / "build" / "bin" / name, root / name):
                    if cand.is_file():
                        return str(cand)
        return shutil.which("llama-cli") or shutil.which("llama-completion")

    def _load_low_ram(self) -> None:
        gguf = self._gguf_path()
        try:
            from llama_cpp import Llama

            self._llama = Llama(
                model_path=gguf,
                n_gpu_layers=0,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                verbose=False,
            )
            self._backend = "llama_cpp"
            return
        except Exception:
            pass
        cli = self._find_llama_cli()
        if cli is None:
            raise RuntimeError(
                "Low-RAM profile needs the STQ kernel (llama.cpp PR #22836). "
                "Either pip-install llama-cpp-python from that branch or "
                "build llama.cpp and set LLAMA_CPP_DIR."
            )
        self._llama_cli = (cli, gguf)
        self._backend = "llama_cli"

    def _generate_low_ram(self, instruction: str) -> str:
        if self._backend == "llama_cpp":
            try:
                resp = self._llama.create_chat_completion(
                    messages=[{"role": "user", "content": instruction}],
                    max_tokens=512,
                    temperature=0.0,
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                out = self._llama(instruction, max_tokens=512, temperature=0.0)
                return out["choices"][0]["text"].strip()
        cli, gguf = self._llama_cli
        proc = subprocess.run(
            [
                cli, "--model", gguf,
                "-ngl", "0",
                "--jinja",
                "-n", "512",
                "--temp", "0.0",
                "-p", instruction,
            ],
            capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()

    # ------------------------------------------------------------------
    # High-end profile: INT4 ONNX (from GPTQ-Int4) via ORT GenAI
    # ------------------------------------------------------------------
    def _build_onnx(self, model_dir: Path) -> None:
        ep = _select_execution_provider()
        cmd = [
            sys.executable, "-m",
            "onnxruntime_genai.models.builder",
            "--model_name", INT4_REPO,
            "--precision", "int4",
            "--execution_provider", ep,
            "--output_dir", str(model_dir),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            if ep == "qnn":
                cmd[cmd.index("qnn")] = "cpu"
                subprocess.run(cmd, check=True)
            else:
                raise

    def _load_high_end(self) -> None:
        model_dir = self.cache_dir / "hymt-onnx-int4"
        try:
            import onnxruntime_genai as og

            if not (model_dir / "genai_config.json").exists():
                self._build_onnx(model_dir)
            self._og = og
            self._og_model = og.Model(str(model_dir))
            self._og_tokenizer = og.Tokenizer(self._og_model)
            self._backend = "ort_genai"
            return
        except Exception:
            pass
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(INT4_REPO)
        model = AutoModelForCausalLM.from_pretrained(INT4_REPO, device_map="auto")
        model.eval()
        self._hf = (tok, model)
        self._backend = "transformers"

    def _generate_high_end(self, instruction: str) -> str:
        if self._backend == "ort_genai":
            og = self._og
            prompt = self._og_tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                add_generation_prompt=True,
            )
            params = og.GeneratorParams(self._og_model)
            gen = og.Generator(self._og_model, params)
            gen.append_tokens(self._og_tokenizer.encode(prompt))
            out_tokens: list[int] = []
            while not gen.is_done():
                gen.generate_next_token()
                out_tokens.append(gen.get_next_tokens()[0])
            return self._og_tokenizer.decode(out_tokens).strip()
        # transformers fallback
        tok, model = self._hf
        tokenized = tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        out = model.generate(tokenized.to(model.device), max_new_tokens=512)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _generate(self, instruction: str) -> str:
        if self.profile == "low_ram":
            return self._generate_low_ram(instruction)
        return self._generate_high_end(instruction)

    # ------------------------------------------------------------------
    # Core: single sentence
    # ------------------------------------------------------------------
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang.startswith("zh"):
            text = self.preprocess_chinese(text)
        return self._generate(_instruction(text, tgt_lang))

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
        return [self.translate(t, src_lang, tgt_lang) for t in texts]

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
        intermediate = self.translate(text, src_lang, pivot_lang)
        return self.translate(intermediate, pivot_lang, tgt_lang)

    # ------------------------------------------------------------------
    # Chinese-specific pre-processing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess_chinese(text: str) -> str:
        text = re.sub(
            r"(?<=[\u4e00-\u9fff\u3400-\u4dbf])\s+(?=[\u4e00-\u9fff\u3400-\u4dbf])",
            "",
            text,
        )
        return text.strip()


# Backwards-compatible alias so orchestrator.py and conftest.py keep
# importing NLLBTranslator() unchanged.
NLLBTranslator = HYMTTranslator
