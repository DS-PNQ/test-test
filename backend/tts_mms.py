# Standardized input/output for MMS-TTS

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from transformers import VitsModel, AutoTokenizer


# MMS-TTS model IDs keyed by simple language code.
# Vietnamese has a dedicated high-quality checkpoint (MMS-TTS-vie).
MMS_TTS_MODELS = {
    "vi": "facebook/mms-tts-vie",
    "en": "facebook/mms-tts-eng",
    "zh": "facebook/mms-tts-zho",
    "zh_hans": "facebook/mms-tts-zho",
    "zh_hant": "facebook/mms-tts-zho",
}


class MMSTTS:
    """MMS-TTS (VITS architecture) wrapper for on-device TTS evaluation.

    Lazily loads per-language models on first use to save memory —
    only the three in-scope languages (vi, en, zh) are supported.
    """

    def __init__(self):
        self._models: dict[str, VitsModel] = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language: str,
        output_path: str | Path,
        *,
        sample_rate: int | None = None,
    ) -> str:
        """Convert text to speech and write a WAV file.

        Parameters
        ----------
        text : str
            Text to synthesize.
        language : str
            Language code (``"vi"``, ``"en"``, ``"zh"``).
        output_path : str | Path
            Destination WAV file path.
        sample_rate : int | None
            Override the model's native sample rate (usually 16000).
            ``None`` uses the model's default.

        Returns
        -------
        str
            Absolute path to the written WAV file.
        """
        model, tokenizer = self._get_model_and_tokenizer(language)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs)

        waveform = output.waveform[0].cpu().numpy()
        sr = sample_rate or model.config.sampling_rate

        # Normalize to 16-bit PCM range
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform_int16 = (waveform * 32767).astype(np.int16)

        scipy.io.wavfile.write(str(output_path), sr, waveform_int16)
        return str(output_path.resolve())

    def synthesize_to_array(
        self,
        text: str,
        language: str,
    ) -> tuple[np.ndarray, int]:
        """Synthesize and return the raw waveform array + sample rate."""
        model, tokenizer = self._get_model_and_tokenizer(language)
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs)

        waveform = output.waveform[0].cpu().numpy()
        return waveform, model.config.sampling_rate

    def get_supported_languages(self) -> list[str]:
        """Return the list of supported language codes."""
        return list(MMS_TTS_MODELS.keys())

    # ------------------------------------------------------------------
    # Internals — lazy loading
    # ------------------------------------------------------------------

    def _get_model_and_tokenizer(
        self, language: str
    ) -> tuple[VitsModel, AutoTokenizer]:
        if language not in MMS_TTS_MODELS:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported: {list(MMS_TTS_MODELS.keys())}"
            )

        if language not in self._models:
            model_id = MMS_TTS_MODELS[language]
            self._tokenizers[language] = AutoTokenizer.from_pretrained(model_id)
            self._models[language] = VitsModel.from_pretrained(model_id)
            self._models[language].eval()

        return self._models[language], self._tokenizers[language]
