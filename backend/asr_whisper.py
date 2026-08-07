# Standardized input/output for Whisper Small

from dataclasses import dataclass, field

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass
class ASRResult:
    """Result from Whisper transcription."""
    text: str
    language: str
    segments: list = field(default_factory=list)


# Whisper language tokens for the three in-scope languages.
WHISPER_LANG_TOKENS = {
    "vi": "<|vi|>",
    "en": "<|en|>",
    "zh": "<|zh|>",
}

MODEL_NAME = "openai/whisper-small"


class WhisperASR:
    """Whisper Small wrapper for on-device ASR evaluation.

    Provides transcription with optional language hint. If no language is given
    the model's language-detection head chooses automatically (useful for the
    conversation-mode scenario).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        sample_rate: int = 16_000,
    ) -> ASRResult:
        """Transcribe an audio file to text.

        Parameters
        ----------
        audio_path : str
            Path to audio file (any format ``librosa`` can read).
        language : str | None
            ISO-639-1 code (``"vi"``, ``"en"``, ``"zh"``). ``None`` lets
            Whisper auto-detect.
        sample_rate : int
            Target sample rate for Whisper (always 16 kHz).

        Returns
        -------
        ASRResult
            Transcription result with text, detected language, and segments.
        """
        # Load & resample to 16 kHz (Whisper requirement)
        waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

        # Prepare processor inputs
        input_features = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features

        # Build generation kwargs
        gen_kwargs: dict = {
            "max_new_tokens": 440,
            "return_timestamps": True,
        }
        if language is not None:
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **gen_kwargs)

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        # Detect language from the first predicted token if not provided
        detected_lang = language or self._detect_language(predicted_ids)

        return ASRResult(
            text=transcription,
            language=detected_lang,
            segments=[],  # Segment-level detail can be added later if needed
        )

    def transcribe_array(
        self,
        waveform: np.ndarray,
        language: str | None = None,
        *,
        sample_rate: int = 16_000,
    ) -> ASRResult:
        """Transcribe from a NumPy array instead of a file path."""
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        input_features = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features

        gen_kwargs: dict = {"max_new_tokens": 440}
        if language is not None:
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **gen_kwargs)

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        detected_lang = language or self._detect_language(predicted_ids)

        return ASRResult(text=transcription, language=detected_lang, segments=[])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_language(self, predicted_ids: torch.Tensor) -> str:
        """Best-effort language detection from the generated token sequence."""
        decoded_with_special = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False
        )[0]
        for code, token in WHISPER_LANG_TOKENS.items():
            if token in decoded_with_special:
                return code
        return "und"  # undetermined
