# OnSpeak47 — Omni Voice

**On-device speech translation for Vietnamese ↔ English and Vietnamese ↔ Chinese.**

Edge AI Challenge, Phase 2 | Public Services domain | Android-first, wearable-ready

---

## Pipeline

```
[User speaks]
    → Whisper Small — transcribes speech to text
    → NLLB-200 Distilled 600M — translates text between VN/EN/CN
    → MMS-TTS — converts translated text back to speech
[Translated speech plays]
```

Three models, three stages. No language-branching — every input goes through the same path.

## Project Structure

```
OnSpeak47/
├── backend/                    # Python pipeline modules
│   ├── asr_whisper.py         # Whisper Small ASR wrapper
│   ├── translation_nllb.py    # NLLB-200 translation (batch, pivot, corpus)
│   ├── tts_mms.py             # MMS-TTS speech synthesis
│   └── orchestrator.py        # ASR → Translation → TTS pipeline
│
├── tests_local/               # Local quality tests (no device needed)
│   ├── conftest.py            # Fixtures for all test modules
│   ├── test_01_asr.py         # Whisper transcription accuracy
│   ├── test_02_translation.py # NLLB BLEU scoring (parallel sentences)
│   ├── test_03_tts.py         # MMS-TTS synthesis validation
│   ├── test_04_vizh_corpus.py # Large-corpus VI↔ZH BLEU evaluation
│   ├── test_05_pipeline.py    # End-to-end pipeline tests
│   └── data/
│       └── parallel_sentences.json  # 40 hand-curated test sentences
│
├── optimize/                  # ONNX export & mobile optimization
│   ├── 01_export_onnx.py     # Export all models to ONNX
│   ├── 02_prune_vocab.py     # Prune NLLB vocabulary to VN/EN/CN
│   ├── 03_quantize_aimet.py  # INT8 quantization (generic + AIMET)
│   └── 04_qah_submit.py     # Qualcomm AI Hub submission
│
├── android/                   # Android app (OmniVoice)
│   ├── app/src/main/
│   │   ├── java/com/omnivoice/onspeak47/
│   │   │   ├── OmniVoiceApp.java
│   │   │   ├── LoadingActivity.java
│   │   │   ├── TranslationActivity.java
│   │   │   ├── pipeline/     # ASR, Translation, TTS, Tokenizer
│   │   │   ├── audio/        # AudioRecorder, AudioPlayer
│   │   │   └── util/         # LanguageConfig, TensorUtils, FileUtils
│   │   └── res/              # Layouts, values, raw language XMLs
│   └── build.gradle
│
├── docs/                      # Generated documentation
│   └── translation_quality_results.md
│
└── requirements.txt           # Python dependencies
```

## Environment Setup (Python — local quality tests)

Windows PowerShell, from the `OnSpeak47/` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```powershell
python -c "import torch, transformers, sentencepiece, soundfile, librosa, scipy, jiwer, sacrebleu; print('ok')"
```

## Running Tests

### Translation quality (NLLB BLEU)
```powershell
python -m pytest tests_local/test_02_translation.py -v
```

### VI↔ZH large corpus evaluation
```powershell
python -m pytest tests_local/test_04_vizh_corpus.py -v -s
```

### Full pipeline (ASR → Translation → TTS)
```powershell
python -m pytest tests_local/test_05_pipeline.py -v -s
```

### All tests
```powershell
python -m pytest tests_local/ -v -s
```

Results are written to `docs/translation_quality_results.md`.

## ONNX Export

```powershell
# Uncomment optimum/onnxruntime in requirements.txt first
pip install optimum onnxruntime

# Export all models
python optimize/01_export_onnx.py --verify

# Prune NLLB vocabulary to VN/EN/CN
python optimize/02_prune_vocab.py

# Quantize to INT8
python optimize/03_quantize_aimet.py --method onnx_int8

# Export MMS-TTS models (vi / en / zh) for on-device speech output
python optimize/export_mms_tts.py --verify
```

## Android App

The Android app is in `android/`. To build:

1. Copy ONNX model files to `android/app/src/main/assets/`
2. Open in Android Studio
3. Build → Make Project (or `gradlew assembleDebug`)

Required assets:
- `encoder_model_int8.onnx` (NLLB encoder)
- `decoder_model_merged_int8.onnx` (NLLB decoder)
- `sentencepiece_bpe.model` (NLLB tokenizer)
- `whisper_encoder.onnx`
- `whisper_decoder.onnx`
- `whisper_preprocess.onnx` / `whisper_postprocess.onnx` / `whisper_vocab.json`
- `mms_tts_vi.onnx` + `mms_tts_vi_vocab.json` (MMS-TTS Vietnamese)
- `mms_tts_en.onnx` + `mms_tts_en_vocab.json` (MMS-TTS English)
- `mms_tts_zh.onnx` + `mms_tts_zh_vocab.json` (MMS-TTS Chinese)

> The `mms_tts_*` files are optional — if not bundled, TTSModule falls back
> to the Android system TextToSpeech engine for that language.

## Architecture

- **Modular pipeline stages** — clear I/O contracts between ASR, Translation, TTS
- **ONNX Runtime** for all model inference (same as RTranslator-2.00)
- **Qualcomm AI Hub** as primary optimization path (not vendor-locked)
- **Wearable-ready** — pipeline design allows hardware swap without redesign
