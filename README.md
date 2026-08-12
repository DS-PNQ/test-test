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
# requirements.txt already lists the export stack (optimum, onnx, onnxruntime,
# onnxruntime-extensions); on Linux you may additionally `pip install aimet-torch`
# to enable the AIMET-only quantization method.
pip install -r requirements.txt

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

1. Copy ONNX model files to `android/app/src/main/assets/` (see "Required assets")
2. Open the `android/` folder in **Android Studio** — the project uses **Gradle 9.5**
   and requires **JDK 17+** (Android Studio bundles a compatible JDK; there is no
   `gradlew` wrapper committed, so run `gradle assembleDebug` from `android/`
   when building on the command line)
3. Build → Make Project (or press Run ▶ on a connected device / emulator)

> The APK is built for **arm64-v8a only** (`abiFilters 'arm64-v8a'` in `app/build.gradle`).

> `whisper_decoder.onnx` is exported as a **dynamic int8** quantized model (was
> 100% fp32 ~774 MB, now ~195 MB). The fp32 original is backed up at
> `onnx_models/whisper_decoder_fp32.onnx`; regenerate via
> `python optimize/06_quantize_whisper.py`. Validation confirmed int8 preserves
> the top-1 token ordering across the full 51,865-token vocab.

Required assets:
- `encoder_model_int8.onnx` (NLLB encoder)
- `decoder_model_merged_int8.onnx` (NLLB decoder — slimmed to drop the redundant
  fp32 embedding, see `optimize/05_slim_decoder.py`)
- `sentencepiece_bpe.model` (NLLB tokenizer)
- `whisper_encoder.onnx`
- `whisper_decoder.onnx` (dynamic int8)
- `whisper_preprocess.onnx` / `whisper_postprocess.onnx` / `whisper_vocab.json`
- `mms_tts_vi.onnx` + `mms_tts_vi_vocab.json` (MMS-TTS Vietnamese)
- `mms_tts_en.onnx` + `mms_tts_en_vocab.json` (MMS-TTS English)

> APK size notes: the release APK is **~1.6 GB** (under the 2 GB target). The
> ~1.2 GB HuggingFace `hf_cache/` folder (a build-time byproduct the app never
> reads) is excluded from packaging via `aaptOptions.ignoreAssetsPattern
> "hf_cache"` in `app/build.gradle`. Only arm64-v8a native libs are included.


> There is **no `mms_tts_zh` MMS-TTS asset** — Meta never released an MMS-TTS
> Mandarin checkpoint (only Hakka `mms-tts-hak` and Min-Nan `mms-tts-nan`), so
> Chinese TTS uses the device's Android **system** TextToSpeech engine.
>
> The system fallback in `TTSModule` is fully automatic:
> - The engine is created on the main thread and binds **explicitly** to the
>   best available engine — preferring `com.google.android.tts` (Google TTS),
>   otherwise the first installed engine that declares a TTS service — so it no
>   longer depends on the device's "default engine" setting.
> - On the first failed Chinese synthesis it **auto-opens the system
>   voice-data installer** (`ACTION_INSTALL_TTS_DATA`) so the user can download
>   the Mandarin (普通话) voice, and it surfaces an actionable reason through
>   `getLastError()` / the TTS Toast (missing voice data, no default engine, or
>   no TTS engine installed at all).
>
> Requirements for Chinese speech output:
> - The APK declares `<queries>` for the TTS service intent (required for
>   Android 11+ **package visibility**), so the app can detect the engine
>   you install.
> - An Android TTS engine with **Chinese voice data installed** (usually
>   Google TTS → Settings → Text-to-speech → Install voice data → 普通话).
> - On emulators, use a **Google APIs / Play Store** AVD image — plain AOSP
>   images ship **no** TTS engine at all.
>
> To add proper fully-offline Chinese TTS you'd integrate a Mandarin neural TTS
> that ships ONNX (e.g. a Sherpa-ONNX VITS `zh` model with its G2P front-end) —
> see `docs/` for details.

> The `mms_tts_*` files are optional — if a model isn't bundled, `TTSModule`
> falls back to the Android system TextToSpeech engine for that language (which
> is always the path used for Chinese).

## Architecture

- **Modular pipeline stages** — clear I/O contracts between ASR, Translation, TTS
- **ONNX Runtime** for all model inference (same as RTranslator-2.00)
- **Qualcomm AI Hub** as primary optimization path (not vendor-locked)
- **Wearable-ready** — pipeline design allows hardware swap without redesign
- **Two-backend TTS** — neural MMS-TTS (ONNX) for VN/EN when bundled, with the
  Android system TextToSpeech engine as an automatic fallback (and the only
  backend for Chinese)
