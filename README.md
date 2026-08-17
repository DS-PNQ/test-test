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

**ASR runs streaming-class fast**: the Whisper decoder is a custom KV-cache
export (constant per-token cost instead of O(n²)) and the encoder accepts the
audio's real length (a 3 s clip no longer pays a full 30 s encoder window).
End-to-end ASR on ~3 s clips measured **~10× faster** (en 5515→533 ms,
vi 5160→570 ms, desktop ort 1.22) with quality equal or better — see
`docs/optimization_results.md`.

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
│   ├── test_06_onnx_parity.py # ONNX parity GATE — quality vs PyTorch + regression baseline
│   ├── gen_parity_reference.py     # Generates the PyTorch greedy reference
│   ├── verify_preprocess_onnx.py   # whisper_preprocess.onnx sanity check
│   ├── diagnose_whisper_paths.py   # Decodes two fixtures on every asset variant
│   ├── baselines/             # Recorded regression baseline (auto-created)
│   ├── output/                # Latest gate/parity results (JSON)
│   └── data/
│       ├── parallel_sentences.json     # 40 hand-curated test sentences
│       └── audio_samples/              # ASR parity fixtures (WAV + reference text)
│
├── optimize/                  # ONNX export & mobile optimization
│   ├── 01_export_onnx.py     # Export all models to ONNX (original pipeline)
│   ├── 02_prune_vocab.py     # Prune NLLB vocabulary to VN/EN/CN
│   ├── 03_quantize_aimet.py  # INT8 quantization (generic + AIMET)
│   ├── 04_qah_submit.py     # Qualcomm AI Hub submission
│   ├── 05_slim_decoder.py   # Slim NLLB decoder (drop fp32 embedding)
│   ├── 06_quantize_whisper.py       # Legacy: quantize the cache-less decoder export
│   ├── 07_preoptimize.py     # Offline ORT graph pre-opt (*.opt.onnx) — run in ort==1.22 venv
│   ├── 08_quantize_whisper_encoder.py  # Legacy: int8 the fixed-3000 encoder export
│   ├── 09_export_whisper_decoder_kv.py # KV-cached Whisper decoder (CURRENT — streaming ASR)
│   ├── 10_export_whisper_encoder_dyn.py # Dynamic-length Whisper encoder (CURRENT)
│   └── export_mms_tts.py     # MMS-TTS export
│
├── android/                   # Android app (OmniVoice)
│   ├── app/src/main/
│   │   ├── java/com/omnivoice/onspeak47/
│   │   │   ├── OmniVoiceApp.java
│   │   │   ├── LoadingActivity.java
│   │   │   ├── TranslationActivity.java
│   │   │   ├── pipeline/     # ASR, Translation, TTS, PipelineOrchestrator, Tokenizer
│   │   │   ├── audio/        # AudioRecorder, AudioPlayer
│   │   │   └── util/         # LanguageConfig, OrtSessionConfig, TensorUtils, FileUtils
│   │   └── res/              # Layouts, values, raw language XMLs
│   └── build.gradle
│
├── docs/
│   ├── architecture.md           # Pipeline architecture overview
│   └── optimization_results.md   # Runtime/RAM + streaming-ASR optimization log
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

### ONNX parity GATE (required after ANY model/asset change)
Runs the exact on-device inference scheme (ORT + KV-cached greedy decode,
mirroring the Java modules) against the bundled assets and compares against
the PyTorch greedy reference plus a recorded regression baseline. Use a
python with **onnxruntime==1.22.0** (the version the app ships):

```powershell
python -m venv .venv-ort122
.venv-ort122\Scripts\pip install "onnxruntime==1.22.0" numpy pytest jiwer sacrebleu sentencepiece
.venv-ort122\Scripts\python -m pytest tests_local\test_06_onnx_parity.py -v
```

8/8 must pass (6 NLLB directions + Whisper en/vi). Latest recorded run:
`tests_local/output/onnx_parity_results.json`.

## ONNX Export

`*.onnx` files are **gitignored** — a fresh checkout must regenerate the
assets before building the app (see "Required assets" for the exact list).

### Whisper ASR assets (current pipeline)

The current Whisper graphs are custom exports (both validated against eager
HuggingFace before quantizing; both installed as dynamic int8):

```powershell
python optimize/09_export_whisper_decoder_kv.py    # KV-cached decoder → whisper_decoder.onnx
python optimize/10_export_whisper_encoder_dyn.py   # dynamic-length encoder → whisper_encoder.onnx

# Pre-optimize offline — MUST run under onnxruntime==1.22.0 (matches the app):
.venv-ort122\Scripts\python optimize\07_preoptimize.py --models whisper_decoder.onnx whisper_encoder.onnx

# GATE — 8/8 must pass before shipping:
.venv-ort122\Scripts\python -m pytest tests_local\test_06_onnx_parity.py -v
```

- `09` decoder: one uniform graph (no `If` branch) — prefill feeds the whole
  prefix with empty pasts; every step feeds ONE token with an empty
  `encoder_hidden_states`. Decoder presents are cumulative (cycled zero-copy);
  encoder cross-K/V are returned at prefill and re-fed each step. The app
  (`ASRModule.decodeWithCache`) mirrors this contract and falls back to
  whole-sequence decode on cache-less graphs.
- `10` encoder: accepts any mel length — `ASRModule` feeds only the audio's
  real frames (the 3000-frame pad tail is skipped for short clips).

Both read weights from the committed `android/app/src/main/assets/hf_cache/`
(no HF download needed) and back up replaced assets under `onnx_models/`.
Rollback paths are documented in `docs/optimization_results.md`.

### NLLB / MMS-TTS assets (original pipeline)

```powershell
# requirements.txt already lists the export stack (optimum, onnx, onnxruntime,
# onnxruntime-extensions); on Linux you may additionally `pip install aimet-torch`
# to enable the AIMET-only quantization method.
pip install -r requirements.txt

python optimize/01_export_onnx.py --verify      # Export all models
python optimize/02_prune_vocab.py               # Prune NLLB vocabulary to VN/EN/CN
python optimize/03_quantize_aimet.py --method onnx_int8   # Quantize to INT8
python optimize/05_slim_decoder.py              # Slim the NLLB merged decoder
python optimize/export_mms_tts.py --verify      # MMS-TTS models (vi / en)
.venv-ort122\Scripts\python optimize\07_preoptimize.py --models encoder_model_int8.onnx  # pre-opt NLLB encoder
```

## Android App

The Android app is in `android/`. To build:

1. Ensure the ONNX assets exist in `android/app/src/main/assets/` (they are
   **gitignored** — on a fresh checkout regenerate them via the commands in
   "ONNX Export"; on a machine that already ran the scripts they are in place
   and a plain rebuild is enough)
2. Open the `android/` folder in **Android Studio** — the project uses **Gradle 9.5**
   and requires **JDK 17+** (Android Studio bundles a compatible JDK; there is no
   `gradlew` wrapper committed, so run `gradle assembleDebug` from `android/`
   when building on the command line)
3. Build → Make Project (or press Run ▶ on a connected device / emulator)

> The APK is built for **arm64-v8a only** (`abiFilters 'arm64-v8a'` in `app/build.gradle`).

> Whisper assets: the decoder ships as a **KV-cached dynamic-int8** graph
> (~196 MB, was 100% fp32 ~774 MB; the old cache-less export cost O(n²)
> whole-sequence decode) and the encoder as a **dynamic-length dynamic-int8**
> graph (~98 MB, accepts any mel length). The app prefers the offline
> pre-optimized siblings (`*.opt.onnx`, loaded with NO_OPT for fast session
> creation) and falls back to the base files. The fp32 originals and every
> replaced asset are backed up under `onnx_models/`; regenerate with
> `optimize/09` / `optimize/10` + `optimize/07` (see "ONNX Export"), then run
> the parity gate. Validation: fp32 exports match eager HuggingFace logits
> <1e-3 (prefill AND cached step); gate transcripts — en matches the PyTorch
> reference exactly.

Required assets:
- `encoder_model_int8.onnx` or `encoder_model_int8.opt.onnx` (NLLB encoder)
- `decoder_model_merged_int8.onnx` (NLLB decoder — slimmed to drop the redundant
  fp32 embedding, see `optimize/05_slim_decoder.py`; deliberately NOT pre-optimized)
- `sentencepiece_bpe.model` (NLLB tokenizer; `language_token_map.json` optional)
- `whisper_encoder.opt.onnx` (dynamic-length int8 — regenerate via `optimize/10` + `07`)
- `whisper_decoder.opt.onnx` (KV-cached int8 — regenerate via `optimize/09` + `07`)
- `whisper_preprocess.onnx` / `whisper_postprocess.onnx` / `whisper_vocab.json`
- `mms_tts_vi.onnx` + `mms_tts_vi_vocab.json` (MMS-TTS Vietnamese)
- `mms_tts_en.onnx` + `mms_tts_en_config.json` + `mms_tts_en_vocab.json` (MMS-TTS English)

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
- **Streaming-class ASR decode** — KV-cached zero-copy greedy decode
  (`decodeWithCache`: prefill-held encoder K/V, cumulative decoder presents,
  per-step cost constant) with whole-sequence fallback; row-only logits argmax
  (no full `[seq × 51865]` Java materialization per step)
- **Short-window encoder** — dynamic-length encoder input; short clips feed only
  their real mel frames (~10× less encoder compute for a 3 s clip)
- **ONNX Runtime** for all model inference (same as RTranslator-2.00); offline
  pre-optimized graphs (`*.opt.onnx`) loaded with NO_OPT; mobile-tuned session
  options (`OrtSessionConfig`: low-RAM arena/mem-pattern handling, NNAPI/XNNPACK A/B flags)
- **Qualcomm AI Hub** as primary optimization path (not vendor-locked)
- **Wearable-ready** — pipeline design allows hardware swap without redesign;
  resident model weights ~1.29 GB (see `docs/optimization_results.md`)
- **Two-backend TTS** — neural MMS-TTS (ONNX) for VN/EN when bundled, with the
  Android system TextToSpeech engine as an automatic fallback (and the only
  backend for Chinese)
