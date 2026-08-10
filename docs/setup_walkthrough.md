# OmniVoice (OnSpeak47) — Setup & Usage Walkthrough

**On-device speech translation for Vietnamese ↔ English and Vietnamese ↔ Chinese.**
Every utterance runs through the same three-stage pipeline — no language branching:

```
[You speak] → Whisper Small (ASR) → NLLB-200 (translate) → MMS-TTS (speak back) → [Audio plays]
```

This guide covers: (1) preparing the Python environment, (2) producing the model
assets, (3) building the Android app, a nd (4) using it.

---

## 0. Prerequisites

| Tool | Version / Notes |
|------|-----------------|
| Python | 3.10+ (3.12 confirmed working) |
| Android Studio | Hedgehog or newer |
| Android SDK / NDK | API 34, CMake 3.22.1, NDK r25+ |
| Android device | arm64-v8a only, `minSdk 26` (Android 8.0+), a Snapdragon device preferred |
| Disk space | ~4 GB free (models + HF cache + build) |

> The app compiles **only `arm64-v8a`** (`build.gradle`), so it will not run on an
> x86 emulator. Use a physical device.

---

## 1. Python environment

From the repository root (`d:\StudioProjects\realdemo\`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```powershell
python -c "import torch, transformers, sentencepiece, soundfile, librosa, scipy, onnx, onnxruntime; print('ok')"
```

---

## 2. Produce the model assets

The app loads everything from `android/app/src/main/assets/`. The current pipeline
uses **four** ONNX model families:

| Model | Files | Provided by |
|-------|-------|-------------|
| NLLB-200 (translation) | `encoder_model_int8.onnx`, `decoder_model_merged_int8.onnx`, `sentencepiece_bpe.model` | `optimize/01_export_onnx.py` (Xenova quantized) |
| Whisper Small (ASR) | `whisper_encoder.onnx`, `whisper_decoder.onnx`, `whisper_preprocess.onnx`, `whisper_postprocess.onnx`, `whisper_vocab.json` | `optimize/01_export_onnx.py` |
| MMS-TTS vi/en | `mms_tts_vie.onnx`, `mms_tts_eng.onnx`, `mms_tts_<iso3>_charset.txt`, `mms_tts_<iso3>_tokenizer_config.json` | `optimize/05_export_mms_tts_onnx.py` |
| MMS-TTS zh | *(none)* | **no public MMS Mandarin checkpoint exists** — see note below |

### 2a. NLLB + Whisper

```powershell
python optimize/01_export_onnx.py --verify         # downloads NLLB int8 + exports Whisper
python optimize/02_prune_vocab.py                  # prune NLLB vocab to VN/EN/CN
python optimize/03_quantize_aimet.py --method onnx_int8   # (optional) further INT8 quant
```

`01_export_onnx.py` writes into `onnx_models/`; copy the resulting files into
`android/app/src/main/assets/`.

### 2b. MMS-TTS (text-to-speech)

```powershell
python optimize/05_export_mms_tts_onnx.py            # exports vi + en, with smoke tests
```

Each run produces a graph validated against the exact contract the Java feeds:
input `input_ids` (`int64 [batch, seq_len]`) → output `waveform` (`float [batch, n_samples]`).
The script also copies `*_tokenizer_config.json` so the Java `TextMapper` can
interleave blank tokens exactly like the HF `VitsTokenizer`.

After export, copy the six runtime files into `android/app/src/main/assets/`:

```
mms_tts_vie.onnx  mms_tts_vie_charset.txt  mms_tts_vie_tokenizer_config.json
mms_tts_eng.onnx  mms_tts_eng_charset.txt  mms_tts_eng_tokenizer_config.json
```

> **Chinese note:** there is no published `mms-tts-zho` checkpoint — the MMS corpus
> never shipped a Mandarin recording. `TTSModule` transparently falls back to the
> **Android system TTS** for Chinese (and for any language whose ONNX asset is
> missing). The pipeline still completes end-to-end; speech out just comes from the
> phone's built-in engine instead of the neural model.

---

## 3. Build & install the Android app

1. Copy **all** the assets from step 2 into `android/app/src/main/assets/`.
   Check the folder contents — you should see the four model families listed above.
2. Open the `android/` folder in **Android Studio**.
3. Let Gradle sync, then **Build → Make Project** (or from a shell):
   ```powershell
   cd android
   .\gradlew assembleDebug
   ```
4. Install and run on your device:
   ```powershell
   .\gradlew installDebug
   ```
   (or the **Run ▶** button in Android Studio).

`build.gradle` sets `noCompress 'onnx', 'model', 'json'` and `largeHeap="true"`,
both required for the multi-hundred-MB graphs to load from assets.

---

## 4. Using the app

### First launch — model loading
The app opens on `LoadingActivity`, which initializes the three models **in
parallel** and shows progress:

```
Loading ASR model (Whisper Small)...   (Whisper encoder/decoder ONNX sessions)
Loading Translation model (NLLB-200)... (encoder + decoder int8)
Loading TTS model (MMS-TTS)...          (MMS sessions + Android TTS fallback)
Ready!
```

It then moves to the **Translation** screen automatically. Loading takes a few
seconds on first launch (per-language ONNX sessions stay warm after that).

### The translation screen — walkie-talkie style
The UI (`TranslationActivity`) is a push-to-talk screen:

- **Two language spinners** at the top — **Source → Target**. Defaults to
  `Tiếng Việt → English`. Chinese (`中文 (简体)`) is also available.
- **Two text cards** — the top shows the **transcript** (what Whisper heard), the
  bottom shows the **translation** (what NLLB produced).
- **A big "Talk" button** at the bottom.

### Translating
1. Grant the **microphone permission** when prompted (required for ASR).
2. Pick your source and target languages with the spinners.
3. **Press and hold** the Talk button, speak your sentence, then release.
4. The screen shows `Listening...` while recording, then `Processing...` while the
   pipeline runs. When it finishes you get:
   - the transcript (source),
   - the translation (target),
   - a timing line like `ASR: 810ms | Translation: 240ms | TTS: 60ms | Total: 1110ms`,
   - and the translated speech **plays aloud** through the speaker.

Each stage's latency is logged to logcat under the tags `ASRModule`,
`TranslationModule`, `TTSModule`, and `PipelineOrchestrator`.

---

## 5. Reference — pipeline internals

| Stage | Class | File(s) it opens in assets |
|-------|-------|----------------------------|
| ASR | `ASRModule` | `whisper_encoder.onnx`, `whisper_decoder.onnx` |
| Translate | `TranslationModule` | `encoder_model_int8.onnx`, `decoder_model_merged_int8.onnx`, `sentencepiece_bpe.model` |
| TTS (preferred) | `TTSModule` → `MmsOnnxTTS` | `mms_tts_<iso3>.onnx`, `mms_tts_<iso3>_charset.txt`, `mms_tts_<iso3>_tokenizer_config.json` |
| TTS (fallback) | `TTSModule` → `Android TextToSpeech` | — |

- **Language mapping** is centralized in `util/LanguageConfig.java`:
  `vi/en/zh` ↔ display names ↔ NLLB FLORES codes (`vie_Latn`, `eng_Latn`, `zho_Hans`).
- **MMS-ONNX graph contract** (must match `MmsOnnxTTS.java`): exactly one input
  `input_ids` (int64) and one output `waveform` (float32). `TextMapper` converts
  text to ids and applies `add_blank` from the tokenizer config.
- **Chinese TTS** uses the system engine because no MMS Mandarin model exists.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| App stuck on loading screen | A required asset is missing | Confirm all files from §2 are in `android/app/src/main/assets/` |
| "ASR / Translation init failed" | Corrupt or wrong-size ONNX | Re-run `01_export_onnx.py --verify`; check file sizes, especially the int8 NLLB decoder |
| Translated speech never plays, or plays in English only | MMS-TTS ONNX asset missing for that language | TTS falls back to the system engine; verify `mms_tts_<iso3>.onnx` + charset + config exist |
| `mms_tts_zho.onnx` not found | — | **This is expected**; no public MMS Mandarin checkpoint exists. Chinese relies on the Android system TTS |
| `IndexError` during a new manual export | Dummy token ids out of vocab range | Fixed in `05_export_mms_tts_onnx.py` — the trace input is clamped to `[1, vocab_size)` |
| Build fails with "file too large" | Assets compressed in APK | `noCompress` is already set in `build.gradle` — don't remove it |

---

*Generated for Edge AI Challenge Phase 2 — Public Services domain.*
