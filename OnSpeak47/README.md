# OnSpeak47

## Environment Setup (local accuracy/quality tests)

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

This installs CPU-only PyTorch plus everything needed to run and score the three
local pipeline stages (`backend/`) against test data (`tests_local/`): Whisper
Small (ASR), NLLB-200-distilled-600M (translation), and MMS-TTS (speech synthesis).

The `optimum`/`onnxruntime` dependencies in `requirements.txt` are commented out —
uncomment them later when starting the ONNX export / Qualcomm AI Hub work in
`optimize/`.
