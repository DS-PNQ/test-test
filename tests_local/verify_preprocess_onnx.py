# OmniVoice — Phase 2.2: verify whisper_preprocess.onnx.
#
# Runs the bundled pre-processing graph (raw PCM -> log-mel) on the parity
# fixtures and compares its output against the Java DSP implementation
# (ported 1:1 in test_06_onnx_parity.extract_mel, which mirrors
# ASRModule.extractMelFeatures). If they match, the on-device native path is
# equivalent to the Java fallback and can be trusted to replace it.
#
# Usage: <venv>/Scripts/python tests_local/verify_preprocess_onnx.py
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests_local"))
from test_06_onnx_parity import ASSETS, extract_mel, read_wav_pcm  # noqa: E402


def main() -> None:
    pre = ort.InferenceSession(str(ASSETS / "whisper_preprocess.onnx"),
                               providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.shape, i.type) for i in pre.get_inputs()])
    print("outputs:", [(o.name, o.shape, o.type) for o in pre.get_outputs()])

    in_name = pre.get_inputs()[0].name
    rank1 = len(pre.get_inputs()[0].shape) == 1

    for wav in sorted((ROOT / "tests_local/data/audio_samples").glob("parity_*.wav")):
        samples = read_wav_pcm(wav)
        feed_shape = (len(samples),) if rank1 else (1, len(samples))
        out = pre.run(None, {in_name: samples.reshape(feed_shape).astype(np.float32)})[0]

        java = extract_mel(samples)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        diff = np.abs(out - java)
        print(f"{wav.name}: onnx{out.shape} vs java{java.shape} | "
              f"max|d|={diff.max():.5f} mean|d|={diff.mean():.6f} "
              f"| corr={np.corrcoef(out.ravel(), java.ravel())[0,1]:.5f}")

        assert out.shape == java.shape, f"shape mismatch for {wav.name}"
        assert diff.max() < 0.05, (
            f"{wav.name}: mel divergence too large (max|d|={diff.max():.4f}) — "
            "the ONNX graph and the Java DSP disagree; do not enable the native path")

    print("PREPROCESS PARITY OK")


if __name__ == "__main__":
    main()
