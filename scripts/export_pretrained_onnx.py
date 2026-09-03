"""Export a pretrained PyTorch reference detector to verified ONNX inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def export_detector(name: str) -> None:
    import onnx
    import onnxruntime as ort
    import torch

    device = torch.device("cpu")
    artifact_dir = PROJECT_ROOT / "models" / name
    checkpoint = artifact_dir / "model.pt"
    output = artifact_dir / "model.onnx"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if name == "rawnet2_pretrained":
        from src.lava.models.pytorch.rawnet2_pretrained import load_pretrained, target_samples
    elif name == "aasist_pretrained":
        from src.lava.models.pytorch.aasist_pretrained import load_pretrained, target_samples
    else:
        raise ValueError(f"Unsupported pretrained detector: {name}")

    torch.manual_seed(42)
    model = load_pretrained(checkpoint, device).eval()
    sample = torch.randn(1, target_samples(), dtype=torch.float32) * 0.01
    with torch.no_grad():
        torch_logits = model(sample).cpu().numpy()

    torch.onnx.export(
        model,
        sample,
        str(output),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"waveform": sample.numpy()})[0]
    max_logit_difference = float(np.max(np.abs(torch_logits - onnx_logits)))
    torch_exp = np.exp(torch_logits - np.max(torch_logits, axis=1, keepdims=True))
    onnx_exp = np.exp(onnx_logits - np.max(onnx_logits, axis=1, keepdims=True))
    torch_p_fake = torch_exp[:, 0] / torch_exp.sum(axis=1)
    onnx_p_fake = onnx_exp[:, 0] / onnx_exp.sum(axis=1)
    max_probability_difference = float(np.max(np.abs(torch_p_fake - onnx_p_fake)))
    if max_probability_difference > 1e-4:
        raise RuntimeError(
            f"{name} ONNX parity failed: P(FAKE) difference={max_probability_difference}"
        )

    print(f"Exported: {output}")
    print(f"Size: {output.stat().st_size:,} bytes")
    print(f"Logit max abs difference: {max_logit_difference:.10g}")
    print(f"P(FAKE) max abs difference: {max_probability_difference:.10g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("rawnet2_pretrained", "aasist_pretrained"))
    args = parser.parse_args()
    export_detector(args.model)


if __name__ == "__main__":
    main()
