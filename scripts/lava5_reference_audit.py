"""Read-only native checkpoint parameter and ONNX parity audit; isolated torch environment."""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rawnet2", "aasist"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lava_5")
    args = parser.parse_args()
    import torch
    import numpy as np
    import onnxruntime as ort
    from benchmark.lava5 import read_csv, sha256
    from src.lava.preprocessing.waveform import load_waveform
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = ROOT / "models" / (args.model + "_pretrained")
    module = importlib.import_module("src.lava.models.pytorch." + args.model + "_pretrained")
    model = module.load_pretrained(directory / "model.pt", torch.device("cpu")).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"])
    rows = read_csv(args.output / "protocol/test_samples.csv")
    selected = [r for label in (0, 1) for r in [q for q in rows if int(q["label"]) == label][:3]]
    differences = []
    for row in selected:
        x = load_waveform(ROOT / row["path"], sample_rate=16000, target_samples=64600)[None]
        with torch.no_grad():
            logits = model(torch.from_numpy(x))
            native = torch.softmax(logits, dim=1).numpy()[:, 0]
        exported = session.run(["logits"], {"waveform": x})[0]
        exported = np.exp(exported - exported.max(axis=1, keepdims=True))
        exported = exported[:, 0] / exported.sum(axis=1)
        np.testing.assert_allclose(native, exported, atol=1e-4, rtol=1e-4)
        differences.append(float(np.max(np.abs(native - exported))))
    result = dict(status="PASS", model=args.model, parameter_count=sum(p.numel() for p in model.parameters()),
        buffers_elements=sum(p.numel() for p in model.buffers()), source="actual strictly loaded native model.parameters()",
        checkpoint_sha256=sha256(directory / "model.pt"), onnx_sha256=sha256(directory / "model.onnx"),
        native_class_order="FAKE=0, REAL=1; LAVA probability is softmax[:,0]", parity_samples=len(selected),
        max_score_difference=max(differences), torch_version=torch.__version__,
        original_training_dataset="ASVspoof 2019 logical access (reference README; no independent training log)",
        provenance_evidence="2021-main/LA/Baseline-RawNet2/README.md" if args.model == "rawnet2" else "aasist-main/README.md",
        caveat="Reference code read for audit only; production evaluation uses self-contained ONNX export")
    target = args.output / "protocol" / (args.model + "_native_audit.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
