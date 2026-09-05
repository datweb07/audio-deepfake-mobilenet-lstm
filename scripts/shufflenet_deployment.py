"""ShuffleNet-only Keras 3 -> TensorFlow/Keras 2.15 deployment conversion.

The source bundle in ``ok/`` is preserved.  No training or threshold calibration
is performed.  ``export-source`` is the only operation that imports the isolated
Keras 3 runtime; all production validation uses the pinned TensorFlow 2.15 stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_DIR = ROOT / "ok"
SOURCE = SOURCE_DIR / "model.keras"
WORK = ROOT / "outputs/shufflenet_conversion"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaves(model, prefix: str = ""):
    """Yield stable named leaf weight groups across Keras 2 and Keras 3."""
    if hasattr(model, "layers"):
        for layer in model.layers:
            yield from leaves(layer, prefix + "/" + layer.name)
    elif hasattr(model, "layer"):
        yield from leaves(model.layer, prefix + "/wrapped")
    elif model.weights:
        yield prefix, model


def prepare() -> None:
    import numpy as np
    from src.lava.data.loader import load_split
    from src.preprocessing import process_audio_file

    paths, labels = load_split("validation")
    selected = [i for cls in (0, 1) for i in [j for j, y in enumerate(labels) if y == cls][:2]]
    features = [process_audio_file(paths[i]) for i in selected]
    features += [
        np.zeros_like(features[0]),
        np.random.default_rng(42).uniform(0, 255, features[0].shape).astype("float32"),
    ]
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / "parity_inputs.npy", np.stack(features))
    write_json(
        WORK / "parity_inputs.json",
        {
            "split": "validation",
            "paths": [paths[i] for i in selected],
            "labels": [labels[i] for i in selected],
            "extra": ["zeros", "seed42 uniform"],
        },
    )


def export_source() -> None:
    isolated = ROOT / "outputs/efficientnet_conversion/keras3"
    if not isolated.is_dir():
        raise RuntimeError("Isolated Keras 3 runtime is unavailable")
    sys.path.insert(0, str(isolated))
    import numpy as np
    import keras
    import tensorflow as tf
    from src.lava.models.tensorflow.shufflenetv2_lstm import ChannelShuffle, ChannelSplit  # noqa: F401

    if keras.__version__ != "3.13.2":
        raise RuntimeError(f"Source export requires Keras 3.13.2, got {keras.__version__}")
    model = keras.models.load_model(SOURCE, compile=False, safe_mode=True)
    if model.name != "shufflenetv2_lstm_audio_deepfake":
        raise ValueError("Not the expected ShuffleNetV2-LSTM checkpoint")
    groups: dict[str, list[str]] = {}
    arrays: dict[str, object] = {}
    for name, layer in leaves(model):
        keys = []
        for value in layer.get_weights():
            key = f"w{len(arrays):04d}"
            arrays[key] = value
            keys.append(key)
        groups[name] = keys
    np.savez(WORK / "source_weights.npz", **arrays)
    with zipfile.ZipFile(SOURCE) as archive:
        archive_metadata = json.loads(archive.read("metadata.json"))
    write_json(
        WORK / "source_export.json",
        {
            "groups": groups,
            "params": model.count_params(),
            "source_sha256": sha(SOURCE),
            "source_archive_metadata": archive_metadata,
            "keras_version": keras.__version__,
            "tensorflow_version": tf.__version__,
        },
    )
    inputs = np.load(WORK / "parity_inputs.npy")
    embedding_model = keras.Model(
        model.inputs, model.get_layer("time_distributed_shufflenetv2").output
    )
    scores = np.concatenate([np.asarray(model(x[None], training=False)) for x in inputs])
    embeddings = np.concatenate(
        [np.asarray(embedding_model(x[None], training=False)) for x in inputs]
    )
    np.savez(WORK / "source_predictions.npz", scores=scores, embeddings=embeddings)
    print("ShuffleNet source load PASS", model.count_params(), scores.ravel(), flush=True)


def convert() -> None:
    import numpy as np
    import tensorflow as tf
    from src.lava.models.tensorflow.shufflenetv2_lstm import (
        ShuffleNetV2LSTMDetector,
        build_model,
    )

    exported = read_json(WORK / "source_export.json")
    if sha(SOURCE) != exported["source_sha256"]:
        raise ValueError("Source model changed after Keras 3 export")
    model, _ = build_model(None)
    groups = dict(leaves(model))
    if set(groups) != set(exported["groups"]):
        missing = sorted(set(exported["groups"]) - set(groups))
        extra = sorted(set(groups) - set(exported["groups"]))
        raise ValueError(f"Weight layer coverage mismatch; missing={missing}, extra={extra}")
    with np.load(WORK / "source_weights.npz", allow_pickle=False) as arrays:
        for name, layer in groups.items():
            values = [arrays[key] for key in exported["groups"][name]]
            expected_shapes = [tuple(weight.shape) for weight in layer.weights]
            if [value.shape for value in values] != expected_shapes:
                raise ValueError(f"Weight shape mismatch: {name}")
            layer.set_weights(values)
    if model.count_params() != exported["params"]:
        raise ValueError("Parameter count changed during conversion")
    ShuffleNetV2LSTMDetector().validate_model(model)
    inputs = np.load(WORK / "parity_inputs.npy")
    expected = np.load(WORK / "source_predictions.npz")
    embedding_model = tf.keras.Model(
        model.inputs, model.get_layer("time_distributed_shufflenetv2").output
    )
    scores = np.concatenate([model(x[None], training=False).numpy() for x in inputs])
    embeddings = np.concatenate(
        [embedding_model(x[None], training=False).numpy() for x in inputs]
    )
    np.testing.assert_allclose(scores, expected["scores"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(embeddings, expected["embeddings"], rtol=1e-3, atol=1e-4)
    converted = WORK / "converted.keras"
    model.save(converted)
    loaded = tf.keras.models.load_model(converted, compile=False)
    restored = np.concatenate([loaded(x[None], training=False).numpy() for x in inputs])
    np.testing.assert_allclose(scores, restored, rtol=1e-6, atol=1e-7)
    report = {
        "status": "PASS",
        "source_sha256": sha(SOURCE),
        "converted_sha256": sha(converted),
        "source_keras": exported["keras_version"],
        "target_tensorflow": tf.__version__,
        "parameter_count": model.count_params(),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "weight_layers": len(groups),
        "parity_samples": len(inputs),
        "score_max_abs_difference": float(np.max(np.abs(scores - expected["scores"]))),
        "embedding_max_abs_difference": float(
            np.max(np.abs(embeddings - expected["embeddings"]))
        ),
        "save_reload_max_abs_difference": float(np.max(np.abs(scores - restored))),
        "source_scores": expected["scores"].ravel().tolist(),
        "converted_scores": scores.ravel().tolist(),
    }
    write_json(WORK / "conversion_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def publish() -> None:
    import tensorflow as tf
    from src.lava.artifacts import write_json_atomic
    from src.lava.data.manifest import validate_manifest_files
    from src.lava.models.tensorflow.shufflenetv2_lstm import ShuffleNetV2LSTMDetector

    metadata = read_json(SOURCE_DIR / "metadata.json")
    threshold = read_json(SOURCE_DIR / "threshold.json")
    report = read_json(WORK / "conversion_report.json")
    converted = WORK / "converted.keras"
    manifest = validate_manifest_files()
    if metadata.get("detector_name") != "shufflenetv2_lstm":
        raise ValueError("Wrong detector metadata")
    selection = metadata.get("selection", {})
    if selection.get("best_stage") != "scratch" or selection.get("test_used") is not False:
        raise ValueError("Invalid checkpoint-selection provenance")
    if metadata.get("best_epoch") != selection.get("best_epoch"):
        raise ValueError("Best epoch metadata mismatch")
    if threshold.get("source") != "validation" or threshold.get("threshold") != metadata.get("final_threshold"):
        raise ValueError("Threshold provenance mismatch")
    if not 0.0 <= float(threshold["threshold"]) <= 1.0:
        raise ValueError("Threshold outside [0,1]")
    if manifest["manifest_hash"] != metadata.get("training_manifest_hash"):
        raise ValueError("Training manifest hash mismatch")
    if report.get("status") != "PASS" or sha(SOURCE) != report.get("source_sha256"):
        raise ValueError("Stale or failed conversion report")
    if sha(converted) != report.get("converted_sha256"):
        raise ValueError("Converted artifact hash mismatch")
    if report.get("parameter_count") != metadata.get("parameter_count"):
        raise ValueError("Parameter count metadata mismatch")

    detector = ShuffleNetV2LSTMDetector()
    spec = detector.spec
    targets = (spec.model_artifact, spec.threshold_artifact, spec.metadata_artifact)
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing ShuffleNet deployment")
    # Verify the target-format model before changing the production bundle.
    detector.model = tf.keras.models.load_model(converted, compile=False)
    detector.validate_model(detector.model)
    metadata["training_framework_version"] = metadata.get("framework_version")
    metadata["framework_version"] = tf.__version__
    metadata["serialized_size"] = converted.stat().st_size
    metadata["conversion"] = report
    metadata["inference"] = {"precision": "float32", "training": False}
    spec.model_artifact.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(converted, spec.model_artifact)
    write_json_atomic(spec.threshold_artifact, threshold)
    write_json_atomic(spec.metadata_artifact, metadata)
    detector.load()
    print("ShuffleNet deployment published; validation threshold 0.12 retained.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "export-source", "convert", "publish"])
    args = parser.parse_args()
    {"prepare": prepare, "export-source": export_source, "convert": convert, "publish": publish}[
        args.operation
    ]()
