"""MnasNet-only, offline Keras 3 -> Keras 2.15 deployment preparation.

Run with production Python. Only export-source prepends the isolated Keras 3
package directory. No training, test calibration, or other detector writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / "outputs/mnasnet_conversion"
SOURCE = ROOT / "mnasnet_lstm_deployment/model.keras"
LIFECYCLE = SOURCE.with_name("lifecycle_state.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leaves(model, prefix=""):
    """Named leaf weight groups, independent of trainable-weight ordering."""
    if hasattr(model, "layers"):
        for layer in model.layers:
            yield from leaves(layer, prefix + "/" + layer.name)
    elif hasattr(model, "layer"):
        yield from leaves(model.layer, prefix + "/wrapped")
    elif model.weights:
        yield prefix, model


def prepare():
    import numpy as np
    from src.lava.data.loader import load_split
    from src.preprocessing import process_audio_file
    paths, labels = load_split("validation")
    selected = [i for cls in (0, 1) for i in [j for j, y in enumerate(labels) if y == cls][:2]]
    features = [process_audio_file(paths[i]) for i in selected]
    features += [np.zeros_like(features[0]), np.random.default_rng(42).uniform(0, 255, features[0].shape).astype("float32")]
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / "parity_inputs.npy", np.stack(features))
    write_json(WORK / "parity_inputs.json", {"split": "validation", "paths": [paths[i] for i in selected], "labels": [labels[i] for i in selected], "extra": ["zeros", "seed42 uniform"]})


def export_source():
    # Process-local imports only: never replace Keras in the app environment.
    isolated = ROOT / "outputs/efficientnet_conversion/keras3"
    if isolated.is_dir():
        sys.path.insert(0, str(isolated))
    import numpy as np
    import keras
    import tensorflow as tf
    if keras.__version__ != "3.13.2":
        raise RuntimeError("Source export requires isolated Keras 3.13.2")
    model = keras.models.load_model(SOURCE, compile=False, safe_mode=True)
    if model.name != "mnasnet_lstm_audio_deepfake":
        raise ValueError("Not the expected MnasNet checkpoint")
    groups, arrays = {}, {}
    for name, layer in leaves(model):
        keys = []
        for value in layer.get_weights():
            key = f"w{len(arrays):04d}"
            arrays[key] = value
            keys.append(key)
        groups[name] = keys
    np.savez(WORK / "source_weights.npz", **arrays)
    with zipfile.ZipFile(SOURCE) as archive:
        config = json.loads(archive.read("config.json"))
    write_json(WORK / "source_export.json", {"config": config, "groups": groups, "params": model.count_params(), "source_sha256": sha(SOURCE), "keras_version": keras.__version__, "tensorflow_version": tf.__version__})
    inputs = np.load(WORK / "parity_inputs.npy")
    embedding_model = keras.Model(model.inputs, model.get_layer("time_distributed_mnasnet").output)
    scores = np.concatenate([np.asarray(model(x[None], training=False)) for x in inputs])
    embeddings = np.concatenate([np.asarray(embedding_model(x[None], training=False)) for x in inputs])
    np.savez(WORK / "source_predictions.npz", scores=scores, embeddings=embeddings)
    print("Native source load PASS", model.count_params(), scores.ravel(), flush=True)


def clean_config(value):
    if isinstance(value, list):
        return [clean_config(x) for x in value]
    if not isinstance(value, dict):
        return "swish" if value == "silu" else value
    if value.get("class_name") == "DTypePolicy":
        return value["config"]["name"]
    return {k: clean_config(v) for k, v in value.items() if k not in {"module", "registered_name", "shared_object_id"}}


def rebuild_graph(description):
    """Translate only serialization syntax; retain the source graph/scaling."""
    import tensorflow as tf
    cfg = description["config"]
    tensors = {}

    def resolve(value):
        if isinstance(value, dict) and value.get("class_name") == "__keras_tensor__":
            name, node, output = value["config"]["keras_history"]
            if node != 0 or output != 0:
                raise ValueError("Unsupported multi-node/output graph")
            return tensors[name]
        if isinstance(value, list):
            return [resolve(x) for x in value]
        return value

    allowed = {"ReLU", "Rescaling", "Normalization", "ZeroPadding2D", "Conv2D", "BatchNormalization", "Activation", "DepthwiseConv2D", "GlobalAveragePooling2D", "Reshape", "Multiply", "Dropout", "Add", "LSTM", "Dense"}
    for item in cfg["layers"]:
        kind = item["class_name"]
        lc = clean_config(item["config"])
        name = lc["name"]
        if kind == "InputLayer":
            if any(lc.get(k, False) for k in ("sparse", "ragged", "optional")):
                raise ValueError("Unsupported input semantics")
            tensors[name] = tf.keras.Input(batch_shape=lc["batch_shape"], dtype=lc["dtype"], name=name)
            continue
        if kind == "TimeDistributed":
            nested = rebuild_graph(item["config"]["layer"])
            layer = tf.keras.layers.TimeDistributed(nested, name=name, trainable=lc["trainable"], dtype=lc["dtype"])
        else:
            if kind not in allowed:
                raise ValueError(f"Unsupported layer {kind}")
            for key in ("quantization_config", "seed" if kind == "LSTM" else "unused"):
                if key in lc:
                    if lc[key] is not None:
                        raise ValueError(f"Cannot discard non-null {key}")
                    lc.pop(key)
            layer = tf.keras.layers.deserialize({"class_name": kind, "config": lc})
        nodes = item["inbound_nodes"]
        if len(nodes) != 1:
            raise ValueError("Expected one inbound node")
        kwargs = {k: v for k, v in nodes[0].get("kwargs", {}).items() if v is not None}
        tensors[name] = layer(*resolve(nodes[0]["args"]), **kwargs)
    def endpoint(ref):
        if len(ref) != 3 or ref[1:] != [0, 0]:
            raise ValueError("Unexpected model endpoints")
        return tensors[ref[0]]
    return tf.keras.Model(endpoint(cfg["input_layers"]), endpoint(cfg["output_layers"]), name=cfg["name"], trainable=cfg["trainable"])


def convert():
    import numpy as np
    import tensorflow as tf
    from src.lava.models.tensorflow.mnasnet_lstm import MnasNetLSTMDetector
    exported = read_json(WORK / "source_export.json")
    if sha(SOURCE) != exported["source_sha256"]:
        raise ValueError("Source changed after export")
    model = rebuild_graph(exported["config"])
    groups = dict(leaves(model))
    if set(groups) != set(exported["groups"]):
        raise ValueError("Weight layer coverage mismatch")
    with np.load(WORK / "source_weights.npz", allow_pickle=False) as arrays:
        for name, layer in groups.items():
            values = [arrays[k] for k in exported["groups"][name]]
            if [v.shape for v in values] != [tuple(w.shape) for w in layer.weights]:
                raise ValueError(f"Weight shape mismatch: {name}")
            layer.set_weights(values)
    if model.count_params() != exported["params"]:
        raise ValueError("Parameter count changed")
    MnasNetLSTMDetector().validate_model(model)
    inputs = np.load(WORK / "parity_inputs.npy")
    expected = np.load(WORK / "source_predictions.npz")
    embedding_model = tf.keras.Model(model.inputs, model.get_layer("time_distributed_mnasnet").output)
    scores = np.concatenate([model(x[None], training=False).numpy() for x in inputs])
    embeddings = np.concatenate([embedding_model(x[None], training=False).numpy() for x in inputs])
    np.testing.assert_allclose(scores, expected["scores"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(embeddings, expected["embeddings"], rtol=1e-3, atol=1e-4)
    converted = WORK / "converted.keras"
    model.save(converted)
    loaded = tf.keras.models.load_model(converted, compile=False)
    restored = np.concatenate([loaded(x[None], training=False).numpy() for x in inputs])
    np.testing.assert_allclose(scores, restored, rtol=1e-6, atol=1e-7)
    report = {"status": "PASS", "source_sha256": sha(SOURCE), "converted_sha256": sha(converted), "source_keras": exported["keras_version"], "target_tensorflow": tf.__version__, "parameter_count": model.count_params(), "input_shape": model.input_shape, "output_shape": model.output_shape, "weight_layers": len(groups), "parity_samples": len(inputs), "score_max_abs_difference": float(np.max(np.abs(scores - expected['scores']))), "embedding_max_abs_difference": float(np.max(np.abs(embeddings - expected['embeddings']))), "save_reload_max_abs_difference": float(np.max(np.abs(scores-restored))), "source_scores": expected['scores'].ravel().tolist(), "converted_scores": scores.ravel().tolist()}
    write_json(WORK / "conversion_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def publish():
    import tensorflow as tf
    from src.lava.artifacts import write_json_atomic
    from src.lava.data.manifest import validate_manifest_files
    from src.lava.models.tensorflow.mnasnet_lstm import MnasNetLSTMDetector
    report = read_json(WORK / "conversion_report.json")
    converted = WORK / "converted.keras"
    metadata = read_json(SOURCE.with_name("metadata.json"))
    threshold = read_json(SOURCE.with_name("threshold.json"))
    lifecycle = read_json(LIFECYCLE)
    manifest = validate_manifest_files()
    if metadata["detector_name"] != "mnasnet_lstm" or lifecycle["detector_name"] != "mnasnet_lstm":
        raise ValueError("Wrong detector bundle")
    if not lifecycle["selection_finalized"] or not lifecycle["production_model_saved"]:
        raise ValueError("Incomplete lifecycle")
    if metadata["best_epoch"] != lifecycle["best_epoch"] or metadata["selection"]["test_used"]:
        raise ValueError("Invalid checkpoint selection provenance")
    if threshold["source"] != "validation" or threshold["threshold"] != metadata["final_threshold"]:
        raise ValueError("Threshold provenance mismatch")
    if not 0 <= threshold["threshold"] <= 1:
        raise ValueError("Invalid threshold")
    if manifest["manifest_hash"] != metadata["training_manifest_hash"] or manifest["manifest_hash"] != lifecycle["training_manifest_hash"]:
        raise ValueError("Manifest mismatch")
    if report["status"] != "PASS" or sha(SOURCE) != report["source_sha256"] or sha(converted) != report["converted_sha256"]:
        raise ValueError("Stale conversion report")
    if report["parameter_count"] != metadata["parameter_count"]:
        raise ValueError("Metadata parameter count mismatch")
    with (ROOT / "data/manifests/split_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                path = Path(row["path"])
                if not path.is_absolute():
                    path = ROOT / path
                if sha(path) != row["sha256"]:
                    raise ValueError(f"Changed test audio: {path}")
    detector = MnasNetLSTMDetector()
    spec = detector.spec
    if any(p.exists() for p in (spec.model_artifact, spec.threshold_artifact, spec.metadata_artifact)):
        raise FileExistsError("Refusing to overwrite MnasNet deployment")
    metadata["training_framework_version"] = metadata["framework_version"]
    metadata["framework_version"] = tf.__version__
    metadata["serialized_size"] = converted.stat().st_size
    metadata["conversion"] = report
    metadata["inference"] = {"default_batch_size": 1, "precision": "float32", "training": False}
    spec.model_artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(converted, spec.model_artifact)
    detector.load()
    write_json_atomic(spec.threshold_artifact, threshold)
    write_json_atomic(spec.metadata_artifact, metadata)
    print("MnasNet deployment published; original validation threshold retained.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "export-source", "convert", "publish"])
    args = parser.parse_args()
    {"prepare": prepare, "export-source": export_source, "convert": convert, "publish": publish}[args.operation]()
