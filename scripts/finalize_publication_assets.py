"""Derive publication assets from stored evidence; never load or train a model."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom, binomtest
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lava.evaluation_metrics import compute_eer

PAPER = ROOT / "papers"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
OUT5 = ROOT / "outputs/lava_5"
OUT6 = ROOT / "outputs/lava_6"
NAMES = ["mobilenetv3_lstm", "efficientnet_b0_lstm", "mnasnet_lstm", "shufflenetv2_lstm", "rawnet2", "aasist"]
DISPLAY = dict(zip(NAMES, ["MobileNetV3", "EfficientNet-B0", "MnasNet-A1", "ShuffleNetV2", "RawNet2", "AASIST"]))
COLORS = dict(zip(NAMES, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_path(name: str) -> Path:
    return (OUT6 if name == "shufflenetv2_lstm" else OUT5) / "clean" / name / "scores.csv"


def scores() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    baseline = read_csv(score_path(NAMES[0]))
    ids = [row["sample_id"] for row in baseline]
    labels = np.array([int(row["true_label"]) for row in baseline])
    result = {}
    for name in NAMES:
        rows = read_csv(score_path(name))
        if [row["sample_id"] for row in rows] != ids:
            raise ValueError(f"Sample order mismatch: {name}")
        result[name] = {
            "p": np.array([float(row["p_fake"]) for row in rows]),
            "pred": np.array([int(row["predicted_label"]) for row in rows]),
        }
    return labels, result


def dataset_tables() -> None:
    meta = json.loads((ROOT / "data/manifests/manifest_metadata.json").read_text(encoding="utf-8"))
    write_csv(TABLES / "table_2_dataset_integrity.csv", [dict(
        Scanned=meta["scanned_samples"], Included=meta["included_samples"], Excluded=meta["excluded_samples"],
        Duplicate_groups=meta["duplicate_groups"], Redundant_duplicates=meta["redundant_duplicate_files"],
        Conflict_groups=meta["label_conflict_groups"], Conflict_files=meta["label_conflict_files"],
        Seed=meta["random_seed"], Manifest_SHA256=meta["manifest_hash"], Split_claim=meta["split_claim"],
    )])
    manifest = read_csv(ROOT / "data/manifests/split_manifest.csv")
    split_rows = []
    for split in ("train", "validation", "test"):
        selected = [row for row in manifest if row["split"] == split]
        split_rows.append(dict(Split=split, REAL=sum(int(row["label"]) == 0 for row in selected), FAKE=sum(int(row["label"]) == 1 for row in selected), Total=len(selected)))
    write_csv(TABLES / "table_3_canonical_splits.csv", split_rows)


def detector_tables() -> None:
    fields = ["Model", "Group", "Framework", "Input", "Duration_s", "Backbone", "Temporal", "Embedding", "Initialization", "Params", "Provenance"]
    values = [
        ["MobileNetV3Small-LSTM", "Lightweight", "TensorFlow 2.15", "6 Mel images", 3.0, "MobileNetV3Small", "LSTM(128)", 576, "ImageNet", 1308401, "LAVA-trained"],
        ["ShuffleNetV2-1.0x-LSTM", "Lightweight", "TensorFlow 2.15", "6 Mel images", 3.0, "ShuffleNetV2-1.0x", "LSTM(128)", 1024, "Scratch", 1868441, "LAVA-trained"],
        ["MnasNet-A1-LSTM", "Lightweight", "TensorFlow 2.15", "6 Mel images", 3.0, "MnasNet-A1", "LSTM(128)", 1280, "Scratch", 3369255, "LAVA-trained"],
        ["EfficientNet-B0-LSTM", "Lightweight", "TensorFlow 2.15", "6 Mel images", 3.0, "EfficientNet-B0", "LSTM(128)", 1280, "ImageNet", 4779300, "LAVA-trained; warm-up checkpoint"],
        ["RawNet2", "Reference", "ONNX 1.17.3 / PyTorch source", "Waveform", 4.0375, "Sinc/residual/attention", "GRU", "native", "External checkpoint", 17621410, "External reference"],
        ["AASIST", "Reference", "ONNX 1.17.3 / PyTorch source", "Waveform", 4.0375, "Sinc/2-D encoder/graph attention", "Graph readout", "native", "External checkpoint", 297866, "External reference"],
    ]
    write_csv(TABLES / "table_4_detector_specification.csv", [dict(zip(fields, row)) for row in values])
    fields = ["Model", "Initialization", "Training_mode", "Optimizer", "Initial_LR", "Fine_tune_LR", "BN_policy", "Early_stopping", "Selection", "Threshold"]
    values = [
        ["MobileNetV3Small-LSTM", "ImageNet", "Frozen warm-up; partial tail fine-tune", "Adam", "1e-4", "1e-5", "Frozen", "Per stage", "Global minimum validation loss", "Validation FAKE-F1; 0.82"],
        ["ShuffleNetV2-1.0x-LSTM", "Scratch", "Full end-to-end from epoch 1", "Adam", "3e-4", "N/A", "56/56 trainable", "Stopped 28; restored 16", "Minimum validation loss", "Validation FAKE-F1; 0.12"],
        ["MnasNet-A1-LSTM", "Scratch", "Full end-to-end from epoch 1", "Adam", "1e-4", "N/A", "49/49 trainable", "Stopped 39; restored 27", "Minimum validation loss", "Validation FAKE-F1; 0.90"],
        ["EfficientNet-B0-LSTM", "ImageNet", "Frozen warm-up only (deployed)", "Adam", "1e-4", "Not completed", "Frozen", "Warm-up lifecycle", "Best warm-up epoch 47", "Validation FAKE-F1; 0.90"],
        ["RawNet2", "External", "Not trained in current LAVA run", "N/A", "N/A", "N/A", "N/A", "N/A", "External checkpoint", "Default uncalibrated; 0.50"],
        ["AASIST", "External", "Not trained in current LAVA run", "N/A", "N/A", "N/A", "N/A", "N/A", "External checkpoint", "Default uncalibrated; 0.50"],
    ]
    write_csv(TABLES / "table_5_training_provenance.csv", [dict(zip(fields, row)) for row in values])


def result_tables() -> None:
    clean = read_csv(OUT6 / "tables/table_2_clean_6_models.csv")
    normalized, classwise = [], []
    for row in clean:
        tn, fp, fn, tp = (int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        name = row["Model"]
        normalized.append(dict(Model=DISPLAY[name], Accuracy=float(row["accuracy"]), Precision_FAKE=float(row["precision"]), Recall_FAKE=float(row["recall"]), F1_FAKE=float(row["f1"]), Macro_F1=float(row["macro_f1"]), ROC_AUC=float(row["roc_auc"]), EER=float(row["eer"])))
        real_p, real_r = tn / (tn + fn), tn / (tn + fp)
        fake_p, fake_r = tp / (tp + fp), tp / (tp + fn)
        classwise.append(dict(Model=DISPLAY[name], REAL_Precision=real_p, REAL_Recall=real_r, REAL_F1=2 * real_p * real_r / (real_p + real_r), FAKE_Precision=fake_p, FAKE_Recall=fake_r, FAKE_F1=2 * fake_p * fake_r / (fake_p + fake_r)))
    write_csv(TABLES / "table_6_clean_performance.csv", normalized)
    write_csv(TABLES / "table_7_classwise_performance.csv", classwise)
    mappings = [
        ("table_3_robustness_diagnostic_6_models.csv", "table_8_robustness_summary.csv"),
        ("table_4_efficiency_6_models.csv", "table_9_efficiency.csv"),
        ("table_5_pareto_diagnostic_6_models.csv", "table_10_pareto.csv"),
    ]
    for source, target in mappings:
        write_csv(TABLES / target, read_csv(OUT6 / "tables" / source))


def full_test_statistics(iterations: int = 1000) -> None:
    labels, values = scores()
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    rng = np.random.default_rng(42)
    draws = {name: [] for name in NAMES}
    for _ in range(iterations):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for name in NAMES:
            draws[name].append((f1_score(labels[indices], values[name]["pred"][indices], zero_division=0), roc_auc_score(labels[indices], values[name]["p"][indices]), compute_eer(labels[indices], values[name]["p"][indices])[0]))
    intervals = []
    for name in NAMES:
        array = np.asarray(draws[name])
        row = {"Model": DISPLAY[name], "Scope": "full canonical test; stratified percentile bootstrap; seed 42", "Iterations": iterations}
        for index, metric in enumerate(("F1", "AUC", "EER")):
            row[f"{metric}_lower"], row[f"{metric}_upper"] = np.percentile(array[:, index], [2.5, 97.5])
        intervals.append(row)
    write_csv(TABLES / "table_11_bootstrap_ci.csv", intervals)
    correct = {name: values[name]["pred"] == labels for name in NAMES}
    pairs = []
    for left, right in itertools.combinations(NAMES, 2):
        n01 = int(np.sum(~correct[left] & correct[right]))
        n10 = int(np.sum(correct[left] & ~correct[right]))
        total = n01 + n10
        p_value = float(binomtest(n01, total, .5).pvalue) if total else 1.
        log_value = min(0., float(np.log(2) + logsumexp(binom.logpmf(np.arange(min(n01, n10) + 1), total, .5)))) if total else 0.
        lower, upper = np.percentile(np.asarray(draws[left])[:, 0] - np.asarray(draws[right])[:, 0], [2.5, 97.5])
        pairs.append(dict(Model_A=DISPLAY[left], Model_B=DISPLAY[right], A_wrong_B_right=n01, A_right_B_wrong=n10, McNemar_p=p_value, McNemar_log10_p=log_value / np.log(10), F1_difference_lower=lower, F1_difference_upper=upper))
    previous = -np.inf
    for rank, index in enumerate(sorted(range(len(pairs)), key=lambda item: pairs[item]["McNemar_log10_p"])):
        previous = max(previous, min(0., pairs[index]["McNemar_log10_p"] + np.log10(len(pairs) - rank)))
        pairs[index]["Holm_adjusted_log10_p"] = previous
        pairs[index]["Holm_adjusted_p"] = 10. ** previous
    write_csv(TABLES / "table_12_pairwise_full_test.csv", pairs)


def save(fig, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flow(filename: str, title: str, blocks: list[str]) -> None:
    fig, axis = plt.subplots(figsize=(10, 1.05 * len(blocks)))
    axis.axis("off")
    for index, block in enumerate(blocks):
        y_value = 1 - (index + .5) / len(blocks)
        axis.text(.5, y_value, block, ha="center", va="center", bbox=dict(boxstyle="round,pad=.45", facecolor="#F4F8FC", edgecolor="#0072B2"))
        if index:
            axis.annotate("", (.5, y_value + .035), (.5, y_value + 1 / len(blocks) - .035), arrowprops=dict(arrowstyle="->", color="#444"))
    axis.set_title(title, fontsize=13, weight="bold")
    save(fig, filename)


def publication_figures() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    flow("dataset_integrity_pipeline.png", "Checksum-aware canonical dataset construction", ["Raw audio inventory", "SHA-256 checksum equivalence groups", "Cross-label conflict quarantine", "Same-label canonical representative", "Seeded stratified split", "Checksum-group-disjoint train / validation / test"])
    flow("lightweight_temporal_pipeline.png", "Shared lightweight spectro-temporal inference", ["Mono / 22,050 Hz / 3.0 s", "Six chronological 0.5-s segments", "STFT / 128-band log-Mel / 224×224×3", "TimeDistributed CNN embedding", "LSTM(128) / Dense(64) / Dropout(0.4)", "Sigmoid P(FAKE)"])
    flow("training_provenance_strategies.png", "Heterogeneous training and artifact provenance", ["ImageNet: MobileNet warm-up → fine-tune; EfficientNet warm-up checkpoint", "Scratch: MnasNet and ShuffleNet end-to-end from epoch 1", "External: RawNet2 and AASIST → ONNX parity → score adapter", "Common artifact, threshold, and P(FAKE) evaluation boundary"])
    clean = read_csv(TABLES / "table_6_clean_performance.csv")
    positions, width = np.arange(6), .22
    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, key, label in [(-width, "F1_FAKE", "F1"), (0, "Macro_F1", "Macro-F1"), (width, "ROC_AUC", "ROC-AUC")]:
        axis.bar(positions + offset, [float(row[key]) for row in clean], width, label=label)
    axis.set_ylim(0, 1.05)
    axis.set_xticks(positions, [row["Model"] for row in clean], rotation=25, ha="right")
    axis.set_ylabel("Score (higher is better)")
    axis.legend(ncols=3)
    save(fig, "clean_metric_grouped_6_models.png")
    summaries = read_csv(OUT6 / "tables/table_2_clean_6_models.csv")
    maximum = max(max(int(row[key]) for key in ("tn", "fp", "fn", "tp")) for row in summaries)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), constrained_layout=True)
    for axis, row in zip(axes.flat, summaries):
        matrix = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for i, j in itertools.product(range(2), repeat=2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > maximum / 2 else "black")
        axis.set_title(DISPLAY[row["Model"]])
        axis.set_xticks([0, 1], ["REAL", "FAKE"])
        axis.set_yticks([0, 1], ["REAL", "FAKE"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.colorbar(image, ax=axes, shrink=.75, label="Samples")
    save(fig, "confusion_matrix_panel_6_models.png")


def inventory() -> None:
    roots = [ROOT / name for name in ("models", "docs", "papers", "benchmark", "src", "configs", "data/manifests")]
    roots += [OUT5 / "protocol", OUT5 / "clean", OUT5 / "diagnostic_100", OUT5 / "efficiency", OUT6]
    rows = []
    for base in roots:
        if not base.exists():
            continue
        for path in ([base] if base.is_file() else base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
                continue
            relative = path.relative_to(ROOT).as_posix()
            role = "main evidence" if any(token in relative for token in ("lava_6", "manifest", "metadata.json", "threshold")) else "appendix/support"
            rows.append(dict(Path=relative, Type=path.suffix.lower() or "binary", Bytes=path.stat().st_size, SHA256=sha(path), Current="archive" not in path.parts, Paper_use=role))
    write_csv(PAPER / "REPOSITORY_EVIDENCE_INVENTORY.csv", rows)
    lines = ["# Repository Evidence Inventory", "", f"Generated from {len(rows)} evidence/support files without loading a detector.", "", "The CSV companion records path, type, size, SHA-256, currency, and paper role. Vendored conversion runtimes and archives are excluded because they are not publication inputs.", "", "| Scope | Files |", "|---|---:|"]
    for prefix in ("models/", "data/manifests/", "outputs/lava_5/", "outputs/lava_6/", "docs/", "papers/", "benchmark/", "src/"):
        lines.append(f"| `{prefix}` | {sum(row['Path'].startswith(prefix) for row in rows)} |")
    (PAPER / "REPOSITORY_EVIDENCE_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    dataset_tables()
    detector_tables()
    result_tables()
    full_test_statistics()
    publication_figures()
    inventory()
    print(json.dumps({"status": "PASS", "tables": len(list(TABLES.glob("*.csv"))), "figures": len(list(FIGURES.glob("*.png"))), "statistics_scope": "full canonical test", "model_inference": False, "training": False}, indent=2))


if __name__ == "__main__":
    main()
