"""Figures/tables/statistics from verified experimental scores only."""
import itertools
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, binom, norm
from scipy.special import logsumexp
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, roc_auc_score

from benchmark.lava5 import MODELS, read_csv, write_csv, sha256, metrics, verify_protocol
from src.lava.artifacts import write_json_atomic
from src.lava.evaluation_metrics import compute_eer
from benchmark.pareto import pareto_frontier

LABELS = {"mobilenetv3_lstm": "MobileNetV3", "efficientnet_b0_lstm": "EfficientNet-B0", "mnasnet_lstm": "MnasNet-A1", "rawnet2": "RawNet2 (external)", "aasist": "AASIST (external)"}
COLORS = dict(zip(MODELS, ["#2474B5", "#E69F00", "#009E73", "#CC4678", "#7758A6"]))
_DIAGNOSTIC = False


def markdown_blocks(blocks):
    """Separate prose blocks without breaking contiguous Markdown table rows."""
    result = ""
    previous = ""
    for block in blocks:
        if not block:
            continue
        separator = "\n" if previous.startswith("|") and block.startswith("|") else "\n\n"
        result += (separator if result else "") + block
        previous = block
    return result + "\n"


def load_result(folder, canonical, diagnostic=False):
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    if summary["status"] != ("DIAGNOSTIC" if diagnostic else "BENCHMARKED") or summary["samples"] != len(canonical):
        raise ValueError(f"Not a full benchmark: {folder}")
    if sha256(folder / "scores.csv") != summary["scores_sha256"]:
        raise ValueError(f"Score digest mismatch: {folder}")
    rows = read_csv(folder / "scores.csv")
    if any(float(r["threshold"]) != summary["threshold"] for r in rows):
        raise ValueError("Per-sample threshold differs from result summary")
    if [r["sample_id"] for r in rows] != [r["sample_id"] for r in canonical]:
        raise ValueError("Sample alignment mismatch")
    y = np.array([int(r["true_label"]) for r in rows])
    p = np.array([float(r["p_fake"]) for r in rows])
    np.testing.assert_array_equal(y, [int(r["label"]) for r in canonical])
    recomputed = metrics(y, p, summary["threshold"])
    for key, value in recomputed.items():
        if not np.isclose(value, summary[key], atol=1e-12):
            raise ValueError(f"Metric mismatch: {folder}/{key}")
    pred = (p >= summary["threshold"]).astype(int)
    np.testing.assert_array_equal(pred, [int(r["predicted_label"]) for r in rows])
    np.testing.assert_array_equal(pred == y, [bool(int(r["correct"])) for r in rows])
    return dict(summary=summary, y=y, p=p, pred=pred, rows=rows)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if _DIAGNOSTIC:
        fig.text(.5, -.01, "DIAGNOSTIC SUBSET - NOT FULL TEST ROBUSTNESS", ha="center", fontsize=9, color="#B3261E")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bar(figures, filename, names, values, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([LABELS[n] for n in names], values, color=[COLORS[n] for n in names])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=.18)
    save(fig, figures / filename)


def matrix_figure(figures, filename, names, values, title, limits=None, columns=None):
    fig, ax = plt.subplots(figsize=(max(7, len(columns or names) * .9), 5))
    options = dict(vmin=limits[0], vmax=limits[1]) if limits else {}
    im = ax.imshow(values, cmap="viridis", aspect="auto", **options)
    ax.set_xticks(range(len(columns or names)), columns or [LABELS[n] for n in names], rotation=35, ha="right")
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    for i in range(len(names)):
        for j in range(len(values[i])):
            red, green, blue, _ = im.cmap(im.norm(values[i][j]))
            ink = "#111111" if .2126 * red + .7152 * green + .0722 * blue > .55 else "white"
            ax.text(j, i, f"{values[i][j]:.3f}", ha="center", va="center", color=ink, fontsize=8)
    save(fig, figures / filename)


def diagrams(figures):
    chains = {
        "mobilenetv3_lstm": ["Audio: 3 s / 22.05 kHz", "6 chronological Mel images\n224 x 224 x 3", "MobileNetV3Small\nshared across segments", "LSTM(128) / Dense(64)\nDropout / sigmoid"],
        "efficientnet_b0_lstm": ["Audio: 3 s / 22.05 kHz", "6 chronological Mel images\n224 x 224 x 3", "EfficientNet-B0\nshared across segments", "LSTM(128) / Dense(64)\nDropout / sigmoid"],
        "mnasnet_lstm": ["Audio: 3 s / 22.05 kHz", "6 chronological Mel images\n224 x 224 x 3", "MnasNet-A1\nshared across segments", "LSTM(128) / Dense(64)\nDropout / sigmoid"],
        "rawnet2": ["Waveform: 64600 / 16 kHz", "Sinc front end\nResidual 1D blocks", "Attention / GRU", "Classifier\nsoftmax spoof score"],
        "aasist": ["Waveform: 64600 / 16 kHz", "Sinc / residual 2D encoder", "Spectral + temporal graphs\nHeterogeneous graph attention", "Pooling / master nodes\nReadout / classifier"],
    }
    def flow(path, blocks, title):
        fig, ax = plt.subplots(figsize=(10, max(3, len(blocks) * .8)))
        ax.axis("off")
        for i, text in enumerate(blocks):
            y = 1 - (i + .5) / len(blocks)
            ax.text(.5, y, text, ha="center", va="center", fontsize=11,
                    bbox=dict(boxstyle="round,pad=.6", facecolor="#EDF4FB", edgecolor="#2474B5"))
            if i:
                ax.annotate("", xy=(.5, y + .045), xytext=(.5, y + 1 / len(blocks) - .05), arrowprops=dict(arrowstyle="->"))
        ax.set_title(title, pad=20)
        save(fig, path)
    for name, blocks in chains.items():
        flow(figures / f"{name}_architecture.png", blocks, LABELS[name] + " - inference schematic")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    boxes = [(.5, .94, "LAVA-5 INTERIM BENCHMARK"),
             (.25, .69, "LAVA-trained lightweight group\nMobileNetV3 / EfficientNet-B0 / MnasNet-A1\nMel-sequence + LSTM family"),
             (.76, .69, "External reference group\nRawNet2 / AASIST\nNative waveform architectures / ONNX"),
             (.5, .43, "Unified test samples and score contract\nDistinct training provenance and duration"),
             (.5, .23, "Clean / noise / codec / simulated replay / efficiency"),
             (.5, .04, "Measured multi-objective Pareto analysis")]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=.65", facecolor="#EDF4FB", edgecolor="#2474B5"))
    for start, end in [((.5,.89),(.25,.79)), ((.5,.89),(.76,.79)), ((.25,.59),(.5,.49)), ((.76,.59),(.5,.49)), ((.5,.37),(.5,.28)), ((.5,.18),(.5,.09))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->"))
    save(fig, figures / "lava_5_model_overview.png")
    flow(figures / "lava_benchmark_pipeline.png", ["Dataset / checksum integrity manifest", "Canonical test split (no resplitting)",
        "5 existing checkpoints / fixed thresholds", "Clean inference / shared stress generation", "Robustness / isolated efficiency",
        "Aligned scores / statistics / error analysis", "Tables / figures / Pareto / report"], "Inference-only benchmarking pipeline")


def statistics(output, clean, canonical, iterations=1000):
    names = list(clean)
    if len(names) != 5:
        return
    directory = output / "error_analysis"
    key = {n: r["summary"]["scores_sha256"] for n, r in clean.items()}
    stamp = directory / "statistics_state.json"
    state = dict(scores=key, iterations=iterations, method_version=3, figure_dpi=300)
    if stamp.exists() and json.loads(stamp.read_text()) == state:
        return
    y = clean[names[0]]["y"]
    correct = np.array([clean[n]["pred"] == y for n in names])
    agreement = np.array([[np.mean(clean[a]["pred"] == clean[b]["pred"]) for b in names] for a in names])
    overlap = np.array([[np.sum(~correct[i] & ~correct[j]) for j in range(5)] for i in range(5)])
    write_csv(directory / "agreement_matrix.csv", [dict(Model=n, **dict(zip(names, agreement[i]))) for i, n in enumerate(names)])
    for filename in ("agreement_heatmap.png", "model_agreement_heatmap.png"):
        matrix_figure(directory, filename, names, agreement, "Pairwise decision agreement", (0, 1))
    matrix_figure(directory, "error_overlap_matrix.png", names, overlap, "Shared wrong predictions (counts)")
    filters = {"all_5_wrong": ~correct.any(axis=0), "all_5_correct": correct.all(axis=0),
        "lightweight_wrong_reference_correct": (~correct[:3]).all(axis=0) & correct[3:].all(axis=0),
        "lightweight_correct_reference_wrong": correct[:3].all(axis=0) & (~correct[3:]).all(axis=0)}
    for label, mask in filters.items():
        write_csv(directory / f"{label}.csv", [dict(sample_id=canonical[i]["sample_id"], path=canonical[i]["path"], label=int(y[i])) for i in np.flatnonzero(mask)], ["sample_id", "path", "label"])
    errors = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, result in clean.items():
        for label, axis in [(0, axes[0]), (1, axes[1])]:
            mask = (y == label) & (result["pred"] != y)
            axis.hist(result["p"][mask], bins=np.linspace(0, 1, 21), histtype="step", label=LABELS[name], color=COLORS[name])
        for i in np.flatnonzero(result["pred"] != y):
            errors.append(dict(Model=name, sample_id=canonical[i]["sample_id"], true_label=int(y[i]), p_fake=float(result["p"][i]), error="FP" if y[i] == 0 else "FN"))
    axes[0].set_title("False positives: raw FAKE score")
    axes[1].set_title("False negatives: raw FAKE score")
    for axis in axes:
        axis.set_xlabel("Uncalibrated model score (not confidence)")
        axis.set_ylabel("Count")
    axes[1].legend(fontsize=7)
    save(fig, directory / "confidence_error_distribution.png")
    write_csv(directory / "errors.csv", errors, ["Model", "sample_id", "true_label", "p_fake", "error"])
    for error, reverse in [("FP", True), ("FN", False)]:
        selected = sorted([r for r in errors if r["error"] == error], key=lambda r: r["p_fake"], reverse=reverse)
        write_csv(directory / f"extreme_score_{error}.csv", selected, ["Model", "sample_id", "true_label", "p_fake", "error"])
    # One shared stratified resample per iteration preserves paired comparisons.
    rng = np.random.default_rng(42)
    indices = [np.flatnonzero(y == label) for label in (0, 1)]
    draws = {n: [] for n in names}
    for iteration in range(iterations):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in indices])
        for n in names:
            r = clean[n]
            draws[n].append([f1_score(y[idx], r["pred"][idx], zero_division=0), roc_auc_score(y[idx], r["p"][idx]), compute_eer(y[idx], r["p"][idx])[0]])
    intervals = []
    for n in names:
        draws[n] = np.asarray(draws[n])
        for j, metric in enumerate(["F1", "AUC", "EER"]):
            lo, hi = np.percentile(draws[n][:, j], [2.5, 97.5])
            intervals.append(dict(Model=n, metric=metric, lower=float(lo), upper=float(hi), iterations=iterations, method="stratified test-set percentile bootstrap; not training-seed uncertainty"))
    write_csv(directory / "bootstrap_95_ci.csv", intervals)
    pairs = []
    for a, b in itertools.combinations(range(5), 2):
        n01, n10 = int(np.sum(~correct[a] & correct[b])), int(np.sum(correct[a] & ~correct[b]))
        p = float(binomtest(n01, n01+n10, .5).pvalue) if n01+n10 else 1.0
        log_p = min(0., float(np.log(2) + logsumexp(binom.logpmf(np.arange(min(n01, n10) + 1), n01+n10, .5)))) if n01+n10 else 0.
        lo, hi = np.percentile(draws[names[a]][:, 0] - draws[names[b]][:, 0], [2.5, 97.5])
        pairs.append(dict(ModelA=names[a], ModelB=names[b], A_wrong_B_right=n01, A_right_B_wrong=n10,
                          mcnemar_exact_p=p, mcnemar_log10_p=log_p / np.log(10),
                          p_value_note="floating-point underflow; use log10 p, not literal zero" if p == 0 else "finite",
                          f1_difference_lower=float(lo), f1_difference_upper=float(hi)))
    previous_log = -np.inf
    for rank, i in enumerate(sorted(range(len(pairs)), key=lambda i: pairs[i]["mcnemar_log10_p"])):
        previous_log = max(previous_log, min(0., pairs[i]["mcnemar_log10_p"] + np.log10(len(pairs)-rank)))
        pairs[i]["holm_adjusted_log10_p"] = float(previous_log)
        pairs[i]["holm_adjusted_p"] = float(10. ** previous_log)
    write_csv(directory / "paired_comparison.csv", pairs)
    write_json_atomic(stamp, state)


def generate_report(output):
    global _DIAGNOSTIC
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    protocol = verify_protocol(output)
    _DIAGNOSTIC = protocol.get("evaluation_scope") == "DIAGNOSTIC_SUBSET"
    canonical = read_csv(output / "protocol/test_samples.csv")
    figures, tables = output / "figures", output / "tables"
    recording_formats = Counter((r["label_name"], r["sample_rate"], r["channels"]) for r in canonical)
    write_csv(tables / "dataset_recording_formats.csv", [dict(Label=label, SampleRate=rate, Channels=channels, Samples=count) for (label, rate, channels), count in sorted(recording_formats.items())])
    clean = {n: r for n in MODELS if (r := load_result(output / "clean" / n, canonical, _DIAGNOSTIC)) is not None}
    audits_by_name = {a["model"]: a for a in protocol["models"]}
    for name, result in clean.items():
        if result["summary"]["threshold"] != audits_by_name[name]["threshold"] or result["summary"]["manifest_hash"] != protocol["manifest"]["manifest_hash"]:
            raise ValueError("Clean result threshold/manifest differs from sealed protocol")
    names = list(clean)
    for metric, filename, label in [("accuracy", "accuracy", "Accuracy"), ("f1", "f1", "FAKE F1"), ("macro_f1", "macro_f1", "Macro F1"), ("roc_auc", "roc_auc", "ROC AUC"), ("eer", "eer", "EER")]:
        if names:
            bar(figures, filename + "_comparison.png", names, [clean[n]["summary"][metric] for n in names], label)
    for curve in ("roc", "pr", "det"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for n, r in clean.items():
            if curve == "pr":
                precision, recall, _ = precision_recall_curve(r["y"], r["p"])
                x, y = recall, precision
            else:
                x, tpr, _ = roc_curve(r["y"], r["p"])
                y = tpr if curve == "roc" else 1 - tpr
                if curve == "det":
                    x, y = norm.ppf(np.clip(x, 1e-4, 1-1e-4)), norm.ppf(np.clip(y, 1e-4, 1-1e-4))
            ax.plot(x, y, label=LABELS[n], color=COLORS[n])
        ax.set_xlabel("Recall" if curve == "pr" else "False-positive rate" + (" (normal deviate)" if curve == "det" else ""))
        ax.set_ylabel({"pr": "Precision", "roc": "True-positive rate", "det": "False-negative rate (normal deviate)"}[curve])
        ax.set_title(f"{curve.upper()} - {len(clean)} measured detectors")
        if names:
            ax.legend(fontsize=8)
            save(fig, figures / f"{curve}_comparison_{len(clean)}_models.png")
        else:
            plt.close(fig)
    for n, r in clean.items():
        s = r["summary"]
        fig, ax = plt.subplots(figsize=(4, 4))
        values = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        ax.imshow(values, cmap="Blues", vmin=0, vmax=len(canonical))
        for i, j in itertools.product(range(2), repeat=2):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["REAL", "FAKE"])
        ax.set_yticks([0, 1], ["REAL", "FAKE"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(LABELS[n] + f"; threshold={s['threshold']:.2f}")
        save(fig, output / "clean" / n / "confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(r["y"], r["p"])
        ax.plot(fpr, tpr, color=COLORS[n])
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=LABELS[n])
        save(fig, output / "clean" / n / "roc_curve.png")
    master, robustness_rows, efficiency_rows, specs = [], [], [], []
    condition_results = {}
    for audit in protocol["models"]:
        n = audit["model"]
        load_path = output / "protocol" / f"{n}_load.json"
        load = json.loads(load_path.read_text()) if load_path.exists() else {}
        native_path = output / "protocol" / f"{n}_native_audit.json"
        native = json.loads(native_path.read_text()) if native_path.exists() else {}
        params = native.get("parameter_count", load.get("parameter_count"))
        specs.append(dict(Model=n, Group=audit["group"], Framework=audit["framework"], Input=audit["input_type"], Duration=audit["duration"], Params=params,
                          Artifact=audit["artifact"], Format=Path(audit["artifact"]).suffix,
                          Threshold=audit["threshold"], ThresholdSource=audit["threshold_source"], LoadStatus=load.get("load_status", "NOT_RUN"),
                          ThresholdPath=audit["threshold_path"], MetadataPath=audit["metadata_path"],
                          TrainedWithLAVA=audit["trained_with_lava_pipeline"],
                          OriginalTrainingDataset=native.get("original_training_dataset", audit["original_training_dataset"]),
                          Architecture=audit["source_metadata"].get("architecture", "native waveform / Sinc / residual / GRU" if n == "rawnet2" else "native waveform / Sinc / encoder / spectral-temporal heterogeneous graph attention"),
                          BenchmarkEligible=load.get("load_status") == "PASS",
                          Source=audit["checkpoint_origin"], TrainingManifest=audit["manifest_compatibility"], Initialization=audit["source_metadata"].get("initialization", audit["source_metadata"].get("pretraining", "external")),
                          ParameterCountSource=native.get("source", load.get("parameter_count_source", "NOT_RUN"))))
        s = clean[n]["summary"] if n in clean else {}
        row = dict(Model=n, Group=audit["group"], Framework=audit["framework"], ArtifactSource=audit["artifact"], TrainingProvenance=audit["checkpoint_origin"],
                   CleanAccuracy=s.get("accuracy"), CleanF1=s.get("f1"), MacroF1=s.get("macro_f1"), AUC=s.get("roc_auc"), EER=s.get("eer"),
                   NoiseDeg=None, CompressionDeg=None, ReplayDeg=None, MeanRobustnessDeg=None,
                   Params=params, SizeMB=audit["size_bytes"] / 1024**2, MemoryMB=None, LatencyMeanMs=None, LatencyP95Ms=None, Throughput=None, RTF=None, ParetoFront=None, Status="NOT_RUN")
        for suite, column, expected in [("noise", "NoiseDeg", 4), ("compression", "CompressionDeg", 4), ("replay", "ReplayDeg", 1)]:
            conditions = sorted((output / "protocol/conditions" / suite).glob("*.csv"))
            degradation = []
            for path in conditions:
                result = load_result(output / "robustness" / suite / path.stem / n, canonical, _DIAGNOSTIC)
                if result is not None and s:
                    if result["summary"]["threshold"] != audit["threshold"] or result["summary"]["manifest_hash"] != protocol["manifest"]["manifest_hash"]:
                        raise ValueError("Stress result threshold/manifest differs from sealed protocol")
                    condition_results[(n, suite, path.stem)] = result["summary"]
                    degradation.append(s["f1"] - result["summary"]["f1"])
            if len(degradation) == expected:
                row[column] = float(np.mean(degradation))
        if all(row[k] is not None for k in ("NoiseDeg", "CompressionDeg", "ReplayDeg")):
            # Mean over every completed stress condition, not arbitrary category weights.
            row["MeanRobustnessDeg"] = float(np.mean([s["f1"] - value["f1"] for (model, _, _), value in condition_results.items() if model == n]))
        ep = output / "efficiency" / n / "summary.json"
        if ep.exists():
            e = json.loads(ep.read_text())
            row.update(MemoryMB=e["peak_sampled_rss_mb"], LatencyMeanMs=e["end_to_end"]["mean_ms"], LatencyP95Ms=e["end_to_end"]["p95_ms"], Throughput=e["throughput"], RTF=e["rtf"])
            efficiency_rows.append(dict(Model=n, Params=params, SizeMiB=row["SizeMB"], MemoryMiB=row["MemoryMB"],
                PreprocessMs=e["preprocessing"]["mean_ms"], ModelOnlyMs=e["model_only"]["mean_ms"], EndToEndMs=row["LatencyMeanMs"], P95Ms=row["LatencyP95Ms"], Throughput=row["Throughput"], RTF=row["RTF"], LoadSeconds=e["load_seconds"]))
        row["Status"] = ("DIAGNOSTIC" if _DIAGNOSTIC else "BENCHMARKED") if s and row["MeanRobustnessDeg"] is not None and ep.exists() else ("DIAGNOSTIC_PARTIAL" if _DIAGNOSTIC else "CLEAN_AND_EFFICIENCY" if ep.exists() else "CLEAN_ONLY") if s else "NOT_RUN"
        master.append(row)
        robustness_rows.append({k: row[k] for k in ("Model", "CleanF1", "NoiseDeg", "CompressionDeg", "ReplayDeg", "MeanRobustnessDeg")})
    write_csv(tables / "table_1_detector_specification.csv", specs)
    write_csv(output / "protocol/artifact_audit.csv", specs)
    if clean:
        write_csv(tables / "table_2_clean.csv", [dict(Model=n, **clean[n]["summary"]) for n in names])
    write_csv(tables / "table_3_robustness.csv", robustness_rows)
    if efficiency_rows:
        write_csv(tables / "table_4_efficiency.csv", efficiency_rows)
    for column, filename, label in [("Params", "parameters_bar.png", "Parameters (native or Keras; see count convention)"), ("SizeMB", "model_size_bar.png", "Inference artifact size (MiB)"),
        ("MemoryMB", "memory_bar.png", "Sampled process peak RSS (MiB)"), ("LatencyMeanMs", "end_to_end_latency_bar.png", "End-to-end latency (ms)"), ("Throughput", "throughput_bar.png", "Recordings / second"), ("RTF", "rtf_bar.png", "RTF (native model duration denominator)")]:
        available = [r for r in master if r[column] is not None]
        if available:
            bar(figures, filename, [r["Model"] for r in available], [r[column] for r in available], label)
    if efficiency_rows:
        bar(figures, "inference_latency_bar.png", [r["Model"] for r in efficiency_rows], [r["ModelOnlyMs"] for r in efficiency_rows], "Warm model-only latency (ms)")
    stress_figures(figures, master, clean, condition_results)
    load_checks = [json.loads((output / "protocol" / f"{n}_load.json").read_text()) if (output / "protocol" / f"{n}_load.json").exists() else {} for n in MODELS]
    native_checks = [json.loads((output / "protocol" / f"{n}_native_audit.json").read_text()) if (output / "protocol" / f"{n}_native_audit.json").exists() else {} for n in ("rawnet2", "aasist")]
    complete = (all(r["Status"] == ("DIAGNOSTIC" if _DIAGNOSTIC else "BENCHMARKED") for r in master)
                and all(r.get("load_status") == "PASS" and r.get("adapter_parity") == "PASS" for r in load_checks)
                and all(r.get("status") == "PASS" for r in native_checks))
    if complete:
        frontier = {r["Model"] for r in pareto_frontier(master, {"EER": "min", "MeanRobustnessDeg": "min", "RTF": "min"})}
        for row in master:
            row["ParetoFront"] = row["Model"] in frontier
        write_csv(output / "pareto/pareto_results.csv", master)
        write_csv(tables / "table_5_pareto.csv", [{k: r[k] for k in ("Model", "EER", "MeanRobustnessDeg", "RTF", "ParetoFront")} for r in master])
        objectives = ["EER", "MeanRobustnessDeg", "RTF"]
        dominance = [dict(Model=a["Model"], **{b["Model"]: int(all(a[k] <= b[k] for k in objectives) and any(a[k] < b[k] for k in objectives)) for b in master}) for a in master]
        write_csv(output / "pareto/dominance_matrix.csv", dominance)
        pareto_figures(output, master)
    write_csv(output / "lava_5_results.csv", master)
    statistics(output, clean, canonical)
    diagrams(figures)
    status = ("LAVA-5 DIAGNOSTIC SUBSET COMPLETE" if complete else "LAVA-5 DIAGNOSTIC SUBSET - PARTIAL EXECUTION") if _DIAGNOSTIC else ("LAVA-5 INTERIM BENCHMARK COMPLETE" if complete else "LAVA-5 INTERIM BENCHMARK - PARTIAL EXECUTION")
    report = [f"# {status}", "", "## Protocol and provenance", "",
        f"{'Diagnostic subset of canonical test' if _DIAGNOSTIC else 'Canonical test'}: {len(canonical)} samples. Manifest: `{protocol['manifest']['manifest_hash']}`.",
        "No retraining, no test-set threshold fitting, no resplitting. REAL=0, FAKE=1; score >= stored threshold means FAKE.",
        "Three LAVA-trained lightweight models and two externally pretrained reference anti-spoofing models. ShuffleNetV2 is TRAINING_PENDING and excluded from every experimental table and plot.",
        "MobileNet: ImageNet initialization and two-stage training. EfficientNet: selected warm-up checkpoint, not a completed fine-tune lifecycle. MnasNet: scratch initialization, best early-stopped checkpoint. Training budgets/pretraining differ; this is NOT an isolated architecture/pretraining comparison.",
        "RawNet2/AASIST: external native PyTorch checkpoints, self-contained ONNX inference. Reference READMEs describe ASVspoof 2019 LA training; no original training logs or LAVA-train overlap audit is available. See native checkpoint audit JSONs.",
        "Lightweight input: 3 s, 22.05 kHz, six chronological 224x224x3 Mel images. Reference input: 64600 samples / 16 kHz = 4.0375 s, native waveform front ends. Native duration and framework differences remain explicit caveats.",
        "Important reference preprocessing deviation: existing LAVA adapters use scipy resample_poly and prefix/zero-padding; reference data_utils uses librosa and repetition padding for short clips. The current adapters are retained unchanged. Native-checkpoint/ONNX parity uses identical adapter tensors and does NOT prove equivalence to original-paper preprocessing. These are LAVA-adapted external-reference results, not paper-result reproductions.",
        "Thresholds remain unchanged: lightweight validation-F1 thresholds; reference defaults 0.5, NOT LAVA-calibrated. Scores are NOT demonstrated to be calibrated real-world probabilities. EER threshold is diagnostic only and is never deployed.",
        "", "## Result tables", ""]
    report += ["| Model | Accuracy | F1 | AUC | EER | Status |", "|---|---:|---:|---:|---:|---|"]
    for r in master:
        val = lambda k: "NOT_RUN" if r[k] is None else f"{r[k]:.6f}"
        report.append(f"| {r['Model']} | {val('CleanAccuracy')} | {val('CleanF1')} | {val('AUC')} | {val('EER')} | {r['Status']} |")
    report += ["", "CSV tables are generated from checked per-sample scores, artifact metadata and measured runtime JSONs; empty CSV fields mean unavailable, never zero.",
        "", "## Robustness", "", "Noise: synthetic white Gaussian noise at 20/10/5/0 dB. This is not recorded environmental background noise. One seeded waveform per sample/condition reused across all models. FLOAT WAV prevents clipping. Whole retained-prefix RMS defines SNR (including silence).",
        "Compression: FFmpeg MP3 128/64 kbps, Opus 64 kbps, AAC 96 kbps; PCM16 codec input. Codec delay, rate conversion, quantization and channel effects are part of the practical round-trip condition. Clean inference uses the untouched original, not codec-normalized audio.",
        "Replay: synthetic direct/echo taps and band limitation documented in protocol/replay_generation.json; NOT measured-RIR or physical replay. Physical replay NOT_AVAILABLE. Unseen/cross-dataset NOT_AVAILABLE; no verified labeled external dataset supplied.",
        "Stress retains up to 4.1 seconds at original rate, mono channel mean, then each unchanged adapter applies its own input policy. Mean robustness degradation is the arithmetic mean of F1_clean minus F1_condition over nine stress conditions. Negative values mean improvement; no clipping or arbitrary score.",
        "", "## Efficiency and Pareto", "", "CPU, float32, batch1, one computational thread per backend, sequential isolated processes; 10 warm-ups and 50 measured repetitions per component. Returned arrays synchronize CPU work. Load and process startup excluded from warm timings; graph tracing excluded. Whole-worker RSS (20ms samples) is not isolated model memory. TF graph and ONNX kernels differ. RTF uses each model's native input duration, so latency is also reported directly.",
        "TensorFlow count includes nontrainable BN state; native PyTorch count covers parameters and buffers are reported separately. Serialized size is the inference artifact; companion MobileNet fallback weights are additionally fingerprinted in protocol (deployment bundle can be larger).",
        "Pareto minimizes EER, mean F1 degradation, end-to-end RTF. It is generated only when all five have complete selected objectives. No arbitrary weighted ranking. Radar uses within-five-model min-max normalization and is visualization only, not an independent ranking.",
        "", "## Error analysis and uncertainty", "", "Paired sample IDs checked before aggregation. Error CSVs distinguish all-five errors and strict unanimous lightweight/reference disagreements. Extreme-score errors are NOT calibrated high-confidence claims.",
        "1000 stratified test-set bootstrap resamples, seed42, percentile95% F1/AUC/EER intervals; paired resamples for F1 differences. Exact McNemar with Holm correction over ten pairs. These are test-set uncertainty, NOT multi-seed training uncertainty; speaker/source dependence is unknown.",
        "", "## Limitations and acceptance", "", "Only checksum-group-disjoint is supported; speaker/generator/source IDs are UNKNOWN. Recording sample-rate/channel distributions by class are reported separately; source/format confounding cannot be ruled out. Repeated inspection of this test set in prior development may weaken independent hold-out claims. No six-model or FULL LAVA claim is supported.",
        "Current completion status is derived from full score files and required measured suites. No synthetic example numbers enter these results.", "", "## Figures", ""]
    if _DIAGNOSTIC:
        report.insert(2, "DIAGNOSTIC: user-selected fixed stratified subset, seed42, no selection using predictions. Clean baseline is derived from the exact same IDs in the full clean run. Degradation and exploratory Pareto here apply only to this subset, NOT full test robustness. Do not copy these metrics into official full-test robustness columns.")
    elif (output / "protocol/robustness_execution_plan.json").exists():
        plan = json.loads((output / "protocol/robustness_execution_plan.json").read_text())
        if plan.get("scope") == "DIAGNOSTIC_SUBSET":
            report.insert(2, f"Full test clean evaluation is retained. User selected robustness on a fixed {plan['samples']}-sample diagnostic subset first. Full-test robustness and official three-objective Pareto remain NOT_RUN. See [separate diagnostic report](../diagnostic_{plan['samples']}/report/LAVA_5_BENCHMARK_REPORT.md); its degradations use the matched subset clean baseline.")
    for audit in protocol["models"]:
        for limitation in audit["source_metadata"].get("limitations", []):
            report.insert(2, f"Artifact caveat ({audit['model']}): {limitation}")
    report += [f"![{p.stem}](../figures/{p.name})" for p in sorted(figures.glob("*.png"))
               if "_comparison_" not in p.stem or "_models" not in p.stem or p.stem.endswith(f"_{len(clean)}_models")]
    report_path = output / "report/LAVA_5_BENCHMARK_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_blocks(report), encoding="utf-8")
    write_json_atomic(output / "report/acceptance.json", dict(status=status, scope="DIAGNOSTIC_SUBSET" if _DIAGNOSTIC else "CANONICAL_TEST", samples=len(canonical), clean_models=len(clean), completed_models=sum(r["Status"] in {"BENCHMARKED", "DIAGNOSTIC"} for r in master), shuffle_excluded=True, retraining=False, full_acceptance=complete and not _DIAGNOSTIC))
    print(status, flush=True)


def stress_figures(figures, master, clean, results):
    for suite, conditions in [("noise", ["snr_20", "snr_10", "snr_5", "snr_0"]), ("compression", ["mp3_128k", "mp3_64k", "opus_64k", "aac_96k"]), ("replay", ["synthetic_channel"])]:
        available = [n for n in clean if all((n, suite, c) in results for c in conditions)]
        if not available:
            continue
        for metric in ("f1", "roc_auc", "eer"):
            fig, ax = plt.subplots(figsize=(8, 4))
            for n in available:
                labels = ["Clean"] + (["20", "10", "5", "0"] if suite == "noise" else conditions)
                ax.plot(labels, [clean[n]["summary"][metric]] + [results[n, suite, c][metric] for c in conditions], "o-", color=COLORS[n], label=LABELS[n])
            if suite == "noise":
                ax.set_xlabel("AWGN SNR (dB)")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=15)
            ax.legend(fontsize=8)
            prefix = "codec" if suite == "compression" else suite
            suffix = "vs_snr" if suite == "noise" else "comparison"
            save(fig, figures / f"{prefix}_{'auc' if metric == 'roc_auc' else metric}_{suffix}.png")
        degradation = np.array([[clean[n]["summary"]["f1"] - results[n, suite, c]["f1"] for c in conditions] for n in available])
        bar(figures, ("codec" if suite == "compression" else suite) + "_degradation_bar.png", available, degradation.mean(axis=1), "F1 clean minus stress (lower better)")
        if suite == "compression":
            matrix_figure(figures, "codec_degradation_heatmap.png", available, degradation, "Codec F1 degradation", columns=conditions)
    observed = {(s, c) for _, s, c in results}
    columns = [pair for pair in [("noise", f"snr_{v}") for v in (20, 10, 5, 0)]
               + [("compression", v) for v in ("mp3_128k", "mp3_64k", "opus_64k", "aac_96k")]
               + [("replay", "synthetic_channel")] if pair in observed]
    available = [n for n in clean if columns and all((n, s, c) in results for s, c in columns)]
    if available:
        values = [[clean[n]["summary"]["f1"] - results[n, s, c]["f1"] for s, c in columns] for n in available]
        matrix_figure(figures, "robustness_heatmap.png", available, values, "F1 degradation vs untouched clean", columns=[c for _, c in columns])
    for x, y, size, filename in [("Params", "LatencyMeanMs", None, "parameter_vs_latency_scatter.png"), ("SizeMB", "EER", None, "size_vs_eer_scatter.png"),
                                 ("EER", "LatencyMeanMs", "SizeMB", "bubble_eer_latency.png"), ("MeanRobustnessDeg", "CleanF1", "Params", "bubble_robustness_f1.png")]:
        available = [r for r in master if r[x] is not None and r[y] is not None]
        if available:
            fig, ax = plt.subplots(figsize=(8, 5))
            maximum = max(r[size] for r in available) if size else 1
            for r in available:
                ax.scatter(r[x], r[y], s=80 + (r[size] / maximum * 600 if size else 0), color=COLORS[r["Model"]], alpha=.7)
                ax.annotate(LABELS[r["Model"]], (r[x], r[y]), fontsize=8)
            ax.set(xlabel=x, ylabel=y, title=f"Bubble area encodes {size}" if size else "Measured model comparison")
            save(fig, figures / filename)


def pareto_figures(output, master):
    for x, filename in [("EER", "pareto_2d_eer_rtf.png"), ("MeanRobustnessDeg", "pareto_2d_robustness_rtf.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in master:
            ax.scatter(r[x], r["RTF"], marker="*" if r["ParetoFront"] else "o", s=140, color=COLORS[r["Model"]], label=LABELS[r["Model"]])
        ax.set(xlabel=x, ylabel="End-to-end RTF", title="Stars: three-objective non-dominated set (2D projection)")
        ax.legend(fontsize=8)
        save(fig, output / "pareto" / filename)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for r in master:
        ax.scatter(r["EER"], r["MeanRobustnessDeg"], r["RTF"], color=COLORS[r["Model"]], marker="*" if r["ParetoFront"] else "o", s=90, label=LABELS[r["Model"]])
    ax.set(xlabel="EER", ylabel="Mean F1 degradation", zlabel="RTF")
    ax.legend(fontsize=7)
    save(fig, output / "pareto/pareto_3d.png")
    columns = ["CleanF1", "MeanRobustnessDeg", "LatencyMeanMs", "SizeMB", "MemoryMB"]
    if all(r[k] is not None for r in master for k in columns):
        values = np.array([[r[k] for k in columns] for r in master])
        span = np.ptp(values, axis=0)
        normed = np.divide(values-values.min(axis=0), span, out=np.full_like(values, .5), where=span != 0)
        normed[:, 1:] = 1 - normed[:, 1:]
        angles = np.linspace(0, 2*np.pi, len(columns), endpoint=False)
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "polar"})
        for i, r in enumerate(master):
            ax.plot(np.r_[angles, angles[0]], np.r_[normed[i], normed[i, 0]], color=COLORS[r["Model"]], label=LABELS[r["Model"]])
        ax.set_xticks(angles, ["F1", "Robustness", "Latency", "Size", "Memory"])
        ax.set_ylim(0, 1)
        ax.set_title("Within-five min-max utility; visualization only, not ranking", pad=30)
        ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=8)
        save(fig, output / "figures/radar_5_models.png")
        write_json_atomic(output / "protocol/radar_normalization.json", dict(columns=columns, minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist(), direction=["higher", "lower", "lower", "lower", "lower"], constant_value=.5, primary_ranking=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat[:3], ["CleanF1", "MeanRobustnessDeg", "RTF"]):
        ax.bar([LABELS[r["Model"]] for r in master], [r[key] for r in master], color=[COLORS[r["Model"]] for r in master])
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
    for r in master:
        axes[1, 1].scatter(r["EER"], r["RTF"], color=COLORS[r["Model"]], marker="*" if r["ParetoFront"] else "o")
    axes[1, 1].set(xlabel="EER", ylabel="RTF", title="3-objective Pareto projection")
    save(fig, output / "figures/composite_comparison.png")
