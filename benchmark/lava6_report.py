"""Aggregate authoritative LAVA-5 artifacts with ShuffleNet-only measurements.

This module never loads a detector.  It validates and combines stored score files,
so running it cannot re-execute the five historical detectors or train anything.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, binom, norm
from scipy.special import logsumexp
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve

import config
from benchmark.lava5 import read_csv, write_csv
from benchmark.lava5_report import load_result
from benchmark.pareto import pareto_frontier
from src.lava.artifacts import write_json_atomic
from src.lava.evaluation_metrics import compute_eer

ROOT = Path(config.BASE_DIR)
OLD = ROOT / "outputs/lava_5"
OUT = ROOT / "outputs/lava_6"
DIAG = "diagnostic_100"
NAMES = ["mobilenetv3_lstm", "efficientnet_b0_lstm", "mnasnet_lstm", "rawnet2", "aasist", "shufflenetv2_lstm"]
LIGHTWEIGHT = {"mobilenetv3_lstm", "efficientnet_b0_lstm", "mnasnet_lstm", "shufflenetv2_lstm"}
LABELS = {"mobilenetv3_lstm": "MobileNetV3", "efficientnet_b0_lstm": "EfficientNet-B0", "mnasnet_lstm": "MnasNet-A1", "rawnet2": "RawNet2 (external)", "aasist": "AASIST (external)", "shufflenetv2_lstm": "ShuffleNetV2"}
COLORS = dict(zip(NAMES, ["#2474B5", "#E69F00", "#009E73", "#CC4678", "#7758A6", "#D55E00"]))


def _save(fig, path: Path, diagnostic=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if diagnostic:
        fig.text(.5, -.01, "DIAGNOSTIC 100-SAMPLE SUBSET - NOT FULL-TEST ROBUSTNESS", ha="center", fontsize=8, color="#B3261E")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bar(path, rows, key, ylabel, diagnostic=False):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([LABELS[r["Model"]] for r in rows], [r[key] for r in rows], color=[COLORS[r["Model"]] for r in rows])
    ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=.18)
    _save(fig, path, diagnostic)


def _load_sets():
    full_rows = read_csv(OUT / "protocol/test_samples.csv")
    diag_rows = read_csv(OUT / DIAG / "protocol/test_samples.csv")
    full, diag = {}, {}
    for name in NAMES:
        full_root = OUT if name == "shufflenetv2_lstm" else OLD
        diag_root = OUT / DIAG if name == "shufflenetv2_lstm" else OLD / DIAG
        full[name] = load_result(full_root / "clean" / name, full_rows, False)
        diag[name] = load_result(diag_root / "clean" / name, diag_rows, True)
        if full[name] is None or diag[name] is None:
            raise ValueError(f"Missing clean result for {name}")
    return full_rows, diag_rows, full, diag


def _stress(diag_rows):
    values = {}
    conditions = {"noise": ["snr_20", "snr_10", "snr_5", "snr_0"], "compression": ["mp3_128k", "mp3_64k", "opus_64k", "aac_96k"], "replay": ["synthetic_channel"]}
    for name, suite in itertools.product(NAMES, conditions):
        root = OUT / DIAG if name == "shufflenetv2_lstm" else OLD / DIAG
        for condition in conditions[suite]:
            result = load_result(root / "robustness" / suite / condition / name, diag_rows, True)
            if result is None:
                raise ValueError(f"Missing diagnostic result: {name}/{suite}/{condition}")
            values[name, suite, condition] = result
    return conditions, values


def _clean_figures(full):
    figures = OUT / "figures"
    for metric, filename, label in [("accuracy", "accuracy_comparison_6_models.png", "Accuracy"), ("f1", "f1_comparison_6_models.png", "FAKE F1"), ("macro_f1", "macro_f1_comparison_6_models.png", "Macro F1"), ("roc_auc", "roc_auc_comparison_6_models.png", "ROC AUC"), ("eer", "eer_comparison_6_models.png", "EER")]:
        _bar(figures / filename, [dict(Model=n, value=full[n]["summary"][metric]) for n in NAMES], "value", label)
    for curve in ("roc", "pr", "det"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in NAMES:
            result = full[name]
            if curve == "pr":
                y, x, _ = precision_recall_curve(result["y"], result["p"])
            else:
                x, tpr, _ = roc_curve(result["y"], result["p"]); y = tpr if curve == "roc" else 1 - tpr
                if curve == "det":
                    x = norm.ppf(np.clip(x, 1e-4, 1-1e-4)); y = norm.ppf(np.clip(y, 1e-4, 1-1e-4))
            ax.plot(x, y, label=LABELS[name], color=COLORS[name])
        ax.set_xlabel("Recall" if curve == "pr" else "False-positive rate" + (" (normal deviate)" if curve == "det" else ""))
        ax.set_ylabel({"pr": "Precision", "roc": "True-positive rate", "det": "False-negative rate (normal deviate)"}[curve])
        ax.set_title(f"{curve.upper()} comparison - six detectors"); ax.legend(fontsize=8)
        _save(fig, figures / f"{curve}_comparison_6_models.png")
    name = "shufflenetv2_lstm"; result = full[name]; summary = result["summary"]; target = OUT / "clean" / name
    matrix = np.array([[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(matrix, cmap="Blues")
    for i, j in itertools.product(range(2), repeat=2): ax.text(j, i, str(matrix[i,j]), ha="center", va="center")
    ax.set_xticks([0,1], ["REAL","FAKE"]); ax.set_yticks([0,1], ["REAL","FAKE"]); ax.set(xlabel="Predicted", ylabel="True", title="ShuffleNetV2")
    _save(fig, target / "confusion_matrix.png")
    for curve in ("roc", "precision_recall", "det"):
        fig, ax = plt.subplots(figsize=(5,4))
        if curve == "precision_recall":
            precision, recall, _ = precision_recall_curve(result["y"], result["p"]); x, y = recall, precision; xlabel, ylabel = "Recall", "Precision"
        else:
            x, tpr, _ = roc_curve(result["y"], result["p"]); y = tpr; xlabel, ylabel = "False-positive rate", "True-positive rate"
            if curve == "det": y = 1-tpr; xlabel, ylabel = "False-positive rate", "False-negative rate"
        ax.plot(x, y, color=COLORS[name]); ax.set(xlabel=xlabel, ylabel=ylabel, title=f"ShuffleNetV2 {curve.replace('_',' ').upper()}")
        _save(fig, target / f"{curve}_curve.png")


def _robustness_figures(diag, conditions, stress):
    figures = OUT / "figures"
    for suite, conds in conditions.items():
        for metric in ("f1", "roc_auc", "eer"):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for name in NAMES:
                labels = ["Clean"] + (["20", "10", "5", "0"] if suite == "noise" else conds)
                vals = [diag[name]["summary"][metric]] + [stress[name,suite,c]["summary"][metric] for c in conds]
                ax.plot(labels, vals, "o-", color=COLORS[name], label=LABELS[name])
            ax.set_ylabel("AUC" if metric == "roc_auc" else metric.upper()); ax.set_xlabel("AWGN SNR (dB)" if suite == "noise" else suite.title()); ax.legend(fontsize=7)
            prefix = "codec" if suite == "compression" else suite
            suffix = "vs_snr" if suite == "noise" else "comparison"
            _save(fig, figures / f"{prefix}_{'auc' if metric == 'roc_auc' else metric}_{suffix}_6_models.png", True)
        rows = []
        for name in NAMES:
            clean_f1 = diag[name]["summary"]["f1"]
            rows.append(dict(Model=name, value=float(np.mean([clean_f1-stress[name,suite,c]["summary"]["f1"] for c in conds]))))
        _bar(figures / f"{('codec' if suite == 'compression' else suite)}_degradation_bar_6_models.png", rows, "value", "Mean F1 degradation (lower better)", True)
    cols = [(s,c) for s, cs in conditions.items() for c in cs]
    matrix = np.array([[diag[n]["summary"]["f1"]-stress[n,s,c]["summary"]["f1"] for s,c in cols] for n in NAMES])
    fig, ax = plt.subplots(figsize=(11,5)); im=ax.imshow(matrix,cmap="viridis",aspect="auto")
    ax.set_xticks(range(len(cols)),[c for _,c in cols],rotation=35,ha="right"); ax.set_yticks(range(6),[LABELS[n] for n in NAMES]); ax.set_title("Diagnostic F1 degradation"); fig.colorbar(im,ax=ax)
    _save(fig, figures / "robustness_heatmap_6_models.png", True)


def _statistics(full_rows, full, diagnostic):
    directory = OUT / "error_analysis"; directory.mkdir(parents=True, exist_ok=True)
    y = full[NAMES[0]]["y"]
    correct = np.array([full[n]["pred"] == y for n in NAMES])
    agreement=np.array([[np.mean(full[a]["pred"]==full[b]["pred"]) for b in NAMES] for a in NAMES])
    write_csv(directory/"agreement_matrix_6_models.csv",[dict(Model=n,**dict(zip(NAMES,agreement[i]))) for i,n in enumerate(NAMES)])
    fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(agreement,cmap="viridis",vmin=0,vmax=1); fig.colorbar(im,ax=ax)
    ax.set_xticks(range(6),[LABELS[n] for n in NAMES],rotation=35,ha="right"); ax.set_yticks(range(6),[LABELS[n] for n in NAMES]); ax.set_title("Pairwise decision agreement")
    _save(fig,directory/"agreement_heatmap_6_models.png")
    filters={"all_6_correct":correct.all(axis=0),"all_6_wrong":~correct.any(axis=0)}
    lw=[NAMES.index(n) for n in NAMES if n in LIGHTWEIGHT]; refs=[NAMES.index(n) for n in NAMES if n not in LIGHTWEIGHT]
    filters["lightweight_correct_reference_wrong"]=correct[lw].all(axis=0)&(~correct[refs]).all(axis=0)
    filters["lightweight_wrong_reference_correct"]=(~correct[lw]).all(axis=0)&correct[refs].all(axis=0)
    for label,mask in filters.items():
        write_csv(directory/f"{label}.csv",[dict(sample_id=full_rows[i]["sample_id"],path=full_rows[i]["path"],label=int(y[i])) for i in np.flatnonzero(mask)],["sample_id","path","label"])
    unique=[]
    for i,n in enumerate(NAMES):
        mask=(~correct[i]) & np.delete(correct,i,axis=0).all(axis=0)
        for j in np.flatnonzero(mask): unique.append(dict(Model=n,sample_id=full_rows[j]["sample_id"],true_label=int(y[j]),p_fake=float(full[n]["p"][j])))
    write_csv(directory/"model_unique_failures.csv",unique,["Model","sample_id","true_label","p_fake"])
    # Statistical rows extend the exact historical 100-sample diagnostic scope.
    # Full-clean agreement/error membership above remains based on all 2,737 IDs.
    old_ci=read_csv(OLD/DIAG/"error_analysis/bootstrap_95_ci.csv")
    dy=diagnostic[NAMES[0]]["y"]
    dcorrect=np.array([diagnostic[n]["pred"] == dy for n in NAMES])
    rng=np.random.default_rng(42); groups=[np.flatnonzero(dy==v) for v in (0,1)]; draws={n:[] for n in NAMES}
    for _ in range(1000):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        for n in NAMES:
            draws[n].append([f1_score(dy[idx],diagnostic[n]["pred"][idx],zero_division=0),roc_auc_score(dy[idx],diagnostic[n]["p"][idx]),compute_eer(dy[idx],diagnostic[n]["p"][idx])[0]])
    new_ci=[]
    for j,metric in enumerate(("F1","AUC","EER")):
        lo,hi=np.percentile(np.asarray(draws["shufflenetv2_lstm"])[:,j],[2.5,97.5]); new_ci.append(dict(Model="shufflenetv2_lstm",metric=metric,lower=float(lo),upper=float(hi),iterations=1000,method="stratified test-set percentile bootstrap; not training-seed uncertainty"))
    write_csv(directory/"bootstrap_95_ci_6_models.csv",old_ci+new_ci)
    old_pairs=read_csv(OLD/DIAG/"error_analysis/paired_comparison.csv")
    for row in old_pairs:
        row["correction"] = "historical five-model Holm-adjusted result reused unchanged"
    new_pairs=[]; sidx=NAMES.index("shufflenetv2_lstm")
    for name in NAMES[:-1]:
        a=NAMES.index(name); n01=int(np.sum(~dcorrect[a]&dcorrect[sidx])); n10=int(np.sum(dcorrect[a]&~dcorrect[sidx])); total=n01+n10
        p=float(binomtest(n01,total,.5).pvalue) if total else 1.; logp=min(0.,float(np.log(2)+logsumexp(binom.logpmf(np.arange(min(n01,n10)+1),total,.5)))) if total else 0.
        lo,hi=np.percentile(np.asarray(draws[name])[:,0]-np.asarray(draws["shufflenetv2_lstm"])[:,0],[2.5,97.5])
        new_pairs.append(dict(ModelA=name,ModelB="shufflenetv2_lstm",A_wrong_B_right=n01,A_right_B_wrong=n10,mcnemar_exact_p=p,mcnemar_log10_p=logp/np.log(10),p_value_note="floating-point underflow; use log10 p" if p==0 else "finite",f1_difference_lower=float(lo),f1_difference_upper=float(hi),correction="Holm correction within five new ShuffleNet comparisons; historical family preserved"))
    previous_log = -np.inf
    for rank, index in enumerate(sorted(range(len(new_pairs)), key=lambda i: new_pairs[i]["mcnemar_log10_p"])):
        previous_log = max(previous_log, min(0., new_pairs[index]["mcnemar_log10_p"] + np.log10(len(new_pairs) - rank)))
        new_pairs[index]["holm_adjusted_log10_p"] = float(previous_log)
        new_pairs[index]["holm_adjusted_p"] = float(10. ** previous_log)
    write_csv(directory/"paired_comparison_6_models.csv",old_pairs+new_pairs)


def _diagrams():
    figures=OUT/"figures"
    def flow(path,blocks,title):
        fig,ax=plt.subplots(figsize=(10,max(3,len(blocks)*.75))); ax.axis("off")
        for i,text in enumerate(blocks):
            y=1-(i+.5)/len(blocks); ax.text(.5,y,text,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",facecolor="#EDF4FB",edgecolor="#2474B5"))
            if i: ax.annotate("",xy=(.5,y+.04),xytext=(.5,y+1/len(blocks)-.04),arrowprops=dict(arrowstyle="->"))
        ax.set_title(title); _save(fig,path)
    flow(figures/"shufflenetv2_lstm_architecture.png",["3-second audio / 22.05 kHz","6 chronological Mel images / 224 x 224 x 3","TimeDistributed ShuffleNetV2-1.0x","Embedding sequence / LSTM(128)","Dense(64, ReLU) / Dropout(0.4)","Sigmoid P(FAKE)"],"ShuffleNetV2-LSTM inference architecture")
    flow(figures/"lava_6_model_overview.png",["LAVA six-detector benchmark","Lightweight: MobileNetV3 / ShuffleNetV2 / MnasNet / EfficientNet\nReference: RawNet2 / AASIST","Unified REAL=0, FAKE=1 score interface","Full clean / diagnostic robustness / efficiency","Diagnostic three-objective Pareto analysis"],"LAVA six-detector evaluation")


def generate():
    protocol=json.loads((OUT/"protocol/protocol.json").read_text(encoding="utf-8")); audits={a["model"]:a for a in protocol["models"]}
    full_rows,diag_rows,full,diag=_load_sets(); conditions,stress=_stress(diag_rows)
    _clean_figures(full); _robustness_figures(diag,conditions,stress); _statistics(full_rows,full,diag); _diagrams()
    tables=OUT/"tables"; figures=OUT/"figures"; master=[]; specs=[]; efficiency=[]; robustness=[]
    old_master={r["Model"]:r for r in read_csv(OLD/"lava_5_results.csv")}
    for n in NAMES:
        audit=audits[n]; meta=audit["source_metadata"]; params=int(meta.get("parameter_count") or old_master[n]["Params"])
        ep=(OUT if n=="shufflenetv2_lstm" else OLD)/"efficiency"/n/"summary.json"; e=json.loads(ep.read_text())
        clean=full[n]["summary"]; dclean=diag[n]["summary"]
        deg={suite:float(np.mean([dclean["f1"]-stress[n,suite,c]["summary"]["f1"] for c in cs])) for suite,cs in conditions.items()}
        mean=float(np.mean([dclean["f1"]-stress[n,s,c]["summary"]["f1"] for s,cs in conditions.items() for c in cs]))
        row=dict(Model=n,Group=audit["group"],Framework=audit["framework"],ArtifactSource=audit["artifact"],TrainingProvenance=audit["checkpoint_origin"],CleanAccuracy=clean["accuracy"],CleanF1=clean["f1"],MacroF1=clean["macro_f1"],AUC=clean["roc_auc"],EER=clean["eer"],DiagnosticEER=dclean["eer"],NoiseDeg=deg["noise"],CompressionDeg=deg["compression"],ReplayDeg=deg["replay"],Unseen="NOT_AVAILABLE",MeanRobustnessDeg=mean,Params=params,SizeMB=audit["size_bytes"]/1024**2,MemoryMB=e["peak_sampled_rss_mb"],LatencyMeanMs=e["end_to_end"]["mean_ms"],LatencyP95Ms=e["end_to_end"]["p95_ms"],Throughput=e["throughput"],RTF=e["rtf"],ParetoFront=None,Status="FULL_CLEAN_DIAGNOSTIC_ROBUSTNESS_EFFICIENCY")
        master.append(row); robustness.append({k:row[k] for k in ("Model","NoiseDeg","CompressionDeg","ReplayDeg","MeanRobustnessDeg")})
        efficiency.append(dict(Model=n,Params=params,SizeMiB=row["SizeMB"],MemoryMiB=row["MemoryMB"],PreprocessMs=e["preprocessing"]["mean_ms"],ModelOnlyMs=e["model_only"]["mean_ms"],EndToEndMs=row["LatencyMeanMs"],P95Ms=row["LatencyP95Ms"],Throughput=row["Throughput"],RTF=row["RTF"],LoadSeconds=e["load_seconds"]))
        specs.append(dict(Model=n,Group=audit["group"],Framework=audit["framework"],Input=audit["input_type"],Duration=audit["duration"],Architecture=meta.get("architecture","native waveform architecture"),Params=params,ArtifactOrigin=audit["checkpoint_origin"],Initialization=meta.get("initialization",meta.get("pretraining","external")),TrainingPolicy=meta.get("training_policy","external reference checkpoint"),ManifestCompatibility=audit["manifest_compatibility"]))
    frontier={r["Model"] for r in pareto_frontier(master,{"DiagnosticEER":"min","MeanRobustnessDeg":"min","RTF":"min"})}
    for r in master:r["ParetoFront"]=r["Model"] in frontier
    write_csv(OUT/"lava_6_results.csv",master); write_csv(tables/"table_1_detector_specification_6_models.csv",specs); write_csv(tables/"table_2_clean_6_models.csv",[dict(Model=n,**full[n]["summary"]) for n in NAMES]); write_csv(tables/"table_3_robustness_diagnostic_6_models.csv",robustness); write_csv(tables/"table_4_efficiency_6_models.csv",efficiency); write_csv(tables/"table_5_pareto_diagnostic_6_models.csv",[dict(Model=r["Model"],DiagnosticEER=r["DiagnosticEER"],MeanRobustnessDeg=r["MeanRobustnessDeg"],RTF=r["RTF"],ParetoFront=r["ParetoFront"]) for r in master])
    objectives=("DiagnosticEER","MeanRobustnessDeg","RTF"); dominance=[dict(Model=a["Model"],**{b["Model"]:int(all(a[k]<=b[k] for k in objectives) and any(a[k]<b[k] for k in objectives)) for b in master}) for a in master]
    write_csv(OUT/"pareto/pareto_results_6_models.csv",master); write_csv(OUT/"pareto/dominance_matrix_6_models.csv",dominance)
    for key,file,label in [("Params","parameters_bar_6_models.png","Parameters"),("SizeMB","model_size_bar_6_models.png","Artifact size (MiB)"),("MemoryMB","memory_bar_6_models.png","Peak process RSS (MiB)"),("LatencyMeanMs","end_to_end_latency_bar_6_models.png","End-to-end latency (ms)"),("Throughput","throughput_bar_6_models.png","Recordings/s"),("RTF","rtf_bar_6_models.png","RTF")]: _bar(figures/file,master,key,label)
    for x,file in [("DiagnosticEER","pareto_eer_rtf_6_models.png"),("MeanRobustnessDeg","pareto_robustness_rtf_6_models.png")]:
        fig,ax=plt.subplots(figsize=(8,5))
        for r in master: ax.scatter(r[x],r["RTF"],s=130,marker="*" if r["ParetoFront"] else "o",color=COLORS[r["Model"]],label=LABELS[r["Model"]])
        ax.set(xlabel=x,ylabel="RTF",title="Diagnostic three-objective Pareto (stars)");ax.legend(fontsize=7);_save(fig,OUT/"pareto"/file,True)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection="3d")
    for r in master:ax.scatter(r["DiagnosticEER"],r["MeanRobustnessDeg"],r["RTF"],color=COLORS[r["Model"]],marker="*" if r["ParetoFront"] else "o",label=LABELS[r["Model"]])
    ax.set(xlabel="Diagnostic-subset EER",ylabel="Mean diagnostic F1 degradation",zlabel="RTF");ax.legend(fontsize=7);_save(fig,OUT/"pareto/pareto_3d_6_models.png",True)
    report=["# LAVA Six-Detector Incremental Benchmark","","Full clean evaluation covers 2,737 canonical test samples for six detectors. Robustness is limited to the pre-existing fixed stratified 100-sample diagnostic subset; it is not full-test robustness. No model was trained, and no historical detector was re-executed.","","## Provenance","","MobileNetV3, EfficientNet-B0, MnasNet-A1, and ShuffleNetV2 are LAVA-trained lightweight detectors. RawNet2 and AASIST are externally pretrained reference checkpoints evaluated through LAVA adapters. Initializations and training provenance are heterogeneous.","","## Clean results","","| Model | Accuracy | F1 | Macro F1 | AUC | EER |","|---|---:|---:|---:|---:|---:|"]
    for r in master:report.append(f"| {LABELS[r['Model']]} | {r['CleanAccuracy']:.6f} | {r['CleanF1']:.6f} | {r['MacroF1']:.6f} | {r['AUC']:.6f} | {r['EER']:.6f} |")
    report += ["","## Robustness and efficiency","", "Noise, codec, and simulated-channel results reuse exactly the LAVA-5 stressed waveforms. Physical replay and external unseen evaluation are NOT_AVAILABLE. Pareto minimizes full-clean EER, diagnostic mean F1 degradation, and measured end-to-end RTF; it is exploratory because robustness is diagnostic.","", "| Model | Noise dF1 | Codec dF1 | Replay dF1 | Latency ms | RTF | Pareto |","|---|---:|---:|---:|---:|---:|---|"]
    for r in master:report.append(f"| {LABELS[r['Model']]} | {r['NoiseDeg']:.6f} | {r['CompressionDeg']:.6f} | {r['ReplayDeg']:.6f} | {r['LatencyMeanMs']:.3f} | {r['RTF']:.6f} | {r['ParetoFront']} |")
    report += ["","## Limitations","","The benchmark is complete for six-model clean evaluation and the selected diagnostic robustness scope, but not a full-test robustness benchmark. The external references use uncalibrated 0.5 thresholds and different native input durations. Speaker/source/generator metadata and an unseen external dataset are unavailable. Efficiency is desktop CPU evidence, not edge-device validation.","","## Figures",""]
    report += [f"![{p.stem}](../figures/{p.name})" for p in sorted(figures.glob("*.png"))]
    path=OUT/"report/LAVA_FULL_BENCHMARK_REPORT.md";path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n\n".join(report)+"\n",encoding="utf-8")
    acceptance=dict(status="PASS",scope="FULL_CLEAN_AND_DIAGNOSTIC_ROBUSTNESS",models=NAMES,clean_samples_per_model=2737,diagnostic_samples_per_condition=100,clean_model_count=6,efficiency_model_count=6,stress_model_condition_results=54,agreement_shape=[6,6],pareto_input_count=6,existing_models_reexecuted=[],no_retraining=True,shuffle_manifest="MANIFEST_MATCHED",physical_replay="NOT_AVAILABLE",unseen="NOT_AVAILABLE",full_test_robustness="NOT_RUN",official_pareto="NOT_RUN",diagnostic_pareto="PASS",six_detector_current_scope_complete=True,full_lava_benchmark_complete=False)
    write_json_atomic(OUT/"report/acceptance.json",acceptance);print(json.dumps(acceptance,indent=2))


if __name__ == "__main__": generate()
