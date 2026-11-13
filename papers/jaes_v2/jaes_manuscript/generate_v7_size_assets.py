"""Render the V7 size table and figure from the verified score analysis."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[2]
EVIDENCE = ROOT / "outputs/jaes_v2/phase10/size1280_v1/size_sensitivity.json"
ARMS = ("erm640", "channel640", "erm1280", "channel1280")
ENDPOINTS = {
    "A_clean_AUC": "A clean ROC-AUC",
    "A_macro_stress_AUC": "A macro stress ROC-AUC",
    "A_clean_FA": "A clean REAL false alarms",
    "A_clean_miss": "A clean FAKE misses",
    "A_stress_FA": "A stress REAL false alarms",
    "A_stress_miss": "A stress FAKE misses",
    "B_clean_FA": "B clean REAL false alarms",
    "A05_stress_miss": "A05 stressed FAKE misses",
    "A06_stress_miss": "A06 stressed FAKE misses",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if (data["status"], data["seed"], data["score_rows_per_model"]) != ("SIZE_SENSITIVITY_VERIFIED", 9207, 2576):
        raise ValueError("Unexpected size-sensitivity evidence")
    endpoints = data["endpoints"]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Fixed-epoch training-size sensitivity for seed 9207 on the same development sources. "
        r"Column headings give the training size. A has 112 sources per class and B has 144 REAL sources. "
        r"Pooled stress counts use three views per source at threshold 0.5. "
        r"Each attack row uses 18 sources.}\label{tab:size-sensitivity}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Development endpoint & ERM 640 & Channel 640 & ERM 1,280 & Channel 1,280 \\", r"\colrule",
    ]
    for key, label in ENDPOINTS.items():
        endpoint = endpoints[key]
        counts = endpoint.get("counts")
        cells = ([f"{counts[arm]['errors']}/{counts[arm]['views']}" for arm in ARMS] if counts
                 else [f"{endpoint['per_model'][arm]:.4f}" for arm in ARMS])
        lines.append(" & ".join([label, *cells]) + r" \\")
    lines.extend([r"\botrule", r"\end{tabular}", r"\end{table*}", ""])
    table = PAPER / "tables/size_sensitivity_v7.tex"
    table.write_text("\n".join(lines), encoding="utf-8")

    panels = (
        ("A_macro_stress_AUC", "Macro stress ROC-AUC", 1, 1.0),
        ("A_stress_FA", "Stressed REAL false alarms", 100, 60),
        ("A_stress_miss", "Stressed FAKE misses", 100, 90),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.55), layout="constrained")
    for ax, (key, title, scale, maximum) in zip(axes, panels, strict=True):
        values = endpoints[key]["per_model"]
        for offset, pair, color, hatch, label in (
            (-.18, ARMS[::2], "#315c80", None, "ERM"),
            (+.18, ARMS[1::2], "#d5803b", "///", "Channel"),
        ):
            bars = ax.bar([offset, 1 + offset], [values[k] * scale for k in pair], width=.32,
                          color=color, edgecolor="#273743", linewidth=.45, hatch=hatch, label=label)
            ax.bar_label(bars, fmt="%.2f" if scale == 1 else "%.1f", fontsize=8, padding=2)
        ax.set(title=title, xticks=[0, 1], xticklabels=["640", "1,280"], ylim=(0, maximum),
               xlabel="Training recordings", ylabel="ROC-AUC" if scale == 1 else "Rate (%)")
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    figure = PAPER / "figures/size_sensitivity_v7.pdf"
    fig.savefig(figure, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    provenance = {
        "scope": "TRAINING_SIZE_SENSITIVITY_9207",
        "source": EVIDENCE.relative_to(ROOT).as_posix(),
        "source_sha256": digest(EVIDENCE),
        "analysis": "scripts/jaes_v2/size1280_analysis.py",
        "checkpoints": data["selected_checkpoints"],
        "run_seals": data["seal_sha256"],
        "outputs": {p.relative_to(PAPER).as_posix(): digest(p) for p in (table, figure)},
    }
    (PAPER / "size_sensitivity_asset_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
