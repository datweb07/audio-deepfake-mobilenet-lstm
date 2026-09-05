"""Inventory repository evidence for the paper; read-only except generated CSV/Markdown."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def classify(path: Path):
    rel = path.relative_to(ROOT).as_posix()
    lower = rel.lower()
    experiment = "LAVA-5" if "lava_5" in lower else "deployment/conversion" if "conversion" in lower else "legacy/other"
    model = next((m for m in ("mobilenetv3_lstm", "efficientnet_b0_lstm", "mnasnet_lstm", "rawnet2", "aasist", "shufflenetv2_lstm") if m in lower), "shared")
    if "clean" in lower: candidate = "Clean results"
    elif "robustness" in lower or any(x in lower for x in ("noise", "codec", "replay")): candidate = "Robustness"
    elif "efficiency" in lower or "latency" in lower or "rtf" in lower: candidate = "Efficiency"
    elif "pareto" in lower: candidate = "Pareto"
    elif "error_analysis" in lower or "agreement" in lower or "bootstrap" in lower: candidate = "Error/statistics"
    elif "architecture" in lower or "pipeline" in lower or "overview" in lower: candidate = "Methods"
    elif "metadata" in lower or "protocol" in lower or "audit" in lower: candidate = "Provenance/protocol"
    else: candidate = "Supplement/reproducibility"
    usable = "YES" if path.suffix.lower() in {".csv", ".json", ".md", ".txt", ".png", ".svg", ".pdf"} else "SUPPORTING"
    if "diagnostic_100" in lower: usable = "DIAGNOSTIC_ONLY"
    return [rel, path.suffix.lower() or "none", experiment, model, candidate, usable, path.stat().st_size]


def main():
    files = []
    for base in ("outputs", "docs", "models", "papers", "benchmark", "src/lava", "configs", "scripts"):
        files += [p for p in (ROOT/base).rglob("*") if p.is_file()]
    files += [ROOT/name for name in ("train.py", "evaluate.py", "predict.py", "app.py", "README.md", "AUDIT_REPORT.md", "requirements.txt", "requirements-torch.txt") if (ROOT/name).is_file()]
    rows = [classify(p) for p in sorted(set(files))]
    target = ROOT/"papers/REPOSITORY_EVIDENCE_INVENTORY.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["relative_path","file_type","experiment","model","section_candidate","paper_usability","bytes"]); w.writerows(rows)
    counts = {}
    for r in rows: counts[r[4]] = counts.get(r[4],0)+1
    md = ["# Repository Evidence Inventory", "", f"Generated from {len(rows)} files. The complete file-level inventory is `REPOSITORY_EVIDENCE_INVENTORY.csv`.", "", "| Section candidate | Files |", "|---|---:|"]
    md += [f"| {k} | {v} |" for k,v in sorted(counts.items())]
    md += ["", "`DIAGNOSTIC_ONLY` denotes the fixed 100-sample robustness scope; it must not be represented as full-test evidence. Binary model files are supporting provenance/deployment artifacts rather than directly inspectable numeric paper results."]
    (ROOT/"papers/REPOSITORY_EVIDENCE_INVENTORY.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(f"Inventoried {len(rows)} files")

if __name__ == "__main__": main()
