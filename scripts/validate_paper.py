"""Fail-closed consistency checks for the evidence-backed LAVA manuscripts."""
from pathlib import Path
import csv, json, re

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers"
en = (PAPER/"LAVA_FULL_PAPER_EN.md").read_text(encoding="utf-8")
vi = (PAPER/"LAVA_FULL_PAPER_VI.md").read_text(encoding="utf-8")
tex = (PAPER/"LAVA_FULL_PAPER.tex").read_text(encoding="utf-8")
bib = (PAPER/"references.bib").read_text(encoding="utf-8")

for text in (en, vi, tex):
    assert "2,737" in text or "2.737" in text
    assert "ShuffleNet" in text and ("excluded" in text or "bị loại" in text)
    assert "RawNet2" in text and "AASIST" in text
for forbidden in ("climate", "hydrology", "salinity", "25,380 training", "71,237 evaluation", "10,000 unseen"):
    assert forbidden.lower() not in (en+vi+tex).lower(), forbidden

en_figs = re.findall(r"\]\(figures/([^\)]+)\)", en)
vi_figs = re.findall(r"\]\(figures/([^\)]+)\)", vi)
tex_figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
assert en_figs == vi_figs and set(en_figs) == set(tex_figs) and len(en_figs) == 10
assert all((PAPER/"figures"/name).is_file() for name in en_figs)
assert len(re.findall(r"\*\*Table [1-6]\.", en)) == 6
assert len(re.findall(r"\*\*Bảng [1-6]\.", vi)) == 6
assert len(re.findall(r"\\begin\{table\*?\}", tex)) == 6
assert tex.count("{") == tex.count("}"), "unbalanced LaTeX braces"

en_refs = en.split("## References", 1)[1].strip()
vi_refs = vi.split("## Tài liệu tham khảo", 1)[1].strip()
assert en_refs == vi_refs, "English and Vietnamese reference lists differ"

bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
used = set(re.findall(r"\\cite\{([^}]+)\}", tex))
assert used == bib_keys, (used, bib_keys)
assert len(bib_keys) == 8

with (ROOT/"outputs/lava_5/tables/table_2_clean.csv").open() as f:
    clean = list(csv.DictReader(f))
assert len(clean) == 5 and all(str(round(float(r["f1"]),4)) in en for r in clean)
with (ROOT/"outputs/lava_5/diagnostic_100/tables/table_5_pareto.csv").open() as f:
    pareto = list(csv.DictReader(f))
assert len(pareto) == 5
for row in pareto:
    values = (float(row["EER"]), float(row["MeanRobustnessDeg"]), float(row["RTF"]))
    for value in values:
        rendered = f"{value:.4f}"
        en_rendered = rendered.replace("-", "−", 1) if rendered.startswith("-") else rendered
        tex_rendered = rendered.replace("0.", ".", 1)
        assert en_rendered in en and tex_rendered in tex, (row["Model"], rendered)
acceptance = json.loads((ROOT/"outputs/lava_5/report/diagnostic_scope_acceptance.json").read_text())
assert acceptance["status"] == "PASS" and acceptance["no_retraining"]
assert acceptance["full_test_robustness"] == "NOT_RUN" and not acceptance["full_lava5_complete"]
result = dict(status="PASS", figures=10, tables=6, citations=8,
              clean_models=5, diagnostic_conditions=45,
              full_test_robustness="NOT_RUN", shuffle_excluded=True,
              latex_compilation="NOT_AVAILABLE_ON_HOST")
(PAPER/"PAPER_VALIDATION.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result, indent=2))
