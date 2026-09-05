# Table Manifest

| Table | Contents | Source CSV/JSON | Generation/verification | Section |
|---:|---|---|---|---|
| 1 | detector, category, runtime, input, architecture, parameters, provenance | `outputs/lava_5/tables/table_1_detector_specification.csv`; native audit JSONs | `benchmark/lava5_report.py`; `scripts/lava5_reference_audit.py` | 3.4 |
| 2 | full clean Accuracy, Precision, Recall, F1, Macro-F1, AUC, EER, confusion counts | `outputs/lava_5/tables/table_2_clean.csv` | recomputed by `benchmark/lava5_report.py::load_result` | 4.2 |
| 3 | matched diagnostic clean F1 and category/overall ΔF1 | `outputs/lava_5/diagnostic_100/tables/table_3_robustness.csv` | `benchmark/lava5_report.py` | 4.3 |
| 4 | parameters, size, RSS, preprocessing/model/end-to-end timing, throughput, RTF | `outputs/lava_5/tables/table_4_efficiency.csv` | `benchmark/lava5_runtime.py`; report generator | 4.4 |
| 5 | diagnostic EER, degradation, RTF, Pareto membership | `outputs/lava_5/diagnostic_100/tables/table_5_pareto.csv` | `benchmark/pareto.py`; report generator | 4.5 |
| 6 | stratified full-test bootstrap 95% intervals | `outputs/lava_5/error_analysis/bootstrap_95_ci.csv` | `benchmark/lava5_report.py::statistics`, seed 42, 1,000 iterations | 4.7 |

Values in Markdown and LaTeX are rounded display copies of these generated files. CSV precision remains authoritative.
