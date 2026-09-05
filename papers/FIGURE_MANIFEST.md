# Figure Manifest

All copied paper figures are 300-DPI PNGs regenerated from measured CSV/JSON data by `benchmark/lava5_report.py`. Diagnostic figures contain the visible watermark “DIAGNOSTIC SUBSET — NOT FULL TEST ROBUSTNESS.”

| Figure | Title | Paper file | Source evidence | Generator | Section |
|---:|---|---|---|---|---|
| 1 | LAVA five-detector overview | `figures/lava_5_model_overview.png` | registry specifications and `outputs/lava_5/protocol/protocol.json` | `benchmark/lava5_report.py::diagrams` | 3.1 |
| 2 | Benchmark pipeline | `figures/lava_benchmark_pipeline.png` | benchmark execution design and canonical protocol | `benchmark/lava5_report.py::diagrams` | 3.1 |
| 3 | Clean ROC comparison | `figures/roc_comparison_5_models.png` | five full `clean/*/scores.csv` files, 2,737 rows/model | `benchmark/lava5_report.py` | 4.2 |
| 4 | Clean DET comparison | `figures/det_comparison_5_models.png` | five full clean score files | `benchmark/lava5_report.py` | 4.2 |
| 5 | F1 versus AWGN SNR | `figures/noise_f1_vs_snr.png` | `outputs/lava_5/diagnostic_100/robustness/noise/` | `benchmark/lava5_report.py::stress_figures` | 4.3 |
| 6 | Robustness degradation heatmap | `figures/robustness_heatmap.png` | 45 diagnostic stress summaries and matched subset clean scores | `benchmark/lava5_report.py::stress_figures` | 4.3 |
| 7 | End-to-end latency | `figures/end_to_end_latency_bar.png` | five `efficiency/*/summary.json` files | `benchmark/lava5_report.py` | 4.4 |
| 8 | Parameter comparison | `figures/parameters_bar.png` | loaded Keras counts and native strict-load audit | `benchmark/lava5_report.py` | 4.4 |
| 9 | Diagnostic Pareto EER–RTF projection | `figures/pareto_2d_eer_rtf.png` | diagnostic Pareto CSV; three objectives | `benchmark/lava5_report.py::pareto_figures` | 4.5 |
| 10 | Prediction agreement | `figures/model_agreement_heatmap.png` | aligned full-test predictions in five score files | `benchmark/lava5_report.py::statistics` | 4.6 |

Supplementary figures remain under `outputs/lava_5/figures/`, `outputs/lava_5/error_analysis/`, and `outputs/lava_5/diagnostic_100/figures/`; they are not copied into the main paper to avoid an image dump.
