# Figure manifest — six-detector paper

| Figure | Title | Paper file | Generated from | Script | Section |
|---:|---|---|---|---|---|
| 1 | Six-detector LAVA overview | `figures/lava_6_model_overview.png` | six-model protocol | `benchmark/lava6_report.py` | 3.1 |
| 2 | Benchmark pipeline | `figures/lava_benchmark_pipeline.png` | historical protocol | `benchmark/lava5_report.py` | 3.1 |
| 3 | Clean ROC | `figures/roc_comparison_6_models.png` | six full score files | `benchmark/lava6_report.py` | 4.2 |
| 4 | Clean DET | `figures/det_comparison_6_models.png` | six full score files | same | 4.2 |
| 5 | Noise F1 | `figures/noise_f1_vs_snr_6_models.png` | diagnostic scores | same | 4.3 |
| 6 | Robustness heatmap | `figures/robustness_heatmap_6_models.png` | 54 stress summaries | same | 4.3 |
| 7 | End-to-end latency | `figures/end_to_end_latency_bar_6_models.png` | six efficiency JSONs | same | 4.4 |
| 8 | Parameters | `figures/parameters_bar_6_models.png` | metadata/load audits | same | 4.4 |
| 9 | Diagnostic Pareto | `figures/pareto_eer_rtf_6_models.png` | six aggregate rows | same | 4.5 |
| 10 | Agreement | `figures/agreement_heatmap_6_models.png` | aligned full predictions | same | 4.6 |

Robustness and Pareto figures are explicitly marked diagnostic; none is a terminal screenshot.
