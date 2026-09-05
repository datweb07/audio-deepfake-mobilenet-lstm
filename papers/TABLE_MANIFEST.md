# Table manifest — six-detector paper

| Table | Columns | Programmatic source | Section |
|---:|---|---|---|
| 1 | detector, group, input, architecture, params, provenance | `outputs/lava_6/tables/table_1_detector_specification_6_models.csv` | 3.4 |
| 2 | accuracy, precision, recall, F1, macro-F1, AUC, EER, confusion | `outputs/lava_6/tables/table_2_clean_6_models.csv` | 4.2 |
| 3 | noise/codec/replay/overall degradation | `outputs/lava_6/tables/table_3_robustness_diagnostic_6_models.csv` | 4.3 |
| 4 | params, size, RSS, latency, throughput, RTF | `outputs/lava_6/tables/table_4_efficiency_6_models.csv` | 4.4 |
| 5 | diagnostic EER, degradation, RTF, Pareto | `outputs/lava_6/tables/table_5_pareto_diagnostic_6_models.csv` | 4.5 |
| 6 | bootstrap intervals | `outputs/lava_6/error_analysis/bootstrap_95_ci_6_models.csv` | 4.7 |

All tables are derived by `benchmark/lava6_report.py`; five-model rows are reused and ShuffleNet rows are appended.
