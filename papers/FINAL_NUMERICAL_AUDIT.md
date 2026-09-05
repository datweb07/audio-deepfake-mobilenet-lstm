# Final Numerical Audit

| Paper metric | Paper value | Source | Artifact value | Difference | Status |
|---|---:|---|---:|---:|---|
| Test samples | 2737 | `data/manifests/manifest_metadata.json` | 2737 | 0 | PASS (rounding) |
| ShuffleNet clean F1 | 0.9824 | `papers/tables/table_6_clean_performance.csv` | 0.9824109824 | 1.09824e-05 | PASS (rounding) |
| ShuffleNet AUC | 0.9929 | `papers/tables/table_6_clean_performance.csv` | 0.9928828496 | 1.71504e-05 | PASS (rounding) |
| ShuffleNet EER | 0.0146 | `papers/tables/table_6_clean_performance.csv` | 0.0146299484 | 2.99484e-05 | PASS (rounding) |
| MobileNet E2E ms | 43.81 | `papers/tables/table_9_efficiency.csv` | 43.806352 | 0.003648 | PASS (rounding) |
| ShuffleNet E2E ms | 62.52 | `papers/tables/table_9_efficiency.csv` | 62.521652 | 0.001652 | PASS (rounding) |
| ShuffleNet RTF | 0.0208 | `papers/tables/table_9_efficiency.csv` | 0.020840551 | 4.0551e-05 | PASS (rounding) |
| ShuffleNet mean diagnostic degradation | 0.1623 | `papers/tables/table_8_robustness_summary.csv` | 0.1622553145 | 4.46855e-05 | PASS (rounding) |
