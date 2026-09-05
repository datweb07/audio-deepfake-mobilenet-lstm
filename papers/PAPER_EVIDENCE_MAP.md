# Paper Evidence Map

| Claim | Section | Source artifact | Verified value/scope | Verified? |
|---|---|---|---|---|
| Canonical included/test counts | 3.2 | `data/manifests/manifest_metadata.json` | 18,232 / 2,737 | YES |
| Checksum-group-disjoint claim | 3.2 | `data/manifests/manifest_metadata.json` | verified | YES |
| Preprocessing tensor contract | 3.3 | `src/preprocessing.py` | 6x224x224x3 float32 | YES |
| Six detector architecture/provenance | 3.4--3.12 | `models/*/metadata.json; src/lava/models` | six rows | YES |
| Clean metrics | 4.2 | `outputs/lava_6/lava_6_results.csv` | six models / 2,737 samples | YES |
| Class-wise and confusion counts | 4.3 | `outputs/lava_5/clean/*/scores.csv; outputs/lava_6/clean/shufflenetv2_lstm/scores.csv` | six models | YES |
| Diagnostic robustness | 4.5--4.8 | `outputs/lava_6/robustness/robustness_summary_6_models.csv` | 100 samples / nine conditions | YES |
| Efficiency | 4.9 | `outputs/lava_6/efficiency/efficiency_summary_6_models.csv` | one thread / 10 warmup / 50 runs | YES |
| Pareto membership | 4.10 | `outputs/lava_6/pareto/pareto_results_6_models.csv` | MobileNet, ShuffleNet, AASIST | YES |
| Bootstrap and pairwise tests | 4.12 | `papers/tables/table_11_bootstrap_ci.csv; table_12_pairwise_full_test.csv` | full canonical test | YES |
