# LAVA paper evidence map — six-detector update

| Paper claim | Section | Source artifact | Metric / figure / table | Verified? |
|---|---|---|---|---|
| Six artifacts load and expose P(FAKE) | Abstract, 4.1 | `outputs/lava_6/protocol/protocol.json`; ShuffleNet and historical load audits | artifact gate | YES |
| ShuffleNet is manifest matched | 3.5, 4.1 | `models/shufflenetv2_lstm/metadata.json`; `data/manifests/manifest_metadata.json` | identical manifest hash | YES |
| ShuffleNet is scratch/end-to-end with 56 trainable BN layers | 3.5 | `models/shufflenetv2_lstm/metadata.json` | training policy | YES |
| Dataset counts and checksum-group-disjoint scope | 3.2 | `data/manifests/manifest_metadata.json` | integrity inventory | YES |
| Full clean results, 2,737 samples/model | 4.2 | five historical clean outputs plus `outputs/lava_6/clean/shufflenetv2_lstm` | Table 2; ROC/DET | YES |
| ShuffleNet F1/AUC/EER = 0.9824/0.9929/0.0146 | Abstract, 4.2 | `outputs/lava_6/clean/shufflenetv2_lstm/summary.json` | Table 2 | YES |
| Robustness is diagnostic 100 samples | 3.8, 4.3, 5 | `outputs/lava_6/report/acceptance.json` | Table 3 | YES |
| Stress waveforms are reused from LAVA-5 | 3.8 | six-model and historical condition protocols | robustness figures | YES |
| ShuffleNet mean diagnostic degradation = 0.1623 | 4.3 | `outputs/lava_6/lava_6_results.csv` | Table 3 | YES |
| ShuffleNet latency/RTF = 62.52 ms/0.02084 | 4.4 | `outputs/lava_6/efficiency/shufflenetv2_lstm/summary.json` | Table 4 | YES |
| Diagnostic frontier = MobileNet, ShuffleNet, AASIST | 4.5 | `outputs/lava_6/pareto/pareto_results_6_models.csv` | Table 5 | YES |
| Six-model agreement | 4.6 | `outputs/lava_6/error_analysis/` | Figure 10 | YES |
| RawNet2/AASIST are external references | 3.4–3.5 | pretrained metadata; `docs/SOURCE_PROVENANCE.md` | Table 1 | YES |
| Unseen and physical replay unavailable | 3.8, 5 | `outputs/lava_6/report/acceptance.json` | limitation | YES |
| No old detector re-executed; no training | Appendix | six-model protocol and acceptance | provenance | YES |
