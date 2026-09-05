# LAVA Paper Evidence Map

Repository evidence overrides narrative text. `YES (diagnostic)` means evidence exists only for the fixed 100-sample scope and must not be generalized to all 2,737 test files.

| Paper claim | Section | Source artifact/file | Metric/table/figure | Verified? |
|---|---|---|---|---|
| Five evaluated models; ShuffleNet excluded | Abstract, 3.4, 4.1 | `outputs/lava_5/protocol/protocol.json` | detector scope | YES |
| RawNet2/AASIST externally pretrained | 2.2–2.3, 3.5 | `outputs/lava_5/protocol/artifact_audit.csv`; `docs/SOURCE_PROVENANCE.md` | provenance | YES |
| Native-to-ONNX score parity | 3.5 | `outputs/lava_5/protocol/rawnet2_native_audit.json`; `aasist_native_audit.json` | maximum difference, six samples | YES |
| 18,722 scanned; 18,232 included | 3.2 | `data/manifests/manifest_metadata.json` | inventory counts | YES |
| 435 duplicate groups and 30 conflict files | 3.2 | same metadata; `duplicate_report.csv`; `label_conflicts.csv` | integrity counts | YES |
| checksum-group-disjoint only | 3.2, Limitations | `manifest_metadata.json`; `split_manifest.csv` | split claim | YES |
| 12,762/2,733/2,737 split | 3.2 | `manifest_metadata.json` | split counts | YES |
| lightweight preprocessing parameters | 3.3 | `config.py`; `src/preprocessing.py` | 22.05 kHz, 3 s, six segments, STFT/Mel/image | YES |
| reference duration and adapter deviation | 3.3 | `src/lava/preprocessing/waveform.py`; `outputs/lava_5/report/LAVA_5_BENCHMARK_REPORT.md` | 64,600/16 kHz; padding caveat | YES |
| architecture and parameter counts | 3.4, Table 1 | `table_1_detector_specification.csv`; model source; native audits | Table 1, Fig. 1 | YES |
| EfficientNet is warm-up-only | 3.5, Limitations | `models/efficientnet_b0_lstm/metadata.json`; `docs/efficientnet_warmup_deployment.md` | epoch 47; validation loss | YES |
| MnasNet scratch best epoch 27 | 3.5 | `models/mnasnet_lstm/metadata.json`; `docs/mnasnet_deployment.md` | lifecycle evidence | YES |
| score semantics REAL=0/FAKE=1 | 3.6 | `src/lava/score_semantics.py`; `config.py` | Eq. 1 | YES |
| lightweight thresholds validation-derived | 3.6 | threshold/metadata files; `artifact_audit.csv` | 0.82/0.90/0.90 | YES |
| reference thresholds default, uncalibrated | 3.6, Limitations | reference `threshold.json`; `artifact_audit.csv` | 0.5 | YES |
| clean results, 2,737/model | 4.2 | five `outputs/lava_5/clean/*/scores.csv`; summaries | Table 2; Figs. 3–4 | YES |
| robustness subset 100 (58/42), seed 42 | 3.8, 4.3 | `diagnostic_100/protocol/protocol.json`; `robustness_execution_plan.json` | scope | YES (diagnostic) |
| AWGN/codec/replay conditions | 3.8 | `diagnostic_100/protocol/*_generation.json`; condition manifests | Figs. 5–6 | YES (diagnostic) |
| mean degradation values | 4.3 | `diagnostic_100/tables/table_3_robustness.csv` | Table 3 | YES (diagnostic) |
| no physical replay or unseen result | 3.8, Limitations | main and diagnostic generated reports | NOT_AVAILABLE | YES |
| CPU timing protocol and results | 3.9, 4.4 | five `efficiency/*/summary.json`; execution environment | Table 4; Fig. 7 | YES |
| exploratory Pareto membership | 3.10, 4.5 | diagnostic `pareto_results.csv`; `dominance_matrix.csv` | Table 5; Fig. 9 | YES (diagnostic) |
| official full-test Pareto NOT_RUN | 3.10, Limitations | `report/diagnostic_scope_acceptance.json` | acceptance state | YES |
| all-correct/all-wrong counts | 4.6 | `error_analysis/all_5_correct.csv`; `all_5_wrong.csv` | error overlap | YES |
| pairwise agreement | 4.6 | `error_analysis/agreement_matrix.csv` | Fig. 10 | YES |
| bootstrap intervals | 4.7 | `error_analysis/bootstrap_95_ci.csv`; `statistics_state.json` | Table 6 | YES |
| no architecture/preprocessing/threshold mutation during benchmark | Appendix | sealed source/artifact hashes; acceptance JSON | acceptance PASS | YES |
| no model retraining | Entire paper | benchmark modules import no trainer; acceptance JSON `no_retraining=true` | protocol | YES |

Literature descriptions are supported by the eight verified entries in `references.bib`; no empirical LAVA value is sourced from external literature.
