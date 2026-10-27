# Frozen conference baseline

Audit started 24 October 2026; execution continued 25 October 2026 (Asia/Novosibirsk).

The authoritative accepted file is `papers/PaperID_804_final.docx`. Its SHA-256 is `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`, verified again before implementation. Paragraphs, tables and the OMML equations were inspected. The file with spaces in its name is not authoritative.

Before implementation, `outputs/jaes_v2/audit/baseline_freeze.json` recorded hashes and sizes for **2,376** existing source, test, model, canonical-manifest and LAVA-5/LAVA-6 output files. It also records the initial Git state (only the previous prompt document was untracked) and HEAD `e7bcf296145228bccd96d9bdab5745d624282c59`. No legacy tracked source has been edited. The preflight command rechecks every protected file. This is an integrity snapshot, not a separate Git checkout or an assertion that the old code is bug-free.

Repository interfaces inspected: detector registry/specs/contracts; MobileNet and common Keras adapters; production artifact restoration; canonical manifests/integrity metadata; preprocessing; metrics; training lifecycle policies; legacy ordered stress generation and benchmark aggregation; source provenance; relevant regression tests. Source fingerprints cover all production Python modules. Reference repositories were treated as read-only; we did not claim line-by-line review of every vendored repository.

## Evidence inherited

- 18,722 scanned / 18,232 retained recordings; train/validation/test 12,762/2,733/2,737. Only byte-checksum separation is established.
- Four locally trained Mel models; external RawNet2 and AASIST checkpoints. EfficientNet is a warm-up artifact, not a completed fine-tuning comparison.
- Six real detector artifacts are present; MobileNet has additional weights-only deployment fallbacks. V2 seals the primary bundle and possible fallbacks, and checks score parity with the existing registry API.
- Legacy metadata has UNKNOWN speaker/source/generator/corpus/parent identities. Filename patterns cannot recover authoritative identities.
- The 100-item stress diagnostic, single desktop timing and exploratory Pareto results are prior work.

## Known inherited regression failures

Selected tests ran against unchanged baseline files: 37 of 40 passed outside the Windows sandbox. Three failures remain:

1. `test_lava5_benchmark.Lava5Tests.test_display_missing_results_is_optional`: `benchmark_card(None, ...)` dereferences `spec.name` in the packaged fallback.
2. `test_decision_display.DecisionDisplayTest.test_single_score_chart`: the test expects `st.pyplot`, but its mock receives no call.
3. `test_streamlit_ui.StreamlitUITest.test_available_models_do_not_show_missing_artifact_panel`: an existing benchmark expander violates the test's expectation of no expander.

These are not evidence that V2 broke the UI: neither legacy implementations nor tests import V2, and their before/after hashes match. They do prevent an all-green regression claim. Full tracebacks and suite counts are in `outputs/jaes_v2/audit/validation_v2/validation.json`. Five additional permission errors seen inside the sandbox disappeared outside it; those are reported separately from the three persistent failures. The legacy code was preserved instead of changing unrelated historical behavior in this research slice.

## Phase 2 amendment (25 October 2026)

The preceding failure description is historical Phase 1 status. The user explicitly authorized resolving these three issues with backups and a separately recorded freeze exception. All three were reproduced and resolved; selected legacy tests now pass 40/40. See LEGACY_REGRESSION_RESOLUTION.md and `outputs/jaes_v2/audit/phase2_baseline_amendment.json`. The original snapshot remains unchanged. Only the benchmark-card source and two stale/mismatched tests have approved hash changes; paper, models, canonical manifests, historical outputs and all 71 files of pilot v3 retain their prior hashes.
