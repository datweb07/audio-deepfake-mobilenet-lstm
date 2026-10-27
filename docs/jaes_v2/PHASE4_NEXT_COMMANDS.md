# Phase 4 next commands and prerequisites

Run from the repository root using the existing `.venv`. The delivered experiment is complete; model-free verification is the next reproducibility action:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase4 verify --run outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm --report outputs/jaes_v2/audit/phase4_mobile_verify_next.json
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase4 verify --run outputs/jaes_v2/c1_nuisance_controls_dev_v1_shufflenetv2_lstm --report outputs/jaes_v2/audit/phase4_shuffle_verify_next.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.audit_phase4 --output outputs/jaes_v2/audit/phase4_source_matching_recheck_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase4_feature_diagnostics --output outputs/jaes_v2/audit/phase4_feature_recheck_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.plan_phase4_precision --run outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm --output outputs/jaes_v2/audit/phase4_precision_next.json --target-effect 0.10 --half-width 0.05
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.validate --output outputs/jaes_v2/audit/phase4_validation_next
```

Reports/directories must be new. Required existing inputs: pinned metadata, Phase 3 sealed cache/scores and delivered Phase 4 cache/run directories. Expected outputs: source matching/coverage/draft, exact feature diagnostics, reproducible metrics/contrasts/precision/figures, test report. Those operations do not call a neural predictor. Allow roughly one minute and small derived-output space; no new audio download.

Delivered config: `configs/jaes_v2/phase4_nuisance_dev_v1.json`; selection: `phase4_nuisance_selection_v1.json`. Preparation took 84.69 s and stored about 74.37 MB including 288 shared new waveforms. Concurrent CPU detector jobs took 118.79 / 148.61 s, each producing 288 new scores and reusing 224 Phase 3 scores; no latency benchmark is claimed. The actual code supports `phase4 prepare --config` and `phase4 infer --config --model` for a reviewed new experiment ID; do not rerun the delivered ID or manufacture a matched scope.

Next scientific input for A confirmation: authorized metadata/audio with repeated exact source content across declared systems, substantially more independent source groups, resolved reference/target voice limits and an untouched cohort. Inspect provenance and precision before acquisition/inference. A full MLAAD account-access decision belongs to the user; no gated-content workaround exists in this workflow. For B transfer, obtain a separate official-label/provenance cohort and explicitly register a transfer hypothesis; it cannot confirm A.

For C2, exact checkpoint terms/hash and a compatible isolated author runtime are prerequisites to a bounded load/quality/memory test. Current production/Torch environments were not upgraded. No unsupported training or confirmation command is supplied. C3 needs a validated student and independent calibration first.
