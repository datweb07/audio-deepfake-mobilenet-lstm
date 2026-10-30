# Phase6 commands and Phase7 gates

All commands run at D:/audio-deepfake-mobilenet-lstm in PowerShell. Existing output IDs reject overwrite. Do not rerun completed training to reproduce tables.

Executed stages:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment select
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment preflight
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment cache
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment train --config configs/jaes_v2/p6_p0_b4_v1.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment finetune --output-name partial_finetune_sanity_v2
```

The nine executed config files are p6_p0_{b0,b1,b2,b4_control,b4,rb}_v1.json and p6_p1_{b0,b1,b2}_v1.json. Braces here describe names, not a CLI option. Failed feature_cache_v1/fine_tune_v1 remain; the valid cache is feature_cache_v2. The first fine-tune directory's exact name is partial_finetune_sanity_v1.

Safe score-only regeneration to fresh output IDs:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_report --output outputs/jaes_v2/phase6/reports_another
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.build_master_research_report --audit-output outputs/jaes_v2/phase6/master_audit_another
```

Validation:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.validate --output outputs/jaes_v2/audit/phase6_validation_another
```

An independently provisioned/pinned compatible TensorFlow GPU runtime may run the bounded sanity command below. No GPU environment installation or full fine-tuning is claimed here:

```powershell
python -m scripts.jaes_v2.phase6_experiment finetune --device gpu --output-name partial_finetune_gpu_sanity_v1
```

Phase7: increase balanced per-speaker cells to include A05/A06; validate online strong augmentation and partial representation training; budget at least three seeds; register primary contrasts/clean margins and secondary correction before external labels; resolve a technical distinction or reject the mechanism; choose/storage-audit unused external data; separately lock calibration use. Longer fine-tuning is not exposed through a fake CLI flag. Conference venue/DOI/publication date and copyright terms are still needed before journal-extension eligibility can be certified. No immediate dataset download is required.
