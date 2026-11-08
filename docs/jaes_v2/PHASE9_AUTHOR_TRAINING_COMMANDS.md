# Phase 9 — Author command handoff

Run from the repository root in PowerShell. These are verified audit/preparation commands. The legacy preflight performs read-only train/development integrity checks and writes a projection. Its PASS applies to its specific checks.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_medium_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment validate --output outputs/jaes_v2/phase9/regression_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase9_parity --output outputs/jaes_v2/phase9/parity_v2.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment preflight --config configs/jaes_v2/phase8/p8_foundation_erm_partial_s6107_v1.json --output outputs/jaes_v2/phase9/foundation_preflight_v2.json
```

Medium-plus is already prepared. `python -m scripts.jaes_v2.phase9_data` is a one-time command in a fresh workspace with the foundation and medium artifacts. It refuses to overwrite the existing output directory. Audit artifacts are under `outputs/jaes_v2/phase9`.

For existing historical runs the audit above is the all-in-one verification/aggregation path. Historical monitor and verify are read-only and use a run directory. Existing training and resume entry points are Phase 8 only. A Phase 9 train/resume/monitor/aggregate launch package requires a new run contract after gradient/validation parity, resource approval and new manifest registration. No Phase 9 training command is authorized by this handoff. On a future interrupted pilot preserve the latest committed checkpoint, verify its seal and require the original fingerprint for resume. Stop for OOM, D below 15 GiB, low C/pagefile commit or incomplete guards. The next job dependency is the fixed-engine 100-update pilot, followed by matched medium replication only if it passes.
