# LAVA Journal V2 research status

Updated 25 October 2026, Asia/Novosibirsk.

Conference baseline: **FROZEN** (integrity verified, not all legacy tests green).
Journal V2: **IN DEVELOPMENT**.
C1: **PILOT_EVALUATED — CHANNEL ORDER ONLY**.
C2: **DESIGNED — NOVELTY/TEACHER GATES PENDING**.
C3: **DESIGNED — DEPENDS ON VALIDATED STUDENT AND CALIBRATION DATA**.

## Delivered execution

The actual MobileNet pilot is `outputs/jaes_v2/c1_noise_opus_pilot_v3/`. It contains 12 development source records, 60 condition waveforms, 60 paired scores, metrics/statistics, 27 descriptive contrasts, PNG/SVG and run metadata with source/config/checkpoint hashes. Six examples per class is a feasibility cohort, not independent evaluation. Baseline v1 evidence is preserved.

Pilot v1/v2 remain historical execution evidence. Their decoded samples/scores agree, but a timestamped PEAK chunk made libsndfile WAV container hashes change. V3 fixes this writer issue. Verification evidence is `outputs/jaes_v2/audit/reproduction_verification.json`; tests are recorded in `outputs/jaes_v2/audit/validation_v2/validation.json` (19 V2 passes; 37/40 selected legacy passes).

For this small cohort only: clean 0/12 errors, AWGN 3/12, codec 0/12, noise→codec 6/12, codec→noise 4/12. All six noise→codec errors are FAKE misses; there are no REAL false alarms at the inherited 0.82 threshold. Overall order-error contrast is 2/12 with descriptive parent-bootstrap interval [0, 5/12]. It does not establish a nonzero population effect. At this fixed threshold, error ordering and diagnostic EER ordering differ: threshold transfer and ranking must be analyzed separately.

The source audit finds mean original durations about 4.33 s (REAL) versus 104.42 s (FAKE), although the detector sees at most the first 3 s. This flags collection heterogeneity; it is not proof the classifier uses duration. Metadata-only leave-one-parent-out accuracy is 7/12 on this tiny cohort, so no population shortcut conclusion is justified. Original lineage remains UNKNOWN.

## Exact commands (PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.runner --preflight
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.runner --aggregate outputs/jaes_v2/c1_noise_opus_pilot_v3
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests/jaes_v2 -v
```

The aggregation command reads persisted scores and does not import TensorFlow/PyTorch or run a model. Rebuilding derived statistics/figures is explicit; input score/manifest/config hashes must match. Baseline paths are never written.

To regenerate every waveform, verify file hashes and rebuild the figures/statistics without a model, use a new report filename:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.reproduce --run outputs/jaes_v2/c1_noise_opus_pilot_v3 --report outputs/jaes_v2/audit/reproduction_verification_next.json
```

For an independent execution check using the **same pilot cohort**, choose a new ID:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.runner --config configs/jaes_v2/c1_pilot_v3.json --experiment-id c1_noise_opus_replication_v1
```

Expected output: `outputs/jaes_v2/c1_noise_opus_replication_v1/`. It remains a PILOT and does not create independent scientific confirmation. Reusing an existing ID fails deliberately. For tests in the managed Windows sandbox, temporary-file replacement can require an approved unsandboxed run.

To persist both test suites in a new audit directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.validate --output outputs/jaes_v2/audit/validation_v3
```

Expect nonzero exit until the three inherited legacy test failures described in BASELINE_AUDIT.md are separately reconciled. This is not a V2 unit-test failure.

## Next research steps and blockers

- **SOURCE_BLOCKER:** no blocker for this C1 slice. Public-dataset importer, generator-factor estimator and codec delay/control experiments are not implemented. Frozen baseline has three inherited regression failures.
- **DATA_BLOCKER:** no acquired metadata-rich independent generator/corpus cohort; UNKNOWN identities cannot be repaired by filenames. Near-duplicate audit and teacher pretraining-overlap review remain.
- **COMPUTE_BLOCKER:** 4 GiB GPU present but teacher training feasibility unmeasured; no full-training estimate. CPU pilot is complete.
- **NOVELTY_BLOCKER:** paired consistency/distillation, class asymmetry and sequential channels all have close prior work. No three-contribution novelty claim is supported.
- **VALIDATION_BLOCKER:** no external confirmation, matched trained baselines, multi-seed results, teacher validation, C2 ablation or C3 matched-cost evaluation.
- **JOURNAL_ELIGIBILITY_BLOCKER:** accepted conference venue/copyright/publication details missing; official current Clarivate profile not verified.

Proceed by selecting an actual small license-compatible public corpus and inspecting its metadata, then implement its importer and hold-out protocol. Before training C2, finish the nearest-work technical distinction and bounded teacher validation. Do not invent a long training command for an objective not yet specified; a truthful blocked research stage is preferable to an empty executable.
