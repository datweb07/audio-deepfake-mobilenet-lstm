# Phase 3 completion and next-stage commands

The delivered 192-record experiment is complete. Do not rerun its inference to regenerate tables. From the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase3 verify --run outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm --report outputs/jaes_v2/audit/phase3_mobile_verify_next.json
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase3 verify --run outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm --report outputs/jaes_v2/audit/phase3_shuffle_verify_next.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.validate --output outputs/jaes_v2/audit/phase3_validation_next
```

Required inputs: delivered run directories and the shared waveform cache. The two verification commands import neither TensorFlow nor Torch, verify seals and rebuild statistics/figures. Report names and validation directories must be new. Expected outputs: PASS reproduction JSON; suite counts and failure details in validation.json. No new audio or model inference is required. Allow roughly 20–60 seconds total depending on machine load and small derived-output space.

For a separately reviewed new public development cohort, `scripts.jaes_v2.public_data` provides discover/select/acquire/import; exact options and existing inputs are in PHASE3_DATA_ACQUISITION.md. A new config must contain the actual imported manifest/selection hashes before `phase3 prepare` and `phase3 infer` can execute. The delivered resource measurement is raw audio 43,492,994 bytes and shared cache 349,010,430 bytes; the cache is reused by both detectors. Preparation took about 190 seconds; concurrent detector jobs took about 524 and 649 seconds. These are resource observations, not latency benchmarks or guarantees for future cohorts.

Phase 4 C1 confirmation has no execution command because its gate is NOT_LOCKED. Required inputs: independent accessible source release/license, exact untouched cohort IDs and hashes, verified grouping/factors, appropriate clean/order baseline, precision rationale and multiplicity/threshold lock. C2 has no training command for a proposed method: teacher weight license/hash/load/quality and distinct mechanism remain unresolved. C3 has no routing/calibration command: validated student, independent calibration and measured total cost remain unresolved. This prevents placeholder commands being mistaken for implemented capabilities.
