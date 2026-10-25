# LAVA Journal V2 implementation

This package is an isolated research extension. Existing registry IDs, artifacts, entry points and conference results are unchanged. Importing the package, provenance/analysis or aggregate command does not load a neural framework. Only the live predictor bridge imports TensorFlow; PyTorch remains isolated in its existing environment.

Run from the repository root using the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.runner --preflight
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.runner --config configs/jaes_v2/c1_pilot_v3.json --experiment-id c1_example_new_id
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.runner --aggregate outputs/jaes_v2/c1_example_new_id
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.data_audit --manifest outputs/jaes_v2/c1_example_new_id/manifest.csv --output outputs/jaes_v2/audit/example_new_id
```

All run IDs and audit output directories must be new. The existing first delivered run is `c1_noise_opus_pilot_v3`, so use `--aggregate` to inspect it instead of rerunning into its directory. A failure retains metadata/error traceback and any partial score rows; it does not promote the run to completion. A subsequent attempt needs a new ID. Automatic resume is intentionally not implemented.

`conditions.csv` records source/parent/label, chain order/hash, transformation parameters/seeds, actual source/output rates and lengths, common headroom gain, noise SNR, codec alignment limitations and waveform checksum. `scores.csv` records each attempted prediction, fixed threshold, status, error and component timing. `calibrated_score=NOT_FITTED` explicitly distinguishes scores from calibrated probabilities. Diagnostic timings exclude transformation generation but include variant-file decoding, preprocessing and inference; they are a single-pass observation, not repeated deployment benchmarks.

`analysis.py` rejects missing/duplicate/failed/mismatched pairs and mixed model/seed/threshold strata. It never optimizes a threshold. Parent bootstrap estimates equal-weight parent means, not independent transformed-file uncertainty. Undefined EER for one-class inputs and intervals with fewer than two parents remain missing.

`SourceRecord` and `audit_partitions` are reusable extension contracts. Training/development/calibration/test/confirmation labels are explicit. Source/target speaker overlaps are checked when requested; UNKNOWN identity cannot certify a hold-out. Exact decoded PCM checks are separate from perceptual deduplication, which remains unavailable. Strict pilot scope prevents silently treating legacy evaluation as confirmation.

No new dependencies were installed. FFmpeg is required for codecs; FFprobe additionally records the encoded stream's actual rate. Absence of either is explicit. Supported conventional codecs are Opus, MP3 and AAC, not neural resynthesis. Environmental-noise and measured-RIR transforms require supplied, licensed assets; they were unit-tested with fixtures and are not claimed as a measured-room research dataset.

Scientific status, protocol caveats, known baseline test failures and next commands: `docs/jaes_v2/RESEARCH_STATUS.md`.
