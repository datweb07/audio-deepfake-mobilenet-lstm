# LAVA Journal V2 implementation

## Phase 4 current extension

Source matching and matched-cohort/interaction contracts are in `source_matching.py` and `matched_analysis.py`. `nuisance_controls.py`, `phase4.py`, `phase4_analysis.py` and `phase4_figures.py` implement a sealed bounded within-record fallback, reusing Phase 3 scores. `precision.py` plans from cluster means; `confirmation_gate.py` reviews metadata without opening labels. Feature diagnostics use production-parity replicas only; no production preprocessing is replaced.

Delivered: matched fixed-pair study blocked; 32-record nuisance pilot, 576 successful new scores and 448 reused scores; 79 V2 / 40 selected legacy passing tests. This is supporting validation/engineering, not international novelty or confirmation. CLI: `python -m src.lava.jaes_v2.phase4` with prepare/infer/aggregate/verify. Reproducible audit/planning commands: docs/jaes_v2/PHASE4_NEXT_COMMANDS.md. Historical phases follow.

## Phase 3 current extension

`acquisition.py` + `mlaad.py` implement pinned official metadata discovery, author annotation parsing, deterministic language/group selection, bounded noncommercial acquisition and local import. `scripts.jaes_v2.public_data` exposes those operations. `datasets.py` keeps ASVspoof compatibility and adds optional public-source provenance fields.

`phase3.py` provides `prepare`, `infer`, `aggregate` and `verify`. One sealed seven-condition waveform cache serves both MobileNet/ShuffleNet artifacts. `public_analysis.py` enforces complete paired cohorts, single-class denominators and existing guarded factorial estimation; `public_figures.py` exports executed SVG/300-DPI PNG. `shortcut.py` and audit_public_shortcuts group known source components and mark constant features noninformative. No new neural training, threshold fitting or confirmatory CLI is introduced.

Delivered Phase 3: real 192-record MLAAD-tiny development, 2,688/2,688 successful scores, 58 V2 / 40 selected legacy passing tests. Public development is not independent confirmation or a novelty certificate. Commands/inputs/resources: `docs/jaes_v2/PHASE3_NEXT_COMMANDS.md`. Earlier sections below record historical phases.

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

## Phase 2 continuation

`controls.py` adds reference-anchored absolute noise and local-SNR noise policies, headroom/PCM/resampling/identity controls and net-delay diagnostics. `phase2.py` executes the 14-condition bounded study and `control_analysis.py` validates full paired cohorts before analysis. None changes the sealed Phase 1 protocol or artifacts.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase2 --aggregate outputs/jaes_v2/c1_channel_controls_dev_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.reproduce_phase2 --run outputs/jaes_v2/c1_channel_controls_dev_v1 --report outputs/jaes_v2/audit/phase2_reproduction_next.json --waveforms
```

`datasets.py` implements official ASVspoof 2019 LA parsing with separate public-source fields and an adapter interface. It never downloads/copies audio or infers generators from attack IDs. Its local-release CLI and pending inputs are documented in PUBLIC_DATASET_IMPORTER.md. `overlap.py` separately reports identity overlaps and candidate-only acoustic near duplicates. `factorial.py` computes cluster-aware generator/attack × channel contrasts; missing/unknown/confounded factors fail rather than being invented.

The optional-spec production bug and two stale legacy tests are resolved by three explicitly recorded baseline amendments. Current final tests: 37 V2 / 40 selected legacy, all passing. The original snapshot remains immutable. Encoder/decoder component delays are not separately measured; the codec diagnostic reports net residual lag and available preskip metadata. No public audio was acquired, so the executed study is LEGACY DEVELOPMENT, not external confirmation.
