# LAVA Journal V2 Phase 2 research status

Updated 25 October 2026, Asia/Novosibirsk.

**Phase 2: CODE COMPLETE — EXTERNAL DATA PENDING.**
Conference scientific artifacts: preserved. Three explicit legacy source/test amendments are backed up and registered separately; the original freeze was not overwritten.
C1: **DEVELOPMENT_EVALUATED — CONFOUNDER-AWARE CHANNEL STUDY**.
C2: teacher/prior-art feasibility only; no learner trained.
C3: prerequisites/metrics/baselines only; no policy implemented.
Scientific novelty: **NOT ESTABLISHED**. No external confirmation was performed.

## Actual new execution

Run: `outputs/jaes_v2/c1_channel_controls_dev_v1/`.
Config: `configs/jaes_v2/c1_channel_controls_dev_v1.json`.
64 legacy validation recordings (32 REAL/32 FAKE), seed 2804, MobileNet existing checkpoint and inherited threshold 0.82. There are 64 checksum-local parent keys; the number of truly independent original recordings/speakers is UNKNOWN. No new training or threshold fitting occurred.

Fourteen conditions yield **896/896 successful predictions, 0 failures, 0 exclusions**: no-headroom clean, prepared clean, lower-headroom sensitivity, PCM16, FFmpeg PCM identity, resampling-only, native/aligned Opus, and noise plus both orders under two noise policies. Runtime was 362.12 s; artifacts approximately 205 MiB. This is a bounded exploratory study, not a warm latency benchmark or powered confirmation.

Paired raw scores, condition evidence, component diagnostics, 60 descriptive contrasts and PNG/SVG are stored. The primary fixed-power order error difference was 0.125 with descriptive local-parent bootstrap 95% interval approximately [0.046875, 0.219141]. Local-SNR gave the same decisions and error contrast, despite after-codec noise power ranging 0.91486–1.00499 times the fixed reference (mean 0.96852). Raw-score effects were not identical. Unknown source grouping and development selection prevent population/causal/novelty claims.

Impulse and all 64 speech diagnostics selected residual lag 0. Opus skip metadata was 312 samples at a 48 kHz encoded clock. Encoder/decoder component delay is NOT_SEPARATELY_IDENTIFIABLE. The native codec path was never silently aligned.

## Regression and reproduction

- Before: V2 19/19, selected legacy 37/40.
- After: **V2 37/37, selected legacy 40/40**, no skips. Evidence: `outputs/jaes_v2/audit/phase2_validation_v2/validation.json`.
- Reproduction: 907 sealed files verified; all 896 waveform files regenerated with hash parity; model-free aggregation passed. Evidence: `outputs/jaes_v2/audit/phase2_reproduction.json`.
- Original 2,376-file snapshot retained, with exactly three approved source/test exceptions recorded in `phase2_baseline_amendment.json`. All model/manifest/paper/historical-result hashes match. All 71 files of Phase 1 pilot v3 remain unchanged; that pilot was not rerun.

## Public data integration

ASVspoof 2019 LA is the primary official-source importer target. Typed official-protocol parsing, audio availability/labels, source records, separate official/research splits, checksums and partition/decoded audits are implemented. Generator identity and model-relative familiarity are not inferred from attack IDs. Public development execution rejects train/eval/test/calibration records.

Actual external audio installed/downloaded in this phase: **0 bytes**. The official small license request failed DNS inside the sandbox and HTTP 403 on an approved retry. There was no access-control bypass or bulk download. Protocol fixtures test correctness only and are not a public corpus. Acquisition evidence: `outputs/jaes_v2/datasets/asvspoof2019_acquisition_v1/source_metadata.json`.

The factorial estimator is implemented and tested on hand-computed fixtures, including unequal cluster sizes, shared speaker dependence, missing cells, one-class handling and corpus/factor confounding. It was deliberately not run on legacy UNKNOWN attack/generator identities.

The 64-source audit found no exact decoded duplicate or high-correlation candidate under its limited prefix screen; it cannot prove speaker/generator/original-parent independence. Shortcut accuracies were duration .515625, sample rate .53125, silence .625, RMS .59375 and combined .640625, using development-only leave-one-local-parent-out folds. See DATASET_SHORTCUT_AUDIT.md.

## Exact next commands

Run from the repository root, using the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.runner --preflight
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase2 --aggregate outputs/jaes_v2/c1_channel_controls_dev_v1
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests/jaes_v2 -v
```

For explicit full waveform reproduction, use a new report filename:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.reproduce_phase2 --run outputs/jaes_v2/c1_channel_controls_dev_v1 --report outputs/jaes_v2/audit/phase2_reproduction_next.json --waveforms
```

For a persisted full selected regression check, use a new output directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.validate --output outputs/jaes_v2/audit/phase2_validation_next
```

Tests manipulating temporary files may require an approved unsandboxed run on this Windows setup. Do not rerun the original 12-item pilot or rerun inference simply to read results. Existing run directories are never overwritten. Aggregation intentionally rebuilds derived reports after hash validation.

For future actual public audio, see PUBLIC_DATASET_IMPORTER.md. Its CLI is implemented, but the actual local release and selection-file paths are explicitly pending. The bounded runner supports a validated public dev manifest through `ControlConfig.public_manifest`; no confirmation flag is exposed.

## Remaining blockers

- DATA_BLOCKER: accessible official protocol/audio subset, release checksums and verified attack/source lineage needed. No real public integration claim yet.
- COMPUTE_BLOCKER: existing Torch is CPU-only; the 4 GiB physical GPU is not a validated teacher runtime. No environment upgrade or full training was attempted.
- NOVELTY_BLOCKER: protocol hardening and development effects do not establish a distinction from close prior work.
- VALIDATION_BLOCKER: generator-factor external evaluation, controlled trained baselines, multi-seed, component-delay separation and independent confirmation remain absent. True original grouping remains unknown.
- JOURNAL_ELIGIBILITY_BLOCKER: accepted venue/copyright/publication details and official current indexing confirmation remain unresolved.

Next milestone: provide a license-compatible official ASVspoof LA development subset with its original CM protocol, validate it through the importer, lock factor membership relative to actual checkpoint training evidence, and execute public development before registering an untouched confirmation cohort. C1_CONFIRMATORY_PROTOCOL.md is deliberately NOT LOCKED.
