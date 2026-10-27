# LAVA Journal V2 Phase 3 research status

Updated 27 October 2026, Asia/Novosibirsk.

**Phase 3: EXTERNAL_DEVELOPMENT_EVALUATED.** Real public acquisition/import, shared controlled waveforms and two existing detector runs are complete. Scientific novelty: **NOT_ESTABLISHED**. Confirmation: **CONFIRMATION_NOT_LOCKED**. C2 teacher feasibility and C3 prerequisites remain pending. The dated Phase 2 record below is historical.

## Executed Phase 3 evidence

- MLAAD-tiny official repository, immutable revision `9143e5ea709575ebab6bec52840a1043aada7bb1`.
- 192 actual downloaded audio / 43,492,994 bytes; every upstream content SHA-256 matches. FAKE CC-BY-NC-4.0, REAL retained M-AILABS original notice; noncommercial research purpose, no audio publication.
- 96 English REAL + 96 English FAKE, four author-annotated system groups, 24 FAKE each. All selected source audio is 16 kHz. No performance-based cohort selection.
- Four FAKE original source-text speakers balanced six recordings per speaker/system. Synthesized/target voice, REAL speaker/parent and detector-relative familiarity remain UNKNOWN. Independent population units remain UNKNOWN.
- Seven prespecified conditions / **1,344 shared waveform files**: native input, prepared clean, PCM16, Opus24, fixed-absolute noise10dB, both orders. Exact same cache reused by both detectors; no retraining/calibration.
- **MobileNet: 1,344/1,344 successful; ShuffleNet: 1,344/1,344 successful; zero failures/exclusions.** Thresholds retained at .82 and .12. Concurrent CPU runs took 524.15 and 648.89 s; not speed benchmarks.

| Artifact / selected condition | Error | REAL false alarm | FAKE miss | ROC-AUC | Diagnostic EER |
|---|---:|---:|---:|---:|---:|
| MobileNet prepared clean | .28125 | .50000 | .06250 | .81955 | .23958 |
| MobileNet noise→codec | .48958 | .02083 | .95833 | .67795 | .35417 |
| MobileNet codec→noise | .47917 | .09375 | .86458 | .68316 | .36458 |
| ShuffleNet prepared clean | .24479 | .25000 | .23958 | .84679 | .23958 |
| ShuffleNet noise→codec | .38021 | .11458 | .64583 | .60059 | .43750 |
| ShuffleNet codec→noise | .36979 | .22917 | .51042 | .64475 | .38542 |

Full executed tables and interpretation: PHASE3_RESEARCH_FINDINGS.md. Both artifacts have only a 2/192 net order-error difference, concealing nine or thirteen additional FAKE misses offset by fewer REAL false alarms. Neither architecture superiority nor a novel mechanism follows. Headroom also changes results relative to native input. Ranking diagnostics do not authorize fitting replacement thresholds.

## Phase 3 validity, tests and reproduction

The actual public-source factorial estimator now has 12 annotation-defined pairwise group/channel contrasts per detector, using FAKE misses and joint original source-text-speaker bootstrap. Four source clusters give limited uncertainty support; these are observational descriptive results. Single-class AUC/EER/F1 are missing. No multiplicity-adjusted significance or unseen-generator claim is asserted.

No exact byte overlap with the canonical legacy manifest, within-cohort decoded duplicate, or bounded near-duplicate candidate was detected. This does not certify speaker/source independence. REAL source lineage and exhaustive decoded legacy overlap remain unresolved.

Grouped numeric shortcuts attain about .849 on original recordings and .880 on prepared inputs; prepared silence-only about .849. Folds retain known source components but REAL remains locally grouped; fold imbalance and dummy baseline are reported. Constant sample-rate features are noninformative. These diagnostics expose available dataset fingerprints, not detector feature causality.

Regression: **58/58 V2, 40/40 selected legacy, zero failures/errors/skips**, persisted in `outputs/jaes_v2/audit/phase3_validation_v2/validation.json`. Baseline preflight preserves 2,376 protected files with the same three Phase 2 amendments. All 979 files from the delivered Phase 1/2 runs remain unchanged; no old inference was rerun.

Reproduction reports: `phase3_mobilenet_reproduction_v1.json` and `phase3_shufflenet_reproduction_v1.json` under outputs/jaes_v2/audit. Both verify 19 sealed run files plus the shared cache, reproduce exact statistics and import neither TensorFlow nor Torch. This is model-free aggregation/hash verification, not a claim that all Phase 3 waveforms were regenerated a second time.

## Phase 3 evidence paths and next steps

- Discovery/import: `outputs/jaes_v2/datasets/`; raw audio: `data/public/mlaad_tiny_9143e5e_dev_v1/`.
- Shared waveforms/audits: `outputs/jaes_v2/c1_mlaad_tiny_dev_v1_waveforms/` (349,010,430 bytes).
- Scores/statistics/SVG/300-DPI PNG: detector directories `c1_mlaad_tiny_dev_v1_mobilenetv3_lstm/` and `c1_mlaad_tiny_dev_v1_shufflenetv2_lstm/`.
- Six programmatic research tables: `outputs/jaes_v2/audit/phase3_report_v1/`.
- Grouped shortcuts: `outputs/jaes_v2/audit/phase3_shortcuts_v1/`.
- Selection/config: `configs/jaes_v2/mlaad_tiny_dev_selection_v1.json` and `c1_mlaad_tiny_dev_v1.json`.

Executable commands, required inputs and resource estimates: PHASE3_NEXT_COMMANDS.md. Use aggregate/verify for delivered results; new acquisitions/runs require new names. No confirmatory CLI flag or placeholder C2/C3 training command exists.

RADAR dev metadata/license was downloaded and verified; audio and evaluation labels were not acquired. ASVspoof official README again returned 403; its adapter remains supported. Details: PHASE3_DATASET_DECISION.md, PHASE3_DATA_ACQUISITION.md, PHASE3_PROVENANCE_AUDIT.md and NEAREST_PRIOR_ART.md.

C1 Phase 4 can pursue additional provenance-controlled development and design an untouched confirmation cohort, but execution is blocked on independent lineage/precision/factor/threshold/novelty locks. C2 needs a licensed, loaded and quality-validated teacher plus a distinct mechanism; C3 needs a validated student, independent calibration and measured routing cost. Journal eligibility still requires accepted venue/copyright/publication details and current official indexing verification. No manuscript was promoted to journal-ready.

---

## Historical Phase 2 status — superseded by the Phase 3 record above

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
