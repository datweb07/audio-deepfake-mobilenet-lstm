# Phase6 current update —30 October2026

C1: SUPPORTING_VALIDATION_ENGINEERING. C2: STRONG_BASELINES_PILOT_EVALUATED; NOVELTY_NOT_ESTABLISHED. C3: DEFERRED. Confirmation: NOT_LOCKED. Three international contributions: NOT_ESTABLISHED.

New speaker-balanced cohort160train/56dev (four/class/speaker), nine60-update standard baselines and real partial-backbone sanity completed. A01–A04-only coverage, one seed and UNKNOWN original lineage limit claims. Equal-view consistency improves the declared development contrast; channel/RawBoost reduce misses with false-alarm trade-offs. New loss remains gated. Sequential execution/microbatching address the observed memory-pressure failure; no production environment or sealed historical evidence is overwritten.

Current handoff: LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT.md and LAVA_JOURNAL_V2_RESEARCH_INVENTORY.json, generated from structured evidence. See PHASE6_EXPERIMENT_RESULTS.md and PHASE6_NEXT_COMMANDS.md. Earlier sections below remain historical.

---

# Phase 5 current update — 28 October 2026

**C1: SUPPORTING_VALIDATION / ENGINEERING. C2: STANDARD_BASELINES_PILOT_EVALUATED; NOVELTY_NOT_ESTABLISHED. C3: DEFERRED. INDEPENDENT_CONFIRMATION: NOT_LOCKED. THREE_INTERNATIONAL_CONTRIBUTIONS: NOT_ESTABLISHED.**

ASV controlled foundation:50,224 official CM records; train/dev extraction only; calibration reserved and evaluation unopened. Main pilot:128 train/64 dev, frozen ImageNet MobileNet-LSTM, four6-epoch/48-update cells, one seed. P1 removes gain flips but can worsen noise misses; augmentation changes clean classwise trade-offs. Development FAKE covers7 speakers versus16 for REAL, so support imbalance remains a scientific limitation. Teacher runtime/quality gates are unresolved; no KD/new loss/C3 routing was run.

Measured regression:92/92 JAES V2 and40/40 selected legacy, no failures/errors/skips. Current evidence and full metrics: **PHASE5_RESEARCH_FINDINGS.md**; commands and remaining gates: **PHASE5_NEXT_COMMANDS.md**. Standard controls and software correctness do not establish journal novelty. Earlier sections below are historical and preserved.

---

# Resource preflight and tiers

## Phase 4 measured continuation

Initial free disk: 39,125,602,304 bytes; saved current preflight. No audio/teacher weight acquisition or dependency upgrade. New nuisance cache: 74,368,583 bytes including 288 shared waveforms; preparation 84.69 s. Each detector performed 288 new scored predictions plus one warm-up and reused 224 scores. Concurrent CPU jobs took 118.79 / 148.61 s including loading/reporting, not a speed benchmark. Model-free diagnostics/planning use small additional outputs. Scientific scaling remains gated by source support and independent-group precision, not available disk alone.

## Phase 3 measured continuation

Pre-acquisition free disk: 39,548,977,152 bytes, rechecked rather than assumed. Raw selected audio: 43,492,994 bytes; shared cache/audits: 349,010,430 bytes; metadata/license payloads: 2,634,680 bytes plus API inventories. No full audio archive, dependency upgrade or paid computation.

Preparation: 189.59 s after integrity/shortcut audit. Concurrent CPU artifact jobs: 524.15 s MobileNet / 648.89 s ShuffleNet including load/reports; not latency comparisons. Saved environment/FFmpeg/disk evidence: phase3_before.json and phase3_resource_preflight.json. New acquisitions require a current free-disk check and byte budget.

Measured environment: Python 3.11.9; TensorFlow 2.15.0; NumPy 1.24.3; SciPy 1.11.4; soundfile 0.12.1; scikit-learn 1.3.2; matplotlib 3.8.2; ONNX Runtime 1.17.3; FFmpeg 8.0.1 full build. All were already installed; no dependency versions were changed.

This execution host reports 12 logical CPUs, an NVIDIA RTX 3050 Laptop GPU with 4,096 MiB VRAM and approximately 40.3 GB free on D: before the pilot. These are **current host measurements**, not the conference paper's Ryzen desktop specifications. C1 uses CPU-only TensorFlow, one intra-op and one inter-op thread. GPU presence does not imply a validated TensorFlow CUDA runtime; it was not used.

The isolated `.venv-torch` was also probed: PyTorch `2.2.2+cpu`, `torch.cuda.is_available() == False`, no CUDA runtime. Using the physical GPU for C2 therefore requires a separate compatibility/environment decision before any memory dry run; the current torch environment cannot use it.

Tier 1 completed: protected-file hashes, source integrity, transform tests including actual FFmpeg, small inference, paired risk analysis and diagnostic figures. The first 60-score pilot took approximately 37.7 s including preflight-excluded run work, transforms, model loading, API parity and plotting; consult run_metadata.json for measured stage times. It is not an efficiency benchmark and does not replace the conference timings.

Final pilot v3 took 41.23 s of run work, including 10.47 s transform generation and 3.84 s model loading. It differs from earlier executions in deterministic WAV-container serialization, not cohort/perturbation selection. Timing variability is not a model performance claim.

Tier 2 pending: teacher evaluation, student training, ablations and multiple seeds. The 4 GiB device requires an actual teacher forward/backward dry run and memory profiling before choosing batch size, freezing, caching or checkpointing. No GPU-hour estimate is asserted without that test. A generic large-scale corpus download is inappropriate with current disk headroom.

Tier 3 pending: physical replay, real edge runtime, streaming state/latency and instrumented energy. None has been measured. There is no rental, cloud charge, external upload or full-training job in this execution.

Next resource decision: choose a license-compatible metadata-rich corpus that fits disk, validate a teacher and run a bounded dry run. Do not retrain all six conference artifacts merely to begin V2.

## Phase 2 measured workload

The 64-record / 14-condition CPU development run completed in 362.12 s (about 6 minutes), with 896 successful predictions and no exclusions. Total run artifacts occupied 214,810,029 bytes (about 205 MiB). Its preflight reserved one additional GiB beyond a conservative uncompressed-audio estimate; roughly 39 GB remained available. Full waveform reproduction was model-free and did not retrain anything. Diagnostic per-row timing is not a warm repeated latency benchmark.

No public audio or archive was downloaded. The small Edinburgh license request failed HTTP 403 after an approved network retry. No multi-GB download, GPU workload, dependency upgrade or paid service was started. Public-corpus extraction/cache size remains unverified and must be budgeted before acquisition.
