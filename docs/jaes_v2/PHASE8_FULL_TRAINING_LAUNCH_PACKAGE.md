# Full-training launch package

This package authorizes no automatic long job. The author explicitly runs the commands below after reading FULL_TRAINING_READINESS. Novel-method training is NO-GO; standard baseline research can proceed only when the measured code/resource gates say YES. All commands start in `D:/audio-deepfake-mobilenet-lstm` using the existing `.venv`. No production environment or conference checkpoint is replaced.

## Prerequisites and first order

Keep extracted ASV train/development audio, ImageNet weight cache and ffmpeg/libopus available. Verify the accepted manuscript, old evidence, source contracts and new tests; leave reserved calibration/evaluation closed. Keep at least15GiB D free after the projected outputs, and additional C space for the OS. TEMP/TMP and plotting cache are redirected to `.research_tmp` on D by the CLI. One heavy job at a time uses the inherited OS file mutex; the kernel releases it on process death.

Start with **medium A0 ERM partial**:640train,224A+144Bdevelopment,1warmup+20partial epochs, microbatch1/accum16, fresh seed6107, Adam.001 warmup/.0001 partial, frozen BN. Then run channel augmentation under the same staged schedule. These are development baselines, not confirmatory experiments. The first preflight commands create new JSON files and fail rather than overwrite an existing file. If rerunning preflight, choose a new output name.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment preflight --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json --output outputs/jaes_v2/phase8/preflight/author_medium_erm_6107.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment verify --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_medium_channel_partial_s6107_v1.json
```

After inspecting both classes and resources, run seeds6207/6307 in sequence using their existing configs. Weighted, equal-view, consistency, smoothed-DRO and finite-RawBoost controls have separate existing configs for each seed. Do not silently change LR, threshold, sampling or a config's hash. A new setting requires a new ID/protocol; `phase8_register` can copy an existing setting to a new CPU/GPU identity, not change scientific parameters invisibly. No B0/M3 family exists because the novelty entry gate failed.

## Foundation family, only after medium results and renewed storage gate

Foundation uses all25,380 training sources, natural prevalence (2,580REAL/22,800FAKE), and17,503 model-selection development sources. It still excludes7,341 reserved calibration and all official evaluation. Final evaluation measures every allowed development source under seven conditions. This is substantially more costly than the medium cohort and is NOT_EXECUTED in this phase.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment prepare-data --config configs/jaes_v2/phase8/p8_foundation_erm_partial_s6107_v1.json --output outputs/jaes_v2/phase8/preflight/author_foundation_erm_6107.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_foundation_erm_partial_s6107_v1.json
```

`prepare-data` verifies manifests, source byte hashes and known partition/speaker separation and computes a conservative storage projection. It does not create a giant feature/Mel cache, copy the archive, extract evaluation or certify unknown original lineage. Fine-tuning streams P0 tensors through the real CNN. Head-only foundation training is explicitly rejected because no corresponding frozen cache is prepared. Medium head-only uses the sealed Phase7 cache.

## Outputs and monitoring

Each ID creates `outputs/jaes_v2/phase8/training/<experiment_id>/` once: resolved config; source/code/environment contract; dataset audit; initialization/trainable manifest; per-update source/view/loss/LR journals; atomic optimizer checkpoints; latest pointer; history; selected `best_epoch_*.keras` and its SHA; selected-model scores; condition/attack classwise metrics; resources; terminal status; recursive seal. Checkpoint names include epoch/step/cursor/pending-validation state. Head/partial initialization is fresh; Phase7 training checkpoints are never used as a warm start.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment monitor --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
```

Monitor reads atomic state while the job owns the lock. Its rough ETA uses recent journal filesystem wall intervals; pauses/validation may distort it and final evaluation is excluded. Use the measured resource table for a planning range, not a promised completion time. Default limits are4,000MiB lifetime process peak,15GiB disk reserve and24h per invocation. A budget stop is a recoverable interruption, not a completed experiment. GPU memory/timing is unverified; no CUDA speedup is assumed.

## Resume and failure recovery

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment resume --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
```

Resume reconstructs the same architecture/trainable-variable set, verifies config/data/code/environment, validates checkpoint hashes and the committed update-journal hash chain, restores all Adam variables/iteration and explicit controller cursor/dropout counter, then recreates deterministic epoch/view sampling. Checkpoints occur at complete optimizer boundaries; accumulators are zero there. Interrupted uncommitted microbatches are replayed from that boundary. This is not weight-only restart. CPU bitwise verification is limited to the tested TF2.15/Python/runtime; switching GPU/version is not exact resume.

For power loss/Ctrl+C, preserve every file and resume the last published pointer. Unique `.pending.*` directories remain unpublished evidence and are never loaded as checkpoints or automatically deleted. A committed orphan may be reused only when its sealed controller matches the replayed state. If failure occurs before the first committed checkpoint, exact continuation is unavailable: preserve the failed run and register a fresh ID.

For OOM, close other heavy jobs and restart from a committed boundary only if unchanged config/runtime can meet the budget. Reducing microbatch changes dropout grouping and the config, so it requires a new experiment identity; no false exact-resume claim. For disk stops, make user-approved space available outside protected evidence, or plan a new experiment; do not delete Phase1–7 or conference artifacts. For NaN/Inf, inspect the preserved failure and journal; fail-closed gradients and metrics prevent publication as success. Changing LR/data requires a new protocol. For corrupt checkpoint/hash failure, preserve the corrupt files. An earlier sealed boundary can be selected only through an explicitly audited pointer-recovery procedure; no silent fallback or hash rewrite is provided in this release.

Early stopping stops when the declared criterion fails to improve for5epochs. If guarded selection has no feasible epoch, status is NO_FEASIBLE_CHECKPOINT; there is no fallback winner. Final selected-model development evaluation follows completion or early stop. If interrupted during final evaluation, resume redoes evaluation from the committed finished training state. There is no separate advertised evaluate-development CLI operation.

## Reload verification, aggregation and handback

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment verify --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment aggregate --output outputs/jaes_v2/phase8/author_statistics_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment validate --output outputs/jaes_v2/phase8/author_validation_review
```

Verify checks sealed bytes; selected models have actual reload evaluation during training and separate tiny full-model/export-parity checks. The standalone verify operation does not claim an additional neural forward. Aggregation is model-free and retains incomplete/negative runs separately. Keep all configs, logs, failures, contracts, checkpoint/export hashes, history, scores, classwise/per-attack metrics, resources and seals for research handback. Report independent seed identities and both error classes, not just a best score. A new final journal table must use these real outputs; no expected score is supplied here.
