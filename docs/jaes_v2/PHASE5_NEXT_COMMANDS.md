# Phase 5 commands and Phase 6 prerequisites

Run in PowerShell at `D:/audio-deepfake-mobilenet-lstm` with `.venv/Scripts/python.exe`. Public CLIs were exercised. Existing evidence directories reject overwrite; **do not rerun completed prediction/training commands to regenerate tables**. A clean reproduction on another machine needs matching source snapshot, ImageNet weights and data hashes, plus fresh output identities.

Executed acquisition sequence:

```powershell
.\.venv\Scripts\python.exe -m src.lava.jaes_v2.asv_archive inspect
.\.venv\Scripts\python.exe -m src.lava.jaes_v2.asv_archive extract
.\.venv\Scripts\python.exe -m src.lava.jaes_v2.asv_archive import
```

Completed archive destination `data/public/asvspoof2019_la_v1`; inventory and foundation under `outputs/jaes_v2/datasets/`. These commands are a record of acquisition, not instructions to overwrite/re-extract. Future archives must use new `--destination`, `--audit`, `--output`.

Executed resource/cache sequence:

```powershell
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment preflight --report outputs/jaes_v2/audit/phase5_training_preflight_v2.json
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment cache --cache-output outputs/jaes_v2/c2_feature_cache_v2
```

V1 cache failed at its 600-second cap and remains as failure evidence. V2 pairs policies in the measured batch-two encoder call; it completed in 462.70 seconds. All supplied run configs explicitly pin V2. Its 128 train and 64 development sources have 1,792 shared-policy embeddings and no calibration/eval access. No need to recreate features for model-free aggregation.

One CLI per actual training experiment:

```powershell
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment train --config configs/jaes_v2/c2_overfit_sanity_v1.json
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment train --config configs/jaes_v2/c2_p0_erm_pilot_v2.json
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment train --config configs/jaes_v2/c2_p1_erm_pilot_v2.json
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment train --config configs/jaes_v2/c2_p0_aug_pilot_v2.json
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment train --config configs/jaes_v2/c2_p1_aug_pilot_v2.json
```

The B1 amplitude-only and B4 ordinary-consistency `*_planned_v1.json` configs are **not executed evidence**. B3 is an alias of B2. Do not count implementation as demonstrated training benefit. No new loss, KD or C3 policy is implemented.

Model-free commands that reuse completed scores (choose a fresh report output):

```powershell
.\.venv\Scripts\python.exe -m scripts.jaes_v2.c2_experiment aggregate --run outputs/jaes_v2/c2/c2_p0_erm_pilot_v2
.\.venv\Scripts\python.exe -m scripts.jaes_v2.phase5_report --output outputs/jaes_v2/audit/phase5_report_another_reproduction
```

Reports verify input seals and generate statistics/figures without neural inference. The two retained report versions prove byte-identical reproduction. `phase5_efficiency` is a **separate bounded full-model timing experiment**, not a model-free report command, and refuses to replace its measured JSON.

Verification record:

```powershell
.\.venv\Scripts\python.exe -m scripts.jaes_v2.validate --output outputs/jaes_v2/audit/phase5_validation_final_v3
.\.venv\Scripts\python.exe -m scripts.jaes_v2.finish_phase5
```

These are completed verification destinations; new verification outputs need new names. Interrupted training stores INTERRUPTED/FAILED state and available epoch/history/checkpoints. Automatic optimizer-state resume is not implemented; a new run must use a fresh ID, and must not be described as exact continuation.

Phase 6 prerequisites, before confirmatory training: finish the blocked selective-robustness prior-art technical comparison; decide whether any mechanism differs from known normalization/augmentation/consistency; validate stronger RawBoost and equal-view baselines, including B1/B4; lock a larger source-screened train/dev design; reserve calibration for the declared role; predeclare clean classwise tolerance and primary stress comparison; budget full finetuning on a measured compatible runtime; lock at least three seeds; choose an unused external corpus and perform overlap/storage audit **before** labels guide design. C3 additionally needs a usable stage-2 model and measured stage costs. No risk guarantee under arbitrary domain shift follows from reserved calibration alone. Existing ASV evaluation remains unopened and final confirmation is not locked.

Resource-driven training revision: P0 ERM v1 reached its 180-second cap after 13 epochs; P1 ERM v1 was explicitly interrupted with checkpoint/history retained. Neither enters the main factorial comparison. The four main v2 configurations were fixed to 6 epochs (48 optimizer updates each) before launching that group, with separate source-shuffle/view RNG streams. These are resource decisions, not selection of an algorithm by external/test scores. V1 augmentation runs were never launched.
