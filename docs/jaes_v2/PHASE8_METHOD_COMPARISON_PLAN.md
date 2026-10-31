# Matched comparison plan

Registered configuration hashes live in `outputs/jaes_v2/phase8/protocol/baselines_v1.json`; tiny lifecycle configs use `protocol/tiny_v2.json`. Neither enables calibration/evaluation. No full run was automatically authorized by generating configs.

Main architecture: MobileNetV3Small-LSTM128, fresh ImageNet backbone and independent head seeds6107/6207/6307. Medium train640, development224A+144B; foundation train25,380 (2,580REAL/22,800FAKE), development17,503 excludes reserved speakers. Never interpret oversampled REAL views as new recordings. Natural and balanced protocols have different target distributions and separate IDs.

| Control | Objective/exposure | Research role |
|---|---|---|
| A0 | ERM, clean | ordinary representation baseline |
| A1 | one sampled clean/gain/noise10/Opus24 | channel augmentation |
| M1 | same bank, weighted BCE[1,1.5] | ordinary class emphasis |
| A2 | equal supervision of all four views | all-view exposure control |
| M2/A3 | seven source groups, all four views, adaptive q | smoothed group robustness |
| Consistency | two supervised forwards, .1 probability MSE | ordinary invariance |
| A4 | clean plus two fixed RawBoost realizations | finite bank; not original online reproduction |
| B0 / M3 ablations | NOT_CREATED | entry gate failed |

All three-seed medium/foundation families are prepared; longer jobs remain NOT_EXECUTED. Two-stage partial schedules use1warmup+20finetune epochs; head-only20epochs. These are neither equal-update nor compute-matched. Compare the actual journal, updates, views, repeated-source frequencies, trainable parameters, wall time, validation cost and checkpoints. A fair M2 effect additionally requires uniform q **with the same group-balanced source sampler**; that dedicated new experiment is not yet registered. The natural-sampling equal-view family alone cannot isolate q. Do not promote a DRO advantage to a mechanism claim without that control.

Selection uses primary A development only. Report fixed-threshold .5 REAL FA, FAKE miss, AUC/EER, probability quality, all conditions, all seen attacks and B one-class nulls. The guarded selector uses prospective absolute caps, not the retrospective choice of whichever class looks favorable. Primary scientific contrasts require paired initialization/order within seed and paired known speakers. Report separate seed SD and speaker bootstrap uncertainty; views and seed×recording products are not independent samples. Tiny lifecycle cohorts cannot support population significance.

To escalate a future M3: precise non-equivalent mechanism; locked matched controls including sampling; support across independent training seeds with both clean errors acceptable; independent source/license/overlap and calibration protocol locked before any confirmation. Existing order compositions have been inspected since Phase1 and cannot become untouched tests through renaming. Architecture transfer to ShuffleNet comes only after a credible MobileNet effect; no six-detector retraining or cascade is launched.
