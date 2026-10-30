# Phase 5 research findings — 28 October 2026

Status: **DATA_IMPLEMENTED; MECHANISM_DIAGNOSTIC_EVALUATED; TIER1_SMOKE_TESTED; FOUR_STANDARD_BASELINES_PILOT_EVALUATED.** C1 is supporting validation/engineering. C2 is not an established novel method. C3 is deferred. Confirmation is not locked.

## Controlled foundation and actual execution

The user supplied LA.zip (7,646,913,333 bytes). Selected train/dev extraction retained 3,178,404,021 bytes, with evaluation audio and labels unopened. Official CM records: train25,380, model-selection dev17,503, reserved calibration7,341. Full inventory/header/hash checks cover50,224 CM records; original content/generator lineage remains UNKNOWN. Local CRC/SHA establish local integrity; an official remote checksum is not certified. ODC-By database notices and upstream content conditions are retained. No additional dataset bytes were downloaded.

The actual pilot uses128 train (64/64,20 speakers) and64 development (32/32,16 speakers), plus1,792 frozen ImageNet embeddings across P0/P1 and paired views. Cache v1 stopped at600.006 seconds after1,648 embeddings and is unusable. Batch-two v2 completed in462.704 seconds (18,579,137 bytes including audits). The data are not a substitute for a large controlled training study.

**Important support imbalance:** selected train has20 REAL and20 FAKE official speakers; development has16 REAL speakers (two recordings each) but only7 FAKE speakers (four/five each). Official speaker-disjointness across splits does not make the within-dev class support matched. All group denominators are retained. The min-n4 worst-speaker statistic excludes the nine REAL-only n2 speakers; their individual classwise metrics remain in the full group table. Equal-speaker bootstrap and recording-average effects are different estimands. No original-utterance/source matching is certified. A separate metadata-selected28-record,7-speaker,2-per-class-per-speaker post-hoc sensitivity uses existing scores only: outputs/jaes_v2/audit/phase5_group_coverage.json. It does not replace the primary64-record cohort.

Original and prepared-input grouped shortcut accuracies on dev: duration .59375 (original; prepared constant), silence .5625/.578125, RMS .46875/.53125, numeric combined .578125/.53125. Sample rate is constant/noninformative. No perfect metadata shortcut was observed in these bounded folds; this does not exclude shortcuts. The reused source-audit helper also internally computed mixed local-parent diagnostics, which were discarded; only the separately speaker-grouped within-partition results are reported. Six classwise ASV train/dev versus already-used MLAAD raw/input fingerprints are retained in phase5_corpus_fingerprints.json. They are descriptive, not proof of domain equality or untouched external evaluation.

## Frozen-model preprocessing diagnostic

384 forwards cover32 MLAAD development originals, two old models and three policies at native/half gain.128 P0 bridge/embedding forwards explicitly check parity with sealed Phase4 scores;256 P1/P2 forwards provide new diagnostics. Old weights and operating thresholds remain fixed.

| Model / policy | Native error | Half-gain error | Native REAL FA | Native FAKE miss | Gain decision flips | Mean absolute score shift |
|---|---:|---:|---:|---:|---:|---:|
| mobilenetv3_lstm / P0 | 0.25000 | 0.34375 | 0.43750 | 0.06250 | 3 | 0.100226 |
| mobilenetv3_lstm / P1 | 0.12500 | 0.12500 | 0.12500 | 0.12500 | 0 | 0.000000 |
| mobilenetv3_lstm / P2 | 0.25000 | 0.25000 | 0.31250 | 0.18750 | 0 | 0.000000 |
| shufflenetv2_lstm / P0 | 0.18750 | 0.25000 | 0.25000 | 0.12500 | 2 | 0.062740 |
| shufflenetv2_lstm / P1 | 0.31250 | 0.31250 | 0.50000 | 0.12500 | 0 | 0.000000 |
| shufflenetv2_lstm / P2 | 0.18750 | 0.18750 | 0.25000 | 0.12500 | 0 | 0.000000 |

P0 maximum tensor residual is19.190689/255. Mean LSTM embedding L2 shifts are1.37910 MobileNet and1.30512 ShuffleNet. P1/P2 half-gain feature, embedding and score residuals are zero on this cohort. P1 improves MobileNet but worsens ShuffleNet discrimination; P2 also preserves gain stability with mixed discrimination. This supports a preprocessing contribution to observed gain sensitivity. Silence/reference/precision changes and distribution mismatch prevent attributing every classification change solely to the absolute floor. These frozen-model results do not establish a detector trained for P1. Mathematical derivation, deployment assumptions and silence failure cases are in PHASE5_PREPROCESSING_MECHANISM.md.

## Main controlled training pilot

The16-record overfit sanity reached1.0 train accuracy at epoch2, passed checkpoint reload and gradient checks. A preliminary P0 ERM v1 reached its180-second cap after13 epochs; P1 ERM v1 was controlled-interrupted after11 complete epochs, with checkpoint/history retained. Neither preliminary run enters the main comparison. Resource observations motivated four fresh v2 runs fixed at6 epochs/48 updates each, seed5804, same128/64 originals, initial ImageNet weights, fresh classifier, frozen backbone, batch16, Adam .001 and the same linear LR schedule. Source-order RNG is separate from view-choice RNG. All four completed their declared budget. This resource-informed development design is not an independently preregistered confirmatory study.

| Main cell | Clean error | Clean REAL FA | Clean FAKE miss | Clean AUC | Noise5 error | Noise5 FAKE miss | Mean 5-stress error | Gain-quarter flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c2_p0_erm_pilot_v2 | 0.18750 | 0.06250 | 0.31250 | 0.92969 | 0.40625 | 0.71875 | 0.31250 | 3 |
| c2_p1_erm_pilot_v2 | 0.15625 | 0.28125 | 0.03125 | 0.95801 | 0.46875 | 0.90625 | 0.32500 | 0 |
| c2_p0_aug_pilot_v2 | 0.23438 | 0.28125 | 0.18750 | 0.89941 | 0.40625 | 0.59375 | 0.29688 | 11 |
| c2_p1_aug_pilot_v2 | 0.18750 | 0.31250 | 0.06250 | 0.91504 | 0.39062 | 0.65625 | 0.29375 | 0 |

P1+ERM improves clean error from.1875 to.15625 but raises Noise5 miss from.71875 to.90625. Standard P1 augmentation reaches mean stress error.29375 versus P0 ERM.31250, while clean FAR changes from.0625 to.3125 and clean miss from.3125 to.0625. Even equal clean total error hides a large operating-point trade-off. P0 augmentation keeps Noise5 total error.40625 while swapping REAL/FAKE failures. Gain invariance therefore does not solve noise-induced evidence loss. No acceptable clean-risk margin was prospectively validated; these are not formal robustness wins.

Six conditions per64 originals give384 paired scores per main run,1,536 total. Variants are repeated measurements, not additional independent recordings. Full accuracy, precision, recall, specificity, F1, macro-F1, ROC-AUC, PR-AUC and diagnostic EER are stored in each metrics.json. Scores use fixed.5, without calibration fitting. EER thresholds are never used as policy thresholds. Per-speaker/per-attack denominators, clean-to-stress flips, raw-score shifts, worst-group sensitivity and2,000-replicate descriptive speaker bootstrap are in phase5_report_v2/summary.json. No p-values, multiplicity-adjusted discovery or independent-training-seed uncertainty are claimed. Noise interaction recording mean is−.078125, equal-speaker estimate−.044643 with descriptive95%[−.107143,0]; this is exploratory and composition dependent.

## Measured cost and reproducibility

All students have1,308,401 parameters and5,916,718-byte full Keras artifacts. End-to-end measured CPU cost includes decode/headroom, preprocessing and the full backbone+head, one declared3-second dev recording, five warm repetitions per model:

| Main cell | Train seconds | Model ms | Preprocess ms | End-to-end ms | Offline RTF |
|---|---:|---:|---:|---:|---:|
| c2_p0_aug_pilot_v2 | 96.99 | 276.81 | 11.06 | 291.63 | 0.09721 |
| c2_p0_erm_pilot_v2 | 93.51 | 295.13 | 11.38 | 309.51 | 0.10317 |
| c2_p1_aug_pilot_v2 | 85.07 | 299.24 | 14.86 | 318.36 | 0.10612 |
| c2_p1_erm_pilot_v2 | 82.24 | 307.29 | 14.52 | 324.18 | 0.10806 |

Timing process lifetime peak spans462.75–505.59 MiB across sequential model loads; it is not isolated per-model peak. Resource forward/backward preflight peak was1,147.29 MiB. No GPU was used. Warm offline RTF is not a streaming/energy certification, and small latency differences between identical architectures can reflect measurement order/noise. Teacher costs remain unmeasured.

Final regression:92/92 JAES V2 and40/40 selected legacy, zero failures/errors/skips, phase5_validation_final_v3/validation.json. Actual model-free reaggregation and figures were reproduced without TensorFlow/Torch imports; current report/reproduction are phase5_report_v2 and phase5_report_reproduced_v2. V1 reports remain superseded because their middle panel used inconsistent model colors; numerical statistics are unchanged. Eight requested figure topics are covered by five standalone multi-panel PNG artifacts: mechanism, gain/decision sensitivity, learning curves, clean/stress+classwise+ablation, efficiency/risk. Source and protected-artifact verification is recorded by phase5_final_integrity.json.

## Teacher, novelty and next decision

NII teacher: exact local artifact/hash absent; incompatible installed author-runtime prerequisites; quality/memory/latency unmeasured; corpus overlap with ASV/MLAAD not excluded. Existing environments remain unchanged. The tested two-logit FAKE-index helper is a software contract, not a measured checkpoint. No KD was run. B1 and B4 are implemented and tested but not executed scientifically; B3 is merged with B2.

P1 uses established relative normalization; B2 is standard augmentation. PCL-NET is direct paired-consistency prior art; DK-CAST overlaps codec-aware weighting/distillation; RawBoost is a required stronger augmentation baseline. The selective-robust-training primary-source access gate is unresolved. **No new loss or three international contributions have been established.** C1 remains supporting evidence. C2 now has a controlled data/training foundation and falsifiable negative/positive development observations. C3 remains deferred.

Phase6 must resolve prior-art differences, strengthen B1/RawBoost/equal-view consistency controls, address class/speaker support, lock larger source-screened splits and clean-risk tolerances, then budget at least three seeds and untouched external evaluation. Independent confirmation is NOT LOCKED. The ASV evaluation archive remains unopened. No extra dataset download is needed for the present Phase5 pilot. See PHASE5_NEXT_COMMANDS.md.
