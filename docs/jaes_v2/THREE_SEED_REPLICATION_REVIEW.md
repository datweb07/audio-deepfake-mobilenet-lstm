# Three-seed replication review

**12 November 2026. Six sealed development runs verified.** The earlier resource block was a historical pre-training snapshot. The author subsequently completed both 9307 arms. V5 adds their development evidence.

## Integrity and fitted comparisons

The official verify CLI passed for both 9307 directories including every checkpoint file referenced by the seals. The analysis also verified all six seals, latest pointers, selected checkpoint controllers, the registry and manifest hashes, frozen source fingerprints, and identical within-pair initial weights and full training source order. The three initial weight fingerprints differ between seed blocks. Each run has 840 optimizer updates, 21 validation epochs and 2,576 unique predictions with matching labels, speaker/attack identity, seven conditions, FAKE orientation and fixed 0.5 threshold. The archived metric and score object hashes were recomputed exactly. Both 9307 selected exports and both selected/final checkpoint tensors were restored and hashed. Optimizer iteration counters matched the checkpoint steps. No forward pass, training update, calibration or new scoring was performed.

| Experiment | Selected checkpoint | Final checkpoint |
| --- | --- | --- |
| `p10_medium_erm_s9107_v2` | `u000560_e014_c000` | `u000840_e021_c000` |
| `p10_medium_channel_s9107_v2` | `u000760_e019_c000` | `u000840_e021_c000` |
| `p10_medium_erm_s9207_v2` | `u000520_e013_c000` | `u000840_e021_c000` |
| `p10_medium_channel_s9207_v2` | `u000720_e018_c000` | `u000840_e021_c000` |
| `p10_medium_erm_s9307_v2` | `u000360_e009_c000` | `u000840_e021_c000` |
| `p10_medium_channel_s9307_v2` | `u000680_e017_c000` | `u000840_e021_c000` |

The selected checkpoint is the recorded minimum clean BCE, using earlier epoch to break ties. The matched comparison holds architecture, sources, initialization, source scheduling, optimizer, budget and the checkpoint rule fixed within each seed. Model weights and selected epochs differ after fitting. The registered augmentation policy is the treatment.

## Per-arm development evidence

| Seed/arm | A clean AUC | Macro stress AUC | A clean REAL FA | A clean FAKE miss | A stress REAL FA | A stress FAKE miss | B clean REAL FA | A05 stress miss | A06 stress miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9107 ERM | 0.9872 | 0.6715 | 3/112 | 6/112 | 54/336 | 222/336 | 19/144 | 48/54 | 41/54 |
| 9107 CHANNEL | 0.9814 | 0.8026 | 3/112 | 9/112 | 45/336 | 155/336 | 31/144 | 50/54 | 45/54 |
| 9207 ERM | 0.9851 | 0.5946 | 4/112 | 9/112 | 167/336 | 143/336 | 26/144 | 38/54 | 16/54 |
| 9207 CHANNEL | 0.9809 | 0.7839 | 4/112 | 9/112 | 70/336 | 134/336 | 20/144 | 47/54 | 31/54 |
| 9307 ERM | 0.9822 | 0.5498 | 4/112 | 10/112 | 44/336 | 266/336 | 21/144 | 49/54 | 45/54 |
| 9307 CHANNEL | 0.9856 | 0.7934 | 6/112 | 8/112 | 61/336 | 119/336 | 30/144 | 43/54 | 28/54 |

## Descriptive paired differences and speaker uncertainty

| Metric | 9107 delta | 9207 delta | 9307 delta | Mean ± sample SD | Range | 9307 speaker 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean_auc | -0.0058 | -0.0041 | +0.0033 | -0.0022 ± 0.0049 | -0.0058 to +0.0033 | — |
| macro_stress_auc | +0.1312 | +0.1893 | +0.2436 | +0.1880 ± 0.0562 | +0.1312 to +0.2436 | +0.1884 to +0.3071 |
| A_clean_FA | +0.00 | +0.00 | +1.79 | +0.60 ± 1.0310 | +0.00 to +1.79 | -3.57 to +9.82 |
| A_clean_miss | +2.68 | +0.00 | -1.79 | +0.30 ± 2.2470 | -1.79 to +2.68 | -4.46 to +1.79 |
| A_stress_FA | -2.68 | -28.87 | +5.06 | -8.83 ± 17.7809 | -28.87 to +5.06 | -1.79 to +10.42 |
| A_stress_miss | -19.94 | -2.68 | -43.75 | -22.12 ± 20.6225 | -43.75 to -2.68 | -47.92 to -39.58 |
| B_clean_FA | +8.33 | -4.17 | +6.25 | +3.47 ± 6.6970 | -4.17 to +8.33 | -1.39 to +15.28 |
| A05_stress_miss | +3.70 | +16.67 | -11.11 | +3.09 ± 13.8992 | -11.11 to +16.67 | -16.67 to -5.26 |
| A06_stress_miss | +7.41 | +27.78 | -31.48 | +1.23 ± 30.1080 | -31.48 to +27.78 | -43.86 to -17.54 |

Error differences are percentage points. The original seed aggregator resamples speakers within stratum while retaining paired sources and all their stress views. The 9307 displayed intervals use 10,000 valid draws. Pointwise intervals quantify shared-cohort variation conditional on each fitted pair. Sample SD (ddof=1) describes variability between three fitted seed blocks. The supplement records each arm’s mean, sample SD, range and paired differences and retains all seeds’ cluster intervals.

## Transformation-order evidence

| Seed/arm | A REAL median score gap | A FAKE median score gap | B REAL median score gap | A REAL first-only errors | A FAKE second-only errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9107 ERM | +0.2540 | +0.3440 | +0.1682 | 43/112 | 48/112 |
| 9107 CHANNEL | +0.0884 | +0.0248 | +0.0810 | 19/112 | 17/112 |
| 9207 ERM | +0.3428 | +0.2780 | +0.3568 | 63/112 | 40/112 |
| 9207 CHANNEL | +0.1415 | +0.0327 | +0.1196 | 31/112 | 26/112 |
| 9307 ERM | +0.2033 | +0.2375 | +0.1694 | 37/112 | 41/112 |
| 9307 CHANNEL | +0.2024 | +0.0538 | +0.1754 | 44/112 | 33/112 |

The signed gap is score(noise then Opus) minus score(Opus then noise). The direction of the A class-specific order asymmetry appears in all three fitted pairs. The original standalone Noise10 and Opus24 development diagnostics cover 9107 and 9207. The nuisance model was fitted only for 9107. No standalone or held-out inference was run for 9307.

## What repeats and what reverses

- Stress macro AUC increases and stressed FAKE misses decrease in all three development pairs. ERM stress AUC is 0.6053 ± 0.0616, channel 0.7933 ± 0.0094 and the paired gain +0.1880 ± 0.0562. The smaller observed channel SD is descriptive for these seeds.
- Clean AUC falls in 9107/9207 and rises in 9307. Clean A FAKE misses increase, tie, then decrease. Stress A REAL false alarms improve in the first two seeds and worsen in 9307. B clean REAL false alarms rise in 9107 and 9307 and fall in 9207. The registered selected-pair 9307 guard flags B clean REAL false alarms alone. The +6.25-point observed difference exceeds the +2-point tolerance even though its speaker interval includes zero.
- A05/A06 increase in 9107 and 9207, then fall in 9307. A05 has six corrected stress-view errors, all in noise-before-Opus. A06 has 18 corrected and one introduced across the stress views. Recoveries are distributed across several speakers. The net A05/A06 improvements span five and six speakers, with largest individual net reductions of two and five views. A06 ERM burden is 16/54 in 9207 versus 45/54 in 9307. The 9307 subgroup mean FAKE scores rise under all three conditions. These score shifts and threshold crossings localize the effects. The class-only attack subsets do not support ROC-AUC. Ranking improves on the full A cohort in every registered stress condition in 9307. The causal mechanism behind the attack reversal remains untested.
- Held-out metrics were replayed from the original 1,088 shards and 17,408 scores for four models. The original A17–A19 adverse results still apply to that comparison. No result extends them to 9307 or to a new corpus.

## Resource evidence and decision

Seed 9307 optimizer/validation times were 39.9/37.7 min for ERM and 78.9/37.8 min for channel. Peak resident memory from the recorded CPU logs was approximately 714/896 MiB. The latest read-only resource snapshot recorded about 17.59 GiB free on D, 11.58 GiB free on C, 6.39 GiB available RAM and 12.16 GiB available commit. This is a point measurement.

**NO_FURTHER_EXPERIMENTS_FOR_DECEMBER.** The new pair directly addresses the initialization question for the existing cohort and exposes why the claim must remain endpoint-specific. A balanced nested 1,280 cohort is available, but a larger run would answer a separate training-size question. Fixing updates halves average per-source exposure. Fixing epochs doubles updates and changes the compute estimand. The released 640 profile has no approved isolated 1,280 exact-resume lifecycle, and checkpoint retention plus verification time still need qualification. Cross-corpus evaluation would add external validity if an independent subset were verified. The 34.5 GB DF archive exceeds current free space. The published maps show overlap and the candidate VCC subset still lacks audio/member and lineage reconciliation. MLAAD-tiny already informed development. For this revision, preserve these two limitations and reserve the remaining time for supervisor review and submission QA.

[Registered 9307 comparison](../../outputs/jaes_v2/phase10/medium_pair_s9307_v2.json) · [Three-seed aggregate](../../outputs/jaes_v2/phase10/empirical_execution_v1/paired_seed_evidence_9107_9207_9307.json) · [Verification and statistics](../../outputs/jaes_v2/three_seed_revision/verification_and_statistics.json) · [Weight restore](../../outputs/jaes_v2/three_seed_revision/9307_weight_restore.json) · [External audit](SUPERVISOR_EXTERNAL_AND_VENUE_AUDIT.md)
