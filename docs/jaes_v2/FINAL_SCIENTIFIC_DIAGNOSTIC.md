# Scientific diagnostic of the matched seed 9107 pair

09/11/2026, Asia/Ho_Chi_Minh. ERM and channel completed 840 updates in 21 epochs. We verified full run seals, selected and final checkpoint seals, the manifest and audio bytes, source order, contracts and historical qualification. The selected checkpoints are ERM 560 and channel 760. The experiment differs in augmentation policy. Each score grid contains 2,576 rows from 368 source recordings and seven conditions. Row alignment used source, condition and stratum and checked parent, speaker, attack and label against the development manifest. The threshold is 0.5 with ties assigned to FAKE. [Machine-readable audit](../../outputs/jaes_v2/phase10/final_diagnostic_v1/scientific_evidence.json)

## Ranking and paired decisions

Paired bootstrap uses 10,000 draws, seed 91109. A has seven speakers and B has nine. Each draw keeps all recordings and their views inside a sampled speaker. Intervals below are descriptive for the selected models. They do not quantify variation over training seeds. Invalid draws are counted. Sparse attack support and few speaker clusters limit precision.

| Endpoint | ERM | Channel | Channel−ERM and 95% interval |
| --- | ---: | ---: | --- |
| A clean REAL false alarm | 3/112 | 3/112 | +0.00 pp, [-2.68, +2.68] (10000/10000 valid) |
| A clean FAKE miss | 6/112 | 9/112 | +2.68 pp, [+0.00, +6.25] (10000/10000 valid) |
| B clean REAL false alarm | 19/144 | 31/144 | +8.33 pp, [+1.39, +17.36] (10000/10000 valid) |
| A stress REAL false alarm | 54/336 | 45/336 | -2.68 pp, [-8.63, +4.76] (10000/10000 valid) |
| A stress FAKE miss | 222/336 | 155/336 | -19.94 pp, [-25.31, -14.58] (10000/10000 valid) |

Clean A ROC-AUC / EER is 0.9872 / 4.46% for ERM and 0.9814 / 4.91% for channel. Stress macro AUC changes 0.6715 → 0.8026, delta +0.1312, 95% interval [+0.02, +0.24] (10000/10000 valid). AUC preserves the three-condition macro; ROC/DET coordinates in JSON are descriptive only. B has no two-class metrics. The selected endpoints fail the registered A clean miss, B clean false-alarm and A06 stress miss guards. No conclusion follows for every epoch.

| Condition | A REAL false alarms ERM → channel | A FAKE misses ERM → channel | B REAL false alarms ERM → channel |
| --- | --- | --- | --- |
| clean | 3/112 → 3/112 | 6/112 → 9/112 | 19/144 → 31/144 |
| gain_half | 0/112 → 0/112 | 15/112 → 14/112 | 4/144 → 7/144 |
| gain_quarter | 7/112 → 0/112 | 20/112 → 17/112 | 3/144 → 1/144 |
| noise5 | 1/112 → 8/112 | 99/112 → 64/112 | 2/144 → 7/144 |
| opus64 | 6/112 → 4/112 | 4/112 → 6/112 | 40/144 → 38/144 |
| A_then_B | 48/112 → 28/112 | 38/112 → 37/112 | 34/144 → 32/144 |
| B_then_A | 5/112 → 9/112 | 85/112 → 54/112 | 2/144 → 13/144 |

## Attack-specific errors

Each entry is ERM → channel. Stress contains three views per source. All-stress wrong counts the intersection across the three conditions. Full seven-condition failure counts and cross-model transitions are retained in JSON.

| Attack (sources) | Clean | Noise5 | Noise→codec | Codec→noise | All-stress wrong | Total stress and paired 95% delta interval (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| A01 (19) | 0/19 → 0/19 | 14/19 → 2/19 | 0/19 → 0/19 | 7/19 → 0/19 | 0 → 0 | 21/57 → 2/57, [-40.00, -25.93] (10000/10000 valid) |
| A02 (19) | 0/19 → 1/19 | 19/19 → 19/19 | 9/19 → 9/19 | 19/19 → 18/19 | 9 → 9 | 47/57 → 46/57, [-13.73, +10.00] (10000/10000 valid) |
| A03 (19) | 1/19 → 0/19 | 17/19 → 7/19 | 7/19 → 0/19 | 14/19 → 1/19 | 7 → 0 | 38/57 → 8/57, [-71.93, -33.33] (10000/10000 valid) |
| A04 (19) | 2/19 → 2/19 | 14/19 → 3/19 | 1/19 → 0/19 | 12/19 → 1/19 | 1 → 0 | 27/57 → 4/57, [-56.25, -25.00] (10000/10000 valid) |
| A05 (18) | 0/18 → 0/18 | 17/18 → 17/18 | 14/18 → 16/18 | 17/18 → 17/18 | 14 → 16 | 48/54 → 50/54, [-3.70, +12.70] (10000/10000 valid) |
| A06 (18) | 3/18 → 6/18 | 18/18 → 16/18 | 7/18 → 12/18 | 16/18 → 17/18 | 6 → 12 | 41/54 → 45/54, [-6.67, +22.92] (10000/10000 valid) |

## Clean B false alarms by speaker

| Speaker | ERM → channel | New false alarms / recovered | New alarms within 0.1 of ERM threshold | Mean score shift |
| --- | --- | ---: | ---: | ---: |
| LA_0099 | 0/16 → 0/16 | 0 / 0 | 0 | +0.023 |
| LA_0100 | 0/16 → 6/16 | 6 / 0 | 1 | +0.330 |
| LA_0102 | 4/16 → 5/16 | 1 / 0 | 0 | +0.079 |
| LA_0103 | 4/16 → 5/16 | 2 / 1 | 0 | +0.011 |
| LA_0104 | 2/16 → 2/16 | 0 / 0 | 0 | +0.027 |
| LA_0105 | 1/16 → 1/16 | 0 / 0 | 0 | +0.068 |
| LA_0106 | 3/16 → 3/16 | 1 / 1 | 0 | +0.034 |
| LA_0107 | 5/16 → 9/16 | 4 / 0 | 0 | +0.250 |
| LA_0108 | 0/16 → 0/16 | 0 / 0 | 0 | +0.054 |

New alarms total 14 and two were recovered, with one new alarm within 0.1 of the original threshold. New-alarm score shifts span 0.227 to 0.708 (median 0.462). Mean scores rose in all nine clusters, but LA_0100 and LA_0107 account for 10 of the 12 net added errors. The rise combines broad score drift with concentrated threshold crossings. Speaker IDs identify clusters only.

## Order and matched components

Transition columns count both correct, both wrong, wrong only under noise→codec, and wrong only under codec→noise, in that order. Logit diagnostics clip probabilities to [1e-6, 1−1e-6] before transformation. Original probabilities and decisions are unchanged.

| Group | Model | Order error transitions | Mean score / logit difference | Mean excess composition | Noise10 / Opus24 errors |
| --- | --- | --- | --- | --- | --- |
| A_PAIRED/0 | erm | 64 / 5 / 43 / 0 | +0.3206 / +2.1967 | -0.0228 | 6/112 / 21/112 |
| A_PAIRED/0 | channel | 84 / 9 / 19 / 0 | +0.1020 / +0.4848 | +0.0094 | 11/112 / 5/112 |
| A_PAIRED/1 | erm | 26 / 37 / 1 / 48 | +0.3477 / +2.5883 | +0.1285 | 82/112 / 4/112 |
| A_PAIRED/1 | channel | 58 / 37 / 0 / 17 | +0.0798 / +0.6137 | +0.0234 | 54/112 / 7/112 |
| B_REAL_ONLY/0 | erm | 110 / 2 / 32 / 0 | +0.2612 / +1.9797 | -0.1677 | 4/144 / 68/144 |
| B_REAL_ONLY/0 | channel | 112 / 13 / 19 / 0 | +0.0954 / +0.4620 | -0.0172 | 13/144 / 36/144 |
| A01 | erm | 12 / 0 / 0 / 7 | +0.3210 / +2.8320 | +0.1542 | 7/19 / 0/19 |
| A01 | channel | 19 / 0 / 0 / 0 | +0.0072 / +0.5251 | +0.0044 | 0/19 / 0/19 |
| A02 | erm | 0 / 9 / 0 / 10 | +0.4121 / +2.6879 | +0.1742 | 18/19 / 0/19 |
| A02 | channel | 1 / 9 / 0 / 9 | +0.1851 / +0.7971 | +0.0584 | 18/19 / 0/19 |
| A03 | erm | 5 / 7 / 0 / 7 | +0.2974 / +2.4592 | +0.1015 | 13/19 / 0/19 |
| A03 | channel | 18 / 0 / 0 / 1 | +0.0571 / +0.6472 | +0.0280 | 1/19 / 0/19 |
| A04 | erm | 7 / 1 / 0 / 11 | +0.4477 / +3.3741 | +0.1289 | 10/19 / 1/19 |
| A04 | channel | 18 / 0 / 0 / 1 | +0.0328 / +0.7898 | -0.0222 | 1/19 / 1/19 |
| A05 | erm | 1 / 14 / 0 / 3 | +0.1847 / +1.5845 | +0.0773 | 17/18 / 0/18 |
| A05 | channel | 1 / 16 / 0 / 1 | +0.0659 / +0.3189 | +0.0199 | 17/18 / 0/18 |
| A06 | erm | 1 / 6 / 1 / 10 | +0.4185 / +2.5369 | +0.1324 | 17/18 / 3/18 |
| A06 | channel | 1 / 12 / 0 / 5 | +0.1328 / +0.5874 | +0.0532 | 17/18 / 6/18 |

### Waveform summaries

| Condition | Median RMS | Median peak | Median bandwidth95 (Hz) | Maximum absolute lag (samples) | Nonzero unit-range proxy cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.03026 | 0.2500 | 1641 | 0 | 0 |
| noise10 | 0.03174 | 0.2569 | 5338 | 0 | 0 |
| opus24 | 0.02977 | 0.2456 | 1464 | 0 | 0 |
| A_then_B | 0.02967 | 0.2421 | 2909 | 0 | 0 |
| B_then_A | 0.03126 | 0.2528 | 5315 | 0 | 0 |

Median forward difference-waveform SNR is 10.70 dB and median reverse insertion SNR is 9.85 dB. The ratios describe different operations. The lag estimate is restricted to ±441 samples.


Both exports restored with the selected controller weight hashes. Smoke replay of clean and both orders across three sources and two models gave 18 anchor scores, max drift 0, zero decision mismatches. The full diagnostic emitted 1472 new predictions in 646.6 s. Registered FFmpeg binary and P0 fingerprints were checked. No model training, official evaluation or calibration was performed.

Both orders reuse a fixed-absolute noise realization seeded by the sample ID and the registered view version, with 10 dB clean reference SNR and 24 kb/s conventional libopus. Loading mixes to mono, resamples to 22.05 kHz by polyphase filtering and takes up to three seconds. Common clean headroom limits the peak to 0.25. Codec paths convert to PCM16 for encoding and decode to float32. FFmpeg container timestamps and prefix trim or zero-pad align the length, without explicit lag correction or post-channel gain normalization. P0 handles model-length normalization separately. The code reconstructed and compared both chain waveforms bitwise with the registered implementation for every source. It retained source-level RMS, peak, length, energy, bandwidth95, unit-range clipping proxies and a limited cross-correlation lag estimate.

Post-codec effective residual is computed against codec(clean), while reverse-order measured insertion SNR is relative to that same coded clean waveform. Codec(noise + clean) − codec(clean) includes nonlinear coding effects. It is a difference waveform, so this ratio cannot be interpreted as isolated additive noise power. Standalone Noise5 and Opus64 are reported as diagnostics outside the matched factorial contrast. Excess composition subtracts the Noise10-only and Opus24-only scores from the mean composed score and adds clean; a nonzero value can arise from waveform processing or detector nonlinearity.

| Model | Raw order-score AUC for class | After nuisance residualization | FAKE−REAL mean raw order-score gap and 95% interval | Mean residual gap |
| --- | ---: | ---: | --- | ---: |
| erm | 0.515 | 0.538 | +0.0271, [-0.06, +0.11] (10000/10000 valid) | +0.0304 |
| channel | 0.396 | 0.432 | -0.0222, [-0.04, -0.01] (10000/10000 valid) | -0.0128 |

The nuisance screen predicts contrasts with ridge regression, holding out one development speaker at a time. Regressors are waveform differences in RMS, peak, bandwidth95, estimated lag and clipping proxy, plus clean duration and the residual SNR contrast. The two detectors were already selected on this development cohort. The residual AUC is descriptive, and unmeasured confounders remain. Residualization does not identify a forensic target.

A05 erm: 17/18 sources were clean-correct and missed on at least one stress view. 1/48 stressed misses were within 0.1 of threshold, median miss score 0.048.

A05 channel: 17/18 sources were clean-correct and missed on at least one stress view. 4/50 stressed misses were within 0.1 of threshold, median miss score 0.252.

A06 erm: 15/18 sources were clean-correct and missed on at least one stress view. 2/41 stressed misses were within 0.1 of threshold, median miss score 0.117.

A06 channel: 11/18 sources were clean-correct and missed on at least one stress view. 14/45 stressed misses were within 0.1 of threshold, median miss score 0.326.

## Findings and limits

A05 has zero clean and Opus24-only misses in either model, but 17/18 under Noise10 alone and 14–16/18 under noise→codec. A06 already has clean errors, increases to 17/18 under Noise10 in both models, and shows a rise from 7/18 to 12/18 under noise→codec with channel augmentation. Several stressed errors remain far from the threshold. These profiles are consistent with noisy-condition separation problems, with a distinct ordered-composition pattern for A06. They cannot establish cue destruction. The two attack intervals include zero for the channel-minus-ERM stress difference.

For order-score class ranking, residualization moved channel from AUC 0.396 to 0.432 in the declared FAKE direction, or 0.568 with the sign reversed. ERM remained close to 0.5. The change of sign across models, broad positive order-score shift across classes and modest residual separation do not support a stable class-discriminative target. Unmeasured nuisance and detector nonlinearities remain possible. [C2 decision](FINAL_C2_AND_TRAINING_DECISION.md)

The selected channel checkpoint improves aggregate stressed ranking and reduces stressed FAKE misses overall, while A05/A06 remain difficult and clean B false alarms increase. Reducing mean order-score differences does not guarantee better guard outcomes. This rejects that proposed shortcut criterion. Error direction under a fixed threshold can arise from similar score shifts in REAL and FAKE. Detector disagreement and waveform nuisance measures do not identify forensic evidence degradation. Guard failure applies to the selected pair. The study has one prospective training seed and development speakers only.

[Class and attack figures](../../outputs/jaes_v2/phase10/final_diagnostic_v1/class_and_attack.png) · [Order and speaker figures](../../outputs/jaes_v2/phase10/final_diagnostic_v1/order_and_speaker.png) · [C2 decision and training authorization](FINAL_C2_AND_TRAINING_DECISION.md)
