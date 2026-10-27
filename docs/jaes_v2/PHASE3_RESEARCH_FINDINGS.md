# Phase 3 executed research findings

Updated 27 October 2026. Scope: **EXPLORATORY PUBLIC DEVELOPMENT**. Novelty **NOT_ESTABLISHED**; confirmation **NOT_LOCKED**.

Actual MLAAD-tiny cohort: 192 audio / 96 REAL / 96 FAKE, four author-annotated English system groups, 24 FAKE each. Source revision and license evidence are in PHASE3_DATASET_DECISION.md and PHASE3_DATA_ACQUISITION.md.

Four known original source-text speakers contribute to FAKE, with balanced six recordings per speaker/system. REAL speaker/original identity is UNKNOWN. Local or annotated parent keys and source-text speakers are not a certified number of independent population units.

Seven conditions share 1,344 immutable waveform files between detectors. They use identical sample IDs, noise instances and codec outputs. Scores and thresholds remain separate per artifact. No retraining, calibration, confirmation data or performance-based sample selection was used.

| Detector | Threshold | Successful / attempted | Failures | Elapsed seconds |
|---|---:|---:|---:|---:|
| mobilenetv3_lstm | 0.8200 | 1344/1344 | 0 | 524.15 |
| shufflenetv2_lstm | 0.1200 | 1344/1344 | 0 | 648.89 |

Timing includes artifact loading and report generation in two concurrent CPU processes; this is not a latency/fair-speed benchmark. Complete paired scores are persisted; failed predictions are never removed.

| Detector | Condition | Error | REAL false alarm | FAKE miss | ROC-AUC | Diagnostic EER |
|---|---|---:|---:|---:|---:|---:|
| mobilenetv3_lstm | clean_native | 0.2552 | 0.3854 | 0.1250 | 0.8266 | 0.1979 |
| mobilenetv3_lstm | clean | 0.2812 | 0.5000 | 0.0625 | 0.8196 | 0.2396 |
| mobilenetv3_lstm | pcm16 | 0.2865 | 0.5104 | 0.0625 | 0.8190 | 0.2396 |
| mobilenetv3_lstm | codec | 0.2135 | 0.3229 | 0.1042 | 0.8724 | 0.2188 |
| mobilenetv3_lstm | fixed_noise | 0.4635 | 0.0833 | 0.8438 | 0.6786 | 0.3542 |
| mobilenetv3_lstm | fixed_A_then_B | 0.4896 | 0.0208 | 0.9583 | 0.6780 | 0.3542 |
| mobilenetv3_lstm | fixed_B_then_A | 0.4792 | 0.0938 | 0.8646 | 0.6832 | 0.3646 |
| shufflenetv2_lstm | clean_native | 0.2240 | 0.2083 | 0.2396 | 0.8703 | 0.2292 |
| shufflenetv2_lstm | clean | 0.2448 | 0.2500 | 0.2396 | 0.8468 | 0.2396 |
| shufflenetv2_lstm | pcm16 | 0.2396 | 0.2500 | 0.2292 | 0.8468 | 0.2396 |
| shufflenetv2_lstm | codec | 0.2552 | 0.2500 | 0.2604 | 0.8418 | 0.2604 |
| shufflenetv2_lstm | fixed_noise | 0.3854 | 0.2500 | 0.5208 | 0.6570 | 0.3646 |
| shufflenetv2_lstm | fixed_A_then_B | 0.3802 | 0.1146 | 0.6458 | 0.6006 | 0.4375 |
| shufflenetv2_lstm | fixed_B_then_A | 0.3698 | 0.2292 | 0.5104 | 0.6447 | 0.3854 |

EER is a ranking diagnostic calculated after execution, not a replacement decision threshold. Binary F1/AUC/EER are missing for FAKE-only system groups; miss-rate denominators are 24 for those groups and 96 for pooled FAKE.

## Interpretation supported by these executions

Both artifacts lose detection quality under noisy chains. Their inherited thresholds also show large class-asymmetric shifts: more FAKE misses can coincide with fewer REAL false alarms. On this balanced cohort both artifacts show only a 2/192 increase in overall error for noise→codec versus codec→noise, while MobileNet has nine additional FAKE misses and ShuffleNet thirteen. The aggregate difference conceals cancellation between the two classes. This is an exploratory artifact observation, not proof of a new general mechanism.

Headroom preparation changes the native-baseline results; therefore it cannot be treated as neutral transport. The separate native/prepared/PCM conditions must stay in the report. Ranking and operating-point behavior differ across treatments, so the result does not justify fitting a new threshold on this cohort and claiming robustness.

The two artifacts are not matched training replicas. Their cross-artifact agreement does not establish architectural superiority, global generator ordering or independence from legacy training. Four source-text clusters give limited support for interaction uncertainty. No follow-up condition was selected by these observed scores.

| Detector | FAKE system | Order miss contrast | Descriptive source-text-speaker 95% interval | Clusters |
|---|---|---:|---|---:|
| mobilenetv3_lstm | Cartesia.ai (Sonic-3) | 0.1250 | [0.0, 0.25] | 4 |
| mobilenetv3_lstm | Chatterbox | 0.0417 | [0.0, 0.125] | 4 |
| mobilenetv3_lstm | DeepGram | 0.1250 | [-0.08333333333333333, 0.3333333333333333] | 4 |
| mobilenetv3_lstm | Edge-TTS | 0.0833 | [0.0, 0.25] | 4 |
| shufflenetv2_lstm | Cartesia.ai (Sonic-3) | 0.1250 | [0.0, 0.25] | 4 |
| shufflenetv2_lstm | Chatterbox | 0.1250 | [0.0, 0.25] | 4 |
| shufflenetv2_lstm | DeepGram | 0.2083 | [0.16666666666666666, 0.29166666666666663] | 4 |
| shufflenetv2_lstm | Edge-TTS | 0.0833 | [-0.08333333333333334, 0.25] | 4 |

Order means noise→codec minus codec→noise. Positive means more FAKE misses in the first order. Intervals resample four annotated source-text speakers, not four independent synthesized identities. They are descriptive, non-simultaneous and cannot support significance or population equivalence. Every pairwise group/channel result is included in table4_interactions.csv; no favorable subset is selected.

Generator×channel contrasts are observational FAKE-miss difference-in-differences with common English/original-source support and joint source-text-speaker bootstrap. Checkpoint identity, reference/synthesized voice composition and model-relative familiarity remain unresolved. Numerical identifiability of these annotation-defined cells is not causal identification.

Shortcut results, including the actual prepared model inputs, are in phase3_shortcuts_v1/shortcuts.csv and table5_confounders.csv. Constant sample rate has no predictive content and is explicitly missing. The original LOPO constant-feature artifact is not treated as a result. Strong silence/numeric diagnostic performance means balanced language and rate do not remove dataset fingerprints.

The supported result is external execution plus measured transfer/order/threshold behavior for these two artifacts on this cohort. No new forensic mechanism, generator superiority, detector improvement, SOTA, three international contributions or JAES readiness follows. See NEAREST_PRIOR_ART.md and PHASE3_CONFIRMATION_GATE.md.

Programmatic tables: outputs/jaes_v2/audit/phase3_report_v1/table1_…table6_*.csv. Executed figures: figures/ in each detector run (SVG and 300 DPI PNG). No journal Results section or accepted manuscript was altered.
