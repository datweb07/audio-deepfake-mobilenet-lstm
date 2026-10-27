# Phase 4 executed research findings

27 October 2026. **WITHIN_RECORD_DEVELOPMENT_EVALUATED; SOURCE_MATCHED_EXPERIMENT_BLOCKED.**

The six Phase 3 system pairs share zero selected original source IDs. Full tiny metadata contains 290 same-language multi-system source groups, but the largest English fixed-system-pair overlap is three sources. Author-source/text matches exist at level 3; none was certified level 4, and no REAL mapping was verified. A sparse incomplete-block analysis would require a different estimand/design; these findings do not prove all conceivable matched development designs impossible.

The executed fallback is c1_nuisance_controls_dev_v1: 32 existing English public audio files (16 REAL, 16 FAKE). FAKE has four systems × four original source-text speakers, one seeded recording per cell. Those source recordings are different across systems. REAL original identity remains UNKNOWN. There are 20 connected known/local components, not 20 certified independent speakers. No additional audio was downloaded.

Three versioned interventions (global half-amplitude, RMS anchor with cap, observed-window edge repacking) each precede clean/noise→codec/codec→noise. Their fixed-absolute noise is anchored once to each intervention’s clean reference, with the original sample-derived unit-noise seed. No post-channel normalization occurs. Existing native/prepared/PCM/codec/noise/order scores are reused, not predicted again.

| Artifact | New predictions | Reused scores | New failures | Runtime seconds |
|---|---:|---:|---:|---:|
| mobilenetv3_lstm | 288/288 | 224 | 0 | 118.79 |
| shufflenetv2_lstm | 288/288 | 224 | 0 | 148.61 |

Both artifact jobs reuse the same 288 new waveforms. Two separate warm-up forwards are recorded outside scored counts. Concurrent CPU timing includes loading/reporting and is not an efficiency comparison.

| Artifact | Primary change in FAKE order contrast | Descriptive 95% interval | Source-text clusters |
|---|---:|---|---:|
| mobilenetv3_lstm | 0.0625 | [0.0, 0.1875] | 4 |
| shufflenetv2_lstm | 0.1875 | [-0.125, 0.4375] | 4 |

Primary = [(noise→codec miss − codec→noise miss) after edge repacking] minus the same difference at prepared baseline, averaged equally across four original source-text-speaker clusters. This is a within-record preprocessing sensitivity, not a generator effect. All 18 descriptive contrasts per artifact are retained; no p-values/significance. Edge repacking shifts segment alignment and can remove low-energy phonetic tails, so it is not a certified semantic intervention.

| Artifact | Policy | Condition | REAL false alarm | FAKE miss | ROC-AUC | Decision flips vs same-channel baseline |
|---|---|---|---:|---:|---:|---:|
| mobilenetv3_lstm | phase3_baseline | clean | 0.4375 | 0.0625 | 0.8516 | 0 |
| mobilenetv3_lstm | phase3_baseline | fixed_A_then_B | 0.0000 | 1.0000 | 0.7344 | 0 |
| mobilenetv3_lstm | phase3_baseline | fixed_B_then_A | 0.0625 | 0.9375 | 0.7734 | 0 |
| mobilenetv3_lstm | global_half_v1 | clean | 0.6250 | 0.0625 | 0.7578 | 3 |
| mobilenetv3_lstm | global_half_v1 | fixed_A_then_B | 0.0000 | 1.0000 | 0.6641 | 0 |
| mobilenetv3_lstm | global_half_v1 | fixed_B_then_A | 0.0625 | 0.9375 | 0.7461 | 0 |
| mobilenetv3_lstm | rms_anchor_v1 | clean | 0.4375 | 0.0625 | 0.8242 | 0 |
| mobilenetv3_lstm | rms_anchor_v1 | fixed_A_then_B | 0.0000 | 0.9375 | 0.6055 | 1 |
| mobilenetv3_lstm | rms_anchor_v1 | fixed_B_then_A | 0.0625 | 0.8750 | 0.7695 | 1 |
| mobilenetv3_lstm | edge_repack_v1 | clean | 0.3750 | 0.1250 | 0.8633 | 2 |
| mobilenetv3_lstm | edge_repack_v1 | fixed_A_then_B | 0.0625 | 1.0000 | 0.5078 | 1 |
| mobilenetv3_lstm | edge_repack_v1 | fixed_B_then_A | 0.1875 | 0.8750 | 0.5586 | 3 |
| shufflenetv2_lstm | phase3_baseline | clean | 0.2500 | 0.1250 | 0.8984 | 0 |
| shufflenetv2_lstm | phase3_baseline | fixed_A_then_B | 0.1250 | 0.7500 | 0.6797 | 0 |
| shufflenetv2_lstm | phase3_baseline | fixed_B_then_A | 0.2500 | 0.6250 | 0.7031 | 0 |
| shufflenetv2_lstm | global_half_v1 | clean | 0.3750 | 0.1250 | 0.8594 | 2 |
| shufflenetv2_lstm | global_half_v1 | fixed_A_then_B | 0.2500 | 0.7500 | 0.6914 | 6 |
| shufflenetv2_lstm | global_half_v1 | fixed_B_then_A | 0.2500 | 0.5000 | 0.7188 | 2 |
| shufflenetv2_lstm | rms_anchor_v1 | clean | 0.2500 | 0.1250 | 0.8906 | 0 |
| shufflenetv2_lstm | rms_anchor_v1 | fixed_A_then_B | 0.2500 | 0.7500 | 0.6445 | 6 |
| shufflenetv2_lstm | rms_anchor_v1 | fixed_B_then_A | 0.2500 | 0.4375 | 0.7227 | 3 |
| shufflenetv2_lstm | edge_repack_v1 | clean | 0.3125 | 0.1250 | 0.8281 | 1 |
| shufflenetv2_lstm | edge_repack_v1 | fixed_A_then_B | 0.3750 | 0.8125 | 0.5234 | 7 |
| shufflenetv2_lstm | edge_repack_v1 | fixed_B_then_A | 0.3125 | 0.5000 | 0.5859 | 9 |

Severe noisy-chain misses remain after these controls. They do not establish that the detector uses silence as a shortcut or that a generator-specific mechanism survives content matching. Thresholds remain .82/.12; AUC/EER are diagnostics, not fitted operating points.

## Measured feature-level explanation

On all 32 exact prepared inputs, half-amplitude waveform parity is exact. Fixed production Mel floor gives maximum tensor change 19.19068909 on the 0–255 input scale and mean absolute change 3.44061016. In a model-free diagnostic only, scaling the floor from 1e-10 to 2.5e-11 reduces the maximum tensor residual to 0.0000610352. This isolates the absolute-floor contribution to feature non-invariance up to floating precision; it does not measure detector performance under an altered preprocessor. Production preprocessing and accepted results are unchanged.

The pure-amplitude clean sensitivity caused three MobileNet and two ShuffleNet decisions to flip on these 32 records. This cautions against attributing all channel shifts to lost synthesis artifacts. The model-free floor counterfactual is an engineering diagnosis, not a trained method or world-first contribution.

## Decision

C1 should currently be a **supporting validation / engineering contribution**, not one of three established primary international novelties. The source-matched scientific claim remains untested; confirmation is NOT_LOCKED. C2 needs a useful, licensed and compatible teacher or justified teacher-free alternative, controlled training data and a mechanism distinct from augmentation/KD/consistency. C3 remains deferred.

Programmatic tables: outputs/jaes_v2/audit/phase4_report_v1/. Source audit: outputs/jaes_v2/audit/phase4_source_matching/. Figures: source audit figures/ and both nuisance artifact figures/. Precision assumptions and next-stage requirements are documented separately.
