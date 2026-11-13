# Training-size sensitivity at fixed epochs

Seed 9207. Each arm uses 21 epochs. The 640 and 1,280 cohorts use 840 and 1,680 updates, respectively.
The contrast measures sensitivity to cohort expansion and its associated compute on the shared development set.
The pointwise 95% intervals use matched speaker resampling within strata. They describe evaluation-cohort uncertainty for one fitted seed block.

| Endpoint | ERM 640 | Channel 640 | ERM 1,280 | Channel 1,280 | Δ640 | Δ1,280 | Δsize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_clean_AUC | 0.9851 | 0.9809 | 0.9900 | 0.9860 | -0.0041 [-0.0108, +0.0010] | -0.0039 [-0.0080, -0.0004] | +0.0002 [-0.0053, +0.0077] |
| A_macro_stress_AUC | 0.5946 | 0.7839 | 0.7062 | 0.8143 | +0.1893 [+0.0932, +0.2684] | +0.1081 [+0.0317, +0.2041] | -0.0812 [-0.1421, -0.0148] |
| A_clean_FA | 4/112 | 4/112 | 1/112 | 3/112 | +0.0000 [-0.0268, +0.0268] | +0.0179 [+0.0000, +0.0357] | +0.0179 [+0.0000, +0.0357] |
| A_clean_miss | 9/112 | 9/112 | 3/112 | 8/112 | +0.0000 [-0.0357, +0.0446] | +0.0446 [+0.0179, +0.0625] | +0.0446 [-0.0000, +0.0893] |
| A_stress_FA | 167/336 | 70/336 | 5/336 | 161/336 | -0.2887 [-0.3720, -0.1905] | +0.4643 [+0.3393, +0.5863] | +0.7530 [+0.6250, +0.8780] |
| A_stress_miss | 143/336 | 134/336 | 248/336 | 64/336 | -0.0268 [-0.1042, +0.0357] | -0.5476 [-0.6220, -0.4732] | -0.5208 [-0.6339, -0.4077] |
| B_clean_FA | 26/144 | 20/144 | 18/144 | 22/144 | -0.0417 [-0.0694, -0.0069] | +0.0278 [-0.0486, +0.1181] | +0.0694 [-0.0069, +0.1528] |
| A05_stress_miss | 38/54 | 47/54 | 50/54 | 36/54 | +0.1667 [+0.0370, +0.2982] | -0.2593 [-0.4444, -0.1053] | -0.4259 [-0.7000, -0.1852] |
| A06_stress_miss | 16/54 | 31/54 | 50/54 | 11/54 | +0.2778 [+0.1176, +0.4375] | -0.7222 [-0.8958, -0.5370] | -1.0000 [-1.1765, -0.8333] |

## Transformation order

Each contrast uses Noise → Opus minus Opus → Noise within source.

| Stratum and class | Sources | Metric | Δ640 | Δ1,280 | Δsize |
| --- | ---: | --- | ---: | ---: | ---: |
| A_PAIRED/0 | 112 | FAKE score | -0.2099 [-0.2505, -0.1770] | +0.0764 [+0.0475, +0.1021] | +0.2863 [+0.2274, +0.3482] |
| A_PAIRED/0 | 112 | Error rate | -0.2857 [-0.4732, -0.1071] | +0.2679 [+0.1696, +0.3839] | +0.5536 [+0.3212, +0.7946] |
| A_PAIRED/1 | 112 | FAKE score | -0.2012 [-0.2438, -0.1648] | -0.1183 [-0.1700, -0.0609] | +0.0829 [+0.0293, +0.1288] |
| A_PAIRED/1 | 112 | Error rate | +0.1250 [+0.0536, +0.1964] | +0.0268 [-0.0446, +0.0893] | -0.0982 [-0.1875, -0.0089] |
| B_REAL_ONLY/0 | 144 | FAKE score | -0.2148 [-0.2410, -0.1853] | +0.0743 [+0.0234, +0.1169] | +0.2891 [+0.2135, +0.3509] |
| B_REAL_ONLY/0 | 144 | Error rate | -0.4306 [-0.5417, -0.2847] | +0.2917 [+0.1667, +0.4167] | +0.7222 [+0.4653, +0.9236] |

## Selected checkpoints and guard decisions

- erm640 selected `u000520_e013_c000`.
- erm1280 selected `u001600_e020_c000`.
- channel640 selected `u000720_e018_c000`.
- channel1280 selected `u001280_e016_c000`.
- Size 640 guard status `NO_FEASIBLE_CHECKPOINT`. Failed constraints are A05_stress_miss, A06_stress_miss.
- Size 1280 guard status `NO_FEASIBLE_CHECKPOINT`. Failed constraints are A_clean_miss, B_clean_FA.

## Execution costs

| Arm | Training seconds | Validation seconds | Checkpoint seconds | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| erm640 | 2488.0 | 2479.1 | 18.2 | 795.4 |
| erm1280 | 4716.2 | 2258.3 | 37.5 | 807.2 |
| channel640 | 4943.2 | 2373.0 | 18.0 | 769.6 |
| channel1280 | 9620.5 | 2274.6 | 38.7 | 763.3 |

The fixed threshold is 0.5. This development comparison reuses the same evaluation population and uses one selected initialization. Its effect includes the additional updates at fixed epochs. It does not estimate a fixed-compute causal effect or a population-average size effect.

[Score replay, exact denominators, interval draw counts and artifact hashes](../../outputs/jaes_v2/phase10/size1280_v1/size_sensitivity.json)
