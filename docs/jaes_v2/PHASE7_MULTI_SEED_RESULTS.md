# PHASE7 MULTI SEED RESULTS

## Phase7 Findings

| experiment_id | seed | training_policy | epochs | clean_REAL_FA | clean_FAKE_miss | stress_REAL_FA | stress_FAKE_miss | B_clean_REAL_FA | B_stress_REAL_FA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 6107 | HEAD_ONLY | 1 | 0.232143 | 0.446429 | 0.258929 | 0.675595 | 0.111111 | 0.127315 |
| p7_f1_s6107_v1 | 6107 | PARTIAL_BACKBONE_FINETUNE | 1 | 0.258929 | 0.267857 | 0.235119 | 0.717262 | 0.145833 | 0.094907 |
| p7_h0_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.017857 | 0.107143 | 0.636905 | 0.267857 | 0.118056 | 0.537037 |
| p7_h0_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.026786 | 0.071429 | 0.389881 | 0.613095 | 0.104167 | 0.298611 |
| p7_h0_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.071429 | 0.071429 | 0.779762 | 0.321429 | 0.250000 | 0.724537 |
| p7_h1_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.008929 | 0.098214 | 0.392857 | 0.226190 | 0.145833 | 0.534722 |
| p7_h1_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.000000 | 0.133929 | 0.000000 | 0.627976 | 0.034722 | 0.009259 |
| p7_h1_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.044643 | 0.089286 | 0.550595 | 0.184524 | 0.263889 | 0.567130 |
| p7_h2_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.035714 | 0.080357 | 0.113095 | 0.410714 | 0.180556 | 0.182870 |
| p7_h2_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.008929 | 0.071429 | 0.175595 | 0.398810 | 0.118056 | 0.152778 |
| p7_h2_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.044643 | 0.062500 | 0.261905 | 0.354167 | 0.201389 | 0.275463 |
| p7_h3_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.000000 | 0.116071 | 0.366071 | 0.315476 | 0.083333 | 0.416667 |
| p7_h3_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.000000 | 0.098214 | 0.080357 | 0.461310 | 0.097222 | 0.097222 |
| p7_h3_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.026786 | 0.089286 | 0.199405 | 0.375000 | 0.180556 | 0.245370 |
| p7_h4_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.044643 | 0.107143 | 0.919643 | 0.077381 | 0.083333 | 0.930556 |
| p7_h4_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.017857 | 0.080357 | 0.824405 | 0.354167 | 0.125000 | 0.837963 |
| p7_h4_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.017857 | 0.098214 | 0.895833 | 0.125000 | 0.138889 | 0.849537 |

Primary H3–H2 mean signed miss contrast = -0.003968; independent-seed SD = 0.081741. Individual effects: {"6107": -0.09523809523809525, "6207": 0.06249999999999999, "6307": 0.020833333333333322}. Only1/3blocks favor consistency; 0/3pass both registered clean guardrails. The Phase6 favorable single-seed conclusion does not establish a stable accepted improvement here. Standard consistency is not a new method; these are descriptive development results.

## Standard Baseline Comparisons

| method | seeds | clean_REAL_FA_mean | clean_FAKE_miss_mean | clean_AUC_mean | stress_REAL_FA_mean | stress_FAKE_miss_mean | B_clean_REAL_FA_mean | B_stress_REAL_FA_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | 3 | 0.038690 | 0.083333 | 0.976004 | 0.602183 | 0.400794 | 0.157407 | 0.520062 |
| H1 | 3 | 0.017857 | 0.107143 | 0.976190 | 0.314484 | 0.346230 | 0.148148 | 0.370370 |
| H2 | 3 | 0.029762 | 0.071429 | 0.977280 | 0.183532 | 0.387897 | 0.166667 | 0.203704 |
| H3 | 3 | 0.008929 | 0.101190 | 0.972630 | 0.215278 | 0.383929 | 0.120370 | 0.253086 |
| H4 | 3 | 0.026786 | 0.095238 | 0.971354 | 0.879960 | 0.185516 | 0.115741 | 0.872685 |

H0ERM; H1channel; H2two-view BCE; H3same pairs plus0.1probabilityMSE; H4finite RawBoost. Each source has one sampled first view per epoch. H2/H3 also supervise clean and therefore have7680supervised views versus3840for one-view baselines. They are equal-update comparisons, not equal-compute comparisons.

Representation pilot:

| experiment_id | epochs | training_policy | clean_REAL_FA | clean_FAKE_miss | stress_REAL_FA | stress_FAKE_miss | clean_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 1 | HEAD_ONLY | 0.232143 | 0.446429 | 0.258929 | 0.675595 | 0.750797 |
| p7_f1_s6107_v1 | 1 | PARTIAL_BACKBONE_FINETUNE | 0.258929 | 0.267857 | 0.235119 | 0.717262 | 0.810427 |

F1 minus matchedF0: {"clean_AUC": 0.05963010204081631, "clean_FAKE_miss": -0.1785714285714286, "clean_REAL_FA": 0.0267857142857143, "stress_FAKE_miss": 0.04166666666666674, "stress_REAL_FA": -0.023809523809523836}. Clean ranking improves in this short pilot while stressed FAKE miss worsens; there is no demonstrated robustness improvement or convergence claim. F2channel/F3equal-view/F4consistency representation ablations and three-seed fine-tuning are NOT_EXECUTED under the boundedF0/F1priority.

## Multi-Seed Statistical Evidence

```json
{
  "clean_guardrails_by_seed": {
    "6107": {
      "FAKE_miss_delta": 0.03571428571428571,
      "REAL_FA_delta": -0.03571428571428571,
      "passes": false
    },
    "6207": {
      "FAKE_miss_delta": 0.026785714285714288,
      "REAL_FA_delta": -0.008928571428571428,
      "passes": false
    },
    "6307": {
      "FAKE_miss_delta": 0.026785714285714288,
      "REAL_FA_delta": -0.01785714285714286,
      "passes": false
    }
  },
  "first": "H3",
  "group_support": "LIMITED_CLUSTER_PRECISION",
  "mean": -0.003968253968253978,
  "p_value": null,
  "paired_seed_effects": {
    "6107": -0.09523809523809525,
    "6207": 0.06249999999999999,
    "6307": 0.020833333333333322
  },
  "primary": true,
  "second": "H2",
  "seed_bootstrap_ci95": [
    -0.09523809523809525,
    0.06249999999999999
  ],
  "seed_count": 3,
  "seed_std": 0.08174145656791475,
  "speaker_bootstrap_ci95": [
    -0.01884920634920636,
    0.01190476190476189
  ],
  "speaker_count": 7,
  "uncertainty": "DESCRIPTIVE_ONLY; speaker interval conditional on trained seeds; seed interval conditional on same7speakers; only3seed blocks and7clusters"
}
```

Seed std is computed across independently trained blocks, not bootstrap replicates. Speaker bootstrap resamples7known speaker effects while retaining all paired conditions and trained seeds. Seed bootstrap separately resamples3paired blocks on the same speakers. Both intervals are descriptive with LIMITED_CLUSTER_PRECISION; no strong population significance or automatic hierarchical-bootstrap guarantee. Secondary comparisons are exploratory, no selective p-values.

## Classwise Robustness Analysis

| method | attack | unique_fake_per_seed_condition | speakers | clean_miss_mean | stress_miss_mean |
| --- | --- | --- | --- | --- | --- |
| H0 | A01 | 19 | 7 | 0.000000 | 0.286550 |
| H0 | A02 | 19 | 7 | 0.000000 | 0.374269 |
| H0 | A03 | 19 | 7 | 0.035088 | 0.614035 |
| H0 | A04 | 19 | 7 | 0.070175 | 0.333333 |
| H0 | A05 | 18 | 7 | 0.000000 | 0.561728 |
| H0 | A06 | 18 | 7 | 0.407407 | 0.234568 |
| H1 | A01 | 19 | 7 | 0.000000 | 0.070175 |
| H1 | A02 | 19 | 7 | 0.000000 | 0.561404 |
| H1 | A03 | 19 | 7 | 0.000000 | 0.175439 |
| H1 | A04 | 19 | 7 | 0.070175 | 0.070175 |
| H1 | A05 | 18 | 7 | 0.000000 | 0.728395 |
| H1 | A06 | 18 | 7 | 0.592593 | 0.500000 |
| H2 | A01 | 19 | 7 | 0.000000 | 0.011696 |
| H2 | A02 | 19 | 7 | 0.000000 | 0.842105 |
| H2 | A03 | 19 | 7 | 0.000000 | 0.087719 |
| H2 | A04 | 19 | 7 | 0.052632 | 0.005848 |
| H2 | A05 | 18 | 7 | 0.000000 | 0.839506 |
| H2 | A06 | 18 | 7 | 0.388889 | 0.574074 |
| H3 | A01 | 19 | 7 | 0.000000 | 0.017544 |
| H3 | A02 | 19 | 7 | 0.017544 | 0.766082 |
| H3 | A03 | 19 | 7 | 0.035088 | 0.140351 |
| H3 | A04 | 19 | 7 | 0.087719 | 0.017544 |
| H3 | A05 | 18 | 7 | 0.000000 | 0.833333 |
| H3 | A06 | 18 | 7 | 0.481481 | 0.561728 |
| H4 | A01 | 19 | 7 | 0.000000 | 0.210526 |
| H4 | A02 | 19 | 7 | 0.017544 | 0.105263 |
| H4 | A03 | 19 | 7 | 0.000000 | 0.345029 |
| H4 | A04 | 19 | 7 | 0.052632 | 0.228070 |
| H4 | A05 | 18 | 7 | 0.000000 | 0.160494 |
| H4 | A06 | 18 | 7 | 0.518519 | 0.055556 |

Each attack group has18–19FAKE recordings on7speakers, with only2–3per speaker. The table averages condition/seed miss rates within attack and is descriptive. Full `attack_metrics.csv` includes score quantiles and explicit null two-class metrics. `speaker_metrics.csv` retains every REAL-only speaker with missing FAKE denominator as null; small groups are not removed. Different cohort size, attack mixture, seed and update count prevent attributing cross-phase changes solely to the addition ofA05/A06. Within-Phase7 matched comparisons are the supported estimands.

## Clean–Robustness Trade-Offs

| first | second | mean | seed_std | clean_guardrails_by_seed |
| --- | --- | --- | --- | --- |
| H3 | H2 | -0.003968 | 0.081741 | {"6107": {"FAKE_miss_delta": 0.03571428571428571, "REAL_FA_delta": -0.03571428571428571, "passes": false}, "6207": {"FAKE_miss_delta": 0.026785714285714288, "REAL_FA_delta": -0.008928571428571428, "passes": false}, "6307": {"FAKE_miss_delta": 0.026785714285714288, "REAL_FA_delta": -0.01785714285714286, "passes": false}} |
| H3 | H1 | 0.037698 | 0.184075 | {"6107": {"FAKE_miss_delta": 0.017857142857142863, "REAL_FA_delta": -0.008928571428571428, "passes": true}, "6207": {"FAKE_miss_delta": -0.03571428571428571, "REAL_FA_delta": 0.0, "passes": true}, "6307": {"FAKE_miss_delta": 0.0, "REAL_FA_delta": -0.01785714285714286, "passes": true}} |
| H3 | H4 | 0.198413 | 0.079266 | {"6107": {"FAKE_miss_delta": 0.008928571428571438, "REAL_FA_delta": -0.044642857142857144, "passes": true}, "6207": {"FAKE_miss_delta": 0.01785714285714285, "REAL_FA_delta": -0.017857142857142856, "passes": true}, "6307": {"FAKE_miss_delta": -0.008928571428571425, "REAL_FA_delta": 0.008928571428571428, "passes": true}} |
| H1 | H0 | -0.054563 | 0.076710 | {"6107": {"FAKE_miss_delta": -0.008928571428571425, "REAL_FA_delta": -0.008928571428571428, "passes": true}, "6207": {"FAKE_miss_delta": 0.0625, "REAL_FA_delta": -0.026785714285714284, "passes": false}, "6307": {"FAKE_miss_delta": 0.017857142857142863, "REAL_FA_delta": -0.02678571428571428, "passes": true}} |
| H4 | H0 | -0.215278 | 0.037920 | {"6107": {"FAKE_miss_delta": 0.0, "REAL_FA_delta": 0.026785714285714288, "passes": false}, "6207": {"FAKE_miss_delta": 0.008928571428571438, "REAL_FA_delta": -0.008928571428571428, "passes": true}, "6307": {"FAKE_miss_delta": 0.026785714285714288, "REAL_FA_delta": -0.05357142857142857, "passes": false}} |

The predeclared2pp clean delta limit applies within paired comparisons at fixed0.5. It is a sample-level screen, not a certified real-world bound. Lower FAKE miss accompanied by higher REAL false alarm is a trade-off, never universal superiority. Figures05/12 show both error classes. No new threshold was fitted after inspecting stress results.

Secondary descriptive Pareto screen relative to H0, explicitly posthoc and distinct from the registered H3/H2 test:

| method | clean_screen_vs_H0 | stress_REAL_FA | stress_FAKE_miss | lifecycle_seconds_mean | nondominated_stress_errors_within_screen |
| --- | --- | --- | --- | --- | --- |
| H0 | True | 0.602183 | 0.400794 | 19.561808 | False |
| H1 | False | 0.314484 | 0.346230 | 20.112182 | False |
| H2 | True | 0.183532 | 0.387897 | 21.862857 | True |
| H3 | False | 0.215278 | 0.383929 | 19.845208 | False |
| H4 | False | 0.879960 | 0.185516 | 20.573749 | False |


