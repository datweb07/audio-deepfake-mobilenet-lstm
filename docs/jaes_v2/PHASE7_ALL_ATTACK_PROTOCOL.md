# PHASE7 ALL ATTACK PROTOCOL

## All-Attack Experimental Design

Selection and protocol are locked independently of scores.

```json
{
  "calibration_accessed": false,
  "counts": {
    "development_A": {
      "attacks": {
        "A01": 19,
        "A02": 19,
        "A03": 19,
        "A04": 19,
        "A05": 18,
        "A06": 18
      },
      "fake": 112,
      "n": 224,
      "real": 112,
      "speakers": 7
    },
    "development_B": {
      "attacks": {},
      "fake": 0,
      "n": 144,
      "real": 144,
      "speakers": 9
    },
    "train": {
      "attacks": {
        "A01": 53,
        "A02": 54,
        "A03": 54,
        "A04": 54,
        "A05": 53,
        "A06": 52
      },
      "fake": 320,
      "n": 640,
      "real": 320,
      "speakers": 20
    }
  },
  "decoded_duplicates": 0,
  "evaluation_accessed": false,
  "foundation_hashes": {
    "development": "dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a",
    "train": "7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72"
  },
  "known_partition_audit": "PASS",
  "manifest_sha256": "029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319",
  "original_source_lineage": "ORIGINAL_SOURCE_LINEAGE_UNKNOWN",
  "per_cell": 16,
  "scope": "EXPLORATORY_DEVELOPMENT; A01-A06 SEEN IN TRAIN; B speakers may have prior Phase5 exposure",
  "seed": 7107,
  "selection_id": "medium_all_attack_v1",
  "selection_rows_sha256": "52b65bc3cac4cd5b415a353dd400969058cb2774223c80c0f26476d05794d9bb",
  "status": "LOCKED_ALL_ATTACK_SELECTION"
}
```

| split | speaker_id | REAL | FAKE | A01 | A02 | A03 | A04 | A05 | A06 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | LA_0069 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| development | LA_0070 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| development | LA_0071 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| development | LA_0072 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| development | LA_0075 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| development | LA_0076 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| development | LA_0077 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| development | LA_0099 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0100 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0102 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0103 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0104 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0105 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0106 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0107 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0108 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | LA_0079 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0080 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0081 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0082 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0083 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0084 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0085 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0086 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0087 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0088 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0089 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0090 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0091 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0092 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0093 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0094 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0095 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0096 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0097 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0098 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |

StratumA defines a matched-support controlled estimand. StratumB tests false-alarm sensitivity on9REAL-only speakers. No fake records are imputed. Per-attack FAKE-only ROC-AUC/EER/PR-AUC are undefined and stored null; the full112REAL reference is not silently replicated per attack.

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


