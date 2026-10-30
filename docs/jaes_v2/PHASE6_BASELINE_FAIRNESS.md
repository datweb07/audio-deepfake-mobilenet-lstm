# Phase6 baseline fairness

All methods share the selected sources, detached frozen-backbone cache, model class, pinned ImageNet bytes, initialization seed5806, batch16, six epochs/60 updates, Adam .001 linear decay, fixed.5 and clean-dev-BCE checkpoint selection. Training source permutation uses a different RNG instance from view choice. Main initial weights are hashed before updates.

B1 samples each source from discrete gain{1,.5} with equal probability, at every source/update. Both are finite precomputed waveforms with shared peak headroom.25; half is seen and quarter is a held-out gain value. No random clipping or unknown reference gain is required. P0/P1 retain their declared feature normalization.

B2 samples clean/half/noise10/Opus24. B4_CONTROL and B4 sample exactly the same bank and second clean view; same pair log hash and initial-weight hash are verified. Their only objective difference is.1 probability MSE. Dropout masks remain stochastic independently in the two forwards, so clean/clean pairs also regularize dropout disagreement. B4/control equality isolates this particular standard consistency objective; it does not make two-view and one-view training compute-equivalent.

RB samples clean plus two fixed algorithm5 realizations from unchanged MIT author functions. It uses22050Hz/maxF8000 and does not reproduce the original online16k exposure. Global numpy reference RNG is locked/restored; source/license hashes are in vendor/rawboost/PROVENANCE.json. No label changes, length drift or hidden post-channel normalization are introduced.

Cache arrays are materialized once, avoiding repeated Phase5 NPZ decompression per sample. This is engineering optimization, not a learned research contribution. Actual exposure/cost:

| Run | Updates | Source examples | Supervised views | Head training forwards | Backward passes | Run seconds | Peak MiB |
|---|---|---|---|---|---|---|---|
| p6_p0_b0_v1 | 60 | 960 | 960 | 60 | 60 | 8.197240600000441 | 541.51953125 |
| p6_p0_b1_v1 | 60 | 960 | 960 | 60 | 60 | 8.88612060000014 | 543.13671875 |
| p6_p0_b2_v1 | 60 | 960 | 960 | 60 | 60 | 8.430270599999858 | 540.015625 |
| p6_p0_b4_control_v1 | 60 | 960 | 1920 | 120 | 60 | 9.685003900000083 | 567.93359375 |
| p6_p0_b4_v1 | 60 | 960 | 1920 | 120 | 60 | 9.505638500000714 | 565.0234375 |
| p6_p0_rb_v1 | 60 | 960 | 960 | 60 | 60 | 8.464471300001605 | 539.72265625 |
| p6_p1_b0_v1 | 60 | 960 | 960 | 60 | 60 | 8.43149209999865 | 540.26171875 |
| p6_p1_b1_v1 | 60 | 960 | 960 | 60 | 60 | 8.46226470000147 | 542.390625 |
| p6_p1_b2_v1 | 60 | 960 | 960 | 60 | 60 | 8.447896800000308 | 537.09765625 |

Primary pair audit:

```json
{
  "comparison": [
    "p6_p0_b4_v1",
    "p6_p0_b4_control_v1"
  ],
  "equal_view_fairness": "PASS",
  "contrast": {
    "estimate": -0.07142857142857142,
    "parents": 7,
    "ci95": [
      -0.13095238095238096,
      -0.023809523809523808
    ],
    "ci_status": "DESCRIPTIVE_PERCENTILE"
  },
  "clean_guardrail_passes": true,
  "clean_far_change": 0.0,
  "clean_miss_change": 0.0,
  "screening_improvement": true,
  "scope": "Single-seed exploratory screen; no significance or confirmed novelty"
}
```
