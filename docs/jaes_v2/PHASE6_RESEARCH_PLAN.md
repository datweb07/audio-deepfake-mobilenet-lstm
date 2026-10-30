# Phase6 locked exploratory research plan —30 October2026

Nine staged head-only baselines use a new within-speaker-balanced cohort, one seed, six epochs, sixty updates, batch16 and the same ImageNet initialization. P0: B0, B1, B2, B4_CONTROL, B4, RB; P1: B0, B1, B2. P1 consistency/RawBoost Cartesian expansion is not automatic. The protocol was written before main training; primary comparison B4 versus equal-view control, mean speaker-uniform FAKE miss over noise5 and both order chains, with at most one additional clean error in each class. This small-cohort guardrail is a screen, not an operational risk guarantee.

Reserved calibration and official evaluation remain unused. The cohort changes from Phase5, so better absolute values cannot establish improvement over Phase5. Particularly, A05/A06 are absent; clean saturation on A01–A04 is not all-attack success. Medium/large plans are unexecuted projections. No new loss is authorized while the method distinction and proxy validity gates fail.

Locked protocol:

```json
{
  "status": "LOCKED_BEFORE_MAIN_TRAINING",
  "date": "2026-10-30",
  "confirmation_enabled": false,
  "selection_sha256": "45ff123a734e11935e9e7f7a1ee9259f950cc82f7441bf06b6ca96cd56d87965",
  "primary_comparison": [
    "p6_p0_b4_v1",
    "p6_p0_b4_control_v1"
  ],
  "primary_metric": "Equal-official-speaker mean FAKE miss over noise5,A_then_B,B_then_A",
  "clean_risk_margin": 0.03571428571428571,
  "margin_rationale": "Exploratory screen: at most one additional clean error in each 28-record class; not a deployment-certified tolerance",
  "operating_threshold": 0.5,
  "calibration_fitted": false,
  "uncertainty": "Paired known-speaker bootstrap; single seed; no discovery/significance claims",
  "maximum_training_seconds": 180,
  "epochs": 6,
  "optimizer_steps": 60,
  "seed": 5806,
  "finite_rawboost_realizations": 2,
  "bank": "All methods access same cache; B4 vs control same two supervised views and updates; one-view methods have half view exposure",
  "candidate_new_loss_enabled": false,
  "medium_large_training_enabled": false
}
```
