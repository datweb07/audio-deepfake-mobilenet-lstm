# Phase8 actual experiment results

| experiment_id | status | scope | steps |
| --- | --- | --- | --- |
| p8_tiny_dro_head_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_dro_head_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_resume_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_resume_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v1 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6207_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6207_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6307_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6307_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_resume_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_resume_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_guarded_infeasible_head_s6107_v3 | NO_FEASIBLE_CHECKPOINT | TINY_LIFECYCLE | 2 |
| p8_tiny_weighted_head_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_weighted_head_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |

All completed quality rows use12train and14development sources (12primary A,2REAL-only B), A01–A06 seen attacks. These are lifecycle/objective pilots, not a generalization study. Original implementation, exact-resume pairs and optimized-validation reruns are not independent scientific replications. M1/M2 optimize standard losses; no M3 run exists. Full medium/foundation training, independent calibration/confirmation, GPU and cascade are NOT_EXECUTED.

Guarded negative integration: NO_FEASIBLE_CHECKPOINT, epoch1 of a declared5epoch maximum, patience1; no export fallback. Historical ordinary consistency and RawBoost trade-offs remain unchanged.

CPU exact-resume evidence:

```json
{
  "head": {
    "status": "PASS",
    "checks": {
      "final_weights": true,
      "optimizer": true,
      "sample_order_views_losses_lr": true,
      "step": true,
      "counter": true,
      "history": true,
      "export_scores": true
    },
    "numerical_tolerance": 0,
    "scope": "CPU TF2.15 same environment; fresh processes",
    "final_weights_hash": "5b72bf0e6ff9251662ebe0dce9944004267dd3bb7fb3c8693c97bb76d08760bb",
    "optimizer_hash": "a95699f9c07e107d731f9b09934637854839e197a8000ef57a48191f2a8d7827",
    "steps": 4
  },
  "partial": {
    "status": "PASS",
    "checks": {
      "final_weights": true,
      "optimizer": true,
      "sample_order_views_losses_lr": true,
      "step": true,
      "counter": true,
      "history": true,
      "export_scores": true
    },
    "numerical_tolerance": 0,
    "scope": "CPU TF2.15 same environment; exact optimizer boundaries; fresh processes; warmup/partial transition included",
    "final_weights_hash": "294eb9a9177a8a3c9637dc6c2167b8c69bc43d3dc1ed4540f197eae0c5f6a520",
    "optimizer_hash": "2057f8b1483bdb01ca1c073a599c6ca7d906af0c04d38e8b46ae4ca7725b3954",
    "steps": 4
  }
}
```

Selected-model classwise values and measured costs are in the model-free per_run.csv/classwise_conditions.json; each run also retains all seen-attack metrics with one-class AUC/REAL-FA nulls. No posthoc threshold tuning or claim of superiority from tiny scores.
