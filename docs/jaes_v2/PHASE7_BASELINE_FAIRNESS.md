# PHASE7 BASELINE FAIRNESS

## Training and Fine-Tuning Architecture

Source bytes → deterministic official-speaker/class selection → shared seeded waveform views → P0/P1 preprocessing → MobileNetV3Small → LSTM128 → Dense64/dropout → sigmoid FAKE score. Source labels are REAL0/FAKE1. P0 calls production preprocessing; P1 uses per-segment relative power flooring with explicit silent-segment behavior. Both retain the temporal image dimensions. Head-only training materializes detached frozen-backbone embeddings once in RAM, reusing the same source/condition cells across objectives. BCE supervises each sampled view; equal-two-view control averages two BCE terms; B4 adds0.1 mean squared probability discrepancy. ImageNet weight hash, seed, LR decay, Adam updates, view logs, clean-development-BCE checkpoint selection and reload evaluation are recorded. Partial fine-tuning uses raw/preprocessed train tensors, calls the backbone inside the tape, reuses the existing tail-unfreeze policy, freezes BN statistics, creates a fresh optimizer after trainability changes, and audits actual variable gradients/updates and checkpoint parity. Frozen embeddings cannot support CNN updates. Microbatching bounds memory; its tiny sanity loss is not development performance.

Phase7 head-only uses pinned ImageNet embeddings and a freshly seeded shared architecture. F0/F1 instead share1epoch,40updates,Adam0.0001,microbatch1/accum16, identical training source order and initialization. F1 reuses the existing tail-unfreeze policy, keeps all BN frozen, creates the optimizer after trainability changes, backpropagates through actual CNN operations and verifies frozen bytes and changed kernels. Versioned run configs reject unknown keys/unsafe paths and check their hashes against the locked protocol. Checkpoints publish atomically and are reloaded for all evaluation. Last-update optimizer arrays are retained as evidence; exact continuation is RESUME_NOT_SUPPORTED. Train/view/development and resource budgets are enforced without reusing protected conference weights.

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

## RawBoost and Consistency Controls

H2/H3 use the same initialization SHA and identical training-view log per seed, including independently stochastic dropout in paired forward calls. The sole objective difference is0.1MSE. Finite RawBoost algorithm5 supplies2fixed realizations at22050Hz/maxF8000, with original author functions/license unchanged. Finite realizations are shared across seeds; training seeds change head/order/view selection.

```json
{
  "ISD_parameters": {
    "P": 10,
    "g_sd": 2
  },
  "algorithm": 5,
  "amplitude": "Reference internal processing; reject peak>1.00001; no hidden post-channel normalization",
  "convolution_parameters": {
    "N_f": 5,
    "maxBW": 1000,
    "maxBiasLinNonLin": 20,
    "maxCoeff": 100,
    "maxF": 8000,
    "maxG": 0,
    "minBW": 100,
    "minBiasLinNonLin": 5,
    "minCoeff": 10,
    "minF": 20,
    "minG": 0,
    "nBands": 5
  },
  "decision": "Additional extraction alone exceeds locked head-only budget; do not substitute finite-view results",
  "head_run_budget_seconds": 240,
  "implemented": true,
  "max_filter_hz": 8000,
  "mean_seconds": 0.3652749857142875,
  "projected_six_epoch_extra_extraction_seconds": 935.103963428576,
  "quality_experiment": "NOT_EXECUTED",
  "resampling": "NONE; original16k setting adapted directly to22050Hz frequency-calibrated filters",
  "sampling_rate": 22050,
  "status": "AWAITING_COMPUTE",
  "tested": true
}
```

The stateless online adapter changes per-example realization across epochs, isolates global numpy state and preserves labels/length. Frequency parameters remainHz at22,050Hz without resampling. It is not exact16k original reproduction. Online quality is NOT_EXECUTED when extra extraction violates the locked budget; Figure09 is deliberately not generated.

