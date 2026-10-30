# Phase6 partial fine-tuning feasibility

A separate real train-tensor path reuses the production builder and existing tail-unfreeze policy, without editing production. BatchNorm is frozen, a fresh optimizer is created after trainability changes, and loss reaches selected CNN variables. Cached detached embeddings are never called fine-tuning.

First concurrent attempt terminated without completed gradients; its exact root cause is UNKNOWN. The contemporaneous cache raised an allocation error. Failed directories and source snapshot remain. Sequential execution and microbatch1 with two-step gradient accumulation then passed. The CLI now rejects concurrent heavy jobs with a tested file mutex; this does not reserve system RAM against other apps. The system volume is low on free space, a practical constraint on larger jobs.

Actual train-only sanity, not development quality:

```json
{
  "status": "SMOKE_TESTED",
  "scope": "OVERFIT_SANITY_RESOURCE_PREFLIGHT",
  "training_policy": "PARTIAL_BACKBONE_FINETUNE",
  "seed": 5806,
  "sources": [
    "ASVspoof2019_LA:LA_T_1075018",
    "ASVspoof2019_LA:LA_T_1113457"
  ],
  "data_role": "TRAIN_ONLY",
  "steps": 5,
  "batch": 1,
  "gradient_accumulation": 2,
  "effective_batch": 2,
  "preprocessing_seconds": 0.04903580000063812,
  "head_only_forward_seconds": 0.6505297000003338,
  "head_only_backward_seconds": 0.36144469999999274,
  "head_only_gradient_norm": 4.490053653717041,
  "elapsed_seconds": 9.81548440000006,
  "trainable_parameters": 646481,
  "total_parameters": 1308401,
  "selected_backbone_kernels": [
    "expanded_conv_10/squeeze_excite/Conv/kernel:0",
    "expanded_conv_10/squeeze_excite/Conv_1/kernel:0",
    "expanded_conv_10/project/kernel:0",
    "Conv_1/kernel:0"
  ],
  "changed_backbone_kernels": [
    "expanded_conv_10/squeeze_excite/Conv/kernel:0",
    "expanded_conv_10/squeeze_excite/Conv_1/kernel:0",
    "expanded_conv_10/project/kernel:0",
    "Conv_1/kernel:0"
  ],
  "gradients": {
    "expanded_conv_10/squeeze_excite/Conv/kernel:0": 0.2547600269317627,
    "expanded_conv_10/squeeze_excite/Conv/bias:0": 0.020035885274410248,
    "expanded_conv_10/squeeze_excite/Conv_1/kernel:0": 0.23635178804397583,
    "expanded_conv_10/squeeze_excite/Conv_1/bias:0": 0.02518070861697197,
    "expanded_conv_10/project/kernel:0": 0.32761093974113464,
    "Conv_1/kernel:0": 0.45991718769073486,
    "temporal_lstm/lstm_cell/kernel:0": 1.2389395236968994,
    "temporal_lstm/lstm_cell/recurrent_kernel:0": 0.22725650668144226,
    "temporal_lstm/lstm_cell/bias:0": 0.07371043413877487,
    "classifier_dense/kernel:0": 0.3783886134624481,
    "classifier_dense/bias:0": 0.07067957520484924,
    "probability_fake/kernel:0": 0.1936441957950592,
    "probability_fake/bias:0": 0.007385313510894775
  },
  "frozen_variables_unchanged": 200,
  "batchnorm_frozen": 34,
  "save_load_max_residual": 0.0,
  "peak_ram_mib": 837.37890625,
  "gpu_available": false,
  "checkpoint_bytes": 5916699,
  "initial_loss": 0.677266925573349,
  "final_loss": 0.5458562970161438,
  "limitations": "Two train recordings / five updates; no development quality comparison; prefix/middle layers intentionally frozen",
  "output_hashes": {
    "progress.json": "dd00d770c969f32246c0608670def2d88f700ffbf42e1758848880276d3abf57",
    "sanity.keras": "0e36448923fd41248844feb886a2858a3222aa8233c755ba8cdeef4aef2d2ebb",
    "training_history.csv": "fdd71d5e279b6fdb990d118f86db38757d155c1b12d6b707501c7ef460ede7b6"
  }
}
```

The frozen prefix/middle intentionally has no trainable gradient; four selected tail kernels have nonzero gradients and changed bytes. Five effective-batch2 updates over two real train recordings show implementability and decreasing loss, not generalization. Full-data/multiseed fine-tuning is NOT_EXECUTED. Use measured microbatch cost to budget compatible isolated GPU/CPU studies; physical RTX hardware does not validate the installed Windows CUDA runtime. The external `--device gpu` sanity command checks visible TensorFlow GPU and has no disguised CPU fallback.
