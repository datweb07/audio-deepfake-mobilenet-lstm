# PHASE7 FINE TUNING RESULTS

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

## Computational and Storage Resources

```json
{
  "CIM_probe": "Read denied by host; physical/commit memory measured with Windows GlobalMemoryStatusEx",
  "concurrency": "Existing OS file mutex; one heavyTensorFlow job; PID ownership audit in job_owners",
  "cpu_count": 12,
  "date": "2026-10-31 Asia/Novosibirsk",
  "disk": {
    "C:/": {
      "free_bytes": 1744568320,
      "total_bytes": 325344104448
    },
    "D:/": {
      "free_bytes": 26325270528,
      "total_bytes": 157286395904
    }
  },
  "gpu_inventory": {
    "returncode": 0,
    "stderr": "",
    "stdout": "NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 595.71"
  },
  "gpu_tensorflow": "No GPU visible in executed current TensorFlowCPU cache/training",
  "memory": {
    "available_commit": 4586565632,
    "available_physical": 6092017664,
    "available_virtual": 140731997741056,
    "extended_virtual": 0,
    "length": 64,
    "load_percent": 62,
    "total_commit": 27264434176,
    "total_physical": 16452980736,
    "total_virtual": 140737488224256
  },
  "new_dependencies_installed": false,
  "pip_cache_actual": {
    "returncode": 0,
    "stderr": "",
    "stdout": "c:\\users\\thanh dat\\appdata\\local\\packages\\sandbox.{3424b7d1-ae6a-402f-8ef2-d7d24d2d9471}\\ac\\pip\\cache"
  },
  "production_environments_modified": false,
  "python": "D:\\audio-deepfake-mobilenet-lstm\\.venv\\Scripts\\python.exe",
  "temp": {
    "TEMP": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp",
    "TMP": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp",
    "checkpoints": "D:\\audio-deepfake-mobilenet-lstm\\outputs\\jaes_v2\\phase7\\training",
    "matplotlib_cache": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp\\matplotlib",
    "pip_cache_default": "C:\\Users\\THANH DAT\\AppData\\Local\\pip\\Cache",
    "python_temp": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp",
    "tensorflow_temporary": "Process TEMP/TMP onD; no system environment change"
  },
  "tensorflow": "2.15.0",
  "wsl_inventory": {
    "returncode": 4294967295,
    "stderr": "",
    "stdout": "Access is denied. \n\nError code: Wsl/E_ACCESSDENIED"
  }
}
```

Cache actual:

```json
{
  "arrays": {
    "development_P0": [
      2576,
      6,
      576
    ],
    "train_P0": [
      3840,
      6,
      576
    ]
  },
  "benchmark_only": false,
  "contract_hash": "7411d09b8d9efedf5fd5b69f0129652335f1c5b034ca7a25a33bddaf83b28d12",
  "elapsed_seconds": 1776.5667477000002,
  "gpu_available": false,
  "peak_ram_mib": 959.32421875,
  "resumed_shards": 0,
  "sources": 1008,
  "status": "SEALED_PHASE7_CACHE",
  "views": 6416
}
```

Training resources:

| experiment_id | elapsed_seconds | update_seconds_total | peak_ram_mib | optimizer_updates | source_examples_seen | supervised_views_seen | model_training_forwards | backward_calls | checkpoint_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 12.506894 | 6.477249 | 626.136719 | 40 | 640 | 640 | 640 | 640 | 5916718 |
| p7_f1_s6107_v1 | 1472.984895 | 48.669298 | 883.273438 | 40 | 640 | 640 | 640 | 640 | 5916699 |
| p7_h0_s6107_v1 | 17.454715 | 7.225866 | 631.628906 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h0_s6207_v1 | 18.022321 | 7.495766 | 633.785156 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h0_s6307_v1 | 23.208389 | 10.856521 | 633.832031 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h1_s6107_v1 | 20.377338 | 7.607627 | 635.992188 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h1_s6207_v1 | 19.773671 | 7.199822 | 638.687500 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h1_s6307_v1 | 20.185537 | 9.552592 | 633.992188 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h2_s6107_v1 | 22.733548 | 10.222911 | 650.046875 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h2_s6207_v1 | 16.987935 | 8.666293 | 647.390625 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h2_s6307_v1 | 25.867089 | 13.359128 | 651.187500 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h3_s6107_v1 | 19.772342 | 9.057026 | 650.335938 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h3_s6207_v1 | 17.604633 | 9.544557 | 650.816406 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h3_s6307_v1 | 22.158650 | 10.731365 | 647.113281 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h4_s6107_v1 | 17.654819 | 7.361591 | 632.394531 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h4_s6207_v1 | 25.623703 | 9.261843 | 629.824219 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h4_s6307_v1 | 18.442723 | 8.462098 | 635.988281 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |

Full-student inference:

| experiment_id | parameters | checkpoint_bytes | median_seconds | offline_rtf | peak_process_lifetime_mib |
| --- | --- | --- | --- | --- | --- |
| p7_h0_s6107_v1 | 1308401 | 5916718 | {"decode": 0.002616699999634875, "end_to_end": 0.24510800000007293, "model": 0.23322110000026441, "preprocessing": 0.009560999999848718} | 0.106609 | 656.878906 |
| p7_h3_s6107_v1 | 1308401 | 5916718 | {"decode": 0.0023830999998608604, "end_to_end": 0.2480652000003829, "model": 0.23644430000058492, "preprocessing": 0.009863700000096287} | 0.107895 | 679.839844 |
| p7_h4_s6107_v1 | 1308401 | 5916718 | {"decode": 0.0024597999999969034, "end_to_end": 0.2576280000002953, "model": 0.24508510000032402, "preprocessing": 0.010029600000052596} | 0.112055 | 689.761719 |

Cache cost is shared preprocessing, not hidden inside fast head-only training. Timing excludes process startup/model construction and separates full-student inference from embedding-head updates. CPU runtime, physical GPU and actual GPU execution are distinct. Process lifetime peaks include allocator/activation effects but do not isolate activation memory. No system CUDA stack or production environment was changed.
Matched full-student F0/F1 timing:

| experiment_id | parameters | median_seconds | offline_rtf | process_lifetime_peak_mib |
| --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 1308401 | {"decode": 0.002626599999530299, "end_to_end": 0.278780700000425, "model": 0.2663632999992842, "preprocessing": 0.010294499999872642} | 0.121255 | 467.316406 |
| p7_f1_s6107_v1 | 1308401 | {"decode": 0.0025995000005423208, "end_to_end": 0.2883762999999817, "model": 0.27558030000000144, "preprocessing": 0.010318200000256184} | 0.125429 | 487.542969 |

The F1 lifecycle includes complete raw-view CNN evaluation; its optimizer-update seconds are reported separately. Comparing that lifecycle directly with cached-head scoring does not measure training overhead. Future work should profile/compile full inference with parity checks before longer experiments; no unmeasured speedup is claimed.

