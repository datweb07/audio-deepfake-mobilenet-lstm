# Phase7 GPU, calibration and external readiness

Validated research training uses the existing Windows TensorFlow2.15 CPU environment. A detected RTX3050Laptop4GiB does not establish CUDA visibility. No CUDA/WSL environment or package was installed, and no production environment was replaced. `resource_inventory_v1.json` records actual driver/distro/resource checks. Treat unverified compatibility as NOT_VERIFIED.

The portable package consists of the existing architecture, new versioned configs/manifests/checkpoints and `scripts/jaes_v2/phase7_gpu_probe.py`. On a separately provisioned Linux/WSL GPU runtime, pin Python/TensorFlow dependencies, copy the selected source/hash inventory and ImageNet weight artifact, verify the same preprocessing/sample-rate contract, then run:

```powershell
python -X utf8 -m scripts.jaes_v2.phase7_gpu_probe --output outputs/jaes_v2/phase7/gpu_portability_probe_v1
```

This parser is implemented and syntactically checked; GPU runtime execution is NOT_EXECUTED. It rejects missing GPU and performs one real synthetic gradient/save-load test. It is not a full GPU training launch. The main Phase7 lifecycle remains validated CPU-only; a future GPU port must preserve initialization, source schedule, BN policy and numerical parity and measure microbatch1 memory before longer runs. Budget/disk/version/driver verification is required on that target, with no blanket request to install a stack here.

External confirmation: NOT_LOCKED. MLAAD-tiny has already influenced development. RADAR audio and In-the-Wild have not been acquired. ASV evaluation remains unopened. Before any new external audio/labels: source/version/license/expanded-size, overlap, available original lineage, selected checkpoints, frozen hyperparameters, primary tests and operating-point/calibration policy must be recorded. Existing importer/protocol adapters can be reused only after those gates. `phase7_readiness.py` is fail-closed; constructing a proposal alone does not authorize label use.

Reserved calibration remains7341records, unused. `CalibrationContract` validates a separately locked fitting role/checkpoint/hash and supplies the temperature-scaling contract; `temperature_transform` only applies a supplied positive temperature and never fits on reserved labels. Ranking measures ordering, temperature fitting concerns probability quality, threshold selection chooses an operating rule, and risk control needs assumptions/independent evidence. None substitutes for the others. NLL/Brier/ECE are descriptive reliability diagnostics at the existing fixed threshold.

C3: DEFERRED. Candidate first stage is the selected research MobileNet student; potential second stage must have verified data overlap, correct score semantics and actual quality before selection. The retained RawNet2/AASIST artifacts are heterogeneous references, not automatically qualified teachers or second stages. Current measured offline CPU latency does not establish cascade risk–coverage, edge latency or arbitrary-shift guarantees. No cascade executed.
