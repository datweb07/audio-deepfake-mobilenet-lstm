# Resource preflight and tiers

Measured environment: Python 3.11.9; TensorFlow 2.15.0; NumPy 1.24.3; SciPy 1.11.4; soundfile 0.12.1; scikit-learn 1.3.2; matplotlib 3.8.2; ONNX Runtime 1.17.3; FFmpeg 8.0.1 full build. All were already installed; no dependency versions were changed.

This execution host reports 12 logical CPUs, an NVIDIA RTX 3050 Laptop GPU with 4,096 MiB VRAM and approximately 40.3 GB free on D: before the pilot. These are **current host measurements**, not the conference paper's Ryzen desktop specifications. C1 uses CPU-only TensorFlow, one intra-op and one inter-op thread. GPU presence does not imply a validated TensorFlow CUDA runtime; it was not used.

The isolated `.venv-torch` was also probed: PyTorch `2.2.2+cpu`, `torch.cuda.is_available() == False`, no CUDA runtime. Using the physical GPU for C2 therefore requires a separate compatibility/environment decision before any memory dry run; the current torch environment cannot use it.

Tier 1 completed: protected-file hashes, source integrity, transform tests including actual FFmpeg, small inference, paired risk analysis and diagnostic figures. The first 60-score pilot took approximately 37.7 s including preflight-excluded run work, transforms, model loading, API parity and plotting; consult run_metadata.json for measured stage times. It is not an efficiency benchmark and does not replace the conference timings.

Final pilot v3 took 41.23 s of run work, including 10.47 s transform generation and 3.84 s model loading. It differs from earlier executions in deterministic WAV-container serialization, not cohort/perturbation selection. Timing variability is not a model performance claim.

Tier 2 pending: teacher evaluation, student training, ablations and multiple seeds. The 4 GiB device requires an actual teacher forward/backward dry run and memory profiling before choosing batch size, freezing, caching or checkpointing. No GPU-hour estimate is asserted without that test. A generic large-scale corpus download is inappropriate with current disk headroom.

Tier 3 pending: physical replay, real edge runtime, streaming state/latency and instrumented energy. None has been measured. There is no rental, cloud charge, external upload or full-training job in this execution.

Next resource decision: choose a license-compatible metadata-rich corpus that fits disk, validate a teacher and run a bounded dry run. Do not retrain all six conference artifacts merely to begin V2.

## Phase 2 measured workload

The 64-record / 14-condition CPU development run completed in 362.12 s (about 6 minutes), with 896 successful predictions and no exclusions. Total run artifacts occupied 214,810,029 bytes (about 205 MiB). Its preflight reserved one additional GiB beyond a conservative uncompressed-audio estimate; roughly 39 GB remained available. Full waveform reproduction was model-free and did not retrain anything. Diagnostic per-row timing is not a warm repeated latency benchmark.

No public audio or archive was downloaded. The small Edinburgh license request failed HTTP 403 after an approved network retry. No multi-GB download, GPU workload, dependency upgrade or paid service was started. Public-corpus extraction/cache size remains unverified and must be budgeted before acquisition.
