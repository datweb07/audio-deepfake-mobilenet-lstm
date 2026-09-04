# LAVA-5 interim benchmark (inference only)

Do not run any trainer to reproduce this experiment. Existing production artifacts
and the current canonical test split are inputs, not outputs of this workflow.

**Current execution choice:** the user selected full clean evaluation plus a fixed
100-sample robustness diagnostic first (58 REAL, 42 FAKE, stratified seed42).
Official full-test robustness stays NOT_RUN. Diagnostic curves and exploratory
Pareto must not be presented as full-test results.

After execution, verify the selected scope without rerunning inference:

```powershell
.\.venv\Scripts\python.exe scripts/lava5_acceptance.py --samples 100
.\.venv\Scripts\python.exe -m unittest tests.test_lava5_benchmark tests.test_score_semantics tests.test_manifest_integrity tests.test_pareto
```

The acceptance checker recomputes metrics, verifies paired sample identities and
checks model/threshold/manifest seals. Its PASS means the chosen full-clean plus
diagnostic scope passed, not that full-test robustness or FULL LAVA is complete.

After full clean finishes, prepare the paired subset and run the three suites:

One-command version (also measures official efficiency and regenerates both reports):

```powershell
.\.venv\Scripts\python.exe -m benchmark.lava5 diagnostic-execute --diagnostic-samples 100
```

Or run each diagnostic robustness stage separately:

```powershell
.\.venv\Scripts\python.exe -m benchmark.lava5 diagnostic-prepare --diagnostic-samples 100
.\.venv\Scripts\python.exe -m benchmark.lava5 stress --suite noise --output outputs/lava_5/diagnostic_100
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite noise --output outputs/lava_5/diagnostic_100
.\.venv\Scripts\python.exe -m benchmark.lava5 stress --suite compression --output outputs/lava_5/diagnostic_100
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite compression --output outputs/lava_5/diagnostic_100
.\.venv\Scripts\python.exe -m benchmark.lava5 stress --suite replay --output outputs/lava_5/diagnostic_100
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite replay --output outputs/lava_5/diagnostic_100
```

The subset clean baseline is derived by sample ID from verified full-clean scores,
not from full-test aggregate F1. `diagnostic-prepare` can be rerun to collect a model
whose full clean result finished later. The default commands below describe the
future full robustness workflow, not a completed experiment.

## Fixed scope

| Logical ID | Loaded registry ID | Provenance |
|---|---|---|
| mobilenetv3_lstm | mobilenetv3_lstm | LAVA / ImageNet initialization |
| efficientnet_b0_lstm | efficientnet_b0_lstm | LAVA / ImageNet / warm-up checkpoint |
| mnasnet_lstm | mnasnet_lstm | LAVA / scratch / early-stopped best |
| rawnet2 | rawnet2_pretrained | External PyTorch checkpoint / ONNX inference |
| aasist | aasist_pretrained | External PyTorch checkpoint / ONNX inference |

ShuffleNet is pending and is never selected by this runner. Bare `rawnet2` and
`aasist` IDs in the **general registry** still refer to self-trained models;
the interim runner explicitly maps these report IDs to existing external artifacts.

## Commands (PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe -m benchmark.lava5 prepare
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite load
.\.venv-torch\Scripts\python.exe scripts/lava5_reference_audit.py --model rawnet2
.\.venv-torch\Scripts\python.exe scripts/lava5_reference_audit.py --model aasist
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite clean
.\.venv\Scripts\python.exe -m benchmark.lava5 report
.\.venv\Scripts\python.exe -m benchmark.lava5 stress --suite noise
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite noise
.\.venv\Scripts\python.exe -m benchmark.lava5 stress --suite compression
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite compression
.\.venv\Scripts\python.exe -m benchmark.lava5 stress --suite replay
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite replay
.\.venv\Scripts\python.exe -m benchmark.lava5 run --suite efficiency
.\.venv\Scripts\python.exe -m benchmark.lava5 report
```

On Linux use the corresponding environment's `python` command. FFmpeg must be on
PATH for codec generation. The original reference folders are consulted only by
the native PyTorch provenance/parity audit; ONNX benchmark inference is independent
of those folders. Do not install torch into the TensorFlow environment.

`prepare` verifies manifests and every test audio SHA256; it does not build or
resplit anything. It fingerprints model, threshold, metadata, and MobileNet's
companion weights. It refuses changed protocols in an existing result directory;
choose a fresh `--output` directory for a new experiment. Completed score files are
checksum-verified on reuse. A valid partial score prefix can resume after interruption;
a torn row fails closed for manual inspection. No test threshold search occurs.

`--limit 10` on `run` is explicitly DIAGNOSTIC and writes under `diagnostic/`, never
into official clean/robustness results. Default inference uses ALL canonical test
samples. Workers are sequential CPU-only processes, one thread, float32, batch1.
No full-Mel RAM cache, no parallel resident models, no dataset/model mutation.

## Methodological boundaries

- Fixed thresholds: lightweight validation-derived; external references retain
  default 0.5. A threshold is not a probability calibration.
- External native architecture is preserved in the ONNX graph, but existing LAVA
  waveform adapters zero-pad short clips and use `resample_poly`; original reference
  loaders repeat-pad and use librosa. Document this deviation, not paper parity.
- Native reference duration is 4.0375 s; lightweight duration is 3 s. RTF uses each
  duration; compare direct latency too. Training provenance and budgets differ.
- Noise is seeded AWGN, not recorded background environments. Replay is a documented
  synthetic channel, not measured RIR or physical replay. Codec round-trips include
  PCM16 quantization. Shared stress manifests fingerprint every generated waveform.
- Generated stress WAVs can require several GiB; capacity is checked before each
  suite. They are not duplicated per detector and never added to the source dataset.
- Per-suite degradation is mean F1_clean minus F1_condition. Overall degradation is
  the unweighted mean across nine individual stress conditions (4 noise,4 codec,1
  replay), not a weighted combined ranking. Negative degradation is retained.
- Bootstrap: 1000 shared stratified test resamples, seed42; percentile CIs do not
  quantify training-seed uncertainty. McNemar p-values use Holm correction.
- Missing data is empty/NOT_RUN, not zero. Pareto requires all five measured models
  with complete objectives. Radar is normalized visualization only.
- TensorFlow parameters include BN state; native PyTorch parameter counts exclude
  buffers. Raw native buffers are separately recorded in the audit JSON.
- `load_seconds` excludes interpreter/framework import but includes artifact load.
  Warm timings exclude graph tracing/startup; RSS measures the whole process, not
  isolated model allocations. Do not run other model jobs during efficiency timing.

The paper update describes **five-model interim** scope. The generated report and
`report/acceptance.json` are authoritative for what actually completed. Unknown
speaker/source/generator identity prevents stronger split or unseen claims.

## Streamlit

The optional sidebar benchmark card is read-only and shown only if the local
`outputs/lava_5` protocol and master result exist and match the active artifact's
hashes. It never alters prediction, preprocessing, microphone input or threshold.
`outputs/` is gitignored. To show measured cards in deployment, explicitly distribute
only `protocol/protocol.json` and `lava_5_results.csv` at that location alongside the
matching artifacts; do not upload stress audio or private test samples accidentally.
