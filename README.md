# LAVA — Deepfake Voice Multi-Detector Foundation

LAVA provides one evaluation contract across TensorFlow lightweight detectors and native PyTorch anti-spoofing references:

- `REAL = 0`, `FAKE = 1`
- every detector returns a finite `P(FAKE)` in `[0, 1]`
- `P(FAKE) >= threshold` means `FAKE`
- every detector has one final model, one validation-calibrated threshold, and one metadata artifact

The repository is a **multi-model software foundation**, not yet a completed six-model scientific benchmark. Only a fully trained detector with a manifest-compatible artifact can be marked `BENCHMARKED`; smoke tests never receive that status.

## Detector registry

| Registry name | Model | Group | Framework | Input | Pretraining policy |
|---|---|---|---|---|---|
| `mobilenetv3_lstm` | MobileNetV3Small-LSTM | lightweight | TensorFlow 2.15 | 6 chronological Mel-RGB segments | verified ImageNet |
| `efficientnet_b0_lstm` | EfficientNet-B0-LSTM | lightweight | TensorFlow 2.15 | same Mel sequence | verified ImageNet |
| `shufflenetv2_lstm` | ShuffleNetV2-1.0x-LSTM | lightweight | TensorFlow 2.15 | same Mel sequence | scratch experimental; ImageNet weights not verified |
| `mnasnet_lstm` | MnasNet-A1-1.0-LSTM | lightweight | TensorFlow 2.15 | same Mel sequence | scratch experimental; ImageNet weights not verified |
| `rawnet2` | RawNet2 | reference | PyTorch, isolated worker | 16 kHz waveform | native scratch protocol |
| `aasist` | AASIST | reference | PyTorch, isolated worker | 16 kHz waveform and native graph front end | native scratch protocol |

The four lightweight models share duration normalization, 6 chronological segments, Mel generation, `224×224×3` images, `LSTM(128)`, `Dense(64, ReLU)`, `Dropout(0.4)`, and a sigmoid `P(FAKE)` head. Their backbone embedding dimensions are not forced to match: MobileNetV3Small 576, ShuffleNetV2 1024, MnasNet-A1 1280, and EfficientNet-B0 1280. RawNet2 and AASIST retain native waveform architectures and are not converted to Mel-CNN-LSTM variants.

RawNet2 and AASIST both pass shape/forward tests at 48,000 and 64,600 samples. The 48,000-sample path is currently the LAVA common-duration experimental stratum; metadata explicitly marks primary duration-comparison eligibility as false until native-vs-common performance/fidelity experiments have been run.

## Environments

TensorFlow/lightweight environment:

```powershell
cd D:\audio-deepfake-mobilenet-lstm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Isolated PyTorch/reference environment:

```powershell
cd D:\audio-deepfake-mobilenet-lstm
python -m venv .venv-torch
.\.venv-torch\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements-torch.txt
```

Do not install PyTorch into `.venv`. The TensorFlow orchestrator invokes `.venv-torch\Scripts\python.exe` as a JSON worker, so benchmark code does not import both frameworks into one process. Set `LAVA_TORCH_PYTHON` only when the torch interpreter is stored elsewhere.

## Canonical dataset manifest

Only `data\REAL` and `data\FAKE` are production inputs. Build and verify the canonical split before training:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.lava.data.manifest build
python -m src.lava.data.manifest check
```

Artifacts:

```text
data/manifests/dataset_manifest.csv
data/manifests/split_manifest.csv
data/manifests/duplicate_report.csv
data/manifests/label_conflicts.csv
data/manifests/manifest_metadata.json
```

Cross-label byte-identical files are quarantined from every split. For same-label duplicates, the lexicographically first path is retained and all redundant copies are excluded. Therefore duplicate/checksum leakage across train, validation, and test is impossible. Missing speaker/source/generator metadata remains `UNKNOWN`; the only valid current claim is **checksum-group-disjoint split**, not speaker-, generator-, source-, or dataset-disjoint evaluation.

## Train

The default command remains backward-compatible:

```powershell
python train.py
python train.py --model mobilenetv3_lstm
```

Other detectors:

```powershell
python train.py --model efficientnet_b0_lstm
python train.py --model shufflenetv2_lstm
python train.py --model mnasnet_lstm
python train.py --model rawnet2
python train.py --model aasist
```

TensorFlow training is selected from explicit registry policy metadata. Verified ImageNet models (MobileNetV3Small and EfficientNet-B0) use one public lifecycle with internal warm-up and fine-tuning. Each stage creates fresh, stage-local `EarlyStopping` and `ReduceLROnPlateau` callbacks plus an independent checkpoint under `outputs/checkpoints/<detector>/warmup_best.keras` or `finetune_best.keras`. Temporary fine-tuning degradation is allowed: there is no global-patience cutoff against the warm-up score. Only after both stages finish does the trainer compare their best `val_loss`, load the global winner, calibrate its threshold on validation, and publish one production model.

Scratch models (ShuffleNetV2 and MnasNet-A1) instead train the complete backbone, BatchNormalization layers, LSTM, and classifier end-to-end from epoch 1 with Adam at `3e-4`. They use one continuous `fit` lifecycle and `outputs/checkpoints/<detector>/best.keras`; a random backbone is never frozen or given partial-tail learning as its first opportunity. The startup policy summary prints initialization, trainable backbone parameters, BatchNorm status, and learning rate. Recovery metadata is written to `outputs/checkpoints/<detector>/lifecycle_state.json`; interruption preserves the validation-selected checkpoint and never publishes partial weights. The independent test split is never used for checkpointing, early stopping, scheduling, diagnostics, or threshold calibration.

Run a non-production lifecycle test first:

```powershell
python train.py --model efficientnet_b0_lstm --smoke-test
python train.py --model shufflenetv2_lstm --smoke-test
python train.py --model mnasnet_lstm --smoke-test
python train.py --model rawnet2 --smoke-test
python train.py --model aasist --smoke-test
```

`--smoke-test` performs tiny build/train/save/load verification and never overwrites production artifacts.

## Artifact contract

The existing baseline paths are preserved for compatibility:

```text
models/lava_mobilenetv3_lstm.keras
models/best_threshold.txt
models/model_metadata.json
```

Every new detector uses:

```text
models/<detector>/model.keras   # TensorFlow
models/<detector>/model.pt      # PyTorch
models/<detector>/threshold.json
models/<detector>/metadata.json
```

Metadata records the manifest hash, score semantics, pretraining stratum, parameters, seed, framework, hardware summary, and final validation threshold. A clean benchmark refuses artifacts trained on a different or unknown manifest hash.

## Evaluate and predict

```powershell
python evaluate.py
python evaluate.py --model efficientnet_b0_lstm
python evaluate.py --model rawnet2
```

Evaluation always reads the canonical independent test split and reports Accuracy, Precision, Recall, F1, Macro F1, ROC-AUC from raw `P(FAKE)`, EER, confusion matrix, classification report, and threshold.

```powershell
python predict.py --audio "data\REAL\11241.wav"
python predict.py --model efficientnet_b0_lstm --audio "path\to\audio.wav"
python predict.py --model rawnet2 --audio "path\to\audio.wav"
```

Prediction output includes model, framework, predicted label, confidence, raw `P(FAKE)`, and threshold.

## Benchmark

Clean benchmark, one model or all registered models:

```powershell
python -m benchmark.runner --models mobilenetv3_lstm --suite clean
python -m benchmark.runner --models all --suite clean
```

Efficiency benchmark uses batch size 1, warm-up, repeated timings, and reports mean/median/std/P95 for preprocessing, warm model inference, and end-to-end inference. Cold load/process startup is reported separately and excluded from warm model latency:

```powershell
python -m benchmark.runner --models all --suite efficiency --warmup 10 --runs 50
```

Clean per-sample output is written to `outputs/benchmark/clean/<model>/scores.csv`; summaries are aggregated under `outputs/benchmark/`. A run with `--limit` or fewer than 10 efficiency repetitions is diagnostic and receives `SMOKE_TESTED`, never `BENCHMARKED`.

Noise, compression, simulated replay, and unseen-data modules currently expose explicit `NOT_RUN` skeletons. Pareto analysis requires at least two completed models and complete objective columns; otherwise it fails with an insufficiency message instead of inventing a frontier.

## Streamlit

```powershell
python -m streamlit run app.py
```

The selector shows only detectors with a model, threshold, metadata, and successful load check. Streamlit never trains models. MobileNetV3Small-LSTM remains the default when its artifact is valid.

## Verification

TensorFlow/core suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Native torch suite:

```powershell
.\.venv-torch\Scripts\python.exe -m unittest tests.test_rawnet2_shapes tests.test_aasist_shapes tests.test_torch_training_smoke -v
```

The reference repositories in the workspace are read-only research sources. Production code never imports them at runtime. See `docs/SOURCE_PROVENANCE.md` for the adaptation boundary and licensing notes.
