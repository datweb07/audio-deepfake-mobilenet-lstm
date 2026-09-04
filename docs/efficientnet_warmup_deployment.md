# EfficientNet-B0-LSTM warm-up checkpoint deployment

This deployment deliberately uses `efficientnet_b0_warmup_checkpoint/warmup_best.keras`.
It is not a completed fine-tuning/global-best run. The supplied lifecycle records
best warm-up epoch 47, validation loss 0.1698165237903595, and unfinished fine-tuning.
The root `best_threshold.txt` and `model_metadata.json` belong to MobileNet and
must not be reused for EfficientNet.

## Verified conversion and validation

- Parameters: 4,779,300; input `(B,6,224,224,3)`; embeddings `(B,6,1280)`;
  output `(B,1)` sigmoid P(FAKE).
- Maximum Keras 3 -> Keras 2.15 score difference: 2.682209014892578e-7.
- Maximum embedding difference: 1.4781951904296875e-5.
- Save/reload score difference: 0 on the six parity inputs.
- Validation calibration: 2,733 samples, FAKE-class F1 0.9583694709453601,
  threshold 0.90. The threshold search follows `src/metrics.py` unchanged.
- Validation ROC-AUC: 0.9884533178420326; confusion matrix
  `[[1532,41],[55,1105]]` (rows true REAL/FAKE, columns predicted REAL/FAKE).
- Canonical manifest hash:
  `8b55591d58d3658b8cafe0e77b6ebdedbaa67be2e339730a7276fe9b10958df9`.

These validation values are not independent test performance. Evidence:
`outputs/efficientnet_conversion/conversion_report.json`, `calibration_report.json`,
`validation_scores.csv` and the production metadata. No test-based tuning is used.

## Independent test evaluation (2026-09-04)

Executed `python evaluate.py --model efficientnet_b0_lstm` without a subset limit,
with the frozen validation threshold 0.90, on 2,737 samples (REAL 1,575; FAKE 1,162).

| Metric | Value (CLI precision) |
|---|---:|
| Accuracy | 0.9635 |
| Precision (FAKE) | 0.9724 |
| Recall (FAKE) | 0.9406 |
| F1 (FAKE) | 0.9563 |
| Macro F1 | 0.9624 |
| ROC-AUC | 0.9877 |
| EER | 0.0400 |

Confusion matrix: TN=1544, FP=31, FN=69, TP=1093. Raw CLI log including
classification report: `outputs/efficientnet_conversion/test_evaluation.txt`.
Test F1 is close to validation F1, but this is same-dataset/checksum-disjoint
evidence only; it does not establish robustness or cross-dataset performance.

Regression suite: `python -m unittest discover -s tests -q` ran 59 tests,
OK with 6 environment-dependent skips. This includes the EfficientNet registry,
native archive reload, bounded inference, parity, threshold provenance, and a
Streamlit AppTest upload-to-prediction smoke test. Log:
`outputs/efficientnet_conversion/regression_tests.txt`.

## Conversion and calibration

Run from the repository root. Existing production Python remains TF/Keras 2.15.
Only source export uses isolated Keras 3 packages (never upgrade the app environment).
The tested export uses Keras 3.13.2 on the existing TensorFlow 2.15 backend, with
process-local imports; conversion and serving use the original Keras 2.15.

```powershell
.\.venv\Scripts\python.exe -m pip install --target outputs/efficientnet_conversion/keras3 --no-deps keras==3.13.2 optree==0.17.0 namex==0.1.0 ml-dtypes==0.5.1 typing_extensions==4.15.0
.\.venv\Scripts\python.exe scripts/efficientnet_warmup.py prepare
.\.venv\Scripts\python.exe scripts/efficientnet_warmup.py export-source
.\.venv\Scripts\python.exe scripts/efficientnet_warmup.py convert
.\.venv\Scripts\python.exe scripts/efficientnet_warmup.py publish
```

`publish` refuses to overwrite an existing deployment. It checks manifest integrity
and validation/test audio checksums, calibrates only on validation using the
existing FAKE-class F1 threshold search, then publishes:

- `models/efficientnet_b0_lstm/model.keras`
- `models/efficientnet_b0_lstm/threshold.json`
- `models/efficientnet_b0_lstm/metadata.json`

The conversion preserves the source graph, including ImageNet rescaling and
Normalization state. Weights are mapped by named leaf layers with exact shape
and coverage checks. Native Keras 3 embeddings/scores are compared with the
converted Keras 2.15 model on four validation recordings plus zero/random input.
Save/reload parity must pass. Conversion records and per-validation-sample scores
are saved under `outputs/efficientnet_conversion/`.

Important limitation: checkpoint tensors alone cannot verify the original Kaggle
audio preprocessing implementation. A matching training manifest proves split
identity, not preprocessing identity or generalization to new recordings.

## Run locally

```powershell
.\.venv\Scripts\python.exe evaluate.py --model efficientnet_b0_lstm
.\.venv\Scripts\python.exe predict.py --model efficientnet_b0_lstm --audio "path\to\audio.wav"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The existing registry and Streamlit selector discover the complete canonical
bundle without changes to other detectors. No retraining is performed.

EfficientNet alone now defaults to inference batch size 1. The former shared
training batch size 16 expands to 96 images and caused a measured CPU OOM during
calibration. `LAVA_EFFICIENTNET_INFERENCE_BATCH_SIZE` may override this on a
machine with sufficient memory. This does not change any training batch size,
BatchNorm inference mode, weights, input ordering or decision threshold.

## Deploy to Streamlit Cloud

Commit all three files in `models/efficientnet_b0_lstm/` to the deployed branch.
The converted `.keras` contains real model bytes, not a pointer to the source
checkpoint, and does not require the conversion environment or reference repos.
Include `src/lava/models/tensorflow/efficientnet_b0_lstm.py` with its bounded
inference adapter. No requirements/app/other-model change is needed. Restart the app
and select **EfficientNet-B0-LSTM**. Cloud execution must still be checked after
deployment; local verification alone is not proof of a successful cloud rollout.
