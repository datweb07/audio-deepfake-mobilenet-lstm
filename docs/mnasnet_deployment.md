# MnasNet-A1-LSTM deployment

Source bundle: `mnasnet_lstm_deployment/` (preserved without modification).
The lifecycle is COMPLETE, early stopping occurred at epoch 39, and the exported
weights are from best epoch 27 (validation loss 0.5457665324211121).
This is a completed scratch/end-to-end lifecycle, not an unfinished checkpoint.
The original validation-calibrated threshold 0.90 is retained; no test tuning.

The training recipe includes label smoothing and L2 regularization, so its
validation loss should not be directly compared with another model's unregularized
binary cross-entropy. Use common evaluation metrics instead.

## Conversion

Keras 3.13.2 source -> Keras 2.15 native archive. The source graph, all named leaf
weights, BN state, activations, LSTM/head and score semantics are retained.
Parameters: 3,369,255. Input `(B,6,224,224,3)`, embeddings `(B,6,1280)`,
sigmoid output `(B,1)` = P(FAKE), REAL=0 / FAKE=1.

Six parity inputs (four validation recordings, zeros, seeded random tensor):
maximum score difference 2.0265579223632812e-6; maximum embedding difference
0.00128173828125 (elementwise relative/absolute tolerance test passed).
Save/reload score difference: 0. Report and golden arrays reside in
`outputs/mnasnet_conversion/`. Parity verifies conversion on this local backend;
it does not independently reproduce the original Kaggle hardware/preprocessing.

Offline reproduction from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/mnasnet_deployment.py prepare
.\.venv\Scripts\python.exe scripts/mnasnet_deployment.py export-source
.\.venv\Scripts\python.exe scripts/mnasnet_deployment.py convert
.\.venv\Scripts\python.exe scripts/mnasnet_deployment.py publish
```

Only `export-source` uses the previously installed isolated Keras 3.13.2 package
directory `outputs/efficientnet_conversion/keras3` (read-only reuse). It never
changes the app environment. All other operations use production Keras 2.15.
`publish` refuses to overwrite a deployment, verifies selection/threshold/manifest
provenance and test audio checksums, and copies only the converted MnasNet bundle.

## Serving

Runtime bundle: `models/mnasnet_lstm/{model.keras,threshold.json,metadata.json}`.
Only `src/lava/models/tensorflow/mnasnet_lstm.py` is changed: bounded inference
defaults to one recording at a time, `training=False`, float32. It can be
overridden via `LAVA_MNASNET_INFERENCE_BATCH_SIZE` on suitably provisioned hardware.
Training batch, optimizer, architecture and all other detectors are unchanged.

```powershell
python evaluate.py --model mnasnet_lstm
python predict.py --model mnasnet_lstm --audio "data\REAL\11241.wav"
python predict.py --model mnasnet_lstm --audio "data\FAKE\1.wav"
python -m streamlit run app.py
```

The existing Streamlit selector discovers the bundle automatically. Deploy the
three bundle files plus the MnasNet adapter to the GitHub branch used by Streamlit
Cloud, then reboot/select MnasNet-A1-1.0-LSTM. No app/requirements/other-model edit
is needed. Do not deploy conversion environments or source checkpoints. Cloud
execution still requires verification after the user pushes the changes.

## Independent test evaluation (2026-09-04)

Command: `python evaluate.py --model mnasnet_lstm` (no subset limit).
Canonical manifest unchanged:
`8b55591d58d3658b8cafe0e77b6ebdedbaa67be2e339730a7276fe9b10958df9`.
All 2,737 test recordings were evaluated (1,575 REAL, 1,162 FAKE), with the
original validation threshold 0.90. Test data did not select weights/threshold.

| Metric | Result (CLI precision) |
|---|---:|
| Accuracy | 0.9697 |
| Precision (FAKE) | 0.9599 |
| Recall (FAKE) | 0.9690 |
| F1 (FAKE) | 0.9645 |
| Macro F1 | 0.9690 |
| ROC-AUC (raw scores) | 0.9886 |
| EER | 0.0310 |

Confusion matrix: TN=1,528, FP=47, FN=36, TP=1,126.
Evidence: `outputs/mnasnet_conversion/test_evaluation_retry.txt`.
The first attempt encountered a load-time MemoryError; a separate retry completed
successfully without code, dataset, threshold or architecture changes. The failed
log is retained as `test_evaluation.txt`. Avoid concurrent model-heavy processes
on memory-constrained machines.

These are internal checksum-group-disjoint test results, not speaker-disjoint or
cross-dataset generalization evidence. They do not guarantee microphone/live-audio
performance. This detector remains in the scratch experimental stratum.

## Regression and inference checks

`python -m unittest discover -s tests -q`: 62 tests, 56 passed, 6 skipped,
zero failures (119.769 seconds). Evidence:
`outputs/mnasnet_conversion/regression_tests.txt`. Includes real MnasNet model
loading, score parity, threshold provenance, Streamlit AppTest selector and
uploaded-audio prediction, plus existing regression tests. AppTest is not a live
Streamlit Cloud deployment test.

CLI predictions completed successfully with the retained threshold:

| File | Folder label | Prediction | P(FAKE) |
|---|---|---|---:|
| `data/REAL/11241.wav` | REAL | REAL | 0.2436 |
| `data/FAKE/1.wav` | FAKE | FAKE | 0.9553 |

Evidence: `outputs/mnasnet_conversion/predict_real.txt` and `predict_fake.txt`.
Serialized production model size: 14,082,109 bytes. `pip check` reports no broken
requirements. Neither other-model code nor shared app/training/preprocessing code
was edited for this integration.
