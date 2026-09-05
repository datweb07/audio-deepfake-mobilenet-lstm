# ShuffleNetV2-1.0x-LSTM deployment

The immutable source bundle is `ok/`. It contains the scratch/end-to-end best
checkpoint selected at epoch 16; training stopped at epoch 28. The recorded best
validation loss is 0.08210305869579315. Its validation-calibrated threshold 0.12
is retained. Test data are not used for checkpoint or threshold selection.

The source archive was saved by Keras 3.13.2 and cannot be loaded directly by the
production TensorFlow/Keras 2.15 runtime. `scripts/shufflenet_deployment.py`
performs an offline, ShuffleNet-only conversion. It loads the source in the
existing isolated Keras 3 runtime, exports named weights and reference outputs,
rebuilds the unchanged production ShuffleNet architecture under Keras 2.15, and
requires embedding and score parity before publishing.

```powershell
.\.venv\Scripts\python.exe scripts/shufflenet_deployment.py prepare
.\.venv\Scripts\python.exe scripts/shufflenet_deployment.py export-source
.\.venv\Scripts\python.exe scripts/shufflenet_deployment.py convert
.\.venv\Scripts\python.exe scripts/shufflenet_deployment.py publish
```

The production contract is:

```text
models/shufflenetv2_lstm/
  model.keras
  threshold.json
  metadata.json
```

The existing registry and Streamlit selector discover this bundle without an
`app.py` change. The conversion neither retrains the model nor changes its input,
architecture, weights, score semantics, threshold, or canonical manifest.

Conversion evidence (six parity tensors): 1,868,441 parameters and 115 named
weight groups were transferred. Maximum absolute score difference was
`1.7462298274040222e-10`, maximum embedding difference was
`2.1457672119140625e-06`, and Keras 2.15 save/reload score difference was zero.
The source SHA-256 is
`2e1d7853886772928e3c73046d65ad012a17bf2681b2c1ca71cf2375dd582aa4`.

Validation commands:

```powershell
python -m unittest tests.test_shufflenet_deployment tests.test_shufflenet_shapes tests.test_shufflenet_training_step -v
python predict.py --model shufflenetv2_lstm --audio "data\REAL\11241.wav"
python predict.py --model shufflenetv2_lstm --audio "data\FAKE\1.wav"
python evaluate.py --model shufflenetv2_lstm
python -m streamlit run app.py
```

## Canonical test evaluation

`python evaluate.py --model shufflenetv2_lstm` completed on all 2,737 canonical
test recordings with the unchanged validation threshold 0.12:

| Metric | Value |
|---|---:|
| Accuracy | 0.9850 |
| Precision (FAKE) | 0.9795 |
| Recall (FAKE) | 0.9854 |
| F1 (FAKE) | 0.9824 |
| Macro F1 | 0.9847 |
| ROC-AUC | 0.9929 |
| EER | 0.0146 |

Confusion matrix: TN=1,551, FP=24, FN=17, TP=1,145. The raw CLI output is
retained at `outputs/shufflenet_conversion/test_evaluation.txt`. These are
checksum-group-disjoint internal-test results, not unseen-dataset evidence.

CLI smoke predictions also passed: `data/REAL/11241.wav` produced P(FAKE)
0.0004 and REAL; `data/FAKE/1.wav` produced P(FAKE) 0.9966 and FAKE.
