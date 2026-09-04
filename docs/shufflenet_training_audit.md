# ShuffleNet training audit (2026-09-04)

## Confirmed defect and minimal fix

The current `train.py` uses `src.dataset.create_tf_dataset` for canonical
training rows. These rows are label-ordered, but the shuffle buffer was only
`min(sample_count, 1024)`. A shuffle over a homogeneous prefix cannot introduce
the other class. The last batches were also homogeneous. This is an optimization
and BatchNorm-distribution hazard, not an architecture or label mapping defect.

Measured through the production dataset path (audio decoding replaced with a
tiny tensor only to test ordering; all canonical training labels retained):

| Configuration | First 20 batches REAL/FAKE | Last 20 batches REAL/FAKE |
|---|---:|---:|
| Before, prefix buffer 1024 | 320 / 0 | 0 / 314 |
| After, full path/label shuffle | 182 / 138 | 183 / 131 |

The final batch is smaller, explaining the 314-sample final window.
The regression failed before the change and passed after it. The fix shuffles
all path/label records **before** audio decoding. It does not cache all Mel
tensors and does not change the manifest, split, class weights or augmentation.
Because the defective loader is shared, its ordering correction applies to all
TensorFlow training consumers. No existing model weights/inference are changed.

## What the screenshots do and do not establish

Validation AUC remains roughly 0.87–0.97, while scores are biased toward FAKE at
the diagnostic threshold 0.5. This is not evidence of permanently dead ranking
or permanently zero gradients. Train/validation loss discrepancy can include
generalization error, score bias and train/inference BatchNorm differences.
BatchNorm is trainable (56/56), but training uses batch statistics whereas
inference uses moving statistics; long single-class runs can bias the latter.

Keras binary cross-entropy can recover the pre-sigmoid logits and compute stable
sigmoid cross-entropy. A saturated wrong prediction does not by itself imply
zero loss gradient. The added test checks a wrong logit of +20 against REAL.
Lack of weight decay does not prove exploding gamma. The local reference's
parameter grouping in fact excludes one-dimensional parameters from decay.

The best shown validation loss is at epoch 2. Current early stopping patience
is 12: if no improvement follows, stopping would occur at epoch 14. The supplied
screenshots end before that event, so they do not prove its actual stopping epoch.

No matching full Kaggle checkpoint/history for the pictured run is available
locally: the local ShuffleNet lifecycle records interruption before epoch 1,
and older checkpoints belong to a different failed freeze-policy run. Thus the
ordering bug is **reproduced in current code**, but its exact contribution to
the remote run cannot be quantified without the remote code/checkpoint and a
controlled repeat. This is not a claim that overfitting is solved experimentally.

## Deliberately unchanged

- ShuffleNet topology, channel shuffle, LSTM/head and all other architectures.
- Full end-to-end scratch policy: backbone 1,253,604 / 1,269,784 parameters,
  all 56 BN layers trainable.
- Adam 3e-4, no newly added clipping, smoothing or decay; BN defaults unchanged.
- Early stopping, LR scheduler, validation checkpoint selection and threshold.
- Microphone, app/CSS, preprocessing, manifests and production artifacts.

Do not combine several unverified optimizer changes with the ordering fix and
attribute any improvement to a single cause. Investigate BN moving-statistic
quality and optimizer regularization next if a correctly shuffled run still
shows the failure; tune only on training/validation, never test.

## Verification / next run

```powershell
python -m unittest tests.test_training_shuffle tests.test_shufflenet_shapes tests.test_shufflenet_training_step tests.test_training_policy -q
python train.py --model shufflenetv2_lstm
```

Only unit/shape/gradient/two-update/save-load tests are run during this fix,
not full training. Preserve the prior Kaggle run as evidence. A fresh training
run is needed to measure validation improvement; no quality/speed is promised.

Verification outcome: 12 relevant tests passed in isolated runs (10 ordering,
shape/serialization/policy/metadata tests; 2 training-step/saturated-BCE tests).
The initial combined run failed on C: temporary-disk exhaustion and an unrelated
MnasNet gradient allocation; this is not counted as a passing full suite. Setting
TEMP/TMP for the test process to `outputs/shufflenet_audit_tmp` on D: allowed the
ShuffleNet save/load tests to pass. Only approximately 5.5 MB was free on C: at
that check; D: had approximately 25 GB. No user files were deleted.

Successful commands (from the project root):

```powershell
$env:TEMP = 'D:\audio-deepfake-mobilenet-lstm\outputs\shufflenet_audit_tmp'
$env:TMP = $env:TEMP
python -m unittest tests.test_shufflenet_training_step -v
python -m unittest tests.test_training_shuffle tests.test_shufflenet_shapes tests.test_training_policy.TrainingPolicyResolverTest tests.test_training_policy.TensorFlowTrainabilityPolicyTest tests.test_training_policy.ScratchMetadataTest -q
```

The canonical manifest hash remains
`8b55591d58d3658b8cafe0e77b6ebdedbaa67be2e339730a7276fe9b10958df9`.

Primary references:
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
- https://docs.pytorch.org/vision/main/_modules/torchvision/models/shufflenetv2.html

Local evidence: `src/dataset.py`, `config.py`, `train.py`,
`src/lava/training/tensorflow_lifecycle.py`, `ShuffleNetV2/utils.py`,
installed `keras/src/backend.py`, and the regression tests listed above.
