# ShuffleNet symbolic split compatibility

The Kaggle traceback is a Keras 3 Functional graph build failure, not a manifest
failure or training collapse. Calling `tf.split` directly on a symbolic
`KerasTensor` is rejected. The production fix moves exactly the same equal split
into the registered, weightless `LAVA>ChannelSplit` layer. Explicit output shapes
are supplied for split/shuffle so nested TimeDistributed shape inference works.

The topology's mathematical operations, channel ordering, trainable weights,
preprocessing, labels, scratch policy and optimizer are unchanged. The serialized
graph now contains a named custom Layer instead of an implicit TensorFlow op
wrapper; consumers must import the ShuffleNet module to register it (the registry
already does this). This is not a general Keras 3-to-Keras 2 checkpoint converter.

Shapes: `(B,6,224,224,3)` -> `(B,6,1024)` -> `(B,1)` sigmoid P(FAKE).
Total parameters: 1,868,441. Backbone total/trainable: 1,269,784 / 1,253,604.
All 56 backbone BN layers remain trainable.

Manifest check passes with unchanged hash:
`8b55591d58d3658b8cafe0e77b6ebdedbaa67be2e339730a7276fe9b10958df9`.
The real `train.py --model shufflenetv2_lstm --smoke-test` command passes on
local Python 3.11 / TensorFlow 2.15 / Keras 2.15 CPU. It uses one training and one
validation recording per class, performs checkpoint selection and save/load,
and does not publish production model/threshold artifacts.

The same real smoke command also returns `SMOKE_TESTED` with isolated Keras
3.13.2 packages on the local TensorFlow 2.15 CPU backend. This verifies Keras 3
symbolic build, optimization, checkpoint selection and save/load, not the exact
Python 3.12/TensorFlow/GPU combination on Kaggle. Production dependencies were
not upgraded. `pip check` passes.

`python -m unittest tests.test_shufflenet_shapes -q`: four tests pass under
Keras 2.15 and all four pass under Keras 3.13.2. Coverage includes exact split
values vs tf.split, clone serialization, channel permutation, stage shapes and
full-model inference save/load parity. No full training was run.

## Kaggle

Update the extracted code (not just the zip still sitting in the input folder).
Restart the kernel if the old module was already imported. In the notebook:

```python
%cd /kaggle/working/lava
import sys, tensorflow as tf, keras
print(sys.version, tf.__version__, keras.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
from src.lava.models.tensorflow.shufflenetv2_lstm import ChannelSplit
print(ChannelSplit)
!python -m src.lava.data.manifest check
!python train.py --model shufflenetv2_lstm --smoke-test
```

Only after `SMOKE_TESTED`, start a fresh full run:

```python
!python train.py --model shufflenetv2_lstm
```

`cuInit UNKNOWN ERROR (303)` is separate: the supplied traceback actually exits
on `tf.split`. The symbolic fix cannot provision a Kaggle GPU or repair its CUDA
runtime. If no GPU is listed, verify notebook accelerator settings/restart the
session before committing to full training; CPU smoke can still be valid.
