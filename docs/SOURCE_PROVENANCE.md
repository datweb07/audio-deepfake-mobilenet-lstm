# LAVA production source provenance

This file distinguishes production implementations under `src/lava/` from the read-only repositories used to understand the original architectures. None of the reference repository directories is a runtime dependency.

| Final detector | Production implementation | Reference source/files consulted | Reuse decision | Framework/input contract | Important deviation or constraint |
|---|---|---|---|---|---|
| MobileNetV3Small-LSTM | `src/model.py`, wrapped by `src/lava/models/tensorflow/mobilenetv3_lstm.py` | Existing production baseline; old MobileNet references were contextual only | existing code reused through wrapper | TensorFlow; 6 Mel-RGB images | Architecture intentionally unchanged; verified ImageNet weights |
| EfficientNet-B0-LSTM | `src/lava/models/tensorflow/efficientnet_b0_lstm.py` | `efficientnet-master/efficientnet/model.py`, `tfkeras.py`, `preprocessing.py`, tests and README; Keras application behavior verified in installed TF | architecture studied; production uses maintained `tf.keras.applications.EfficientNetB0` | TensorFlow; same Mel sequence | Built-in model contains input rescaling; no second normalization is added |
| ShuffleNetV2-1.0x-LSTM | `src/lava/models/tensorflow/shufflenetv2_lstm.py` | `ShuffleNetV2/blocks.py`, `network.py`, `train.py`, README | independently reimplemented from architecture concepts | TensorFlow; same Mel sequence | No reference-folder import and no unverified weight conversion; scratch experimental stratum |
| MnasNet-A1-1.0-LSTM | `src/lava/models/tensorflow/mnasnet_lstm.py` | `mnasnet/mnasnet_model.py`, `mnasnet_models.py`, `mnas_utils.py`, config and preprocessing files | TF1 topology studied, then independently ported to Keras 2.15 | TensorFlow; same Mel sequence | No TF1 runtime dependency and no unverified weight conversion; scratch experimental stratum |
| RawNet2 | `src/lava/models/pytorch/rawnet2.py` | `rawnet2-antispoofing-main/model.py`, `main.py`, `data_utils_LA.py`, `model_config_RawNet2.yaml` | MIT architecture adapted and modernized | PyTorch worker; mono 16 kHz waveform | Device-safe buffers/checkpoints; native logits `[spoof, bonafide]` are converted by adapter to `P(FAKE)` |
| AASIST | `src/lava/models/pytorch/aasist.py` | `aasist-main/models/AASIST.py`, `main.py`, `data_utils.py`, `evaluation.py`, config, LICENSE/NOTICE | MIT architecture adapted and modernized | PyTorch worker; mono 16 kHz waveform | Native graph branches retained; obsolete `torchcontrib` flow removed; test partition never selects checkpoints |

## Other reference repositories

| Repository | Decision | Reason |
|---|---|---|
| `ConvNets-TensorFlow2-master/` | inspiration only | Broad image-classifier patterns were useful context, but production uses focused implementations and maintained Keras applications where available. |
| `EfficientWord-Net-main/` | rejected for detector implementation | It is a wake-word/embedding use case rather than a deepfake anti-spoofing architecture. |
| `deepfake-audio-detection/` | historical context only | It is not imported and does not define a LAVA detector. |
| `enhancing-deepfake-detection-using-mobilenet-lstm-hybrid-model-main/` | historical provenance only | The current production baseline, not this folder, remains authoritative. |
| `mobilenetv3.pytorch/` | historical context only | LAVA's production MobileNet baseline remains TensorFlow/Keras. |

## Licensing boundary

- RawNet2 and AASIST reference repositories include MIT license files. Production adaptations retain architecture provenance in this report and module docstrings.
- ShuffleNetV2 and MnasNet production modules are independent implementations of published architectural concepts; source files are not runtime-copied or imported, and unverified pretrained weights are deliberately excluded.
- EfficientNet production delegates to TensorFlow/Keras rather than copying the reference package.
- Before external redistribution or publication, project owners should preserve applicable notices and perform a final license review of every distributed source and weight artifact.
