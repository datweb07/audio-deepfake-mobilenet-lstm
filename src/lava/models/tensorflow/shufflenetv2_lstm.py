"""Independent TensorFlow implementation of ShuffleNetV2 1.0x + LSTM."""

from __future__ import annotations

import tensorflow as tf

import config
from src.lava.models.tensorflow.base import KerasLightweightDetector
from src.lava.models.tensorflow.specs import SHUFFLENET_SPEC as SPEC
from src.lava.models.tensorflow.temporal_classifier import build_temporal_classifier


@tf.keras.utils.register_keras_serializable(package="LAVA")
class ChannelSplit(tf.keras.layers.Layer):
    """Equal channel split inside a Layer, valid for Keras 2 and 3 graphs."""

    def call(self, inputs: tf.Tensor):
        return tuple(tf.split(inputs, num_or_size_splits=2, axis=-1))

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if shape[-1] is None or shape[-1] % 2:
            raise ValueError("ChannelSplit requires a known even channel count")
        shape[-1] //= 2
        return (tuple(shape), tuple(shape))


@tf.keras.utils.register_keras_serializable(package="LAVA")
class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(inputs)
        batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        tf.debugging.assert_equal(channels % self.groups, 0)
        channels_per_group = channels // self.groups
        output = tf.reshape(inputs, [batch, height, width, self.groups, channels_per_group])
        output = tf.transpose(output, [0, 1, 2, 4, 3])
        return tf.reshape(output, [batch, height, width, channels])

    def compute_output_shape(self, input_shape):
        # Keras 3 TimeDistributed asks nested layers for symbolic output shapes.
        return tuple(input_shape)

    def get_config(self) -> dict[str, int]:
        return {**super().get_config(), "groups": self.groups}


def _pointwise(x: tf.Tensor, filters: int, name: str, activate: bool = True) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(name=f"{name}_relu")(x) if activate else x


def _depthwise(x: tf.Tensor, stride: int, name: str) -> tf.Tensor:
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    return tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)


def shuffle_unit(x: tf.Tensor, out_channels: int, stride: int, name: str) -> tf.Tensor:
    if stride not in (1, 2):
        raise ValueError("ShuffleNetV2 unit stride must be 1 or 2")
    if out_channels % 2:
        raise ValueError("ShuffleNetV2 output channels must be even")
    branch_channels = out_channels // 2
    input_channels = x.shape[-1]
    if stride == 1:
        if input_channels != out_channels or input_channels is None or int(input_channels) % 2:
            raise ValueError("Stride-1 ShuffleNet unit requires even input_channels == out_channels")
        branch1, branch2 = ChannelSplit(name=f"{name}_split")(x)
        branch2 = _pointwise(branch2, branch_channels, f"{name}_b2_pw1")
        branch2 = _depthwise(branch2, 1, f"{name}_b2")
        branch2 = _pointwise(branch2, branch_channels, f"{name}_b2_pw2")
    else:
        branch1 = _depthwise(x, 2, f"{name}_b1")
        branch1 = _pointwise(branch1, branch_channels, f"{name}_b1_pw")
        branch2 = _pointwise(x, branch_channels, f"{name}_b2_pw1")
        branch2 = _depthwise(branch2, 2, f"{name}_b2")
        branch2 = _pointwise(branch2, branch_channels, f"{name}_b2_pw2")
    output = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")([branch1, branch2])
    return ChannelShuffle(groups=2, name=f"{name}_shuffle")(output)


def build_backbone() -> tf.keras.Model:
    inputs = tf.keras.layers.Input((*config.IMAGE_SIZE, config.CHANNELS), name="image")
    x = tf.keras.layers.Conv2D(24, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, padding="same", name="stem_pool")(x)
    for stage_index, (repeats, channels) in enumerate(zip((4, 8, 4), (116, 232, 464)), start=2):
        x = shuffle_unit(x, channels, 2, f"stage{stage_index}_unit1")
        for unit_index in range(2, repeats + 1):
            x = shuffle_unit(x, channels, 1, f"stage{stage_index}_unit{unit_index}")
    x = _pointwise(x, 1024, "head")
    outputs = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    return tf.keras.Model(inputs, outputs, name="shufflenetv2_1_0x")


def build_model(weights: str | None = None) -> tuple[tf.keras.Model, tf.keras.Model]:
    if weights not in (None,):
        raise ValueError("ShuffleNetV2 ImageNet weights are not yet verified; use weights=None")
    return build_temporal_classifier(detector_name=SPEC.name, backbone=build_backbone())


class ShuffleNetV2LSTMDetector(KerasLightweightDetector):
    def __init__(self) -> None:
        super().__init__(SPEC, build_model)
