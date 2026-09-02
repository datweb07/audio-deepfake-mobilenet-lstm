"""TensorFlow 2 reimplementation of the audited MnasNet-A1 1.0 backbone."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

import config
from src.lava.models.tensorflow.base import KerasLightweightDetector
from src.lava.models.tensorflow.specs import MNASNET_SPEC as SPEC
from src.lava.models.tensorflow.temporal_classifier import build_temporal_classifier


BN_ARGS = {"momentum": 0.99, "epsilon": 1e-3}


@dataclass(frozen=True)
class BlockSpec:
    repeats: int
    kernel: int
    stride: int
    expansion: int
    input_filters: int
    output_filters: int
    se_ratio: float | None = None
    skip: bool = True


A1_BLOCKS = (
    BlockSpec(1, 3, 1, 1, 32, 16, skip=False),
    BlockSpec(2, 3, 2, 6, 16, 24),
    BlockSpec(3, 5, 2, 3, 24, 40, se_ratio=0.25),
    BlockSpec(4, 3, 2, 6, 40, 80),
    BlockSpec(2, 3, 1, 6, 80, 112, se_ratio=0.25),
    BlockSpec(3, 5, 2, 6, 112, 160, se_ratio=0.25),
    BlockSpec(1, 3, 1, 6, 160, 320),
)


def _bn_relu(x: tf.Tensor, name: str) -> tf.Tensor:
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn", **BN_ARGS)(x)
    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)


def mnas_block(
    x: tf.Tensor,
    *,
    output_filters: int,
    kernel: int,
    stride: int,
    expansion: int,
    se_ratio: float | None,
    skip: bool,
    name: str,
) -> tf.Tensor:
    identity = x
    input_filters = int(x.shape[-1])
    expanded_filters = input_filters * expansion
    if expansion != 1:
        x = tf.keras.layers.Conv2D(expanded_filters, 1, padding="same", use_bias=False, name=f"{name}_expand_conv")(x)
        x = _bn_relu(x, f"{name}_expand")
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_depthwise")(x)
    x = _bn_relu(x, f"{name}_depthwise")
    if se_ratio is not None:
        reduced = max(1, int(input_filters * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_se_pool")(x)
        se = tf.keras.layers.Conv2D(reduced, 1, activation="relu", name=f"{name}_se_reduce")(se)
        se = tf.keras.layers.Conv2D(expanded_filters, 1, activation="sigmoid", name=f"{name}_se_expand")(se)
        x = tf.keras.layers.Multiply(name=f"{name}_se_scale")([x, se])
    x = tf.keras.layers.Conv2D(output_filters, 1, padding="same", use_bias=False, name=f"{name}_project_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_project_bn", **BN_ARGS)(x)
    if skip and stride == 1 and input_filters == output_filters:
        x = tf.keras.layers.Add(name=f"{name}_skip")([identity, x])
    return x


def build_backbone() -> tf.keras.Model:
    inputs = tf.keras.layers.Input((*config.IMAGE_SIZE, config.CHANNELS), name="image")
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = _bn_relu(x, "stem")
    for stage_index, spec in enumerate(A1_BLOCKS, start=1):
        for repeat_index in range(spec.repeats):
            x = mnas_block(
                x,
                output_filters=spec.output_filters,
                kernel=spec.kernel,
                stride=spec.stride if repeat_index == 0 else 1,
                expansion=spec.expansion,
                se_ratio=spec.se_ratio,
                skip=spec.skip,
                name=f"stage{stage_index}_block{repeat_index + 1}",
            )
    x = tf.keras.layers.Conv2D(1280, 1, padding="same", use_bias=False, name="head_conv")(x)
    x = _bn_relu(x, "head")
    outputs = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    return tf.keras.Model(inputs, outputs, name="mnasnet_a1_1_0")


def build_model(weights: str | None = None) -> tuple[tf.keras.Model, tf.keras.Model]:
    if weights not in (None,):
        raise ValueError("MnasNet-A1 ImageNet weights are not yet verified; use weights=None")
    return build_temporal_classifier(detector_name=SPEC.name, backbone=build_backbone())


class MnasNetLSTMDetector(KerasLightweightDetector):
    def __init__(self) -> None:
        super().__init__(SPEC, build_model)
