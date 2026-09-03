"""Shared lightweight Mel-sequence classifier used only by new LAVA backbones."""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

import config


BackboneBuilder = Callable[[str | None], tf.keras.Model]


def build_temporal_classifier(
    *,
    detector_name: str,
    backbone: tf.keras.Model,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.layers.Input(
        shape=(config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS),
        name="mel_segment_sequence",
    )
    backbone.trainable = False
    embeddings = tf.keras.layers.TimeDistributed(
        backbone,
        name=f"time_distributed_{detector_name.replace('_lstm', '')}",
    )(inputs)
    temporal = tf.keras.layers.LSTM(
        config.LSTM_UNITS,
        return_sequences=False,
        name="temporal_lstm",
    )(embeddings)
    features = tf.keras.layers.Dense(
        config.DENSE_UNITS, activation="relu", name="classifier_dense"
    )(temporal)
    features = tf.keras.layers.Dropout(
        config.DROPOUT_RATE, name="classifier_dropout"
    )(features)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="probability_fake"
    )(features)
    model = tf.keras.Model(inputs, outputs, name=f"{detector_name}_audio_deepfake")
    return model, backbone


def freeze_backbone(backbone: tf.keras.Model) -> None:
    backbone.trainable = False


def enable_scratch_end_to_end(backbone: tf.keras.Model) -> None:
    """Make every learnable backbone layer, including BatchNorm, trainable."""
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = True
        if isinstance(layer, tf.keras.Model):
            enable_scratch_end_to_end(layer)


def parameter_status(backbone: tf.keras.Model) -> dict[str, int]:
    """Report unique parameter elements after the requested trainability policy."""
    if not backbone.built:
        return {"total": 0, "trainable": 0, "frozen": 0}
    total = sum(int(tf.keras.backend.count_params(weight)) for weight in backbone.weights)
    trainable = sum(
        int(tf.keras.backend.count_params(weight)) for weight in backbone.trainable_weights
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def unfreeze_backbone(backbone: tf.keras.Model, tail_layers: int) -> None:
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - tail_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
