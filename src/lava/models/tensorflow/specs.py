"""TensorFlow detector metadata safe to import without importing TensorFlow."""

from pathlib import Path

import config
from src.lava.artifacts import tensorflow_artifacts
from src.lava.contracts import DetectorSpec


MOBILENET_SPEC = DetectorSpec(
    name="mobilenetv3_lstm", display_name="MobileNetV3Small-LSTM", group="lightweight",
    framework="tensorflow", input_type="mel_sequence", sample_rate=config.SAMPLE_RATE,
    audio_duration=config.AUDIO_DURATION, num_segments=config.NUM_SEGMENTS,
    model_artifact=Path(config.MODEL_PATH), threshold_artifact=Path(config.THRESHOLD_PATH),
    metadata_artifact=Path(config.MODEL_METADATA_PATH), pretraining_status="VERIFIED_IMAGENET",
)


def _new(name: str, display_name: str, pretraining_status: str) -> DetectorSpec:
    model, threshold, metadata = tensorflow_artifacts(name)
    return DetectorSpec(
        name=name, display_name=display_name, group="lightweight", framework="tensorflow",
        input_type="mel_sequence", sample_rate=config.SAMPLE_RATE, audio_duration=config.AUDIO_DURATION,
        num_segments=config.NUM_SEGMENTS, model_artifact=model, threshold_artifact=threshold,
        metadata_artifact=metadata, pretraining_status=pretraining_status,
    )


EFFICIENTNET_SPEC = _new("efficientnet_b0_lstm", "EfficientNet-B0-LSTM", "VERIFIED_IMAGENET")
SHUFFLENET_SPEC = _new("shufflenetv2_lstm", "ShuffleNetV2-1.0x-LSTM", "PRETRAINING_NOT_VERIFIED")
MNASNET_SPEC = _new("mnasnet_lstm", "MnasNet-A1-1.0-LSTM", "PRETRAINING_NOT_VERIFIED")
