"""TensorFlow detector metadata safe to import without importing TensorFlow."""

from pathlib import Path

import config
from src.lava.artifacts import mobilenet_artifacts, tensorflow_artifacts
from src.lava.contracts import DetectorSpec, Initialization, TrainingPolicy


_MOBILENET_MODEL, _MOBILENET_THRESHOLD, _MOBILENET_METADATA = mobilenet_artifacts()

MOBILENET_SPEC = DetectorSpec(
    name="mobilenetv3_lstm", display_name="MobileNetV3Small-LSTM", group="lightweight",
    framework="tensorflow", input_type="mel_sequence", sample_rate=config.SAMPLE_RATE,
    audio_duration=config.AUDIO_DURATION, num_segments=config.NUM_SEGMENTS,
    model_artifact=_MOBILENET_MODEL, threshold_artifact=_MOBILENET_THRESHOLD,
    metadata_artifact=_MOBILENET_METADATA, pretraining_status="VERIFIED_IMAGENET",
    initialization=Initialization.IMAGENET_PRETRAINED,
    training_policy=TrainingPolicy.PRETRAINED_TRANSFER,
)


def _new(
    name: str,
    display_name: str,
    pretraining_status: str,
    initialization: Initialization,
    training_policy: TrainingPolicy,
) -> DetectorSpec:
    model, threshold, metadata = tensorflow_artifacts(name)
    return DetectorSpec(
        name=name, display_name=display_name, group="lightweight", framework="tensorflow",
        input_type="mel_sequence", sample_rate=config.SAMPLE_RATE, audio_duration=config.AUDIO_DURATION,
        num_segments=config.NUM_SEGMENTS, model_artifact=model, threshold_artifact=threshold,
        metadata_artifact=metadata, pretraining_status=pretraining_status,
        initialization=initialization, training_policy=training_policy,
    )


EFFICIENTNET_SPEC = _new(
    "efficientnet_b0_lstm", "EfficientNet-B0-LSTM", "VERIFIED_IMAGENET",
    Initialization.IMAGENET_PRETRAINED, TrainingPolicy.PRETRAINED_TRANSFER,
)
SHUFFLENET_SPEC = _new(
    "shufflenetv2_lstm", "ShuffleNetV2-1.0x-LSTM", "PRETRAINING_NOT_VERIFIED",
    Initialization.SCRATCH, TrainingPolicy.SCRATCH_END_TO_END,
)
MNASNET_SPEC = _new(
    "mnasnet_lstm", "MnasNet-A1-1.0-LSTM", "PRETRAINING_NOT_VERIFIED",
    Initialization.SCRATCH, TrainingPolicy.SCRATCH_END_TO_END,
)
