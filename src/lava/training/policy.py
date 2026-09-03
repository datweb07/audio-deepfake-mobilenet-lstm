"""Scientific data-role guards shared by every training framework."""

from src.lava.contracts import DetectorSpec, Initialization, TrainingPolicy


TRAINING_SPLITS = ("train", "validation")
TEST_SPLIT = "test"


def require_validation_source(source_split: str) -> None:
    if source_split != "validation":
        raise ValueError("Checkpoint selection, early stopping, and threshold calibration require validation data")


def assert_test_isolation(training_splits: tuple[str, ...] = TRAINING_SPLITS) -> None:
    if TEST_SPLIT in training_splits:
        raise ValueError("Test leakage detected: test split must not participate in training or calibration")


def resolve_training_policy(spec: DetectorSpec) -> TrainingPolicy:
    """Validate initialization/policy compatibility before any optimizer is built."""
    policy = spec.training_policy
    if policy == TrainingPolicy.PRETRAINED_TRANSFER:
        if spec.framework != "tensorflow":
            raise ValueError("Pretrained transfer policy currently requires a TensorFlow detector")
        if spec.initialization != Initialization.IMAGENET_PRETRAINED:
            raise ValueError(f"Refusing to freeze a non-pretrained backbone for {spec.name}")
        if spec.pretraining_status != "VERIFIED_IMAGENET":
            raise ValueError(f"Verified ImageNet weights are required for transfer policy: {spec.name}")
    elif policy == TrainingPolicy.SCRATCH_END_TO_END:
        if spec.framework != "tensorflow" or spec.initialization != Initialization.SCRATCH:
            raise ValueError(f"Scratch end-to-end policy mismatch for {spec.name}")
        if spec.pretraining_status == "VERIFIED_IMAGENET":
            raise ValueError(f"Scratch policy cannot silently discard verified weights: {spec.name}")
    elif policy == TrainingPolicy.NATIVE_REFERENCE:
        if spec.framework != "pytorch" or spec.initialization != Initialization.NATIVE:
            raise ValueError(f"Native reference policy mismatch for {spec.name}")
    else:
        raise ValueError(f"Unsupported training policy for {spec.name}: {policy}")
    return policy


def weights_for_policy(spec: DetectorSpec) -> str | None:
    policy = resolve_training_policy(spec)
    if policy == TrainingPolicy.PRETRAINED_TRANSFER:
        return "imagenet"
    if policy == TrainingPolicy.SCRATCH_END_TO_END:
        return None
    raise ValueError(f"Native detector weights are managed by its framework-specific trainer: {spec.name}")
