"""Scientific data-role guards shared by every training framework."""


TRAINING_SPLITS = ("train", "validation")
TEST_SPLIT = "test"


def require_validation_source(source_split: str) -> None:
    if source_split != "validation":
        raise ValueError("Checkpoint selection, early stopping, and threshold calibration require validation data")


def assert_test_isolation(training_splits: tuple[str, ...] = TRAINING_SPLITS) -> None:
    if TEST_SPLIT in training_splits:
        raise ValueError("Test leakage detected: test split must not participate in training or calibration")

