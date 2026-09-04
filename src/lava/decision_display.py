"""Presentation only: never calibrate scores or change decision thresholds."""

from src.lava.artifacts import load_json
from src.lava.errors import ArtifactNotReadyError


SCORE_NOTICE = (
    "Raw FAKE score (P(FAKE)) is a model output, not a verified probability of "
    "authenticity. Threshold selection does not calibrate probabilities."
)


def threshold_description(spec) -> str:
    """Prefer the threshold's provenance; do not assume calibration from value."""
    try:
        if spec.threshold_artifact.suffix.lower() == ".json":
            source = load_json(spec.threshold_artifact).get("source", "")
        else:
            source = load_json(spec.metadata_artifact).get("threshold_source", "")
    except (ArtifactNotReadyError, OSError):
        source = ""
    source = str(source).lower()
    if "default" in source or "uncalibrated" in source or "no_calibration" in source:
        return "Default threshold — not tuned on this dataset's validation split."
    if source.startswith("validation"):
        return "Validation-selected decision threshold (not probability calibration)."
    return "Threshold provenance unavailable — validation selection is not verified."


def decision_explanation(result) -> str:
    operator = ">=" if result.probability_fake >= result.threshold else "<"
    return (
        f"Raw FAKE score {result.probability_fake:.4f} {operator} "
        f"threshold {result.threshold:.4f}: classified as {result.prediction}. "
        "The decision uses this threshold, not a comparison of REAL and FAKE bars."
    )
