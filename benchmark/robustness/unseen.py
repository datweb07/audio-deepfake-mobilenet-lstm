"""Unseen-data contract blocked by unavailable generator/dataset metadata."""

STATUS = "NOT_RUN"


def status() -> dict[str, str]:
    return {
        "status": STATUS,
        "generator_disjoint": STATUS,
        "dataset_disjoint": STATUS,
        "reason": "generator_id and dataset_id are UNKNOWN in current manifest",
    }

