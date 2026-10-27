"""Separate identity overlap dimensions and candidate-only acoustic similarity."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .controls import estimate_delay
from .provenance import SourceRecord, known
from .transforms import Audio


def overlap_dimensions(records: list[SourceRecord]) -> dict:
    result = {}
    for dimension in ("speaker_id", "source_speaker_id", "target_speaker_id", "generator_id", "attack_id", "dataset_id", "parent_recording_id", "checksum"):
        groups = defaultdict(set)
        missing = 0
        for record in records:
            value = getattr(record, dimension)
            if known(value):
                groups[value].add(record.split)
            else:
                missing += 1
        result[dimension] = {"known_ids": len(groups), "unknown_records": missing,
                             "cross_partition_ids": {value: sorted(splits) for value, splits in groups.items() if len(splits) > 1}}
    return result


def near_duplicate_candidates(audio: dict[str, Audio], *, threshold: float = .98, max_pairs: int = 10000) -> list[dict]:
    """Gain/short-shift-tolerant correlation flags; never delete or certify identities."""
    if not .5 <= threshold <= 1 or len(audio) * (len(audio) - 1) // 2 > max_pairs:
        raise ValueError("Invalid threshold or excessive quadratic comparison budget")
    signals = {}
    for name, clip in audio.items():
        divisor = gcd(clip.sample_rate, 8000)
        values = resample_poly(clip.samples, 8000 // divisor, clip.sample_rate // divisor).astype(np.float64)
        signals[name] = values
    candidates = []
    for first, second in combinations(sorted(signals), 2):
        a, b = signals[first], signals[second]
        if not np.any(a) or not np.any(b):
            continue
        if abs(len(a) - len(b)) > max(80, .05 * max(len(a), len(b))):
            continue
        diagnostic = estimate_delay(a, b, max_lag=160)
        if diagnostic["normalized_cross_correlation"] >= threshold:
            candidates.append({"sample_a": first, "sample_b": second, "status": "CANDIDATE_REVIEW_REQUIRED",
                               "similarity": diagnostic["normalized_cross_correlation"],
                               "lag_at_8khz": diagnostic["lag_samples"], "threshold": threshold,
                               "automatic_exclusion": False})
    return candidates
