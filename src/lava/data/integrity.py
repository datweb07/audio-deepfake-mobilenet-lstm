"""Apply explicit checksum-based integrity policy to an audio inventory."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


INCLUDED = "INCLUDED"
LABEL_CONFLICT = "LABEL_CONFLICT"
DUPLICATE_EXCLUDED = "DUPLICATE_EXCLUDED"


def apply_integrity_policy(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Exclude label conflicts and retain one canonical member per same-label hash."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sha256"])].append(dict(record))

    annotated: list[dict[str, object]] = []
    for checksum in sorted(grouped):
        members = sorted(grouped[checksum], key=lambda row: str(row["path"]).casefold())
        labels = {int(row["label"]) for row in members}
        group_id = f"sha256:{checksum}"
        if len(labels) > 1:
            for row in members:
                row.update(
                    duplicate_group_id=group_id,
                    canonical_sample_id="",
                    integrity_status=LABEL_CONFLICT,
                    included=False,
                    exclusion_reason="identical bytes occur under both REAL and FAKE labels",
                )
                annotated.append(row)
            continue

        canonical_id = str(members[0]["sample_id"])
        for index, row in enumerate(members):
            included = index == 0
            row.update(
                duplicate_group_id=group_id,
                canonical_sample_id=canonical_id,
                integrity_status=INCLUDED if included else DUPLICATE_EXCLUDED,
                included=included,
                exclusion_reason="" if included else "same-label byte duplicate; canonical member retained",
            )
            annotated.append(row)
    return sorted(annotated, key=lambda row: str(row["path"]).casefold())


def integrity_summary(records: Iterable[dict[str, object]]) -> dict[str, int]:
    rows = list(records)
    duplicate_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        duplicate_groups[str(row["duplicate_group_id"])].append(row)
    actual_duplicate_groups = [members for members in duplicate_groups.values() if len(members) > 1]
    conflict_groups = {
        str(row["duplicate_group_id"])
        for row in rows
        if row["integrity_status"] == LABEL_CONFLICT
    }
    return {
        "scanned_samples": len(rows),
        "included_samples": sum(bool(row["included"]) for row in rows),
        "excluded_samples": sum(not bool(row["included"]) for row in rows),
        "duplicate_groups": len(actual_duplicate_groups),
        "redundant_duplicate_files": sum(max(0, len(members) - 1) for members in actual_duplicate_groups),
        "label_conflict_groups": len(conflict_groups),
        "label_conflict_files": sum(row["integrity_status"] == LABEL_CONFLICT for row in rows),
    }

