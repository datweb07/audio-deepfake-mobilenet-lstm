"""Validated C1 pilot configurations. Confirmatory runs are intentionally gated."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from .evidence import object_hash
from .transforms import Transform


@dataclass(frozen=True)
class C1Config:
    experiment_id: str
    model_id: str
    per_class: int
    seed: int
    bootstrap_replicates: int
    sample_rate: int
    max_seconds: float
    common_peak: float
    transform_a: dict
    transform_b: dict
    split_manifest: str = "data/manifests/split_manifest.csv"
    scope: str = "PILOT"
    expected_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", self.experiment_id):
            raise ValueError("Invalid experiment ID")
        if self.scope not in {"SMOKE", "PILOT", "DEVELOPMENT"}:
            raise ValueError("Confirmatory execution requires external protocol; legacy pilot cannot certify it")
        if self.model_id not in {"mobilenetv3_lstm", "shufflenetv2_lstm"}:
            raise ValueError("Initial C1 runner supports the two lightweight candidates")
        if not 1 <= self.per_class <= 100 or self.bootstrap_replicates < 100:
            raise ValueError("Invalid pilot size or bootstrap count")
        if not 0 < self.common_peak <= 0.5 or not 0 < self.max_seconds <= 3.0 or self.sample_rate != 22050:
            raise ValueError("Initial Mel student contract: <=3s, 22050Hz, documented peak headroom <=0.5")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        first, second = Transform(**self.transform_a), Transform(**self.transform_b)
        if first.key == second.key:
            raise ValueError("A and B require distinct stable transform keys")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return object_hash(self.to_dict())


def load_config(path: Path) -> C1Config:
    return C1Config(**json.loads(path.read_text(encoding="utf-8")))
