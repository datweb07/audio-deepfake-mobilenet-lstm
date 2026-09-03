"""Export and verify the MobileNet weights-only deployment fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.artifacts import (  # noqa: E402
    _load_production_numpy_weights,
    _save_numpy_weights,
    load_production_model,
    validate_model_contract,
)
from src.model import build_hybrid_model  # noqa: E402


def main() -> None:
    model_path = Path(config.MODEL_PATH)
    weights_path = Path(config.MODEL_WEIGHTS_PATH)
    numpy_weights_path = Path(config.MODEL_NUMPY_WEIGHTS_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"Production model not found: {model_path}")

    source = load_production_model(compile=False, weights_path=str(weights_path))
    validate_model_contract(source)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    source.save_weights(weights_path)

    restored, _ = build_hybrid_model(weights=None)
    restored.load_weights(weights_path)
    validate_model_contract(restored)
    # Always serialize ordered tensors from the canonical rebuilt model. The
    # full-model object may expose a different nested trainable-weight order.
    _save_numpy_weights(restored, str(numpy_weights_path))
    numpy_restored = _load_production_numpy_weights(str(numpy_weights_path))

    sample = np.zeros((1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS), dtype=np.float32)
    expected = source(sample, training=False).numpy()
    actual = restored(sample, training=False).numpy()
    numpy_actual = numpy_restored(sample, training=False).numpy()
    max_abs_difference = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-6, atol=1e-7):
        raise RuntimeError(f"Deployment weights parity failed: max abs diff={max_abs_difference}")
    numpy_max_abs_difference = float(np.max(np.abs(expected - numpy_actual)))
    if not np.allclose(expected, numpy_actual, rtol=1e-6, atol=1e-7):
        raise RuntimeError(
            f"NumPy deployment weights parity failed: max abs diff={numpy_max_abs_difference}"
        )

    print(f"Exported: {weights_path}")
    print(f"Parameters: {restored.count_params():,}")
    print(f"P(FAKE) parity max abs difference: {max_abs_difference:.10g}")
    print(f"Exported serialization-independent fallback: {numpy_weights_path}")
    print(f"NumPy P(FAKE) parity max abs difference: {numpy_max_abs_difference:.10g}")


if __name__ == "__main__":
    main()
