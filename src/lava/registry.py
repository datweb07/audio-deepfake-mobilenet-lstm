"""Lazy detector registry; importing it never imports PyTorch."""

from __future__ import annotations

from collections.abc import Callable

from src.lava.contracts import DetectorSpec, LAVADetector
from src.lava.errors import DetectorNotFoundError


Factory = Callable[[], LAVADetector]
_SPECS: dict[str, DetectorSpec] = {}
_FACTORIES: dict[str, Factory] = {}


def register(spec: DetectorSpec, factory: Factory) -> None:
    if spec.name in _SPECS:
        raise ValueError(f"Detector already registered: {spec.name}")
    _SPECS[spec.name] = spec
    _FACTORIES[spec.name] = factory


def names() -> tuple[str, ...]:
    return tuple(_SPECS)


def specs() -> tuple[DetectorSpec, ...]:
    return tuple(_SPECS.values())


def get_spec(name: str) -> DetectorSpec:
    try:
        return _SPECS[name]
    except KeyError as exc:
        raise DetectorNotFoundError(f"Unknown detector '{name}'. Available: {', '.join(names())}") from exc


def create(name: str) -> LAVADetector:
    get_spec(name)
    return _FACTORIES[name]()


def _register_builtins() -> None:
    from src.lava.models.tensorflow.specs import (
        EFFICIENTNET_SPEC,
        MNASNET_SPEC,
        MOBILENET_SPEC,
        SHUFFLENET_SPEC,
    )

    def mobile_factory():
        from src.lava.models.tensorflow.mobilenetv3_lstm import MobileNetV3LSTMDetector
        return MobileNetV3LSTMDetector()

    def efficientnet_factory():
        from src.lava.models.tensorflow.efficientnet_b0_lstm import EfficientNetB0LSTMDetector
        return EfficientNetB0LSTMDetector()

    def shufflenet_factory():
        from src.lava.models.tensorflow.shufflenetv2_lstm import ShuffleNetV2LSTMDetector
        return ShuffleNetV2LSTMDetector()

    def mnasnet_factory():
        from src.lava.models.tensorflow.mnasnet_lstm import MnasNetLSTMDetector
        return MnasNetLSTMDetector()

    register(MOBILENET_SPEC, mobile_factory)
    register(EFFICIENTNET_SPEC, efficientnet_factory)
    register(SHUFFLENET_SPEC, shufflenet_factory)
    register(MNASNET_SPEC, mnasnet_factory)
    from src.lava.models.pytorch.specs import (
        AASIST_SPEC,
        RAWNET2_SPEC,
        aasist_factory,
        rawnet2_factory,
    )

    register(RAWNET2_SPEC, rawnet2_factory)
    register(AASIST_SPEC, aasist_factory)

    from src.lava.models.pytorch.specs import (
        RAWNET2_PRETRAINED_SPEC,
        AASIST_PRETRAINED_SPEC,
        rawnet2_pretrained_factory,
        aasist_pretrained_factory,
    )
    register(RAWNET2_PRETRAINED_SPEC, rawnet2_pretrained_factory)
    register(AASIST_PRETRAINED_SPEC, aasist_pretrained_factory)


_register_builtins()
