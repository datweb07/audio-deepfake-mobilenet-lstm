"""Unified single-audio prediction CLI returning the LAVA P(FAKE) contract."""

from __future__ import annotations

import argparse
import os

from src.lava.score_semantics import classify_probability
from src.lava.artifacts import load_threshold
from src.lava.registry import create, get_spec, names


def main(audio_path: str, model_name: str = "mobilenetv3_lstm") -> None:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    spec = get_spec(model_name)
    detector = create(model_name)
    detector.load()
    threshold = load_threshold(spec)
    probability = float(detector.predict_scores([audio_path])[0])
    result = classify_probability(probability, threshold)
    print(f"Model: {spec.display_name}")
    print(f"Framework: {spec.framework}")
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print(f"Raw P(FAKE): {result.probability_fake:.4f}")
    print(f"Threshold: {result.threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=names(), default="mobilenetv3_lstm")
    parser.add_argument("--audio", required=True, help="Path to WAV/FLAC/MP3/OGG/M4A audio")
    arguments = parser.parse_args()
    main(arguments.audio, arguments.model)
