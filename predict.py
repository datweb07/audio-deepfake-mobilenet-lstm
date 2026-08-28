"""Single-file prediction CLI."""

from __future__ import annotations

import argparse
import os

import tensorflow as tf

from src.inference import predict_audio
from src.metrics import resolve_model_path


def main(audio_path: str) -> None:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model_path = resolve_model_path()
    model = tf.keras.models.load_model(model_path)
    result = predict_audio(model, audio_path)
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print(f"Raw probability P(FAKE): {result.probability_fake:.4f}")
    print(f"Threshold: {result.threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect deepfake audio")
    parser.add_argument("--audio", required=True, help="Path to WAV/FLAC/MP3/OGG/M4A audio")
    arguments = parser.parse_args()
    main(arguments.audio)
