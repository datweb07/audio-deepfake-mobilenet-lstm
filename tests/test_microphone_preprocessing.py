from __future__ import annotations

import io
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.lava.preprocessing.microphone import (
    MicrophoneQualityError,
    prepare_microphone_recording,
)
from src.preprocessing import process_audio_file


def wav_bytes(audio: np.ndarray, sample_rate: int = 48_000) -> bytes:
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


class MicrophonePreprocessingTest(unittest.TestCase):
    def test_valid_speech_is_trimmed_without_gain_normalization(self) -> None:
        sample_rate = 48_000
        speech_time = np.arange(3 * sample_rate, dtype=np.float32) / sample_rate
        speech = (0.2 * np.sin(2 * np.pi * 220 * speech_time)).astype(np.float32)
        audio = np.concatenate(
            [np.zeros(sample_rate // 2, dtype=np.float32), speech, np.zeros(sample_rate // 2, dtype=np.float32)]
        )
        prepared = prepare_microphone_recording(wav_bytes(audio, sample_rate))
        self.assertEqual(prepared.quality.source_sample_rate, sample_rate)
        self.assertAlmostEqual(prepared.quality.original_duration, 4.0, places=2)
        self.assertGreaterEqual(prepared.quality.retained_duration, 3.0)
        self.assertLess(prepared.quality.retained_duration, 3.4)
        with sf.SoundFile(io.BytesIO(prepared.wav_bytes)) as output:
            restored = output.read(dtype="float32")
        self.assertLessEqual(float(np.max(np.abs(restored))), 0.201)
        self.assertGreater(float(np.max(np.abs(restored))), 0.19)

    def test_silence_is_rejected_before_inference(self) -> None:
        with self.assertRaisesRegex(MicrophoneQualityError, "too quiet"):
            prepare_microphone_recording(wav_bytes(np.zeros(3 * 48_000, dtype=np.float32)))

    def test_clipped_capture_is_rejected_before_inference(self) -> None:
        with self.assertRaisesRegex(MicrophoneQualityError, "clipped"):
            prepare_microphone_recording(wav_bytes(np.ones(3 * 48_000, dtype=np.float32)))

    def test_short_capture_is_rejected_before_inference(self) -> None:
        audio = np.full(48_000, 0.1, dtype=np.float32)
        with self.assertRaisesRegex(MicrophoneQualityError, "too short"):
            prepare_microphone_recording(wav_bytes(audio))

    def test_prepared_capture_enters_the_unchanged_model_pipeline(self) -> None:
        sample_rate = 48_000
        time = np.arange(3 * sample_rate, dtype=np.float32) / sample_rate
        audio = (0.15 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)
        prepared = prepare_microphone_recording(wav_bytes(audio, sample_rate))
        path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
                handle.write(prepared.wav_bytes)
                path = handle.name
            tensor = process_audio_file(path)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        self.assertEqual(tensor.shape, (6, 224, 224, 3))
        self.assertTrue(np.all(np.isfinite(tensor)))


if __name__ == "__main__":
    unittest.main()
