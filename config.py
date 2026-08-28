"""Central configuration for the root audio-deepfake implementation."""

from __future__ import annotations

import os


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_DIR = os.path.join(DATA_DIR, "REAL")
FAKE_DIR = os.path.join(DATA_DIR, "FAKE")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

PHASE1_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_phase1.keras")
PHASE2_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_phase2.keras")
LEGACY_PHASE1_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_phase1.h5")
LEGACY_PHASE2_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_phase2.h5")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "best_threshold.txt")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

for directory in (MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, LOGS_DIR, REAL_DIR, FAKE_DIR):
    os.makedirs(directory, exist_ok=True)


# Audio and spectrogram
SAMPLE_RATE = 22_050
AUDIO_DURATION = 3.0
SEGMENT_DURATION = 0.5
N_MELS = 128
FMIN = 20
FMAX = 8_000
HOP_LENGTH = 512
N_FFT = 2_048
TOP_DB = 80.0
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


# Image/model input
IMAGE_SIZE = (224, 224)
CHANNELS = 3
INPUT_VALUE_MIN = 0.0
INPUT_VALUE_MAX = 255.0


# Derived temporal geometry
NUM_SEGMENTS = int(round(AUDIO_DURATION / SEGMENT_DURATION))
TOTAL_SAMPLES = int(round(SAMPLE_RATE * AUDIO_DURATION))
SEGMENT_SAMPLES = int(round(SAMPLE_RATE * SEGMENT_DURATION))

if NUM_SEGMENTS * SEGMENT_SAMPLES != TOTAL_SAMPLES:
    raise ValueError("AUDIO_DURATION must be exactly divisible by SEGMENT_DURATION")


# Dataset and reproducibility
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
SHUFFLE_BUFFER_SIZE = 1_024

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")


# Training
BATCH_SIZE = 16
PHASE1_EPOCHS = 50
PHASE2_EPOCHS = 50
PHASE1_LR = 1e-4
PHASE2_LR = 1e-5
FINE_TUNE_LAST_LAYERS = 20
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.4


# Inference and threshold calibration
DEFAULT_THRESHOLD = 0.5
THRESHOLD_SEARCH_MIN = 0.10
THRESHOLD_SEARCH_MAX = 0.90
THRESHOLD_SEARCH_STEP = 0.01
REAL_LABEL = 0
FAKE_LABEL = 1
REAL_NAME = "REAL"
FAKE_NAME = "FAKE"
