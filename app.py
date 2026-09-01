"""Artifact-gated Streamlit UI for trained LAVA detectors."""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import config
from src.lava.score_semantics import classify_probability
from src.lava.artifacts import artifact_readiness, load_threshold
from src.lava.registry import create, get_spec, specs
from src.preprocessing import create_mel_spectrogram_db, load_audio, segment_audio


st.set_page_config(page_title="LAVA Deepfake Voice Detection", page_icon="🎙️", layout="wide")


@st.cache_resource
def load_detector(model_name: str):
    detector = create(model_name)
    detector.load()
    return detector, load_threshold(detector.spec)


def available_specs():
    available = []
    for spec in specs():
        ready, _ = artifact_readiness(spec)
        if not ready:
            continue
        try:
            detector = create(spec.name)
            detector.load()
        except Exception:
            continue
        available.append(spec)
    return available


def main() -> None:
    st.title("LAVA Deepfake Voice Detection")
    st.caption("REAL=0, FAKE=1; every detector returns P(FAKE).")
    candidates = available_specs()
    if not candidates:
        st.warning("No valid trained detector found. Run: python train.py")
        return
    default_index = next((index for index, spec in enumerate(candidates) if spec.name == "mobilenetv3_lstm"), 0)
    selected_name = st.selectbox(
        "Detector", [spec.name for spec in candidates], index=default_index,
        format_func=lambda name: get_spec(name).display_name,
    )
    selected_spec = get_spec(selected_name)
    try:
        detector, threshold = load_detector(selected_name)
    except Exception as exc:
        st.warning(str(exc))
        return
    st.caption(f"Framework: {selected_spec.framework} · Input: {selected_spec.input_type}")

    uploaded = st.file_uploader(
        "Upload audio", type=[extension.lstrip(".") for extension in config.SUPPORTED_AUDIO_EXTENSIONS]
    )
    if uploaded is None:
        return
    st.audio(uploaded)
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.getvalue())
            temporary_path = handle.name
        probability = float(detector.predict_scores([temporary_path])[0])
        result = classify_probability(probability, threshold)
        display_audio = load_audio(temporary_path)
    except Exception as exc:
        st.error(f"Could not process audio: {exc}")
        return
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    waveform_column, mel_column = st.columns(2)
    with waveform_column:
        figure, axis = plt.subplots(figsize=(7, 2.5))
        axis.plot([index / config.SAMPLE_RATE for index in range(len(display_audio))], display_audio, linewidth=0.6)
        axis.set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
        figure.tight_layout(); st.pyplot(figure); plt.close(figure)
    with mel_column:
        mel_db = create_mel_spectrogram_db(segment_audio(display_audio)[0])
        figure, axis = plt.subplots(figsize=(7, 2.5))
        image = axis.imshow(mel_db, aspect="auto", origin="lower", extent=[0, config.SEGMENT_DURATION, 0, config.N_MELS])
        axis.set(xlabel="Time (s)", ylabel="Mel bin", title="Display Mel spectrogram")
        figure.colorbar(image, ax=axis, format="%+2.0f dB")
        figure.tight_layout(); st.pyplot(figure); plt.close(figure)
    color = "red" if result.prediction == config.FAKE_NAME else "green"
    st.markdown(f"<h2 style='text-align:center;color:{color}'>Prediction: {result.prediction}</h2>", unsafe_allow_html=True)
    first, second, third = st.columns(3)
    first.metric("Confidence", f"{result.confidence * 100:.2f}%")
    second.metric("Raw P(FAKE)", f"{result.probability_fake:.4f}")
    third.metric("Threshold", f"{result.threshold:.4f}")


if __name__ == "__main__":
    main()
