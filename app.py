"""Responsive artifact-gated dashboard for trained LAVA detectors."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import config
from src.lava.artifacts import artifact_diagnostics, load_threshold
from src.lava.registry import create, get_spec, specs
from src.lava.score_semantics import classify_probability
from src.preprocessing import create_mel_spectrogram_db, load_audio, segment_audio


st.set_page_config(page_title="LAVA Voice Detection", layout="wide", initial_sidebar_state="auto")
SHOW_DETECTOR_DIAGNOSTICS = os.getenv("LAVA_SHOW_DETECTOR_DIAGNOSTICS", "0").strip().lower() in {
    "1", "true", "yes", "on",
}


def inject_styles() -> None:
    stylesheet = Path(__file__).resolve().parent / "assets" / "app.css"
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def load_detector(model_name: str):
    """Keep one runtime resident to stay within Community Cloud memory limits."""
    cached = st.session_state.get("_lava_loaded_detector")
    if cached is not None and cached[0] == model_name:
        return cached[1], cached[2]
    detector = create(model_name)
    detector.load()
    threshold = load_threshold(detector.spec)
    st.session_state["_lava_loaded_detector"] = (model_name, detector, threshold)
    return detector, threshold


def available_specs():
    available, rejected = [], {}
    for spec in specs():
        issues = artifact_diagnostics(spec)
        if issues:
            rejected[spec.name] = issues
        else:
            available.append(spec)
    for name, reasons in rejected.items():
        if SHOW_DETECTOR_DIAGNOSTICS:
            print(f"[LAVA artifact probe] {name}: {'; '.join(reasons)}")
    return available, rejected


def configure_plot(axis) -> None:
    axis.set_facecolor("#f5f9ff")
    axis.grid(axis="y", color="#dbeafe", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#93c5fd")
    axis.tick_params(colors="#334155", labelsize=8)
    axis.xaxis.label.set_color("#1d4ed8")
    axis.yaxis.label.set_color("#1d4ed8")
    axis.title.set_color("#172554")


def render_pipeline(input_type: str) -> None:
    source = "Waveform 16 kHz" if input_type == "waveform" else "Audio 22.05 kHz"
    representation = "Native front end" if input_type == "waveform" else "6 Mel segments"
    nodes = (source, "Normalize", representation, "Detector", "P(FAKE)")
    pieces = []
    for index, node in enumerate(nodes):
        class_name = "pipeline-node primary" if index == len(nodes) - 1 else "pipeline-node"
        pieces.append(f'<div class="{class_name}">{html.escape(node)}</div>')
        if index < len(nodes) - 1:
            pieces.append('<div class="pipeline-arrow">&#8594;</div>')
    st.markdown(f'<div class="pipeline">{"".join(pieces)}</div>', unsafe_allow_html=True)


def render_waveform(audio: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.1), facecolor="#f5f9ff")
    timeline = np.arange(audio.size, dtype=np.float32) / config.SAMPLE_RATE
    axis.plot(timeline, audio, color="#2563eb", linewidth=0.8)
    axis.fill_between(timeline, audio, 0, color="#60a5fa", alpha=0.28)
    axis.set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform", xlim=(0, config.AUDIO_DURATION))
    configure_plot(axis)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_full_mel(segments: np.ndarray) -> None:
    mel_db = np.concatenate([create_mel_spectrogram_db(segment) for segment in segments], axis=1)
    figure, axis = plt.subplots(figsize=(8, 3.1), facecolor="#f5f9ff")
    image = axis.imshow(
        mel_db, aspect="auto", origin="lower", cmap="magma",
        extent=[0, config.AUDIO_DURATION, 0, config.N_MELS], vmin=-config.TOP_DB, vmax=0,
    )
    axis.set(xlabel="Time (s)", ylabel="Mel bin", title="Full-audio Mel spectrogram")
    configure_plot(axis)
    colorbar = figure.colorbar(image, ax=axis, format="%+2.0f dB", pad=0.02)
    colorbar.ax.tick_params(colors="#334155", labelsize=8)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_segment_profile(segments: np.ndarray) -> None:
    rms = np.sqrt(np.mean(np.square(segments), axis=1))
    peak = np.max(np.abs(segments), axis=1)
    positions = np.arange(1, len(segments) + 1)
    figure, axis = plt.subplots(figsize=(8, 3.1), facecolor="#f5f9ff")
    axis.bar(positions - 0.16, rms, width=0.32, color="#7c3aed", label="RMS energy")
    axis.bar(positions + 0.16, peak, width=0.32, color="#f97316", label="Peak amplitude")
    axis.set(
        xlabel="Chronological segment", ylabel="Normalized amplitude",
        title="Temporal segment profile", xticks=positions,
    )
    axis.legend(frameon=False, fontsize=8)
    configure_plot(axis)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_probability(probability_fake: float, threshold: float) -> None:
    probabilities = [1.0 - probability_fake, probability_fake]
    figure, axis = plt.subplots(figsize=(8, 3.1), facecolor="#f5f9ff")
    bars = axis.barh(["REAL", "FAKE"], probabilities, color=["#14b8a6", "#f04452"], height=0.45)
    axis.axvline(threshold, color="#f59e0b", linestyle="--", linewidth=1.5, label="Decision threshold")
    axis.set(xlabel="Probability", xlim=(0, 1), title="Decision distribution")
    for bar, value in zip(bars, probabilities):
        axis.text(min(value + 0.02, 0.94), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    configure_plot(axis)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_sidebar(candidates):
    st.sidebar.markdown('<div class="lava-wordmark">LAVA</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-label">Detector</div>', unsafe_allow_html=True)
    default_index = next((i for i, spec in enumerate(candidates) if spec.name == "mobilenetv3_lstm"), 0)
    selected_name = st.sidebar.selectbox(
        "Select model", [spec.name for spec in candidates], index=default_index,
        format_func=lambda name: get_spec(name).display_name, label_visibility="collapsed",
    )
    selected_spec = get_spec(selected_name)
    st.sidebar.markdown('<div class="sidebar-label">Dashboard</div>', unsafe_allow_html=True)
    show_analysis = st.sidebar.toggle("Show signal analysis", value=True)
    compact = st.sidebar.toggle("Compact charts", value=False)
    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-label">Runtime</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="sidebar-copy">{html.escape(selected_spec.framework)}<br>'
        f'Input: {html.escape(selected_spec.input_type)}<br>Score contract: P(FAKE)</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-label">Authors</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<ul class="author-list">'
        '<li>Phan Khắc Anh Tuấn</li>'
        '<li>Nguyễn Phương Chinh</li>'
        '<li>Lại Thành Đạt</li>'
        '<li>Nguyễn Tấn Khiêm</li>'
        '<li>Trương Thành Đạt</li>'
        '</ul>',
        unsafe_allow_html=True,
    )
    return selected_name, selected_spec, show_analysis, compact


def main() -> None:
    inject_styles()
    candidates, rejected = available_specs()
    if not candidates:
        st.markdown('<div class="hero-kicker">System status</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">No trained detector available.</h1>', unsafe_allow_html=True)
        with st.expander("Detector diagnostics", expanded=True):
            for name, reasons in rejected.items():
                st.markdown(f"**{name}**")
                for reason in reasons:
                    st.code(reason)
        return

    selected_name, selected_spec, show_analysis, compact = render_sidebar(candidates)
    if rejected and SHOW_DETECTOR_DIAGNOSTICS:
        with st.sidebar.expander("Unavailable detectors"):
            for name, reasons in rejected.items():
                st.write(name)
                for reason in reasons:
                    st.code(reason)

    st.markdown('<div class="hero-kicker">Audio authenticity intelligence</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Deepfake voice detection, clearly explained.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-copy">Upload an audio file to inspect its authenticity score, temporal '
        'structure and spectral profile through a single deployment dashboard.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<h2 class="section-heading">Analysis pipeline</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">One score contract across TensorFlow and reference detectors: '
        'REAL = 0, FAKE = 1.</p>', unsafe_allow_html=True,
    )
    render_pipeline(selected_spec.input_type)

    st.markdown('<h2 class="section-heading">Audio input</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-copy">Active detector: {html.escape(selected_spec.display_name)}</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload audio", type=[ext.lstrip(".") for ext in config.SUPPORTED_AUDIO_EXTENSIONS],
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.markdown(
            '<div class="footer-note">Supported formats: WAV, FLAC, MP3, OGG and M4A. '
            'Inference runs only after a file is provided.</div>', unsafe_allow_html=True,
        )
        return

    st.audio(uploaded)
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.getvalue())
            temporary_path = handle.name
        with st.spinner("Running detector and signal analysis..."):
            detector, threshold = load_detector(selected_name)
            probability = float(detector.predict_scores([temporary_path])[0])
            result = classify_probability(probability, threshold)
            display_audio = load_audio(temporary_path)
    except Exception as exc:
        st.error(f"Could not process audio: {exc}")
        return
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    st.markdown(
        '<div class="result-panel"><div><div class="result-eyebrow">Detection result</div>'
        f'<div class="result-value">{html.escape(result.prediction)}</div></div>'
        f'<div class="result-model">{html.escape(selected_spec.display_name)}<br>'
        f'{html.escape(uploaded.name)}</div></div>', unsafe_allow_html=True,
    )
    first, second, third, fourth = st.columns(4)
    first.metric("Confidence", f"{result.confidence * 100:.2f}%")
    second.metric("P(FAKE)", f"{result.probability_fake:.4f}")
    third.metric("Threshold", f"{result.threshold:.4f}")
    fourth.metric("Normalized audio", f"{config.AUDIO_DURATION:.1f} s")

    st.markdown('<h2 class="section-heading">Decision dashboard</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">The threshold is calibrated per detector; ROC-oriented analysis '
        'should always use the raw P(FAKE) score.</p>', unsafe_allow_html=True,
    )
    render_probability(result.probability_fake, result.threshold)

    if show_analysis:
        st.markdown('<h2 class="section-heading">Signal analysis</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Visualization uses the shared three-second LAVA display '
            'policy and does not alter detector inference.</p>', unsafe_allow_html=True,
        )
        segments = segment_audio(display_audio)
        if compact:
            render_waveform(display_audio)
            render_full_mel(segments)
            render_segment_profile(segments)
        else:
            waveform_column, mel_column = st.columns(2)
            with waveform_column:
                render_waveform(display_audio)
            with mel_column:
                render_full_mel(segments)
            segment_column, detail_column = st.columns(2)
            with segment_column:
                render_segment_profile(segments)
            with detail_column:
                st.markdown(
                    '<div class="panel"><strong>Signal contract</strong><br><br>'
                    f'{config.SAMPLE_RATE:,} Hz visualization rate<br>'
                    f'{config.NUM_SEGMENTS} chronological segments<br>'
                    f'{config.N_MELS} Mel bands<br>'
                    f'{config.IMAGE_SIZE[0]} × {config.IMAGE_SIZE[1]} model images<br><br>'
                    '<span style="color:#6e6e73">Charts describe the input signal. They are not '
                    'separate classifier outputs.</span></div>', unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="footer-note">LAVA reports P(FAKE) under a shared score contract. '
        'A prediction is model evidence, not proof of authorship or intent.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
