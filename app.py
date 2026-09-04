"""Responsive artifact-gated dashboard for trained LAVA detectors."""

from __future__ import annotations

import html
import os
import tempfile
import io
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder

import config
from src.lava.artifacts import artifact_diagnostics, load_threshold
from src.lava.registry import create, get_spec, specs
from src.lava.benchmark_display import benchmark_card
from src.lava.score_semantics import classify_probability
from src.lava.decision_display import SCORE_NOTICE, decision_explanation, threshold_description
from src.lava.preprocessing.microphone import (
    MicrophoneQualityError,
    prepare_microphone_recording,
)
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

def render_interactive_plot(figure, height=320) -> None:
    """Chuyển đổi Matplotlib figure sang ảnh tương tác toàn màn hình (bứt phá iframe)"""
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches='tight', facecolor=figure.get_facecolor(), transparent=False)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    
    html_code = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/viewerjs/1.11.3/viewer.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/viewerjs/1.11.3/viewer.min.js"></script>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: transparent; }}
        #img-container {{ width: 100%; height: {height}px; display: flex; justify-content: center; align-items: center; }}
        img {{ max-width: 100%; max-height: 100%; object-fit: contain; cursor: zoom-in; border-radius: 4px; }}
    </style>
    
    <div id="img-container">
        <img id="interactive-img" src="data:image/png;base64,{img_b64}" alt="LAVA Signal Analysis">
    </div>
    
    <script>
        const img = document.getElementById('interactive-img');
        const frame = window.frameElement; 
        
        let origStyles = {{}};
        if (frame) {{
            origStyles = {{
                position: frame.style.position,
                zIndex: frame.style.zIndex,
                top: frame.style.top,
                left: frame.style.left,
                width: frame.style.width,
                height: frame.style.height,
            }};
        }}

        const viewer = new Viewer(img, {{
            inline: false,
            button: true,
            navbar: false,
            title: false,
            tooltip: true,
            movable: true,
            rotatable: true,
            scalable: true,
            zoomable: true,
            transition: false,
            toolbar: {{
                zoomIn: 1, zoomOut: 1, oneToOne: 1, reset: 1, 
                rotateLeft: 1, rotateRight: 1, flipHorizontal: 1, flipVertical: 1
            }},
        }});

        img.addEventListener('show', function () {{
            if (frame) {{
                frame.style.position = 'fixed';
                frame.style.zIndex = '99999999'; 
                frame.style.top = '0';
                frame.style.left = '0';
                frame.style.width = '100vw';
                frame.style.height = '100vh';
            }}
        }});

        img.addEventListener('hidden', function () {{
            if (frame) {{
                frame.style.position = origStyles.position;
                frame.style.zIndex = origStyles.zIndex;
                frame.style.top = origStyles.top;
                frame.style.left = origStyles.left;
                frame.style.width = origStyles.width;
                frame.style.height = origStyles.height;
            }}
        }});
    </script>
    """
    components.html(html_code, height=height)

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
    render_interactive_plot(figure)
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
    render_interactive_plot(figure)
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
    render_interactive_plot(figure)
    plt.close(figure)

def render_probability(probability_fake: float, threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.1), facecolor="#f5f9ff")
    axis.axvspan(0, threshold, color="#14b8a6", alpha=0.12, label="REAL: score < threshold")
    axis.axvspan(threshold, 1, color="#f04452", alpha=0.12, label="FAKE: score >= threshold")
    axis.axvline(threshold, color="#d97706", linestyle="--", linewidth=1.5,
                 label=f"Threshold {threshold:.4f}")
    axis.scatter([probability_fake], [0.5], color="#2563eb", s=100, zorder=4, clip_on=False)
    axis.annotate(f"Raw score {probability_fake:.4f}", (probability_fake, 0.5),
                  xytext=(0, 18), textcoords="offset points",
                  ha="left" if probability_fake < 0.15 else "right" if probability_fake > 0.85 else "center",
                  color="#1d4ed8", fontsize=10)
    axis.set(xlabel="Raw FAKE score (not calibrated confidence)", xlim=(0, 1),
             ylim=(0, 1), yticks=[], title="Score against decision threshold")
    axis.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.25))
    configure_plot(axis)
    figure.tight_layout()
    render_interactive_plot(figure)
    plt.close(figure)

def render_sidebar(candidates):
    st.sidebar.markdown('<div class="lava-wordmark">LAVA</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-label">Detector</div>', unsafe_allow_html=True)
    default_index = next((i for i, spec in enumerate(candidates) if spec.name == "mobilenetv3_lstm"), 0)
    detector_names = {spec.display_name: spec.name for spec in candidates}
    selected_display_name = st.sidebar.selectbox(
        "Select model", list(detector_names), index=default_index, label_visibility="collapsed",
    )
    selected_name = detector_names[selected_display_name]
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
    measured = benchmark_card(selected_spec)
    if measured is not None:
        with st.sidebar.expander("LAVA-5 measured benchmark", expanded=False):
            st.caption(f"Canonical test: {measured['TestSamples']} samples. {measured['Provenance']}.")
            st.write(f"F1: {float(measured['CleanF1']):.4f} | AUC: {float(measured['AUC']):.4f} | EER: {float(measured['EER']):.4f}")
            if measured.get("LatencyMeanMs") and measured.get("RTF"):
                st.write(f"Benchmark CPU latency: {float(measured['LatencyMeanMs']):.1f} ms | RTF: {float(measured['RTF']):.3f}")
            st.caption("Measured on the benchmark machine, not this browser/server. Training provenance differs between models.")
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
        '<p class="hero-copy">Upload or record audio to inspect its authenticity score, temporal '
        'structure and spectral profile through a single deployment dashboard.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<h2 class="section-heading">Analysis pipeline</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">One score contract across TensorFlow and reference detectors: '
        'REAL = 0, FAKE = 1.</p>', unsafe_allow_html=True,
    )
    render_pipeline(selected_spec.input_type)

    st.markdown(
        f'<div style="display: flex; align-items: baseline; justify-content: space-between; margin: 2.7rem 0 1rem 0;">'
        f'<h2 class="section-heading" style="margin: 0;">Audio input</h2>'
        f'<span class="section-copy" style="margin: 0;">Active detector: {html.escape(selected_spec.display_name)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "Input method",
        ("Upload file", "Record microphone"),
        horizontal=True,
        label_visibility="collapsed",
    )
    audio_bytes: bytes | None = None
    audio_name = ""
    audio_format = "audio/wav"
    microphone_quality = None

    if input_method == "Upload file":
        uploaded = st.file_uploader(
            "Upload audio", type=[ext.lstrip(".") for ext in config.SUPPORTED_AUDIO_EXTENSIONS],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()
            audio_name = uploaded.name
            audio_format = uploaded.type or "audio/wav"
    else:
        st.markdown(
            '<div class="panel" style="text-align: center;">'
            '<strong>Microphone capture</strong><br>'
            '<span style="color: var(--muted); font-size: 0.9rem;">'
            'Speak one continuous sentence for 3–5 seconds. Use a quiet room, keep a stable distance from the microphone and stop the recording after speaking. The capture must pass quality checks before inference.</span></div>',
            unsafe_allow_html=True,
        )
        
        st.markdown('<div style="display: flex; justify-content: center; margin: 2.5rem 0;">', unsafe_allow_html=True)
        raw_audio_bytes = audio_recorder(
            text="",  
            recording_color="#f04452", 
            neutral_color="#14b8a6",   
            icon_name="microphone",
            icon_size="3x",
            key="lava_microphone_recorder"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if raw_audio_bytes:
            try:
                prepared = prepare_microphone_recording(raw_audio_bytes)
            except MicrophoneQualityError as exc:
                st.warning(str(exc))
                return
            audio_bytes = prepared.wav_bytes
            microphone_quality = prepared.quality
            audio_name = "Live microphone capture"
            audio_format = "audio/wav"

    if audio_bytes is None:
        guidance = (
            "Supported formats: WAV, FLAC, MP3, OGG and M4A."
            if input_method == "Upload file"
            else "Microphone access requires localhost or an HTTPS deployment."
        )
        st.markdown(
            f'<div class="footer-note">{guidance} Inference runs only after audio is provided.</div>',
            unsafe_allow_html=True,
        )
        return

    st.audio(audio_bytes, format=audio_format)
    if microphone_quality is not None:
        quality_one, quality_two, quality_three, quality_four = st.columns(4)
        quality_one.metric("Captured", f"{microphone_quality.original_duration:.1f} s")
        quality_two.metric("Speech retained", f"{microphone_quality.retained_duration:.1f} s")
        quality_three.metric("Signal level", f"{microphone_quality.rms_dbfs:.1f} dBFS")
        quality_four.metric("Clipping", f"{microphone_quality.clipping_ratio * 100:.2f}%")
        st.caption(
            "Capture quality passed. No gain normalization, denoising or voice transformation was applied."
        )

    suffix = os.path.splitext(audio_name)[1].lower() or ".wav"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio_bytes)
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
        f'{html.escape(audio_name)}</div></div>', unsafe_allow_html=True,
    )
    first, second, third = st.columns(3)
    first.metric("Raw FAKE score", f"{result.probability_fake:.4f}", help=SCORE_NOTICE)
    second.metric("Decision threshold", f"{result.threshold:.4f}")
    third.metric("Model input", f"{selected_spec.audio_duration:.1f} s")
    st.caption(decision_explanation(result))

    st.markdown('<h2 class="section-heading">Decision dashboard</h2>', unsafe_allow_html=True)
    st.caption(threshold_description(selected_spec))
    st.caption(SCORE_NOTICE)
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
        'A microphone recording can still produce a false positive because devices, rooms and '
        'codecs differ from training data. A prediction is model evidence, not proof of authorship '
        'or intent.</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
