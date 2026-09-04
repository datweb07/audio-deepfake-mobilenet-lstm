"""Backend-specific inference only; shared benchmark orchestration stays model-independent."""
import platform
import time
import threading
import numpy as np
from benchmark.efficiency import _rss_mib
from src.lava.registry import create
from src.lava.score_semantics import validate_p_fake


class Runtime:
    def __init__(self, registry_name):
        self.detector = create(registry_name)
        self.external = registry_name.endswith("_pretrained")

    def load(self):
        self.rss_before = _rss_mib()
        start = time.perf_counter()
        self.detector.load()
        self.load_seconds = time.perf_counter() - start
        self.rss_after = _rss_mib()
        if self.external:
            import onnxruntime
            from src.lava.preprocessing.waveform import load_waveform
            self.version = onnxruntime.__version__
            self.preprocess = lambda path: load_waveform(path, sample_rate=16000, target_samples=64600)
            self.forward = self.detector._run_waveform
            self.input_shape, self.output_shape = [1, 64600], [1, 2]
            self.parameter_count_source = "checkpoint metadata (pending independent verification)"
        else:
            import tensorflow as tf
            from src.preprocessing import process_audio_file
            self.version = tf.__version__
            self.preprocess = lambda path: process_audio_file(path)[None]
            model = self.detector.model
            infer = tf.function(lambda x: model(x, training=False), input_signature=[tf.TensorSpec([1, 6, 224, 224, 3], tf.float32)])
            self.forward = lambda values: infer(tf.convert_to_tensor(values, dtype=tf.float32)).numpy().reshape(-1)
            self.input_shape, self.output_shape = list(model.input_shape), list(model.output_shape)
            self.parameter_count_source = "loaded Keras model.count_params (includes BN state)"

    def predict(self, path):
        return float(validate_p_fake(self.forward(self.preprocess(path))).reshape(-1)[0])

    def parameter_count(self):
        return self.detector.parameter_count()


def efficiency(runtime, path, duration, warmup=10, runs=50):
    peak = [_rss_mib()]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.02):
            value = _rss_mib()
            if value is not None:
                peak[0] = max(peak[0] or 0, value)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    def timed(call):
        for _ in range(warmup):
            call()
        values = []
        for _ in range(runs):
            start = time.perf_counter()
            call()
            values.append((time.perf_counter() - start) * 1000)
        return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                    std_ms=float(np.std(values)), p95_ms=float(np.percentile(values, 95)), runs_ms=values)
    try:
        features = runtime.preprocess(path)
        model = timed(lambda: runtime.forward(features))
        prep = timed(lambda: runtime.preprocess(path))
        end = timed(lambda: runtime.predict(path))
    finally:
        stop.set()
        monitor.join()
    return dict(status="BENCHMARKED", warmup=warmup, runs=runs, batch_size=1,
        device="CPU", threads=1, precision="float32", framework_version=runtime.version,
        model_only=model, preprocessing=prep, end_to_end=end, duration=duration,
        throughput=1000 / end["mean_ms"], rtf=end["mean_ms"] / (1000 * duration),
        load_seconds=runtime.load_seconds, rss_before_load_mb=runtime.rss_before,
        rss_after_load_mb=runtime.rss_after, peak_sampled_rss_mb=peak[0],
        memory_note="Whole worker process RSS sampled every 20ms; not model-only or exact allocator peak",
        timing_note="CPU synchronous graph/ONNX inference; startup, trace and warm-up excluded. ORT vs TF kernel difference remains.",
        hardware=dict(platform=platform.platform(), processor=platform.processor()), representative_file=path)
