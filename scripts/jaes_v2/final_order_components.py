"""Inference-only reconstruction of the sealed Phase 10 order components."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from scipy.signal import correlate, correlation_lags

from scripts.jaes_v2.phase6_experiment import resource_lock
from scripts.jaes_v2.phase10_medium import registered
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.controls import NoisePlan
from src.lava.jaes_v2.c2_data import paired_views
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv, write_csv, write_json
from src.lava.jaes_v2.phase10_resources import require_resources, snapshot
from src.lava.jaes_v2.preprocessing import Preprocessor
from src.lava.jaes_v2.transforms import Transform, diagnostics, load_audio, measured_snr

BASE = Path('outputs/jaes_v2/phase10')
OUT = BASE / 'final_diagnostic_v1'
METHODS = ('erm', 'channel')
COMPONENTS = ('noise10', 'opus24')
ORDERS = ('A_then_B', 'B_then_A')


def views(audio, record, ffmpeg):
    if record.split != 'development':
        raise ValueError('Development-only diagnostic')
    existing, evidence = paired_views(audio, record.sample_id, 'development', ffmpeg)
    clean = existing['clean']
    noise = NoisePlan.build(clean, target_snr_db=10, seed=evidence['sample_seed'])
    noisy, noise_evidence = noise.insert(clean, 'fixed_absolute')
    codec = Transform('opus24', 'codec', {'encoder': 'libopus', 'bitrate': '24k', 'extension': '.ogg'})
    coded, codec_evidence = codec.apply(clean, seed=0, ffmpeg=ffmpeg)
    reverse, reverse_evidence = noise.insert(coded, 'fixed_absolute')
    if not np.array_equal(reverse.samples, existing['B_then_A'].samples):
        raise ValueError('Standalone codec is inconsistent with registered reverse order')
    forward, _ = codec.apply(noisy, seed=0, ffmpeg=ffmpeg)
    if not np.array_equal(forward.samples, existing['A_then_B'].samples):
        raise ValueError('Standalone noise is inconsistent with registered forward order')
    measured = dict(noise=noise_evidence, codec=codec_evidence, reverse=reverse_evidence,
                    forward_codec=evidence['conditions']['A_then_B'],
                    forward_residual_snr_db=measured_snr(coded.samples, forward.samples))
    return dict(existing, noise10=noisy, opus24=coded), measured


def nuisance(clean, changed):
    info = diagnostics(changed)
    ref, values = clean.samples.astype(float), changed.samples.astype(float)
    spectrum = np.abs(np.fft.rfft(values)) ** 2
    corr = correlate(values, ref, method='fft')
    lags = correlation_lags(len(values), len(ref))
    allowed = abs(lags) <= 441
    info.update(clipping_proxy_fraction=float(np.mean(abs(values) >= 1)),
                energy=float(np.dot(values, values)),
                bandwidth95_hz=float(np.searchsorted(np.cumsum(spectrum), .95 * sum(spectrum))
                                     * changed.sample_rate / len(values)),
                alignment_lag_samples=int(lags[allowed][np.argmax(corr[allowed])]),
                relative_waveform_rms=float(np.sqrt(np.mean((values - ref) ** 2))))
    return info


def run(root, *, seed=9107, output=OUT):
    before = require_resources(root)
    out = root / output
    target = out / 'matched_components.json'
    if target.exists():
        raise FileExistsError(target)
    start = perf_counter()
    paths = {m: root / BASE / f'p10_medium_{m}_s{seed}_v2/scientific' for m in METHODS}
    _, config = registered(f'p10_medium_erm_s{seed}_v2')
    train, development, b_ids, _ = prior.dataset(root, config)
    assert config.source_manifest == config.development_manifest == prior.MEDIUM
    digest = object_hash([row.to_dict() for row in train + development])
    anchors = {}
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise ValueError('Missing registered FFmpeg')
    current_source = engine.fingerprints(root)
    for method, path in paths.items():
        prior.verify_tree(path)
        result = engine.read(path / 'result.json')
        state = result['state']
        if (state['contract']['actual_sources'] != digest or state['contract']['source'] != current_source
                or state['step'] != 840 or state['epoch'] != 21):
            raise ValueError('Sealed source, code or budget mismatch')
        selected = path / 'checkpoints' / state['best']['tag']
        prior.verify_tree(selected)
        if engine.read(selected / 'controller.json')['model_hash'] != result['selected_weights']:
            raise ValueError('Selected checkpoint weights mismatch')
        anchors[method] = {(r['sample_id'], r['condition_id']): float(r['raw_score'])
                           for r in read_csv(path / 'scores.csv')}
    tf = engine.phase10_tensorflow()
    models, footprint = {}, {}
    for method, path in paths.items():
        started = perf_counter()
        model = tf.keras.models.load_model(path / 'selected.keras', compile=False)
        restore_seconds = perf_counter() - started
        if prior.variable_hash(model.weights) != engine.read(path / 'result.json')['selected_weights']:
            raise ValueError('Restored model weights mismatch')
        models[method] = prior.segmentwise_inference(tf, model)
        backbone = model.get_layer('time_distributed_mobilenetv3small').layer
        backbone_ids = {id(weight) for weight in backbone.weights}
        footprint[method] = dict(total=model.count_params(),
            partial_trainable=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
            warmup_trainable=sum(int(np.prod(w.shape)) for w in model.trainable_weights
                                 if id(w) not in backbone_ids),
            nontrainable=sum(int(np.prod(w.shape)) for w in model.non_trainable_weights),
            serialized_bytes=(path / 'selected.keras').stat().st_size, restore_seconds=restore_seconds)
    preprocessor = Preprocessor('P0')
    predictions, waveform_rows = [], []
    smoke = [next(r for r in development if r.label == label and r.sample_id not in b_ids) for label in (0, 1)]
    smoke.append(next(r for r in development if r.sample_id in b_ids))
    anchor_errors = []

    def inputs(record):
        require_resources(root)
        audio = load_audio(prior.source_path(root, record.path), sample_rate=22050, max_seconds=3)
        return views(audio, record, ffmpeg)

    def predict(method, tensor):
        score = float(models[method](tensor).numpy().reshape(-1)[0])
        if not 0 <= score <= 1:
            raise ValueError('Nonfinite or out-of-range prediction')
        return score

    for record in smoke:
        generated, _ = inputs(record)
        for condition in ('clean', *ORDERS):
            tensor = preprocessor.extract(generated[condition].samples)[None]
            for method in METHODS:
                original = anchors[method][record.sample_id, condition]
                score = predict(method, tensor)
                if abs(score - original) > 3e-5 or (score >= .5) != (original >= .5):
                    raise ValueError('Sealed score anchor replay mismatch')
                anchor_errors.append(abs(score - original))
        for condition in COMPONENTS:
            tensor = preprocessor.extract(generated[condition].samples)[None]
            for method in METHODS:
                predictions.append(dict(method=method, sample_id=record.sample_id, condition_id=condition,
                                        raw_score=predict(method, tensor)))

    if seed != 9107:
        write_json(out / 'anchor_replay.json', dict(seed=seed, scores=len(anchor_errors),
                   maximum_error=max(anchor_errors), expected_component_predictions=4*len(development)))
        import platform
        times = []
        for repeat in range(3):
            for record in development[:12]:
                require_resources(root)
                started = perf_counter()
                audio = load_audio(prior.source_path(root, record.path), sample_rate=22050, max_seconds=3)
                tensor = preprocessor.extract(audio.samples)[None]
                ready = perf_counter()
                predict('erm', tensor)
                finished = perf_counter()
                times.append(dict(sample_id=record.sample_id, repeat=repeat,
                                  waveform_seconds=len(audio.samples)/audio.sample_rate,
                                  preprocessing_seconds=ready-started, forward_seconds=finished-ready,
                                  end_to_end_seconds=finished-started))
        write_json(out / 'efficiency.json', dict(cpu=platform.processor(), system=platform.platform(),
                   tensorflow=tf.__version__, batch=1, threads=1, warmup='anchor replay before timing',
                   snapshot=snapshot(root), models=footprint, timing=times))

    for index, record in enumerate(development, 1):
        generated, metadata = inputs(record)
        identity = dict(sample_id=record.sample_id, parent_recording_id=record.parent_recording_id,
                        speaker_id=record.speaker_id, attack_id=record.attack_id, label=record.label,
                        stratum='B_REAL_ONLY' if record.sample_id in b_ids else 'A_PAIRED')
        waveform_rows.append(dict(identity, metadata=metadata,
                                  conditions={c: nuisance(generated['clean'], generated[c])
                                              for c in ('clean', *COMPONENTS, *ORDERS)}))
        if record not in smoke:
            for condition in COMPONENTS:
                tensor = preprocessor.extract(generated[condition].samples)[None]
                for method in METHODS:
                    predictions.append(dict(method=method, sample_id=record.sample_id, condition_id=condition,
                                            raw_score=predict(method, tensor)))
        if index % 40 == 0:
            print(f'Matched components {index}/{len(development)}', flush=True)
    if len(predictions) != len(development) * 4:
        raise ValueError('Incomplete component score grid')
    write_csv(out / 'component_scores.csv', predictions)
    write_json(target, dict(status='PASS', seed=seed, date=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
                            new_predictions=len(predictions), anchor_replays=len(anchor_errors),
                            anchor_max_abs=max(anchor_errors), cutoff=3e-5,
                            score_sha256=file_hash(out / 'component_scores.csv'), actual_sources=digest,
                            selected_sha256={m:file_hash(p / 'selected.keras') for m,p in paths.items()},
                            ffmpeg_version=subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0],
                            ffmpeg_sha256=file_hash(Path(ffmpeg)), preprocessor=preprocessor.fingerprint,
                            before=before, after=snapshot(root), elapsed_seconds=perf_counter()-start,
                            evaluation_access=False, calibration_access=False, training=False,
                            waveforms=waveform_rows))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(9107, 9207), default=9107)
    args = parser.parse_args()
    output = OUT if args.seed == 9107 else Path('outputs/jaes_v2/supervisor_major_revision/components_9207')
    root = Path.cwd()
    with resource_lock(root):
        out = root / output
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            tempfile.tempdir = temporary
            try:
                run(root, seed=args.seed, output=output)
            finally:
                tempfile.tempdir = None
