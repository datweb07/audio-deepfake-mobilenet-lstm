"""Bounded Phase 9 comparison. Uses development inputs and never opens reserved roles."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.jaes_v2.phase6_experiment import resource_lock
from src.lava.jaes_v2.c2_student import peak_process_mib
from src.lava.jaes_v2.evidence import file_hash
from src.lava.jaes_v2.phase8_objectives import bce, weighted_sum
from src.lava.jaes_v2.phase8_training import (
    Inputs, code_contract, dataset, initialize, load_config, model_factory,
    segmentwise_gradient, segmentwise_inference, variable_hash,
)
from src.lava.jaes_v2.phase9_protocol import pointwise_backbone

ROOT = Path(__file__).resolve().parents[2]
ID = 'p8_medium_erm_partial_s6107_v1'
RUN = ROOT / 'outputs/jaes_v2/phase8/training' / ID


def difference(first, second):
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Shape or finite-value parity failed')
    scale = max(np.linalg.norm(a.ravel()), np.linalg.norm(b.ravel()))
    product = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
    return {
        'shape': list(a.shape), 'finite': True, 'max_abs': float(np.max(np.abs(a - b))),
        'relative_l2': float(np.linalg.norm((a - b).ravel()) / scale) if scale else 0.,
        'cosine': float(np.dot(a.ravel(), b.ravel()) / product) if product else None,
    }


def probe():
    start = perf_counter()
    tf = initialize('cpu')
    cfg = load_config(ROOT, RUN / 'resolved_config.json')
    train, dev, b_ids, _ = dataset(ROOT, cfg)
    model, backbone, forward = model_factory(tf, cfg)
    warmup = RUN / 'best_epoch_001.keras'
    model.load_weights(warmup)
    variables = tuple(model.trainable_weights)
    initial = model.get_weights()
    initial_trainable = [variable.numpy().copy() for variable in variables]
    frozen = model.non_trainable_weights
    frozen_hash = variable_hash(frozen)
    layer_methods = [getattr(layer, 'convolution_op', None) for layer in backbone.layers]
    vjp_view = pointwise_backbone(tf, backbone)
    inputs = Inputs(ROOT, cfg)

    def objective(x, y, coeff, counter, clean, paired, *, embeddings=False):
        p = forward(x, True, counter, 0, embeddings=embeddings)
        loss = weighted_sum(y, p, coeff)
        if paired:
            q = forward(clean, True, counter, 1, embeddings=embeddings)
            loss = .5 * (loss + weighted_sum(y, q, coeff))
        return loss, bce(y, p)

    @tf.function(reduce_retracing=True)
    def standard(x, y, coeff, counter, clean, paired):
        with tf.GradientTape() as tape:
            loss, per_example = objective(x, y, coeff, counter, clean, paired)
        return loss, tape.gradient(loss, variables), per_example

    paths = {
        'full': standard,
        'segmentwise': segmentwise_gradient(tf, backbone, variables, objective),
        'pointwise': segmentwise_gradient(tf, vjp_view, variables, objective),
    }
    optimizer = tf.keras.optimizers.Adam(cfg.finetune_learning_rate)
    optimizer.build(variables)
    optimizer_initial = [v.numpy().copy() for v in optimizer.variables()]
    startup_seconds = perf_counter() - start
    cases = []
    primary = [record for record in dev if record.sample_id not in b_ids]
    for label, condition in ((0, 'clean'), (1, 'opus64')):
        record = next(record for record in primary if record.label == label)
        t = perf_counter()
        x = tf.convert_to_tensor(inputs.get(record, condition)[None])
        preparation = perf_counter() - t
        args = (x, tf.constant([float(label)]), tf.constant([1.]), tf.constant(104 + label), x, False)
        measurements = {}
        for name, gradient in paths.items():
            model.set_weights(initial)
            for variable, value in zip(optimizer.variables(), optimizer_initial):
                variable.assign(value)
            t = perf_counter()
            value, gradients, _ = gradient(*args)
            arrays = [g.numpy() for g in gradients]
            gradient_seconds = perf_counter() - t
            optimizer.apply_gradients(zip(gradients, variables))
            weights = [v.numpy() for v in variables]
            del gradients
            if variable_hash(frozen) != frozen_hash:
                raise ValueError('Frozen state drift')
            measurements[name] = (float(value), arrays, weights, gradient_seconds)
        ref = measurements['full']
        result = {}
        for name in ('segmentwise', 'pointwise'):
            value, gradients, weights, seconds = measurements[name]
            flat = lambda seq: np.concatenate([g.ravel() for g in seq])
            result[name] = {
                'loss_abs': abs(value - ref[0]), 'gradient_global': difference(flat(ref[1]), flat(gradients)),
                'gradient_global_norm': float(np.linalg.norm(flat(gradients))),
                'reference_global_norm': float(np.linalg.norm(flat(ref[1]))),
                'gradient_seconds_including_trace_if_first_call': seconds,
                'per_tensor': {v.name: {'gradient': difference(a, b), 'post_adam_weight': difference(p, q),
                                       'adam_update': difference(p - start, q - start)}
                               for v, a, b, p, q, start in zip(variables, ref[1], gradients, ref[2], weights,
                                                             initial_trainable)},
            }
        cases.append({'source': record.sample_id, 'label': label, 'condition': condition, 'preparation_seconds': preparation,
                      'full_loss': ref[0], 'full_gradient_seconds': ref[3], 'comparisons': result})
    model.set_weights(initial)
    segmented = segmentwise_inference(tf, model)
    compiled = tf.function(lambda x: forward(x, training=False), reduce_retracing=True)
    # Same input tensors, compilation warmup excluded from timed calls.
    views = [inputs.get(r, 'clean')[None] for r in primary[:4]]
    for function in (compiled, segmented):
        function(views[0]).numpy()
    validation = {}
    predictions = {}
    for name, function in (('full_compiled', compiled), ('segmentwise', segmented)):
        t = perf_counter()
        predictions[name] = [float(function(x).numpy().reshape(-1)[0]) for x in views]
        validation[name] = {'forward_seconds': perf_counter() - t, 'scores': predictions[name]}
    validation['difference'] = difference(predictions['full_compiled'], predictions['segmentwise'])
    validation['threshold_flips'] = sum((a >= .5) != (b >= .5) for a, b in zip(*predictions.values()))
    if [getattr(layer, 'convolution_op', None) for layer in backbone.layers] != layer_methods:
        raise ValueError('Original convolution methods changed')
    return {'scope': 'bounded two-case arithmetic and four-view timing', 'warmup_sha256': file_hash(warmup),
            'tf': tf.__version__, 'manifest_sha256': cfg.manifest_sha256, 'code_contract': code_contract(ROOT),
            'prospective_source_sha256': file_hash(ROOT/'src/lava/jaes_v2/phase9_protocol.py'),
            'audit_script_sha256': file_hash(Path(__file__)), 'startup_and_dataset_seconds': startup_seconds,
            'frozen_unchanged': True, 'original_layer_methods_unchanged': True,
            'cases': cases, 'validation': validation, 'input_timing': dict(inputs.timings),
            'elapsed_seconds': perf_counter() - start, 'peak_process_mib': peak_process_mib(),
            'calibration_used': False, 'official_evaluation_used': False, 'registered_training_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs/jaes_v2/phase9'):
        parser.error('Output must be under outputs/jaes_v2/phase9')
    if output.exists():
        raise FileExistsError(output)
    tmp = ROOT / '.research_tmp'
    tmp.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(tmp), TMP=str(tmp))
    with resource_lock(ROOT):
        result = probe()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(output)


if __name__ == '__main__':
    main()
