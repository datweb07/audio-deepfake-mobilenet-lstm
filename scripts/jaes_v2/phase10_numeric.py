"""Acceptance fixtures for the real Phase 10 model and ASVspoof inputs."""
from time import perf_counter

import numpy as np

from src.lava.jaes_v2 import phase10_training as p10
from src.lava.jaes_v2 import phase8_training as prior
from src.lava.jaes_v2.phase10_resources import require_resources, snapshot


def numerical_diff(ref, candidate, tolerance):
    a, b = np.asarray(ref, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Invalid numerical parity tensor')
    delta = np.abs(b - a)
    limit = tolerance['atol'] + tolerance['rtol'] * np.abs(a)
    return dict(shape=list(a.shape), max_abs=float(delta.max()),
                pass_thresholds=bool((delta <= limit).all()))


def numeric_parity(root, tolerance, contract):
    require_resources(root)
    start = perf_counter()
    tf = p10.phase10_tensorflow()
    c = p10.qualification_config(root, 'lifecycle')
    _, dev, _, _ = prior.dataset(root, c)
    real = next(r for r in dev if r.label == 0)
    fake = next(r for r in dev if r.label == 1)
    inputs = prior.Inputs(root, c)
    model, backbone, forward = prior.model_factory(tf, c)
    warmup, partial, variables, cnn, objective = p10.kernel(tf, backbone, model, forward)
    frozen = [v for v in model.weights if id(v) not in {id(v) for v in variables}]
    frozen_hash = prior.variable_hash(frozen)
    optimizer = tf.keras.optimizers.Adam(c.learning_rate)
    optimizer.build(variables)
    slots = tuple(optimizer.variables())

    @tf.function(reduce_retracing=True)
    def standard(x, y, coeff, counter):
        with tf.GradientTape() as tape:
            loss, _ = objective(x, y, coeff, counter, x, False)
        return loss, tuple(tape.gradient(loss, variables))

    full_infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    segmented = prior.segmentwise_inference(tf, model)
    fixtures = (
        ((real, 'clean'),), ((fake, 'clean'),), ((real, 'noise5'),), ((fake, 'noise5'),),
        ((real, 'opus64'),), ((fake, 'A_then_B'),), ((real, 'B_then_A'),),
        ((real, 'clean'), (fake, 'noise5')), ((fake, 'opus64'), (real, 'A_then_B')),
    )
    evidence = []
    for index, fixture in enumerate(fixtures):
        require_resources(root)
        is_partial = index >= 2
        active = [i for i, v in enumerate(variables) if is_partial or id(v) not in cnn]
        optimizer.learning_rate.assign(c.finetune_learning_rate if is_partial else c.learning_rate)
        x = np.stack([inputs.get(r, condition) for r, condition in fixture])
        y = tf.constant([r.label for r, _ in fixture])
        coeff = tf.constant([1 / len(fixture)], tf.float32)
        originals = [v.numpy() for v in variables]
        original_slots = [v.numpy() for v in slots]
        # For two-example fixtures exercise both true mixed inference and accumulated
        # training gradients with the same stateless dropout micro-counters.
        paths = {}
        for name, fn in (('standard', standard), ('candidate', partial if is_partial else warmup)):
            for variable, old in zip((*variables, *slots), (*originals, *original_slots)):
                variable.assign(old)
            accum = [tf.zeros_like(v) for v in variables]
            loss = 0.
            for micro in range(len(fixture)):
                value, gradients = fn(x[micro:micro + 1], y[micro:micro + 1], coeff,
                                      tf.constant(index * 2 + micro))
                accum = [a + g for a, g in zip(accum, gradients)]
                loss += float(value.numpy())
            optimizer.apply_gradients((accum[i], variables[i]) for i in active)
            paths[name] = dict(loss=loss, gradient=[accum[i].numpy() for i in active],
                               optimizer=[v.numpy() for v in slots], weight=[v.numpy() for v in variables],
                               update=[v.numpy() - old for v, old in zip(variables, originals)],
                               score=(full_infer(x) if name == 'standard' else segmented(x)).numpy())
        if prior.variable_hash(frozen) != frozen_hash:
            raise ValueError('Frozen tensor changed')
        first, second = paths['standard'], paths['candidate']
        losses = numerical_diff(first['loss'], second['loss'], tolerance['loss'])
        scores = numerical_diff(first['score'], second['score'], tolerance['score'])
        tensors = {}
        for category, names in (
            ('gradient', [variables[i].name for i in active]),
            ('optimizer', [v.name for v in slots]),
            ('weight', [v.name for v in variables]), ('update', [v.name for v in variables]),
        ):
            tensors[category] = {name: numerical_diff(a, b, tolerance[category])
                                 for name, a, b in zip(names, first[category], second[category], strict=True)}
        passed = losses['pass_thresholds'] and scores['pass_thresholds'] and all(
            t['pass_thresholds'] for group in tensors.values() for t in group.values())
        evidence.append(dict(stage='partial' if is_partial else 'warmup',
                             views=[[r.sample_id, condition, r.label] for r, condition in fixture],
                             loss=losses, score=scores, tensors=tensors, pass_thresholds=passed))
    return dict(**contract, pass_thresholds=all(item['pass_thresholds'] for item in evidence),
                fixtures=evidence, frozen_unchanged=True, elapsed_seconds=perf_counter() - start,
                resources=snapshot(root))
