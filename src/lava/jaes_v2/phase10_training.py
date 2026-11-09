"""Fixed-backend prospective qualification. Historical Phase 8 is read-only."""
from collections import Counter
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import platform
from importlib.metadata import version
from time import perf_counter

import numpy as np

from . import phase8_training as prior
from .evidence import file_hash, object_hash, write_csv, write_json
from .phase8_objectives import bce
from .phase10_resources import ResourceStop, require_resources, snapshot
from .phase10_validation import MARGINS, guard_counts, metrics, score, select

BASE = Path('outputs/jaes_v2/phase10')
BACKEND = 'FULL_SHAPE_RECOMPUTE'
CAPS = {'A_clean_FA': .10, 'A_clean_miss': .15}
LEGACY_CONFIG = Path('configs/jaes_v2/phase8/p8_medium_channel_partial_s6107_v1.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def strict_controller(directory, contract):
    prior.verify_tree(directory)
    state = read(directory / 'controller.json')
    if state['contract'] != contract:
        raise ValueError('Resume contract mismatch')
    return state


def committed_controller(out, contract):
    pointer = read(out / 'latest.json')
    directory = prior.safe_path(out / 'checkpoints', pointer['checkpoint'])
    if file_hash(directory / 'controller.json') != pointer['sha256']:
        raise ValueError('Checkpoint pointer checksum mismatch')
    return directory, strict_controller(directory, contract)


def fingerprints(root):
    paths = ('src/lava/jaes_v2/phase10_training.py', 'src/lava/jaes_v2/phase10_validation.py',
             'src/lava/jaes_v2/phase10_resources.py', 'src/lava/jaes_v2/phase9_protocol.py',
             'src/lava/jaes_v2/phase7_training.py', 'src/lava/jaes_v2/evidence.py',
             'scripts/jaes_v2/phase10_experiment.py', 'scripts/jaes_v2/phase10_numeric.py',
             'configs/jaes_v2/phase10/qualification_v2.json')
    return {**prior.code_contract(root), **{p: file_hash(root / p) for p in paths}}


def qualification_config(root, profile):
    original = prior.load_config(root, root / LEGACY_CONFIG)
    if profile == 'lifecycle':
        return replace(original, training_seed=9107, training_scope='TINY_LIFECYCLE',
                       tiny_per_attack=1, batch_size=4, gradient_accumulation=4, finetune_epochs=2,
                       maximum_updates=9, checkpoint_every_updates=2)
    if profile != '100':
        raise ValueError('Unknown qualification profile')
    return replace(original, training_seed=9107, maximum_updates=100, checkpoint_every_updates=10)


def qualification_id(profile):
    if profile == 'lifecycle':
        return 'p10_lifecycle_channel_s9107_v2'
    if profile == '100':
        return 'p10_engine_100_channel_s9107_v2'
    raise ValueError('Unknown qualification profile')


def phase10_tensorflow():
    for key, value in {'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_DETERMINISTIC_OPS': '1'}.items():
        os.environ[key] = value
    return prior.initialize('cpu')


def kernel(tf, backbone, model, forward):
    variables = tuple(model.trainable_weights)
    cnn = {id(v) for v in backbone.trainable_weights}

    def objective(x, y, coeff, counter, unused, paired, *, embeddings=False):
        p = forward(x, True, counter, 0, embeddings=embeddings)
        loss = bce(y, p)
        return tf.reduce_sum(loss * coeff), loss

    @tf.function(reduce_retracing=True)
    def warmup(x, y, coeff, counter):
        # The CNN is frozen and the warm-up optimizer only updates the head.
        features = tf.stack([backbone(x[:, i], training=False) for i in range(x.shape[1])], axis=1)
        with tf.GradientTape() as tape:
            loss, _ = objective(features, y, coeff, counter, features, False, embeddings=True)
        active = tuple(v for v in variables if id(v) not in cnn)
        grad = iter(tape.gradient(loss, active))
        return loss, tuple(tf.zeros_like(v) if id(v) in cnn else next(grad) for v in variables)

    # Keep the reference frame batching shape during recomputation. Running
    # individual frames can cross a hard-swish derivative boundary in float32.
    extractor = tf.keras.Model(model.inputs, model.get_layer('time_distributed_mobilenetv3small').output)
    recomputed = tf.recompute_grad(lambda x: extractor(x, training=True))

    @tf.function(reduce_retracing=True)
    def partial(x, y, coeff, counter):
        with tf.GradientTape() as tape:
            features = recomputed(x)
            loss, _ = objective(features, y, coeff, counter, x, False, embeddings=True)
        return loss, tuple(tape.gradient(loss, variables))

    return warmup, partial, variables, cnn, objective


def run(root, profile, role, *, resume=False, stop_at=None, policy='CLEAN_BCE', reference=None,
        prospective=None):
    """Run a qualification or registered medium job with an update-atomic resume pointer."""
    require_resources(root)
    started = perf_counter()
    if prospective is None:
        c = qualification_config(root, profile)
        base_id = qualification_id(profile)
        if role not in ('uninterrupted', 'resumable'):
            raise ValueError('Invalid comparator role')
    else:
        base_id, c, registry_hash = prospective
        if profile != 'medium' or role != 'scientific' or not base_id.startswith('p10_medium_'):
            raise ValueError('Invalid prospective experiment role')
    if policy not in ('CLEAN_BCE', 'GUARDED_ROBUST_SELECTION') or (
            policy == 'GUARDED_ROBUST_SELECTION') != (reference is not None):
        raise ValueError('Selection requires a registered reference')
    out = root / BASE / base_id / role
    if (out / 'seal.json').exists():
        raise ValueError('Sealed run cannot resume')
    if stop_at is not None and not 0 < stop_at < c.maximum_updates:
        raise ValueError('Stop step must precede final update')
    if policy == 'GUARDED_ROBUST_SELECTION':
        seal = prior.verify_tree(reference)
        if 'tier_b_reference.json' not in seal['output_hashes']:
            raise ValueError('Tier B reference missing')
        ref_path = reference / 'tier_b_reference.json'
        reference_data = read(ref_path)
    else:
        ref_path = reference_data = None
    tf = phase10_tensorflow()
    train, dev, b_ids, audit = prior.dataset(root, c)
    inputs = prior.Inputs(root, c)
    model, backbone, forward = prior.model_factory(tf, c)
    if any(layer.trainable for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)):
        raise ValueError('BatchNorm must stay frozen')
    warmup, partial, variables, cnn, _ = kernel(tf, backbone, model, forward)
    optimizer = tf.keras.optimizers.Adam(c.learning_rate)
    optimizer.build(variables)
    frozen = [v for v in model.weights if id(v) not in {id(v) for v in variables}]
    frozen_hash = prior.variable_hash(frozen)
    infer = prior.segmentwise_inference(tf, model)
    source_digest = object_hash([r.to_dict() for r in train + dev])
    contract = dict(experiment=base_id, profile=profile, backend=BACKEND, policy=policy,
                    config=c.__dict__, actual_sources=source_digest, b_ids=sorted(r.sample_id for r in dev if r.sample_id in b_ids),
                    source=fingerprints(root), initial_weights=prior.variable_hash(model.weights),
                    runtime=dict(python=platform.python_version(), system=platform.platform(),
                                 tensorflow=tf.__version__, packages={p: version(p) for p in ('keras', 'numpy', 'soundfile', 'librosa', 'scikit-learn')},
                                 deterministic_ops=True, one_dnn=False, threads=1,
                                 native_threads={name: os.getenv(name) for name in
                                                 ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}),
                    reference_sha256=file_hash(ref_path) if ref_path else None,
                    threshold=.5, margins=MARGINS, caps=CAPS, tier_b_every=3,
                    augmentation_rng='SeedSequence(training_seed+200000,epoch,cursor)',
                    dropout_rng='stateless training_seed and micro counter')
    if prospective is not None:
        contract['registry_sha256'] = registry_hash
    if reference_data is not None:
        keys = ('actual_sources', 'b_ids', 'backend', 'runtime', 'initial_weights', 'tier_b_every',
                'source', 'threshold', 'augmentation_rng', 'dropout_rng')
        expected = {key: contract[key] for key in keys}
        expected['config'] = dict(contract['config'], augmentation_policy='clean')
        if reference_data['matched_contract'] != expected:
            raise ValueError('Reference source, initialization, budget or numerical contract mismatch')
    state = dict(contract=contract, step=0, epoch=0, cursor=0, micro=0, journal=[], history=[],
                 best=None, bad_epochs=0, warning=None, frozen_hash=frozen_hash)
    if resume:
        directory, state = committed_controller(out, contract)
        tf.train.Checkpoint(model=model, optimizer=optimizer).read(str(directory / 'state')).assert_consumed()
        if (int(optimizer.iterations.numpy()) != state['step'] or state['frozen_hash'] != frozen_hash
                or prior.variable_hash(model.weights) != state['model_hash']
                or prior.variable_hash(optimizer.variables()) != state['optimizer_hash']):
            raise ValueError('Restored model, optimizer or cursor mismatch')
        if stop_at is not None and stop_at <= state['step']:
            raise ValueError('Interruption must advance the checkpoint')
        if state['warning'] == 'UNEXPECTED_THROUGHPUT_STALL':
            state['warning'] = None
        write_json(out / 'resume_events' / f"u{state['step']:06d}.json",
                   dict(step=state['step'], optimizer_hash=state['optimizer_hash'],
                        model_hash=state['model_hash'], controller_sha256=file_hash(directory / 'controller.json')))
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'contract.json', contract)
        write_json(out / 'data_audit.json', dict(audit, calibration_used=False, official_evaluation_used=False))
    timing = Counter(startup_and_restore=perf_counter() - started)
    samples = []
    invocation_updates = 0
    collapse_epochs = 0

    def guard():
        if perf_counter() - started > c.resource_limits['seconds_per_invocation']:
            raise ResourceStop('INVOCATION_TIME_LIMIT')
        return require_resources(root)

    def commit():
        guard()
        t = perf_counter()
        state.update(model_hash=prior.variable_hash(model.weights),
                     optimizer_hash=prior.variable_hash(optimizer.variables()))
        tag = f"u{state['step']:06d}_e{state['epoch']:03d}_c{state['cursor']:03d}"
        target = out / 'checkpoints' / tag
        if target.exists():
            if strict_controller(target, contract) != state:
                raise ValueError('Orphan checkpoint differs from deterministic replay')
        else:
            prior.save_checkpoint(tf, model, optimizer, target, state)
        write_json(out / 'latest.json', dict(checkpoint=tag, sha256=file_hash(target / 'controller.json')))
        timing['checkpoint_seconds'] += perf_counter() - t
        return target

    def publish(status):
        remaining = c.maximum_updates - state['step']
        seconds = timing['train_seconds']
        write_json(out / 'status.json', dict(status=status, step=state['step'], epoch=state['epoch'],
                   last_update=state['journal'][-1] if state['journal'] else None,
                   validation=state['history'][-1] if state['history'] else None,
                   best=state['best'], early_stopping_counter=state['bad_epochs'], warning=state['warning'],
                   latest=read(out / 'latest.json') if (out / 'latest.json').exists() else None,
                   resources=samples[-1] if samples else None, gpu_vram=None,
                   updates_per_second=invocation_updates / seconds if seconds else None,
                   remaining_train_seconds=remaining * seconds / invocation_updates if invocation_updates else None))

    try:
        if not resume:
            commit()
        order = None
        while state['step'] < c.maximum_updates:
            samples.append(guard())
            epoch, cursor = state['epoch'], state['cursor']
            if order is None:
                order = prior.epoch_order(train, c, epoch)
            is_partial = epoch >= c.warmup_epochs
            lr = (c.finetune_learning_rate * (1 - .5 * (epoch - c.warmup_epochs) / c.finetune_epochs)
                  if is_partial else c.learning_rate)
            optimizer.learning_rate.assign(lr)
            records = [train[i] for i in order[cursor:cursor + c.batch_size]]
            rng = np.random.default_rng(np.random.SeedSequence([c.training_seed + 200000, epoch, cursor]))
            views = [(r, str(rng.choice(prior.BANKS[c.augmentation_policy]))) for r in records]
            step_started = perf_counter()
            active = [i for i, v in enumerate(variables) if is_partial or id(v) not in cnn]
            accum = [tf.zeros_like(v) for v in variables]
            loss = 0.
            gradient = partial if is_partial else warmup
            for record, condition in views:
                value, gs = gradient(inputs.get(record, condition)[None], tf.constant([record.label]),
                                     tf.constant([1 / len(views)], tf.float32), tf.constant(state['micro']))
                tf.debugging.assert_all_finite(value, 'Nonfinite loss')
                for g in gs:
                    tf.debugging.assert_all_finite(g, 'Invalid micro-gradient')
                accum = [a + g for a, g in zip(accum, gs)]
                loss += float(value.numpy())
                state['micro'] += 1
            norm = float(tf.linalg.global_norm([accum[i] for i in active]).numpy())
            if not math.isfinite(norm) or norm == 0 or norm > 1e5 or loss > 20:
                raise FloatingPointError(f'Invalid update norm={norm} loss={loss}')
            before = prior.variable_hash([variables[i] for i in active])
            optimizer.apply_gradients((accum[i], variables[i]) for i in active)
            for i in active:
                tf.debugging.assert_all_finite(variables[i], 'Invalid updated weight')
            if before == prior.variable_hash([variables[i] for i in active]):
                raise FloatingPointError('No active weight changed')
            elapsed = perf_counter() - step_started
            timing['train_seconds'] += elapsed
            invocation_updates += 1
            state['step'] += 1
            state['cursor'] += len(views)
            state['journal'].append(dict(step=state['step'], epoch=epoch, lr=lr, loss=loss, norm=norm,
                                         sources=[[r.sample_id, v] for r, v in views], micro=state['micro']))
            if elapsed > 300:
                state['warning'] = 'UNEXPECTED_THROUGHPUT_STALL'
            at_epoch = state['cursor'] == len(order)
            final = state['step'] == c.maximum_updates
            if at_epoch or final:
                evaluation_epoch = epoch + 1
                if at_epoch:
                    state.update(epoch=evaluation_epoch, cursor=0)
                    order = None
                is_b = evaluation_epoch % 3 == 0 or final
                tier = 'B' if is_b else 'A'
                rows, measured = score(dev, b_ids, tier, inputs, {'fixed': infer}, guard)
                timing['validation_seconds'] += measured['elapsed_seconds']
                row_metrics = metrics(rows['fixed'])
                clean = row_metrics['A_PAIRED/clean']
                objective = clean['nll'] if policy == 'CLEAN_BCE' else None
                guard_result = None
                if policy == 'GUARDED_ROBUST_SELECTION' and is_b:
                    guard_result = select(rows['fixed'], reference_data['updates'][str(state['step'])], MARGINS, CAPS)
                    objective = guard_result['objective']
                collapse = clean['fp'] == clean['n_REAL'] or clean['fn'] == clean['n_FAKE']
                collapse_epochs = state['history'][-1]['collapse_epochs'] + 1 if collapse and state['history'] else int(collapse)
                if collapse_epochs >= 2 and evaluation_epoch >= 3:
                    state['warning'] = 'PERSISTENT_PREDICTION_COLLAPSE'
                state['history'].append(dict(epoch=evaluation_epoch, step=state['step'], tier=tier, clean=clean, collapse_epochs=collapse_epochs,
                                             guard_counts=guard_counts(rows['fixed']) if is_b else None, guard=guard_result))
                if objective is not None:
                    rank = [objective, clean['nll'], evaluation_epoch]
                    if state['best'] is None or rank < state['best']['rank']:
                        state['best'] = dict(rank=rank, tag=f"u{state['step']:06d}_e{state['epoch']:03d}_c{state['cursor']:03d}")
                        state['bad_epochs'] = 0
                    else:
                        state['bad_epochs'] += 1
            pause = state['step'] == stop_at or state['warning'] == 'UNEXPECTED_THROUGHPUT_STALL'
            if final or at_epoch or pause or state['step'] % c.checkpoint_every_updates == 0:
                commit()
            if pause and not final:
                publish('PAUSE_AT_CHECKPOINT')
                return
            publish('WARNING' if state['warning'] else 'RUNNING')
        if state['best'] is None:
            publish('SCIENTIFIC_NO_GO')
            return
        final_hash = prior.variable_hash(model.weights)
        selected = out / 'checkpoints' / state['best']['tag']
        selected_state = strict_controller(selected, contract)
        tf.train.Checkpoint(model=model, optimizer=optimizer).read(str(selected / 'state')).assert_consumed()
        if prior.variable_hash(model.weights) != selected_state['model_hash']:
            raise ValueError('Selected checkpoint changed')
        rows, measured = score(dev, b_ids, 'C', inputs, {'fixed': infer}, guard)
        timing['validation_seconds'] += measured['elapsed_seconds']
        t = perf_counter()
        guard()
        model.save(out / 'selected.keras')
        timing['export_seconds'] += perf_counter() - t
        write_csv(out / 'scores.csv', rows['fixed'])
        write_json(out / 'result.json', dict(state=state, final_weights=final_hash, selected_weights=selected_state['model_hash'],
                                            scores_hash=object_hash(rows['fixed']), metrics=metrics(rows['fixed'])))
        publish('QUALIFICATION_COMPLETE' if prospective is None else 'SCIENTIFIC_COMPLETE')
        prior.seal_tree(out)
    except (ResourceStop, tf.errors.ResourceExhaustedError, MemoryError) as exc:
        state['warning'] = str(exc) if isinstance(exc, ResourceStop) else 'TENSOR_ALLOCATION_FAILURE'
        publish('STOP_FOR_RESOURCE_SAFETY')
        raise
    except (FloatingPointError, tf.errors.InvalidArgumentError) as exc:
        state['warning'] = str(exc)
        publish('STOP_FOR_NUMERICAL_FAILURE')
        raise
    finally:
        # Keep timings outside the run seal. Resume invocations retain distinct reports.
        evidence = out.parent / f'{role}_{os.getpid()}_{state["step"]}_resources.json'
        write_json(evidence, dict(timing=timing, input_timing=inputs.timings, samples=samples,
                                  checkpoint_bytes=sum(p.stat().st_size for p in (out / 'checkpoints').rglob('*') if p.is_file()),
                                  final_resources=snapshot(root)))
