"""Preflight, manually qualify and run the fixed-epoch nested size experiment."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

if __name__ == '__main__':
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

import numpy as np

from scripts.jaes_v2 import size1280_profile as profile
from scripts.jaes_v2.final_score_diagnostic import align, expected_rows
from scripts.jaes_v2.phase6_experiment import resource_lock
from scripts.jaes_v2.phase10_numeric import numerical_diff
from src.lava.jaes_v2 import phase10_training as engine, phase8_training as prior, size1280_training as runner
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv
from src.lava.jaes_v2.phase10_resources import MIN_D_FREE, WRITE_RESERVE, require_resources
from src.lava.jaes_v2.phase10_validation import compare, metrics, score, select

ROOT = profile.ROOT
OUT = ROOT / profile.BASE
QUALIFICATION = OUT / 'qualification'
SOURCE = (
    'scripts/jaes_v2/phase10_size1280.py', 'scripts/jaes_v2/size1280_profile.py',
    'src/lava/jaes_v2/size1280_training.py', 'scripts/jaes_v2/size1280_analysis.py',
)


def release():
    return dict(registry_sha256=file_hash(ROOT / profile.REGISTRY),
                adapter={p: file_hash(ROOT / p) for p in SOURCE})


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def preflight(spec, template, data):
    train, dev, b_ids, audit = data
    q_storage = profile.storage(ROOT, spec, qualification=True)
    s_storage = profile.storage(ROOT, spec, current=q_storage['measurement'])
    old_checkpoint_sizes = []
    timing = {}
    for method in profile.METHODS:
        base, _ = profile.reference(ROOT, method)
        old_checkpoint_sizes.extend(sum(p.stat().st_size for p in d.iterdir() if p.is_file())
                                    for d in (base / 'checkpoints').iterdir() if d.is_dir())
        logs = [engine.read(p) for p in base.parent.glob('scientific_*_resources.json')]
        if logs:
            seconds = Counter()
            for log in logs:
                seconds.update(log['timing'])
            timing[method] = dict(seconds)
    extrapolated = sum(2 * r.get('train_seconds', 0) + r.get('validation_seconds', 0)
                       + 2 * r.get('checkpoint_seconds', 0) + r.get('export_seconds', 0)
                       + r.get('startup_and_restore', 0)
                       for r in timing.values()) if len(timing) == 2 else None
    profile.weight_cache(spec)
    total_required = (s_storage['disk_required_bytes'] + q_storage['disk_required_bytes']
                      - MIN_D_FREE - WRITE_RESERVE)
    return dict(status='STATIC_PREFLIGHT_PASS', measured_at=datetime.now().astimezone().isoformat(),
                release=release(), train=len(train), dev=len(dev), b=len(b_ids), cohort=audit,
                batch=template.batch_size, updates_per_epoch=spec['updates_per_epoch'],
                epochs=template.warmup_epochs + template.finetune_epochs, updates=spec['maximum_updates'],
                max_observed_640_checkpoint_bytes=max(old_checkpoint_sizes),
                qualification_storage=q_storage, scientific_pair_storage=s_storage,
                full_pair_storage_including_qualification_peak_bytes=total_required,
                additional_disk_including_qualification_peak_bytes=max(
                    0, total_required - q_storage['measurement']['d_free']),
                scientific_output_exists={m: profile.directory(ROOT, spec, m, 'scientific').exists()
                                          for m in profile.METHODS},
                reference_resource_timings=timing, extrapolated_pair_seconds_before_margin=extrapolated)


def smoke(spec, template, data):
    train, dev, b_ids, _ = data
    tol = engine.read(ROOT / 'configs/jaes_v2/phase10/qualification_v2.json')['numeric']
    init, stages = {}, {}
    for method in profile.METHODS:
        require_resources(ROOT)
        config = profile.run_config(template, method, spec)
        tf, model, backbone, forward = runner.initialize(ROOT, spec, config, spec['initial_weights'])
        init[method] = prior.variable_hash(model.weights)
        if method == 'erm':
            del model, backbone, forward
    warmup, partial, variables, cnn, objective = engine.kernel(tf, backbone, model, forward)
    if not cnn or len(cnn) == len(variables):
        raise ValueError('Unexpected warm-up/partial variable partition')
    frozen_hash = prior.variable_hash(model.weights)
    inputs = prior.Inputs(ROOT, config)
    for epoch, fn in ((0, warmup), (1, partial)):
        order = prior.epoch_order(train, config, epoch)
        batch = runner.update_plan(train, config, epoch, 0, order)
        cases = [next(pair for pair in batch if pair[0].label == label) for label in (0, 1)]
        evidence = []
        for i, (record, condition) in enumerate(cases):
            require_resources(ROOT)
            x = inputs.get(record, condition)[None]
            if tuple(x.shape[1:]) != tuple(model.input_shape[1:]) or not np.isfinite(x).all():
                raise ValueError('Invalid input geometry or values')
            y, coeff, counter = tf.constant([record.label]), tf.constant([1 / config.batch_size], tf.float32), tf.constant(i)
            result, grads = fn(x, y, coeff, counter)
            with tf.GradientTape() as tape:
                reference, _ = objective(x, y, coeff, counter, x, False)
            ref_grads = tape.gradient(reference, variables)
            active = [j for j, v in enumerate(variables) if epoch or id(v) not in cnn]
            diffs = [numerical_diff(ref_grads[j], grads[j], tol['gradient']) for j in active]
            loss = numerical_diff(reference, result, tol['loss'])
            heads = [g for g, v in zip(grads, variables, strict=True) if id(v) not in cnn]
            conv = [g for g, v in zip(grads, variables, strict=True) if id(v) in cnn]
            head_norm, conv_norm = (float(tf.linalg.global_norm(group).numpy()) for group in (heads, conv))
            p = float(model(x, training=False).numpy().reshape(-1)[0])
            if (not all(d['pass_thresholds'] for d in diffs) or not loss['pass_thresholds']
                    or not np.isfinite([head_norm, conv_norm]).all() or head_norm <= 0
                    or (conv_norm <= 0 if epoch else conv_norm != 0) or not 0 <= p <= 1):
                raise ValueError('Warm-up/partial numerical smoke failed')
            evidence.append(dict(id=record.sample_id, label=record.label, view=condition, score=p,
                                 loss=loss, head_norm=head_norm, cnn_norm=conv_norm,
                                 gradients_max_abs=max(d['max_abs'] for d in diffs), tensors=len(diffs)))
        stages[str(epoch)] = evidence
    if prior.variable_hash(model.weights) != frozen_hash:
        raise ValueError('Weights changed during zero-update smoke')
    return dict(status='NUMERICAL_SMOKE_PASS', release=release(), initial_weights=init, stages=stages,
                optimizer_updates=0, frozen_weights_unchanged=True)


def verified(spec, template, data, method, role='scientific'):
    run = profile.directory(ROOT, spec, method, role)
    sealed = prior.verify_tree(run)['output_hashes']
    required = ('contract.json', 'latest.json', 'result.json', 'scores.csv', 'selected.keras', 'status.json')
    if not set(required) <= sealed.keys():
        raise ValueError('Incomplete artifact seal')
    config = profile.run_config(template, method, spec, qualification=role != 'scientific')
    contract = runner.contract(ROOT, spec, config, method, role, data, release())
    final, state = engine.committed_controller(run, contract)
    result = engine.read(run / 'result.json')
    step = config.maximum_updates
    epoch, cursor = divmod(step, spec['updates_per_epoch'])
    cursor *= config.batch_size
    expected_status = 'SCIENTIFIC_COMPLETE' if role == 'scientific' else 'QUALIFICATION_COMPLETE'
    status = engine.read(run / 'status.json')
    if (engine.read(run / 'contract.json') != contract or result['state'] != state
            or (state['step'], state['epoch'], state['cursor'], state['micro']) != (step, epoch, cursor, step * config.batch_size)
            or (status['status'], status['step'], status['epoch']) != (expected_status, step, epoch)
            or final.name != f'u{step:06d}_e{epoch:03d}_c{cursor:03d}'
            or result['final_weights'] != state['model_hash']):
        raise ValueError('Completion, budget or final state mismatch')
    train, dev, b_ids, _ = data
    if len(state['journal']) != step:
        raise ValueError('Incomplete optimizer journal')
    for i, record in enumerate(state['journal']):
        ep, offset = divmod(i, spec['updates_per_epoch'])
        if offset == 0:
            order = prior.epoch_order(train, config, ep)
        planned = runner.update_plan(train, config, ep, offset * config.batch_size, order)
        if ((record['step'], record['epoch'], record['micro'], record['lr'])
                != (i + 1, ep, (i + 1) * config.batch_size, runner.learning_rate(config, ep))
                or record['sources'] != [[r.sample_id, condition] for r, condition in planned]
                or not np.isfinite([record['loss'], record['norm']]).all() or record['norm'] <= 0):
            raise ValueError('Optimizer record, LR, source ordering or augmentation differs')
    validation_steps = list(range(spec['updates_per_epoch'], step, spec['updates_per_epoch'])) + [step]
    expected_history = [(s, (s - 1) // spec['updates_per_epoch'] + 1,
                         'B' if s == step or (s - 1) // spec['updates_per_epoch'] % 3 == 2 else 'A')
                        for s in validation_steps]
    if [(h['step'], h['epoch'], h['tier']) for h in state['history']] != expected_history:
        raise ValueError('Validation cadence differs')
    best = min(state['history'], key=lambda h: (h['clean']['nll'], h['epoch']))
    best_epoch, best_cursor = divmod(best['step'], spec['updates_per_epoch'])
    tag = f"u{best['step']:06d}_e{best_epoch:03d}_c{best_cursor * config.batch_size:03d}"
    if state['best']['tag'] != tag:
        raise ValueError('Selected checkpoint differs from minimum clean BCE')
    selected = engine.strict_controller(run / 'checkpoints' / tag, contract)
    if result['selected_weights'] != selected['model_hash']:
        raise ValueError('Selected checkpoint weight hash mismatch')
    raw = read_csv(run / 'scores.csv')
    align(raw, expected_rows(dev, b_ids), prior.DEV)
    rows = [dict(r, label=int(r['label']), threshold=float(r['threshold']),
                 raw_score=float(r['raw_score']), predicted_label=int(r['predicted_label'])) for r in raw]
    if object_hash(rows) != result['scores_hash'] or metrics(rows) != result['metrics']:
        raise ValueError('Score or metric replay mismatch')
    return run, result, rows


def verify_export(spec, template, data, run, result, rows, *, parity=False):
    """Reload a sealed export and compare with its saved selected weights and scores."""
    tf = engine.phase10_tensorflow()
    model = tf.keras.models.load_model(run / 'selected.keras', compile=False)
    if prior.variable_hash(model.weights) != result['selected_weights']:
        raise ValueError('Exported model does not match the selected checkpoint')
    if not parity:
        return dict(exported_weights=result['selected_weights'])
    _, dev, b_ids, _ = data
    inputs = prior.Inputs(ROOT, template)
    standard = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    fixed = prior.segmentwise_inference(tf, model)
    both, timing = score(dev, b_ids, 'C', inputs, dict(standard=standard, fixed=fixed),
                         lambda: require_resources(ROOT))
    evidence = compare(both['standard'], both['fixed'], **spec['qualification']['validation_parity'])
    if not evidence['pass_thresholds'] or object_hash(both['fixed']) != object_hash(rows):
        raise ValueError('Full validation parity or exact exported score replay failed')
    return dict(exported_weights=result['selected_weights'], parity=evidence, timing=timing)


def training_budget(spec, method, role, resume):
    path = profile.directory(ROOT, spec, method, role)
    if (path / 'seal.json').exists() or path.exists() != resume:
        raise ValueError('Output collision or missing run for resume')
    if resume and not (path / 'latest.json').is_file():
        raise ValueError('No committed checkpoint to resume')
    q = spec['qualification']
    if role != 'scientific':
        maximum = q['maximum_updates']
        paths = [profile.directory(ROOT, spec, q['arm'], r) for r in ('uninterrupted', 'resumable')]
    else:
        maximum = spec['maximum_updates']
        paths = [profile.directory(ROOT, spec, m, role) for m in profile.METHODS]
    every = spec['checkpoint_every_updates']
    expected = {0, maximum, *range(every, maximum + 1, every)}
    remaining = 0
    for candidate in paths:
        if (candidate / 'seal.json').is_file():
            continue
        committed = {int(p.name[1:7]) for p in (candidate / 'checkpoints').glob('u??????_e???_c???')
                     if (p / 'seal.json').is_file()}
        remaining += len(expected - committed)
    evidence = profile.storage(ROOT, spec, qualification=role != 'scientific', remaining=remaining)
    if evidence['additional_disk_bytes']:
        raise ValueError(f"Additional free bytes required on D: {evidence['additional_disk_bytes']}")
    return path


def qualification_report(spec, template, data):
    q = spec['qualification']
    arms = [verified(spec, template, data, q['arm'], role) for role in ('uninterrupted', 'resumable')]
    left, right = [arm[1] for arm in arms]
    if any((arms[0][0] / 'resume_events').glob('*.json')):
        raise ValueError('Uninterrupted comparator has a resume event')
    fields = ('contract', 'step', 'epoch', 'cursor', 'micro', 'journal', 'history', 'best',
              'bad_epochs', 'frozen_hash', 'model_hash', 'optimizer_hash')
    same = {f: left['state'][f] == right['state'][f] for f in fields}
    same.update({f: left[f] == right[f] for f in ('final_weights', 'selected_weights', 'scores_hash', 'metrics')})
    if not all(same.values()):
        raise ValueError(f'Exact resume mismatch: {same}')
    cut = q['split_step']
    event = engine.read(arms[1][0] / 'resume_events' / f'u{cut:06d}.json')
    checkpoint = arms[1][0] / 'checkpoints' / f"u{cut:06d}_e001_c000"
    controller = engine.strict_controller(checkpoint, left['state']['contract'])
    other = engine.strict_controller(arms[0][0] / 'checkpoints' / checkpoint.name, left['state']['contract'])
    if (controller != other or event != dict(step=cut, model_hash=controller['model_hash'],
            optimizer_hash=controller['optimizer_hash'], controller_sha256=file_hash(checkpoint / 'controller.json'))):
        raise ValueError('Warm-up boundary checkpoint or recorded restore differs')
    if (controller['model_hash'] == left['state']['model_hash']
            or controller['warmup_cnn_hash'] != left['state']['warmup_cnn_hash']
            or any(arm['state']['first_partial_cnn_hash'] == controller['warmup_cnn_hash']
                   for arm in (left, right))
            or left['state']['first_partial_cnn_hash'] != right['state']['first_partial_cnn_hash']):
        raise ValueError('CNN warm-up/partial transition differs')
    export = verify_export(spec, template, data, *arms[0], parity=True)
    second = verify_export(spec, template, data, *arms[1])
    timing = {}
    for role in ('uninterrupted', 'resumable'):
        summed = Counter()
        for log in (QUALIFICATION / 'resources').glob(f'{role}_*.json'):
            summed.update(engine.read(log)['timing'])
        timing[role] = dict(summed)
    sample = timing['uninterrupted']
    reference = preflight(spec, template, data)
    baseline = reference['reference_resource_timings']
    plan = None
    if len(baseline) == 2 and all(sample.get(k, 0) > 0 for k in ('warmup_seconds', 'partial_seconds')):
        partial_updates = q['maximum_updates'] - cut
        channel = sample['warmup_seconds'] + sample['partial_seconds'] * (spec['maximum_updates'] - cut) / partial_updates
        ratio = baseline['erm']['train_seconds'] / baseline['channel']['train_seconds']
        overhead = sum(r.get('validation_seconds', 0) + 2 * r.get('checkpoint_seconds', 0)
                       + r.get('startup_and_restore', 0) + r.get('export_seconds', 0) for r in baseline.values())
        plan = dict(channel_warmup_seconds=sample['warmup_seconds'],
                    sampled_channel_partial_updates=partial_updates,
                    projected_pair_seconds=channel * (1 + ratio) + overhead,
                    limitation='Two partial updates include trace startup. ERM and validation use historical timings.')
    return dict(status='SIZE1280_QUALIFICATION_PASS', release=release(),
                max_updates=q['maximum_updates'], split_step=cut, matches=same, validation=export,
                warmup_cnn_hash=controller['warmup_cnn_hash'],
                first_partial_cnn_hash=left['state']['first_partial_cnn_hash'],
                resumed_export=second, invocation_timings_seconds=timing, runtime_projection=plan,
                keras_archive_hash_equal=file_hash(arms[0][0] / 'selected.keras') == file_hash(arms[1][0] / 'selected.keras'),
                seals={p.name: file_hash(p / 'seal.json') for p, _, _ in arms})


def require_smoke():
    path = QUALIFICATION / 'numerical_smoke.json'
    if not path.is_file():
        raise ValueError('Run the zero-update smoke before qualification')
    report = engine.read(path)
    if (report['status'] != 'NUMERICAL_SMOKE_PASS' or report['release'] != release()
            or report['optimizer_updates'] != 0):
        raise ValueError('Missing or stale zero-update numerical smoke')


def require_qualification():
    require_smoke()
    path = QUALIFICATION / 'qualification.json'
    if not path.is_file():
        raise ValueError('Run the manual bounded qualification before scientific training')
    report = engine.read(path)
    if (report['status'] != 'SIZE1280_QUALIFICATION_PASS' or report['release'] != release()
            or not all(report['matches'].values()) or not report['validation']['parity']['pass_thresholds']):
        raise ValueError('Missing or stale size1280 qualification evidence')
    for role in ('uninterrupted', 'resumable'):
        path = QUALIFICATION / role
        if file_hash(path / 'seal.json') != report['seals'][role]:
            raise ValueError('Qualification seal changed')
        prior.verify_tree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'smoke', 'qualify', 'qualify-resume',
                                         'qualify-verify', 'train', 'resume', 'verify', 'compare'))
    parser.add_argument('--arm', choices=profile.METHODS)
    parser.add_argument('--role', choices=('uninterrupted', 'resumable'))
    args = parser.parse_args()
    if args.mode in ('train', 'resume', 'verify') and not args.arm:
        parser.error(f'{args.mode} requires --arm')
    if args.arm and args.mode not in ('train', 'resume', 'verify'):
        parser.error('--arm is only supported by train, resume and verify')
    if (args.mode == 'qualify' and not args.role) or (args.role and args.mode not in ('qualify', 'qualify-resume')):
        parser.error('--role is required for qualify and optional for qualify-resume')
    spec, template = profile.registry()
    data = profile.cohort(ROOT, spec, template)
    if args.mode == 'preflight':
        result = preflight(spec, template, data)
        result['qualification_pass'] = False
        result['scientific_ready_for_fresh_pair'] = False
        if (QUALIFICATION / 'qualification.json').is_file():
            require_qualification()
            result['qualification_pass'] = True
            result['scientific_ready_for_fresh_pair'] = (
                not result['scientific_pair_storage']['additional_disk_bytes']
                and not any(result['scientific_output_exists'].values()))
    elif args.mode == 'compare':
        from scripts.jaes_v2.size1280_analysis import analyze, markdown_report
        require_qualification()
        fits = [verified(spec, template, data, arm) for arm in profile.METHODS]
        result = analyze(ROOT, spec, data, fits)
        path = OUT / 'size_sensitivity.json'
        if path.exists():
            if engine.read(path) != result:
                raise ValueError('Existing analysis differs from verified replay')
        else:
            write_new(path, result)
        report_path = ROOT / 'docs/jaes_v2/TRAINING_SIZE_SENSITIVITY_RESULTS.md'
        content = markdown_report(result)
        if report_path.exists():
            if report_path.read_text(encoding='utf-8') != content:
                raise ValueError('Existing Markdown report differs from verified replay')
        else:
            with report_path.open('x', encoding='utf-8') as stream:
                stream.write(content)
    elif args.mode == 'verify':
        with resource_lock(ROOT):
            require_resources(ROOT)
            path, result, rows = verified(spec, template, data, args.arm)
            result = dict(status='SCIENTIFIC_VERIFICATION_PASS', selected=result['state']['best']['tag'],
                          score_rows=len(rows), seal_sha256=file_hash(path / 'seal.json'),
                          **verify_export(spec, template, data, path, result, rows))
    else:
        with resource_lock(ROOT), tempfile.TemporaryDirectory(prefix='size1280_', dir=ROOT / engine.BASE) as temp:
            os.environ.update(TEMP=temp, TMP=temp)
            tempfile.tempdir = temp
            if args.mode == 'smoke':
                path = QUALIFICATION / 'numerical_smoke.json'
                if path.exists():
                    require_smoke()
                    result = engine.read(path)
                else:
                    result = smoke(spec, template, data)
                    write_new(path, result)
            elif args.mode == 'qualify-verify':
                require_smoke()
                path = QUALIFICATION / 'qualification.json'
                if path.exists():
                    require_qualification()
                    result = engine.read(path)
                else:
                    result = qualification_report(spec, template, data)
                    write_new(path, result)
            else:
                bounded = args.mode.startswith('qualify')
                role = (args.role or 'resumable') if bounded else 'scientific'
                arm = spec['qualification']['arm'] if bounded else args.arm
                resume = args.mode in ('resume', 'qualify-resume')
                if bounded:
                    require_smoke()
                else:
                    require_qualification()
                    if arm == 'channel':
                        erm_path, erm_result, erm_rows = verified(spec, template, data, 'erm')
                        verify_export(spec, template, data, erm_path, erm_result, erm_rows)
                path = training_budget(spec, arm, role, resume)
                config = profile.run_config(template, arm, spec, qualification=bounded)
                cut = spec['qualification']['split_step'] if args.mode == 'qualify' and role == 'resumable' else None
                runner.Run(ROOT, spec, config, arm, role, data, release()).execute(resume=resume, stop_at=cut)
                result = engine.read(path / 'status.json')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
