"""Launch the registered Phase 10 matched medium runs after pilot acceptance."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from dataclasses import replace

if __name__ == '__main__':
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

from scripts.jaes_v2.phase6_experiment import resource_lock
from scripts.jaes_v2.phase10_experiment import ROOT, OUT
from src.lava.jaes_v2 import phase10_training as engine, phase8_training as prior
from src.lava.jaes_v2.evidence import file_hash, read_csv, write_json
from src.lava.jaes_v2.phase10_resources import require_resources
from src.lava.jaes_v2.phase10_validation import select


REGISTRY = Path('configs/jaes_v2/phase10/prospective_medium_v1.json')
ENGINE = 'src/lava/jaes_v2/phase10_training.py'
PILOT = 'p10_engine_100_channel_s9107_v2'


def registered(experiment):
    registry = engine.read(ROOT / REGISTRY)
    entries = {row['experiment_id']: row for row in registry['paired_runs']}
    row = entries.get(experiment)
    if row is None:
        raise ValueError('Experiment ID is outside the prospective registry')
    template = prior.load_config(ROOT, ROOT / registry['template'])
    if (registry['backend'] != engine.BACKEND or registry['medium_manifest_sha256'] != template.manifest_sha256
            or registry['checkpoint_policy'] != 'CLEAN_BCE' or registry['calibration_enabled']
            or registry['official_evaluation_enabled'] or not registry['fixed_budget']
            or registry['threshold'] != .5 or row['augmentation_policy'] not in ('clean', 'channel')):
        raise ValueError('Prospective data, objective or engine contract mismatch')
    config = replace(template, training_seed=row['seed'], augmentation_policy=row['augmentation_policy'],
                     maximum_updates=registry['maximum_updates'],
                     checkpoint_every_updates=registry['checkpoint_every_updates'])
    return registry, config


def admission(registry):
    release = registry.get('release')
    if not registry.get('launch_authorized') or release is None:
        raise ValueError('Scientific launch gate is closed')
    proof = OUT / 'benchmark_resume_v2.json'
    if file_hash(proof) != release['benchmark_sha256']:
        raise ValueError('Qualification benchmark evidence changed')
    benchmark = engine.read(proof)
    if not benchmark['pass_thresholds'] or benchmark['steps'] != 100 or benchmark['interrupted_at'] != 55:
        raise ValueError('Benchmark has not passed')
    current = engine.fingerprints(ROOT)
    if current != {**benchmark['source'], ENGINE: release['engine_sha256']}:
        raise ValueError('Release source differs from the audited adapter and frozen pilot')
    if file_hash(Path(__file__)) != release['launcher_sha256']:
        raise ValueError('Scientific launcher changed')
    for role in ('uninterrupted', 'resumable'):
        directory = OUT / PILOT / role
        if file_hash(directory / 'seal.json') != release['pilot_seals'][role]:
            raise ValueError('Pilot seal changed')
        prior.verify_tree(directory)
    return benchmark


def preflight(experiment):
    registry, config = registered(experiment)
    admission(registry)
    resources = require_resources(ROOT)
    train, dev, b_ids, audit = prior.dataset(ROOT, config)
    updates = len(prior.epoch_order(train, config, 0)) // config.batch_size
    if config.maximum_updates != updates * (config.warmup_epochs + config.finetune_epochs):
        raise ValueError('Fixed budget does not match the dataset and epoch schedule')
    return config, dict(status='MEDIUM_PREFLIGHT_PASS', experiment_id=experiment,
                        backend=engine.BACKEND, seed=config.training_seed, augmentation=config.augmentation_policy,
                        train=len(train), development_a=len(dev) - sum(r.sample_id in b_ids for r in dev),
                        development_b=sum(r.sample_id in b_ids for r in dev),
                        updates=config.maximum_updates, resources=resources, data_audit=audit,
                        registry_sha256=file_hash(ROOT / REGISTRY))


def comparison(experiment, peer):
    first, second = [OUT / name / 'scientific' for name in (experiment, peer)]
    for directory in (first, second):
        prior.verify_tree(directory)
    a, b = [engine.read(directory / 'result.json') for directory in (first, second)]
    ca, cb = (r['state']['contract'] for r in (a, b))
    if (ca['config']['augmentation_policy'] != 'clean' or cb['config']['augmentation_policy'] != 'channel'
            or {**ca, 'experiment': cb['experiment'],
                'config': {**ca['config'], 'augmentation_policy': 'channel'}} != cb):
        raise ValueError('Expected a matched ERM/channel pair in that order')

    def scores(directory):
        return [dict(row, label=int(row['label']), raw_score=float(row['raw_score']),
                     predicted_label=int(row['predicted_label']))
                for row in read_csv(directory / 'scores.csv')]

    guard = select(scores(second), scores(first), engine.MARGINS, engine.CAPS)
    return dict(status='MATCHED_DEVELOPMENT_COMPARISON', seed=ca['config']['training_seed'],
                erm=experiment, channel=peer, erm_metrics=a['metrics'], channel_metrics=b['metrics'],
                channel_guard_against_selected_erm=guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'train', 'resume', 'monitor', 'verify', 'compare'))
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--peer')
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    registered(args.experiment)
    run = OUT / args.experiment / 'scientific'
    if args.mode == 'monitor':
        result = engine.read(run / 'status.json')
    elif args.mode == 'verify':
        result = prior.verify_tree(run)
    elif args.mode == 'compare':
        if not args.peer:
            parser.error('compare requires --peer')
        registered(args.peer)
        result = comparison(args.experiment, args.peer)
    elif args.mode == 'preflight':
        _, result = preflight(args.experiment)
    else:
        with resource_lock(ROOT):
            config, _ = preflight(args.experiment)
            tmp = OUT / 'tmp'
            tmp.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=tmp):
                pass
            os.environ.update(TEMP=str(tmp), TMP=str(tmp))
            tempfile.tempdir = str(tmp)
            engine.run(ROOT, 'medium', 'scientific', resume=args.mode == 'resume',
                       stop_at=args.stop_at, prospective=(args.experiment, config, file_hash(ROOT / REGISTRY)))
        return
    if args.output:
        target = args.output.resolve()
        if not target.is_relative_to(OUT) or target.exists() or target.suffix != '.json':
            parser.error('Use a new JSON output under outputs/jaes_v2/phase10')
        write_json(target, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
