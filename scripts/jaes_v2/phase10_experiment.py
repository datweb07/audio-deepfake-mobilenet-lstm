"""Phase 10 read-only audits and gated qualification commands."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

# Set this before NumPy loads OpenBLAS. Its default worker pool can consume
# substantial Windows commit before the resource gate even runs.
if __name__ == '__main__':
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

from src.lava.jaes_v2 import phase8_training as prior
from src.lava.jaes_v2 import phase10_training as p10
from src.lava.jaes_v2.evidence import file_hash, read_csv, write_csv, write_json
from src.lava.jaes_v2.phase10_resources import pressure, require_resources, snapshot
from src.lava.jaes_v2.phase10_validation import compare, guard_counts, metrics, plan, score, select
from scripts.jaes_v2.phase6_experiment import resource_lock

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / p10.BASE
CONFIG = ROOT / 'configs/jaes_v2/phase10/qualification_v2.json'
HISTORICAL = ROOT / 'outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1'


def resource_report():
    import winreg
    from importlib.metadata import version
    import platform
    measurement = snapshot(ROOT)
    pagefile = {}
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                       r'SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management') as key:
        for name in ('PagingFiles', 'ExistingPageFiles'):
            try:
                pagefile[name] = winreg.QueryValueEx(key, name)[0]
            except FileNotFoundError:
                pagefile[name] = None
    return dict(measured_at=datetime.now().astimezone().isoformat(), **measurement,
                pagefile=pagefile, temp=os.getenv('TEMP'), tmp=os.getenv('TMP'),
                cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                native_threads={name: os.getenv(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
                python=platform.python_version(), packages={p: version(p) for p in ('tensorflow', 'keras', 'numpy', 'soundfile')},
                issue=pressure(measurement), status='RESOURCE_BLOCKED' if pressure(measurement) else 'RESOURCE_PREFLIGHT_PASS')


def evidence_contract():
    return dict(source=p10.fingerprints(ROOT), acceptance_sha256=file_hash(CONFIG))


def validation_parity(output):
    require_resources(ROOT)
    startup = perf_counter()
    tf = p10.phase10_tensorflow()
    c = p10.qualification_config(ROOT, '100')
    _, records, b_ids, _ = prior.dataset(ROOT, c)
    sealed = prior.verify_tree(HISTORICAL)
    checkpoint = HISTORICAL / p10.read(HISTORICAL / 'best.json')['path']
    if checkpoint.name not in sealed['output_hashes']:
        raise ValueError('Selected model absent from seal')
    model = tf.keras.models.load_model(checkpoint, compile=False)
    standard = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    optimized = prior.segmentwise_inference(tf, model)
    inputs = prior.Inputs(ROOT, c)
    startup = perf_counter() - startup
    both, timing = score(records, b_ids, 'C', inputs, dict(standard=standard, optimized=optimized),
                         lambda: require_resources(ROOT))
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in both.items():
        write_csv(output.with_name(f'{output.stem}_{name}.csv'), rows)
    result = compare(both['standard'], both['optimized'], **p10.read(CONFIG)['validation'])
    return dict(**evidence_contract(), **result, sources=len(records), expected_rows=len(records) * len(prior.DEV),
                checkpoint_sha256=file_hash(checkpoint), startup_seconds=startup,
                timing=timing, input_timing=inputs.timings, resources=snapshot(ROOT),
                calibration_used=False, official_evaluation_used=False)


def integrity():
    from src.lava.jaes_v2.runtime import preflight
    inventory = p10.read(ROOT / 'docs/jaes_v2/LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE9.json')
    checked = []
    def visit(obj):
        if isinstance(obj, dict):
            if {'path', 'sha256', 'bytes'} <= obj.keys():
                path = prior.safe_path(ROOT, obj['path'])
                if path.stat().st_size != obj['bytes'] or file_hash(path) != obj['sha256']:
                    raise ValueError(f'Phase 9 inventory artifact changed: {path}')
                checked.append(obj['path'])
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)
    visit(inventory)
    historical = preflight(ROOT)
    sealed = {}
    for run in sorted(HISTORICAL.parent.glob('p8_medium_*_s6107_v1')):
        if run.name not in {v['experiment_id'] for v in inventory['selected_models'].values()}:
            continue
        seal = prior.verify_tree(run)
        sealed[run.name] = dict(entries=len(seal['output_hashes']), sha256=file_hash(run / 'seal.json'))
    if len(sealed) != 2:
        raise ValueError('Both historical runs are required')
    return dict(phase9_entries=len(checked), seals=sealed, legacy=historical['baseline_freeze_verification'],
                **evidence_contract())


def historical_scores():
    """Exercise the prospective cohort and guards with real sealed predictions."""
    c = p10.qualification_config(ROOT, '100')
    _, records, b_ids, _ = prior.dataset(ROOT, c)
    plans = {tier: list(plan(records, b_ids, tier)) for tier in ('A', 'B', 'C')}
    paths = [HISTORICAL, HISTORICAL.parent / 'p8_medium_channel_partial_s6107_v1']
    tier_b = []
    summaries = {}
    for directory in paths:
        prior.verify_tree(directory)
        csv = read_csv(directory / 'scores.csv')
        by_key = {(r['sample_id'], r['condition_id']): r for r in csv}
        expected = {(r.sample_id, condition) for r, condition, _ in plans['C']}
        if len(csv) != len(by_key) or by_key.keys() != expected:
            raise ValueError('Incomplete or duplicate historical score rows')
        normalized = {}
        for record, condition, stratum in plans['C']:
            key = record.sample_id, condition
            row = by_key[key]
            p = float(row['raw_score'])
            if (int(row['label']) != record.label or row['attack_id'] != record.attack_id
                    or row['speaker_id'] != record.speaker_id or row['stratum'] != stratum
                    or float(row['threshold']) != .5 or int(row['predicted_label']) != int(p >= .5)):
                raise ValueError(f'Manifest/score identity mismatch: {key}')
            normalized[key] = dict(row, raw_score=p, label=record.label,
                                   threshold=.5, predicted_label=int(p >= .5),
                                   parent_recording_id=record.parent_recording_id, status='OK')
        rows_b = [normalized[r.sample_id, condition] for r, condition, _ in plans['B']]
        tier_b.append(rows_b)
        summaries[directory.name] = dict(counts=guard_counts(rows_b), metrics=metrics(list(normalized.values())),
                                         scores_sha256=file_hash(directory / 'scores.csv'))
    return dict(scope='historical_score_replay_no_model_inference', tier_rows={k: len(v) for k, v in plans.items()},
                development_sources=len(records), models=summaries,
                retrospective_guard=select(tier_b[1], tier_b[0], p10.MARGINS, p10.CAPS),
                **evidence_contract())


def compare_resume(profile):
    base = p10.qualification_id(profile)
    dirs = [OUT / base / role for role in ('uninterrupted', 'resumable')]
    for directory in dirs:
        prior.verify_tree(directory)
    a, b = [p10.read(directory / 'result.json') for directory in dirs]
    if any(result['state']['contract']['source'] != p10.fingerprints(ROOT) for result in (a, b)):
        raise ValueError('Comparator source differs from the current acceptance contract')
    fields = ('contract', 'step', 'epoch', 'cursor', 'micro', 'journal', 'history',
              'best', 'bad_epochs', 'frozen_hash', 'model_hash', 'optimizer_hash')
    match = {field: a['state'][field] == b['state'][field] for field in fields}
    match.update({k: a[k] == b[k] for k in ('final_weights', 'selected_weights', 'scores_hash', 'metrics')})
    steps = p10.qualification_config(ROOT, profile).maximum_updates
    stop_at = p10.read(CONFIG)['benchmark_split_step' if profile == '100' else 'lifecycle_split_step']
    event = dirs[1] / 'resume_events' / f'u{stop_at:06d}.json'
    split = event.is_file() and any(
        file_hash(path) == p10.read(event)['controller_sha256'] and p10.read(path)['step'] == stop_at
        for path in (dirs[1] / 'checkpoints').glob('*/controller.json'))
    return dict(**evidence_contract(), pass_thresholds=all(match.values()) and a['state']['step'] == steps and split,
                steps=steps, interrupted_at=stop_at, matches=match,
                archive_hash_equal=file_hash(dirs[0] / 'selected.keras') == file_hash(dirs[1] / 'selected.keras'))


def require_report(name):
    report = p10.read(OUT / name)
    if not report['pass_thresholds'] or any(report.get(k) != v for k, v in evidence_contract().items()):
        raise ValueError(f'Missing or stale acceptance: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('resources', 'integrity', 'historical-scores', 'validation-parity', 'numeric-parity',
                                         'qualify', 'resume', 'compare-resume', 'monitor'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', choices=('lifecycle', '100'), default='lifecycle')
    parser.add_argument('--role', choices=('uninterrupted', 'resumable'), default='uninterrupted')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    if output and (not output.is_relative_to(OUT) or output.suffix != '.json' or output.exists()):
        parser.error('Use a new JSON output under outputs/jaes_v2/phase10')
    if args.mode == 'resources':
        result = resource_report()
    elif args.mode == 'integrity':
        result = integrity()
    elif args.mode == 'historical-scores':
        result = historical_scores()
    elif args.mode == 'compare-resume':
        result = compare_resume(args.profile)
    elif args.mode == 'monitor':
        base = p10.qualification_id(args.profile)
        result = p10.read(OUT / base / args.role / 'status.json')
    else:
        if args.mode in ('validation-parity', 'numeric-parity') and output is None:
            parser.error('--output required for parity evidence')
        require_resources(ROOT)
        tmp = OUT / 'tmp'
        tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp):
            pass
        os.environ.update(TEMP=str(tmp), TMP=str(tmp))
        tempfile.tempdir = str(tmp)
        with resource_lock(ROOT):
            if args.mode == 'validation-parity':
                result = validation_parity(output)
            elif args.mode == 'numeric-parity':
                from scripts.jaes_v2.phase10_numeric import numeric_parity
                result = numeric_parity(ROOT, p10.read(CONFIG)['numeric'], evidence_contract())
            else:
                require_report('numeric_parity_v2.json')
                if args.profile == '100':
                    require_report('lifecycle_resume_v2.json')
                    require_report('validation_parity_v2.json')
                if args.pause and (args.role != 'resumable' or args.mode == 'resume'):
                    parser.error('--pause is for the first resumable invocation only')
                stop = p10.read(CONFIG)['benchmark_split_step' if args.profile == '100' else 'lifecycle_split_step']
                p10.run(ROOT, args.profile, args.role, resume=args.mode == 'resume', stop_at=stop if args.pause else None)
                return
    if output:
        write_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
