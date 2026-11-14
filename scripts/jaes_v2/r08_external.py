"""R08 metadata, gated bounded acquisition, development replay and external scoring."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gc
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import mean, stdev
from zipfile import ZipFile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

import numpy as np
import soundfile as sf

from scripts.jaes_v2 import r08_data as data
from scripts.jaes_v2.empirical_readiness import models_and_data, write_once
from scripts.jaes_v2.final_score_diagnostic import bootstrap_auc, cluster_delta, draws, interval, metrics, transitions
from scripts.jaes_v2.phase6_experiment import resource_lock
from scripts.jaes_v2.phase10_experiment import ROOT, resource_report
from scripts.jaes_v2.size1280_profile import FOUNDATION, LOCK
from src.lava.jaes_v2 import phase10_training as engine, phase8_training as prior
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv, write_csv
from src.lava.jaes_v2.phase10_resources import MIN_D_FREE, WRITE_RESERVE, require_resources

OUT = ROOT / 'outputs/jaes_v2/r08_external_v1'
CONTRACT = ROOT / 'configs/jaes_v2/r08_external_v1/protocol.json'
RIGHTS = ROOT / 'docs/jaes_v2/R08_DATA_RIGHTS_AND_PROVENANCE.md'
ACCESS = OUT / 'author_data_access_approval.json'
SCORING = OUT / 'author_scoring_approval.json'
SHAPE = [6, 224, 224, 3]
SEEDS = (9107, 9207, 9307)
METHODS = ('erm', 'channel')
BOOT_SEED = 91109


def fingerprint():
    return {**engine.fingerprints(ROOT), **{p.relative_to(ROOT).as_posix(): file_hash(p)
            for p in (Path(__file__), Path(data.__file__),
                      ROOT / 'scripts/jaes_v2/final_score_diagnostic.py',
                      ROOT / 'scripts/jaes_v2/empirical_readiness.py')}}


def runtime():
    return dict(python=platform.python_version(), platform=platform.platform(), libsndfile=sf.__libsndfile_version__,
                packages={p: version(p) for p in ('tensorflow', 'keras', 'numpy', 'soundfile', 'scipy',
                                                  'librosa', 'scikit-learn')})


def approval(path, scope, bindings):
    record = engine.read(path) if path.is_file() else {}
    if record.get('scope') != scope or any(record.get(k) != v for k, v in bindings.items()):
        raise ValueError(f'Missing or mismatched authorization: {path.name}')
    for key in ('author', 'approved_at', 'approval_reference'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f'Missing approval field: {key}')
    if datetime.fromisoformat(record['approved_at']).utcoffset() is None:
        raise ValueError('Approval timestamp needs a timezone')
    return record


def access():
    proof = OUT / 'data_use_confirmation.md'
    record = approval(ACCESS, 'AUTHOR_APPROVAL_R08_DATA_ACCESS', dict(
        dataset=data.RELEASE, revision=data.REVISION, rights_sha256=file_hash(RIGHTS),
        intended_use='LOCAL_RESEARCH_AND_AGGREGATE_REPORTING',
        destination='outputs/jaes_v2/r08_external_v1/audio'))
    if not proof.is_file() or file_hash(proof) != record.get('confirmation_sha256') or not proof.read_text(encoding='utf-8').strip():
        raise ValueError('Attach the actual written data-use confirmation and bind its SHA-256')
    cap = record.get('max_download_bytes')
    if type(cap) is not int or not 0 < cap <= data.ZIP_BYTES + data.META_BUDGET:
        raise ValueError('Invalid download budget')
    if record.get('method') not in ('range', 'local-zip'):
        raise ValueError('Invalid approved acquisition method')
    return record


def overlap_baseline():
    files = [ROOT / FOUNDATION / f'{part}_manifest.csv' for part in ('train', 'development', 'calibration')]
    files += sorted((ROOT / 'outputs/jaes_v2/datasets/mlaad_tiny_dev_v1').glob('*manifest.csv'))
    records = [row for file in files for row in read_csv(file)]
    held = ROOT / LOCK
    records += engine.read(held)['evaluation_manifest']['rows']
    ids = {Path(str(r.get(k, '')).replace('\\','/').split(':')[-1]).stem.casefold() for r in records
           for k in ('sample_id', 'official_recording_id', 'source_recording_id')}
    speakers = {data.speaker_key(r.get(k, '')) for r in records
                for k in ('speaker_id', 'source_speaker_id', 'target_speaker_id')}
    hashes = {r.get('sha256') or r.get('checksum') for r in records}
    manifests = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [*files, held]}
    return ids, speakers, hashes, manifests


def guard(bytes_needed=0):
    measure = require_resources(ROOT)
    if measure['d_free'] < MIN_D_FREE + WRITE_RESERVE + bytes_needed:
        raise ValueError('Insufficient D space above the frozen reserve')
    return measure


@contextmanager
def archive_for(record=None):
    if record and record['method'] == 'local-zip':
        path = OUT / 'release_in_the_wild.zip'
        if record['max_download_bytes'] < data.ZIP_BYTES or path.stat().st_size != data.ZIP_BYTES or file_hash(path) != data.ZIP_SHA256:
            raise ValueError('Unapproved budget or full-archive checksum mismatch')
        with ZipFile(path) as archive:
            yield archive, None
    else:
        budget = data.META_BUDGET if record is None else record['max_download_bytes']
        ledger = OUT / 'transfer'
        previous = sum(engine.read(p)['bytes'] for p in ledger.glob('*.json')) if record else 0
        def debit(count):
            if record:
                write_once(ledger / f'{datetime.now().strftime("%Y%m%dT%H%M%S%f")}.json',
                           dict(bytes=count, approval_sha256=file_hash(ACCESS)))
        with data.RemoteZip(budget - previous, guard, debit=debit) as stream:
            with ZipFile(stream) as archive:
                yield archive, stream


def metadata():
    with archive_for() as (archive, stream):
        rows, _ = data.inventory(archive)
        _, support = data.shortlist(rows, 0, 16)
        return dict(status='PINNED_METADATA_VERIFIED', revision=data.REVISION, metadata_sha256=data.META_SHA256,
                    archive_bytes=data.ZIP_BYTES, etag=data.ETAG, audio_bytes=0,
                    metadata_transfer_bytes=stream.transferred, classes=dict(Counter(str(r['label']) for r in rows)),
                    speaker_labels=len({r['speaker_id'] for r in rows}), normalized_support=support,
                    speakers_with_16_each=sum(min(n) >= 16 for n in support.values()),
                    speakers_with_8_each=sum(min(n) >= 8 for n in support.values()))


def atomic_audio(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    tmp.write_bytes(payload)
    tmp.replace(path)


def audio_path(row):
    data.member_name(row['sample_id'])
    return prior.safe_path(OUT, 'audio/' + row['sample_id'])


def check_unique(rows, known=()):
    for field in ('sample_id', 'sha256', 'decoded_sha256'):
        if len({r[field] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate selected {field}')
    if any(r['sha256'] in known for r in rows):
        raise ValueError('Exact audio overlap with protected cohorts')
    groups = [r['source_group'] for r in rows if r['source_group'] is not None]
    if len(set(groups)) != len(groups):
        raise ValueError('Repeated known source group')


def acquire(pilot):
    record = access()
    receipt = engine.read(OUT / 'pilot.json') if not pilot else None
    binding = file_hash(ACCESS)
    if receipt and (receipt['access_sha256'] != binding or receipt['status'] != 'DECODE_PILOT_PASS'):
        raise ValueError('Current access authorization needs its own decode pilot')
    ids, speakers, hashes, baseline = overlap_baseline()
    aliases_path = OUT / 'speaker_aliases.json'
    aliases = engine.read(aliases_path) if aliases_path.is_file() else {}
    if any(data.speaker_key(k) != k or data.speaker_key(v) != v for k, v in aliases.items()):
        raise ValueError('Alias mapping must use canonical values')
    if record.get('speaker_aliases_sha256') != (file_hash(aliases_path) if aliases_path.is_file() else None):
        raise ValueError('Alias mapping differs from the approved input')
    with archive_for(record) as (archive, stream):
        rows, members = data.inventory(archive)
        _, support = data.shortlist(rows, 0, 16, ids, speakers, aliases)
        count, quota = (4, 4) if pilot else (16, 16)
        fallback_reason = None
        if not pilot and sum(min(n) >= quota for n in support.values()) < count:
            fallback_reason = 'Fewer than 16 metadata-eligible speakers with 16 clips of each class'
            quota = 8
        chosen, _ = data.shortlist(rows, count, quota, ids, speakers, aliases)
        guard(sum(members[r['member']].file_size for r in chosen if not audio_path(r).is_file()) + data.MAX_MEMBER)
        plan_path = OUT / ('pilot_plan.json' if pilot else 'subset_plan.json')
        plan = dict(rows=chosen, access_sha256=binding, source=fingerprint(), overlap_manifests=baseline,
                    selection_seed=data.SEED, metadata_sha256=data.META_SHA256, profile='pilot' if pilot else 'external')
        if plan_path.is_file():
            if engine.read(plan_path) != plan:
                raise ValueError('Acquisition resume plan changed')
        else:
            write_once(plan_path, plan)
        valid = []
        for row in chosen:
            path = audio_path(row)
            info = members[row['member']]
            try:
                guard(info.file_size)
                if path.is_file():
                    import zlib
                    if path.stat().st_size != info.file_size or path.stat().st_size > data.MAX_MEMBER:
                        raise ValueError('Cached WAV size differs')
                    payload = path.read_bytes()
                    if zlib.crc32(payload) != info.CRC:
                        raise ValueError('Cached WAV CRC differs')
                else:
                    payload = data.read_member(archive, row)
                audio = data.qualified_audio(payload)
                if audio['shape'] != SHAPE:
                    raise ValueError(f'Unexpected P0 shape: {audio["shape"]}')
                valid.append(dict(row, **audio))
                check_unique(valid, hashes)
                if not path.exists():
                    atomic_audio(path, payload)
            except (ValueError, OSError, RuntimeError) as exc:
                failure = OUT / 'acquisition_failures' / f'{datetime.now().strftime("%Y%m%dT%H%M%S%f")}.json'
                write_once(failure, dict(member=row['member'], reason=str(exc), acquisition_stage=plan['profile']))
                raise
        return dict(status='DECODE_PILOT_PASS' if pilot else 'SUBSET_AUDIO_PASS', rows=valid,
                    rows_sha256=object_hash(valid), access_sha256=binding, source=plan['source'],
                    overlap_manifests=baseline, fallback_reason=fallback_reason,
                    selection_seed=data.SEED, real=sum(r['label']==0 for r in valid),
                    fake=sum(r['label']==1 for r in valid), speakers=count,
                    audio_bytes=sum(r['bytes'] for r in valid),
                    remote_transfer_bytes=stream.transferred if stream else 0, final_resources=guard(),
                    lineage='Original source and parent unverified. No known mapped overlap under available checks.',
                    invalid_files=0, decoder=runtime(), external_predictions=0,
                    overlap_scope='Available manifest recording IDs, canonical and supplied speaker aliases, exact bytes. '
                    'Decoded duplicates checked within selection. Near-duplicates and external parent links unverified.')


def restore(tf, entry):
    path = prior.safe_path(ROOT, entry['model_path'])
    if file_hash(path) != entry['model_sha256']:
        raise ValueError('Frozen model checksum mismatch')
    model = tf.keras.models.load_model(path, compile=False)
    if list(model.input_shape[1:]) != SHAPE or prior.variable_hash(model.weights) != entry['weight_sha256']:
        raise ValueError('Frozen input or restored weight mismatch')
    return model, prior.segmentwise_inference(tf, model)


def score_value(value):
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError('Invalid P(FAKE)')
    return result, int(result >= .5)


def replay():
    tf = engine.phase10_tensorflow()
    fixtures = {}
    models = []
    tolerance = engine.read(ROOT / 'configs/jaes_v2/phase10/qualification_v2.json')['validation']['max_score']
    for seed in SEEDS:
        guard()
        runs, dev, b_ids, config, _ = models_and_data(ROOT, seed)
        examples = [min((r for r in dev if r.sample_id not in b_ids and r.label == y), key=lambda r: r.sample_id)
                    for y in (0, 1)]
        examples += [min((r for r in dev if r.sample_id in b_ids), key=lambda r: r.sample_id)]
        examples.sort(key=lambda r: r.sample_id)
        for method in METHODS:
            path, contract, state, result = runs[method]
            if contract['config']['preprocessing_policy'] != 'P0' or contract['threshold'] != .5:
                raise ValueError('Model P0 or threshold mismatch')
            entry = dict(seed=seed, method=method, id=f'{method}_{seed}', checkpoint=state['best']['tag'],
                         model_path=(path/'selected.keras').relative_to(ROOT).as_posix(),
                         model_sha256=file_hash(path/'selected.keras'), weight_sha256=result['selected_weights'],
                         seal_sha256=file_hash(path/'seal.json'))
            model, forward = restore(tf, entry)
            archived = {r['sample_id']: r for r in read_csv(path/'scores.csv') if r['condition_id']=='clean'}
            audit = []
            inputs = prior.Inputs(ROOT, config)
            for row in examples:
                guard()
                x = data.clean_input(prior.source_path(ROOT, row.path))
                if not np.array_equal(x, inputs.get(row, 'clean')):
                    raise ValueError('P0 input parity failed')
                a, label = score_value(forward(x[None]).numpy().item())
                b, _ = score_value(forward(x[None]).numpy().item())
                expected = archived[row.sample_id]
                delta = abs(a - float(expected['raw_score']))
                if a != b or delta > tolerance or label != int(expected['predicted_label']) or row.label != int(expected['label']):
                    raise ValueError('Development deterministic score replay failed')
                audit.append(dict(sample_id=row.sample_id, label=row.label, score=a, absolute_error=delta))
            fixtures[entry['id']] = audit
            models.append(entry)
            del forward, model
            tf.keras.backend.clear_session()
            gc.collect()
    return dict(status='DEVELOPMENT_REPLAY_PASS', models=models, fixtures=fixtures, tolerance=tolerance,
                source=fingerprint(), runtime=runtime(), external_predictions=0, training_updates=0)


def verify_rows(rows):
    check_unique(rows)
    for row in rows:
        if file_hash(audio_path(row)) != row['sha256']:
            raise ValueError('Locked audio checksum mismatch')


def lock_protocol():
    access()
    subset, rep = [engine.read(OUT/p) for p in ('subset.json', 'inference_qualification.json')]
    rows = subset['rows']
    if (subset['status'] != 'SUBSET_AUDIO_PASS' or subset['access_sha256'] != file_hash(ACCESS)
            or subset['rows_sha256'] != object_hash(rows) or rep['status'] != 'DEVELOPMENT_REPLAY_PASS'
            or rep['source'] != fingerprint() or subset['source'] != rep['source'] or rep['runtime'] != runtime()):
        raise ValueError('Data or inference qualification mismatch')
    quota = 8 if subset['fallback_reason'] else 16
    support = Counter((r['speaker_id'], r['label']) for r in rows)
    if (len(support) != 32 or len({s for s, _ in support}) != 16 or set(support.values()) != {quota}
            or any(r['label'] not in (0,1) or r['shape'] != SHAPE for r in rows)
            or [m['id'] for m in rep['models']] != [f'{m}_{s}' for s in SEEDS for m in METHODS]):
        raise ValueError('Locked class/speaker quotas or model list changed')
    for name, digest in subset['overlap_manifests'].items():
        if file_hash(prior.safe_path(ROOT,name)) != digest:
            raise ValueError('Overlap reference changed')
    verify_rows(rows)
    return dict(schema=1, evaluation_id='r08_external_v1', dataset=data.RELEASE, revision=data.REVISION,
                metadata_sha256=data.META_SHA256, publisher_archive_sha256=data.ZIP_SHA256,
                selected_rows_sha256=object_hash(rows), rows=rows, models=rep['models'], source=rep['source'],
                subset_sha256=file_hash(OUT/'subset.json'), replay_sha256=file_hash(OUT/'inference_qualification.json'),
                data_use_review_sha256=file_hash(RIGHTS), data_access_sha256=file_hash(ACCESS),
                runtime=rep['runtime'], condition='native', preprocess='P0', max_seconds=3, batch_size=1,
                orientation='P(FAKE)', threshold=.5, predictions=len(rows)*len(rep['models']),
                scope='CROSS_DATASET_DIAGNOSTIC_WITH_LINEAGE_LIMITATIONS', bootstrap={'draws':10000,'seed':BOOT_SEED,
                'publication_intervals':False,'reason':'Parent/source dependence not verified across speaker groups'},
                resource_policy='Existing Phase 10 CPU reserve thresholds', overlap_manifest_hashes=subset['overlap_manifests'])


def load_lock():
    lock = engine.read(CONTRACT)
    if lock != lock_protocol():
        raise ValueError('Protocol differs from current qualified data, source or models')
    approval(SCORING, 'AUTHOR_APPROVAL_R08_EXTERNAL_SCORING', dict(
        dataset=data.RELEASE, revision=data.REVISION, protocol_sha256=file_hash(CONTRACT),
        selected_rows_sha256=lock['selected_rows_sha256'], predictions=lock['predictions'],
        model_hashes={m['id']:m['model_sha256'] for m in lock['models']},
        overlap_audit_sha256=lock['subset_sha256'], rights_sha256=lock['data_use_review_sha256']))
    for model in lock['models']:
        p = prior.safe_path(ROOT, model['model_path'])
        if file_hash(p)!=model['model_sha256'] or file_hash(p.parent/'seal.json')!=model['seal_sha256']:
            raise ValueError('Model or its seal changed')
    return lock


def identity(lock, entry, row):
    return dict(evaluation_id=lock['evaluation_id'], protocol_sha256=file_hash(CONTRACT),
                model_id=entry['id'], seed=entry['seed'], method=entry['method'], checkpoint=entry['checkpoint'],
                model_sha256=entry['model_sha256'], sample_id=row['sample_id'], speaker_id=row['speaker_id'],
                source_group=row['source_group'], label=row['label'], audio_sha256=row['sha256'],
                condition='native', preprocess='P0', threshold=.5, status='OK')


def check_prediction(row, expected):
    p, pred = score_value(row['raw_score'])
    if row != dict(expected, raw_score=p, predicted_label=pred):
        raise ValueError('Prediction identity or threshold mismatch')
    return dict(row, p=p, y=expected['label'], error=int(pred != expected['label']))


def score(lock):
    out = OUT/'run'
    if (out/'seal.json').exists():
        raise FileExistsError('Evaluation already sealed')
    tf = engine.phase10_tensorflow()
    for entry in lock['models']:
        if all(shard(out, entry, row).is_file() for row in lock['rows']):
            for row in lock['rows']:
                check_prediction(engine.read(shard(out, entry, row)), identity(lock, entry, row))
            continue
        guard()
        model, forward = restore(tf, entry)
        for row in lock['rows']:
            guard()
            expected = identity(lock, entry, row)
            path = shard(out, entry, row)
            if path.exists():
                check_prediction(engine.read(path), expected)
                continue
            audio = audio_path(row)
            if file_hash(audio) != row['sha256']:
                raise ValueError('Audio changed before scoring')
            x = data.clean_input(audio)
            p, pred = score_value(forward(x[None]).numpy().item())
            write_once(path, dict(expected, raw_score=p, predicted_label=pred))
        if prior.variable_hash(model.weights) != entry['weight_sha256']:
            raise ValueError('Inference changed weights')
        del forward, model
        tf.keras.backend.clear_session()
        gc.collect()


def shard(out, entry, row):
    return out/'shards'/entry['id']/(sha256(row['sample_id'].encode()).hexdigest()+'.json')


def summarize(parts, bootstrap=False):
    speakers = sorted({r['speaker_id'] for r in next(iter(parts.values()))})
    weights = draws(speakers, np.random.default_rng(BOOT_SEED)) if bootstrap else None
    per_model, paired = {}, {}
    for model, rows in parts.items():
        m = metrics(rows)
        fp, fn = m['false_alarm'], m['miss']
        m['fpr'] = fp['errors']/fp['n'] if fp['n'] else None
        m['fnr'] = fn['errors']/fn['n'] if fn['n'] else None
        m['confusion'] = dict(tn=fp['n']-fp['errors'], fp=fp['errors'], fn=fn['errors'], tp=fn['n']-fn['errors'])
        per_model[model] = m
    for seed in SEEDS:
        a, b = [f'{method}_{seed}' for method in METHODS]
        left, right = parts[a], parts[b]
        if [r['sample_id'] for r in left] != [r['sample_id'] for r in right]:
            raise ValueError('Unpaired sources')
        delta = {k: per_model[b][k]-per_model[a][k] if None not in (per_model[b][k],per_model[a][k]) else None
                 for k in ('auc','fpr','fnr')}
        pairs = {}
        for y in (0,1):
            first, second = [[r for r in part if r['y']==y] for part in (left,right)]
            pairs[str(y)] = transitions([r['error'] for r in first],[r['error'] for r in second])
            if weights is not None and first:
                pairs[str(y)]['speaker_interval'] = cluster_delta(first,second,speakers,weights)
        auc_ci = interval(bootstrap_auc(right,speakers,weights)-bootstrap_auc(left,speakers,weights)) if weights is not None else None
        paired[str(seed)] = dict(delta=delta, class_transitions=pairs, auc_interval=auc_ci)
    stats = {}
    for key in ('auc','fpr','fnr'):
        values = [x['delta'][key] for x in paired.values() if x['delta'][key] is not None]
        stats[key] = dict(mean=mean(values) if values else None, sample_sd=stdev(values) if len(values)>1 else None)
    return dict(models=per_model, paired=paired, across_seed_delta=stats, speakers=len(speakers))


def finalize(lock):
    out = OUT/'run'
    parts = {}
    expected_paths = set()
    for entry in lock['models']:
        rows = []
        for row in lock['rows']:
            path = shard(out,entry,row)
            expected_paths.add(path)
            rows.append(check_prediction(engine.read(path), identity(lock,entry,row)))
        parts[entry['id']] = rows
    if set((out/'shards').rglob('*.json')) != expected_paths or len(expected_paths) != lock['predictions']:
        raise ValueError('Incomplete or extra prediction shards')
    development, heldout = reference_metrics()
    report = dict(status=lock['scope'],protocol_sha256=file_hash(CONTRACT),
                  scoring_approval_sha256=file_hash(SCORING), predictions=len(expected_paths),
                  bootstrap=lock['bootstrap'], reference_development_clean=development,
                  reference_within_release_clean=heldout, **summarize(parts))
    final = out/'metrics.json'
    if (out/'seal.json').exists():
        prior.verify_tree(out)
        if engine.read(final) != report:
            raise ValueError('Sealed external metrics replay differs')
        return report
    rows = [{k:v for k,v in row.items() if k not in ('p','y','error')} for part in parts.values() for row in part]
    target = out/'scores.csv'
    pending = target.with_suffix('.pending')
    write_csv(pending,rows)
    pending.replace(target)
    if final.exists():
        if engine.read(final) != report:
            raise ValueError('Existing metrics differ')
    else:
        write_once(final,report)
    prior.seal_tree(out)
    prior.verify_tree(out)
    return report


def reference_metrics():
    base = ROOT / engine.BASE
    held = base/'empirical_execution_v1/heldout_28f940ea7d151177/metrics.json'
    baseline = engine.read(OUT/'protected_before.json')['files']
    if file_hash(held) != baseline[held.relative_to(ROOT).as_posix()]:
        raise ValueError('Historical held-out metrics changed')
    dev = {}
    for seed in SEEDS:
        for method in METHODS:
            path = base/f'p10_medium_{method}_s{seed}_v2/scientific'
            result = path/'result.json'
            sealed = engine.read(path/'seal.json')['output_hashes']
            if file_hash(result) != sealed['result.json']:
                raise ValueError('Historical development metrics changed')
            dev[f'{method}_{seed}'] = engine.read(result)['metrics']['A_PAIRED/clean']
    return dev, {m:r['conditions']['clean'] for m,r in engine.read(held)['models'].items()}


def write_results(report):
    path = ROOT/'docs/jaes_v2/R08_CROSS_DATASET_EVALUATION_RESULTS.md'
    lines = ['# R08 cross-dataset results', '', report['status'], '',
             f"Verified predictions: {report['predictions']}. Speaker groups: {report['speakers']}.", '',
             '| Model | ROC-AUC | EER | REAL false alarms | FAKE misses | FPR | FNR | TN/FP/FN/TP |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for model,r in report['models'].items():
        a,b = r['false_alarm'],r['miss']
        counts = '/'.join(str(r['confusion'][k]) for k in ('tn','fp','fn','tp'))
        lines.append(f"| {model} | {r['auc']:.4f} | {r['eer']:.4f} | {a['errors']}/{a['n']} | {b['errors']}/{b['n']} | {r['fpr']:.4f} | {r['fnr']:.4f} | {counts} |")
    lines += ['', '| Seed | Channel−ERM AUC | ΔFPR | ΔFNR |',
              '| --- | ---: | ---: | ---: |']
    for seed, r in report['paired'].items():
        d = r['delta']
        lines.append(f"| {seed} | {d['auc']:+.4f} | {d['fpr']:+.4f} | {d['fnr']:+.4f} |")
    lines += ['', 'Paired transitions, across-seed mean and sample SD and original clean-domain references are in metrics.json.',
              'These native-view results use the locked 0.5 threshold. Across-corpus differences describe distribution shift.',
              'Parent lineage remains unresolved. Speaker-bootstrap publication intervals are withheld, and the three seeds are descriptive.', '']
    value = '\n'.join(lines)
    if path.exists():
        if path.read_text(encoding='utf-8') != value:
            raise ValueError('Existing R08 report differs from verified replay')
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(value)


def integrity():
    reference = engine.read(OUT/'protected_before.json')
    for relative, digest in reference['files'].items():
        p = ROOT/relative
        if file_hash(p)!=digest:
            raise ValueError(f'Protected evidence changed: {relative}')
        if p.name=='seal.json':
            prior.verify_tree(p.parent)
    if reference['source']!=engine.fingerprints(ROOT):
        raise ValueError('Frozen scientific source changed')
    held = ROOT/engine.BASE/'empirical_execution_v1/heldout_28f940ea7d151177'
    verified = engine.read(held/'analysis/verified_analysis.json')['integrity']['score_shards']
    if {p.name:file_hash(p) for p in (held/'scores').glob('*.json')} != verified:
        raise ValueError('Historical prediction shards changed')
    return dict(status='PROTECTED_INTEGRITY_PASS', files=reference['files'], full_run_seals=8,
                historical_heldout_shards=len(verified))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('resources','metadata','integrity','replay','pilot','acquire','lock','score','verify','scoring-template','zip-preflight'))
    mode = parser.parse_args().mode
    if mode in ('pilot','acquire','score','verify','lock','scoring-template','zip-preflight'):
        access()
    if mode == 'resources':
        result = resource_report()
    elif mode == 'integrity':
        result = integrity()
    else:
        with resource_lock(ROOT):
            guard()
            if mode in ('metadata','replay','pilot','acquire'):
                path = OUT / {'metadata':'metadata.json','replay':'inference_qualification.json',
                              'pilot':'pilot.json','acquire':'subset.json'}[mode]
                if path.exists():
                    raise FileExistsError(path)
                result = {'metadata':metadata,'replay':replay,
                          'pilot':lambda:acquire(True),'acquire':lambda:acquire(False)}[mode]()
                write_once(path,result)
            elif mode == 'lock':
                result = lock_protocol()
                write_once(CONTRACT,result)
            elif mode == 'zip-preflight':
                if access()['method'] != 'local-zip' or access()['max_download_bytes'] < data.ZIP_BYTES:
                    raise ValueError('Full ZIP is outside approved method or byte budget')
                result = guard(data.ZIP_BYTES + 512 * 1024**2)
            elif mode == 'scoring-template':
                lock = engine.read(CONTRACT)
                if lock != lock_protocol():
                    raise ValueError('Qualification inputs changed')
                result = dict(scope='AUTHOR_APPROVAL_R08_EXTERNAL_SCORING', dataset=data.RELEASE,
                    revision=data.REVISION, protocol_sha256=file_hash(CONTRACT),
                    selected_rows_sha256=lock['selected_rows_sha256'], predictions=lock['predictions'],
                    model_hashes={m['id']:m['model_sha256'] for m in lock['models']},
                    overlap_audit_sha256=lock['subset_sha256'], rights_sha256=lock['data_use_review_sha256'],
                    author=None, approved_at=None, approval_reference=None)
                write_once(OUT/'author_scoring_approval.template.json', result)
            else:
                lock = load_lock()
                if mode == 'score':
                    score(lock)
                result = finalize(lock)
                write_results(result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
