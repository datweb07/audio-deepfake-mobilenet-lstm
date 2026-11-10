"""Empirical readiness audit. The ZIP inventory reads directory metadata only."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import shutil
import tempfile

import numpy as np

from scripts.jaes_v2.phase6_experiment import resource_lock
from scripts.jaes_v2.phase10_experiment import integrity
from scripts.jaes_v2.phase10_medium import registered, preflight
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv, write_csv, write_json
from src.lava.jaes_v2.phase10_resources import require_resources, snapshot
from src.lava.jaes_v2.phase10_validation import STRESS

BASE = Path('outputs/jaes_v2/phase10')
OUT = BASE / 'empirical_execution_v1'
ARCHIVE = Path('data/downloads/LA.zip')
PROTOCOL = Path('configs/jaes_v2/empirical_protocol_v1.json')
CONDITIONS = ('clean', *STRESS)
METHODS = ('erm', 'channel')
ATTACKS = ('REAL', *(f'A{i:02}' for i in range(7,20)))


def stable_key(seed, value):
    return sha256(f'{seed}:{value}'.encode()).hexdigest()


def sample_heldout(records, seed, quotas, blocked_ids=(), blocked_speakers=()):
    """Score-blind speaker round robin. Input labels are supplied only after authorization."""
    if set(quotas) != set(ATTACKS):
        raise ValueError('Invalid frozen stratum list')
    seen = set(blocked_ids)
    buckets = {attack: defaultdict(list) for attack in ATTACKS}
    for record in records:
        identity, speaker, attack = (record[k] for k in ('sample_id','speaker_id','attack'))
        if (not identity or not speaker or identity in seen or speaker in blocked_speakers
                or record['split'] != 'evaluation' or attack not in buckets):
            raise ValueError('Invalid, duplicate or overlapping held-out metadata')
        seen.add(identity)
        buckets[attack][speaker].append(record)
    selected, support = [], {}
    for attack, groups in buckets.items():
        speakers = sorted(groups, key=lambda speaker:stable_key(seed,f'{attack}:{speaker}'))
        ordered = [sorted(groups[s],key=lambda row:stable_key(seed,row['sample_id'])) for s in speakers]
        picks = [group[i] for i in range(max(map(len,ordered),default=0)) for group in ordered if i < len(group)]
        chosen = picks[:quotas[attack]]
        selected.extend(chosen)
        support[attack] = dict(requested=quotas[attack],available=len(picks),selected=len(chosen))
    return selected, support


def inventory(root):
    with ZipFile(root/ARCHIVE) as archive:
        entries = archive.infolist()
        audio = [e for e in entries if '/ASVspoof2019_LA_eval/flac/' in e.filename and e.filename.endswith('.flac')]
        protocol = [e.filename for e in entries if e.filename.endswith('/ASVspoof2019.LA.cm.eval.trl.txt')]
        licenses = [e.filename for e in entries if 'license' in e.filename.lower() and not e.is_dir()]
        if len(set(e.filename for e in audio)) != len(audio):
            raise ValueError('Duplicate evaluation archive paths')
    foundation = engine.read(root/'outputs/jaes_v2/datasets/asv2019_foundation_v1/foundation.json')
    digest = file_hash(root/ARCHIVE)
    if digest != foundation['archive_sha256']:
        raise ValueError('Archive does not match the foundation record')
    return dict(path=ARCHIVE.as_posix(), sha256=digest, bytes=(root/ARCHIVE).stat().st_size,
                evaluation_audio_entries=len(audio), evaluation_expanded_bytes=sum(e.file_size for e in audio),
                protocol_paths=protocol, license_paths=licenses, inventory_only=True,
                calibration_manifest_present=(root/'outputs/jaes_v2/datasets/asv2019_foundation_v1/calibration_manifest.csv').is_file(),
                extracted_evaluation_present=(root/'data/public/asvspoof2019_la_v1/LA/ASVspoof2019_LA_eval').is_dir())


def models_and_data(root, seed):
    _, config = registered(f'p10_medium_erm_s{seed}_v2')
    train, dev, b_ids, audit = prior.dataset(root,config)
    digest=object_hash([r.to_dict() for r in train+dev])
    runs={}
    for method in METHODS:
        path=root/BASE/f'p10_medium_{method}_s{seed}_v2/scientific'
        prior.verify_tree(path)
        contract=engine.read(path/'contract.json')
        _, state=engine.committed_controller(path,contract)
        result=engine.read(path/'result.json')
        best=path/'checkpoints'/state['best']['tag']
        prior.verify_tree(best)
        if (state!=result['state'] or state['step']!=840 or state['epoch']!=21
                or contract['actual_sources']!=digest or contract['source']!=engine.fingerprints(root)
                or engine.read(best/'controller.json')['model_hash']!=result['selected_weights']
                or engine.read(path/'status.json')['status']!='SCIENTIFIC_COMPLETE'):
            raise ValueError('Scientific run integrity mismatch')
        runs[method]=(path,contract,state,result)
    _, first, left, _=runs['erm']
    _, second, right, _=runs['channel']
    if ({**first,'experiment':second['experiment'],'config':{**first['config'],'augmentation_policy':'channel'}}!=second
            or [[s[0] for s in j['sources']] for j in left['journal']] != [[s[0] for s in j['sources']] for j in right['journal']]):
        raise ValueError('Pair differs outside registered augmentation')
    return runs,dev,b_ids,config,audit


def write_once(path, value):
    if path.exists():
        raise FileExistsError(path)
    write_json(path,value)


def verify_evaluation_lock(root, lock):
    """Fail closed before a caller can open any evaluation member."""
    if lock.get('authorized') is not True:
        raise ValueError('Held-out evaluation is not authorized')
    if tuple(lock.get('seeds', ())) not in ((9107,9207),(9107,9207,9307)):
        raise ValueError('Lock requires both complete paired seed blocks')
    if lock.get('subset') not in ('primary','fallback') or tuple(lock.get('conditions',())) != CONDITIONS:
        raise ValueError('Evaluation conditions or subset differ from the protocol')
    if (not lock.get('author_approval_record') or not lock.get('resource_subset_reason')
            or lock.get('benchmark_sha256') != file_hash(root/OUT/'benchmark.json')):
        raise ValueError('Missing approval, subset rationale or measured budget evidence')
    files = lock.get('files', {})
    model_paths = {f'{BASE.as_posix()}/p10_medium_{method}_s{seed}_v2/scientific/selected.keras'
                   for seed in lock['seeds'] for method in METHODS}
    required = {PROTOCOL.as_posix(), ARCHIVE.as_posix(),
                'configs/jaes_v2/phase10/prospective_medium_v1.json',
                'scripts/jaes_v2/empirical_readiness.py', *model_paths}
    required.update(k for k in engine.fingerprints(root) if k != 'ffmpeg_binary')
    if not required <= files.keys() or lock.get('ffmpeg_sha256') != engine.fingerprints(root)['ffmpeg_binary']:
        raise ValueError('Incomplete source, data or checkpoint lock')
    for name, digest in files.items():
        if file_hash(prior.safe_path(root,name)) != digest:
            raise ValueError(f'Locked file mismatch: {name}')
    for seed in lock['seeds']:
        models_and_data(root,seed)


def commit_source_scores(directory, lock_hash, sample_id, rows, model_ids, conditions):
    """A complete per-source shard is the atomic unit of future technical resume."""
    key = stable_key(0,sample_id)
    path = directory/f'{key}.json'
    expected = {(model,condition) for model in model_ids for condition in conditions}
    actual = {(r['model'],r['condition']) for r in rows}
    if (actual != expected or len(actual) != len(rows)
            or any(r['sample_id'] != sample_id or not 0 <= r['raw_score'] <= 1
                   or r['predicted_label'] != int(r['raw_score'] >= .5) for r in rows)):
        raise ValueError('Duplicate, incomplete or invalid held-out score shard')
    record = dict(lock_sha256=lock_hash,sample_id=sample_id,rows=rows,rows_sha256=object_hash(rows))
    if path.exists():
        if engine.read(path) != record:
            raise ValueError('Committed score shard differs on resume')
        return False
    write_once(path,record)
    return True


def readiness(root):
    start=integrity()
    archive=inventory(root)
    runs,dev,b_ids,config,_=models_and_data(root,9107)
    plan=engine.read(root/PROTOCOL)
    registry_hash=file_hash(root/'configs/jaes_v2/phase10/prospective_medium_v1.json')
    if plan['registry_sha256']!=registry_hash or plan['heldout_authorized']:
        raise ValueError('Registry mismatch or unexpected evaluation authorization')
    candidates={}
    for method in METHODS:
        identity=f'p10_medium_{method}_s9207_v2'
        path=root/BASE/identity/'scientific'
        try:
            _, check=preflight(identity)
        except engine.ResourceStop as exc:
            check={'status':str(exc),'resources':snapshot(root)}
        candidates[method]=dict(run_exists=path.exists(),preflight=check)
    result=dict(timestamp=datetime.now(timezone(timedelta(hours=7))).isoformat(), archive=archive,
        integrity=start, resource=snapshot(root), seed9207=candidates,
        protocol_sha256=file_hash(root/PROTOCOL), source_sha256=file_hash(Path(__file__)),
        frozen_source_hashes=engine.fingerprints(root), models={
            method:dict(experiment=contract['experiment'],checkpoint=state['best']['tag'],
                        model_sha256=file_hash(path/'selected.keras'),seal_sha256=file_hash(path/'seal.json'),
                        score_sha256=file_hash(path/'scores.csv'))
            for method,(path,contract,state,_) in runs.items()},
        expected_dev=len(dev), real_only_dev=len(b_ids),
        heldout_status='BLOCKED_PENDING_PAIRED_REPLICATION_AND_EVALUATION_AUTHORIZATION',
        evaluation_labels_opened=False,evaluation_audio_opened=False,calibration_opened=False,
        sampling_executed=False,training_executed=False)
    write_once(root/OUT/'readiness.json',result)
    write_once(root/OUT/'heldout_lock_pending.json',dict(authorized=False,
        reason='Seed 9207 pair and explicit evaluation release pending',
        timestamp=result['timestamp'],protocol_sha256=result['protocol_sha256'],
        source_sha256=result['source_sha256'],archive_sha256=archive['sha256'],
        protocol_member_paths=archive['protocol_paths'], verified_9107_models=result['models'],
        seeds_pending=[9207],subset='UNDECIDED_RESOURCE_ONLY',conditions=CONDITIONS,
        sampling_seed=plan['heldout']['sampling_seed'], sampling_executed=False,
        evaluation_manifest=None, scores=None, metrics=None))


def benchmark(root):
    before=require_resources(root)
    runs,dev,b_ids,config,_=models_and_data(root,9107)
    selected=[]
    for group in ([r for r in dev if not r.label and r.sample_id not in b_ids],
                  [r for r in dev if r.label], [r for r in dev if r.sample_id in b_ids]):
        selected.extend(sorted(group,key=lambda r:stable_key(91109,r.sample_id))[:4])
    tf=engine.phase10_tensorflow()
    functions={}
    anchors={}
    for method,(path,_,_,result) in runs.items():
        model=tf.keras.models.load_model(path/'selected.keras',compile=False)
        if prior.variable_hash(model.weights)!=result['selected_weights']:
            raise ValueError('Benchmark model hash mismatch')
        functions[method]=prior.segmentwise_inference(tf,model)
        anchors[method]={(r['sample_id'],r['condition_id']):float(r['raw_score']) for r in read_csv(path/'scores.csv')}
    inputs=prior.Inputs(root,config)
    records=[]
    started=perf_counter()
    maximum_difference=0.
    for record in selected:
        for condition in CONDITIONS:
            require_resources(root)
            tensor=inputs.get(record,condition)[None]
            for method,forward in functions.items():
                clock=perf_counter()
                score=float(forward(tensor).numpy().reshape(-1)[0])
                seconds=perf_counter()-clock
                reference=anchors[method][record.sample_id,condition]
                difference=abs(score-reference)
                if not 0<=score<=1 or difference>3e-5 or (score>=.5)!=(reference>=.5):
                    raise ValueError('Benchmark score replay failed')
                maximum_difference=max(maximum_difference,difference)
                records.append(dict(method=method,sample_id=record.sample_id,condition_id=condition,
                                    raw_score=score,inference_seconds=seconds))
    out=root/OUT
    if (out/'benchmark_scores.csv').exists():
        raise FileExistsError('Benchmark already exists')
    write_csv(out/'benchmark_scores.csv',records)
    write_once(out/'benchmark.json',dict(scope='DEVELOPMENT_ONLY',sources=len(selected),predictions=len(records),
        conditions=CONDITIONS,input_seconds=inputs.timings,mean_forward_seconds=float(np.mean([r['inference_seconds'] for r in records])),
        total_seconds=perf_counter()-started, score_bytes=(out/'benchmark_scores.csv').stat().st_size,
        score_sha256=file_hash(out/'benchmark_scores.csv'), max_abs_replay=maximum_difference,before=before,after=snapshot(root),
        heldout_authorized=False))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('readiness','benchmark'))
    args=parser.parse_args()
    root=Path.cwd()
    with resource_lock(root):
        out=root/OUT
        out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            tempfile.tempdir=tmp
            try:
                (readiness if args.mode=='readiness' else benchmark)(root)
            finally:
                tempfile.tempdir=None
