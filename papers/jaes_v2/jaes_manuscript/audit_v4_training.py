"""Qualify the registered replication and derive a train-only size-study candidate."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import tempfile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

from scripts.jaes_v2.phase10_medium import registered, preflight
from scripts.jaes_v2.phase6_experiment import resource_lock
from scripts.jaes_v2.empirical_readiness import stable_key
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.evidence import read_csv, file_hash, object_hash, write_csv, write_json

BASE = Path('outputs/jaes_v2')
DEST = BASE/'supervisor_major_revision'
HELDOUT = BASE/'phase10/empirical_execution_v1/heldout_28f940ea7d151177/manifest.json'


def cohort(root):
    foundation = BASE/'datasets/asv2019_foundation_v1'
    medium = read_csv(root/prior.MEDIUM)
    train = read_csv(root/foundation/'train_manifest.csv')
    original = {r['sample_id']:r for r in medium if r['research_split']=='train'}
    pool = {r['sample_id']:r for r in train}
    if len(original)!=640 or not original.keys() <= pool.keys():
        raise ValueError('Medium subset is not nested')
    fields = ('checksum','label','speaker_id','attack_id')
    if any(any(row[k]!=pool[sample][k] for k in fields) for sample,row in original.items()):
        raise ValueError('Medium identity mismatch')
    selected = list(original.values())
    quotas = Counter((r['label'],r['attack_id']) for r in selected)
    occupied = {r['checksum'] for r in selected}
    for group, needed in sorted(quotas.items()):
        candidates = defaultdict(list)
        for r in train:
            if (r['label'],r['attack_id'])==group and r['sample_id'] not in original:
                candidates[r['speaker_id']].append(r)
        speakers = Counter(r['speaker_id'] for r in selected if (r['label'],r['attack_id'])==group)
        for queue in candidates.values():
            queue.sort(key=lambda r:stable_key(12809307,r['sample_id']),reverse=True)
        for _ in range(needed):
            while True:
                speaker = min((s for s,q in candidates.items() if q),key=lambda s:(speakers[s],s))
                row = candidates[speaker].pop()
                if row['checksum'] not in occupied:
                    break
            selected.append(row)
            occupied.add(row['checksum'])
            speakers[speaker]+=1
    blocked = {
        'development':read_csv(root/foundation/'development_manifest.csv'),
        'calibration':read_csv(root/foundation/'calibration_manifest.csv'),
        'heldout':engine.read(root/HELDOUT)['rows'],
    }
    overlap={}
    for role, rows in blocked.items():
        checks={'sample_id':'sample_id','speaker_id':'speaker_id','checksum':'checksum'}
        if role=='heldout':
            checks['checksum']='sha256'
        overlap[role]={}
        for k,m in checks.items():
            values={row.get(m) for row in rows}-{None,'','UNKNOWN'}
            if not values:
                raise ValueError(f'Missing overlap field: {role}/{m}, fields={list(rows[0])}')
            overlap[role][k]=len({r[k] for r in selected}&values)
        if any(overlap[role].values()):
            raise ValueError(f'Overlap with {role}')
    if len(occupied)!=1280 or Counter((r['label'],r['attack_id']) for r in selected)!= {k:2*v for k,v in quotas.items()}:
        raise ValueError('Unexpected class/attack support')
    for r in selected:
        if file_hash(prior.source_path(root,r['path']))!=r['checksum']:
            raise ValueError('Candidate audio bytes differ from manifest')
    path=root/DEST/'candidate_train1280.csv'
    write_csv(path,sorted(selected,key=lambda r:r['sample_id']))
    write_json(root/DEST/'size_candidate.json',dict(status='COHORT_AUDITED_PROFILE_NOT_QUALIFIED',
        sources=len(selected),original_sources=len(original),speakers=len({r['speaker_id'] for r in selected}),
        label_counts=Counter(r['label'] for r in selected),
        attack_counts=Counter(r['attack_id'] for r in selected if r['label']=='1'),
        speaker_counts=Counter(r['speaker_id'] for r in selected), overlap=overlap,
        manifest_sha256=file_hash(path), source_hashes={str(p):file_hash(root/p)
            for p in (Path(prior.MEDIUM),foundation/'train_manifest.csv',HELDOUT)},
        deterministic_selection_seed=12809307, global_generator_lineage_verified=False))


def smoke(root):
    import numpy as np
    proof = {}
    tf = engine.phase10_tensorflow()
    for method,condition in [('erm','clean'),('channel','noise10')]:
        exp=f'p10_medium_{method}_s9307_v2'
        if (root/BASE/'phase10'/exp).exists():
            raise FileExistsError(exp)
        config, audit = preflight(exp)
        train, dev, _, _ = prior.dataset(root,config)
        records = prior.epoch_order(train,config,0)
        record = train[records[0]]
        model, backbone, forward = prior.model_factory(tf,config)
        warmup, partial, variables, _, _ = engine.kernel(tf,backbone,model,forward)
        with tempfile.TemporaryDirectory(dir=root/DEST) as temporary:
            old = tempfile.tempdir
            tempfile.tempdir = temporary
            try:
                inputs = prior.Inputs(root,config)
                data = inputs.get(record,condition)[None]
                if not np.array_equal(data,inputs.get(record,condition)[None]):
                    raise ValueError('Nondeterministic input')
                args = (tf.convert_to_tensor(data),tf.constant([record.label]),tf.ones(1),tf.constant(0))
                losses = []
                for step in (warmup,partial):
                    loss, grads = step(*args)
                    if not np.isfinite(loss.numpy()) or not all(np.isfinite(g.numpy()).all() for g in grads):
                        raise ValueError('Nonfinite numerical smoke')
                    if not any(np.any(g.numpy()) for g in grads):
                        raise ValueError('Zero gradient smoke')
                    losses.append(float(loss.numpy()))
            finally:
                tempfile.tempdir = old
        proof[method]=dict(config_sha256=object_hash(config.__dict__),audit=audit,
            initial_sha256=prior.variable_hash(model.weights),first_order_sha256=object_hash([r.sample_id for r in (train[index] for index in records)]),
            actual_sources_sha256=object_hash([r.to_dict() for r in train+dev]),
            gradient_losses=losses,optimizer_updates=0)
    for key in ('initial_sha256','first_order_sha256','actual_sources_sha256'):
        if proof['erm'][key]!=proof['channel'][key]:
            raise ValueError(f'Matched seed mismatch: {key}')
    for seed in (9107,9207):
        historical=root/BASE/'phase10'/f'p10_medium_erm_s{seed}_v2/scientific/contract.json'
        if engine.read(historical)['initial_weights']==proof['erm']['initial_sha256']:
            raise ValueError('Independent initialization did not change')
    write_json(root/DEST/'seed9307_smoke.json',dict(status='PASS',methods=proof))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('cohort','smoke'))
    args=p.parse_args()
    root=Path.cwd()
    if args.mode=='cohort':
        cohort(root)
    else:
        with resource_lock(root):
            smoke(root)


