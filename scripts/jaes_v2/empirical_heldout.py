"""Deferred official LA inference. Requires a separate authorized, hash-bound lock."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import shutil
import tempfile
from time import perf_counter

import numpy as np

from scripts.jaes_v2.empirical_readiness import (
    ARCHIVE, ATTACKS, CONDITIONS, METHODS, OUT, commit_source_scores,
    models_and_data, stable_key, verify_evaluation_lock, write_once,
)
from scripts.jaes_v2.final_score_diagnostic import (
    SEED, bootstrap_auc, cluster_delta, describe, draws, interval, metrics, transitions,
)
from scripts.jaes_v2.phase6_experiment import resource_lock
from scripts.jaes_v2.heldout_protocol_audit import AUDITOR, FOUNDATION, PRELOCK, scoring_lock
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.controls import NoisePlan, prepare_headroom
from src.lava.jaes_v2.evidence import file_hash, object_hash
from src.lava.jaes_v2.phase10_resources import require_resources
from src.lava.jaes_v2.preprocessing import Preprocessor
from src.lava.jaes_v2.transforms import Transform, load_audio

def waveform_bank(payload, identity, ffmpeg):
    audio=load_audio(BytesIO(payload),sample_rate=22050,max_seconds=3)
    clean,_=prepare_headroom(audio,.25)
    seed=int(object_hash({'sample_id':'ASVspoof2019_LA:'+identity,'seed':5804,'version':'c2-view-1'})[:16],16)
    noise5,_=NoisePlan.build(clean,target_snr_db=5,seed=seed).insert(clean,'fixed_absolute')
    plan=NoisePlan.build(clean,target_snr_db=10,seed=seed)
    noisy,_=plan.insert(clean,'fixed_absolute')
    codec=Transform('opus24','codec',{'encoder':'libopus','bitrate':'24k','extension':'.ogg'})
    ab,_=codec.apply(noisy,seed=0,ffmpeg=ffmpeg)
    coded,_=codec.apply(clean,seed=0,ffmpeg=ffmpeg)
    ba,_=plan.insert(coded,'fixed_absolute')
    return dict(clean=clean,noise5=noise5,A_then_B=ab,B_then_A=ba)


def summarize(rows, model_ids, manifest_rows):
    speakers=sorted({r['speaker_id'] for r in manifest_rows})
    weights=draws(speakers,np.random.default_rng(SEED))
    model_metrics={}
    for model in model_ids:
        part=[r for r in rows if r['model']==model]
        def subset(conditions,label=None,attack=None):
            return [r for r in part if r['condition'] in conditions and (label is None or r['y']==label)
                    and (attack is None or r['attack']==attack)]
        condition_metrics={c:metrics(subset((c,))) for c in CONDITIONS}
        attacks={}
        for attack in ATTACKS[1:]:
            group=subset(CONDITIONS,1,attack)
            by_source=Counter(r['sample_id'] for r in group if r['condition']!='clean' and r['error'])
            if group:
                attacks[attack]=dict(conditions={c:metrics(subset((c,),1,attack)) for c in CONDITIONS},
                                     all_stress_wrong=sum(v==3 for v in by_source.values()))
        order={}
        for label in (0,1):
            first=sorted(subset(('A_then_B',),label),key=lambda r:r['sample_id'])
            second=sorted(subset(('B_then_A',),label),key=lambda r:r['sample_id'])
            order[str(label)]=dict(transition=transitions([r['error'] for r in first],[r['error'] for r in second]),
                score_delta=describe([a['p']-b['p'] for a,b in zip(first,second,strict=True)]))
        model_metrics[model]=dict(conditions=condition_metrics,attacks=attacks,order=order,
            macro_stress_auc=float(np.mean([condition_metrics[c]['auc'] for c in CONDITIONS[1:]])))
    paired={}
    for seed in sorted({m.split('_')[-1] for m in model_ids}):
        left,right=([r for r in rows if r['model']==f'{method}_{seed}'] for method in METHODS)
        specs=[(f'{stage}_{label}',conditions,label,None) for stage,conditions in
               [('clean',('clean',)),('stress',CONDITIONS[1:])] for label in (0,1)]
        specs.extend((attack,CONDITIONS[1:],1,attack) for attack in ATTACKS[1:])
        pairs={}
        for name,conditions,label,attack in specs:
            filtered=[[r for r in part if r['condition'] in conditions and r['y']==label
                       and (attack is None or r['attack']==attack)] for part in (left,right)]
            if filtered[0]:
                pairs[name]=cluster_delta(*filtered,speakers,weights)
        auc_draws=[np.mean([bootstrap_auc([r for r in part if r['condition']==c],speakers,weights)
                            for c in CONDITIONS[1:]],axis=0) for part in (left,right)]
        pairs['macro_stress_auc']=interval(auc_draws[1]-auc_draws[0])
        paired[seed]=pairs
    return dict(models=model_metrics,paired=paired,bootstrap=dict(seed=SEED,draws=10000,unit='speaker'))


def run(root, lock_path, approval_path=None):
    started=perf_counter()
    if approval_path is None:
        raise ValueError('Held-out evaluation is not authorized without separate scoring approval')
    lock=scoring_lock(lock_path, approval_path)
    prelock_path = root/PRELOCK
    if lock['prelock_sha256'] != file_hash(prelock_path):
        raise ValueError('Proposed lock does not match the immutable prelock')
    prelock = engine.read(prelock_path)
    if any(lock[k] != prelock[k] for k in ('seeds','conditions','ffmpeg_sha256','benchmark_sha256')):
        raise ValueError('Proposed lock changes the prelock evaluation design')
    runner = Path(__file__).relative_to(root).as_posix()
    if any(lock['files'].get(name) != digest for name,digest in prelock['files'].items() if name != runner):
        raise ValueError('Proposed lock changes a protected prelock input')
    verify_evaluation_lock(root,lock)
    # The caller must supply the separately approved lock, including this runner and protected identity manifests.
    required=[runner, AUDITOR, 'scripts/jaes_v2/final_score_diagnostic.py',
              *( (FOUNDATION/f'{role}_manifest.csv').as_posix() for role in ('train','development','calibration'))]
    if any(name not in lock['files'] for name in required):
        raise ValueError('Runner or overlap manifest absent from approved lock')
    lock_hash=object_hash(lock)
    selected=dict(lock['evaluation_manifest'],lock_sha256=lock_hash)
    out=root/OUT/f'heldout_{lock_hash[:16]}'
    require_resources(root)
    with resource_lock(root), ZipFile(root/ARCHIVE) as archive:
        out.mkdir(parents=True,exist_ok=True)
        result_path=out/'metrics.json'
        if result_path.exists():
            raise FileExistsError('Evaluation already complete')
        manifest_path=out/'manifest.json'
        if manifest_path.exists():
            if engine.read(manifest_path)!=selected:
                raise ValueError('Evaluation manifest changed')
        else:
            write_once(manifest_path,selected)
        tf=engine.phase10_tensorflow()
        functions={}
        for seed in lock['seeds']:
            runs,*_=models_and_data(root,seed)
            for method,(path,_,_,result) in runs.items():
                model=tf.keras.models.load_model(path/'selected.keras',compile=False)
                if prior.variable_hash(model.weights)!=result['selected_weights']:
                    raise ValueError('Held-out model weights mismatch')
                functions[f'{method}_{seed}']=prior.segmentwise_inference(tf,model)
        pre=Preprocessor('P0')
        ffmpeg=shutil.which('ffmpeg')
        shards=out/'scores'
        rows=[]
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            tempfile.tempdir=temporary
            try:
                for record in selected['rows']:
                    require_resources(root)
                    sample=record['sample_id']
                    shard=shards/(stable_key(0,sample)+'.json')
                    if shard.exists():
                        saved=engine.read(shard)
                        values=saved['rows']
                    else:
                        payload=archive.read(record['member'])
                        if hashlib.sha256(payload).hexdigest()!=record['sha256']:
                            raise ValueError('Evaluation audio checksum changed')
                        values=[]
                        for condition,waveform in waveform_bank(payload,sample,ffmpeg).items():
                            tensor=pre.extract(waveform.samples)[None]
                            for model,forward in functions.items():
                                score=float(forward(tensor).numpy().reshape(-1)[0])
                                values.append(dict(model=model,condition=condition,sample_id=sample,
                                                   raw_score=score,predicted_label=int(score>=.5)))
                    commit_source_scores(shards,lock_hash,sample,values,functions,CONDITIONS)
                    rows.extend(dict(r,**{k:record[k] for k in ('speaker_id','attack')},y=record['label'],
                                     p=r['raw_score'],error=int(r['predicted_label']!=record['label'])) for r in values)
            finally:
                tempfile.tempdir=None
        if len(rows)!=len(selected['rows'])*len(functions)*len(CONDITIONS):
            raise ValueError('Held-out output incomplete')
        write_once(result_path,dict(lock_sha256=lock_hash,manifest_sha256=file_hash(manifest_path),
            status='COMPLETE',predictions=len(rows),subset=lock['subset'],
            elapsed_seconds=perf_counter()-started,**summarize(rows,functions,selected['rows'])))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--proposed-lock',type=Path,required=True)
    parser.add_argument('--scoring-approval',type=Path,required=True)
    args=parser.parse_args()
    run(Path.cwd(),args.proposed_lock,args.scoring_approval.resolve())
