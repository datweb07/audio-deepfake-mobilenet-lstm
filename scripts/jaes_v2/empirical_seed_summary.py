"""Aggregate complete prospective seed pairs without pooling training seeds."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
from pathlib import Path
from statistics import mean
import numpy as np

from scripts.jaes_v2.empirical_readiness import BASE, OUT, METHODS, models_and_data, write_once
from scripts.jaes_v2.final_score_diagnostic import (
    A, B, SEED, align, expected_rows, metrics, transitions, draws, cluster_delta,
    bootstrap_auc, interval, describe,
)
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.evidence import file_hash, read_csv
from src.lava.jaes_v2.phase10_validation import STRESS


def analyze_seed(root, seed):
    runs,dev,b_ids,config,_=models_and_data(root,seed)
    identities=expected_rows(dev,b_ids)
    tables={method:align(read_csv(runs[method][0]/'scores.csv'),identities,prior.DEV) for method in METHODS}
    speakers={s:sorted({r['speaker_id'] for r in identities.values() if r['stratum']==s}) for s in (A,B)}
    random=np.random.default_rng(SEED)
    weights={s:draws(ids,random) for s,ids in speakers.items()}
    specs=[('A_clean_FA',A,('clean',),0,None),('A_clean_miss',A,('clean',),1,None),
           ('B_clean_FA',B,('clean',),0,None),('A_stress_FA',A,STRESS,0,None),
           ('A_stress_miss',A,STRESS,1,None)]
    specs.extend((f'A{i:02}_stress_miss',A,STRESS,1,f'A{i:02}') for i in range(1,7))
    paired={}
    for key,stratum,conditions,label,attack in specs:
        rows=[[r for _,r in sorted(tables[m].items()) if r['stratum']==stratum and r['y']==label
               and r['condition_id'] in conditions and (attack is None or r['attack_id']==attack)] for m in METHODS]
        paired[key]=dict(erm=metrics(rows[0]),channel=metrics(rows[1]),
            transitions=transitions([r['error'] for r in rows[0]],[r['error'] for r in rows[1]]),
            bootstrap=cluster_delta(*rows,speakers[stratum],weights[stratum]))
    endpoints={}
    auc_draws={}
    for method, table in tables.items():
        path,contract,state,result=runs[method]
        stress=[ [r for r in table.values() if r['stratum']==A and r['condition_id']==c] for c in STRESS ]
        auc_draws[method]=np.mean([bootstrap_auc(rows,speakers[A],weights[A]) for rows in stress],axis=0)
        clean=metrics([r for r in table.values() if r['stratum']==A and r['condition_id']=='clean'])
        timing_files=sorted(path.parent.glob('*_resources.json'))
        timings=[engine.read(f) for f in timing_files]
        order={}
        for stratum,label in ((A,'0'),(A,'1'),(B,'0')):
            ids=[sid for sid,r in identities.items() if r['stratum']==stratum and r['label']==label]
            ab=[table[sid,'A_then_B',stratum] for sid in ids]
            ba=[table[sid,'B_then_A',stratum] for sid in ids]
            order[f'{stratum}/{label}']=dict(scores=describe([a['p']-b['p'] for a,b in zip(ab,ba,strict=True)]),
                errors=transitions([r['error'] for r in ab],[r['error'] for r in ba]))
        endpoints[method]=dict(clean=clean,macro_stress_auc=mean(metrics(rows)['auc'] for rows in stress),
            selected_checkpoint=state['best']['tag'],order=order,experiment=contract['experiment'],
            input_sha256={f:file_hash(path/f) for f in ('scores.csv','result.json','seal.json','contract.json','selected.keras')},
            resource_logs={f.name:file_hash(f) for f in timing_files},
            train_seconds=sum(r['timing'].get('train_seconds',0) for r in timings) if timings else None,
            validation_seconds=sum(r['timing'].get('validation_seconds',0) for r in timings) if timings else None,
            peak_rss=max((r['final_resources']['process_peak_rss'] for r in timings),default=None),
            speaker_errors={f'{s}/{sp}/{c}':metrics([r for r in table.values() if r['stratum']==s and
                r['speaker_id']==sp and r['condition_id']==c]) for s in speakers for sp in speakers[s] for c in prior.DEV})
    deltas={name:row['bootstrap']['delta'] for name,row in paired.items()}
    deltas['macro_stress_auc']=endpoints['channel']['macro_stress_auc']-endpoints['erm']['macro_stress_auc']
    return dict(seed=seed,manifest_sha256=config.manifest_sha256,source_count=len(identities),
                endpoints=endpoints,paired=paired,deltas=deltas,
                macro_stress_auc_interval=interval(auc_draws['channel']-auc_draws['erm']))


def run(root):
    evidence={}
    pending={}
    for seed in (9107,9207,9307):
        existing={m:(root/BASE/f'p10_medium_{m}_s{seed}_v2/scientific/seal.json').is_file() for m in METHODS}
        if all(existing.values()):
            evidence[str(seed)]=analyze_seed(root,seed)
        else:
            pending[str(seed)]=existing
    if '9107' not in evidence:
        raise ValueError('Required seed 9107 incomplete')
    effects={}
    if len(evidence)>1:
        for key in evidence['9107']['deltas']:
            values=[row['deltas'][key] for row in evidence.values()]
            effects[key]=dict(mean=mean(values),range=[min(values),max(values)],
                signs=[int(np.sign(v)) for v in values])
    name='paired_seed_evidence_'+'_'.join(evidence)+'.json'
    report=dict(status='COMPLETE_PAIRED_SEEDS_ONLY',seeds=evidence,pending=pending,
        summary=effects,bootstrap=dict(seed=SEED,draws=10000,unit='speaker',strata_separate=True),
        official_evaluation_used=False,calibration_used=False)
    write_once(root/OUT/name,report)
    plot_effects(report,root/OUT/name.replace('.json','.png'))


def plot_effects(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if path.exists():
        raise FileExistsError(path)
    keys=('A_clean_FA','A_clean_miss','B_clean_FA','A_stress_FA','A_stress_miss')
    labels=('A clean REAL FA','A clean FAKE miss','B clean REAL FA','A stress REAL FA','A stress FAKE miss')
    figure,axis=plt.subplots(figsize=(8,4),layout='constrained')
    for index,(seed,result) in enumerate(report['seeds'].items()):
        rows=[result['paired'][key]['bootstrap'] for key in keys]
        values=np.array([row['delta'] for row in rows])*100
        bounds=np.array([row['percentile95'] for row in rows])*100
        y=np.arange(len(keys))+.14*(index-(len(report['seeds'])-1)/2)
        axis.hlines(y,bounds[:,0],bounds[:,1],linewidth=1.5)
        axis.scatter(values,y,label=f'Seed {seed}')
    axis.axvline(0,color='.5',linewidth=1)
    axis.set_yticks(np.arange(len(keys)),labels)
    axis.invert_yaxis()
    axis.set(xlabel='Channel − ERM (percentage points)\n'
             'A clean: 112 sources/class · A stress: 336 views/class · B clean: 144 REAL sources',
             title='Development error changes and paired speaker-bootstrap 95% intervals')
    axis.legend()
    figure.savefig(path,dpi=300)
    plt.close(figure)


if __name__=='__main__':
    run(Path.cwd())
