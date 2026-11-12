"""Verify sealed Phase 10 artifacts and render three-seed development evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

from analyze_locked_heldout import ROOT, OUT, file_hash, object_hash, read, scoring_lock, verified_scores
from scripts.jaes_v2.empirical_readiness import models_and_data
from scripts.jaes_v2.final_score_diagnostic import align, expected_rows, transitions
from scripts.jaes_v2.phase10_medium import admission, comparison, registered
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.evidence import read_csv, write_json
from src.lava.jaes_v2.phase10_validation import STRESS, metrics

PAPER = Path(__file__).resolve().parent
OUTDIR = ROOT/'outputs/jaes_v2/three_seed_revision'
SEEDS = ('9107','9207','9307')
METHODS = ('erm','channel')
RATES = ('A_clean_FA','A_clean_miss','A_stress_FA','A_stress_miss','B_clean_FA','A05_stress_miss','A06_stress_miss')
LABELS = ('A clean REAL FA','A clean FAKE miss','A stress REAL FA','A stress FAKE miss','B clean REAL FA','A05 stress FAKE miss','A06 stress FAKE miss')


def verify():
    base=ROOT/OUT
    evidence=read(base/'paired_seed_evidence_9107_9207_9307.json')
    seeds=evidence['seeds']
    assert tuple(seeds)==SEEDS and not evidence['pending']
    assert not evidence['official_evaluation_used'] and not evidence['calibration_used']
    lock_path=base/'heldout_proposed_9107_9207_v1.json'
    lock=scoring_lock(lock_path,base/'heldout_scoring_author_approval.json')
    lock_hash=object_hash(lock)
    held=base/f'heldout_{lock_hash[:16]}'
    rows, models, _, shards=verified_scores(held,lock,lock_hash)
    assert shards==read(held/'analysis/verified_analysis.json')['integrity']['score_shards']
    assert len(rows)==17408 and len(models)==4 and tuple(lock['seeds'])==(9107,9207)
    reg,_=registered('p10_medium_erm_s9307_v2')
    admission(reg)
    registry_sha=file_hash(ROOT/'configs/jaes_v2/phase10/prospective_medium_v1.json')
    pair=read(base.parent/'medium_pair_s9307_v2.json')
    assert pair==comparison('p10_medium_erm_s9307_v2','p10_medium_channel_s9307_v2')
    audit={}
    ref=None
    for seed in SEEDS:
        runs, dev, b_ids, config, _=models_and_data(ROOT,int(seed))
        ids=expected_rows(dev,b_ids)
        assert len(ids)==368 and len(b_ids)==144
        assert ref is None or ref==ids
        ref=ids
        audit[seed]={}
        score_grids={}
        for method,(path,c,state,result) in runs.items():
            expected=f'p10_medium_{method}_s{seed}_v2'
            assert c['experiment']==expected and c['registry_sha256']==registry_sha
            assert c['config']['manifest_sha256']==reg['medium_manifest_sha256']
            assert c['config']['architecture']=='MobileNetV3Small-LSTM128' and c['backend']=='FULL_SHAPE_RECOMPUTE'
            assert c['config']['training_seed']==int(seed) and c['policy']=='CLEAN_BCE' and c['threshold']==.5
            assert c['config']['augmentation_policy']==('clean' if method=='erm' else 'channel')
            assert state['contract']==c
            journal=state['journal']
            order=[[s for s,_ in j['sources']] for j in journal]
            assert [j['step'] for j in journal]==list(range(1,841)) and all(len(o)==16 for o in order)
            assert [(h['step'],h['epoch'],h['tier']) for h in state['history']]==[
                (40*e,e,'B' if e%3==0 or e==21 else 'A') for e in range(1,22)]
            best=min(state['history'],key=lambda h:(h['clean']['nll'],h['epoch']))
            tag=f"u{best['step']:06d}_e{best['epoch']:03d}_c000"
            assert tag==state['best']['tag']==seeds[seed]['endpoints'][method]['selected_checkpoint']
            assert result['final_weights']==state['model_hash']
            archived=read_csv(path/'scores.csv')
            grid=align(archived,ids,prior.DEV)
            assert len(grid)==2576 and len(archived)==2576
            typed=[dict(r,label=int(r['label']),threshold=float(r['threshold']),
                        raw_score=float(r['raw_score']),predicted_label=int(r['predicted_label'])) for r in archived]
            assert object_hash(typed)==result['scores_hash']
            assert metrics(typed)==result['metrics']
            assert all(file_hash(path/f)==digest for f,digest in seeds[seed]['endpoints'][method]['input_sha256'].items())
            score_grids[method]=grid
            audit[seed][method]=dict(experiment=expected,architecture=c['config']['architecture'],config=c['config'],
                steps=840,epochs=21,validation_epochs=len(state['history']),selected_checkpoint=tag,
                selected_hash=result['selected_weights'],final_hash=result['final_weights'],
                initial_hash=c['initial_weights'],source_order_sha256=object_hash(order),
                source_sha256=c['actual_sources'],score_rows=len(grid),score_hash=result['scores_hash'],
                view_counts=Counter(v for j in journal for _,v in j['sources']),
                final_checkpoint=read(path/'latest.json')['checkpoint'],
                input_sha256={f:file_hash(path/f) for f in ('contract.json','latest.json','result.json','scores.csv','seal.json','selected.keras','status.json','data_audit.json')})
        a,b=[audit[seed][m] for m in METHODS]
        assert all(a[k]==b[k] for k in ('initial_hash','source_order_sha256','source_sha256'))
        attack={}
        for code in ('A05','A06'):
            changes=defaultdict(int)
            conditions={}
            for c in STRESS:
                ids_attack=sorted(s for s,r in ids.items() if r['attack_id']==code)
                aa,bb=[[score_grids[m][s,c,'A_PAIRED'] for s in ids_attack] for m in METHODS]
                assert len(aa)==18
                errors=transitions([r['error'] for r in aa],[r['error'] for r in bb])
                conditions[c]=dict(transitions=errors,
                    mean_fake_score_delta=mean(y['p']-x['p'] for x,y in zip(aa,bb,strict=True)))
                for x,y in zip(aa,bb,strict=True):
                    changes[x['speaker_id']]+=y['error']-x['error']
            attack[code]=dict(conditions=conditions,speaker_net_misses=dict(sorted(changes.items())))
        audit[seed]['attack_diagnostics']=attack
    assert len({audit[s]['erm']['initial_hash'] for s in SEEDS})==3
    for method,expected in [('erm',(.9822225765,.5497714711,266,44,21,49,45)),('channel',(.9855707908,.7933939201,119,61,30,43,28))]:
        row=seeds['9307']['endpoints'][method]
        assert abs(row['clean']['auc']-expected[0])<1e-9 and abs(row['macro_stress_auc']-expected[1])<1e-9
        keys=('A_stress_miss','A_stress_FA','B_clean_FA','A05_stress_miss','A06_stress_miss')
        assert tuple(error_count(seeds['9307'],k,method)[0] for k in keys)==expected[2:]
    assert [tuple(error_count(seeds[s],f'{a}_stress_miss',m)[0] for m in METHODS)
            for s in SEEDS for a in ('A05','A06')]==[(48,50),(41,45),(38,47),(16,31),(49,43),(45,28)]
    return evidence,audit,dict(models=models,predictions=len(rows),shards=len(shards),lock_hash=lock_hash,
        lock_sha256=file_hash(lock_path),manifest_sha256=file_hash(held/'manifest.json'),metrics_sha256=file_hash(held/'metrics.json')),pair


def error_count(block,key,method):
    field='miss' if 'miss' in key else 'false_alarm'
    arm=block['paired'][key][method][field]
    return arm['errors'],arm['n']


def values(block,method):
    ep=block['endpoints'][method]
    return {'clean_auc':ep['clean']['auc'],'macro_stress_auc':ep['macro_stress_auc'],
            **{key: n/d for key in RATES for n,d in [error_count(block,key,method)]},
            **{f'order_median_{k}':v['scores']['quantiles'][3] for k,v in ep['order'].items()}}


def stats(series):
    return dict(mean=mean(series),sample_sd=stdev(series),min=min(series),max=max(series),per_seed=dict(zip(SEEDS,series,strict=True)))


def table(name, label, caption, headers, data):
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',
           f'\\caption{{{caption}}}\\label{{tab:{label}}}',
           r'\begin{tabular}{@{}l'+'r'*(len(headers)-1)+r'@{}}',r'\toprule',
           ' & '.join(headers)+r' \\',r'\colrule']
    lines.extend(' & '.join(map(str,row))+r' \\' for row in data)
    lines.extend([r'\botrule',r'\end{tabular}',r'\end{table*}'])
    if name == 'uncertainty':
        lines.insert(3, r'\setlength{\tabcolsep}{3pt}')
    (PAPER/'tables'/f'{name}_v5.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')


def render(seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#0072B2','#009E73','#D55E00')
    dev,attacks=[],[]
    for s,b in seeds.items():
        for m in METHODS:
            e=b['endpoints'][m]
            cells=[f'{n}/{d}' for key in ('A_stress_FA','A_stress_miss','B_clean_FA') for n,d in [error_count(b,key,m)]]
            dev.append([f'{s} {m.upper()}',e['selected_checkpoint'][1:7].lstrip('0'),f"{e['clean']['auc']:.4f}",f"{e['macro_stress_auc']:.4f}",*cells])
        for attack in ('A05','A06'):
            a,b2=[error_count(b,f'{attack}_stress_miss',m) for m in METHODS]
            attacks.append([f'{s} {attack}',f'{a[0]}/{a[1]}',f'{b2[0]}/{b2[1]}',f'{b2[0]-a[0]:+d}'])
    table('development','development','Development results for all fitted pairs. The selected update identifies the clean-BCE checkpoint. Stressed counts use repeated views.',
          ['Seed and arm','Update','Clean AUC','Stress AUC','Stress FA','Stress miss','B clean FA'],dev)
    table('attack','attack','Development A05/A06 stress FAKE misses over the three stress views. Each row represents 18 distinct sources and 54 transformed views.',
          ['Seed and attack','ERM','Channel',r'$\Delta$ misses'],attacks)
    rows=[]
    for key,label in zip(('macro_stress_auc',*RATES),('A macro stress AUC',*LABELS),strict=True):
        cells=[]
        factor=1 if key=='macro_stress_auc' else 100
        precision=4 if factor==1 else 2
        ds=[]
        for s,b in seeds.items():
            item=b['macro_stress_auc_interval'] if factor==1 else b['paired'][key]['bootstrap']
            x=b['deltas'][key]*factor
            ds.append(x)
            lo,hi=(v*factor for v in item['percentile95'])
            cells.append(r'\shortstack[r]{'+f'{x:+.{precision}f}'+r'\\'+f'{{[{lo:+.{precision}f}, {hi:+.{precision}f}]}}'+'}')
        rows.append([label,*cells,f'{mean(ds):+.{precision}f} '+r'$\pm$ '+f'{stdev(ds):.{precision}f}'])
    table('uncertainty','uncertainty','Channel minus ERM. Brackets give within-seed 95 percent speaker-cluster intervals. Mean and sample SD describe the three fitted contrasts. Error differences are percentage points.',
          ['Endpoint',*SEEDS,'Mean and SD'],rows)
    fig,ax=plt.subplots(figsize=(5.6,2.65),layout='constrained')
    for (s,b),color in zip(seeds.items(),colors,strict=True):
        ax.plot((0,1),[b['endpoints'][m]['macro_stress_auc'] for m in METHODS],'-o',label=s,color=color)
    ax.set(xticks=(0,1),xticklabels=('ERM','Channel'),ylabel='Development A macro stress ROC-AUC',xlim=(-.2,1.2))
    ax.legend(title='Seed')
    fig.savefig(PAPER/'figures/aggregate_v5.pdf',metadata={'CreationDate':None})
    plt.close(fig)
    for keys,labels,name in [(RATES[:5],LABELS[:5],'classwise_errors'),(RATES[-2:],('A05','A06'),'attack_changes')]:
        fig,ax=plt.subplots(figsize=(6.3,2.75 if len(keys)>2 else 2.0),layout='constrained')
        for j,((s,b),color) in enumerate(zip(seeds.items(),colors,strict=True)):
            for i,key in enumerate(keys):
                p=b['paired'][key]['bootstrap']
                y=i+.17*(j-1)
                ax.hlines(y,*(100*x for x in p['percentile95']),color=color)
                ax.scatter(100*p['delta'],y,color=color,label=s if i==0 else None,s=22)
        ax.set(yticks=range(len(keys)),yticklabels=labels,xlabel='Channel minus ERM (percentage points)')
        ax.axvline(0,color='.5',linewidth=.8)
        ax.invert_yaxis()
        ax.legend(title='Seed',ncol=3,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01))
        fig.savefig(PAPER/'figures'/f'{name}_v5.pdf',metadata={'CreationDate':None})
        plt.close(fig)


def main():
    evidence,audit,held,pair=verify()
    seeds=evidence['seeds']
    rows={s:{m:values(b,m) for m in METHODS} for s,b in seeds.items()}
    summary={}
    for key in rows[SEEDS[0]]['erm']:
        series={m:[rows[s][m][key] for s in SEEDS] for m in METHODS}
        summary[key]={**{m:stats(v) for m,v in series.items()},
            'channel_minus_erm':stats([b-a for a,b in zip(series['erm'],series['channel'],strict=True)])}
    OUTDIR.mkdir(parents=True,exist_ok=True)
    write_json(OUTDIR/'verification_and_statistics.json',dict(status='SIX_SEALED_RUNS_VERIFIED',
        v4_pdf_sha256=file_hash(PAPER/'build/LAVA_JAES_Review_Draft_v4.pdf'),
        frozen_fingerprints=engine.fingerprints(ROOT),audit=audit,heldout=held,
        three_seed_summary=summary,guard_9307=pair['channel_guard_against_selected_erm'],
        aggregate_sha256=file_hash(ROOT/OUT/'paired_seed_evidence_9107_9207_9307.json'),
        historical_two_seed_sha256=file_hash(ROOT/OUT/'paired_seed_evidence_9107_9207.json'),
        training_executed=False,new_scoring=False))
    render(seeds)
    for key in ('clean_auc','macro_stress_auc',*RATES):
        print(key,summary[key]['channel_minus_erm'])
    for s,b in seeds.items():
        for m,e in b['endpoints'].items():
            print(s,m,e['selected_checkpoint'],e['train_seconds'],e['validation_seconds'],e['peak_rss'],
                {k:(round(v['scores']['quantiles'][3],4),v['errors']) for k,v in e['order'].items()})
    for k in (*RATES,'macro_stress_auc'):
        print('CI',k,seeds['9307']['macro_stress_auc_interval'] if k=='macro_stress_auc' else seeds['9307']['paired'][k]['bootstrap'])


if __name__=='__main__':
    main()
