"""Model-free sealed Phase 5 aggregation and scientific figures (no predictor calls)."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.lava.jaes_v2.evidence import read_csv,write_csv,write_json,file_hash
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.public_analysis import metrics
from src.lava.jaes_v2.analysis import parent_bootstrap


def summarize_scores(rows):
    conditions=sorted({r['condition_id'] for r in rows}); clean={r['sample_id']:r for r in rows if r['condition_id']=='clean'}
    if len(rows)!=len(clean)*len(conditions) or len({(r['sample_id'],r['condition_id']) for r in rows})!=len(rows):
        raise ValueError('Incomplete/duplicate C2 paired result')
    tables=[]; paired=[]; groups=[]
    for condition in conditions:
        selected=[r for r in rows if r['condition_id']==condition]
        tables.append({'condition':condition,**metrics(selected)})
        clusters=defaultdict(list); shifts=[]; flips=0; correct_to_error=0; error_to_correct=0
        for r in selected:
            c=clean[r['sample_id']]
            if any(r[k]!=c[k] for k in ('label','speaker_id','attack_id','threshold')): raise ValueError('Paired cohort mismatch')
            e=int(r['predicted_label']!=r['label']); ce=int(c['predicted_label']!=c['label'])
            clusters[r['speaker_id']].append(e-ce); shifts.append(float(r['raw_score'])-float(c['raw_score']))
            flips+=r['predicted_label']!=c['predicted_label']; correct_to_error+=int(e and not ce); error_to_correct+=int(ce and not e)
        paired.append({'condition':condition,'decision_flips':flips,'clean_correct_to_error':correct_to_error,'clean_error_to_correct':error_to_correct,
                       'mean_paired_score_shift':float(np.mean(shifts)),'mean_abs_paired_score_shift':float(np.mean(np.abs(shifts))),
                       'speaker_uniform_error_change':parent_bootstrap(clusters,replicates=2000,seed=5804)})
        for field in ('speaker_id','attack_id'):
            for value in sorted({r[field] for r in selected}):
                members=[r for r in selected if r[field]==value]
                groups.append({'condition':condition,'group_field':field,'group':value,**metrics(members)})
    worst=[]
    for condition in conditions:
        eligible=[r for r in groups if r['condition']==condition and r['group_field']=='speaker_id' and r['n']>=4]
        worst.append({'condition':condition,'minimum_group_n':4,'eligible_groups':len(eligible),
                      'worst_speaker_error':max((r['error_rate'] for r in eligible),default=None)})
    return {'metrics':tables,'paired':paired,'groups':groups,'worst_groups':worst,
            'uncertainty_scope':'Descriptive 16-official-speaker bootstrap; equal-speaker estimand; no multi-seed uncertainty or population independence proof'}


def report(root: Path,output: Path):
    output.mkdir(parents=True,exist_ok=False); inputs={}; results={}
    pp=root/'outputs/jaes_v2/phase5_preprocessing_study_v1'
    verify_seal(pp,json.loads((pp/'run_metadata.json').read_text()))
    differences=read_csv(pp/'paired_differences.csv'); pp_metrics=read_csv(pp/'metrics.csv'); summaries=[]
    for model in sorted({r['model_id'] for r in differences}):
        for policy in ('P0','P1','P2'):
            subset=[r for r in differences if r['model_id']==model and r['policy']==policy]
            summaries.append(dict(model=model,policy=policy,n=len(subset),flips=sum(int(r['decision_flip']) for r in subset),
                max_feature_difference=max(float(r['feature_max_difference']) for r in subset),
                mean_embedding_l2=float(np.mean([float(r['embedding_l2_difference']) for r in subset])),
                mean_abs_score_shift=float(np.mean([abs(float(r['raw_score_difference'])) for r in subset]))))
    write_csv(output/'preprocessing_pairs_summary.csv',summaries)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for model in sorted({r['model'] for r in summaries}):
        sub=[r for r in summaries if r['model']==model]
        for ax,key in zip(axes,('max_feature_difference','mean_abs_score_shift','flips')):
            ax.plot([r['policy'] for r in sub],[r[key] for r in sub],'o-',label=model); ax.set_title(key.replace('_',' ')); ax.set_xlabel('Policy, native vs half gain')
    axes[0].legend(fontsize=7); fig.suptitle('32 exploratory MLAAD records; frozen legacy models; altered-input diagnostic')
    fig.tight_layout(); fig.savefig(output/'gain_and_decision_sensitivity.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,2.8)); ax.axis('off')
    labels=['Waveform\nx → a x','STFT / Mel\nP → a² P','P0: max(P, ε)\nP1: max(P / max P, ρ)','Relative dB\n6 images','Frozen ImageNet\n+ trained LSTM']
    for i,label in enumerate(labels):
        ax.text(.02+i*.205,.5,label,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round',facecolor='#dceaf7'),fontsize=9)
        if i<4: ax.annotate('',xy=(.02+(i+1)*.205-.085,.5),xytext=(.02+i*.205+.085,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.set_title('P1 cancels nonzero global scale mathematically; discrimination must be tested')
    fig.tight_layout(); fig.savefig(output/'preprocessing_mechanism.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    runs=sorted((root/'outputs/jaes_v2/c2').glob('*_pilot_v2')) if (root/'outputs/jaes_v2/c2').exists() else []
    scored={}
    for run in runs:
        meta=json.loads((run/'run_metadata.json').read_text())
        if meta['status']!='PILOT_EVALUATED': continue
        verify_seal(run,meta); scored[run.name]=read_csv(run/'validation_scores.csv'); results[run.name]=summarize_scores(scored[run.name])
        inputs[str(run.relative_to(root))]=file_hash(run/'run_metadata.json')
    if results:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for name in results:
            history=read_csv(root/'outputs/jaes_v2/c2'/name/'training_history.csv')
            axes[0].plot([int(r['epoch']) for r in history],[float(r['train_loss']) for r in history],label=name)
            axes[1].plot([int(r['epoch']) for r in history],[float(r['development_clean_bce']) for r in history],label=name)
        axes[0].set_title('Training loss'); axes[1].set_title('Development clean BCE'); axes[1].legend(fontsize=6)
        for ax in axes: ax.set_xlabel('Epoch')
        fig.tight_layout(); fig.savefig(output/'learning_curves.png',dpi=170); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        combined=[]
        for color_index,(name,result) in enumerate(results.items()):
            color=f'C{color_index}'
            clean=next(r for r in result['metrics'] if r['condition']=='clean'); stress=[r for r in result['metrics'] if r['condition']!='clean']
            axes[0].scatter(clean['error_rate'],np.mean([r['error_rate'] for r in stress]),label=name,color=color)
            for condition in ('clean','noise5'):
                r=next(r for r in result['metrics'] if r['condition']==condition)
                axes[1].scatter(r['false_alarm_rate'],r['miss_rate'],marker='o' if condition=='clean' else 'x',label=name+' '+condition,color=color)
            for r in result['metrics']: combined.append({'run':name,**r})
            axes[2].plot([r['condition'] for r in result['metrics']],[r['error_rate'] for r in result['metrics']],'.-',label=name,color=color)
        axes[0].set(xlabel='Clean error',ylabel='Mean five-stress error',title='Clean / robustness trade-off')
        axes[1].set(xlabel='REAL false alarm',ylabel='FAKE miss',title='Clean circles / noise5 crosses')
        axes[2].set_title('Executed 2×2 baseline ablation'); axes[2].tick_params(axis='x',rotation=55); axes[2].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(output/'robustness_classwise_ablation.png',dpi=170); plt.close(fig)
        write_csv(output/'pilot_metrics.csv',combined)
    factorial=[]
    expected=[f'c2_{p}_{b}_pilot_v2' for p,b in (('p0','erm'),('p1','erm'),('p0','aug'),('p1','aug'))]
    if all(name in scored for name in expected):
        cells=[{(r['sample_id'],r['condition_id']):r for r in scored[name]} for name in expected]
        if any(set(c)!=set(cells[0]) for c in cells): raise ValueError('Unmatched factorial cohort')
        for condition in sorted({k[1] for k in cells[0]}):
            cluster=defaultdict(list); utterance=[]
            for key in sorted(k for k in cells[0] if k[1]==condition):
                rows=[c[key] for c in cells]
                if any(r['label']!=rows[0]['label'] or r['speaker_id']!=rows[0]['speaker_id'] for r in rows): raise ValueError('Factorial identity changed')
                errors=[int(r['predicted_label']!=r['label']) for r in rows]
                contrast=(errors[3]-errors[2])-(errors[1]-errors[0])
                cluster[rows[0]['speaker_id']].append(contrast); utterance.append(contrast)
            factorial.append({'condition':condition,'contrast':'(P1_aug-P0_aug)-(P1_erm-P0_erm) error interaction',
                'recording_mean':float(np.mean(utterance)),'speaker_uniform':parent_bootstrap(cluster,replicates=2000,seed=5804),
                'status':'DESCRIPTIVE_SINGLE_SEED_DEVELOPMENT; no multiplicity-adjusted discovery'})
    cost=root/'outputs/jaes_v2/audit/phase5_efficiency.json'
    if cost.exists() and results:
        measured=json.loads(cost.read_text()); fig,ax=plt.subplots(figsize=(7,4))
        for r in measured['results']:
            if r['run'] not in results: continue
            clean=next(m for m in results[r['run']]['metrics'] if m['condition']=='clean')
            stress=np.mean([m['error_rate'] for m in results[r['run']]['metrics'] if m['condition']!='clean'])
            ax.scatter(1000*r['median_seconds']['end_to_end_seconds'],stress,label=r['run'])
        ax.set(xlabel='Measured warm end-to-end CPU milliseconds',ylabel='Mean five-stress development error',title='Tiny pilot cost/risk comparison; same architecture, timing noise remains')
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output/'efficiency_risk.png',dpi=170); plt.close(fig)
        inputs[str(cost.relative_to(root))]=file_hash(cost)
    inputs[str(pp.relative_to(root))]=file_hash(pp/'run_metadata.json')
    write_json(output/'summary.json',{'status':'MODEL_FREE_REAGGREGATED','preprocessing':summaries,'pilots':results,'factorial_interactions':factorial,'input_hashes':inputs,
        'limitations':['All results exploratory development','One seed; tiny frozen-backbone pilot','No independently confirmed novelty','Matched source views are not independent recordings']})
    seal={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'report_metadata.json',{'output_hashes':seal,'input_hashes':inputs,'neural_framework_imported':False})
    return {'pilots':len(results),'output':str(output)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); print(report(Path(__file__).resolve().parents[2],args.output))
