"""Sealed score-only Phase6 statistics, fairness audit and standalone scientific figures."""
from collections import defaultdict,Counter
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.lava.jaes_v2.evidence import read_csv,write_csv,write_json,file_hash
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.analysis import parent_bootstrap
from src.lava.jaes_v2.phase6_data import BASE,DEV_CONDITIONS
from scripts.jaes_v2.phase5_report import summarize_scores


def report(root,output):
    output.mkdir(parents=True,exist_ok=False); summaries={}; metric_rows=[]; metadata={}; input_hashes={}; raw={}
    protocol=json.loads((root/BASE/'protocol_v1.json').read_text())
    for run in sorted((root/BASE/'training').glob('p6_*')):
        m=json.loads((run/'run_metadata.json').read_text())
        if m['status']!='PILOT_EVALUATED': continue
        verify_seal(run,m)
        if m['optimizer_steps']!=protocol['optimizer_steps']: raise ValueError('Unmatched actual optimizer budget')
        rows=read_csv(run/'validation_scores.csv'); summaries[run.name]=summarize_scores(rows); metadata[run.name]=m; raw[run.name]=rows
        metric_rows += [dict(run=run.name,**r) for r in summaries[run.name]['metrics']]
        input_hashes[str(run.relative_to(root))]=file_hash(run/'run_metadata.json')
    if len(summaries)!=9: raise ValueError('Nine registered baselines must complete before the main report')
    a,b=protocol['primary_comparison']; control=metadata[b]; consistency=metadata[a]
    if control['initial_model_weights_sha256']!=consistency['initial_model_weights_sha256']: raise ValueError('Initial weights differ')
    if file_hash(root/BASE/'training'/a/'training_views.csv')!=file_hash(root/BASE/'training'/b/'training_views.csv'): raise ValueError('Paired training exposure differs')
    if any(control[k]!=consistency[k] for k in ('optimizer_steps','supervised_views','source_examples','training_head_forwards','backward_passes')): raise ValueError('Unequal exposure')
    cells={name:{(r['sample_id'],r['condition_id']):r for r in raw[name]} for name in (a,b)}; clusters=defaultdict(list)
    for key,r in cells[a].items():
        if key[1] in ('noise5','A_then_B','B_then_A') and r['label']=='1':
            c=cells[b][key]; clusters[r['speaker_id']].append(int(r['predicted_label']=='0')-int(c['predicted_label']=='0'))
    contrast=parent_bootstrap(clusters,replicates=2000,seed=5806)
    clean={name:next(r for r in summaries[name]['metrics'] if r['condition']=='clean') for name in (a,b)}
    margin=protocol['clean_risk_margin']; passes=all(clean[a][k]<=clean[b][k]+margin for k in ('false_alarm_rate','miss_rate'))
    primary=dict(comparison=[a,b],equal_view_fairness='PASS',contrast=contrast,clean_guardrail_passes=passes,
        clean_far_change=clean[a]['false_alarm_rate']-clean[b]['false_alarm_rate'],clean_miss_change=clean[a]['miss_rate']-clean[b]['miss_rate'],
        screening_improvement=bool(contrast['estimate']<0 and passes),scope='Single-seed exploratory screen; no significance or confirmed novelty')
    # Retention proxy is observed without labels; labels only assess its post-hoc diagnostic association.
    cache=root/BASE/'feature_cache_v2'; view_rows=read_csv(cache/'view_manifest.csv')
    with np.load(cache/'embeddings.npz',allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
    index={(r['policy'],r['sample_id'],r['condition']):r for r in view_rows if r['split']=='development'}
    proxies=[]
    for policy in ('P0','P1'):
        run=f'p6_{policy.lower()}_b0_v1'; lookup={(r['sample_id'],r['condition_id']):r for r in raw[run]}
        for key,r in index.items():
            if key[0]!=policy or key[2]=='clean': continue
            c=index[(policy,key[1],'clean')]; x=arrays[r['array_key']][int(r['array_index'])].reshape(-1); y=arrays[c['array_key']][int(c['array_index'])].reshape(-1)
            denom=float(np.linalg.norm(x)*np.linalg.norm(y)); similarity=float(np.clip(np.dot(x,y)/denom,-1,1)) if denom else 0.
            score=lookup[(key[1],key[2])]
            proxies.append(dict(policy=policy,sample_id=key[1],condition=key[2],speaker_id=score['speaker_id'],label=int(score['label']),
                retention_proxy=(1+similarity)/2,error=int(score['label']!=score['predicted_label'])))
    proxy_summary=[]
    for policy in ('P0','P1'):
        for label in (0,1):
            sub=[r for r in proxies if r['policy']==policy and r['label']==label]; x=[r['retention_proxy'] for r in sub]; y=[r['error'] for r in sub]
            rho=float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None
            proxy_summary.append(dict(policy=policy,label=label,n=len(sub),spearman_rho=rho,interpretation='Post-hoc association; repeated views, no p-value or forensic-retention validation'))
    write_csv(output/'metrics.csv',metric_rows); write_csv(output/'retention_proxy.csv',proxies)
    write_json(output/'summary.json',dict(status='MODEL_FREE_ANALYZED',results=summaries,primary=primary,retention_proxy=proxy_summary,input_hashes=input_hashes,
        seed_count=1,confirmation_locked=False,calibration_fitted=False))
    fig,axes=plt.subplots(2,2,figsize=(13,9)); names=list(summaries); colors={name:f'C{i%10}' for i,name in enumerate(names)}
    for name in names:
        results=summaries[name]['metrics']; clean=next(r for r in results if r['condition']=='clean'); stress=[r for r in results if r['condition'] in ('noise5','A_then_B','B_then_A')]
        axes[0,0].scatter(clean['false_alarm_rate'],np.mean([r['miss_rate'] for r in stress]),label=name,color=colors[name])
        rows=[next(r for r in results if r['condition']==c) for c in DEV_CONDITIONS]
        axes[0,1].plot(range(len(rows)),[r['miss_rate'] for r in rows],'.-',color=colors[name],label=name)
        h=read_csv(root/BASE/'training'/name/'training_history.csv'); axes[1,0].plot([r['epoch'] for r in h],[float(r['development_clean_bce']) for r in h],color=colors[name])
        axes[1,1].bar(name,metadata[name]['elapsed_seconds'],color=colors[name])
    axes[0,0].set(xlabel='Clean REAL false alarm',ylabel='Mean noise/order FAKE miss',title='Exploratory clean/stress trade-off'); axes[0,0].legend(fontsize=6)
    axes[0,1].set_xticks(range(len(DEV_CONDITIONS)),DEV_CONDITIONS,rotation=45,ha='right'); axes[0,1].set_title('Classwise FAKE miss by condition')
    axes[1,0].set(xlabel='Epoch',ylabel='Clean development BCE',title='Actual learning curves'); axes[1,1].set_ylabel('Training + selection/evaluation seconds'); axes[1,1].tick_params(axis='x',rotation=75,labelsize=7)
    fig.tight_layout(); fig.savefig(output/'baseline_classwise_training.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in (b,a):
        rows=[next(r for r in summaries[name]['metrics'] if r['condition']==c) for c in DEV_CONDITIONS]
        axes[0].plot(range(len(rows)),[r['false_alarm_rate'] for r in rows],'.-',label=name)
        axes[1].plot(range(len(rows)),[r['miss_rate'] for r in rows],'.-',label=name)
    for ax in axes: ax.set_xticks(range(len(DEV_CONDITIONS)),DEV_CONDITIONS,rotation=45,ha='right'); ax.legend(fontsize=7)
    axes[0].set_title('Equal-view consistency: REAL false alarm'); axes[1].set_title('Equal-view consistency: FAKE miss'); fig.tight_layout(); fig.savefig(output/'equal_view_ablation.png',dpi=180); plt.close(fig)
    selection=read_csv(root/BASE/'data_selection_v1/manifest.csv'); original=read_csv(root/'outputs/jaes_v2/c2_feature_cache_v2/manifest.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,data,title in zip(axes,(original,selection),('Phase5 development support','Phase6 balanced development support')):
        data=[r for r in data if r['split']=='development']; speakers=sorted({r['speaker_id'] for r in data}); counts=Counter((r['speaker_id'],r['label']) for r in data); positions=np.arange(len(speakers))
        ax.bar(positions-.18,[counts[(s,'0')] for s in speakers],.36,label='REAL'); ax.bar(positions+.18,[counts[(s,'1')] for s in speakers],.36,label='FAKE')
        ax.set_xticks(positions,speakers,rotation=90); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(output/'speaker_coverage.png',dpi=180); plt.close(fig)
    write_json(output/'report_metadata.json',dict(input_hashes=input_hashes,output_hashes={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},neural_inference=False))
    return primary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    print(json.dumps(report(Path(__file__).resolve().parents[2],args.output),indent=2))
