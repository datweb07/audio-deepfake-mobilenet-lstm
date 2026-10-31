"""Deterministic score-only statistics and publication figures, without TensorFlow."""
import argparse,json,os,sys
from pathlib import Path

def figures(root,out,summary):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.lava.jaes_v2.evidence import read_csv,write_json,file_hash
    from src.lava.jaes_v2.phase7_data import BASE,ATTACKS
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'lava-phase7','figure.dpi':150,'savefig.dpi':300})
    out.mkdir(exist_ok=True); saved=[]
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','svg'):
            path=out/(name+'.'+extension); fig.savefig(path,metadata={'Date':None} if extension=='svg' else None);saved.append(path)
        plt.close(fig)
    support=read_csv(root/BASE/'selections/medium_all_attack_v1/selected_support.csv')
    fig,ax=plt.subplots(figsize=(9,7)); matrix=np.array([[int(r[a]) for a in ATTACKS] for r in support]);image=ax.imshow(matrix,aspect='auto',vmin=0,vmax=3,cmap='Blues');ax.set_xticks(range(6),ATTACKS);ax.set_yticks(range(len(support)),[r['split'][:2]+' '+r['speaker_id'] for r in support],fontsize=7);fig.colorbar(image,ax=ax,label='Selected FAKE recordings');save(fig,'01_attack_support')
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(support));ax.bar(x,[int(r['REAL']) for r in support],label='REAL');ax.bar(x,[int(r['FAKE']) for r in support],bottom=[int(r['REAL']) for r in support],label='FAKE');ax.set_xticks(x,[r['speaker_id'] for r in support],rotation=90,fontsize=7);ax.set_ylabel('Unique recordings');ax.legend();save(fig,'02_speaker_class_support')
    fig,ax=plt.subplots(figsize=(11,3.5));ax.axis('off')
    texts=['Official train/dev\ncalibration/eval closed','Speaker + attack selection\n640 /224A /144B','Shared P0 view cache\nindependent seed blocks','Head-only H0–H4\nmatched H2/H3','Paired speaker + seed\nclasswise analysis']
    for i,t in enumerate(texts):
        ax.text(i*.205+.09,.76,t,ha='center',va='center',fontsize=8.5,bbox=dict(boxstyle='round',facecolor='#e5eff7'))
        if i<4:ax.annotate('',xy=(i*.205+.205,.76),xytext=(i*.205+.175,.76),arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.24,'Matched representation pilot (one seed, one epoch)\nF0: frozen cached features  |  F1: P0 tensors → trainable CNN → LSTM\nSame waveforms, initialization, source order and schedule',ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round',facecolor='#f1f6f1'))
    ax.annotate('',xy=(.33,.43),xytext=(.295,.63),arrowprops=dict(arrowstyle='->'))
    ax.annotate('',xy=(.91,.63),xytext=(.78,.32),arrowprops=dict(arrowstyle='->'))
    save(fig,'03_executed_pipeline')
    grouped=summary['seed_aggregates']; methods=[r['method'] for r in grouped];x=np.arange(len(methods))
    for number,keys,title in [('04',('clean_FAKE_miss','stress_FAKE_miss'),'FAKE miss: clean and stressed'),('05',('stress_REAL_FA','stress_FAKE_miss'),'Classwise stressed trade-off')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for offset,key in enumerate(keys):ax.bar(x+(offset-.5)*.36,[r[key+'_mean'] for r in grouped],width=.36,yerr=[r[key+'_std'] or 0 for r in grouped],capsize=3,label=key)
        ax.set_xticks(x,methods);ax.set_ylim(0,1.05);ax.set_ylabel('Error rate');ax.set_title(title+'; error bars = seed SD');ax.legend();save(fig,number+'_classwise')
    attacks=read_csv(out.parent/'attack_metrics.csv'); valid={r['experiment_id']:r for r in summary['experiments'] if r.get('config',{}).get('epochs')==6}
    fig,ax=plt.subplots(figsize=(9,4))
    for method in methods:
        values=[]
        for attack in ATTACKS:
            cells=[float(r['FAKE_miss']) for r in attacks if r['experiment_id'] in valid and valid[r['experiment_id']]['config']['augmentation_policy']==method and r['condition'] in ('noise5','A_then_B','B_then_A') and r['attack_id']==attack];values.append(float(np.mean(cells)))
        ax.plot(ATTACKS,values,marker='o',label=method)
    ax.set_ylim(0,1.05);ax.set_ylabel('Recording-weighted stress FAKE miss within attack');ax.set_title('Seen attacks; A01–A06, same development cohort');ax.legend(ncol=5);save(fig,'06_per_attack')
    per_seed=read_csv(out.parent/'per_seed.csv');fig,ax=plt.subplots(figsize=(8,4))
    for method in methods:
        g=sorted([r for r in per_seed if r['method']==method and r['epochs']=='6'],key=lambda r:r['seed']);ax.plot([r['seed'] for r in g],[float(r['stress_FAKE_miss']) for r in g],marker='o',label=method)
    ax.set_ylabel('Equal-speaker stress FAKE miss');ax.legend(ncol=5);ax.set_title('Independent training blocks; same data and finite waveforms');save(fig,'07_training_seed_variability')
    primary=next((r for r in summary['comparisons'] if r['primary']),None)
    if primary:
        fig,ax=plt.subplots(figsize=(7,4));effects=primary['paired_seed_effects'];ax.bar(list(effects),list(effects.values()));ax.axhline(0,color='black',lw=.8);ax.set_ylabel('H3 minus H2 stress miss');ax.set_title('Matched initialization, source order and two-view supervision');save(fig,'08_equal_view_contrast')
    ft=[r for r in per_seed if r['epochs']=='1']
    if len(ft)==2:
        fig,axes=plt.subplots(1,2,figsize=(9,4))
        for ax,keys in zip(axes,[('clean_REAL_FA','clean_FAKE_miss'),('stress_REAL_FA','stress_FAKE_miss')]):
            for i,r in enumerate(ft):ax.bar(np.arange(2)+(i-.5)*.35,[float(r[k]) for k in keys],width=.35,label=r['experiment_id'].split('_')[1])
            readable=[key.replace('_',' ').replace('REAL FA','REAL false alarm') for key in keys]
            ax.set_xticks(range(2),readable,rotation=10);ax.set_ylim(0,1.05);ax.set_ylabel('Error rate');ax.legend()
        fig.suptitle('Matched one-epoch F0/F1 pilot; not converged performance');save(fig,'10_representation_pilot')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));ax=axes[0];good=[r for r in summary['experiments'] if 'resources' in r]
    update=[r['resources']['update_seconds_total'] for r in good];remaining=[r['resources']['elapsed_seconds']-u for r,u in zip(good,update)]
    ax.bar(range(len(good)),update,label='Optimizer updates');ax.bar(range(len(good)),remaining,bottom=update,label='Validation, checkpoints and other I/O')
    ax.set_xticks(range(len(good)),[r['experiment_id'].replace('p7_','') for r in good],rotation=90,fontsize=6);ax.set_ylabel('Lifecycle seconds (startup excluded)');ax.set_yscale('log');ax.legend(fontsize=7)
    timing=[]
    for path in (root/BASE/'diagnostics_v1/efficiency.json',root/BASE/'representation_efficiency_v1.json'):
        if path.exists():timing.extend(json.loads(path.read_text())['runs'])
    if timing:
        axes[1].bar(range(len(timing)),[r['offline_rtf'] for r in timing]);axes[1].set_xticks(range(len(timing)),[r['experiment_id'].split('_')[1].upper() for r in timing]);axes[1].set_ylabel('Full-student offline RTF');axes[1].set_title('One-source warm CPU median; five repetitions')
    save(fig,'11_measured_training_cost')
    fig,ax=plt.subplots(figsize=(8,5))
    for r in grouped:
        ax.scatter(r['clean_REAL_FA_mean'],r['stress_FAKE_miss_mean'],s=65);ax.annotate(r['method'],(r['clean_REAL_FA_mean'],r['stress_FAKE_miss_mean']),xytext=(4,4),textcoords='offset points')
    ax.set_xlabel('Clean REAL false alarm (seed mean)');ax.set_ylabel('Stress FAKE miss (seed mean)');ax.set_title('Descriptive trade-off; registered guardrail applies to paired deltas');save(fig,'12_clean_robustness_tradeoff')
    diagnostic=root/BASE/'diagnostics_v1/proxy_controls.csv'
    if diagnostic.exists():
        rows=read_csv(diagnostic);names=list(dict.fromkeys(r['condition'] for r in rows));fig,ax=plt.subplots(figsize=(9,4));ax.boxplot([[float(r['embedding_similarity']) for r in rows if r['condition']==c] for c in names],labels=names);ax.tick_params(axis='x',rotation=15);ax.set_ylabel('Mapped frozen-feature cosine');ax.set_title('Seven-source controls; forensic evidence retention UNKNOWN');save(fig,'13_non_specific_proxy')
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');ax.text(.5,.6,'Measured development evidence → stronger standard controls',ha='center',fontsize=13);ax.text(.5,.3,'Distinct forensic mechanism: NOT_ESTABLISHED   |   External confirmation: NOT_LOCKED',ha='center',fontsize=11);save(fig,'14_novelty_evidence_status')
    write_json(out/'figure_metadata.json',dict(source_summary_sha256=file_hash(out.parent/'summary.json'),files={p.name:file_hash(p) for p in saved},
        not_drawn={'09_finite_vs_online':'Online quality experiment NOT_EXECUTED; finite H4 exists but no fabricated comparison'},neural_inference=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];cache=root/'.research_tmp/matplotlib';cache.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
    from src.lava.jaes_v2.phase7_statistics import analyze
    summary=analyze(root,args.output);figures(root,args.output/'figures',summary)
    if 'tensorflow' in sys.modules or 'torch' in sys.modules:raise AssertionError('Report must remain model-free')
    print(json.dumps(dict(experiments=len(summary['experiments']),comparisons=summary['comparisons']),indent=2))
if __name__=='__main__':main()
