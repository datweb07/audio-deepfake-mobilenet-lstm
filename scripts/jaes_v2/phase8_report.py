"""Model-free Phase8 tables/figures and deterministic comprehensive handoff."""
from pathlib import Path
import argparse,json,re,sys
import numpy as np
from src.lava.jaes_v2.evidence import write_json,write_csv,file_hash,read_csv
from src.lava.jaes_v2.phase8_training import BASE,verify_tree

DOCS=Path('docs/jaes_v2')

def table(rows,keys):
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(str(r.get(k,'UNKNOWN')) for k in keys)+' |' for r in rows])

def collect(root):
    result=[]
    for run in sorted((root/BASE/'training').glob('*')):
        if not run.is_dir():continue
        c=json.loads((run/'resolved_config.json').read_text())
        row=dict(experiment_id=run.name,phase=8,config=c,output=run.relative_to(root).as_posix(),status='INCOMPLETE')
        if (run/'status.json').exists():row.update(json.loads((run/'status.json').read_text()))
        if (run/'seal.json').exists():
            verify_tree(run)
            row.update(metrics=json.loads((run/'metrics.json').read_text()),resources=json.loads((run/'resource_measurements.json').read_text()),
                seal_sha256=file_hash(run/'seal.json'),config_sha256=file_hash(run/'resolved_config.json'))
        result.append(row)
    return result

def aggregate(root,out):
    out.mkdir(parents=True,exist_ok=False);runs=collect(root);rows=[];risk=[]
    for run in runs:
        if 'metrics' not in run:continue
        for m in run['metrics']:risk.append(dict(experiment_id=run['experiment_id'],**m))
        clean=next(m for m in run['metrics'] if m['condition']=='clean' and m['stratum']=='A_PAIRED')
        stress=[m for m in run['metrics'] if m['condition'] in ('noise5','A_then_B','B_then_A') and m['stratum']=='A_PAIRED']
        r=run['resources'];c=run['config']
        rows.append(dict(experiment_id=run['experiment_id'],seed=c['training_seed'],objective=c['training_objective'],policy=c['trainable_backbone_policy'],
            scope=c['training_scope'],steps=run['steps'],clean_REAL_FA=clean['REAL_false_alarm'],clean_FAKE_miss=clean['FAKE_miss'],clean_AUC=clean['roc_auc'],
            stress_REAL_FA=float(np.mean([m['REAL_false_alarm'] for m in stress])),stress_FAKE_miss=float(np.mean([m['FAKE_miss'] for m in stress])),
            invocation_seconds=r['elapsed_seconds'],update_seconds=r['timing']['update_seconds'],validation_seconds=r['timing']['validation_seconds'],peak_process_mib=r['peak_process_mib']))
    write_csv(out/'per_run.csv',rows);write_json(out/'classwise_conditions.json',risk)
    write_json(out/'summary.json',dict(runs=runs,per_run=rows,scope='Tiny lifecycle evidence, not scientific superiority; resumed invocations exclude earlier process time'))
    figures(root,out,rows)
    from src.lava.jaes_v2.phase8_training import seal_tree
    seal_tree(out)
    if 'tensorflow' in sys.modules or 'torch' in sys.modules:raise RuntimeError('Report must stay model-free')
    return dict(runs=len(runs),sealed=len(rows),scope='MODEL_FREE_DEVELOPMENT_ONLY')

def figures(root,out,rows):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.research_tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'lava-phase8','savefig.dpi':300})
    folder=out/'figures';folder.mkdir()
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(folder/(name+'.'+ext),metadata={'Date':None} if ext=='svg' else None)
        plt.close(fig)
    # True Phase7 effects, explicitly historical and descriptive.
    comparisons=json.loads((root/'outputs/jaes_v2/phase7/statistics/paired_comparisons.json').read_text())
    p=next(x for x in comparisons if x['primary']);fig,ax=plt.subplots(figsize=(7,4));ax.bar(list(p['paired_seed_effects']),list(p['paired_seed_effects'].values()));ax.axhline(0,color='black',lw=.8);ax.set_ylabel('H3 minus H2 stressed FAKE miss');ax.set_title('Historical Phase7: signs differ across training seeds');save(fig,'01_phase7_instability')
    selected=[r for r in rows if '_resume_' not in r['experiment_id'] and r['experiment_id'].endswith('_v3')]
    if not selected:selected=[r for r in rows if '_resume_' not in r['experiment_id'] and r['experiment_id'].endswith('_v2')]
    if selected:
        fig,ax=plt.subplots(figsize=(8,5))
        for r in selected:
            label=f"{r['objective']} {'partial' if r['policy']=='WARMUP_PARTIAL' else 'head'} seed{r['seed']}"
            ax.scatter(r['stress_REAL_FA'],r['stress_FAKE_miss'],label=label)
        ax.legend(loc='upper right',fontsize=8)
        ax.text(.03,.03,'Some operating points coincide',transform=ax.transAxes,fontsize=8)
        ax.set(xlim=(-.03,1.1),ylim=(-.03,1.1),xlabel='REAL false alarm',ylabel='FAKE miss',title='Tiny pilot classwise errors; no generalization claim');save(fig,'02_tiny_classwise_frontier')
        fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(selected));u=[r['update_seconds'] for r in selected];v=[r['validation_seconds'] for r in selected]
        ax.bar(x,u,label='Update path incl. preparation');ax.bar(x,v,bottom=u,label='Validation incl. transforms');ax.set_xticks(x,[r['experiment_id'].replace('p8_tiny_','') for r in selected],rotation=45,ha='right',fontsize=7);ax.set_ylabel('Measured invocation seconds');ax.legend();save(fig,'03_cpu_resource_components')
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
    for i,text in enumerate(['Verified train/dev\nreserved roles closed','Warmup → partial CNN\nBN unchanged','Optimizer boundary\natomic checkpoint','Fresh process resume\nstate + source journal','Selected export\nfull development']):
        ax.text(.1+i*.2,.6,text,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round',facecolor='#e8f0f7'))
        if i<4:ax.annotate('',xy=(.19+i*.2,.6),xytext=(.17+i*.2,.6),arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.15,'Schematic of implemented lifecycle; no proposed M3 mechanism or performance prediction',ha='center',fontsize=9);save(fig,'04_training_lifecycle_schematic')

def build(root,out,statistics):
    out.mkdir(parents=True,exist_ok=False)
    docs=root/DOCS;old=json.loads((docs/'LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE7.json').read_text())
    old_text=(docs/'LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE7.md').read_text(encoding='utf-8')
    sections={int(n):body for n,body in re.findall(r'^## (\d+)\. [^\n]+\n(.*?)(?=^## \d+\.|\Z)',old_text,re.M|re.S)}
    summary=json.loads((statistics/'summary.json').read_text());runs=summary['runs'];rows=summary['per_run']
    auxiliary=[]
    for path,kind in [(p,'RESOURCE_PREFLIGHT') for p in sorted((root/BASE/'preflight').glob('*.json'))]+[(root/BASE/'benchmark/partial_cpu_v1/measurements.json','RESOURCE_BENCHMARK'),(root/BASE/'audit/resume_integration_v2.json','RESUME_INTEGRATION'),(root/BASE/'audit/resume_integration_v3.json','RESUME_INTEGRATION'),(root/BASE/'audit/representation_export_seed_checks_v3.json','EXPORT_SEED_BN_INTEGRATION'),(root/BASE/'validation_final/validation.json','REGRESSION')]:
        auxiliary.append(dict(experiment_id='p8_aux_'+kind.lower()+'_'+path.parent.name+'_'+path.stem,phase=8,kind=kind,status='EXECUTED_AUXILIARY_NOT_QUALITY_MODEL',output=path.relative_to(root).as_posix(),sha256=file_hash(path)))
    checks=json.loads((root/BASE/'audit/readiness_evidence.json').read_text())
    def doc(name):return (docs/('PHASE8_'+name+'.md')).read_text(encoding='utf-8')
    def inherited(n):return 'Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):\n\n'+sections[n].strip()
    new=dict(old)
    new.update(current_phase=8,updated='2026-10-31 Asia/Novosibirsk',experiment_runs=old['experiments']+runs+auxiliary,
        experiments=old['experiments']+runs+auxiliary,training_configs=[dict(path=p.relative_to(root).as_posix(),sha256=file_hash(p),**json.loads(p.read_text())) for p in sorted((root/'configs/jaes_v2/phase8').glob('*.json'))],
        method_candidates=[dict(id='M1',status='IMPLEMENTED_STANDARD_BASELINE'),dict(id='M2',status='PILOT_EVALUATED_STANDARD_BASELINE'),dict(id='M3a',status='REJECTED_WEIGHTED_AUGMENTATION'),dict(id='M3b',status='REJECTED_WORST_OF_TWO'),dict(id='M3c',status='PROPOSED_ENTRY_GATE_FAIL')],
        novelty_status='NOT_ESTABLISHED',code_readiness=checks['FULL_TRAINING_CODE_READY'],scientific_readiness='NO',resource_requirements=checks['resources'],
        full_training_commands=checks['full_training_commands'],known_blockers=checks['known_blockers'],phase8=dict(actual_runs=len(runs),readiness_evidence_sha256=file_hash(root/BASE/'audit/readiness_evidence.json'),statistics_sha256=file_hash(statistics/'summary.json'),
            reserved_calibration_accessed=False,evaluation_accessed=False,full_data_training_executed=False),
        historical_phase7_edition=dict(markdown_sha256=file_hash(docs/'LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE7.md'),inventory_sha256=file_hash(docs/'LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE7.json')))
    new['models']=old['models']+[dict(model_id=r['experiment_id'],architecture=r['config']['architecture'],seed=r['config']['training_seed'],scope=r['config']['training_scope'],checkpoint=r.get('best'),status=r['status']) for r in runs if 'metrics' in r]
    heads=['Executive Summary','Conference Baseline','Journal Research Objectives','Complete Phase Timeline','Dataset Inventory','Model Inventory','Experiment Inventory','Phase1–4 Findings','Phase5 Findings','Phase6 Findings','Phase7 Findings','Phase8 Research Direction','Mathematical Method','Implementation','Novelty Comparison','Falsification','Baseline Matrix','Actual Phase8 Experiments','Classwise Results','Stability','Representation Training','Full Training Infrastructure','GPU Readiness','Resources','External Readiness','Limits','Contribution Matrix','Negative Results','Evidence Traceability','Reproducibility','Journal Readiness','GO / NO-GO','Exact Training Commands','Phase9 Roadmap']
    content=[
        f"FULL_TRAINING_CODE_READY: {checks['FULL_TRAINING_CODE_READY']}. NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: NO.\n\n"+doc('RESEARCH_PLAN'),
        inherited(2),inherited(3),inherited(4)+'\n\nPhase8 adds standard M1/M2, genuine multi-epoch warmup/partial training, exact-boundary resume tests and an explicit M3 rejection; no full-data experiment was automatically launched.',
        inherited(5)+'\n\n'+inherited(6),inherited(7)+'\n\nPhase8 model rows: '+str(len([r for r in runs if 'metrics' in r]))+'. Fresh-run and interrupted counterpart pairs are lifecycle checks, not additional independent seeds.',
        inherited(8)+'\n\nPhase8 run inventory:\n\n'+table(rows,['experiment_id','objective','policy','scope','steps'])+'\n\nAuxiliary records (not additional quality models):\n\n'+table(auxiliary,['experiment_id','kind','output']),
        inherited(9),inherited(10),inherited(11),inherited(12)+'\n\n'+inherited(16),doc('RESEARCH_PLAN'),doc('C2_METHOD_SPECIFICATION'),
        'New modules: phase8_training.py, phase8_objectives.py; CLI phase8_experiment.py; standard helpers/source data remain inherited and unchanged.\n\n'+doc('FULL_TRAINING_READINESS'),
        'Detailed primary-source matrix: [PHASE8_INTERNATIONAL_NOVELTY_MATRIX.md](PHASE8_INTERNATIONAL_NOVELTY_MATRIX.md). SONAR is now FULL_TEXT_VERIFIED from the official PDF. M1/M2 are standard, M3a/b rejected by algebra; M3c lacks a justified evidence target. No world-first assertion.',
        doc('METHOD_REDIRECTION'),doc('METHOD_COMPARISON_PLAN'),doc('EXPERIMENT_RESULTS'),
        table(rows,['experiment_id','clean_REAL_FA','clean_FAKE_miss','stress_REAL_FA','stress_FAKE_miss','clean_AUC'])+'\n\nThese tiny selected cohorts are for lifecycle testing; stress means here are recording-weighted condition means. Full per-condition/per-attack/null-denominator values remain in each sealed run and model-free tables. They do not replace Phase7 equal-speaker scientific estimands.',
        inherited(16)+'\n\nNew seed initialization/resume checks:\n\n```json\n'+json.dumps(checks['integration'],indent=2)+'\n```\n\nTiny independent ERM seed scores do not establish a stable method effect. Interrupted counterpart shares its seed and is not independent.',
        'Actual warmup/partial gradients, unchanged BN and exported model checks:\n\n```json\n'+json.dumps(checks['representation'],indent=2)+'\n```',
        doc('FULL_TRAINING_READINESS'),doc('GPU_TRAINING_GUIDE'),doc('FULL_TRAINING_RESOURCE_PLAN'),
        'Reserved7,341 calibration records and official evaluation remain unused. External confirmation NOT_LOCKED. MLAAD-tiny is prior development; RADAR/ITW audio not acquired. Selection, hyperparameters, model/seed family, source/license/overlap and calibration/primary tests must lock before confirmation.',
        'Tiny pilots, CPU-only runtime validation, finite augmentation, already-used development compositions, unknown original utterance/generator lineage, unmeasured long-run convergence and absent independent confirmation limit interpretation. Full long jobs and GPU tests are not executed. Early-stopping and guardrails are engineering selection policies, not deployment risk bounds.',
        '| Candidate | Status | Evidence |\n|---|---|---|\n| C1 | supporting empirical/engineering | historical controlled channels |\n| C2 | novelty NOT_ESTABLISHED | standard baselines and rejected M3 specifications |\n| C3 | DEFERRED | no cascade/calibration run |',
        inherited(23)+'\n\n'+doc('NOVELTY_DECISION')+'\n\nFirst Phase8 preflight rejected absolute in-workspace audio paths; the source resolver was corrected before a training output existed. First successful head pilot exposed tf.function retracing. Executed source versions and pilots are retained, not overwritten. Codec preparation dominates the measured first partial pilot; subsequent resource changes are versioned and compared.',
        'Every completed run has config/data/code contracts, initialization hash, source/view update journals, Adam checkpoints and hashes, selected export, scores, classwise metrics, resources and a recursive artifact seal. Historical snapshot and accepted manuscript are verified separately. See outputs/jaes_v2/phase8/audit/readiness_evidence.json and the final integrity audit. A statistical result has to trace to a sealed run, not an illustrative method diagram.',
        doc('NEXT_COMMANDS')+'\n\n```json\n'+json.dumps(checks['tests'],indent=2)+'\n```\n\nHistorical Phase7 master/inventory hashes are retained in the new inventory. The model-free builder produces byte-identical report/JSON from the same fixed evidence. The final source snapshot archives new implementation/config/test/document bytes.',
        inherited(29)+'\n\nPhase8 does not resolve conference venue, disclosure/rights or authenticated Clarivate indexing. A new trainer does not make the paper submission-ready.',
        doc('FULL_TRAINING_READINESS')+'\n\n'+doc('NOVELTY_DECISION'),doc('FULL_TRAINING_LAUNCH_PACKAGE'),
        'First run matched medium standard representation baselines under the handed-off commands. Inspect both class errors and instability; use results to find a residual problem that survives ordinary controls. Pursue one of the two redirection proposals only after a justified evidence target/technical distinction is established. Then lock a new scientific protocol; independent confirmation and C3 follow later.']
    if len(heads)!=len(content):raise ValueError('Missing master section')
    text='# LAVA JOURNAL V2 — MASTER RESEARCH REPORT — PHASE8 EDITION\n\nEvidence-generated; historical Phase7 edition preserved. New results are explicitly separated from inherited evidence.\n\n'
    for i,(head,body) in enumerate(zip(heads,content),1):text+=f'## {i:02}. {head}\n\n{body.strip()}\n\n'
    (out/'LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE8.md').write_text(text,encoding='utf-8')
    write_json(out/'LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE8.json',new)
    result=dict(sections=len(heads),markdown_sha256=file_hash(out/'LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE8.md'),inventory_sha256=file_hash(out/'LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE8.json'))
    write_json(out/'validation.json',result)
    if 'tensorflow' in sys.modules:raise RuntimeError('Master builder imported TensorFlow')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('aggregate','master'));p.add_argument('--output',type=Path,required=True);p.add_argument('--statistics',type=Path);args=p.parse_args();root=Path(__file__).resolve().parents[2]
    if args.mode=='aggregate':result=aggregate(root,args.output)
    else:
        if not args.statistics:p.error('--statistics required')
        result=build(root,args.output,args.statistics)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
