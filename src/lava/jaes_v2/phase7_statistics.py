"""Model-free paired seed and known-speaker descriptive analyses; no independence inflation."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .evidence import read_csv,write_csv,write_json,file_hash,object_hash
from .phase3 import verify_seal
from .phase7_data import BASE,ATTACKS
from .phase7_training import score_metrics

STRESS=('noise5','A_then_B','B_then_A')

def validate_scores(rows,expected):
    keys=[(r['sample_id'],r['condition_id']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate metric/prediction rows')
    if set(keys)!=set(expected): raise ValueError('Missing or extra predictions; cannot silently change denominators')
    for r in rows:
        if float(r['threshold'])!=.5 or int(r['predicted_label'])!=int(float(r['raw_score'])>=.5): raise ValueError('Score semantics mismatch')

def speaker_risks(rows,*,label,conditions=STRESS,stratum='A_PAIRED'):
    groups=defaultdict(list)
    for r in rows:
        if int(r['label'])==label and r['condition_id'] in conditions and r['stratum']==stratum:
            groups[r['speaker_id']].append(int(int(r['predicted_label'])!=label))
    return {k:float(np.mean(v)) for k,v in sorted(groups.items())}

def paired_seed_effects(first,second,*,seed=7107):
    if set(first)!=set(second) or not first: raise ValueError('Paired seed set mismatch')
    speakers=sorted(next(iter(first.values())))
    if not speakers or any(set(x)!=set(speakers) for x in list(first.values())+list(second.values())): raise ValueError('Speaker sets differ')
    values=np.array([[first[s][p]-second[s][p] for p in speakers] for s in sorted(first)])
    per_seed=values.mean(axis=1); rng=np.random.default_rng(seed)
    cluster_draw=values[:,rng.integers(0,len(speakers),(4000,len(speakers)))].mean(axis=(0,2))
    seed_draw=per_seed[rng.integers(0,len(per_seed),(4000,len(per_seed)))].mean(axis=1)
    return dict(mean=float(per_seed.mean()),seed_std=float(per_seed.std(ddof=1)) if len(per_seed)>1 else None,
        paired_seed_effects={str(s):float(v) for s,v in zip(sorted(first),per_seed)},seed_count=len(per_seed),speaker_count=len(speakers),
        speaker_bootstrap_ci95=np.quantile(cluster_draw,[.025,.975]).tolist(),seed_bootstrap_ci95=np.quantile(seed_draw,[.025,.975]).tolist(),
        uncertainty='DESCRIPTIVE_ONLY; speaker interval conditional on trained seeds; seed interval conditional on same7speakers; only3seed blocks and7clusters',
        group_support='LIMITED_CLUSTER_PRECISION' if len(speakers)<20 else 'KNOWN_SPEAKER_CLUSTERS',p_value=None)

def collect(root):
    selection=root/BASE/'selections/medium_all_attack_v1'; source=read_csv(selection/'manifest.csv')
    expected={(r['sample_id'],condition) for r in source if r['split']=='development' for condition in ('clean','gain_half','gain_quarter','noise5','opus64','A_then_B','B_then_A')}
    runs=[]; seen=set()
    for path in sorted((root/BASE/'training').glob('*/run_metadata.json')):
        state=json.loads(path.read_text()); run=path.parent; c=state['config']; identifier=c['experiment_id']
        if identifier in seen or identifier!=run.name: raise ValueError('Duplicate/inconsistent experiment ID')
        seen.add(identifier)
        if state['status']!='PILOT_EVALUATED':
            runs.append(dict(experiment_id=identifier,status=state['status'],config=c,output=run.relative_to(root).as_posix(),failure=state.get('error'))); continue
        verify_seal(run,state)
        if file_hash(run/'best.keras')!=state['checkpoint_sha256'] or json.loads((run/'checkpoint_selection.json').read_text())['checkpoint_sha256']!=state['checkpoint_sha256']: raise ValueError('Checkpoint identity mismatch')
        rows=read_csv(run/'scores.csv'); validate_scores(rows,expected)
        # Labels/official metadata must agree exactly with selection, not just IDs.
        source_map={r['sample_id']:r for r in source}; stratum_map={r['sample_id']:r['stratum'] for r in read_csv(selection/'selection_rows.csv')}
        for r in rows:
            if any(str(r[k])!=str(source_map[r['sample_id']][k]) for k in ('label','speaker_id','attack_id')) or r['stratum']!=stratum_map[r['sample_id']]: raise ValueError('Prediction provenance mismatch')
        metrics=[]
        for condition in sorted({r['condition_id'] for r in rows}):
            for stratum in ('A_PAIRED','B_REAL_ONLY'):
                subset=[r for r in rows if r['condition_id']==condition and r['stratum']==stratum]
                metrics.append(dict(condition=condition,stratum=stratum,**score_metrics(subset)))
        old={(r['condition'],r['stratum']):r for r in json.loads((run/'metrics.json').read_text())}
        if any(old[(r['condition'],r['stratum'])]!=r for r in metrics): raise ValueError('Stale aggregate metrics')
        clean=next(r for r in metrics if r['condition']=='clean' and r['stratum']=='A_PAIRED')
        risks={str(label):speaker_risks(rows,label=label) for label in (0,1)}
        summary=dict(experiment_id=identifier,method=c['augmentation_policy'],seed=c['training_seed'],training_policy=c['training_policy'],epochs=c['epochs'],
            clean_REAL_FA=clean['REAL_false_alarm'],clean_FAKE_miss=clean['FAKE_miss'],clean_AUC=clean['roc_auc'],
            stress_REAL_FA=float(np.mean(list(risks['0'].values()))),stress_FAKE_miss=float(np.mean(list(risks['1'].values()))),
            B_clean_REAL_FA=next(r['REAL_false_alarm'] for r in metrics if r['condition']=='clean' and r['stratum']=='B_REAL_ONLY'),
            B_stress_REAL_FA=float(np.mean(list(speaker_risks(rows,label=0,stratum='B_REAL_ONLY').values()))))
        runs.append(dict(experiment_id=identifier,status=state['status'],config=c,output=run.relative_to(root).as_posix(),metadata=state,
            metadata_sha256=file_hash(path),score_sha256=file_hash(run/'scores.csv'),checkpoint_sha256=state['checkpoint_sha256'],summary=summary,metrics=metrics,rows=rows,
            resources=json.loads((run/'resource_measurements.json').read_text()),speaker_risks=risks))
    return runs

def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False); runs=collect(root); good=[r for r in runs if 'summary' in r]
    summaries=[r['summary'] for r in good]; write_csv(output/'per_seed.csv',summaries)
    grouped=[]
    for method in sorted({s['method'] for s in summaries}):
        cells=[s for s in summaries if s['method']==method and s['epochs']==6 and s['training_policy']=='HEAD_ONLY']
        if not cells: continue
        row=dict(method=method,seeds=len(cells))
        for k in ('clean_REAL_FA','clean_FAKE_miss','clean_AUC','stress_REAL_FA','stress_FAKE_miss','B_clean_REAL_FA','B_stress_REAL_FA'):
            v=np.array([s[k] for s in cells]); row[k+'_mean']=float(v.mean()); row[k+'_std']=float(v.std(ddof=1)) if len(v)>1 else None
        grouped.append(row)
    write_csv(output/'seed_aggregates.csv',grouped)
    comparisons=[]
    for first,second in [('H3','H2'),('H3','H1'),('H3','H4'),('H1','H0'),('H4','H0')]:
        a={r['config']['training_seed']:r for r in good if r['config']['augmentation_policy']==first and r['config']['epochs']==6 and r['config']['training_policy']=='HEAD_ONLY'}
        b={r['config']['training_seed']:r for r in good if r['config']['augmentation_policy']==second and r['config']['epochs']==6 and r['config']['training_policy']=='HEAD_ONLY'}
        if not a or set(a)!=set(b): continue
        if first=='H3' and second=='H2':
            for seed in a:
                if a[seed]['metadata']['initial_model_weights_sha256']!=b[seed]['metadata']['initial_model_weights_sha256'] or file_hash(root/a[seed]['output']/'training_views.csv')!=file_hash(root/b[seed]['output']/'training_views.csv'): raise ValueError('Equal-view fairness violation')
        effect=paired_seed_effects({s:r['speaker_risks']['1'] for s,r in a.items()},{s:r['speaker_risks']['1'] for s,r in b.items()})
        margins=json.loads((root/BASE/'protocol_v1.json').read_text())
        effect.update(first=first,second=second,primary=first=='H3' and second=='H2',clean_guardrails_by_seed={str(s):dict(
            REAL_FA_delta=a[s]['summary']['clean_REAL_FA']-b[s]['summary']['clean_REAL_FA'],FAKE_miss_delta=a[s]['summary']['clean_FAKE_miss']-b[s]['summary']['clean_FAKE_miss'],
            passes=a[s]['summary']['clean_REAL_FA']-b[s]['summary']['clean_REAL_FA']<=margins['clean_REAL_FA_margin'] and a[s]['summary']['clean_FAKE_miss']-b[s]['summary']['clean_FAKE_miss']<=margins['clean_FAKE_miss_margin']) for s in a})
        comparisons.append(effect)
    write_json(output/'paired_comparisons.json',comparisons)
    pareto=[]
    reference={r['summary']['seed']:r['summary'] for r in good if r['summary']['method']=='H0' and r['summary']['epochs']==6}
    for row in grouped:
        cells=[r for r in good if r['summary']['method']==row['method'] and r['summary']['epochs']==6]
        feasible=all(r['summary']['clean_REAL_FA']-reference[r['summary']['seed']]['clean_REAL_FA']<=.02 and r['summary']['clean_FAKE_miss']-reference[r['summary']['seed']]['clean_FAKE_miss']<=.02 for r in cells)
        pareto.append(dict(method=row['method'],clean_screen_vs_H0=feasible,stress_REAL_FA=row['stress_REAL_FA_mean'],stress_FAKE_miss=row['stress_FAKE_miss_mean'],
            lifecycle_seconds_mean=float(np.mean([r['resources']['elapsed_seconds'] for r in cells])),analysis='POSTHOC_DESCRIPTIVE_SECONDARY; primary guardrail remains H3vsH2'))
    for row in pareto:
        row['nondominated_stress_errors_within_screen']=row['clean_screen_vs_H0'] and not any(other['clean_screen_vs_H0'] and other['stress_REAL_FA']<=row['stress_REAL_FA'] and other['stress_FAKE_miss']<=row['stress_FAKE_miss'] and (other['stress_REAL_FA']<row['stress_REAL_FA'] or other['stress_FAKE_miss']<row['stress_FAKE_miss']) for other in pareto)
    write_json(output/'pareto_exploratory.json',pareto)
    attacks=[]; speakers=[]; conditions=[]
    for run in good:
        identifier=run['experiment_id']; rows=run['rows']
        for metric in run['metrics']: conditions.append(dict(experiment_id=identifier,**metric))
        for condition in sorted({r['condition_id'] for r in rows}):
            for attack in ATTACKS:
                subset=[r for r in rows if r['attack_id']==attack and r['condition_id']==condition and r['stratum']=='A_PAIRED']
                metric=score_metrics(subset)
                attacks.append(dict(experiment_id=identifier,condition=condition,attack_id=attack,speakers=len({r['speaker_id'] for r in subset}),**metric))
            for speaker in sorted({r['speaker_id'] for r in rows}):
                subset=[r for r in rows if r['speaker_id']==speaker and r['condition_id']==condition]
                speakers.append(dict(experiment_id=identifier,condition=condition,speaker_id=speaker,stratum=subset[0]['stratum'],**score_metrics(subset)))
    write_json(output/'condition_metrics.json',conditions); write_csv(output/'attack_metrics.csv',attacks); write_csv(output/'speaker_metrics.csv',speakers)
    inventory=[{k:v for k,v in r.items() if k not in ('rows','metrics','speaker_risks')} for r in runs]
    result=dict(status='DEVELOPMENT_ANALYSIS',experiments=inventory,seed_aggregates=grouped,comparisons=comparisons,
        condition_rows=len(conditions),attack_rows=len(attacks),speaker_rows=len(speakers),pareto_exploratory=pareto,
        limitations=['LIMITED_CLUSTER_PRECISION:7paired speakers,9REAL-only speakers','A01-A06 seen training attacks; no unseen-generator claim','Three seeds do not create independent datasets','One-class ROC_AUC/PR_AUC/EER are null; no failed prediction omitted'],
        neural_inference=False,independent_confirmation='NOT_LOCKED')
    write_json(output/'summary.json',result); return result
