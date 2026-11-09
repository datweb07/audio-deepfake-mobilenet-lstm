"""Independent development-score audit and speaker-cluster diagnostics."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, det_curve

from scripts.jaes_v2.final_order_components import BASE, OUT, METHODS, COMPONENTS, ORDERS
from scripts.jaes_v2.phase10_experiment import integrity
from scripts.jaes_v2.phase10_medium import registered
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv, write_json
from src.lava.jaes_v2.phase10_validation import STRESS, select

SEED = 91109
DRAWS = 10000
EPS = 1e-6
A, B = 'A_PAIRED', 'B_REAL_ONLY'


def expected_rows(records, b_ids):
    identities = {}
    for record in records:
        if record.split != 'development' or record.sample_id in identities:
            raise ValueError('Duplicate source or forbidden partition')
        identities[record.sample_id] = dict(parent_recording_id=record.parent_recording_id,
            speaker_id=record.speaker_id, attack_id=record.attack_id, label=str(record.label),
            stratum=B if record.sample_id in b_ids else A)
    if len({row['parent_recording_id'] for row in identities.values()}) != len(identities):
        raise ValueError('Duplicate parent recording')
    return identities


def align(rows, identities, conditions):
    result = {}
    for row in rows:
        sample, condition = row['sample_id'], row['condition_id']
        key = sample, condition, row['stratum']
        if (key in result or sample not in identities or condition not in conditions
                or any(row.get(field) != value for field, value in identities[sample].items())):
            raise ValueError('Duplicate or mismatched score identity')
        score = float(row['raw_score'])
        label = int(row['label'])
        if not 0 <= score <= 1 or float(row['threshold']) != .5 or row['status'] != 'OK' or int(row['predicted_label']) != int(score >= .5):
            raise ValueError('Invalid FAKE orientation, threshold or score')
        result[key] = dict(row, p=score, y=label, error=int((score >= .5) != label))
    if len(result) != len(identities) * len(conditions):
        raise ValueError('Incomplete score grid')
    return result


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), quantiles=np.quantile(x, [0,.1,.25,.5,.75,.9,1]).tolist())


def metrics(rows, curves=False):
    labels = np.array([r['y'] for r in rows])
    scores = np.array([r['p'] for r in rows])
    report = dict(rows=len(rows), sources=len({r['sample_id'] for r in rows}),
                  speakers=len({r['speaker_id'] for r in rows}), auc=None, eer=None, pr_auc=None)
    for label, name in ((0, 'false_alarm'), (1, 'miss')):
        part = [r for r in rows if r['y'] == label]
        report[name] = dict(errors=sum(r['error'] for r in part), n=len(part),
                            scores=describe([r['p'] for r in part]) if part else None)
    if len(set(labels)) == 2:
        fp, tp, thresholds = roc_curve(labels, scores)
        index = np.argmin(abs(fp - (1-tp)))
        report.update(auc=float(roc_auc_score(labels, scores)), eer=float((fp[index]+1-tp[index])/2),
                      pr_auc=float(average_precision_score(labels, scores)))
        if curves:
            det_fp, det_fn, _ = det_curve(labels, scores)
            report['roc'] = dict(fpr=fp.tolist(), tpr=tp.tolist(),
                                 thresholds=[float(t) if np.isfinite(t) else None for t in thresholds])
            report['det'] = dict(fpr=det_fp.tolist(), fnr=det_fn.tolist())
    return report


def transitions(first, second):
    counts = {key: 0 for key in ('both_correct','both_wrong','only_first_wrong','only_second_wrong')}
    for left, right in zip(first, second, strict=True):
        counts[('both_correct','only_second_wrong','only_first_wrong','both_wrong')[2*left+right]] += 1
    return counts


def draws(speakers, rng):
    return rng.multinomial(len(speakers), np.full(len(speakers),1/len(speakers)), size=DRAWS)


def interval(values):
    valid = np.isfinite(values)
    return dict(valid=int(valid.sum()), invalid=int((~valid).sum()),
                percentile95=np.quantile(values[valid], [.025,.975]).tolist() if valid.any() else None)


def cluster_delta(first, second, speakers, weights):
    counts, totals = [], []
    for speaker in speakers:
        indices = [i for i,r in enumerate(first) if r['speaker_id']==speaker]
        counts.append(sum(second[i]['error']-first[i]['error'] for i in indices))
        totals.append(len(indices))
    denominator = weights @ totals
    values = np.divide(weights @ counts, denominator, out=np.full(len(weights), np.nan), where=denominator>0)
    return dict(delta=sum(counts)/sum(totals), **interval(values))


def bootstrap_auc(rows, speakers, weights):
    positives = [np.array([r['p'] for r in rows if r['speaker_id']==s and r['y']==1]) for s in speakers]
    negatives = [np.array([r['p'] for r in rows if r['speaker_id']==s and r['y']==0]) for s in speakers]
    matrix = np.array([[np.sum(p[:, None] > n) + .5*np.sum(p[:, None] == n) for n in negatives] for p in positives])
    numerator = np.einsum('bi,ij,bj->b', weights, matrix, weights)
    denominator = (weights @ [len(p) for p in positives]) * (weights @ [len(n) for n in negatives])
    return np.divide(numerator, denominator, out=np.full(len(weights),np.nan), where=denominator>0)


def analyze(root):
    out = root / OUT
    target = out / 'scientific_evidence.json'
    if target.exists():
        raise FileExistsError(target)
    historical = integrity()
    inventory = engine.read(root/'docs/jaes_v2/LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE10.json')
    for item in inventory['documents']:
        if file_hash(root/item['path']) != item['sha256']:
            raise ValueError('Historical Phase 10 report changed')
    indexed = {item['path']: item['sha256'] for item in inventory['evidence']}
    qualification = {}
    for name in ('numeric_parity_v2','lifecycle_resume_v2','validation_parity_v2','benchmark_resume_v2'):
        path = BASE/(name+'.json')
        fingerprint = file_hash(root/path)
        result = engine.read(root/path)
        if indexed[path.as_posix()] != fingerprint or not result['pass_thresholds']:
            raise ValueError(f'Qualification evidence mismatch: {name}')
        qualification[name] = dict(sha256=fingerprint, **{
            key:result[key] for key in ('rows','decision_flips','steps','interrupted_at','matches') if key in result})
        if name == 'numeric_parity_v2':
            qualification[name]['passed_cases'] = sum(case['pass_thresholds'] for case in result['fixtures'])
    for role in ('uninterrupted','resumable'):
        path = BASE/'p10_engine_100_channel_s9107_v2'/role
        if indexed[(path/'seal.json').as_posix()] != file_hash(root/path/'seal.json'):
            raise ValueError('Pilot seal fingerprint changed')
        prior.verify_tree(root/path)
    _, config = registered('p10_medium_erm_s9107_v2')
    train, dev, b_ids, _ = prior.dataset(root,config)
    identities = expected_rows(dev,b_ids)
    digest = object_hash([r.to_dict() for r in train+dev])
    tables, inputs = {}, {}
    contracts, orders = {}, {}
    sealed_metrics = {}
    for method in METHODS:
        path = root / BASE / f'p10_medium_{method}_s9107_v2/scientific'
        prior.verify_tree(path)
        contract = engine.read(path/'contract.json')
        checkpoint, state = engine.committed_controller(path,contract)
        result = engine.read(path/'result.json')
        best = path/'checkpoints'/state['best']['tag']
        prior.verify_tree(best)
        if (contract['actual_sources'] != digest or contract['source'] != historical['source']
                or state != result['state'] or state['step'] != 840 or state['epoch'] != 21
                or engine.read(best/'controller.json')['model_hash'] != result['selected_weights']
                or engine.read(path/'status.json')['status'] != 'SCIENTIFIC_COMPLETE'):
            raise ValueError('Invalid scientific run provenance')
        contracts[method] = contract
        sealed_metrics[method] = result['metrics']
        orders[method] = [[item[0] for item in entry['sources']] for entry in state['journal']]
        inputs[method] = {name:file_hash(path/name) for name in ('scores.csv','contract.json','seal.json','result.json','selected.keras')}
        inputs[method].update(experiment=contract['experiment'], final=checkpoint.name, best=best.name,
                              best_seal=file_hash(best/'seal.json'), final_seal=file_hash(checkpoint/'seal.json'))
        tables[method] = align(read_csv(path/'scores.csv'), identities, prior.DEV)
    first, second = contracts.values()
    if ({**first, 'experiment':second['experiment'], 'config':{**first['config'],'augmentation_policy':'channel'}} != second
            or first['config']['augmentation_policy']!='clean' or orders['erm']!=orders['channel']):
        raise ValueError('Runs do not share the matched contract')
    guard_rows = []
    for method in METHODS:
        guard_rows.append([{**{k:v for k,v in row.items() if k not in ('p','y','error')},
                            'label':row['y'],'raw_score':row['p'],
                            'predicted_label':int(row['predicted_label'])}
                           for _,row in sorted(tables[method].items())])
    guard = select(guard_rows[1],guard_rows[0],engine.MARGINS,engine.CAPS)
    comparison = root/BASE/'medium_pair_s9107_v2.json'
    if guard != engine.read(comparison)['channel_guard_against_selected_erm']:
        raise ValueError('Guard differs from sealed-score comparison')
    groups = {}
    strata = {s:sorted({r['speaker_id'] for r in identities.values() if r['stratum']==s}) for s in (A,B)}
    rng = np.random.default_rng(SEED)
    weights = {s:draws(speakers,rng) for s,speakers in strata.items()}
    group_specs = [(f'{s}/{c}/{y}',s,(c,),y,None,None) for s in (A,B) for c in prior.DEV for y in ((0,1) if s==A else (0,))]
    group_specs += [(f'A/stress/{y}',A,STRESS,y,None,None) for y in (0,1)]
    for attack in (f'A{i:02}' for i in range(1,7)):
        group_specs += [(f'{attack}/{c}',A,(c,),1,attack,None) for c in prior.DEV]
        group_specs.append((f'{attack}/stress',A,STRESS,1,attack,None))
    group_specs += [(f'B/{speaker}/{condition}',B,(condition,),0,None,speaker)
                    for speaker in strata[B] for condition in prior.DEV]
    group_specs += [(f'A/{speaker}/{condition}/{label}',A,(condition,),label,None,speaker)
                    for speaker in strata[A] for condition in prior.DEV for label in (0,1)]
    for name,stratum,conditions,label,attack,speaker in group_specs:
        paired = [[r for _,r in sorted(tables[m].items()) if r['stratum']==stratum and r['condition_id'] in conditions
                    and r['y']==label and (attack is None or r['attack_id']==attack)
                    and (speaker is None or r['speaker_id']==speaker)] for m in METHODS]
        left,right=paired
        groups[name] = dict(erm=metrics(left), channel=metrics(right),
                            transitions=transitions([r['error'] for r in left],[r['error'] for r in right]))
        if speaker is None:
            groups[name]['paired_bootstrap']=cluster_delta(left,right,strata[stratum],weights[stratum])
        if stratum == B and speaker and conditions == ('clean',):
            worsened=[(x,y) for x,y in zip(left,right,strict=True) if not x['error'] and y['error']]
            groups[name]['new_false_alarms']=[dict(sample_id=x['sample_id'],erm=x['p'],channel=y['p'],
                near_erm=abs(x['p']-.5)<=.1, delta=y['p']-x['p']) for x,y in worsened]
    per_condition = {}
    stress_auc = {}
    for method,table in tables.items():
        per_condition[method] = {}
        for stratum in (A,B):
            for condition in prior.DEV:
                selected=[r for r in table.values() if r['stratum']==stratum and r['condition_id']==condition]
                key = f'{stratum}/{condition}'
                measured = metrics(selected, curves=True)
                archived = sealed_metrics[method][key]
                if (measured['auc'], measured['eer'], measured['pr_auc'],
                        measured['false_alarm']['errors'], measured['miss']['errors'],
                        measured['false_alarm']['n'], measured['miss']['n']) != (
                            archived['roc_auc'], archived['eer'], archived['pr_auc'],
                            archived['fp'], archived['fn'], archived['n_REAL'], archived['n_FAKE']):
                    raise ValueError(f'Recomputed metrics differ for {method}/{key}')
                per_condition[method][key] = measured
        stress_auc[method]=np.mean([bootstrap_auc([r for r in table.values() if r['stratum']==A and r['condition_id']==c],
                                                  strata[A],weights[A]) for c in STRESS],axis=0)
    components_path=out/'matched_components.json'
    components=engine.read(components_path) if components_path.exists() else None
    extra=defaultdict(dict)
    if components:
        if (components['actual_sources']!=digest or components['score_sha256']!=file_hash(out/'component_scores.csv')
                or any(components['selected_sha256'][m]!=inputs[m]['selected.keras'] for m in METHODS)):
            raise ValueError('Component provenance mismatch')
        for row in read_csv(out/'component_scores.csv'):
            key=(row['method'],row['sample_id'])
            condition=row['condition_id']; score=float(row['raw_score'])
            if key[0] not in METHODS or key[1] not in identities or condition not in COMPONENTS or condition in extra[key] or not 0<=score<=1:
                raise ValueError('Invalid component row')
            extra[key][condition]=score
        if len(extra)!=2*len(identities) or any(set(v)!=set(COMPONENTS) for v in extra.values()):
            raise ValueError('Incomplete matched component grid')
    order={}
    records={}
    for method,table in tables.items():
        records[method]={}
        for sample,identity in identities.items():
            views={c:table[sample,c,identity['stratum']] for c in prior.DEV}
            ab,ba=(views[c] for c in ORDERS)
            p_ab,p_ba=np.clip([ab['p'],ba['p']],EPS,1-EPS)
            row=dict(identity, sample_id=sample, order_delta=ab['p']-ba['p'],
                     logit_delta=float(np.log(p_ab/(1-p_ab))-np.log(p_ba/(1-p_ba))),
                     errors={c:r['error'] for c,r in views.items()}, scores={c:r['p'] for c,r in views.items()})
            if components:
                row['excess']=.5*(ab['p']+ba['p'])-sum(extra[method,sample].values())+views['clean']['p']
                row['scores'].update(extra[method,sample])
                row['errors'].update({c:int((p>=.5)!=int(identity['label'])) for c,p in extra[method,sample].items()})
            records[method][sample]=row
    order_specs=[(f'{s}/{y}',s,str(y),None) for s in (A,B) for y in ((0,1) if s==A else (0,))]
    order_specs += [(f'A{i:02}',A,'1',f'A{i:02}') for i in range(1,7)]
    for name,stratum,label,attack in order_specs:
        order[name]={}
        paired=[]
        for method in METHODS:
            rows=[r for r in records[method].values() if r['stratum']==stratum and r['label']==label and (not attack or r['attack_id']==attack)]
            paired.append(rows)
            order[name][method]=dict(score=describe([r['order_delta'] for r in rows]),
                logit=describe([r['logit_delta'] for r in rows]),
                transitions=transitions([r['errors'][ORDERS[0]] for r in rows],[r['errors'][ORDERS[1]] for r in rows]),
                all_stress_wrong=sum(all(r['errors'][c] for c in STRESS) for r in rows),
                all_seven_wrong=sum(all(r['errors'][c] for c in prior.DEV) for r in rows))
            if components:
                order[name][method].update(excess=describe([r['excess'] for r in rows]),
                    component_errors={c:sum(r['errors'][c] for r in rows) for c in COMPONENTS})
        counts=np.array([sum(r['speaker_id']==s for r in paired[0]) for s in strata[stratum]])
        for key in ('order_delta','logit_delta',*(['excess'] if components else [])):
            sums=np.array([sum(b[key]-a[key] for a,b in zip(*paired,strict=True) if a['speaker_id']==s) for s in strata[stratum]])
            denominator=weights[stratum]@counts
            values=np.divide(weights[stratum]@sums,denominator,out=np.full(DRAWS,np.nan),where=denominator>0)
            order[name][key+'_paired_bootstrap']=dict(delta=float(sum(sums)/sum(counts)),**interval(values))
    nuisance_screen = {}
    if components:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        wave = {r['sample_id']:r for r in components['waveforms']}
        for method in METHODS:
            rows = [r for r in records[method].values() if r['stratum']==A]
            fields = ('rms','peak_abs','bandwidth95_hz','alignment_lag_samples','clipping_proxy_fraction')
            features = []
            for row in rows:
                entry = wave[row['sample_id']]
                ab, ba = (entry['conditions'][c] for c in ORDERS)
                features.append([ab[k]-ba[k] for k in fields] +
                    [entry['conditions']['clean']['duration_seconds'],
                     entry['metadata']['forward_residual_snr_db']-entry['metadata']['reverse']['measured_snr_db']])
            features = np.asarray(features)
            labels = np.array([int(r['label']) for r in rows])
            speakers = np.array([r['speaker_id'] for r in rows])
            nuisance_screen[method] = {}
            for key in ('order_delta','logit_delta','excess'):
                values = np.array([r[key] for r in rows])
                predicted = np.empty(len(rows))
                for speaker in strata[A]:
                    held = speakers == speaker
                    fit = make_pipeline(StandardScaler(), Ridge(alpha=1.))
                    predicted[held] = fit.fit(features[~held],values[~held]).predict(features[held])
                residuals = values - predicted
                nuisance_screen[method][key] = dict(raw_auc=float(roc_auc_score(labels,values)),
                    residual_auc=float(roc_auc_score(labels,residuals)),
                    raw_class_mean_difference=float(values[labels==1].mean()-values[labels==0].mean()),
                    residual_class_mean_difference=float(residuals[labels==1].mean()-residuals[labels==0].mean()),
                    nuisance_rmse=float(np.sqrt(np.mean(residuals**2))))
    order_class_gap = {}
    for method in METHODS:
        part = [r for r in records[method].values() if r['stratum']==A]
        order_class_gap[method] = {}
        for key in ('order_delta','logit_delta',*(['excess'] if components else [])):
            class_means = []
            observed = []
            for label in ('0','1'):
                rows = [r for r in part if r['label']==label]
                by_speaker = [[r[key] for r in rows if r['speaker_id']==speaker] for speaker in strata[A]]
                denominator = weights[A] @ [len(group) for group in by_speaker]
                class_means.append(np.divide(weights[A] @ [sum(group) for group in by_speaker],
                    denominator, out=np.full(DRAWS,np.nan), where=denominator>0))
                observed.append(float(np.mean([r[key] for r in rows])))
            order_class_gap[method][key] = dict(fake_minus_real=observed[1]-observed[0],
                                                **interval(class_means[1]-class_means[0]))
    report=dict(date=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),engine='FROZEN_PHASE10',
                inputs=inputs, manifest_sha256=config.manifest_sha256, actual_sources=digest,
                n_sources=len(identities), score_rows={m:len(t) for m,t in tables.items()},
                speakers=strata, conditions=prior.DEV, groups=groups, per_condition=per_condition,
                macro_stress_auc={m:float(np.mean([per_condition[m][f'{A}/{c}']['auc'] for c in STRESS])) for m in METHODS},
                macro_auc_paired_bootstrap=interval(stress_auc['channel']-stress_auc['erm']),
                guard=guard, comparison_sha256=file_hash(comparison),
                order=order, records=records, nuisance_screen=nuisance_screen, order_class_gap=order_class_gap,
                bootstrap=dict(seed=SEED,draws=DRAWS,unit='speaker, strata separate'),
                logit_clip=EPS, historical=historical, qualification=qualification,
                analysis_source_sha256=file_hash(Path(__file__)),
                component_status='PASS' if components else 'NOT_AVAILABLE',
                component_report_sha256=file_hash(components_path) if components else None,
                evaluation_access=False,calibration_access=False,training=False)
    write_json(target,report)
    return report


def figures(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def bars(ax, names, keys, title):
        offsets = np.arange(len(names))
        for method, shift, color in [('erm', -.19, '#476a91'), ('channel', .19, '#b95137')]:
            counts = []
            for key in keys:
                row = report['groups'][key][method]
                counts.append(row['miss'] if row['miss']['n'] else row['false_alarm'])
            artists = ax.bar(offsets+shift,[100*r['errors']/r['n'] for r in counts],.36,
                             label=method.upper(), color=color)
            ax.bar_label(artists, [f"{r['errors']}/{r['n']}" for r in counts], padding=2, fontsize=7)
        ax.set_xticks(offsets,names)
        ax.set_ylim(0,110)
        ax.set_ylabel('Error rate (%)')
        ax.set_title(title)
        ax.legend()

    fig, axes = plt.subplots(2,1,figsize=(9,7),layout='constrained')
    bars(axes[0],['A clean REAL','A clean FAKE','B clean REAL','A stress REAL','A stress FAKE'],
         [f'{A}/clean/0',f'{A}/clean/1',f'{B}/clean/0','A/stress/0','A/stress/1'],'Classwise errors at score 0.5')
    attacks=[f'A{i:02}' for i in range(1,7)]
    bars(axes[1],attacks,[f'{attack}/stress' for attack in attacks],'FAKE misses by attack across three stress views')
    fig.savefig(out/'class_and_attack.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout='constrained')
    for method,style in [('erm','-'),('channel','--')]:
        for label,color in [('0','#476a91'),('1','#b95137')]:
            values=np.sort([r['order_delta'] for r in report['records'][method].values() if r['stratum']==A and r['label']==label])
            axes[0].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,linestyle=style,
                         label=f"{method.upper()} {'REAL' if label=='0' else 'FAKE'} (n={len(values)})")
    axes[0].set(xlabel='FAKE score: noise→codec minus codec→noise',ylabel='Empirical cumulative proportion',title='Within-record score contrasts on stratum A')
    axes[0].legend()
    speakers=report['speakers'][B]
    bars(axes[1],[s.replace('LA_','') for s in speakers],[f'B/{s}/clean' for s in speakers],
         'Clean B REAL false alarms by speaker (LA_ prefix omitted)')
    fig.savefig(out/'order_and_speaker.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    result=analyze(Path.cwd())
    figures(result, Path.cwd()/OUT)
    print(result['n_sources'],result['score_rows'],result['macro_stress_auc'],result['component_status'])
