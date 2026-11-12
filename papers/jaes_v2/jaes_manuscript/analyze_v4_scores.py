"""Generate V4 diagnostics from verified scores, without model inference."""
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from analyze_locked_heldout import ROOT, OUT, file_hash, object_hash, scoring_lock, verified_scores, read
from analyze_v3_scores import training_exposure
from generate_review_figures import load_runs
from scripts.jaes_v2.final_score_diagnostic import metrics, describe, transitions
from src.lava.jaes_v2.evidence import read_csv, write_csv, write_json

DEST = ROOT/'outputs/jaes_v2/supervisor_major_revision'
THRESHOLDS = (.10,.20,.30,.40,.45,.50,.55,.60,.70,.80,.90)
ORDERS = ('A_then_B','B_then_A')
STRESS = ('noise5', *ORDERS)
METHODS = ('erm','channel')


def confusion(rows, threshold):
    counts = Counter((r['y'], r['p'] >= threshold) for r in rows)
    tn, fp, fn, tp = [counts[k] for k in ((0,False),(0,True),(1,False),(1,True))]
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, REAL=tn+fp, FAKE=fn+tp,
                FPR=fp/(tn+fp) if tn+fp else None, TNR=tn/(tn+fp) if tn+fp else None,
                FNR=fn/(fn+tp) if fn+tp else None, TPR=tp/(fn+tp) if fn+tp else None)


def by_source(rows):
    return {r['sample_id']:r for r in rows}


def attack_rows(groups, output):
    summaries, identities = [], []
    for partition, attacks in [('A_PAIRED',('A05','A06')),('heldout',('A17','A18','A19'))]:
        for seed in ('9107','9207'):
            conditions = [c for s,p,m,c in groups if (s,p,m)==(seed,partition,'erm')]
            for attack in attacks:
                for condition in (*conditions,'pooled_stress'):
                    select = STRESS if condition=='pooled_stress' else (condition,)
                    speaker_errors, harmed, recovered, source_ids = defaultdict(list), Counter(), Counter(), set()
                    counts = Counter()
                    misses = {method:Counter() for method in METHODS}
                    for cond in select:
                        pair = [by_source([r for r in groups[seed,partition,m,cond] if r['attack']==attack])
                                for m in METHODS]
                        left, right = pair
                        if not left or left.keys() != right.keys():
                            raise ValueError('Invalid paired attack support')
                        for sample, er in sorted(left.items()):
                            ch = right[sample]
                            if er['speaker_id'] != ch['speaker_id'] or er['y'] != 1 or ch['y'] != 1:
                                raise ValueError('Attack identity mismatch')
                            first, second = er['error'], ch['error']
                            source_ids.add(sample)
                            speaker_errors[er['speaker_id']].append(second-first)
                            harmed[sample] += second > first
                            recovered[sample] += first > second
                            misses['erm'][sample] += first
                            misses['channel'][sample] += second
                            counts[first,second] += 1
                            if condition != 'pooled_stress':
                                identities.append(dict(partition=partition, seed=seed, attack=attack,
                                    condition=cond, source=sample, speaker=er['speaker_id'], erm_miss=first, channel_miss=second))
                    n = sum(counts.values())
                    left = counts[1,1]+counts[1,0]
                    right = counts[1,1]+counts[0,1]
                    sp = list(speaker_errors.values())
                    draws = np.random.default_rng(91109).multinomial(len(sp), np.full(len(sp),1/len(sp)),size=10000)
                    values = (draws @ [sum(x) for x in sp])/(draws @ [len(x) for x in sp])
                    interval = np.quantile(values,[.025,.975]).tolist() if len(sp)>1 else [None,None]
                    summaries.append(dict(partition=partition,seed=seed,attack=attack,condition=condition,
                        sources=len(source_ids),views=n,speakers=len(sp),ERM_misses=left,Channel_misses=right,
                        ERM_FNR=left/n,Channel_FNR=right/n,delta=(right-left)/n,both_wrong=counts[1,1],
                        corrected=counts[1,0],introduced=counts[0,1],both_correct=counts[0,0],
                        distinct_harmed=sum(x>0 for x in harmed.values()),multiple_harms=sum(x>1 for x in harmed.values()),
                        distinct_recovered=sum(x>0 for x in recovered.values()),multiple_recoveries=sum(x>1 for x in recovered.values()),
                        ERM_repeated_misses=sum(x>1 for x in misses['erm'].values()),
                        Channel_repeated_misses=sum(x>1 for x in misses['channel'].values()),
                        pointwise_lower=interval[0],pointwise_upper=interval[1],valid_draws=10000 if len(sp)>1 else 0,
                        invalid_draws=0, status='POST_HOC_EXPLORATORY'))
    write_csv(output/'attack_recovery.csv',summaries)
    write_csv(output/'attack_source_decisions.csv',identities)


def component_summary(groups):
    records = []
    for seed, directory in [('9107',ROOT/OUT.parent/'final_diagnostic_v1'),('9207',DEST/'components_9207')]:
        audit = directory/'matched_components.json'
        if not audit.exists():
            continue
        info = read(audit)
        csv_path = directory/'component_scores.csv'
        values = read_csv(csv_path)
        if len(values)!=1472 or file_hash(csv_path)!=info['score_sha256']:
            raise ValueError('Invalid standalone component scores')
        anchors = {r['sample_id']:r for partition in ('A_PAIRED','B_REAL_ONLY')
                   for r in groups[seed,partition,'erm','clean']}
        wave = {r['sample_id']:r for r in info['waveforms']}
        grid = {(r['method'],r['sample_id'],r['condition_id']):float(r['raw_score']) for r in values}
        if set(wave)!=set(anchors) or set(grid)!={(m,s,c) for m in METHODS for s in anchors for c in ('noise10','opus24')}:
            raise ValueError('Incomplete standalone component identities')
        for method in METHODS:
            for partition in ('A_PAIRED','B_REAL_ONLY'):
                views = {c:by_source(groups[seed,partition,method,c]) for c in ('clean',*ORDERS)}
                for y in sorted({r['y'] for r in views['clean'].values()}):
                    ids = [s for s,r in views['clean'].items() if r['y']==y]
                    component = {c:[grid[method,s,c] for s in ids] for c in ('noise10','opus24')}
                    residual = [.5*(views[ORDERS[0]][s]['p']+views[ORDERS[1]][s]['p'])-
                                component['noise10'][i]-component['opus24'][i]+views['clean'][s]['p']
                                for i,s in enumerate(ids)]
                    records.append(dict(seed=seed,method=method,partition=partition,label=y,
                        source_count=len(ids),source_sha256=file_hash(csv_path),
                        scores={c:describe(p) for c,p in component.items()},
                        errors={c:sum((p>=.5)!=y for p in values) for c,values in component.items()},
                        descriptive_score_interaction=describe(residual),
                        status='POST_HOC_EXPLORATORY'))
    write_json(DEST/'component_summary.json', records)


def main():
    base = ROOT/OUT
    proposed = base/'heldout_proposed_9107_9207_v1.json'
    lock = scoring_lock(proposed, base/'heldout_scoring_author_approval.json')
    digest = object_hash(lock)
    directory = base/f'heldout_{digest[:16]}'
    rows, _, _, shards = verified_scores(directory,lock,digest)
    v3 = read(directory/'analysis/verified_analysis.json')
    if shards != v3['integrity']['score_shards']:
        raise ValueError('Held-out shard fingerprint changed')
    evidence = base/'post9207_consolidated_v1.json'
    grids, metadata, _, sources = load_runs(read(evidence))
    exposure = training_exposure(base,lock['seeds'])
    groups = defaultdict(list)
    for row in rows:
        method, seed = row['model'].split('_')
        groups[seed,'heldout',method,row['condition']].append(row)
    for (seed,method), scores in grids.items():
        for (sample,condition), p in scores.items():
            y, partition, _, speaker, attack = metadata[sample,condition]
            groups[str(seed),partition,method,condition].append(dict(sample_id=sample,speaker_id=speaker,
                                attack=attack,y=y,p=p,error=int((p>=.5)!=y)))
    risk, windows, distribution, curves, order, order_sources = [], [], [], {}, [], []
    for (seed,partition,method,condition), part in sorted(groups.items()):
        key = dict(seed=seed,partition=partition,method=method,condition=condition)
        for t in THRESHOLDS:
            status = ('LOCKED_HELDOUT' if partition=='heldout' else 'PROSPECTIVE_DEVELOPMENT') if t==.5 else 'POST_HOC_EXPLORATORY'
            risk.append(dict(key,threshold=t,status=status,**confusion(part,t)))
        evaluation = metrics(part,curves=True)
        curves['/'.join(key.values())] = {k:evaluation.get(k) for k in ('auc','eer','roc','det')}
        for y in sorted({r['y'] for r in part}):
            values = [r['p'] for r in part if r['y']==y]
            distribution.append(dict(key,label=y,**describe(values)))
            for w in (.025,.05):
                windows.append(dict(key,label=y,window=w,count=sum(abs(p-.5)<=w for p in values),n=len(values)))
        if condition != ORDERS[0]:
            continue
        reverse = by_source(groups[seed,partition,method,ORDERS[1]])
        for y in sorted({r['y'] for r in part}):
            first = sorted((r for r in part if r['y']==y), key=lambda r:r['sample_id'])
            second = [reverse[r['sample_id']] for r in first]
            deltas = [a['p']-b['p'] for a,b in zip(first,second,strict=True)]
            order.append(dict(seed=seed,partition=partition,method=method,label=y,delta=describe(deltas),
                **transitions([r['error'] for r in first],[r['error'] for r in second])))
            for a,b,delta in zip(first,second,deltas,strict=True):
                order_sources.append(dict(seed=seed,partition=partition,method=method,label=y,source=a['sample_id'],
                    speaker=a['speaker_id'],attack=a['attack'],first_score=a['p'],second_score=b['p'],delta=delta,
                    first_error=a['error'],second_error=b['error']))
    DEST.mkdir(parents=True,exist_ok=True)
    write_csv(DEST/'threshold_risk.csv',risk)
    write_csv(DEST/'confusion_at_050.csv',[row for row in risk if row['threshold']==.5])
    write_csv(DEST/'near_threshold.csv',windows)
    write_csv(DEST/'order_source_decisions.csv',order_sources)
    write_json(DEST/'score_and_order_diagnostics.json',dict(status='POST_HOC_EXPLORATORY',distributions=distribution,order=order,curves=curves))
    attack_rows(groups,DEST)
    component_summary(groups)
    fig, axes = plt.subplots(2,2,figsize=(6.8,4.9),layout='constrained')
    for method,color in zip(METHODS,('#17608a','#a94d11'),strict=True):
        part = groups['9207','heldout',method,'A_then_B']
        values = [confusion(part,t) for t in THRESHOLDS]
        for k, style in [('FPR','-'),('FNR','--')]:
            axes[0,0].plot(THRESHOLDS,[v[k] for v in values],style,color=color,label=f'{method.upper()} {k}')
        for label,style in [(0,'-'),(1,'--')]:
            scores=np.sort([r['p'] for r in part if r['y']==label])
            axes[0,1].step(scores,np.arange(1,len(scores)+1)/len(scores),where='post',ls=style,
                           color=color,label=f'{method.upper()} {("REAL","FAKE")[label]}')
        curve = curves[f'9207/heldout/{method}/A_then_B']
        axes[1,0].plot(curve['roc']['fpr'],curve['roc']['tpr'],color=color,label=method.upper())
        det = curve['det']
        transform = lambda values,n: norm.ppf(np.clip(values,.5/n,1-.5/n))
        axes[1,1].plot(transform(det['fpr'],256),transform(det['fnr'],832),color=color,label=method.upper())
    for ax,labels in zip(axes.flat,[('Threshold','Error rate'),('FAKE score','Empirical CDF'),
                                   ('False-alarm rate','True-positive rate'),('False-alarm rate (%)','Miss rate (%)')]):
        ax.set(xlabel=labels[0],ylabel=labels[1]); ax.legend(fontsize=7)
    ticks=[.01,.05,.2,.5,.8,.95]
    axes[1,1].set_xticks(norm.ppf(ticks),[str(round(t*100)) for t in ticks])
    axes[1,1].set_yticks(norm.ppf(ticks),[str(round(t*100)) for t in ticks])
    for ax in axes[0]:
        ax.axvline(.5,color='.5',linewidth=.7)
    fig.savefig(DEST/'operating_points_v4.pdf',metadata={'CreationDate':None})
    plt.close(fig)
    inputs = [proposed,directory/'manifest.json',directory/'metrics.json',evidence]
    write_json(DEST/'score_provenance.json',dict(heldout_predictions=len(rows),shards=len(shards),
        development_predictions=sum(len(g) for g in grids.values()),threshold_grid=THRESHOLDS,
        locked_threshold=.5, no_inference=True, no_training=True, lock_sha256=digest,
        input_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in inputs},development_seals=sources,
        training_exposure=exposure, analysis_script_sha256=file_hash(Path(__file__))))


if __name__ == '__main__':
    main()

