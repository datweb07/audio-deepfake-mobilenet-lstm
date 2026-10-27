"""Within-record preprocessing sensitivity; never a source-matched generator effect."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .analysis import loss, parent_bootstrap
from .nuisance_controls import POLICIES
from .precision import plan_precision
from .public_analysis import metrics, PUBLIC_CONDITIONS
from .shortcut import available_source_clusters

NEW_CHANNELS=('clean','fixed_A_then_B','fixed_B_then_A')


def summarize(rows: list[dict], manifest: list[dict], *, seed: int=4804, replicates: int=2000) -> dict:
    sources={row['sample_id']:row for row in manifest}
    expected={(identity,policy,condition) for identity in sources for policy in POLICIES for condition in NEW_CHANNELS}
    expected|={(identity,'phase3_baseline',condition) for identity in sources for condition in PUBLIC_CONDITIONS}
    if len(sources)!=len(manifest) or len(rows)!=len(expected) or {(r['sample_id'],r['policy'],r['condition_id']) for r in rows}!=expected:
        raise ValueError('Missing/duplicate complete nuisance cells')
    index={}
    for row in rows:
        source=sources[row['sample_id']]
        if source['split']!='development' or row['status']!='OK':
            raise ValueError('Development only and no failed-case deletion')
        for key in ('label','generator_id','parent_recording_id','dataset_id'):
            if str(row[key])!=str(source[key]):
                raise ValueError('Score/source mismatch: '+key)
        score,threshold=float(row['raw_score']),float(row['threshold'])
        if not np.isfinite([score,threshold]).all() or not 0<=score<=1 or not 0<=threshold<=1 or int(row['predicted_label'])!=int(score>=threshold):
            raise ValueError('Invalid score/fixed threshold')
        index[(row['sample_id'],row['policy'],row['condition_id'])]=row
    for field in ('model_id','threshold'):
        if len({str(row[field]) for row in rows})!=1:
            raise ValueError('Mixed detector stratum')
    table=[]
    for policy in ('phase3_baseline',)+POLICIES:
        for condition in PUBLIC_CONDITIONS if policy=='phase3_baseline' else NEW_CHANNELS:
            selected=[index[(identity,policy,condition)] for identity in sources]
            raw=np.array([float(row['raw_score']) for row in selected])
            class_scores={}
            for label in (0,1):
                values=np.array([float(row['raw_score']) for row in selected if int(row['label'])==label])
                class_scores[str(label)]={'n':len(values),'quantiles_05_50_95':np.quantile(values,[.05,.5,.95]).tolist(),
                                         'mean':float(values.mean()),'std':float(values.std()),
                                         'mean_margin_to_fixed_threshold':float(values.mean()-float(selected[0]['threshold']))}
            metric=metrics(selected)
            metric['balanced_error']=.5*(metric['false_alarm_rate']+metric['miss_rate'])
            flips=0; score_shift=[]
            for row in selected:
                reference=index[(row['sample_id'],'phase3_baseline',condition)]
                flips+=int(row['predicted_label'])!=int(reference['predicted_label'])
                score_shift.append(float(row['raw_score'])-float(reference['raw_score']))
            table.append({'policy':policy,'condition':condition,**metric,'decision_flips_vs_same_channel_baseline':flips,
                          'mean_paired_score_shift':float(np.mean(score_shift)),'class_score_distributions':class_scores})
    cluster_map=dict(zip(sources,available_source_clusters(list(sources.values()))))
    contrasts=[]; plans=[]
    for policy in POLICIES:
        for label in ('ALL','REAL','FAKE'):
            for risk in ('error','squared_score_error'):
                groups=defaultdict(list)
                order_groups=defaultdict(list)
                count=0
                for identity,source in sources.items():
                    if label!='ALL' and int(source['label'])!=int(label=='FAKE'):
                        continue
                    def order(member):
                        return loss(index[(identity,member,'fixed_A_then_B')],risk)-loss(index[(identity,member,'fixed_B_then_A')],risk)
                    value=order(policy)
                    groups[cluster_map[identity]].append(value-order('phase3_baseline'))
                    order_groups[cluster_map[identity]].append(value)
                    count+=1
                estimate=parent_bootstrap(groups,replicates=replicates,seed=seed)
                contrasts.append({'estimand':'WITHIN_SOURCE_ORDER_EFFECT','contrast':'controlled order minus baseline order',
                                  'policy':policy,'class':label,'risk':risk,'recordings':count,
                                  'cluster_unit':'connected known original speaker/parent; UNKNOWN REAL stays local',
                                  **estimate,'controlled_order':parent_bootstrap(order_groups,replicates=replicates,seed=seed),
                                  'population_status':'INSUFFICIENT_INDEPENDENT_GROUPS; REAL lineage UNKNOWN',
                                  'primary':policy=='edge_repack_v1' and label=='FAKE' and risk=='error'})
                if label=='FAKE' and risk=='error':
                    plans.append({'policy':policy,**plan_precision(groups)})
    return {'scope':'EXPLORATORY_WITHIN_RECORD_DEVELOPMENT','estimand_type':'WITHIN_SOURCE_ORDER_EFFECT',
            'matched_generator_interaction':'SOURCE_MATCHED_EXPERIMENT_BLOCKED',
            'metrics':table,'contrasts':contrasts,'precision_plans':plans,'threshold_fitting':'NONE',
            'primary':'edge_repack_v1 change in FAKE order miss contrast, equal known source-text-speaker weight',
            'multiplicity':'All 18 descriptive contrasts reported; no p-values/significance; intervals not simultaneous',
            'independent_population_units':'UNKNOWN','limitations':['Not causal attribution to silence or amplitude',
            'Content shift changes segment alignment; low-energy tails can contain phonetic evidence',
            'Only four FAKE original source-text speakers; REAL lineage unresolved','No source-matched generator effect or confirmation']}
