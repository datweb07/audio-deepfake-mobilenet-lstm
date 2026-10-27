"""Complete public-development paired summaries with explicit single-class denominators."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.lava.evaluation_metrics import compute_eer
from .analysis import loss, parent_bootstrap
from .factorial import generator_channel_interaction
from .provenance import UNKNOWN, known

PUBLIC_CONDITIONS = ('clean_native', 'clean', 'pcm16', 'codec', 'fixed_noise', 'fixed_A_then_B', 'fixed_B_then_A')


def validate_public_scores(rows: list[dict], manifest: list[dict]) -> dict:
    sources = {row['sample_id']:row for row in manifest}
    if not sources or len(sources) != len(manifest):
        raise ValueError('Empty/duplicate manifest')
    expected = {(identity, condition) for identity in sources for condition in PUBLIC_CONDITIONS}
    if len(rows) != len(expected) or {(row['sample_id'],row['condition_id']) for row in rows} != expected:
        raise ValueError('Incomplete or duplicate public paired scores')
    result = defaultdict(dict)
    for row in rows:
        original = sources[row['sample_id']]
        for field in ('label','dataset_id','parent_recording_id','generator_id'):
            if str(row[field]) != str(original[field]):
                raise ValueError('Score/manifest identity mismatch: ' + field)
        score, threshold = float(row['raw_score']), float(row['threshold'])
        if row['status'] != 'OK' or int(row['label']) not in (0,1) or not np.isfinite([score,threshold]).all() or \
                not 0 <= score <= 1 or not 0 <= threshold <= 1 or int(row['predicted_label']) != int(score >= threshold):
            raise ValueError('Invalid score or frozen threshold decision')
        result[row['sample_id']][row['condition_id']] = {**row, **{key:original[key] for key in
            ('language','source_corpus','source_speaker_id','parent_scope')}}
    for field in ('model_id','run_id','threshold','seed'):
        if len({str(row[field]) for row in rows}) != 1:
            raise ValueError('Mixed analysis stratum: ' + field)
    return dict(result)


def metrics(rows: list[dict]) -> dict:
    labels = np.array([int(row['label']) for row in rows])
    predicted = np.array([int(row['predicted_label']) for row in rows])
    scores = np.array([float(row['raw_score']) for row in rows])
    real, fake = labels == 0, labels == 1
    both = bool(real.any() and fake.any())
    return {'n':len(rows), 'real':int(real.sum()), 'fake':int(fake.sum()),
            'error_rate':float(np.mean(labels != predicted)),
            'false_alarm_rate':float(predicted[real].mean()) if real.any() else None,
            'miss_rate':float((1-predicted[fake]).mean()) if fake.any() else None,
            'f1':float(f1_score(labels,predicted,zero_division=0)) if both else None,
            'auc':float(roc_auc_score(labels,scores)) if both else None,
            'eer':float(compute_eer(labels,scores)[0]) if both else None,
            'ranking_status':'DEFINED_TWO_CLASS' if both else 'UNDEFINED_SINGLE_CLASS',
            'mean_raw_score':float(scores.mean())}


def summarize_public(rows: list[dict], manifest: list[dict], *, replicates: int, seed: int) -> dict:
    paired = validate_public_scores(rows, manifest)
    generators = sorted({row['generator_id'] for row in manifest if int(row['label']) == 1 and known(row['generator_id'])})
    tables, contrasts = [], []
    for group in ['ALL','REAL'] + generators:
        members = [sample for sample in paired.values() if group == 'ALL' or
                   (group == 'REAL' and int(sample['clean']['label']) == 0) or
                   (group in generators and int(sample['clean']['label']) == 1 and sample['clean']['generator_id'] == group)]
        for condition in PUBLIC_CONDITIONS:
            tables.append({'group':group,'condition':condition,**metrics([sample[condition] for sample in members])})
        # Source-text speaker clusters for fake groups; union of source-recording
        # keys is only a conservative grouping diagnostic, not independence proof.
        for field in ('parent_recording_id','source_speaker_id'):
            if any(not known(str(sample['clean'].get(field,UNKNOWN))) for sample in members):
                contrasts.append({'group':group,'cluster_field':field,'status':'NOT_IDENTIFIABLE_UNKNOWN_CLUSTER'})
                continue
            for risk in ('error','squared_score_error'):
                values = defaultdict(list)
                for sample in members:
                    values[sample['clean'][field]].append(loss(sample['fixed_A_then_B'],risk)-loss(sample['fixed_B_then_A'],risk))
                contrasts.append({'group':group,'cluster_field':field,'risk':risk,
                                  'status':'DESCRIPTIVE_DEVELOPMENT; NOT_POPULATION_INDEPENDENCE',
                                  'recordings':len(members), **parent_bootstrap(values,replicates=replicates,seed=seed)})
    interactions = []
    for first, second in combinations(generators,2):
        selected = []
        for sample in paired.values():
            if int(sample['clean']['label']) != 1 or sample['clean']['generator_id'] not in {first,second}:
                continue
            selected.extend({**row,'factor_group':'G0' if row['generator_id']==first else 'G1'} for row in sample.values())
        for channel in ('fixed_A_then_B','fixed_B_then_A'):
            try:
                estimate = generator_channel_interaction(selected,channel=channel,
                    factor_definition=f'G0 author model_name={first}; G1 author model_name={second}; English FAKE only; checkpoint familiarity UNKNOWN',
                    factor_kind='generator',class_label=1,cluster_field='source_speaker_id',replicates=replicates,seed=seed)
                interactions.append({'G0':first,'G1':second,'channel':channel,'status':'DESCRIPTIVE_IDENTIFIABLE_ANNOTATED_FACTOR',**estimate})
            except ValueError as error:
                interactions.append({'G0':first,'G1':second,'channel':channel,'status':'NOT_IDENTIFIABLE','reason':str(error)})
    return {'scope':'EXPLORATORY_PUBLIC_DEVELOPMENT', 'metric_table':tables,'order_contrasts':contrasts,
            'generator_channel_interactions':interactions, 'model_training_familiarity':UNKNOWN,
            'independent_population_units':UNKNOWN,
            'multiple_comparisons':'All prespecified pairwise descriptions reported; no p-values, significance or discoveries; no selective reporting',
            'limitations':['Author system name is not a checkpoint hash','Source-text speaker is not synthesized-voice identity',
                           'REAL source lineage UNKNOWN; parent-based intervals do not establish independence',
                           'Original-source corpus shared does not equal recording/acoustic environment control',
                           'Group composition and synthesized speakers remain observational confounders']}
