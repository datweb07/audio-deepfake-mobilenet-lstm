"""Fixed Phase 10 score traversal and complete development evidence."""
from time import perf_counter

import numpy as np

from .phase7_training import score_metrics
from .phase8_training import DEV
from .phase9_protocol import guarded_selection

STRESS = ('noise5', 'A_then_B', 'B_then_A')
THRESHOLD = .5
MARGINS = {'A_clean_FA': '0.02', 'A_clean_miss': '0.02', 'B_clean_FA': '0.02',
           'A05_stress_miss': '0.05', 'A06_stress_miss': '0.05'}
TIERS = {'A': ('clean',), 'B': ('clean', *STRESS), 'C': DEV}


def plan(records, b_ids, tier):
    for record in records:
        is_b = record.sample_id in b_ids
        conditions = TIERS[tier] if not is_b or tier == 'C' else (() if tier == 'A' else ('clean',))
        for condition in conditions:
            yield record, condition, 'B_REAL_ONLY' if is_b else 'A_PAIRED'


def score(records, b_ids, tier, inputs, inference, guard):
    """Reuse each input across the compared paths without retaining a cohort cache."""
    rows = {name: [] for name in inference}
    durations = dict.fromkeys(inference, 0.)
    started = perf_counter()
    seen = set()
    for record, condition, stratum in plan(records, b_ids, tier):
        guard()
        key = (record.sample_id, condition)
        if key in seen:
            raise ValueError(f'Duplicate view {key}')
        seen.add(key)
        x = inputs.get(record, condition)[None]
        identity = dict(sample_id=record.sample_id, parent_recording_id=record.parent_recording_id,
                        speaker_id=record.speaker_id, attack_id=record.attack_id, label=record.label,
                        condition_id=condition, stratum=stratum, threshold=THRESHOLD, status='OK')
        for name, forward in inference.items():
            t = perf_counter()
            p = float(forward(x).numpy().reshape(-1)[0])
            durations[name] += perf_counter() - t
            if not 0 <= p <= 1:
                raise ValueError(f'Invalid score {name}: {key}')
            rows[name].append(dict(identity, raw_score=p, predicted_label=int(p >= THRESHOLD)))
    if not seen:
        raise ValueError('Empty scoring plan')
    return rows, dict(forward_seconds=durations, elapsed_seconds=perf_counter() - started)


def metrics(rows):
    keys = sorted({(r['stratum'], r['condition_id']) for r in rows})
    return {f'{s}/{c}': score_metrics([r for r in rows if r['stratum'] == s and r['condition_id'] == c])
            for s, c in keys}


def guard_counts(rows):
    def errors(stratum, conditions, label, attack=None):
        subset = [r for r in rows if r['stratum'] == stratum and r['condition_id'] in conditions
                  and r['label'] == label and (attack is None or r['attack_id'] == attack)]
        return sum(r['predicted_label'] != label for r in subset), len(subset)

    a, b = 'A_PAIRED', 'B_REAL_ONLY'
    return {'A_clean_FA': errors(a, ('clean',), 0), 'A_clean_miss': errors(a, ('clean',), 1),
            'B_clean_FA': errors(b, ('clean',), 0), 'A05_stress_miss': errors(a, STRESS, 1, 'A05'),
            'A06_stress_miss': errors(a, STRESS, 1, 'A06'),
            'A_stress_FA': errors(a, STRESS, 0), 'A_stress_miss': errors(a, STRESS, 1)}


def identity(rows):
    return [{k: v for k, v in r.items() if k not in ('raw_score', 'predicted_label')} for r in rows]


def select(rows, reference, margins, caps):
    if identity(rows) != identity(reference):
        raise ValueError('Reference score cohort or ordering mismatch')
    for predictions in (rows, reference):
        if len({(r['sample_id'], r['condition_id']) for r in predictions}) != len(predictions):
            raise ValueError('Duplicate selection score')
        if any(not 0 <= r['raw_score'] <= 1 or r['predicted_label'] != int(r['raw_score'] >= THRESHOLD)
               for r in predictions):
            raise ValueError('Invalid selection decision')
    counts = guard_counts(rows)
    result = guarded_selection(counts, guard_counts(reference), margins)
    for name, cap in caps.items():
        error, total = counts[name]
        if error / total > cap:
            result['failed'].append(f'{name}_absolute')
    if result['failed']:
        result.update(status='NO_FEASIBLE_CHECKPOINT', objective=None)
    return result


def compare(standard, optimized, *, max_score, max_rank_metric, near_margin):
    if identity(standard) != identity(optimized) or not standard:
        raise ValueError('Incomplete or mismatched parity evidence')
    a = np.array([r['raw_score'] for r in standard])
    b = np.array([r['raw_score'] for r in optimized])
    if not np.isfinite([a, b]).all():
        raise ValueError('Nonfinite parity score')
    error = np.abs(a - b)
    near = error[(np.abs(a - THRESHOLD) <= near_margin) | (np.abs(b - THRESHOLD) <= near_margin)]
    delta = {}
    first, second = metrics(standard), metrics(optimized)
    for group, m in first.items():
        delta[group] = {k: abs(m[k] - second[group][k]) for k in
                        ('roc_auc', 'pr_auc', 'eer', 'REAL_false_alarm', 'FAKE_miss') if m.get(k) is not None}
    rank_diff = max((v for m in delta.values() for k, v in m.items()
                     if k in ('roc_auc', 'pr_auc', 'eer')), default=0.)
    flips = int(np.count_nonzero((a >= THRESHOLD) != (b >= THRESHOLD)))
    ref, actual = guard_counts(standard), guard_counts(optimized)
    return dict(rows=len(a), max_abs=float(error.max()), mean_abs=float(error.mean()),
                near_threshold_rows=len(near), near_threshold_max_abs=float(near.max()) if len(near) else None,
                decision_flips=flips, metrics_delta=delta, standard_counts=ref, optimized_counts=actual,
                pass_thresholds=bool(error.max() <= max_score and rank_diff <= max_rank_metric
                                     and flips == 0 and ref == actual))
