"""Paired development comparison of the sealed seed-9207 size arms."""
from statistics import mean

import numpy as np

from scripts.jaes_v2 import size1280_profile as profile
from scripts.jaes_v2.final_score_diagnostic import align, bootstrap_auc, expected_rows, interval, transitions
from src.lava.jaes_v2 import phase8_training as prior, phase10_training as engine
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv
from src.lava.jaes_v2.phase10_validation import STRESS, guard_counts, metrics, select

A, B = 'A_PAIRED', 'B_REAL_ONLY'
KEYS = ('erm640', 'channel640', 'erm1280', 'channel1280')


def contrast(values):
    a, b, c, d = values
    return {'delta640': b - a, 'delta1280': d - c, 'delta_size': (d - c) - (b - a)}


def report(values, boot):
    sampled = contrast(boot)
    return dict(per_model=dict(zip(KEYS, values, strict=True)), **{
        key: dict(estimate=value, **interval(sampled[key]))
        for key, value in contrast(values).items()
    })


def speaker_draws(ids, spec):
    rng = np.random.default_rng(spec['seed'])
    result = {}
    for stratum in (A, B):
        speakers = sorted({r['speaker_id'] for r in ids.values() if r['stratum'] == stratum})
        n = len(speakers)
        result[stratum] = speakers, rng.multinomial(n, np.full(n, 1 / n), size=spec['draws'])
    return result


def average_by_speaker(rows, value, speakers, weights):
    totals = np.array([sum(r['speaker_id'] == s for r in rows) for s in speakers])
    sums = np.array([sum(value(r) for r in rows if r['speaker_id'] == s) for s in speakers])
    denominator = weights @ totals
    sampled = np.divide(weights @ sums, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
    return float(sums.sum() / totals.sum()), sampled


def endpoints(grids, typed_rows, ids, bootstrap):
    draws = speaker_draws(ids, bootstrap)
    group_metrics = [metrics(rows) for rows in typed_rows]
    main = {}
    for name, conditions in (('A_clean_AUC', ('clean',)), ('A_macro_stress_AUC', STRESS)):
        samples = [np.mean([bootstrap_auc([r for (s, cond, stratum), r in grid.items()
                                           if cond == condition and stratum == A], *draws[A])
                            for condition in conditions], axis=0) for grid in grids]
        values = [mean(g[f'{A}/{condition}']['roc_auc'] for condition in conditions) for g in group_metrics]
        main[name] = dict(report(values, samples), sources=len([r for r in ids.values() if r['stratum'] == A]),
                          conditions=list(conditions))
    rates = (
        ('A_clean_FA', A, ('clean',), 0, None), ('A_clean_miss', A, ('clean',), 1, None),
        ('A_stress_FA', A, STRESS, 0, None), ('A_stress_miss', A, STRESS, 1, None),
        ('B_clean_FA', B, ('clean',), 0, None),
        ('A05_stress_miss', A, STRESS, 1, 'A05'), ('A06_stress_miss', A, STRESS, 1, 'A06'),
    )
    errors = [guard_counts(rows) for rows in typed_rows]
    for name, stratum, conditions, label, attack in rates:
        subsets = [[r for (_, cond, group), r in grid.items() if group == stratum and cond in conditions
                    and r['y'] == label and (attack is None or r['attack_id'] == attack)] for grid in grids]
        pairs = [average_by_speaker(rows, lambda r: r['error'], *draws[stratum]) for rows in subsets]
        point, boot = zip(*pairs, strict=True)
        main[name] = dict(report(point, boot), counts={k: dict(zip(('errors', 'views'), e[name], strict=True))
                                                       for k, e in zip(KEYS, errors, strict=True)},
                          sources=len({r['sample_id'] for r in subsets[0]}))
    order = {}
    for stratum, label in ((A, 0), (A, 1), (B, 0)):
        cohort = sorted(s for s, r in ids.items() if r['stratum'] == stratum and int(r['label']) == label)
        pairs = [[(grid[s, 'A_then_B', stratum], grid[s, 'B_then_A', stratum]) for s in cohort] for grid in grids]
        subgroup = {}
        for field in ('p', 'error'):
            difference = [[dict(left, difference=left[field] - right[field]) for left, right in arm] for arm in pairs]
            points, boots = zip(*(average_by_speaker(rows, lambda r: r['difference'], *draws[stratum])
                                  for rows in difference), strict=True)
            subgroup[field] = report(points, boots)
        subgroup['transitions'] = {k: transitions([l['error'] for l, _ in arm], [r['error'] for _, r in arm])
                                   for k, arm in zip(KEYS, pairs, strict=True)}
        order[f'{stratum}/{label}'] = dict(subgroup, sources=len(cohort), direction='A_then_B minus B_then_A')
    return main, order, dict(zip(KEYS, group_metrics, strict=True))


def analyze(root, spec, data, fits):
    _, dev, b_ids, _ = data
    ids = expected_rows(dev, b_ids)
    raw, resources, checkpoints, seals = [], {}, {}, {}
    for method, (new_dir, new_result, _) in zip(profile.METHODS, fits, strict=True):
        base, base_contract = profile.reference(root, method)
        prior.verify_tree(base)
        baseline = engine.read(base / 'result.json')
        _, final_state = engine.committed_controller(base, base_contract)
        if baseline['state'] != final_state or final_state['step'] != 840 or final_state['epoch'] != 21:
            raise ValueError('Historical fit lifecycle mismatch')
        selected = engine.strict_controller(base / 'checkpoints' / final_state['best']['tag'], base_contract)
        if baseline['selected_weights'] != selected['model_hash']:
            raise ValueError('Historical selected weight mismatch')
        for size, path, result in ((640, base, baseline), (1280, new_dir, new_result)):
            key = f'{method}{size}'
            scores = read_csv(path / 'scores.csv')
            typed = [dict(r, label=int(r['label']), threshold=float(r['threshold']),
                          raw_score=float(r['raw_score']), predicted_label=int(r['predicted_label'])) for r in scores]
            if object_hash(typed) != result['scores_hash'] or metrics(typed) != result['metrics']:
                raise ValueError('Historical or new score replay mismatch')
            raw.append((key, (align(scores, ids, prior.DEV), typed)))
            checkpoints[key] = result['state']['best']
            seals[key] = file_hash(path / 'seal.json')
            log_paths = (path.parent.glob('scientific_*_resources.json') if size == 640
                         else (path.parent / 'resources').glob('scientific_*.json'))
            resources[key] = []
            for log in sorted(log_paths):
                entry = engine.read(log)
                samples = entry.get('samples', []) + [entry['final_resources']]
                resources[key].append(dict(log=log.name, timing=entry['timing'], updates=entry.get('updates'),
                    peak_rss=max(s['process_peak_rss'] for s in samples),
                    minimum_commit=min(s['commit_available'] for s in samples),
                    minimum_disk=min(s['d_free'] for s in samples)))
    raw = dict(raw)
    grids, rows = zip(*(raw[k] for k in KEYS), strict=True)
    clean_and_stress, order, per_model = endpoints(grids, rows, ids, spec['bootstrap'])
    guards = {str(size): select(rows[i + 1], rows[i], engine.MARGINS, engine.CAPS)
              for size, i in ((640, 0), (1280, 2))}
    return dict(status='SIZE_SENSITIVITY_VERIFIED', seed=spec['seed'], threshold=spec['threshold'],
                estimand=spec['estimand'], bootstrap=spec['bootstrap'], development_sources=len(ids),
                score_rows_per_model=len(rows[0]), release_registry_sha256=file_hash(root / profile.REGISTRY),
                endpoints=clean_and_stress, order=order, per_model=per_model, guard=guards,
                selected_checkpoints=checkpoints, resource_invocations=resources, seal_sha256=seals,
                official_evaluation_used=False, calibration_used=False,
                limitation='One initialization, shared development and increased compute at fixed epochs.')


def markdown_report(result):
    def interval_cell(value):
        bounds = value['percentile95']
        limits = 'unavailable' if bounds is None else f'[{bounds[0]:+.4f}, {bounds[1]:+.4f}]'
        return f"{value['estimate']:+.4f} {limits}"

    lines = [
        '# Training-size sensitivity at fixed epochs', '',
        'Seed 9207. Each arm uses 21 epochs. The 640 and 1,280 cohorts use 840 and 1,680 updates, respectively.',
        'The contrast measures sensitivity to cohort expansion and its associated compute on the shared development set.',
        'The pointwise 95% intervals use matched speaker resampling within strata. They describe evaluation-cohort uncertainty for one fitted seed block.', '',
        '| Endpoint | ERM 640 | Channel 640 | ERM 1,280 | Channel 1,280 | Δ640 | Δ1,280 | Δsize |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, endpoint in result['endpoints'].items():
        counts = endpoint.get('counts')
        values = ([f"{counts[k]['errors']}/{counts[k]['views']}" for k in KEYS] if counts
                  else [f"{endpoint['per_model'][k]:.4f}" for k in KEYS])
        fields = [name, *values, *(interval_cell(endpoint[k]) for k in ('delta640', 'delta1280', 'delta_size'))]
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', '## Transformation order', '',
              'Each contrast uses Noise → Opus minus Opus → Noise within source.', '',
              '| Stratum and class | Sources | Metric | Δ640 | Δ1,280 | Δsize |',
              '| --- | ---: | --- | ---: | ---: | ---: |']
    for group, endpoint in result['order'].items():
        for key, label in (('p', 'FAKE score'), ('error', 'Error rate')):
            fields = [group, str(endpoint['sources']), label,
                      *(interval_cell(endpoint[key][k]) for k in ('delta640', 'delta1280', 'delta_size'))]
            lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', '## Selected checkpoints and guard decisions', '']
    for key, checkpoint in result['selected_checkpoints'].items():
        lines.append(f"- {key} selected `{checkpoint['tag']}`.")
    for size, guard in result['guard'].items():
        failed = ', '.join(guard['failed']) or 'none'
        lines.append(f"- Size {size} guard status `{guard['status']}`. Failed constraints are {failed}.")
    lines += ['', '## Execution costs', '',
              '| Arm | Training seconds | Validation seconds | Checkpoint seconds | Peak RSS MiB |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key, invocations in result['resource_invocations'].items():
        seconds = [sum(i['timing'].get(field, 0) for i in invocations)
                   for field in ('train_seconds', 'validation_seconds', 'checkpoint_seconds')]
        peak = max((i['peak_rss'] / 1024**2 for i in invocations), default=None)
        values = [f'{t:.1f}' for t in (*seconds, peak)] if invocations else ['unavailable'] * 4
        lines.append('| ' + ' | '.join([key, *values]) + ' |')
    lines += ['', 'The fixed threshold is 0.5. This development comparison reuses the same evaluation population and uses one selected initialization. Its effect includes the additional updates at fixed epochs. It does not estimate a fixed-compute causal effect or a population-average size effect.',
              '', '[Score replay, exact denominators, interval draw counts and artifact hashes](../../outputs/jaes_v2/phase10/size1280_v1/size_sensitivity.json)', '']
    return '\n'.join(lines)
