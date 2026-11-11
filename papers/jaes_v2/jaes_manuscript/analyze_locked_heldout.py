"""Verify completed held-out shards and export the manuscript analysis."""
import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.jaes_v2.empirical_heldout import summarize
from scripts.jaes_v2.empirical_readiness import CONDITIONS, METHODS, OUT, stable_key
from scripts.jaes_v2.final_score_diagnostic import describe, metrics, transitions
from scripts.jaes_v2.heldout_protocol_audit import scoring_lock
from src.lava.jaes_v2.evidence import file_hash, object_hash

MANUSCRIPT = Path(__file__).resolve().parent
ENDPOINTS = {
    'macro_stress_auc': ('Macro stress ROC-AUC', 'macro_stress_auc'),
    'stress_1': ('Stress FAKE miss', 'A_stress_miss'),
    'stress_0': ('Stress REAL false alarm', 'A_stress_FA'),
    'clean_0': ('Clean REAL false alarm', 'A_clean_FA'),
    'clean_1': ('Clean FAKE miss', 'A_clean_miss'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_scores(directory, lock, lock_hash):
    manifest_path = directory/'manifest.json'
    manifest = read(manifest_path)
    if manifest != dict(lock['evaluation_manifest'], lock_sha256=lock_hash):
        raise ValueError('Manifest differs from the authorized lock')
    result = read(directory/'metrics.json')
    model_ids = [f'{method}_{seed}' for seed in lock['seeds'] for method in METHODS]
    expected = {(model, condition) for model in model_ids for condition in CONDITIONS}
    records = manifest['rows']
    expected_names = {stable_key(0, record['sample_id'])+'.json' for record in records}
    shards = directory/'scores'
    if len(expected_names) != 1088 or {p.name for p in shards.glob('*.json')} != expected_names:
        raise ValueError('Unexpected held-out source shards')
    rows, hashes = [], {}
    for record in records:
        sample = record['sample_id']
        path = shards/(stable_key(0, sample)+'.json')
        shard = read(path)
        values = shard['rows']
        if (shard['sample_id'] != sample or shard['lock_sha256'] != lock_hash
                or shard['rows_sha256'] != object_hash(values) or len(values) != len(expected)
                or {(r['model'], r['condition']) for r in values} != expected):
            raise ValueError(f'Invalid shard: {sample}')
        hashes[path.name] = file_hash(path)
        for row in values:
            score = row['raw_score']
            if row['sample_id'] != sample or not 0 <= score <= 1 or row['predicted_label'] != int(score >= .5):
                raise ValueError(f'Invalid prediction: {sample}')
            label = record['label']
            rows.append(dict(row, speaker_id=record['speaker_id'], attack=record['attack'],
                             y=label, p=score, error=int(row['predicted_label'] != label)))
    if (result['status'] != 'COMPLETE' or result['predictions'] != len(rows) or len(rows) != 17408
            or result['manifest_sha256'] != file_hash(manifest_path)
            or result['lock_sha256'] != lock_hash or result['subset'] != 'primary'):
        raise ValueError('Incomplete or incorrectly bound evaluation')
    replay = summarize(rows, model_ids, records)
    if replay != {key: result[key] for key in replay}:
        raise ValueError('Registered metrics fail exact replay')
    return rows, model_ids, result, hashes


def additional_metrics(rows):
    parts = defaultdict(list)
    by_order = defaultdict(dict)
    for row in rows:
        parts[row['model'], row['condition']].append(row)
        if row['condition'] in ('A_then_B', 'B_then_A'):
            by_order[row['model'], row['sample_id']][row['condition']] = row
    curves = {f'{model}/{condition}': metrics(part, curves=True)
              for (model, condition), part in parts.items()}
    speakers, attack_errors, pooled_errors = Counter(), Counter(), Counter()
    for row in rows:
        key = row['model'], row['condition'], row['y']
        speakers[(*key, row['speaker_id'], 'n')] += 1
        speakers[(*key, row['speaker_id'], 'errors')] += row['error']
        attack_errors[(*key, row['attack'], 'n')] += 1
        attack_errors[(*key, row['attack'], 'errors')] += row['error']
        stage = 'clean' if row['condition']=='clean' else 'stress'
        pooled_errors[row['model'], stage, row['y'], 'n'] += 1
        pooled_errors[row['model'], stage, row['y'], 'errors'] += row['error']
    groups = defaultdict(list)
    for (model, _), views in by_order.items():
        first, second = (views[c] for c in ('A_then_B', 'B_then_A'))
        groups[model, 'class', first['y']].append((first, second))
        groups[model, 'attack', first['attack']].append((first, second))
        groups[model, 'speaker', first['speaker_id'], first['y']].append((first, second))
    def order(group):
        first, second = zip(*group, strict=True)
        return dict(sources=len(group), first_errors=sum(r['error'] for r in first),
                    second_errors=sum(r['error'] for r in second),
                    transition=transitions([r['error'] for r in first],[r['error'] for r in second]),
                    score_delta=describe([a['p']-b['p'] for a,b in group]))
    flatten = lambda counter: {'/'.join(map(str, key)): value for key, value in sorted(counter.items())}
    return dict(condition_curves=curves, speaker_counts=flatten(speakers),
                attack_counts=flatten(attack_errors), pooled_counts=flatten(pooled_errors),
                order={'/'.join(map(str,key)):order(group) for key,group in groups.items()})


def comparison(result, analysis, development):
    by_seed = {}
    for seed, paired in result['paired'].items():
        endpoints = {}
        for key, (label, dev_key) in ENDPOINTS.items():
            arms = []
            for method in METHODS:
                model = f'{method}_{seed}'
                if key == 'macro_stress_auc':
                    value = result['models'][model][key]
                    arms.append(dict(value=value))
                else:
                    stage, cls = key.split('_')
                    prefix = f'{model}/{stage}/{cls}'
                    n = analysis['pooled_counts'][f'{prefix}/n']
                    errors = analysis['pooled_counts'][f'{prefix}/errors']
                    arms.append(dict(value=errors/n, errors=errors, n=n))
            delta = arms[1]['value'] - arms[0]['value']
            if key != 'macro_stress_auc' and abs(delta-paired[key]['delta']) > 1e-12:
                raise ValueError('Classwise delta disagrees with registered comparison')
            endpoints[key] = dict(label=label, arms=arms, delta=delta, heldout_interval=paired[key],
                                  development_delta=development['seeds'][seed]['deltas'][dev_key])
        by_seed[seed] = endpoints
    return by_seed


def condition_comparison(result):
    comparisons = {}
    for seed in result['paired']:
        comparisons[seed] = {}
        for condition in CONDITIONS:
            left, right = (result['models'][f'{method}_{seed}']['conditions'][condition]
                           for method in METHODS)
            row = dict(auc=right['auc']-left['auc'])
            for key in ('false_alarm', 'miss'):
                if right[key]['n'] != left[key]['n']:
                    raise ValueError('Condition comparison denominators differ')
                errors = right[key]['errors']-left[key]['errors']
                row[key] = dict(errors=errors, n=left[key]['n'], rate=errors/left[key]['n'])
            comparisons[seed][condition] = row
    return comparisons


def write_tables(result, by_seed):
    def save(name, label, caption, headers, rows):
        line_break = r' \\'
        lines = [r'\begin{table*}[t]', f'\\caption{{{caption}}}\\label{{tab:{label}}}',
                 r'\centering', r'\small',
                 r'\begin{tabular}{@{}l'+'r'*(len(headers)-1)+'@{}}', r'\toprule',
                 ' & '.join(headers)+line_break, r'\colrule']
        lines.extend(' & '.join(row)+line_break for row in rows)
        lines.extend([r'\botrule', r'\end{tabular}', r'\end{table*}'])
        (MANUSCRIPT/'tables'/f'heldout_{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    seeds = tuple(by_seed)
    headers = ('Endpoint', *(f'{seed} {method}' for seed in seeds for method in ('ERM','Channel')))
    rows = []
    for key, (label, _) in ENDPOINTS.items():
        values = [arm for seed in seeds for arm in by_seed[seed][key]['arms']]
        rows.append([label, *(f"{arm['value']:.4f}" if key=='macro_stress_auc'
                              else f"{arm['errors']}/{arm['n']}" for arm in values)])
    save('endpoints','heldout-endpoints',
         'Selected-model results on the locked held-out subset. Stress error counts pool three views per source. '
         'Macro stress ROC-AUC averages the three condition-specific AUCs.', headers, rows)

    rows = []
    for condition, label in zip(CONDITIONS, ('Clean','Noise5','Noise to Opus','Opus to noise'), strict=True):
        metrics_by_model = [result['models'][f'{method}_{seed}']['conditions'][condition]
                            for seed in seeds for method in METHODS]
        rows.append([label, *(f"{item['auc']:.4f} / {item['false_alarm']['errors']} / {item['miss']['errors']}"
                              for item in metrics_by_model)])
    save('conditions','heldout-conditions',
         'Condition-specific held-out performance. Each cell gives ROC-AUC / REAL false alarms / FAKE misses. '
         'There are 256 REAL and 832 FAKE recordings per condition.', ('Condition', *headers[1:]), rows)


def write_metric_csvs(output, result):
    for filename, by_attack in [('condition_metrics.csv', False), ('attack_metrics.csv', True)]:
        with (output/filename).open('w', encoding='utf-8', newline='') as stream:
            fields = ['model','attack','condition','sources','rows','roc_auc','eer','pr_auc',
                      'REAL_false_alarms','REAL_n','FAKE_misses','FAKE_n']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for model, item in result['models'].items():
                groups = item['attacks'] if by_attack else {'ALL': item}
                for attack, group in groups.items():
                    for condition, metric in group['conditions'].items():
                        writer.writerow(dict(model=model, attack=attack, condition=condition,
                            sources=metric['sources'], rows=metric['rows'], roc_auc=metric['auc'],
                            eer=metric['eer'], pr_auc=metric['pr_auc'],
                            REAL_false_alarms=metric['false_alarm']['errors'], REAL_n=metric['false_alarm']['n'],
                            FAKE_misses=metric['miss']['errors'], FAKE_n=metric['miss']['n']))


def write_report(output, result, analysis):
    by_seed = analysis['development_comparison']
    names = ('Clean', 'Noise5', 'Noise10 to Opus24', 'Opus24 to Noise10')
    models = analysis['integrity']['models']
    endpoint_table = [
        '| Seed | Endpoint | ERM | Channel | Held-out delta (pp) | 95% interval (pp) | Valid / invalid | Development A delta (pp) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for seed, items in by_seed.items():
        for key, item in items.items():
            fmt = lambda arm: (f"{arm['value']:.4f}" if key=='macro_stress_auc' else f"{arm['errors']}/{arm['n']}")
            er, ch = map(fmt, item['arms'])
            interval = item['heldout_interval']
            bounds = interval['percentile95']
            ci = f'[{100*bounds[0]:+.2f}, {100*bounds[1]:+.2f}]' if bounds else 'Undefined'
            endpoint_table.append(
                f"| {seed} | {item['label']} | {er} | {ch} | {100*item['delta']:+.2f} | {ci} | "
                f"{interval['valid']:,} / {interval['invalid']:,} | {100*item['development_delta']:+.2f} |")
    text = [
        '# Locked ASVspoof 2019 LA held-out evaluation', '',
        'The authorized primary subset contains 1,088 recordings from 67 speakers. Each of the four models scored all four conditions. '
        'All 17,408 predictions passed source, model, condition, score range, threshold and shard hash checks. '
        'The registered metrics and paired speaker-cluster intervals were recomputed exactly from the committed scores.', '',
        '## Paired endpoints', '',
        *endpoint_table, '',
        'Differences are channel minus ERM. Macro stressed AUC is the mean of the three stress-condition AUCs. '
        'Percentage-point differences multiply AUC or rate contrasts by 100. Each interval conditions on one seed block '
        'and uses 10,000 paired speaker-cluster draws with seed 91109. The stress rows reuse three views per recording. '
        'Development A and held-out use different speakers and attack sets. Development B is REAL-only and has no matched held-out stratum.', '',
        '## Condition metrics', '',
        '| Model | Condition | ROC-AUC | REAL false alarms | FAKE misses |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for model in models:
        for condition, name in zip(CONDITIONS, names, strict=True):
            row = result['models'][model]['conditions'][condition]
            fa, miss = (row[key] for key in ('false_alarm', 'miss'))
            text.append(f"| {model} | {name} | {row['auc']:.4f} | {fa['errors']}/{fa['n']} | {miss['errors']}/{miss['n']} |")
    text.extend(['', '| Seed | Condition | Delta ROC-AUC | Delta REAL false alarms | Delta FAKE misses |',
                 '| --- | --- | ---: | ---: | ---: |'])
    for seed, conditions in analysis['condition_comparison'].items():
        for condition, name in zip(CONDITIONS, names, strict=True):
            row = conditions[condition]
            fa, miss = (row[key] for key in ('false_alarm', 'miss'))
            text.append(f"| {seed} | {name} | {row['auc']:+.4f} | {fa['errors']:+d}/{fa['n']} | {miss['errors']:+d}/{miss['n']} |")
    text.extend(['', '## Attack-specific misses', '',
        'Each cell gives ERM, channel and the channel-minus-ERM difference. Each attack contains 64 FAKE '
        'recordings per condition and 192 stressed views. AUC is undefined within a FAKE-only attack group.', '',
        '| Seed | Attack | Clean | Noise5 | Noise to Opus | Opus to noise | Stress total | Stress delta 95% interval (pp) | Draws valid / invalid |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for seed in by_seed:
        for attack in (f'A{i:02}' for i in range(7,20)):
            cells = []
            counts = [[analysis['attack_counts'][f'{method}_{seed}/{condition}/1/{attack}/errors']
                       for condition in CONDITIONS] for method in METHODS]
            for er, ch in [*zip(*counts, strict=True), (sum(counts[0][1:]), sum(counts[1][1:]))]:
                cells.append(f'{er}, {ch} ({ch-er:+d})')
            interval = result['paired'][seed][attack]
            bounds = interval['percentile95']
            ci = f'[{100*bounds[0]:+.2f}, {100*bounds[1]:+.2f}]' if bounds else 'Undefined'
            text.append(f'| {seed} | {attack} | '+' | '.join(cells)+
                        f" | {ci} | {interval['valid']:,} / {interval['invalid']:,} |")
    text.extend(['', '## Same-source order transitions', '',
        '| Model | Class | Sources | Noise to Opus only wrong | Opus to noise only wrong | Both wrong | Both correct | Median score contrast |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for model in models:
        for label, name in enumerate(('REAL', 'FAKE')):
            item = analysis['order'][f'{model}/class/{label}']
            tr = item['transition']
            text.append(f"| {model} | {name} | {item['sources']} | {tr['only_first_wrong']} | {tr['only_second_wrong']} | "
                        f"{tr['both_wrong']} | {tr['both_correct']} | {item['score_delta']['quantiles'][3]:+.4f} |")
    text.extend(['', 'The contrast is the FAKE score for Noise10 followed by Opus24 minus the reverse order. '
        'The largest absolute speaker contributions to the order error difference are listed below. '
        'The analysis JSON contains the full speaker and attack tables.', '',
        '| Model | Class | Speaker | Sources | Noise to Opus errors | Opus to noise errors |',
        '| --- | --- | --- | ---: | ---: | ---: |'])
    for model in models:
        for label, name in enumerate(('REAL', 'FAKE')):
            candidates = [(key.split('/')[2], item) for key,item in analysis['order'].items()
                          if key.startswith(f'{model}/speaker/') and key.endswith(f'/{label}')]
            for speaker, item in sorted(candidates, key=lambda pair:
                    (-abs(pair[1]['first_errors']-pair[1]['second_errors']), pair[0]))[:3]:
                text.append(f"| {model} | {name} | {speaker} | {item['sources']} | "
                            f"{item['first_errors']} | {item['second_errors']} |")
    text.extend(['',
        '## Interpretation', ''])
    for seed, items in by_seed.items():
        phrases = []
        for item in items.values():
            dev, held = (item[field] for field in ('development_delta', 'delta'))
            phrases.append(f"{item['label']} moved {100*dev:+.2f} pp on development A and {100*held:+.2f} pp on held-out")
        text.extend([f'**Seed {seed}.** '+'. '.join(phrases)+'.', ''])
    auc_gains = [seed for seed, items in by_seed.items() if items['macro_stress_auc']['delta'] > 0]
    fewer_misses = [seed for seed, items in by_seed.items() if items['stress_1']['delta'] < 0]
    more_false_alarms = [seed for seed, items in by_seed.items() if items['stress_0']['delta'] > 0]
    seed_list = lambda values: ', '.join(values) or 'none'
    text.extend([
        f'Seeds with a positive held-out macro stress AUC contrast: {seed_list(auc_gains)}. '
        f'Seeds with fewer held-out stress FAKE misses: {seed_list(fewer_misses)}. '
        f'Seeds with more held-out stress REAL false alarms: {seed_list(more_false_alarms)}.', '',
    ])
    for seed in by_seed:
        differences = {attack: result['paired'][seed][attack]['delta']
                       for attack in (f'A{i:02}' for i in range(7,20))}
        worsened = [attack for attack,value in differences.items() if value > 0]
        improved = [attack for attack,value in differences.items() if value < 0]
        text.extend([f'Seed {seed} stressed attack misses rose on {seed_list(worsened)} '
                     f'and fell on {seed_list(improved)}. Other attack totals were tied.', ''])
    text.extend(['The contrast describes the two fitted seed blocks and this within-release, attack-balanced subset. '
                 'It provides no estimate of initialization variance, cross-corpus generalization, deployment prevalence or forensic cue retention. '
                 'The source, speaker and audio-byte disjointness audit does not establish complete generator lineage.', ''])
    (output/'HELDOUT_SCIENTIFIC_REPORT.md').write_text('\n'.join(text),encoding='utf-8')


def write_figures(result, analysis, by_seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ('Clean', 'Noise5', 'Noise10 to Opus24', 'Opus24 to Noise10')
    colors = ('#17608a', '#a94d11')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42})

    def save(fig, name):
        fig.savefig(MANUSCRIPT/'figures'/f'heldout_{name}.pdf', metadata={'CreationDate':None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True, sharex=True, layout='constrained')
    for ax, (seed, entries) in zip(axes, by_seed.items(), strict=True):
        for j, item in enumerate(entries.values()):
            interval = item['heldout_interval']['percentile95']
            if interval is not None:
                ax.hlines(j+.12, 100*interval[0], 100*interval[1], color=colors[1])
            ax.plot(100*item['delta'], j+.12, 's', color=colors[1], label='Held-out' if j==0 else None)
            ax.plot(100*item['development_delta'], j-.12, 'o', color=colors[0],
                    label='Development A' if j==0 else None)
        ax.axvline(0, color='.5', lw=.7, ls='--')
        ax.set(title=f'Seed {seed}', xlabel='Channel minus ERM (pp)')
    axes[0].set_yticks(range(len(ENDPOINTS)), [label for label, _ in ENDPOINTS.values()])
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, 'contrasts')

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.3), sharex=True, sharey=True, layout='constrained')
    for ax, condition, label in zip(axes.flat, CONDITIONS, labels, strict=True):
        for i, (key, record) in enumerate((key, record) for key, record in
                                           analysis['condition_curves'].items() if key.endswith('/'+condition)):
            model = key.split('/')[0]
            method, seed = model.split('_')
            ax.plot(record['roc']['fpr'], record['roc']['tpr'], ('-', '--', '-.', ':')[i],
                    color=colors[METHODS.index(method)], label=f'{seed} {method.upper()} ({record["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], color='.7', lw=.7)
        ax.set(title=label, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.supxlabel('False-positive rate')
    fig.supylabel('True-positive rate')
    save(fig, 'roc')

    attacks = [f'A{i:02}' for i in range(7,20)]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), sharey=True, layout='constrained')
    for ax, seed in zip(axes, by_seed, strict=True):
        deltas = np.array([[analysis['attack_counts'][f'channel_{seed}/{condition}/1/{attack}/errors']
                            - analysis['attack_counts'][f'erm_{seed}/{condition}/1/{attack}/errors']
                            for condition in CONDITIONS] for attack in attacks])
        chart = ax.imshow(deltas, cmap='BrBG_r', vmin=-64, vmax=64, aspect='auto')
        for r, counts in enumerate(deltas):
            for c, value in enumerate(counts):
                ax.text(c, r, f'{value:+d}', ha='center', va='center',
                        color='white' if abs(value)>46 else '#151515', fontsize=8)
        ax.set_title(f'Seed {seed}')
        ax.set_yticks(range(len(attacks)), attacks)
        ax.set_xticks(range(4), ['Clean', 'Noise5', 'Noise\nthen Opus', 'Opus\nthen noise'])
    fig.colorbar(chart, ax=axes, label='Difference in misses (of 64)', shrink=.9)
    save(fig, 'attacks')

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), sharey=True, sharex=True, layout='constrained')
    for ax, seed in zip(axes, by_seed, strict=True):
        for j, method in enumerate(METHODS):
            for label in (0, 1):
                q = result['models'][f'{method}_{seed}']['order'][str(label)]['score_delta']['quantiles']
                y = label+(j-.5)*.27
                ax.hlines(y, q[1], q[5], color=colors[j], lw=1)
                ax.hlines(y, q[2], q[4], color=colors[j], lw=3)
                ax.plot(q[3], y, ('o', 's')[j], color=colors[j], label=method.upper() if label==0 else None)
        ax.axvline(0, color='.5', lw=.7)
        ax.set(title=f'Seed {seed}', xlabel='Noise-to-Opus minus Opus-to-noise score')
    axes[0].set_yticks([0,1], ['REAL','FAKE'])
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, 'order')


def run():
    base = ROOT/OUT
    proposed = base/'heldout_proposed_9107_9207_v1.json'
    approval = base/'heldout_scoring_author_approval.json'
    lock = scoring_lock(proposed, approval)
    lock_hash = object_hash(lock)
    directory = base/f'heldout_{lock_hash[:16]}'
    rows, models, result, hashes = verified_scores(directory, lock, lock_hash)
    output = directory/'analysis'
    output.mkdir(exist_ok=True)
    path = output/'scores.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        fields = ['model','method','seed','sample_id','condition','speaker_id','attack','y',
                  'threshold','raw_score','predicted_label','error']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            method, seed = row['model'].split('_')
            writer.writerow(dict(row, method=method, seed=seed, threshold=.5))
    analysis = additional_metrics(rows)
    development_path = base/'post9207_consolidated_v1.json'
    development = read(development_path)
    analysis['development_comparison'] = comparison(result, analysis, development)
    analysis['condition_comparison'] = condition_comparison(result)
    analysis['integrity'] = dict(lock_sha256=lock_hash, proposed_lock_sha256=file_hash(proposed),
        approval_sha256=file_hash(approval), development_sha256=file_hash(development_path),
        metrics_sha256=file_hash(directory/'metrics.json'),
        manifest_sha256=file_hash(directory/'manifest.json'), scores_sha256=file_hash(path),
        sources=len(hashes), predictions=len(rows), models=models, score_shards=hashes,
        exact_metric_replay=True)
    (output/'verified_analysis.json').write_text(json.dumps(analysis, indent=2)+'\n',encoding='utf-8')
    write_report(output, result, analysis)
    write_figures(result, analysis, analysis['development_comparison'])
    write_tables(result, analysis['development_comparison'])
    write_metric_csvs(output, result)
    print(output)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
