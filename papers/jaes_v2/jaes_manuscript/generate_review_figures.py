"""Regenerate the supplemental manuscript figures from sealed development runs."""
from collections import Counter

from generate_evidence import (
    ATTACKS, ENDPOINTS, EVIDENCE, HERE, METHODS, ROOT, csv, hashlib, json, np, plt,
)


COLORS = ('#17608a', '#a94d11')
LABELS = ('ERM', 'Channel')
CONDITIONS = {'clean': 'Clean', 'noise5': 'Noise5', 'A_then_B': 'Noise to Opus',
              'B_then_A': 'Opus to Noise'}
ERROR_LABELS = ('A clean REAL', 'A clean FAKE', 'A stress REAL', 'A stress FAKE', 'B clean REAL')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(evidence):
    grids, histories, sources = {}, {}, {}
    identity = None
    for seed, block in evidence['seeds'].items():
        for method in METHODS:
            endpoint = block['endpoints'][method]
            experiment = endpoint['experiment']
            directory = ROOT/'outputs/jaes_v2/phase10'/experiment/'scientific'
            seal = json.loads((directory/'seal.json').read_text(encoding='utf-8'))['output_hashes']
            hashes = {name: digest(directory/name) for name in ('scores.csv', 'result.json')}
            if any(seal[name] != value for name, value in hashes.items()):
                raise ValueError(f'Seal mismatch for {experiment}')
            if hashes['scores.csv'] != endpoint['input_sha256']['scores.csv']:
                raise ValueError(f'Score provenance mismatch for {experiment}')
            sources[experiment] = hashes
            with (directory/'scores.csv').open(encoding='utf-8', newline='') as stream:
                records = list(csv.DictReader(stream))
            grid, metadata = {}, {}
            for row in records:
                key = row['sample_id'], row['condition_id']
                label, score = int(row['label']), float(row['raw_score'])
                if (key in grid or row['status'] != 'OK' or float(row['threshold']) != .5
                        or not 0 <= score <= 1 or label not in (0, 1)
                        or int(row['predicted_label']) != int(score >= .5)
                        or any(not row[field] for field in ('sample_id', 'parent_recording_id', 'speaker_id'))):
                    raise ValueError(f'Invalid score row for {experiment}: {key}')
                grid[key] = score
                metadata[key] = (label, row['stratum'], row['parent_recording_id'],
                                 row['speaker_id'], row['attack_id'])
            support = Counter((condition, info[1], info[0]) for (_, condition), info in metadata.items())
            expected = {(condition, stratum, label): n
                        for condition in (*CONDITIONS, 'gain_half', 'gain_quarter', 'opus64')
                        for stratum, label, n in [('A_PAIRED', 0, 112), ('A_PAIRED', 1, 112), ('B_REAL_ONLY', 0, 144)]}
            if support != expected or len({source for source, _ in grid}) != 368:
                raise ValueError(f'Invalid development support for {experiment}')
            if any(metadata[source, 'clean'] != info for (source, _), info in metadata.items()):
                raise ValueError(f'Identity changed across views for {experiment}')
            if identity is not None and metadata != identity:
                raise ValueError(f'Nonmatched development rows for {experiment}')
            identity = metadata
            grids[seed, method] = grid
            result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
            state = result['state']
            history = state['history']
            best = state['best']
            bce = np.array([row['clean']['nll'] for row in history])
            selected = int(best['tag'].split('_')[1][1:])
            if (state['contract']['experiment'] != experiment or state['step'] != 840
                    or state['epoch'] != 21 or best['tag'] != endpoint['selected_checkpoint']
                    or [(row['epoch'], row['step']) for row in history] != [(e, e*40) for e in range(1, 22)]
                    or any(row['clean']['n'] != 224 for row in history)
                    or not np.isfinite(bce).all() or bce[selected-1] != min(bce)):
                raise ValueError(f'Invalid epoch history for {experiment}')
            histories[seed, method] = bce, selected
    return grids, identity, histories, sources


def roc(labels, scores):
    order = np.argsort(-scores, kind='stable')
    y, p = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p)-1]
    fpr = np.r_[0, np.cumsum(1-y)[ends] / np.sum(1-y)]
    tpr = np.r_[0, np.cumsum(y)[ends] / np.sum(y)]
    auc = np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)
    return fpr, tpr, auc


def save(fig, name):
    fig.savefig(HERE/'figures'/f'{name}.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def plot_errors(seeds, rates):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True, layout='constrained')
    for ax, seed in zip(axes, seeds):
        for j, method in enumerate(METHODS):
            values = [100*rates[seed, method, key] for key in ENDPOINTS]
            ax.barh(np.arange(5)+(j-.5)*.36, values, height=.34, color=COLORS[j],
                    hatch=('', '//')[j], edgecolor='white', label=LABELS[j])
        ax.set(title=f'Seed {seed}', xlabel='Error rate (%)', xlim=(0, 100))
    axes[0].set_yticks(range(5), ERROR_LABELS)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False)
    save(fig, 'classwise_errors')


def plot_uncertainty(seeds):
    keys = ('macro_stress_auc', 'A_stress_FA', 'A_stress_miss', 'B_clean_FA')
    fig, ax = plt.subplots(figsize=(7, 2.8), layout='constrained')
    for j, (seed, block) in enumerate(seeds.items()):
        for i, key in enumerate(keys):
            interval = (block['macro_stress_auc_interval'] if key == 'macro_stress_auc'
                        else block['paired'][key]['bootstrap'])
            lo, hi = 100*np.array(interval['percentile95'])
            y = i+(j-.5)*.25
            ax.hlines(y, lo, hi, color=COLORS[j])
            ax.plot(100*block['deltas'][key], y, ('o', 's')[j], color=COLORS[j],
                    label=seed if i == 0 else None)
    ax.axvline(0, color='.4', linestyle='--', linewidth=.8)
    ax.set_yticks(range(4), ['A macro stress AUC', 'A stressed REAL', 'A stressed FAKE', 'B clean REAL'])
    ax.set_xlabel('Channel minus ERM (percentage points)')
    ax.invert_yaxis()
    ax.legend(title='Seed', frameon=False)
    save(fig, 'paired_uncertainty')


def plot_roc(seeds, grids, metadata):
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True, sharey=True, layout='constrained')
    aucs = {}
    for (condition, title), ax in zip(CONDITIONS.items(), axes.flat):
        keys = sorted(key for key, info in metadata.items() if key[1] == condition and info[1] == 'A_PAIRED')
        labels = np.array([metadata[key][0] for key in keys])
        for seed_i, seed in enumerate(seeds):
            for j, method in enumerate(METHODS):
                scores = np.array([grids[seed, method][key] for key in keys])
                x, y, auc = roc(labels, scores)
                aucs[seed, method, condition] = auc
                ax.plot(x, y, ('-', '--', '-.', ':')[2*seed_i+j], color=COLORS[j],
                        label=f'{seed} {LABELS[j]} ({auc:.4f})')
        ax.plot([0, 1], [0, 1], color='.7', linewidth=.7)
        ax.set(title=title, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc='lower right', fontsize=8, frameon=False)
    fig.supxlabel('False-positive rate')
    fig.supylabel('True-positive rate')
    for seed, block in seeds.items():
        for method in METHODS:
            endpoint = block['endpoints'][method]
            assert np.isclose(aucs[seed, method, 'clean'], endpoint['clean']['auc'], atol=1e-12, rtol=0)
            macro = np.mean([aucs[seed, method, condition] for condition in list(CONDITIONS)[1:]])
            assert np.isclose(macro, endpoint['macro_stress_auc'], atol=1e-12, rtol=0)
    save(fig, 'development_roc')
    return {'/'.join(key): value for key, value in aucs.items()}


def plot_history(seeds, histories):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.75), sharey=True, layout='constrained')
    for ax, seed in zip(axes, seeds):
        for j, method in enumerate(METHODS):
            bce, epoch = histories[seed, method]
            ax.plot(range(1, 22), bce, ('-', '--')[j], color=COLORS[j], label=LABELS[j])
            ax.plot(epoch, bce[epoch-1], ('o', 's')[j], color=COLORS[j])
        ax.set(title=f'Seed {seed}', xlabel='Completed epoch', xticks=[1, 5, 10, 15, 21])
    axes[0].set_ylabel('Clean development BCE')
    axes[1].legend(frameon=False)
    save(fig, 'checkpoint_dynamics')


def plot_transitions(seeds, grids, metadata):
    fig, axes = plt.subplots(2, 2, figsize=(7, 4), sharex=True, sharey=True, layout='constrained')
    transitions = {}
    for i, seed in enumerate(seeds):
        for label in (0, 1):
            matrix = []
            for condition in CONDITIONS:
                keys = sorted(key for key, info in metadata.items()
                              if key[1] == condition and info[:2] == (label, 'A_PAIRED'))
                values = Counter(tuple(int(grids[seed, m][key] >= .5) == label for m in METHODS)
                                 for key in keys)
                matrix.append([values[pair] for pair in [(True, True), (True, False), (False, True), (False, False)]])
            transitions[f'{seed}/{label}'] = matrix
            ax = axes[i, label]
            heat = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=112, aspect='auto')
            for row, counts in enumerate(matrix):
                for col, count in enumerate(counts):
                    ax.text(col, row, count, ha='center', va='center', color='white' if count > 64 else '#172537')
            ax.set_title(f"Seed {seed} / {('REAL', 'FAKE')[label]}")
            ax.set_yticks(range(4), CONDITIONS.values())
            ax.set_xticks(range(4), ['Both\ncorrect', 'ERM\nonly', 'Channel\nonly', 'Both\nwrong'])
    fig.colorbar(heat, ax=axes, label='Recordings', ticks=[0, 28, 56, 84, 112], shrink=.85)
    save(fig, 'source_transitions')
    return transitions


def run():
    evidence = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    seeds = evidence['seeds']
    grids, metadata, histories, sources = load_runs(evidence)
    rates = {}
    for (seed, method), grid in grids.items():
        for key in (*ENDPOINTS, *ATTACKS):
            group, condition, error = key.split('_')
            label = int(error == 'miss')
            stratum = 'B_REAL_ONLY' if group == 'B' else 'A_PAIRED'
            conditions = ('clean',) if condition == 'clean' else tuple(CONDITIONS)[1:]
            values = [score for row, score in grid.items() if row[1] in conditions
                      and metadata[row][:2] == (label, stratum)
                      and (group not in ('A01','A02','A03','A04','A05','A06') or metadata[row][4] == group)]
            errors = sum(int(score >= .5) != label for score in values)
            reference = seeds[seed]['paired'][key][method]['miss' if label else 'false_alarm']
            if (errors, len(values)) != (reference['errors'], reference['n']):
                raise ValueError(f'Error count mismatch for {seed}/{method}/{key}')
            rates[seed, method, key] = errors/len(values)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.labelsize': 10})
    plot_errors(seeds, rates)
    plot_uncertainty(seeds)
    aucs = plot_roc(seeds, grids, metadata)
    plot_history(seeds, histories)
    transitions = plot_transitions(seeds, grids, metadata)
    (HERE/'review_figure_provenance.json').write_text(json.dumps({
        'consolidated_evidence': {'path': EVIDENCE.relative_to(ROOT).as_posix(), 'sha256': digest(EVIDENCE)},
        'sealed_scientific_inputs': sources,
        'inputs_by_figure': {
            'aggregate': 'consolidated evidence, selected endpoint AUC and paired stress miss counts',
            'attack_changes': 'consolidated evidence, attack deltas and speaker-cluster percentiles',
            'order_scores': 'consolidated evidence, within-source directional score quantiles',
            'speaker_counts': 'consolidated evidence, speaker-specific clean B false alarms',
            'classwise_errors': 'scores.csv, checked against paired error counts in consolidated evidence',
            'paired_uncertainty': 'consolidated evidence, registered speaker-cluster percentiles',
            'development_roc': 'scores.csv, clean and macro AUC checked against consolidated evidence',
            'checkpoint_dynamics': 'result.json, 21 recorded clean development epochs',
            'source_transitions': 'scores.csv, aligned on source ID and condition after metadata checks',
        },
        'condition_auc': aucs, 'source_transition_counts': transitions,
        'transition_columns': ['both_correct', 'erm_only_correct', 'channel_only_correct', 'both_wrong'],
        'transition_rows': list(CONDITIONS),
    }, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    run()
