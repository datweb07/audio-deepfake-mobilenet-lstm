"""Read-only audit of the two sealed Phase 8 medium runs."""
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from src.lava.jaes_v2.evidence import file_hash, object_hash
from src.lava.jaes_v2.phase7_training import score_metrics
from src.lava.jaes_v2 import phase8_training
from src.lava.jaes_v2.phase8_training import code_contract, dataset, load_config, verify_tree

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/jaes_v2/phase8/training'
STRESS = ('noise5', 'A_then_B', 'B_then_A')
CONDITIONS = ('clean', 'gain_half', 'gain_quarter', 'noise5', 'opus64', 'A_then_B', 'B_then_A')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_scores(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(ok, description):
    if not ok:
        raise ValueError(description)


def weight_hashes(path):
    tensors = {}
    with zipfile.ZipFile(path) as archive, h5py.File(io.BytesIO(archive.read('model.weights.h5'))) as weights:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                tensors[name.replace('\\', '/')] = hashlib.sha256(obj[()].tobytes()).hexdigest()
        weights.visititems(collect)
    return tensors


def metrics(rows):
    result = score_metrics(rows)
    tp = result['n_FAKE'] - result['fn']
    tn = result['n_REAL'] - result['fp']
    den = 2 * tp + result['fp'] + result['fn']
    return {**result, 'accuracy': (tp + tn) / len(rows),
            'f1': 2 * tp / den if result['n_REAL'] and result['n_FAKE'] else None}


def audit(method, current_code):
    run = BASE / f'p8_medium_{method}_partial_s6107_v1'
    status = read_json(run / 'status.json')
    require(status['status'] == 'DEVELOPMENT_EVALUATED', f'Incomplete run: {run}')
    seal = verify_tree(run)
    config = load_config(ROOT, run / 'resolved_config.json')
    require(config == load_config(ROOT, ROOT / f'configs/jaes_v2/phase8/{run.name}.json'), 'Config drift')
    train, dev, real_only, partitions = dataset(ROOT, config)
    data_audit = read_json(run / 'dataset_audit.json')
    require(partitions == data_audit['partitions'], 'Partition audit differs')
    require([r.sample_id for r in train] == data_audit['source_ids'], 'Training selection drift')
    require([r.sample_id for r in dev] == data_audit['development_ids'], 'Development selection drift')
    require(not any(item[k] for item in (status, data_audit) for k in ('calibration_accessed', 'evaluation_accessed')),
            'Reserved role accessed')
    contract = read_json(run / 'contract.json')
    require(contract['config_hash'] == object_hash(config.__dict__), 'Config digest mismatch')
    require(contract['actual_sources'] == object_hash([r.to_dict() for r in train + dev]), 'Source fingerprint mismatch')
    require(contract['manifest'] == config.manifest_sha256 and contract['development'] == config.development_sha256,
            'Manifest digest mismatch')
    require({k: v for k, v in contract['code'].items() if not k.endswith('/phase8_training.py')} ==
            {k: v for k, v in current_code.items() if not k.endswith('/phase8_training.py')}, 'Shared source drift')
    chain = object_hash([])
    chains = {0: chain}
    orders = []
    losses = {}
    view_counts = Counter()
    gradient_paths = {}
    for step in range(1, status['steps'] + 1):
        entry = read_json(run / 'updates' / f'{step:08}.json')
        require(entry['step'] == step, 'Journal step mismatch')
        chain = object_hash([chain, entry])
        chains[step] = chain
        orders.append(entry['sample_ids'])
        losses.setdefault(entry['epoch'] + 1, []).append(entry['loss'])
        view_counts.update(v[1] for v in entry['views'])
        gradient_paths.setdefault(entry.get('gradient_path', 'unlabeled'), step)
    checkpoints = {}
    trainer_path = 'src/lava/jaes_v2/phase8_training.py'
    allowed_trainers = {current_code[trainer_path]} | {
        value for name, value in vars(phase8_training).items()
        if name.startswith('PRE_') and name.endswith('_SHA256')
    }
    for checkpoint in sorted((run / 'checkpoints').iterdir()):
        if not (checkpoint / 'seal.json').exists():
            continue
        verify_tree(checkpoint)
        state = read_json(checkpoint / 'controller.json')
        require(state['update_chain'] == chains[state['step']], 'Journal chain mismatch')
        require({k: v for k, v in state['contract'].items() if k != 'code'} ==
                {k: v for k, v in contract.items() if k != 'code'}, 'Checkpoint contract mismatch')
        require({k: v for k, v in state['contract']['code'].items() if not k.endswith('/phase8_training.py')} ==
                {k: v for k, v in contract['code'].items() if not k.endswith('/phase8_training.py')}, 'Checkpoint source drift')
        trainer = state['contract']['code'][trainer_path]
        require(trainer in allowed_trainers, 'Unknown trainer fingerprint')
        checkpoints[checkpoint.name] = (state, trainer)
    pointer = read_json(run / 'latest.json')
    state, _ = checkpoints[pointer['checkpoint']]
    require(pointer['controller_sha256'] == file_hash(run/'checkpoints'/pointer['checkpoint']/'controller.json'),
            'Latest controller hash mismatch')
    history = read_json(run / 'history.json')
    require(state['finished'] and state['step'] == status['steps'] and state['history'] == history, 'Final state mismatch')
    best = read_json(run / 'best.json')
    require(best == status['best'] and best['sha256'] == file_hash(run / best['path']), 'Best model digest mismatch')
    eligible = [row for row in history if row['eligibility'] == 'ELIGIBLE']
    selected = min(eligible, key=lambda row: row['value'])
    require((best['epoch'], best['selection_value']) == (selected['epoch'], selected['value']), 'Selection mismatch')
    rows = read_scores(run / 'scores.csv')
    keys = {(r['sample_id'], r['condition_id']) for r in rows}
    require(len(keys) == len(rows) and keys == {(r.sample_id, c) for r in dev for c in CONDITIONS}, 'Score coverage mismatch')
    by_id = {r.sample_id: r for r in dev}
    for row in rows:
        source = by_id[row['sample_id']]
        require((int(row['label']), row['speaker_id'], row['attack_id']) ==
                (source.label, source.speaker_id, source.attack_id), 'Score metadata mismatch')
        require(row['stratum'] == ('B_REAL_ONLY' if source.sample_id in real_only else 'A_PAIRED'), 'Stratum mismatch')
        require(float(row['threshold']) == .5 and int(row['predicted_label']) == int(float(row['raw_score']) >= .5),
                'Decision mismatch')
    stored_metrics = read_json(run / 'metrics.json')
    attacks = read_json(run / 'attack_metrics.json')
    require(len(stored_metrics) == 14 and len(attacks) == 42, 'Metric coverage mismatch')
    for item in stored_metrics + attacks:
        group = [row for row in rows if row['condition_id'] == item['condition'] and
                 (row['attack_id'] == item['attack'] if 'attack' in item else row['stratum'] == item['stratum'])]
        rebuilt = score_metrics(group)
        for key in ('n', 'n_REAL', 'n_FAKE', 'fp', 'fn', 'REAL_false_alarm', 'FAKE_miss', 'roc_auc', 'pr_auc', 'eer', 'nll'):
            a, b = item[key], rebuilt[key]
            require(a == b if a is None or b is None else abs(a - b) < 1e-12, f'Metric mismatch {key}')
    warm = weight_hashes(run / 'best_epoch_001.keras')
    chosen = weight_hashes(run / best['path'])
    require(warm.keys() == chosen.keys(), 'Export tensor layout differs')
    backbone = [k for k in warm if '/time_distributed/layer/' in k]
    changed = [k for k in backbone if warm[k] != chosen[k]]
    bn = [k for k in backbone if '/batch_normalization' in k]
    require(bool(backbone) and bool(bn), 'Backbone weights missing')
    return {
        'run': run.name, 'seal_entries': len(seal['output_hashes']), 'seal_sha256': file_hash(run/'seal.json'),
        'checkpoint_count': len(checkpoints), 'step': state['step'], 'epoch': state['epoch'],
        'bad_epochs': state['bad_epochs'], 'best': best, 'trainer_versions': dict(Counter(h for _, h in checkpoints.values())),
        'migration': state.get('checkpoint_migration'), 'gradient_first_labeled_step': gradient_paths,
        'segmentwise_gradient_from_step': state.get('segmentwise_gradient_from_step'),
        'pointwise_gradient_from_step': state.get('pointwise_gradient_from_step'),
        'manifest_sha256': contract['manifest'], 'actual_sources_sha256': contract['actual_sources'],
        'partition_audit': partitions, 'initialization_sha256': state['initialization_hash'],
        'views': dict(view_counts), 'resource': read_json(run/'resource_measurements.json'),
        'backbone': {'total': len(backbone), 'changed': changed, 'batch_norm_total': len(bn),
                     'batch_norm_changed': sum(warm[k] != chosen[k] for k in bn),
                     'warmup_sha256': file_hash(run/'best_epoch_001.keras')},
        'history': [{'epoch': row['epoch'], 'dev_nll': row['value'],
                     'train_loss': float(np.mean(losses[row['epoch']]))} for row in history],
        'rows': rows, 'orders': orders,
    }


def summarize(report):
    rows = report.pop('rows')
    report.pop('orders')
    report['metrics'] = {}
    for name in CONDITIONS + ('stress_macro',):
        conditions = STRESS if name == 'stress_macro' else (name,)
        for stratum in ('A_PAIRED', 'B_REAL_ONLY'):
            parts = [metrics([r for r in rows if r['condition_id'] == c and r['stratum'] == stratum]) for c in conditions]
            keys = ('n', 'n_REAL', 'n_FAKE', 'fp', 'fn', 'REAL_false_alarm', 'FAKE_miss',
                    'accuracy', 'f1', 'roc_auc', 'pr_auc', 'eer', 'mean_score')
            report['metrics'][name+'/'+stratum] = {
                k: None if parts[0][k] is None else
                (sum(p[k] for p in parts) if k in ('n', 'n_REAL', 'n_FAKE', 'fp', 'fn') else float(np.mean([p[k] for p in parts])))
                for k in keys}
    report['attacks'] = {}
    for attack in ('A01', 'A02', 'A03', 'A04', 'A05', 'A06'):
        report['attacks'][attack] = {}
        for name, conditions in [(c, (c,)) for c in CONDITIONS] + [('stress', STRESS)]:
            group = [r for r in rows if r['attack_id'] == attack and r['condition_id'] in conditions]
            report['attacks'][attack][name] = [sum(r['predicted_label'] == '0' for r in group), len(group)]
    report['worst_speakers'] = {}
    for stratum in ('A_PAIRED', 'B_REAL_ONLY'):
        for label in ('0', '1'):
            for name, conditions in (('clean', ('clean',)), ('stress', STRESS)):
                groups = {}
                for r in rows:
                    if r['stratum'] == stratum and r['label'] == label and r['condition_id'] in conditions:
                        groups.setdefault(r['speaker_id'], []).append(r)
                if groups:
                    values = [(sum(r['predicted_label'] != label for r in g), len(g), speaker)
                              for speaker, g in groups.items()]
                    worst = max(num/den for num, den, _ in values)
                    report['worst_speakers'][f'{stratum}/{label}/{name}'] = [v for v in sorted(values) if v[0]/v[1] == worst]
    report['flips'] = {}
    clean = {r['sample_id']: r['predicted_label'] for r in rows if r['condition_id'] == 'clean'}
    for condition in CONDITIONS[1:]:
        for stratum, label in (('A_PAIRED','0'), ('A_PAIRED','1'), ('B_REAL_ONLY','0')):
            group = [r for r in rows if r['condition_id'] == condition and r['stratum'] == stratum and r['label'] == label]
            report['flips'][f'{condition}/{stratum}/{label}'] = [
                sum(clean[r['sample_id']] == label and r['predicted_label'] != label for r in group),
                sum(clean[r['sample_id']] != label and r['predicted_label'] == label for r in group), len(group)]
    return report


def main():
    code = code_contract(ROOT)
    erm, channel = (audit(method, code) for method in ('erm', 'channel'))
    require(erm['orders'] == channel['orders'], 'Sample order differs')
    require(erm['initialization_sha256'] == channel['initialization_sha256'], 'Initialization differs')
    print(json.dumps({'erm': summarize(erm), 'channel': summarize(channel)}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
