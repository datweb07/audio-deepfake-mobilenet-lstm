"""Replay locked metrics and describe V3 error transitions without inference."""
from collections import Counter
from analyze_locked_heldout import ROOT, OUT, CONDITIONS, file_hash, object_hash, scoring_lock, verified_scores, read
from src.lava.jaes_v2.evidence import write_json


def decompose(erm, channel):
    if erm.keys() != channel.keys():
        raise ValueError('Unmatched source IDs')
    counts = Counter((erm[key], channel[key]) for key in erm)
    return dict(sources=len(erm), shared_errors=counts[1, 1], corrected=counts[1, 0],
                introduced=counts[0, 1], both_correct=counts[0, 0],
                net=counts[0, 1]-counts[1, 0])


def training_exposure(base, seeds):
    exposure = {}
    for seed in seeds:
        pair, source_orders, matched_contracts = {}, [], []
        for method in ('erm', 'channel'):
            path = base.parent/f'p10_medium_{method}_s{seed}_v2/scientific'
            seal = read(path/'seal.json')['output_hashes']
            for name in ('selected.keras', 'contract.json', 'result.json', 'scores.csv'):
                if file_hash(path/name) != seal[name]:
                    raise ValueError(f'Changed sealed training artifact: {path/name}')
            state = read(path/'result.json')['state']
            contract = read(path/'contract.json')
            journal = state['journal']
            order = [[sample for sample,_ in step['sources']] for step in journal]
            if len(journal) != 840 or state['epoch'] != 21 or sum(map(len,order)) != 13440:
                raise ValueError('Unexpected training budget')
            source_orders.append(order)
            if contract.pop('experiment') != path.parent.name:
                raise ValueError(f'Wrong experiment in {path}')
            policy = contract['config'].pop('augmentation_policy')
            if policy != {'erm': 'clean', 'channel': 'channel'}[method]:
                raise ValueError(f'Unexpected augmentation policy in {path}')
            matched_contracts.append(contract)
            pair[method] = dict(selected_sha256=seal['selected.keras'], selected=state['best']['tag'],
                augmentation_policy=policy,
                contract_sha256=seal['contract.json'], journal_artifact=(path/'result.json').relative_to(ROOT).as_posix(),
                journal_sha256=seal['result.json'], views=Counter(view for step in journal for _,view in step['sources']))
        if source_orders[0] != source_orders[1] or matched_contracts[0] != matched_contracts[1]:
            raise ValueError('Unexpected difference in within-seed source order or run contract')
        exposure[str(seed)] = pair
    return exposure


def main():
    base = ROOT/OUT
    lock_path = base/'heldout_proposed_9107_9207_v1.json'
    lock = scoring_lock(lock_path, base/'heldout_scoring_author_approval.json')
    lock_hash = object_hash(lock)
    directory = base/f'heldout_{lock_hash[:16]}'
    rows, _, result, shards = verified_scores(directory, lock, lock_hash)
    score_path = directory/'analysis/scores.csv'
    integrity = read(directory/'analysis/verified_analysis.json')['integrity']
    if file_hash(score_path) != integrity['scores_sha256'] or shards != integrity['score_shards']:
        raise ValueError('Stored scores differ from their verified hashes')
    errors = {(r['model'], r['condition'], r['sample_id']):r['error'] for r in rows}
    metadata = {r['sample_id']:(r['y'], r['attack']) for r in rows}
    attack_ids = sorted({attack for label,attack in metadata.values() if label == 1})
    comparisons = {}
    for seed in lock['seeds']:
        model_ids = [f'{arm}_{seed}' for arm in ('erm','channel')]
        per_condition, per_attack = {}, {}
        for condition in CONDITIONS:
            arms = [result['models'][model]['conditions'][condition] for model in model_ids]
            per_condition[condition] = dict(auc=[arm['auc'] for arm in arms])
            for label, name, metric in [(0,'REAL','false_alarm'), (1,'FAKE','miss')]:
                source_ids = [key for key,(y,_) in metadata.items() if y == label]
                pair = [{key:errors[model,condition,key] for key in source_ids} for model in model_ids]
                if [(len(arm),sum(arm.values())) for arm in pair] != [
                        (arm[metric]['n'],arm[metric]['errors']) for arm in arms]:
                    raise ValueError('Condition counts differ from locked metrics')
                per_condition[condition][name] = decompose(*pair)
        for attack in attack_ids:
            keys = [key for key,(y,a) in metadata.items() if y == 1 and a == attack]
            views = {}
            harmed, recovered = Counter(), Counter()
            for condition in CONDITIONS:
                pair = [{key:errors[model,condition,key] for key in keys} for model in model_ids]
                views[condition] = decompose(*pair)
                if condition == 'clean':
                    continue
                for key in keys:
                    change = pair[1][key]-pair[0][key]
                    if change == 1:
                        harmed[key] += 1
                    if change == -1:
                        recovered[key] += 1
            per_attack[attack] = dict(by_condition=views,
                stress_net=sum(views[c]['net'] for c in CONDITIONS if c != 'clean'),
                unique_stress_harmed=len(harmed), unique_stress_recovered=len(recovered),
                harmed_multiple_views=sum(n>1 for n in harmed.values()),
                recovered_multiple_views=sum(n>1 for n in recovered.values()),
                harmed_and_recovered=len(harmed.keys() & recovered.keys()))
        comparisons[str(seed)] = dict(conditions=per_condition, attacks=per_attack)
    output = ROOT/'outputs/jaes_v2/v3_score_diagnostics_20261111.json'
    write_json(output, dict(status='POST_HOC_EXPLORATORY', lock_sha256=lock_hash,
        source_sha256={p.relative_to(ROOT).as_posix():file_hash(p)
                       for p in (lock_path, directory/'manifest.json', directory/'metrics.json', score_path)},
        predictions=len(rows), exact_registered_replay=True, threshold=.5, comparisons=comparisons,
        training_exposure_audit=training_exposure(base,lock['seeds'])))
    print(output)


if __name__ == '__main__':
    main()
