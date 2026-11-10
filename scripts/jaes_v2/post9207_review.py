"""Reproduce post-9207 evidence from sealed development runs only."""
from collections import Counter
from pathlib import Path
from statistics import mean

from scripts.jaes_v2.empirical_readiness import (
    BASE, OUT, PROTOCOL, CONDITIONS, integrity, inventory, models_and_data, write_once,
)
from scripts.jaes_v2.empirical_seed_summary import analyze_seed
from scripts.jaes_v2.phase10_medium import comparison
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv
from src.lava.jaes_v2.phase10_resources import snapshot
from src.lava.jaes_v2.phase10_training import read, fingerprints


def attack_details(paths, attack):
    tables = {}
    for method, path in paths.items():
        tables[method] = {
            (r['sample_id'], r['condition_id']): float(r['raw_score'])
            for r in read_csv(path / 'scores.csv') if r['attack_id'] == attack
        }
    erm, channel = tables['erm'], tables['channel']
    ids = sorted({source for source, _ in erm})
    stress = CONDITIONS[1:]
    misses = {m: {c: sum(t[s, c] < .5 for s in ids) for c in CONDITIONS}
              for m, t in tables.items()}
    all_missed = {m: sum(all(t[s, c] < .5 for c in stress) for s in ids)
                  for m, t in tables.items()}
    switches = {}
    for source in ids:
        lost = [c for c in stress if channel[source, c] < .5 <= erm[source, c]]
        recovered = [c for c in stress if erm[source, c] < .5 <= channel[source, c]]
        if lost or recovered:
            switches[source] = dict(harmed=lost, recovered=recovered)
    return dict(sources=len(ids), stress_views=len(ids)*len(stress), misses=misses,
                all_stress_missed=all_missed, transitions=switches,
                harmed_sources=sum(bool(r['harmed']) for r in switches.values()),
                recovered_sources=sum(bool(r['recovered']) for r in switches.values()),
                mean_fake_score={m: {c: mean(t[s,c] for s in ids) for c in CONDITIONS}
                                 for m,t in tables.items()})


def run(root):
    before = fingerprints(root)
    historical = integrity()
    evidence = read(root/OUT/'paired_seed_evidence_9107_9207.json')
    plan = read(root/PROTOCOL)
    model_records, results, attacks = {}, {}, {}
    for seed in plan['required_seeds']:
        runs, _, _, _, audit = models_and_data(root, seed)
        result = analyze_seed(root, seed)
        original = evidence['seeds'][str(seed)]
        for key in ('paired', 'deltas', 'macro_stress_auc_interval', 'endpoints'):
            if result[key] != original[key]:
                raise ValueError(f'Recomputed {seed} {key} differs')
        paired = comparison(f'p10_medium_erm_s{seed}_v2', f'p10_medium_channel_s{seed}_v2')
        if paired != read(root/BASE/f'medium_pair_s{seed}_v2.json'):
            raise ValueError(f'Recomputed {seed} comparison differs')
        result['guard'] = paired['channel_guard_against_selected_erm']
        results[str(seed)] = result
        paths = {}
        for method, (path, contract, state, final) in runs.items():
            identity = f'p10_medium_{method}_s{seed}_v2'
            expected = {(9107,'erm'):560,(9107,'channel'):760,
                        (9207,'erm'):520,(9207,'channel'):720}[seed,method]
            tag = f'u{expected:06d}_e{expected//40:03d}_c000'
            policy = 'clean' if method == 'erm' else 'channel'
            best = min(state['history'], key=lambda row: (row['clean']['nll'], row['epoch']))
            if (contract['experiment'] != identity or contract['config']['training_seed'] != seed
                    or contract['config']['augmentation_policy'] != policy
                    or contract['backend'] != 'FULL_SHAPE_RECOMPUTE'
                    or contract['policy'] != 'CLEAN_BCE' or state['best']['tag'] != tag
                    or len(state['history']) != 21 or len(state['journal']) != 840
                    or best['step'] != expected or best['epoch'] != expected//40):
                raise ValueError(f'Unexpected registered run contract: {identity}')
            paths[method] = path
            selected = path/'checkpoints'/tag
            model_records[identity] = dict(seed=seed, method=method, augmentation=policy,
                template_id=contract['config']['experiment_id'], backend=contract['backend'],
                checkpoint_policy=contract['policy'], step=state['step'], epoch=state['epoch'],
                checkpoint=tag, data_audit=audit, source_order_sha256=object_hash(
                    [[s[0] for s in j['sources']] for j in state['journal']]),
                update_views=dict(Counter(s[1] for j in state['journal'] for s in j['sources'])),
                initial_weights=contract['initial_weights'], selected_weights=final['selected_weights'],
                score_rows=result['source_count']*7,
                files={p.relative_to(root).as_posix():file_hash(p) for p in (
                    *(path/f for f in ('scores.csv','selected.keras','seal.json','contract.json','result.json','status.json')),
                    *(selected/f for f in ('seal.json','controller.json','state.index','state.data-00000-of-00001')))})
        attacks[str(seed)] = {a:attack_details(paths,a) for a in ('A05','A06')}
    archive = inventory(root)
    pending = root/OUT/'heldout_lock_pending.json'
    lock = read(pending)
    files = lock['files']
    for record in model_records.values():
        files.update(record['files'])
    for role in ('train','development','calibration'):
        name = f'outputs/jaes_v2/datasets/asv2019_foundation_v1/{role}_manifest.csv'
        files[name] = file_hash(root/name)
    for name, digest in files.items():
        current = archive['sha256'] if name == archive['path'] else file_hash(root/name)
        if current != digest:
            raise ValueError(f'Prelock source differs from previous evidence: {name}')
    if before != fingerprints(root) or before['ffmpeg_binary'] != lock['ffmpeg_sha256']:
        raise ValueError('Frozen source or FFmpeg changed')
    benchmark = root/OUT/'benchmark.json'
    if file_hash(benchmark) != lock['benchmark_sha256']:
        raise ValueError('Benchmark evidence changed')
    bench = read(benchmark)
    resources = snapshot(root)
    budgets = {}
    for name in ('primary','fallback'):
        budget = plan['heldout'][name]
        count = budget['target']
        budgets[name] = dict(**budget, predictions=count*4*len(CONDITIONS),
            planning_seconds=count*(bench['total_seconds']/bench['sources']
                                    + 2*len(CONDITIONS)*bench['mean_forward_seconds']))
    lock.update(authorized=False, execution_status='HELDOUT_NOT_AUTHORIZED', subset='primary',
                resource_subset_reason='Proposed from development benchmark and direct ZIP access before labels. Recheck resources before approval.',
                models=model_records, archive=archive, benchmark=bench, budgets=budgets,
                resources=resources, score_orientation=plan['score_orientation'], threshold=.5,
                tie='FAKE', preprocessing='P0', transform='Noise5 and fixed-absolute Noise10 + libopus 24 kb/s in both orders',
                metrics={k:plan[k] for k in ('primary','secondary','bootstrap','delta')},
                selection=plan['heldout'], prior_pending_sha256=file_hash(pending),
                blockers=['Explicit author approval for official protocol access and locked execution',
                          'Reconcile CM trial counts and archive IDs, review license, and enforce overlap checks after access approval',
                          'Bind finalized author approval and recheck resources before locked scoring'])
    report=dict(runs=model_records, historical_integrity=historical, frozen_fingerprints=before,
                seeds=results, summary=evidence['summary'], attack_details=attacks,
                protocol_sha256=file_hash(root/PROTOCOL), replicated_summary_sha256=file_hash(
                    root/OUT/'paired_seed_evidence_9107_9207.json'),
                seed_9307='SEED_9307_NO_GO', official_evaluation_used=False, calibration_used=False)
    for name, payload in (('post9207_consolidated_v1.json', report),
                          ('heldout_prelock_9107_9207_v1.json', lock)):
        path = root/OUT/name
        if path.exists():
            existing = read(path)
            # Resource measurements are a dated snapshot, not a reproducibility invariant.
            if object_hash({k:v for k,v in existing.items() if k != 'resources'}) != object_hash(
                    {k:v for k,v in payload.items() if k != 'resources'}):
                raise ValueError(f'Reproduced artifact differs: {path}')
        else:
            write_once(path, payload)


if __name__ == '__main__':
    run(Path.cwd())
