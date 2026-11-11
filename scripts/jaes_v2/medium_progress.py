"""Read-only display of the Phase 10 trainer's committed metadata."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

RUN_ID = re.compile(r'p10_medium_(erm|channel)_s(9107|9207|9307)_v2')
BASE = Path('outputs/jaes_v2/phase10')
NA = 'N/A — not recorded by frozen trainer'
ACTIVE = {'RUNNING', 'WARNING'}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def duration(seconds):
    if seconds is None:
        return NA
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def latest_resume(path):
    events = sorted((path/'resume_events').glob('u[0-9]*.json'), reverse=True)
    return next((event for file in events if (event := read_json(file)) is not None), None)


def render(identity, registry, status, contract=None, *, elapsed=0, stale=False, tick=0, resume=None):
    match = RUN_ID.fullmatch(identity)
    method, seed = match.groups()
    config = (contract or {}).get('config', {})
    limit = registry['maximum_updates']
    if config.get('maximum_updates', limit) != limit:
        return 'Optimizer budget in the run contract differs from the registered budget.'
    epochs = config['warmup_epochs'] + config['finetune_epochs'] if config else None
    method_name = 'ERM' if method == 'erm' else 'Channel Augmentation'
    lines = ['LAVA JOURNAL V2 — TRAINING',
             f"Model: {config.get('architecture', NA)}",
             f"Experiment: {identity}",
             f"Method: {method_name} | Seed: {seed} | Backend: {registry['backend']}"]
    if status is None:
        return '\n'.join([*lines, 'Waiting for readable status.json'])
    step = status.get('step', 0)
    stage = status.get('status', NA)
    filled = min(24, max(0, step * 24 // limit))
    marker = '|/-\\'[tick % 4] if filled < 24 and stage in ACTIVE else ' '
    bar = '█' * filled + '░' * (24 - filled)
    last = status.get('last_update') or {}
    validation = status.get('validation') or {}
    clean = validation.get('clean') or {}
    resources = status.get('resources') or {}
    best = status.get('best') or {}
    latest = status.get('latest') or {}
    eta = status.get('remaining_train_seconds')
    resume_text = f"Recorded at update {resume.get('step', NA)}" if resume else 'No resume event recorded'
    lines.extend([
        f"Epoch {status.get('epoch', NA)}/{epochs if epochs is not None else NA}",
        f"Optimizer updates [{bar}] {step}/{limit} — {100*step/limit:.1f}% {marker}",
        f'Monitor elapsed {duration(elapsed)} | Training wall time {NA}',
        f'Optimizer ETA {duration(eta)} (excludes validation, checkpoints and export)',
        f"Loss {last.get('loss', NA)} | LR {last.get('lr', NA)} | Gradient norm {last.get('norm', NA)}",
        f"Training accuracy and AUC {NA}",
        f"Last completed validation: tier {validation.get('tier', NA)} at update {validation.get('step', NA)}",
        f"Clean ROC-AUC {clean.get('roc_auc', NA)} | Clean BCE {clean.get('nll', NA)}",
        f"Clean REAL false alarms {clean.get('fp', NA)}/{clean.get('n_REAL', NA)} | "
        f"FAKE misses {clean.get('fn', NA)}/{clean.get('n_FAKE', NA)}",
        f"Best checkpoint {best.get('tag', NA)} | Latest {latest.get('checkpoint', NA)}",
        'Resources (GiB) ' + ' | '.join(
            f'{label} {resources[key]/2**30:.2f}' if resources.get(key) is not None else f'{label} {NA}'
            for label,key in (('RAM','ram_available'),('commit','commit_available'),('C free','c_free'),('D free','d_free'))),
        f'System status {stage} | Resume: {resume_text}',
    ])
    if status.get('warning'):
        lines.append(f"Warning {status['warning']}")
    if stage in ACTIVE:
        lines.append('Awaiting next committed progress update')
        if stale:
            lines.append('No recent write. Check the training terminal for validation, delay or process exit.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not RUN_ID.fullmatch(args.experiment):
        parser.error('Expected a registered prospective medium experiment ID')
    registry = read_json(Path('configs/jaes_v2/phase10/prospective_medium_v1.json'))
    if registry is None or args.experiment not in {r['experiment_id'] for r in registry['paired_runs']}:
        parser.error('Run from the repository root with the prospective registry available')
    path = BASE / args.experiment / 'scientific'
    tty = sys.stdout.isatty() and not args.once
    start = time.monotonic()
    tick = 0
    previous = None
    try:
        while True:
            status = read_json(path/'status.json')
            contract = read_json(path/'contract.json')
            resume = latest_resume(path)
            try:
                stale = time.time() - (path/'status.json').stat().st_mtime >= 120
            except OSError:
                stale = False
            key = status, contract, stale, resume
            if tty or key != previous:
                if tty:
                    print('\x1b[H\x1b[J', end='')
                print(render(args.experiment, registry, status, contract,
                             elapsed=time.monotonic()-start, stale=stale, tick=tick, resume=resume), flush=True)
                previous = key
            if args.once or (status and status.get('status') not in ACTIVE):
                return
            tick += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nMonitor stopped.', flush=True)


if __name__ == '__main__':
    main()
