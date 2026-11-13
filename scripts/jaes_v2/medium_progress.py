"""Read-only, line-oriented progress for registered Phase 10 runs."""
import argparse
import json
import math
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path

RUN_ID = re.compile(r'p10_medium_(erm|channel)_s(9107|9207|9307)_v2')
SIZE_ID = re.compile(r'p10_size1280_(erm|channel)_s(9207)_v1')
CHECKPOINT = re.compile(r'u(\d+)_e\d+_c\d+')
BASE = Path('outputs/jaes_v2/phase10')
REGISTRY = Path('configs/jaes_v2/phase10/prospective_medium_v1.json')
SIZE_REGISTRY = Path('configs/jaes_v2/phase10/size1280_v1.json')
ACTIVE = {'RUNNING', 'WARNING'}
TERMINAL = {'PAUSE_AT_CHECKPOINT', 'STOP_FOR_RESOURCE_SAFETY',
            'STOP_FOR_NUMERICAL_FAILURE', 'SCIENTIFIC_NO_GO', 'SCIENTIFIC_COMPLETE'}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def duration(seconds):
    if not number(seconds) or seconds < 0:
        return 'N/A'
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def latest_resume(path):
    for file in sorted((path/'resume_events').glob('u[0-9]*.json'), reverse=True):
        event = read_json(file)
        if event and type(event.get('step')) is int and event['step'] >= 0:
            return event['step']
    return None


def valid_status(status, total, epochs):
    return (isinstance(status, dict) and isinstance(status.get('status'), str)
            and status['status'] in ACTIVE | TERMINAL
            and type(status.get('step')) is int and 0 <= status['step'] <= total
            and type(status.get('epoch')) is int and 0 <= status['epoch'] <= epochs)


def display_epoch(status, epochs):
    completed = status['epoch']
    last = mapping(status.get('last_update'))
    # At the boundary, close the just-validated epoch before opening the next one.
    if last.get('step') == status['step'] and last.get('epoch') == completed - 1:
        return max(1, completed)
    return min(epochs, completed + 1)


def is_complete(path, status, total, epochs):
    if (status['status'], status['step'], status['epoch']) != ('SCIENTIFIC_COMPLETE', total, epochs):
        return False
    sealed = mapping(mapping(read_json(path/'seal.json')).get('output_hashes'))
    return 'result.json' in sealed and 'status.json' in sealed and (path/'result.json').is_file()


def metrics_line(record, fields):
    return [f'{label}: {value:{fmt}}' for key, label, fmt in fields
            if number(value := record.get(key))]


def progress(status, total, *, elapsed=None, complete=False, unicode=True, width=120):
    step = status['step']
    percent = '100.0%' if complete else '<100%' if step == total else f'{min(99.9, 100*step/total):.1f}%'
    active = status['status'] in ACTIVE
    eta = duration(status.get('remaining_train_seconds')) if active or complete else 'N/A'
    timing = f"monitor_elapsed: {duration(elapsed)} - train_ETA: {eta}"
    training = metrics_line(mapping(status.get('last_update')),
                            [('loss','loss','.4f'),('lr','lr','.2e'),('norm','grad_norm','.4f')])
    if width < 104:
        # Compact labels keep both counters visible on standard 80-column consoles.
        timing = f"mon_elapsed: {duration(elapsed)} - train_ETA: {eta}"
    room = width - len(f'{step}/{total} [] {percent} - {timing}') - 1
    metric_room = max(0, room - 25)
    if training and room < 25 + len(training[0]) + 3:
        metric_room = min(max(0, room - 15), len(training[0]) + 3)
    bar_width = min(30, max(1, room - min(metric_room, sum(len(v)+3 for v in training))))
    filled = bar_width if complete else min(bar_width-1, step*bar_width//total)
    bar = ('█' if unicode else '#')*filled + '-'*(bar_width-filled)
    parts = [f'{step}/{total} [{bar}] {percent}', timing, *training]
    return parts


def validation_line(validation):
    clean = mapping(validation.get('clean'))
    values = metrics_line(clean,[('roc_auc','val_auc','.4f'),('nll','val_bce','.4f')])
    for numerator, denominator, label in [('fp','n_REAL','REAL_FA'),('fn','n_FAKE','FAKE_miss')]:
        a, b = clean.get(numerator), clean.get(denominator)
        if type(a) is int and type(b) is int and 0 <= a <= b and b > 0:
            values.append(f'{label}: {a}/{b}')
    return f"Validation (epoch {validation.get('epoch', 'N/A')}): " + ' - '.join(values or ['N/A'])


def checkpoint_step(tag):
    match = CHECKPOINT.fullmatch(tag) if isinstance(tag, str) else None
    return int(match[1]) if match else None


class Terminal:
    def __init__(self, stream, *, live):
        self.stream = stream
        self.live = live
        self.previous = ''

    @property
    def width(self):
        return shutil.get_terminal_size((120, 24)).columns if self.live else 160

    def finish(self):
        if self.previous:
            print(file=self.stream, flush=True)
            self.previous = ''

    def message(self, value):
        self.finish()
        for line in textwrap.wrap(value, width=max(1, self.width-1)):
            print(line, file=self.stream, flush=True)

    def update(self, parts):
        value = ' - '.join(parts)
        if not self.live:
            print(value, file=self.stream, flush=True)
            return []
        available = max(1, self.width-1)
        visible = parts[0][:available]
        omitted = []
        for part in parts[1:]:
            if len(visible)+3+len(part) <= available:
                visible += ' - '+part
            else:
                omitted.append(part)
        if visible != self.previous:
            padding = max(0, min(available, len(self.previous))-len(visible))
            print('\r'+visible+' '*padding, end='', file=self.stream, flush=True)
            self.previous = visible
        return omitted


def monitor(path, config, total, terminal, *, once=False, unicode=True):
    epochs = config['warmup_epochs'] + config['finetune_epochs']
    started = time.monotonic()
    prior = shown_epoch = shown_validation = shown_best = shown_latest = shown_resume = None
    warning = None
    waiting = stale_warning = False
    while True:
        status = read_json(path/'status.json')
        if not valid_status(status,total,epochs):
            if not waiting:
                terminal.message('Waiting for readable status.json; trainer may still be starting or writing.')
            waiting = True
        else:
            waiting = False
            complete = is_complete(path,status,total,epochs)
            stage = status['status']
            epoch = display_epoch(status,epochs)
            validation = mapping(status.get('validation'))
            valid_step = validation.get('step')
            best = mapping(status.get('best')).get('tag')
            latest = mapping(status.get('latest')).get('checkpoint')
            resume = latest_resume(path)
            changed = prior != status or shown_resume != resume
            if changed or complete:
                if epoch != shown_epoch:
                    terminal.message(f'Epoch {epoch}/{epochs}')
                    shown_epoch = epoch
                if resume is not None and resume != shown_resume:
                    terminal.message(f'Resume recorded at update {resume}')
                    shown_resume = resume
                omitted = terminal.update(progress(status,total,elapsed=None if once else time.monotonic()-started,
                                                   complete=complete,unicode=unicode,width=terminal.width))
                new_validation = type(valid_step) is int and 0 < valid_step <= status['step'] and valid_step != shown_validation
                if omitted and (new_validation or stage in TERMINAL):
                    terminal.message(' - '.join(omitted))
                if new_validation:
                    terminal.message(validation_line(validation))
                    if checkpoint_step(best) == valid_step:
                        terminal.message(f'Best checkpoint updated at step {valid_step}: {best}')
                    elif shown_validation is not None and best and best == shown_best:
                        terminal.message('Best checkpoint unchanged.')
                    elif prior is not None and best != shown_best and checkpoint_step(best) is not None:
                        terminal.message(f'Best checkpoint: {best}')
                    shown_validation = valid_step
                elif prior is not None and best != shown_best and checkpoint_step(best) is not None:
                    terminal.message(f'Best checkpoint: {best}')
                if prior is not None and latest != shown_latest and checkpoint_step(latest) is not None and latest != best:
                    terminal.message(f'Checkpoint saved: {latest}')
                shown_best, shown_latest = best, latest
                if status.get('warning') and status['warning'] != warning:
                    terminal.message(f"Warning: {status['warning']}")
                    resources = mapping(status.get('resources'))
                    if stage == 'STOP_FOR_RESOURCE_SAFETY':
                        values = [f'{label} {resources[key]/2**30:.2f} GiB' for label,key in
                                  [('RAM','ram_available'),('commit','commit_available'),('C','c_free'),('D','d_free')]
                                  if number(resources.get(key))]
                        if values:
                            terminal.message('Last resource sample: '+', '.join(values))
                elif stage == 'WARNING' and not status.get('warning') and (not prior or prior['status'] != stage):
                    terminal.message('Status: WARNING')
                warning = status.get('warning')
                stale_warning = False
            if complete:
                terminal.message('Training completed.')
                terminal.message(f'Best checkpoint: {best or "N/A"}')
                terminal.message(f'Status: {stage}')
                return
            if stage in TERMINAL - {'SCIENTIFIC_COMPLETE'}:
                terminal.message(f'Status: {stage}'+(f' | Latest: {latest}' if latest else ''))
                return
            try:
                stale = time.time() - (path/'status.json').stat().st_mtime >= 120
            except OSError:
                stale = False
            if stage == 'SCIENTIFIC_COMPLETE' and changed:
                terminal.message('Waiting for the completion seal.')
            elif stale and not stale_warning:
                terminal.message('Warning: No recent status write. Check the training terminal for validation, checkpoint writing or process exit.')
                stale_warning = True
            if terminal.live and not changed:
                terminal.update(progress(status,total,elapsed=time.monotonic()-started,
                                         unicode=unicode,width=terminal.width))
            prior = status
        if once:
            terminal.finish()
            return
        time.sleep(1)


def supports_unicode(stream):
    try:
        '█'.encode(stream.encoding or 'ascii')
        return True
    except (LookupError, UnicodeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--ascii', action='store_true', help='Use ASCII progress blocks')
    args = parser.parse_args()
    match = SIZE_ID.fullmatch(args.experiment)
    if match:
        registry = read_json(SIZE_REGISTRY)
        if not registry or mapping(registry.get('experiments')).get(match[1]) != args.experiment:
            parser.error('Unregistered size1280 experiment')
        path = BASE / 'size1280_v1' / args.experiment / 'scientific'
        template = registry
    else:
        match = RUN_ID.fullmatch(args.experiment)
        registry = read_json(REGISTRY)
        if not match or not registry or args.experiment not in {r['experiment_id'] for r in registry['paired_runs']}:
            parser.error('Run from the repository root with a registered experiment ID')
        path = BASE / args.experiment / 'scientific'
        template = read_json(Path(registry['template']))
    contract = read_json(path/'contract.json')
    config = mapping(contract.get('config')) if contract else template
    total = registry['maximum_updates']
    if (not config or not all(type(config.get(k)) is int and config[k] >= 0
                              for k in ('warmup_epochs','finetune_epochs'))
            or config['warmup_epochs'] + config['finetune_epochs'] <= 0
            or (contract and config.get('maximum_updates') != total)):
        parser.error('Unreadable epoch schedule or optimizer budget differs from the registry')
    terminal = Terminal(sys.stdout, live=sys.stdout.isatty() and not args.once)
    method, seed = match.groups()
    terminal.message(f'Experiment: {args.experiment}')
    terminal.message(f"Model: {config.get('architecture', 'N/A')} | Method: {'ERM' if method=='erm' else 'Channel'} | Seed: {seed}")
    try:
        monitor(path,config,total,terminal,once=args.once,unicode=not args.ascii and supports_unicode(sys.stdout))
    except KeyboardInterrupt:
        terminal.message('Monitor stopped.')


if __name__ == '__main__':
    main()
