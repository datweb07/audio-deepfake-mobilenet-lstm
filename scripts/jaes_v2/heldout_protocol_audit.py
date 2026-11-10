"""Authorized ZIP protocol audit. This entry point has no scoring path."""
import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re

from scripts.jaes_v2.empirical_readiness import ARCHIVE, ATTACKS, OUT, PROTOCOL, sample_heldout
from src.lava.jaes_v2.evidence import file_hash, object_hash, read_csv

CM = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
EVAL = 'LA/ASVspoof2019_LA_eval/flac/'
LICENSE = 'LA/ASVspoof2019_LA_eval/LICENSE.txt'
FOUNDATION = Path('outputs/jaes_v2/datasets/asv2019_foundation_v1')
PRELOCK = OUT/'heldout_prelock_9107_9207_v1.json'
RUNNER = 'scripts/jaes_v2/empirical_heldout.py'
AUDITOR = 'scripts/jaes_v2/heldout_protocol_audit.py'
OFFICIAL_COUNTS = {0:7355, 1:63882}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require_approval(record, scope, binding, digest):
    if (record.get('scope') != scope or record.get(binding) != digest
            or not all(isinstance(record.get(k), str) and record[k].strip()
                       for k in ('author', 'approved_at', 'approval_reference'))):
        raise ValueError(f'Missing or invalid {scope} approval')


def parse_protocol(data):
    rows = []
    seen = set()
    for line in data.decode('utf-8').splitlines():
        if not line.strip():
            continue
        speaker, identity, _, attack, label = line.split()
        if (not re.fullmatch(r'LA_E_\d+', identity) or identity in seen
                or not re.fullmatch(r'LA_\d+', speaker)):
            raise ValueError('Invalid or duplicate evaluation identity')
        if label == 'bonafide' and attack == '-':
            group = 'REAL'
        elif label == 'spoof' and attack in ATTACKS[1:]:
            group = attack
        else:
            raise ValueError('Unexpected evaluation label or attack')
        seen.add(identity)
        rows.append(dict(sample_id=identity, speaker_id=speaker, attack=group,
                         label=int(label == 'spoof'), split='evaluation'))
    return rows


def inspect_archive(archive, metadata, spec, seed, expected=None):
    members = Counter(archive.namelist())
    if any(members[name] != 1 for name in (CM, LICENSE)):
        raise ValueError('Missing or duplicate CM protocol or evaluation license')
    protocol = archive.read(CM)
    license_bytes = archive.read(LICENSE)
    license_text = license_bytes.decode('utf-8-sig').strip()
    if not license_text:
        raise ValueError('Evaluation license is empty')
    rows = parse_protocol(protocol)
    counts = Counter(r['label'] for r in rows)
    if counts != (OFFICIAL_COUNTS if expected is None else expected):
        raise ValueError(f'Official CM protocol count mismatch: {counts}')
    trial_members = {EVAL+r['sample_id']+'.flac' for r in rows}
    if any(members[name] != 1 for name in trial_members):
        raise ValueError('Missing or duplicate trial audio member')
    blocked_ids = {r['sample_id'].split(':')[-1] for r in metadata}
    blocked_speakers = {r['speaker_id'] for r in metadata}
    blocked_hashes = {r['checksum'] for r in metadata}
    quotas = {a:spec['REAL'] if a == 'REAL' else spec['per_attack'] for a in ATTACKS}
    chosen, support = sample_heldout(rows, seed, quotas, blocked_ids, blocked_speakers)
    seen = set()
    for row in chosen:
        member = EVAL+row['sample_id']+'.flac'
        digest = hashlib.sha256(archive.read(member)).hexdigest()
        if digest in blocked_hashes or digest in seen:
            raise ValueError('Duplicate or overlapping audio bytes')
        seen.add(digest)
        row.update(member=member, sha256=digest)
    audio_members = {name for name in members if name.startswith(EVAL) and name.endswith('.flac')}
    return dict(protocol_sha256=hashlib.sha256(protocol).hexdigest(), support=support, rows=chosen,
                official_counts=dict(real=counts[0], fake=counts[1]), zip_audio_members=len(audio_members),
                zip_without_cm_trial=len(audio_members-trial_members), trial_members_missing=0,
                license=dict(member=LICENSE, sha256=hashlib.sha256(license_bytes).hexdigest(), text=license_text),
                lineage='No verified generator lineage field in the official five-column CM protocol',
                selected_rows_sha256=object_hash(chosen))


def run(root, approval_path, output):
    prelock_path = root/PRELOCK
    prelock_sha = file_hash(prelock_path)
    approval = read(approval_path)
    require_approval(approval, 'AUTHOR_APPROVAL_PROTOCOL_READ', 'prelock_sha256', prelock_sha)
    if approval.get('allow_selected_audio_byte_hashes') is not True:
        raise ValueError('Selected audio byte checks need explicit permission')
    prelock = read(prelock_path)
    if prelock.get('authorized') is not False or prelock.get('execution_status') != 'HELDOUT_NOT_AUTHORIZED':
        raise ValueError('Unexpected prelock state')
    subset = approval.get('subset')
    reason = approval.get('resource_subset_reason')
    if subset not in ('primary','fallback') or not isinstance(reason, str) or not reason.strip():
        raise ValueError('Choose and justify the resource subset before opening the ZIP')
    output = output.resolve()
    if output.parent != (root/OUT).resolve() or output.suffix != '.json' or output.exists():
        raise ValueError('Use a new JSON output in the empirical execution folder')
    files = dict(prelock['files'])
    # Only the two-stage runner supersedes the runner snapshot in the immutable prelock.
    files.update({name:file_hash(root/name) for name in (AUDITOR, RUNNER)})
    for name, digest in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError(f'Prelock input mismatch: {name}')
    metadata = [row for role in ('train','development','calibration')
                for row in read_csv(root/FOUNDATION/f'{role}_manifest.csv')]
    spec = read(root/PROTOCOL)['heldout']
    with ZipFile(root/ARCHIVE) as archive:
        audit = inspect_archive(archive, metadata, spec[subset], spec['sampling_seed'])
    proposed = dict(authorized=False, execution_status='HELDOUT_NOT_AUTHORIZED',
                    prelock_sha256=prelock_sha, protocol_approval_sha256=file_hash(approval_path),
                    subset=subset, resource_subset_reason=reason, seeds=prelock['seeds'],
                    conditions=prelock['conditions'], files=files, ffmpeg_sha256=prelock['ffmpeg_sha256'],
                    benchmark_sha256=prelock['benchmark_sha256'], evaluation_manifest=audit)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(proposed, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def scoring_lock(proposed_path, approval_path):
    proposed = read(proposed_path)
    approval = read(approval_path)
    require_approval(approval, 'AUTHOR_APPROVAL_HELDOUT_SCORING', 'proposed_lock_sha256',
                     file_hash(proposed_path))
    if proposed.get('authorized') is not False or proposed.get('execution_status') != 'HELDOUT_NOT_AUTHORIZED':
        raise ValueError('Expected a protocol-only proposed lock')
    audit = proposed['evaluation_manifest']
    if (object_hash(audit['rows']) != audit['selected_rows_sha256']
            or approval.get('selected_rows_sha256') != audit['selected_rows_sha256']
            or approval.get('license_sha256') != audit['license']['sha256']):
        raise ValueError('Scoring approval does not bind the selected rows and reviewed license')
    return dict(proposed, authorized=True, author_approval_record=approval,
                scoring_approval_sha256=file_hash(approval_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-approval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.protocol_approval, args.output)
