"""Admission for the nested 1,280-recording Phase 10 extension."""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import platform
from zipfile import ZipFile

from scripts.jaes_v2.heldout_protocol_audit import ARCHIVE, CM, parse_protocol
from scripts.jaes_v2.phase10_medium import ROOT, REGISTRY as MEDIUM_REGISTRY, registered
from src.lava.jaes_v2 import phase10_training as engine, phase8_training as prior
from src.lava.jaes_v2.datasets import load_public_manifest
from src.lava.jaes_v2.evidence import file_hash, object_hash
from src.lava.jaes_v2.phase10_resources import GIB, MIN_D_FREE, WRITE_RESERVE, ResourceStop, require_resources
from src.lava.jaes_v2.provenance import audit_partitions

REGISTRY = Path('configs/jaes_v2/phase10/size1280_v1.json')
REGISTRY_SHA256 = 'dfa14e38a1f54aacc6fd7a5e18a364ed37ac9c4104eda549394c1cbf77a82158'
PROFILE = 'phase10_size1280_v1'
BASE = engine.BASE / 'size1280_v1'
FOUNDATION = Path('outputs/jaes_v2/datasets/asv2019_foundation_v1')
LOCK = engine.BASE / 'empirical_execution_v1/heldout_proposed_9107_9207_v1.json'
METHODS = ('erm', 'channel')
ATTACKS = dict(zip(('A01', 'A02', 'A03', 'A04', 'A05', 'A06'), (106, 108, 108, 108, 106, 104)))
TRAIN_HASH = '1fe860164a513eaa17cd42c59af6f08f8d8e3af09317c9baded327789b7109c9'


def registry(root=ROOT):
    path = root / REGISTRY
    if file_hash(path) != REGISTRY_SHA256:
        raise ValueError('Prospective Size1280 registry changed')
    spec = engine.read(path)
    medium, config = registered('p10_medium_erm_s9207_v2')
    inherited = ('architecture', 'sampling', 'batch_size', 'microbatch_size', 'gradient_accumulation',
                 'warmup_epochs', 'finetune_epochs', 'checkpoint_every_updates', 'learning_rate',
                 'finetune_learning_rate', 'scheduler', 'augmentation_seed', 'development_manifest')
    expected = {k: getattr(config, k) for k in inherited}
    expected.update(profile=PROFILE, schema=1, seed=9207, backend=engine.BACKEND,
                    initialization=config.pretrained_weights, optimizer='Adam', updates_per_epoch=80,
                    maximum_updates=1680, checkpoint_policy='CLEAN_BCE', threshold=.5,
                    train_manifest_sha256=TRAIN_HASH,
                    train_manifest='outputs/jaes_v2/supervisor_major_revision/candidate_train1280.csv',
                    validation=medium['validation'], calibration_enabled=False, official_evaluation_enabled=False,
                    experiments={m: f'p10_size1280_{m}_s9207_v1' for m in METHODS},
                    augmentations={m: list(prior.BANKS['clean' if m == 'erm' else m]) for m in METHODS})
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Size1280 registration differs from the fixed design')
    if spec['source'] != engine.fingerprints(root):
        raise ValueError('Frozen numerical source or FFmpeg changed')
    for name, digest in spec['files'].items():
        if file_hash(prior.safe_path(root, name)) != digest:
            raise ValueError(f'Registered input changed: {name}')
    if file_hash(root / spec['train_manifest']) != TRAIN_HASH:
        raise ValueError('Size1280 cohort hash mismatch')
    if file_hash(root / config.development_manifest) != config.development_sha256:
        raise ValueError('Development cohort hash mismatch')
    if spec['files'][MEDIUM_REGISTRY.as_posix()] != file_hash(root / MEDIUM_REGISTRY):
        raise ValueError('Medium registry changed')
    return spec, config


def run_config(template, method, spec, *, qualification=False):
    # The inherited Phase 8 config retains its own admitted manifest identity.
    # The outer contract independently binds the actual train source list and hash.
    return replace(template, augmentation_policy='clean' if method == 'erm' else 'channel',
                   maximum_updates=spec['qualification']['maximum_updates'] if qualification else spec['maximum_updates'])


def cohort(root, spec, config):
    medium, dev, b_ids, _ = prior.dataset(root, config)
    train = sorted(load_public_manifest(root / spec['train_manifest']), key=lambda r: r.sample_id)
    if (len(train) != 1280 or len({r.checksum for r in train}) != 1280
            or Counter(r.label for r in train) != {0: 640, 1: 640}
            or len({r.speaker_id for r in train}) != 20
            or Counter(r.attack_id for r in train if r.label) != ATTACKS):
        raise ValueError('Cohort size, unique audio, speaker or class/attack quotas differ')
    foundation = {r.sample_id: r for r in load_public_manifest(root / FOUNDATION / 'train_manifest.csv')}
    selected = {r.sample_id: r for r in train}
    if (any(r.split != 'train' or r.official_split != 'train' or foundation.get(r.sample_id) != r for r in train)
            or any(selected.get(r.sample_id) != r for r in medium)):
        raise ValueError('Cohort is not the exact nested official train selection')
    for record in train:
        if file_hash(prior.source_path(root, record.path)) != record.checksum:
            raise ValueError(f'Changed audio: {record.sample_id}')
    audit = audit_partitions(train + dev, require_speaker_disjoint=True)
    if len(dev) != 368 or len(b_ids) != 144:
        raise ValueError('Development A/B support changed')
    reserved = {
        role: [(r.sample_id, r.speaker_id, r.checksum)
               for r in load_public_manifest(root / FOUNDATION / f'{role}_manifest.csv')]
        for role in ('development', 'calibration')
    }
    held = engine.read(root / LOCK)['evaluation_manifest']
    if object_hash(held['rows']) != held['selected_rows_sha256']:
        raise ValueError('Locked evaluation identity changed')
    reserved['locked_heldout'] = [(r['sample_id'], r['speaker_id'], r['sha256']) for r in held['rows']]
    with ZipFile(root / ARCHIVE) as archive:
        protocol = archive.read(CM)
    if sha256(protocol).hexdigest() != spec['evaluation_protocol_sha256']:
        raise ValueError('Official evaluation protocol metadata changed')
    eval_rows = parse_protocol(protocol)
    if len(eval_rows) != 71237:
        raise ValueError('Unexpected official evaluation metadata support')
    reserved['official_protocol'] = [(r['sample_id'], r['speaker_id'], None) for r in eval_rows]
    overlap = {}
    for role, rows in reserved.items():
        ids, speakers, hashes = (set(column) for column in zip(*rows, strict=True))
        overlap[role] = {
            'sample_id': sum(r.official_recording_id in ids or r.sample_id in ids for r in train),
            'speaker': len({r.speaker_id for r in train} & speakers),
            'audio_sha256': None if role == 'official_protocol' else len({r.checksum for r in train} & hashes),
        }
        if any(overlap[role].values()):
            raise ValueError(f'Reserved cohort overlap: {role}')
    for epoch in range(config.warmup_epochs + config.finetune_epochs):
        order = prior.epoch_order(train, config, epoch)
        if sorted(order) != list(range(1280)) or len(order) // config.batch_size != spec['updates_per_epoch']:
            raise ValueError('Invalid epoch ordering or optimizer budget')
    return train, dev, b_ids, dict(audit, verified_train_hashes=len(train), nested_medium=len(medium),
        classes=Counter(str(r.label) for r in train), attacks=ATTACKS, speakers=Counter(r.speaker_id for r in train),
        overlap=overlap, official_protocol_rows=len(eval_rows), official_eval_audio_read=False,
        unknown_source_lineage=sum(r.source_recording_id == 'UNKNOWN' for r in train))


def runtime():
    return dict(python=platform.python_version(), system=platform.platform(), tensorflow=version('tensorflow'),
                packages={p: version(p) for p in ('keras', 'numpy', 'soundfile', 'librosa', 'scikit-learn')},
                deterministic_ops=True, one_dnn=False, threads=1,
                native_threads={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})


def reference(root, method):
    out = root / engine.BASE / f'p10_medium_{method}_s9207_v2/scientific'
    seal = engine.read(out / 'seal.json')
    if file_hash(out / 'contract.json') != seal['output_hashes']['contract.json']:
        raise ValueError('Historical initialization contract changed')
    contract = engine.read(out / 'contract.json')
    if contract['runtime'] != runtime():
        raise ValueError('Installed runtime differs from the sealed 640 seed 9207 pair')
    return out, contract


def weight_cache(spec):
    # Use the same location logic as Keras get_file, before model_factory can download.
    cache = Path(os.environ.get('KERAS_HOME', Path.home() / '.keras'))
    path = cache / 'models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5'
    if not path.is_file() or file_hash(path) != spec['imagenet_cache_sha256']:
        raise ValueError(f'Pinned ImageNet initialization cache unavailable: {path}')


def directory(root, spec, method, role):
    if role == 'scientific':
        relative = BASE / spec['experiments'][method] / role
    elif role in ('uninterrupted', 'resumable') and method == spec['qualification']['arm']:
        relative = BASE / 'qualification' / role
    else:
        raise ValueError('Unregistered run role')
    return prior.safe_path(root, relative)


def checkpoint_count(steps, every, stop=None):
    return len({0, steps, *range(every, steps + 1, every)} | ({stop} if stop else set()))


def storage(root, spec, *, qualification=False, remaining=None, current=None):
    plan = spec['resource_plan']
    mib = 1024 ** 2
    size = plan['checkpoint_mib'] * mib
    every = spec['checkpoint_every_updates']
    q = spec['qualification']
    complete = checkpoint_count(spec['maximum_updates'], every)
    bounded = checkpoint_count(q['maximum_updates'], every, q['split_step'])
    if remaining is None:
        remaining = 2 * (bounded if qualification else complete)
    extra = plan['qualification_extra_mib' if qualification else 'pair_extra_mib'] * mib
    if current is None:
        current = require_resources(root)
    free = current['d_free']
    minimum = MIN_D_FREE + WRITE_RESERVE + size * remaining + extra
    return dict(measurement=current, disk_required_bytes=minimum, additional_disk_bytes=max(0, minimum - free),
                current_headroom_gib=(free - MIN_D_FREE - WRITE_RESERVE) / GIB, per_checkpoint_budget_bytes=size,
                planned_remaining_checkpoints=remaining, extra_budget_bytes=extra)


def guard_disk(root, next_write_bytes=0):
    current = require_resources(root)
    if current['d_free'] < MIN_D_FREE + WRITE_RESERVE + next_write_bytes:
        raise ResourceStop('DISK_SPACE_PRESSURE_BEFORE_WRITE')
    return current
