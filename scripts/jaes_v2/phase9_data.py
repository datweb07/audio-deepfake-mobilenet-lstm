"""Prepare a Phase 9 medium-plus training manifest without opening held-out roles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.datasets import load_public_manifest
from src.lava.jaes_v2.evidence import file_hash, write_csv, write_json
from src.lava.jaes_v2.phase7_data import all_attack_select
from src.lava.jaes_v2.phase8_training import source_path
from src.lava.jaes_v2.provenance import audit_partitions


def prepare(root):
    foundation = root / 'outputs/jaes_v2/datasets/asv2019_foundation_v1'
    contract = json.loads((foundation / 'foundation.json').read_text(encoding='utf-8'))
    train = foundation / 'train_manifest.csv'
    medium = root / 'outputs/jaes_v2/phase7/selections/medium_all_attack_v1/manifest.csv'
    if file_hash(train) != contract['manifest_hashes']['train']:
        raise ValueError('Foundation train manifest changed')
    if file_hash(medium) != '029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319':
        raise ValueError('Medium comparison manifest changed')
    selected, _, support = all_attack_select(load_public_manifest(train), per_cell=64, seed=9109, role='train')
    dev = [row for row in load_public_manifest(medium) if row.split == 'development']
    if len(selected) != 2560 or len(dev) != 368:
        raise ValueError('Unexpected training or development support')
    audit = audit_partitions(selected + dev, require_speaker_disjoint=True)
    decoded = set()
    for row in selected + dev:
        path = source_path(root, row.path)
        if file_hash(path) != row.checksum:
            raise ValueError(f'Audio changed for {row.sample_id}')
        samples, rate = sf.read(path, dtype='float32', always_2d=True)
        digest = hashlib.sha256(str((rate, samples.shape)).encode() + samples.astype('<f4').tobytes()).hexdigest()
        if not np.isfinite(samples).all() or digest in decoded:
            raise ValueError(f'Nonfinite or duplicate decoded audio for {row.sample_id}')
        decoded.add(digest)
    out = root / 'outputs/jaes_v2/phase9/medium_plus_v1'
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'train_manifest.csv', [row.to_dict() for row in selected])
    write_csv(out / 'development_manifest.csv', [row.to_dict() for row in dev])
    write_csv(out / 'foundation_train_support.csv', support)
    write_json(out / 'contract.json', {
        'seed': 9109,
        'per_speaker_per_class': 64,
        'train_by_label': dict(Counter(row.label for row in selected)),
        'train_by_attack': dict(Counter(row.attack_id for row in selected if row.label == 1)),
        'train_speakers': len({row.speaker_id for row in selected}),
        'foundation_train_sha256': file_hash(train),
        'medium_parent_sha256': file_hash(medium),
        'train_sha256': file_hash(out / 'train_manifest.csv'),
        'development_sha256': file_hash(out / 'development_manifest.csv'),
        'partition_audit': audit,
        'unique_decoded_waveforms': len(decoded),
        'calibration_used': False,
        'official_evaluation_used': False,
        'training_authorized': False,
    })
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(prepare(Path.cwd()))
