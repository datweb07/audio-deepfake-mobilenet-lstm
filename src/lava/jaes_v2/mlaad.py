"""MLAAD-tiny metadata adapter; author annotations, never folder-only generator identity."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path, PurePosixPath
import json

import numpy as np
import soundfile as sf

from .acquisition import license_gate, pinned_revision, safe_relative
from .datasets import PublicSourceRecord
from .evidence import file_hash, object_hash, write_csv, write_json
from .provenance import UNKNOWN, audit_partitions, known


def source_speaker(original: str) -> str:
    """Parse documented original_file's explicit M-AILABS directory schema, not audio ID."""
    parts = PurePosixPath(original).parts
    if len(parts) == 7 and parts[1] == "by_book" and parts[2] in {"male", "female"} and parts[-2] == "wavs":
        return "M-AILABS:" + "/".join(parts[:4])
    return UNKNOWN


def metadata_inventory(discovery: Path) -> tuple[list[dict], dict]:
    release = json.loads((discovery / "discovery.json").read_text())
    pinned_revision(release["revision"])
    if file_hash(discovery / "inventory.json") != release["inventory_sha256"]:
        raise ValueError("Inventory hash mismatch")
    for row in release["files"]:
        if file_hash(discovery / row["path"]) != row["checksum"]:
            raise ValueError("Authoritative metadata/license changed")
    files = json.loads((discovery / "inventory.json").read_text())
    audio = {row["path"]: row for row in files if row["path"].endswith(".wav")}
    annotations = {}
    orphan_rows = 0
    for path in sorted(discovery.rglob("meta.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="|")
            required = {"path", "original_file", "language", "model_name", "architecture"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"Unsupported MLAAD metadata schema: {path}")
            for row in reader:
                relative = safe_relative(row["path"].removeprefix("./"))
                if relative not in audio:
                    orphan_rows += 1
                    continue
                if relative in annotations:
                    raise ValueError("Conflicting/duplicate metadata annotation; quarantine required")
                if not relative.startswith("fake/") or row["language"] != relative.split('/')[1]:
                    raise ValueError("Metadata class/language mismatch")
                annotations[relative] = (row, path.relative_to(discovery).as_posix())
    records = []
    for path in sorted(audio):
        parts = path.split('/')
        label = int(parts[0] == "fake")
        if parts[0] not in {"fake", "original"} or len(parts) < 3:
            raise ValueError("Unrecognized authoritative audiofolder label layout")
        annotation, annotation_path = annotations.get(path, ({}, UNKNOWN))
        original = annotation.get("original_file", UNKNOWN)
        records.append({"path": path, "sample_id": "mlaad_" + object_hash(path)[:24], "label": label,
                        "language": annotation.get("language", parts[1]),
                        "attack_id": annotation.get("model_name", UNKNOWN),
                        "generator_id": annotation.get("model_name", UNKNOWN),
                        "generator_family": annotation.get("architecture", UNKNOWN),
                        "source_corpus": "M-AILABS" if label == 0 or known(original) else UNKNOWN,
                        "source_recording_id": original,
                        "source_speaker_id": source_speaker(original) if known(original) else UNKNOWN,
                        "speaker_id": UNKNOWN, "target_speaker_id": UNKNOWN,
                        "annotation_path": annotation_path, "expected_size": audio[path]["size"],
                        "expected_sha256": audio[path]["sha256"],
                        "annotation_status": "AUTHOR_MODEL_NAME" if annotation else "AUDIOFOLDER_CLASS_ONLY"})
    summary = {"revision": release["revision"], "audio_records": len(records),
               "class_counts": dict(Counter(str(row['label']) for row in records)),
               "language_label_counts": dict(Counter(f"{row['language']}/{row['label']}" for row in records)),
               "fake_model_names": len({row['generator_id'] for row in records if row['label'] and known(row['generator_id'])}),
               "missing_fake_annotations": sum(row['label'] and row['annotation_status'] != 'AUTHOR_MODEL_NAME' for row in records),
               "metadata_rows_not_in_tiny_audio": orphan_rows,
               "identity_limit": "Author model_name identifies a released system annotation, not a verified checkpoint hash or detector familiarity"}
    return records, summary


def select_development(records: list[dict], *, revision: str, per_group: int, group_count: int,
                       language: str, seed: int, selected_groups: list[str] | None = None,
                       selected_ids: set[str] | None = None) -> dict:
    pinned_revision(revision)
    if per_group < 8 or group_count < 2 or not language or seed < 0:
        raise ValueError("Require >=8 per group, >=2 groups, a language and seed")
    candidates = [row for row in records if row['language'] == language and
                  (selected_ids is None or row['sample_id'] in selected_ids)]
    if selected_ids is not None and selected_ids - {row['sample_id'] for row in candidates}:
        raise ValueError("Unknown selected ID or incompatible language")
    groups = defaultdict(list)
    for row in candidates:
        if row['label'] and row['annotation_status'] == 'AUTHOR_MODEL_NAME' and known(row['source_speaker_id']):
            groups[row['generator_id']].append(row)
    eligible = sorted(group for group, members in groups.items() if len(members) >= per_group and
                      len({row['source_speaker_id'] for row in members}) >= 3)
    chosen = selected_groups if selected_groups is not None else eligible[:group_count]
    if len(chosen) != group_count or len(set(chosen)) != group_count or set(chosen) - set(eligible):
        raise ValueError("Insufficient documented group/source coverage")
    rng = np.random.default_rng(seed)
    selected = []
    # Balance source-text speakers within each fake group where possible. No model scores are read.
    for group in chosen:
        by_speaker = defaultdict(list)
        for row in sorted(groups[group], key=lambda row: row['sample_id']):
            by_speaker[row['source_speaker_id']].append(row)
        queues = {key: [members[i] for i in rng.permutation(len(members))] for key, members in sorted(by_speaker.items())}
        members = []
        while len(members) < per_group:
            for key in queues:
                if queues[key] and len(members) < per_group:
                    members.append(queues[key].pop())
        selected.extend(members)
    real = sorted((row for row in candidates if row['label'] == 0), key=lambda row: row['sample_id'])
    if len(real) < len(selected):
        raise ValueError("Insufficient same-language REAL coverage")
    selected.extend(real[i] for i in sorted(rng.choice(len(real), len(selected), replace=False)))
    return {"scope": "PUBLIC_DEVELOPMENT", "revision": revision, "seed": seed, "language": language,
            "groups": chosen, "per_fake_group": per_group,
            "policy": "Alphabetical eligible author model names unless explicit; seeded source-speaker round-robin FAKE and same-language REAL; no performance input",
            "sample_size_rationale": "24/group default gives worst-case independent-binomial 95% half-width ~0.20; 96/class ~0.10. Source clustering can widen uncertainty: exploratory only",
            "records": sorted(selected, key=lambda row: row['sample_id'])}


def confounding_audit(records: list[PublicSourceRecord]) -> dict:
    result = {}
    for field in ('language', 'source_corpus', 'sample_rate', 'generator_id', 'source_speaker_id'):
        counts = defaultdict(Counter)
        for row in records:
            counts[str(getattr(row, field))][str(row.label)] += 1
        result[field] = {key: dict(value) for key, value in sorted(counts.items())}
    result['language_label_complete_separation'] = all(len(labels) == 1 for labels in result['language'].values())
    result['source_corpus_label_complete_separation'] = all(len(labels) == 1 for labels in result['source_corpus'].values())
    result['source_identity_unknown_real'] = sum(row.label == 0 and not known(row.source_speaker_id) for row in records)
    result['checkpoint_relative_familiarity'] = UNKNOWN
    result['limitations'] = ['source_corpus is source-text/original provenance, not identical recording environment',
                             'REAL speaker/parent lineage not supplied by tiny audiofolder',
                             'FAKE source_speaker is original text speaker, not verified synthesized/target voice']
    return result


def import_mlaad(discovery: Path, audio_root: Path, selection: dict, output: Path) -> dict:
    release = json.loads((discovery / 'discovery.json').read_text())
    acquisition = json.loads((audio_root / 'acquisition.json').read_text())
    license_gate(discovery, purpose='noncommercial_research')
    if selection['scope'] != 'PUBLIC_DEVELOPMENT' or acquisition['status'] != 'ACQUIRED_VERIFIED' or \
            acquisition['selection_hash'] != object_hash(selection) or release['revision'] != selection['revision'] or \
            acquisition['revision'] != selection['revision'] or acquisition['repository'] != release['repository']:
        raise ValueError('Acquisition/release/selection provenance gate failed')
    all_records, summary = metadata_inventory(discovery)
    official = {row['sample_id']: row for row in all_records}
    downloaded = {row['path']: row for row in acquisition['files']}
    if len(downloaded) != len(acquisition['files']) or set(downloaded) != {row['path'] for row in selection['records']}:
        raise ValueError('Acquisition is not the exact selected cohort')
    sources = []
    for row in selection['records']:
        if row != official.get(row['sample_id']):
            raise ValueError('Selection annotation changed from authoritative metadata')
        path = (audio_root / safe_relative(row['path'])).resolve()
        if not path.is_relative_to(audio_root.resolve()):
            raise ValueError('Audio path escaped root')
        digest = file_hash(path)
        if digest != row['expected_sha256'] or digest != downloaded[row['path']]['checksum']:
            raise ValueError('Audio checksum mismatch')
        values, rate = sf.read(path, dtype='float32', always_2d=True)
        if not values.size or not np.isfinite(values).all():
            raise ValueError('Invalid audio')
        parent = 'M-AILABS:' + row['source_recording_id'] if known(row['source_recording_id']) else 'local-sha256:' + digest
        sources.append(PublicSourceRecord(sample_id=row['sample_id'], path=str(path), parent_recording_id=parent,
            dataset_id='MLAAD_TINY', dataset_version=selection['revision'], split='development', label=row['label'],
            checksum=digest, sample_rate=rate, duration=len(values)/rate, source_speaker_id=row['source_speaker_id'],
            generator_id=row['generator_id'], attack_id=row['attack_id'], language=row['language'],
            source_url_or_record=f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/tree/{selection['revision']}",
            license='CC-BY-NC-4.0' if row['label'] else 'M-AILABS original notice',
            provenance_status=row['annotation_status'], parent_scope='AUTHOR_ORIGINAL_FILE' if row['label'] else 'LOCAL_BYTES_ONLY',
            official_split=UNKNOWN, research_split='development', source_recording_id=row['source_recording_id'],
            provenance_source=row['annotation_path'], provenance_confidence='AUTHOR_FIELDS; ORIGINAL_AUDIOFOLDER_LINEAGE_UNKNOWN',
            official_recording_id=row['path'], source_revision=selection['revision'], generator_family=row['generator_family'],
            source_corpus=row['source_corpus']))
    if {row.label for row in sources} != {0,1} or len({row.language for row in sources}) != 1:
        raise ValueError('Same-language two-class development required')
    partition = audit_partitions(sources)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'provenance_manifest.csv', [row.to_dict() for row in sources])
    write_json(output/'selection.json', selection)
    write_json(output/'source_metadata.json', {**summary, 'status':'IMPORTED_REAL_PUBLIC_AUDIO', 'license':acquisition['license'],
        'acquisition_sha256':file_hash(audio_root/'acquisition.json'), 'discovery_sha256':file_hash(discovery/'discovery.json'),
        'selected_class_counts':dict(Counter(str(row.label) for row in sources)),
        'manifest_sha256':file_hash(output/'provenance_manifest.csv')})
    write_json(output/'partition_audit.json', partition)
    write_json(output/'confounding_audit.json', confounding_audit(sources))
    return {'status':'IMPORTED_REAL_PUBLIC_AUDIO', 'records':len(sources), 'partition':partition}
