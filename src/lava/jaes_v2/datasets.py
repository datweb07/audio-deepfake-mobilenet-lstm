"""Official-protocol adapters and explicit public-data provenance (no automatic download)."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re
from typing import Protocol

import soundfile as sf

from .evidence import file_hash, object_hash, read_csv, write_csv, write_json
from .provenance import SourceRecord, UNKNOWN, audit_partitions


@dataclass(frozen=True)
class ProtocolEntry:
    recording_id: str
    speaker_id: str
    attack_id: str
    label: int
    official_split: str
    protocol_line: int


@dataclass(frozen=True)
class PublicSourceRecord(SourceRecord):
    official_split: str = UNKNOWN
    research_split: str = UNKNOWN
    source_recording_id: str = UNKNOWN
    provenance_source: str = UNKNOWN
    provenance_confidence: str = "OFFICIAL_FIELDS_ONLY"
    official_recording_id: str = UNKNOWN
    attack_familiarity: str = UNKNOWN
    source_revision: str = UNKNOWN
    generator_family: str = UNKNOWN
    source_corpus: str = UNKNOWN


class DatasetAdapter(Protocol):
    dataset_id: str
    source_url: str
    license: str
    def parse(self, protocol: Path, official_split: str) -> list[ProtocolEntry]: ...
    def resolve(self, root: Path, entry: ProtocolEntry) -> Path: ...


class ASVspoof2019LA:
    dataset_id = "ASVspoof2019_LA"
    source_url = "https://datashare.ed.ac.uk/handle/10283/3336"
    license = "ODC-By-1.0 (database); retain LICENSE_text.txt and upstream content conditions"
    split_mapping = {"train": "train", "dev": "development", "eval": "test"}

    def parse(self, protocol: Path, official_split: str) -> list[ProtocolEntry]:
        if official_split not in self.split_mapping:
            raise ValueError("Official split must be train/dev/eval")
        entries, identifiers = [], set()
        prefixes = {"train": "LA_T_", "dev": "LA_D_", "eval": "LA_E_"}
        for line_number, line in enumerate(protocol.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"CM protocol line {line_number}: expected speaker recording unused attack key")
            speaker, recording, unused, attack, label = parts
            if not re.fullmatch(r"LA_[TDE]_\d+", recording) or not recording.startswith(prefixes[official_split]):
                raise ValueError(f"Recording/partition mismatch: {recording}")
            if not re.fullmatch(r"LA_\d+", speaker) or unused != "-":
                raise ValueError("Unsupported official LA CM schema")
            if label not in {"bonafide", "spoof"}:
                raise ValueError(f"Unknown official label: {label}")
            if (label == "bonafide" and attack != "-") or (label == "spoof" and not re.fullmatch(r"A\d{2}", attack)):
                raise ValueError("Inconsistent attack/key")
            if recording in identifiers:
                raise ValueError("Duplicate official recording ID")
            identifiers.add(recording)
            entries.append(ProtocolEntry(recording, speaker, attack if label == "spoof" else UNKNOWN,
                                         int(label == "spoof"), official_split, line_number))
        if not entries or {entry.label for entry in entries} != {0, 1}:
            raise ValueError("Official protocol must include REAL and FAKE coverage")
        return entries

    def resolve(self, root: Path, entry: ProtocolEntry) -> Path:
        # dataset-root is the extracted LA directory; no filename-based speaker inference.
        return root / f"ASVspoof2019_LA_{entry.official_split}" / "flac" / f"{entry.recording_id}.flac"


ADAPTERS = {"asvspoof2019_la": ASVspoof2019LA}


def load_public_manifest(path: Path) -> list[PublicSourceRecord]:
    records = [PublicSourceRecord(**{**row, "label": int(row["label"]), "sample_rate": int(row["sample_rate"]),
                                    "duration": float(row["duration"])}) for row in read_csv(path)]
    audit_partitions(records)
    for record in records:
        if record.research_split != record.split:
            raise ValueError("Research partition fields disagree")
    return records


def import_public(adapter: DatasetAdapter, *, dataset_root: Path, protocol: Path, official_split: str,
                  output: Path, metadata_only: bool = False, selected_ids: set[str] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    entries = adapter.parse(protocol, official_split)
    if selected_ids is not None:
        if not selected_ids or selected_ids - {entry.recording_id for entry in entries}:
            raise ValueError("Selection contains IDs outside official protocol")
        entries = [entry for entry in entries if entry.recording_id in selected_ids]
        if {entry.label for entry in entries} != {0, 1}:
            raise ValueError("Selected cohort must retain both classes")
    inventory, records = [], []
    for entry in entries:
        path = adapter.resolve(dataset_root.resolve(), entry)
        if not path.resolve().is_relative_to(dataset_root.resolve()):
            raise ValueError("Audio path escaped dataset root")
        exists = path.is_file()
        inventory.append({**asdict(entry), "path": str(path), "status": "AVAILABLE" if exists else "MISSING_AUDIO"})
        if exists and not metadata_only:
            info = sf.info(path)
            digest = file_hash(path)
            # A stable local variant key is explicit; true source utterance lineage stays UNKNOWN.
            records.append(PublicSourceRecord(
                sample_id=f"{adapter.dataset_id}:{entry.recording_id}", path=str(path),
                parent_recording_id="local-sha256:" + digest, parent_scope="LOCAL_BYTES_ONLY",
                dataset_id=adapter.dataset_id, dataset_version="2019-official-release",
                split=ASVspoof2019LA.split_mapping[official_split], label=entry.label, checksum=digest,
                sample_rate=info.samplerate, duration=info.duration, speaker_id=entry.speaker_id,
                target_speaker_id=entry.speaker_id, attack_id=entry.attack_id, generator_id=UNKNOWN,
                language="en", source_url_or_record=f"{adapter.source_url}#protocol-line-{entry.protocol_line}",
                license=adapter.license, provenance_status="OFFICIAL_PROTOCOL_LOCAL_AUDIO",
                official_split=official_split, research_split=ASVspoof2019LA.split_mapping[official_split],
                provenance_source=f"{protocol.resolve()}#L{entry.protocol_line}", official_recording_id=entry.recording_id))
    missing = sum(row["status"] == "MISSING_AUDIO" for row in inventory)
    status = "METADATA_ONLY" if metadata_only else "MISSING_DATA" if missing else "IMPORTED"
    metadata = {"dataset_id": adapter.dataset_id, "source_url": adapter.source_url, "license": adapter.license,
                "protocol_path": str(protocol.resolve()), "protocol_sha256": file_hash(protocol),
                "status": status, "official_split": official_split, "research_split": ASVspoof2019LA.split_mapping[official_split],
                "protocol_records_selected": len(entries), "class_counts": dict(Counter(str(entry.label) for entry in entries)),
                "audio_available": len(entries) - missing, "audio_missing": missing,
                "audio_downloaded_by_importer": 0, "selection": sorted(selected_ids) if selected_ids else "FULL_PROTOCOL",
                "limitations": ["Attack ID is an official condition, not an exact verified generator identity",
                                "Source-speaker and source-recording lineage are UNKNOWN", "Legacy model training overlap is UNKNOWN"]}
    write_csv(output / "source_inventory.csv", inventory)
    write_json(output / "source_metadata.json", metadata)
    if metadata_only or missing:
        write_csv(output / "provenance_manifest.csv", [], [field.name for field in fields(PublicSourceRecord)])
        write_json(output / "integrity_report.json", {"status": status, "verified_audio_records": 0,
                                                     "reason": "Metadata-only or missing selected audio; no partial benchmark cohort published"})
        write_json(output / "partition_audit.json", {"status": "NOT_RUN", "reason": status})
        if missing and not metadata_only:
            raise FileNotFoundError(f"Missing {missing} selected audio files; inventory preserved; no partial cohort")
        return metadata
    partition = audit_partitions(records)
    from .data_audit import audit_sources
    integrity, _ = audit_sources(dataset_root, records)
    write_csv(output / "provenance_manifest.csv", [record.to_dict() for record in records])
    write_json(output / "integrity_report.json", integrity)
    write_json(output / "partition_audit.json", partition)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=ADAPTERS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--official-split", choices=["train", "dev", "eval"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="One exact official recording ID per line")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    selected = set(args.selection.read_text().split()) if args.selection else None
    report = import_public(ADAPTERS[args.dataset](), dataset_root=args.dataset_root, protocol=args.protocol,
                           official_split=args.official_split, output=args.output, metadata_only=args.metadata_only, selected_ids=selected)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
