"""Storage-safe official LA ZIP inspection/import; evaluation remains unopened."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path,PurePosixPath
import json
import shutil
import stat
import zipfile

import numpy as np
import soundfile as sf

from .datasets import ASVspoof2019LA,PublicSourceRecord
from .evidence import file_hash,write_json,write_csv
from .provenance import audit_partitions,UNKNOWN

RESERVE_BYTES=15*1024**3


def safe_member(name: str) -> str:
    path=PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe ZIP path')
    return path.as_posix()


def inspect_archive(archive: Path) -> dict:
    totals=defaultdict(lambda:dict(files=0,expanded_bytes=0,compressed_bytes=0,audio_files=0))
    seen=set()
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            name=safe_member(item.filename)
            if name in seen or item.flag_bits&1 or stat.S_ISLNK(item.external_attr>>16):
                raise ValueError('Duplicate/encrypted/symlink archive entry')
            seen.add(name)
            if item.is_dir():
                continue
            parts=PurePosixPath(name).parts
            if not parts or parts[0]!='LA':
                raise ValueError('Expected official LA top-level directory')
            key=parts[1] if len(parts)>1 else 'LA'
            total=totals[key]; total['files']+=1; total['expanded_bytes']+=item.file_size
            total['compressed_bytes']+=item.compress_size; total['audio_files']+=name.endswith('.flac')
    selected=sum(total['expanded_bytes'] for name,total in totals.items() if name in {'ASVspoof2019_LA_train','ASVspoof2019_LA_dev'})
    return {'archive':str(archive.resolve()),'archive_bytes':archive.stat().st_size,'partitions':dict(totals),
            'total_expanded_bytes':sum(x['expanded_bytes'] for x in totals.values()),'train_dev_expanded_bytes':selected,
            'free_disk_bytes':shutil.disk_usage(archive.parent).free,'reserve_bytes':RESERVE_BYTES,
            'storage_gate':'PASS' if shutil.disk_usage(archive.parent).free>selected+RESERVE_BYTES+512*1024**2 else 'DATA_STORAGE_BLOCKED',
            'evaluation_labels_read':False,'evaluation_audio_read':False}


def extract_train_dev(archive: Path,destination: Path,output: Path) -> dict:
    report=inspect_archive(archive)
    if report['storage_gate']!='PASS':
        raise ValueError('DATA_STORAGE_BLOCKED')
    if destination.exists() or output.exists():
        raise FileExistsError('New extraction and audit directories required')
    output.mkdir(parents=True); destination.mkdir(parents=True)
    report.update(status='EXTRACTING',archive_sha256=file_hash(archive),archive_authenticity='USER_PROVIDED; OFFICIAL_REMOTE_DIGEST_NOT_VERIFIED')
    write_json(output/'archive_inventory.json',report)
    selected=[]
    protocols={'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
               'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt','LA/README.LA.txt'}
    try:
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                name=safe_member(item.filename)
                if item.is_dir() or not (name in protocols or name.startswith(('LA/ASVspoof2019_LA_train/','LA/ASVspoof2019_LA_dev/'))):
                    continue
                target=(destination/name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError('ZIP destination escaped root')
                target.parent.mkdir(parents=True,exist_ok=True)
                with source.open(item) as src,target.open('xb') as dst:
                    shutil.copyfileobj(src,dst,1024*1024)  # zipfile checks CRC at EOF
                if target.stat().st_size!=item.file_size:
                    raise ValueError('Extracted length mismatch')
                selected.append({'archive_member':name,'bytes':item.file_size,'crc32':f'{item.CRC:08x}'})
                if len(selected)%5000==0:
                    print(f'Extracted {len(selected)} selected files; evaluation unopened',flush=True)
        report.update(status='EXTRACTED_TRAIN_DEV',extracted_files=len(selected),extracted_bytes=sum(r['bytes'] for r in selected),
                      free_disk_after=shutil.disk_usage(destination).free)
        write_csv(output/'extracted_members.csv',selected)
        write_json(output/'archive_inventory.json',report)
        return report
    except Exception as error:
        report.update(status='FAILED_PARTIAL',error=f'{type(error).__name__}: {error}')
        write_json(output/'archive_inventory.json',report); raise


def build_foundation(dataset_root: Path,output: Path,archive_audit: Path,*,seed: int=5804) -> dict:
    if output.exists():
        raise FileExistsError('New foundation output required')
    extraction=json.loads((archive_audit/'archive_inventory.json').read_text())
    if extraction['status']!='EXTRACTED_TRAIN_DEV':
        raise ValueError('Verified extraction required')
    adapter=ASVspoof2019LA(); protocols=dataset_root/'ASVspoof2019_LA_cm_protocols'
    train=adapter.parse(protocols/'ASVspoof2019.LA.cm.train.trn.txt','train')
    dev=adapter.parse(protocols/'ASVspoof2019.LA.cm.dev.trl.txt','dev')
    train_speakers={r.speaker_id for r in train}; dev_speakers={r.speaker_id for r in dev}
    if train_speakers&dev_speakers:
        raise ValueError('Official train/dev speaker leakage')
    rng=np.random.default_rng(seed)
    ordered=sorted(dev_speakers)
    calibration={ordered[i] for i in rng.choice(len(ordered),max(2,len(ordered)//5),replace=False)}
    records=[]
    for entry in train+dev:
        path=adapter.resolve(dataset_root.resolve(),entry); info=sf.info(path); digest=file_hash(path)
        partition='train' if entry.official_split=='train' else 'calibration' if entry.speaker_id in calibration else 'development'
        records.append(PublicSourceRecord(sample_id='ASVspoof2019_LA:'+entry.recording_id,path=str(path),
            parent_recording_id='local-sha256:'+digest,dataset_id=adapter.dataset_id,dataset_version='2019-user-archive-'+extraction['archive_sha256'][:12],
            split=partition,label=entry.label,checksum=digest,sample_rate=info.samplerate,duration=info.duration,
            speaker_id=entry.speaker_id,target_speaker_id=entry.speaker_id,attack_id=entry.attack_id,
            official_split=entry.official_split,research_split=partition,official_recording_id=entry.recording_id,
            language='en',license=adapter.license,source_url_or_record=adapter.source_url,
            source_revision=extraction['archive_sha256'],source_corpus='ASVspoof2019_LA/VCTK',
            provenance_status='OFFICIAL_CM_PROTOCOL_LOCAL_CRC_SHA256',parent_scope='LOCAL_BYTES_ONLY',
            provenance_source=str(protocols/('ASVspoof2019.LA.cm.train.trn.txt' if entry.official_split=='train' else 'ASVspoof2019.LA.cm.dev.trl.txt'))))
        if len(records)%10000==0:
            print(f'Hashed/audio-header-validated {len(records)}/{len(train)+len(dev)} records',flush=True)
    partition_audit=audit_partitions(records,require_speaker_disjoint=True)
    output.mkdir(parents=True)
    for split in ('train','development','calibration'):
        members=[r for r in records if r.split==split]
        if {r.label for r in members}!={0,1}:
            raise ValueError('Both classes required per research partition')
        write_csv(output/(split+'_manifest.csv'),[r.to_dict() for r in members])
    write_json(output/'partition_audit.json',partition_audit)
    report={'status':'CONTROLLED_TRAIN_DEV_FOUNDATION','archive_sha256':extraction['archive_sha256'],
            'counts':{s:dict(recordings=sum(r.split==s for r in records),class_counts=dict(Counter(str(r.label) for r in records if r.split==s)),
                           speakers=len({r.speaker_id for r in records if r.split==s})) for s in ('train','development','calibration')},
            'manifest_hashes':{s:file_hash(output/(s+'_manifest.csv')) for s in ('train','development','calibration')},
            'calibration_speakers':sorted(calibration),'partition_seed':seed,'evaluation_status':'SEALED_IN_ARCHIVE; NOT_EXTRACTED_OR_PARSED',
            'roles':{'train':'TRAIN','development':'MODEL_SELECTION','calibration':'CALIBRATION_RESERVED','MLAAD_tiny':'EXPLORATORY_DEVELOPMENT_ALREADY_USED',
                     'ASVspoof2019_eval':'SEALED_EVALUATION_NOT_CONFIRMATION_LOCKED','RADAR_dev':'METADATA_INSPECTED_NO_AUDIO'},
            'limitations':['Source/parent content and exact generator identity UNKNOWN','Official speaker IDs are not full source-utterance lineage',
                           'SHA256 and CRC establish local integrity; official remote archive checksum not verified','Decoded/perceptual duplicate screening not exhaustive']}
    write_json(output/'foundation.json',report); return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inspect','extract','import'))
    parser.add_argument('--archive',type=Path,default=Path('data/downloads/LA.zip'))
    parser.add_argument('--destination',type=Path,default=Path('data/public/asvspoof2019_la_v1'))
    parser.add_argument('--audit',type=Path,default=Path('outputs/jaes_v2/datasets/asv2019_archive_v1'))
    parser.add_argument('--output',type=Path,default=Path('outputs/jaes_v2/datasets/asv2019_foundation_v1'))
    args=parser.parse_args()
    if args.mode=='inspect': print(json.dumps(inspect_archive(args.archive),indent=2))
    elif args.mode=='extract': print(json.dumps(extract_train_dev(args.archive,args.destination,args.audit),indent=2))
    else: print(json.dumps(build_foundation(args.destination/'LA',args.output,args.audit),indent=2))


if __name__=='__main__':
    main()
