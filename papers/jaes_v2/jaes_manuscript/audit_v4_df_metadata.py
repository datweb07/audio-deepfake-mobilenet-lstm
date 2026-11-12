"""Read official 2019/2021 utterance maps without opening any audio."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
import tarfile


def members(archive, name):
    with TextIOWrapper(archive.extractfile(name), encoding='utf-8-sig') as stream:
        yield from csv.DictReader(stream,delimiter='\t')


def lineage(row, corpus):
    direct = row[f'{corpus}_ID']
    vc = row[f'VC_source_{corpus}_ID']
    target = row['TTS_VC_target_speaker']
    parents = {v for v in (direct,vc) if v and v!='-'}
    speakers = {v.split('_',1)[0] for v in parents}
    if target and target!='-':
        speakers.add(target)
    return parents,speakers,('REAL' if direct and direct!='-' else 'FAKE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('maps',type=Path)
    parser.add_argument('--keys',type=Path)
    args=parser.parse_args()
    root=Path.cwd()
    base=root/'outputs/jaes_v2'
    medium=base/'phase7/selections/medium_all_attack_v1/manifest.csv'
    held=base/'phase10/empirical_execution_v1/heldout_28f940ea7d151177/manifest.json'
    with medium.open(encoding='utf8',newline='') as stream:
        groups={group:set() for group in ('train','development')}
        for row in csv.DictReader(stream):
            groups[row['research_split']].add(row['official_recording_id'])
    groups['heldout']={r['sample_id'] for r in json.loads(held.read_text(encoding='utf8'))['rows']}
    groups['heldout']={s.rsplit(':',1)[-1] for s in groups['heldout']}
    with tarfile.open(args.maps) as archive:
        la={r['ASVspoof_ID']:lineage(r,'VCTK') for r in members(archive,'ASVspoof2019_LA_VCTK_MetaInfo.tsv')}
        fingerprints={}
        for group,ids in groups.items():
            hits=[la[s] for s in sorted(ids) if s in la]
            fingerprints[group]=dict(total=len(ids),mapped=len(hits),
                parents=set().union(*(r[0] for r in hits)),speakers=set().union(*(r[1] for r in hits)))
        keys={}
        if args.keys:
            with tarfile.open(args.keys) as package:
                trial=next(n for n in package.getnames() if n.endswith('/CM/trial_metadata.txt'))
                with TextIOWrapper(package.extractfile(trial),encoding='utf8') as stream:
                    for line in stream:
                        speaker,sample,codec,source,attack,label,trim,subset,vocoder,*_=line.split()
                        if sample in keys:
                            raise ValueError(f'Duplicate official DF key {sample}')
                        keys[sample]=(label,codec,subset,source)
        analyses={}
        for corpus in ('VCTK','VCC'):
            rows={r['ASVspoof_ID']:lineage(r,corpus) for r in members(archive,f'ASVspoof2021_DF_{corpus}_MetaInfo.tsv')}
            overlap={}
            for group,registered in fingerprints.items():
                overlap[group]=dict(registered=registered['total'],mapped=registered['mapped'],
                    df_rows_with_common_parent=sum(bool(r[0]&registered['parents']) for r in rows.values()),
                    df_rows_with_common_speaker=sum(bool(r[1]&registered['speakers']) for r in rows.values()))
            analyses[corpus]=dict(mapped_df_rows=len(rows),counts=Counter(r[2] for r in rows.values()),
                overlap_with_2019_in_comparable_namespace=overlap if corpus=='VCTK' else None,
                df_key_joined=len(rows.keys()&keys.keys()) if keys else None,
                df_key_eval_support=Counter('/'.join(keys[s][:2]) for s in rows.keys()&keys.keys()
                                            if keys[s][2]=='eval') if keys else None)
        license_bytes=archive.extractfile('LICENSE.txt').read()
    output=base/'supervisor_major_revision/df_metadata_audit.json'
    output.write_text(json.dumps(dict(status='METADATA_ONLY',map_sha256=sha256(args.maps.read_bytes()).hexdigest(),
        keys_sha256=sha256(args.keys.read_bytes()).hexdigest() if args.keys else None,keys_rows=len(keys),
        metadata_license='ODC-By 1.0',metadata_license_sha256=sha256(license_bytes).hexdigest(),corpora=analyses,
        limitations=['TTS rows can lack parent audio IDs','VCC/VCTK namespaces cannot be joined by string identity',
                     'No audio-byte or decoded-waveform overlap audit']),indent=2)+'\n',encoding='utf8')
    print(output)


if __name__=='__main__':
    main()
