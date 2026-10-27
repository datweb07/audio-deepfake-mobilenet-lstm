"""Inspect official alternate releases without downloading audio or evaluation labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

from src.lava.jaes_v2.evidence import file_hash, write_json


def get(url, maximum=8_000_000):
    with urllib.request.urlopen(url,timeout=40) as response:
        payload = response.read(maximum+1)
    if len(payload)>maximum:
        raise ValueError('Metadata byte budget exceeded')
    return payload


def main():
    output = Path('outputs/jaes_v2/datasets/phase3_alternative_discovery_v1')
    output.mkdir(parents=True,exist_ok=False)
    report = {'audio_bytes':0,'evaluation_labels_downloaded':False}
    try:
        release = json.loads(get('https://zenodo.org/api/records/22110815'))
        write_json(output/'radar_release.json',release)
        files = release['files']
        selected = {'README.txt','LICENSE.txt','label_RADAR2026-dev.txt','lang_RADAR2026-dev.txt'}
        for item in files:
            if item['key'] not in selected:
                continue
            payload = get(item['links']['self'])
            if len(payload)!=item['size'] or 'md5:'+hashlib.md5(payload).hexdigest()!=item['checksum']:
                raise ValueError('Official RADAR file checksum mismatch')
            (output/item['key']).write_bytes(payload)
        label_counts = Counter(line.split()[2] for line in (output/'label_RADAR2026-dev.txt').read_text().splitlines() if line.strip())
        languages = Counter(line.split()[1] for line in (output/'lang_RADAR2026-dev.txt').read_text().splitlines() if line.strip())
        report['radar'] = {'status':'METADATA_VERIFIED','record_id':release['id'],'version':release['metadata'].get('version','UNKNOWN'),
                           'class_counts':dict(label_counts),'languages':dict(languages),
                           'files':[{k:item[k] for k in ('key','size','checksum')} for item in files],
                           'generator_per_record':'UNKNOWN','paired_clean_per_record':'UNKNOWN',
                           'source':'https://zenodo.org/records/22110815'}
    except Exception as error:
        report['radar'] = {'status':'BLOCKED','error':f'{type(error).__name__}: {error}'}
    try:
        url = 'https://datashare.ed.ac.uk/server/api/core/bitstreams/765597d6-633f-418a-96b0-39274a8d56c9/content'
        payload = get(url,100_000)
        (output/'asv2019_README.txt').write_bytes(payload)
        report['asvspoof'] = {'status':'README_ACQUIRED','bytes':len(payload),'sha256':file_hash(output/'asv2019_README.txt'),'url':url}
    except Exception as error:
        report['asvspoof'] = {'status':'BLOCKED','error':f'{type(error).__name__}: {error}','audio_bytes':0}
    write_json(output/'discovery.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
