"""Post-publication view/index audit, including cropped/transformed cross-partition duplicates."""
from collections import defaultdict
import json
import numpy as np
from .evidence import read_csv,write_json,file_hash
from .phase7_data import BASE,load_cache
from .phase6_data import TRAIN_CONDITIONS,DEV_CONDITIONS

def cross_partition_waveforms(rows):
    hashes=defaultdict(set)
    for r in rows: hashes[r['waveform_sha256']].add(r['split'])
    conflicts=sorted(h for h,roles in hashes.items() if len(roles)>1)
    if conflicts: raise ValueError('Cropped/transformed waveform overlap across partitions: '+str(conflicts[:3]))
    return len(hashes)

def audit(root,config):
    directory,metadata,records=load_cache(root,config);rows=read_csv(directory/'view_manifest.csv')
    index={(r['sample_id'],r['condition']):r for r in rows}
    expected={(r.sample_id,c) for r in records for c in (TRAIN_CONDITIONS if r.split=='train' else DEV_CONDITIONS)}
    if len(index)!=len(rows) or set(index)!=expected:raise ValueError('Cache missing/duplicate/extra view')
    source={r.sample_id:r for r in records};positions=defaultdict(set)
    with np.load(directory/'embeddings.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    for row in rows:
        r=source[row['sample_id']]
        if any(str(row[k])!=str(getattr(r,k)) for k in ('split','speaker_id','attack_id','label')) or row['source_checksum']!=r.checksum or row['policy']!='P0':raise ValueError('Cache/source metadata mismatch')
        key=row['array_key'];i=int(row['array_index']);array=arrays[key]
        if array.shape[1:]!=(6,576) or not 0<=i<len(array) or i in positions[key]:raise ValueError('Invalid/duplicate feature index')
        positions[key].add(i)
    for key,array in arrays.items():
        if positions[key]!=set(range(len(array))) or not np.isfinite(array).all():raise ValueError('Unreferenced/nonfinite feature rows')
    distinct=cross_partition_waveforms(rows)
    result=dict(status='PASS',views=len(rows),sources=len(records),unique_waveform_hashes=distinct,cropped_transformed_cross_partition_overlaps=0,
        cache_metadata_sha256=file_hash(directory/'cache_metadata.json'),view_manifest_sha256=file_hash(directory/'view_manifest.csv'),
        original_source_lineage='UNKNOWN; selected exact PCM/waveform checks do not recover missing ancestry',calibration_data_consumed=False,evaluation_data_consumed=False)
    path=root/BASE/'cache_partition_audit_v1.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,result);return result
