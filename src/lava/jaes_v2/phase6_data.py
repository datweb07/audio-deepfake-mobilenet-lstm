"""New speaker-balanced cohort and bounded finite-view cache; historical data stay immutable."""
from collections import defaultdict,Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import shutil
import numpy as np
import soundfile as sf
from .datasets import load_public_manifest
from .provenance import audit_partitions
from .evidence import read_csv,write_csv,write_json,file_hash,object_hash
from .phase3 import verify_seal
from .c2_data import paired_views
from .transforms import load_audio,Audio
from .preprocessing import Preprocessor
from .rawboost import rawboost,PARAMETERS

BASE=Path('outputs/jaes_v2/phase6')
TRAIN_CONDITIONS=('clean','gain_half','noise10','opus24','rawboost0','rawboost1')
DEV_CONDITIONS=('clean','gain_half','gain_quarter','noise5','opus64','A_then_B','B_then_A')


def speaker_select(records,*,per_cell,seed,role):
    if role not in ('train','development') or per_cell<1 or any(r.split!=role for r in records): raise ValueError('Explicit train/development only')
    groups=defaultdict(lambda:defaultdict(list))
    for r in sorted(records,key=lambda x:x.sample_id): groups[r.speaker_id][r.label].append(r)
    rng=np.random.default_rng(seed); selected=[]; coverage=[]
    for speaker,classes in sorted(groups.items()):
        coverage.append(dict(speaker_id=speaker,split=role,real=len(classes[0]),fake=len(classes[1]),
                             attacks=','.join(sorted({r.attack_id for r in classes[1]}))))
        if not classes[0] or not classes[1]: continue
        if min(len(classes[0]),len(classes[1]))<per_cell: raise ValueError('Requested balanced cell exceeds actual support')
        for label in (0,1):
            attacks=defaultdict(list)
            for r in classes[label]: attacks[r.attack_id].append(r)
            queues={k:[values[i] for i in rng.permutation(len(values))] for k,values in sorted(attacks.items())}
            count=0
            while count<per_cell:
                for queue in queues.values():
                    if queue and count<per_cell: selected.append(queue.pop()); count+=1
    if not selected: raise ValueError('No genuinely supported speaker/class cells')
    return sorted(selected,key=lambda r:r.sample_id),coverage


def select(root,*,per_cell=4,seed=5806):
    output=root/BASE/'data_selection_v1'; output.mkdir(parents=True,exist_ok=False)
    foundation=root/'outputs/jaes_v2/datasets/asv2019_foundation_v1'
    declared=json.loads((foundation/'foundation.json').read_text()); sources=[]; coverage=[]; hashes={}
    for role in ('train','development'):
        path=foundation/(role+'_manifest.csv'); hashes[role]=file_hash(path)
        if hashes[role]!=declared['manifest_hashes'][role]: raise ValueError('Foundation changed')
        selected,table=speaker_select(load_public_manifest(path),per_cell=per_cell,seed=seed,role=role)
        sources+=selected; coverage+=table
    audit=audit_partitions(sources,require_speaker_disjoint=True)
    decoded=defaultdict(list)
    for r in sources:
        path=root/r.path
        if file_hash(path)!=r.checksum: raise ValueError('Source bytes changed')
        x,sr=sf.read(path,dtype='float32',always_2d=True)
        if not np.isfinite(x).all(): raise ValueError('Nonfinite audio')
        key=hashlib.sha256(str((sr,x.shape)).encode()+x.astype('<f4').tobytes()).hexdigest(); decoded[key].append(r)
    if any(len({r.split for r in group})>1 or len({r.label for r in group})>1 for group in decoded.values()): raise ValueError('Decoded overlap conflict')
    write_csv(output/'manifest.csv',[r.to_dict() for r in sources]); write_csv(output/'speaker_class_contingency.csv',coverage)
    write_json(output/'partition_audit.json',audit)
    write_json(output/'selection.json',dict(status='LOCKED_SPEAKER_BALANCED_DEVELOPMENT',seed=seed,per_speaker_per_class=per_cell,
        foundation_hashes=hashes,counts={role:dict(n=sum(r.split==role for r in sources),speakers=len({r.speaker_id for r in sources if r.split==role}),
        real=sum(r.split==role and r.label==0 for r in sources),fake=sum(r.split==role and r.label==1 for r in sources),
        attacks=sorted({r.attack_id for r in sources if r.split==role and r.label==1})) for role in ('train','development')},
        exclusions='Development speakers without FAKE support excluded prospectively; not representative full official dev',
        unknown_lineage='Original utterance/generator checkpoint UNKNOWN',decoded_duplicate_groups=sum(len(v)>1 for v in decoded.values()),
        calibration_accessed=False,evaluation_accessed=False,manifest_sha256=file_hash(output/'manifest.csv')))
    return output


def build_cache(root,*,maximum_seconds=1100):
    from .c2_student import build_student,initialize_tensorflow
    from .runtime import preflight,source_fingerprint
    output=root/BASE/'feature_cache_v2'; selection=root/BASE/'data_selection_v1'
    locked=json.loads((selection/'selection.json').read_text())
    if file_hash(selection/'manifest.csv')!=locked['manifest_sha256']: raise ValueError('Selection changed')
    if shutil.disk_usage(root).free<15*1024**3+256*1024**2: raise ValueError('Disk reserve')
    environment=preflight(root); tf=initialize_tensorflow(); full,backbone,head,extractor=build_student(seed=5806)
    sources=load_public_manifest(selection/'manifest.csv'); output.mkdir(parents=True,exist_ok=False)
    arrays=defaultdict(list); rows=[]; transforms=[]; started=perf_counter()
    try:
        for i,r in enumerate(sources):
            if perf_counter()-started>maximum_seconds: raise TimeoutError('Phase6 cache budget exceeded')
            if file_hash(root/r.path)!=r.checksum: raise ValueError('Source changed')
            audio=load_audio(root/r.path,sample_rate=22050,max_seconds=3)
            views,evidence=paired_views(audio,r.sample_id,r.split,environment['ffmpeg'])
            if r.split=='train':
                for k in range(2): views[f'rawboost{k}']=Audio(rawboost(views['clean'].samples,seed=evidence['sample_seed']+k),22050)
            else: views['gain_half']=Audio(views['clean'].samples*.5,22050)
            for condition in TRAIN_CONDITIONS if r.split=='train' else DEV_CONDITIONS:
                wave=views[condition].samples; sha=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()
                tensors=np.stack([Preprocessor(p).extract(wave) for p in ('P0','P1')])
                vectors=extractor(tf.convert_to_tensor(tensors),training=False).numpy()
                for p,vector in zip(('P0','P1'),vectors):
                    key=r.split+'_'+p; index=len(arrays[key]); arrays[key].append(vector)
                    rows.append(dict(sample_id=r.sample_id,split=r.split,label=r.label,speaker_id=r.speaker_id,attack_id=r.attack_id,
                        policy=p,condition=condition,array_key=key,array_index=index,source_checksum=r.checksum,
                        transform_seed=evidence['sample_seed']+(int(condition[-1]) if condition.startswith('rawboost') else 0),waveform_sha256=sha))
                transforms.append(dict(sample_id=r.sample_id,condition=condition,peak=float(np.max(np.abs(wave))),
                    rms=float(np.sqrt(np.mean(wave.astype('float64')**2))),clipped_fraction=float(np.mean(np.abs(wave)>1)),samples=len(wave),waveform_sha256=sha))
            if (i+1)%16==0: print(f'Phase6 cache {i+1}/{len(sources)}, {len(rows)} embeddings, {perf_counter()-started:.1f}s',flush=True)
        np.savez_compressed(output/'embeddings.npz',**{k:np.stack(v) for k,v in arrays.items()})
        write_csv(output/'view_manifest.csv',rows); write_csv(output/'transform_diagnostics.csv',transforms)
        write_csv(output/'manifest.csv',[r.to_dict() for r in sources])
        from .c2_training import validate_views
        validate_views(rows)
        metadata=dict(status='SEALED_PHASE6_FEATURE_CACHE',scope='CONTROLLED_PILOT',code=source_fingerprint(root),environment=environment,
            elapsed_seconds=perf_counter()-started,selection_sha256=file_hash(selection/'manifest.csv'),seed=5806,
            augmentation='Finite deterministic bank, NOT online stochastic RawBoost; algorithm5, two realizations per train source, fs22050/maxF8000',
            rawboost_parameters=PARAMETERS,pretraining_weight_sha256=file_hash(Path.home()/'.keras/models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5'),
            calibration_accessed=False,evaluation_accessed=False,arrays={k:list(np.stack(v).shape) for k,v in arrays.items()})
        metadata['output_hashes']={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}; write_json(output/'cache_metadata.json',metadata)
        return output
    except BaseException as error:
        write_json(output/'cache_metadata.json',dict(status='FAILED_OR_INTERRUPTED',error=str(error),rows=len(rows),elapsed_seconds=perf_counter()-started)); raise
