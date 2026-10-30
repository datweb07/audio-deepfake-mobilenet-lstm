"""Controlled ASV train/dev selection and shared paired-view embedding cache."""
from __future__ import annotations

from collections import defaultdict,Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .controls import NoisePlan,prepare_headroom
from .datasets import load_public_manifest
from .evidence import file_hash,object_hash,read_csv,write_csv,write_json
from .preprocessing import Preprocessor
from .provenance import audit_partitions
from .transforms import Audio,Transform,load_audio

TRAIN_VIEWS=('clean','gain_half','noise10','opus24')
DEV_VIEWS=('clean','gain_quarter','noise5','opus64','A_then_B','B_then_A')


def select_balanced(records: list, *, per_class: int, seed: int, role: str) -> list:
    if role not in {'train','development'} or per_class<4 or any(r.split!=role for r in records):
        raise ValueError('Only explicit official train or model-selection development is allowed')
    rng=np.random.default_rng(seed); selected=[]
    for label in (0,1):
        cells=defaultdict(lambda:defaultdict(list))
        for row in sorted(records,key=lambda r:r.sample_id):
            if row.label==label:
                cells[row.speaker_id][row.attack_id].append(row)
        queues={}
        for index,(speaker,attacks) in enumerate(sorted(cells.items())):
            attack_order=sorted(attacks)
            attack_order=attack_order[index%len(attack_order):]+attack_order[:index%len(attack_order)]
            pending={attack:[attacks[attack][i] for i in rng.permutation(len(attacks[attack]))] for attack in attack_order}
            queue=[]
            while any(pending.values()):
                for values in pending.values():
                    if values:
                        queue.append(values.pop())
            queues[speaker]=list(reversed(queue))
        if sum(map(len,queues.values()))<per_class:
            raise ValueError('Insufficient official class coverage')
        count=0
        while count<per_class:
            for values in queues.values():
                if values and count<per_class:
                    selected.append(values.pop()); count+=1
    return sorted(selected,key=lambda r:r.sample_id)


def paired_views(audio: Audio,sample_id: str,role: str,ffmpeg: str) -> tuple[dict,dict]:
    if role not in {'train','development'}:
        raise ValueError('No calibration/evaluation augmentation or pair generation')
    clean,headroom=prepare_headroom(audio,.25)
    seed=int(object_hash({'sample_id':sample_id,'seed':5804,'version':'c2-view-1'})[:16],16)
    views={'clean':clean}; evidence={'clean':headroom}
    if role=='train':
        views['gain_half']=Audio(clean.samples*.5,clean.sample_rate)
        plan=NoisePlan.build(clean,target_snr_db=10,seed=seed)
        views['noise10'],evidence['noise10']=plan.insert(clean,'fixed_absolute')
        codec=Transform('opus24','codec',{'encoder':'libopus','bitrate':'24k','extension':'.ogg'})
        views['opus24'],evidence['opus24']=codec.apply(clean,seed=0,ffmpeg=ffmpeg)
    else:
        views['gain_quarter']=Audio(clean.samples*.25,clean.sample_rate)
        noisy5=NoisePlan.build(clean,target_snr_db=5,seed=seed)
        views['noise5'],evidence['noise5']=noisy5.insert(clean,'fixed_absolute')
        codec64=Transform('opus64','codec',{'encoder':'libopus','bitrate':'64k','extension':'.ogg'})
        views['opus64'],evidence['opus64']=codec64.apply(clean,seed=0,ffmpeg=ffmpeg)
        noise=NoisePlan.build(clean,target_snr_db=10,seed=seed)
        codec=Transform('opus24','codec',{'encoder':'libopus','bitrate':'24k','extension':'.ogg'})
        noisy,ne=noise.insert(clean,'fixed_absolute'); coded,ce=codec.apply(clean,seed=0,ffmpeg=ffmpeg)
        views['A_then_B'],evidence['A_then_B']=codec.apply(noisy,seed=0,ffmpeg=ffmpeg)
        views['B_then_A'],evidence['B_then_A']=noise.insert(coded,'fixed_absolute')
    return views,{'sample_seed':seed,'reference_headroom':headroom,'conditions':evidence,
                  'policy':'same waveforms for both preprocessors; same fixed absolute noise per order; no post-channel normalization'}


def build_cache(root: Path,output: Path,*,train_per_class: int=64,dev_per_class: int=32,maximum_seconds: float=600) -> dict:
    from .runtime import preflight
    from .c2_student import build_student,initialize_tensorflow
    environment=preflight(root)
    if environment['free_disk_bytes']<15*1024**3+256*1024**2:
        raise ValueError('15 GiB reserve would be violated')
    foundation=root/'outputs/jaes_v2/datasets/asv2019_foundation_v1'
    report=json.loads((foundation/'foundation.json').read_text())
    selected=[]
    for role,count in (('train',train_per_class),('development',dev_per_class)):
        path=foundation/(role+'_manifest.csv')
        if file_hash(path)!=report['manifest_hashes'][role]:
            raise ValueError('Foundation manifest changed')
        selected.extend(select_balanced(load_public_manifest(path),per_class=count,seed=5804,role=role))
    audit_partitions(selected,require_speaker_disjoint=True)
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'manifest.csv',[r.to_dict() for r in selected])
    write_json(output/'selection.json',{'seed':5804,'train_per_class':train_per_class,'dev_per_class':dev_per_class,
                                     'foundation_hashes':report['manifest_hashes'],'scope':'BOUNDED_TRAIN_DEV_PILOT','calibration_used':False,'evaluation_used':False})
    tf=initialize_tensorflow(); full,backbone,head,extractor=build_student(seed=5804,initialization='imagenet')
    arrays=defaultdict(list); entries=[]; diagnostics=[]; started=time.perf_counter()
    try:
        for source in selected:
            if time.perf_counter()-started>maximum_seconds:
                raise TimeoutError('Feature-cache wall-clock budget exceeded; incomplete cache cannot train')
            path=root/source.path
            if file_hash(path)!=source.checksum:
                raise ValueError('Source changed before feature preparation')
            audio=load_audio(path,sample_rate=22050,max_seconds=3)
            variants,evidence=paired_views(audio,source.sample_id,source.split,environment['ffmpeg'])
            for condition,variant in variants.items():
                waveform_sha=hashlib.sha256(variant.samples.astype('<f4').tobytes()).hexdigest()
                # Resource preflight measured batch two: pair both policies in one
                # frozen-backbone call rather than paying batch-one overhead twice.
                tensors=np.stack([Preprocessor(policy).extract(variant.samples) for policy in ('P0','P1')])
                vectors=extractor(tf.convert_to_tensor(tensors),training=False).numpy()
                for policy,vector in zip(('P0','P1'),vectors):
                    key=source.split+'_'+policy
                    arrays[key].append(vector)
                    entries.append({'sample_id':source.sample_id,'split':source.split,'label':source.label,'speaker_id':source.speaker_id,
                                    'attack_id':source.attack_id,'policy':policy,'condition':condition,'array_key':key,'array_index':len(arrays[key])-1,
                                    'waveform_sha256':waveform_sha,'source_checksum':source.checksum,'transform_seed':evidence['sample_seed']})
            if len(entries)%128==0:
                print(f'Prepared {len(entries)} train/dev embeddings; elapsed {time.perf_counter()-started:.1f}s',flush=True)
        np.savez_compressed(output/'embeddings.npz',**{key:np.stack(values) for key,values in arrays.items()})
        write_csv(output/'view_manifest.csv',entries)
        from .data_audit import audit_sources
        audit,features=audit_sources(root,selected)
        audit.pop('shortcut_baselines',None); audit['shortcut_baseline']='SEPARATELY_GROUPED_BY_RESEARCH_PARTITION'
        write_json(output/'source_shortcut_audit.json',audit); write_csv(output/'source_features.csv',features)
        from .shortcut import grouped_shortcuts
        from .nuisance_controls import signal_features
        from src.preprocessing import normalize_audio_length
        shortcuts={}
        for role in ('train','development'):
            subset=[r for r in selected if r.split==role]
            # Training-only shortcut diagnostic may use development-fold machinery,
            # but this temporary role alias is not a published data partition.
            diagnostic_manifest=[{**r.to_dict(),'split':'development'} for r in subset]
            original=[r for r in features if r['split']==role]
            prepared=[]
            for record in subset:
                audio=load_audio(root/record.path,sample_rate=22050,max_seconds=3)
                clean,_=prepare_headroom(audio,.25)
                prepared.append({'sample_id':record.sample_id,**signal_features(Audio(normalize_audio_length(clean.samples),22050))})
            shortcuts[role]={'original':grouped_shortcuts(original,diagnostic_manifest),
                             'actual_detector_input':grouped_shortcuts(prepared,diagnostic_manifest),
                             'scope':role.upper()+'_ONLY_GROUPED_SHORTCUT_DIAGNOSTIC; no calibration/eval access'}
        write_json(output/'grouped_shortcuts.json',shortcuts)
        from .runtime import source_fingerprint
        if time.perf_counter()-started>maximum_seconds:
            raise TimeoutError('Complete cache including audits exceeded its wall-clock budget')
        environment={**environment,'execution_policy':'Frozen ImageNet feature extraction on explicit ASV train/development only; separately versioned research training permitted'}
        state={'status':'SEALED_C2_FEATURE_CACHE','elapsed_seconds':time.perf_counter()-started,'arrays':{k:list(np.stack(v).shape) for k,v in arrays.items()},
               'code':source_fingerprint(root),'encoder_batch_size':2,'preprocessor_hashes':{p:Preprocessor(p).fingerprint for p in ('P0','P1')},
               'environment':environment,'pretraining':'Keras ImageNet MobileNetV3Small; frozen feature backbone, fresh temporal head',
               'pretraining_weight_sha256':file_hash(Path.home()/'.keras/models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5'),
               'selection_hash':file_hash(output/'manifest.csv'),'pairs_preserve_source':True,'calibration_accessed':False,'evaluation_accessed':False}
        state['output_hashes']={p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'cache_metadata.json',state); return state
    except BaseException as error:
        write_json(output/'cache_metadata.json',{'status':'FAILED_OR_INTERRUPTED','error':f'{type(error).__name__}: {error}',
                                               'elapsed_seconds':time.perf_counter()-started,'completed_view_rows':len(entries)})
        raise
