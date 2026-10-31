"""Versioned all-attack selection and resumable, atomically published P0 cache."""
from collections import defaultdict, Counter
from pathlib import Path
from time import perf_counter
import hashlib, json, os, shutil
import numpy as np
import soundfile as sf
from .evidence import file_hash, object_hash, read_csv, write_csv, write_json
from .datasets import load_public_manifest
from .provenance import audit_partitions
from .phase6_data import TRAIN_CONDITIONS, DEV_CONDITIONS

BASE=Path('outputs/jaes_v2/phase7')
ATTACKS=tuple(f'A{i:02}' for i in range(1,7))

def safe_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or not (root/p).resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe research path')
    return root/p

def all_attack_select(records,*,per_cell=16,seed=7107,role):
    if role not in ('train','development') or per_cell<6 or any(r.split!=role for r in records): raise ValueError('Explicit train/development only, at least six per class')
    groups=defaultdict(lambda:defaultdict(list))
    for r in sorted(records,key=lambda x:x.sample_id): groups[r.speaker_id][r.label].append(r)
    rng=np.random.default_rng(seed); selected=[]; support=[]; real_only=[]
    for ordinal,(speaker,classes) in enumerate(sorted(groups.items())):
        counts=Counter(r.attack_id for r in classes[1])
        support.append(dict(split=role,speaker_id=speaker,real=len(classes[0]),fake=len(classes[1]),**{a:counts[a] for a in ATTACKS}))
        if not classes[1]:
            if role=='development':
                if len(classes[0])<per_cell: raise ValueError('Insufficient REAL-only support')
                real_only.extend(classes[0][i] for i in rng.permutation(len(classes[0]))[:per_cell])
            continue
        if len(classes[0])<per_cell or set(counts)!=set(ATTACKS): raise ValueError('Incomplete speaker/class/attack support')
        selected.extend(classes[0][i] for i in rng.permutation(len(classes[0]))[:per_cell])
        # Rotating remainder assignment avoids always overrepresenting A01-A04.
        order=ATTACKS[ordinal%6:]+ATTACKS[:ordinal%6]
        for j,a in enumerate(order):
            n=per_cell//6+int(j<per_cell%6); cell=[r for r in classes[1] if r.attack_id==a]
            if len(cell)<n: raise ValueError('Insufficient per-attack support')
            selected.extend(cell[i] for i in rng.permutation(len(cell))[:n])
    return sorted(selected,key=lambda x:x.sample_id),sorted(real_only,key=lambda x:x.sample_id),support

def select(root,selection_id='medium_all_attack_v1',per_cell=16,seed=7107):
    out=safe_path(root,str(BASE/'selections'/selection_id)); out.mkdir(parents=True,exist_ok=False)
    foundation=root/'outputs/jaes_v2/datasets/asv2019_foundation_v1'; declared=json.loads((foundation/'foundation.json').read_text())
    selected=[]; real_only=[]; coverage=[]; hashes={}
    for role in ('train','development'):
        path=foundation/(role+'_manifest.csv'); hashes[role]=file_hash(path)
        if hashes[role]!=declared['manifest_hashes'][role]: raise ValueError('Foundation changed')
        a,b,c=all_attack_select(load_public_manifest(path),per_cell=per_cell,seed=seed,role=role)
        selected+=a; real_only+=b; coverage+=c
    records=selected+real_only; audit=audit_partitions(records,require_speaker_disjoint=True); pcm=defaultdict(list)
    for r in records:
        path=root/r.path
        if file_hash(path)!=r.checksum: raise ValueError('Audio checksum changed')
        x,sr=sf.read(path,dtype='float32',always_2d=True)
        if not np.isfinite(x).all(): raise ValueError('Nonfinite audio')
        digest=hashlib.sha256(str((sr,x.shape)).encode()+x.astype('<f4').tobytes()).hexdigest(); pcm[digest].append(r.sample_id)
    if any(len(v)>1 for v in pcm.values()): raise ValueError('Selected decoded duplicate')
    write_csv(out/'manifest.csv',[r.to_dict() for r in records]); write_csv(out/'foundation_support.csv',coverage)
    extra=[]; b_ids={r.sample_id for r in real_only}
    for r in records:
        extra.append(dict(sample_id=r.sample_id,official_partition=r.official_split,research_partition=r.split,
            stratum='B_REAL_ONLY' if r.sample_id in b_ids else 'A_PAIRED',speaker_id=r.speaker_id,attack_id=r.attack_id,label=r.label,
            dataset_version=r.dataset_version,source_checksum=r.checksum,parent_key=r.parent_recording_id,
            selection_seed=seed,selection_strategy='speaker_attack_rotating_remainder_v1'))
    write_csv(out/'selection_rows.csv',extra); write_json(out/'partition_audit.json',audit)
    counts={}
    for name,group in [('train',[r for r in selected if r.split=='train']),('development_A',[r for r in selected if r.split=='development']),('development_B',real_only)]:
        counts[name]=dict(n=len(group),real=sum(r.label==0 for r in group),fake=sum(r.label==1 for r in group),speakers=len({r.speaker_id for r in group}),attacks=dict(sorted(Counter(r.attack_id for r in group if r.label).items())))
    support_selected=[]
    for role,speaker in sorted({(r.split,r.speaker_id) for r in records}):
        g=[r for r in records if r.split==role and r.speaker_id==speaker]
        support_selected.append(dict(split=role,speaker_id=speaker,REAL=sum(r.label==0 for r in g),FAKE=sum(r.label==1 for r in g),**{a:sum(r.attack_id==a for r in g) for a in ATTACKS}))
    write_csv(out/'selected_support.csv',support_selected)
    meta=dict(status='LOCKED_ALL_ATTACK_SELECTION',selection_id=selection_id,seed=seed,per_cell=per_cell,counts=counts,
        manifest_sha256=file_hash(out/'manifest.csv'),selection_rows_sha256=file_hash(out/'selection_rows.csv'),foundation_hashes=hashes,
        original_source_lineage='ORIGINAL_SOURCE_LINEAGE_UNKNOWN',known_partition_audit='PASS',decoded_duplicates=0,
        calibration_accessed=False,evaluation_accessed=False,scope='EXPLORATORY_DEVELOPMENT; A01-A06 SEEN IN TRAIN; B speakers may have prior Phase5 exposure')
    write_json(out/'selection.json',meta); return meta

def selection(root,selection_path):
    directory=safe_path(root,selection_path); m=json.loads((directory/'selection.json').read_text())
    if m['status']!='LOCKED_ALL_ATTACK_SELECTION' or file_hash(directory/'manifest.csv')!=m['manifest_sha256'] or file_hash(directory/'selection_rows.csv')!=m['selection_rows_sha256']: raise ValueError('Selection seal mismatch')
    return directory,m,load_public_manifest(directory/'manifest.csv')

def cache_contract(root,config):
    directory,m,records=selection(root,config['selection_path'])
    if set(config)!={'selection_path','selection_sha256','cache_id','policies','waveform_version','maximum_seconds','disk_reserve_gib'} or config['policies']!=['P0'] or config['waveform_version']!='c2-view-1-rb5-finite2' or config['disk_reserve_gib']<15: raise ValueError('Invalid cache contract')
    if config['selection_sha256']!=m['manifest_sha256']: raise ValueError('Stale selection reference')
    from .preprocessing import Preprocessor
    from .rawboost import PARAMETERS
    sources={r.sample_id:r.checksum for r in records}
    code={p:file_hash(root/p) for p in ('src/lava/jaes_v2/phase7_data.py','src/lava/jaes_v2/c2_data.py','src/lava/jaes_v2/preprocessing.py','src/lava/jaes_v2/rawboost.py','src/lava/jaes_v2/vendor/rawboost/RawBoost.py','src/model.py','src/preprocessing.py','config.py')}
    weight=Path.home()/'.keras/models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5'
    contract=dict(config_hash=object_hash(config),selection_hash=m['manifest_sha256'],source_hash=object_hash(sources),source_files=code,
        preprocessing=Preprocessor('P0').fingerprint,pretraining_weight_sha256=file_hash(weight),rawboost_parameters=PARAMETERS,
        train_conditions=TRAIN_CONDITIONS,development_conditions=DEV_CONDITIONS,rate=22050,window_seconds=3)
    return contract,records

def waveform_views(root,r,ffmpeg):
    from .c2_data import paired_views
    from .transforms import load_audio,Audio
    from .rawboost import rawboost
    views,evidence=paired_views(load_audio(root/r.path,sample_rate=22050,max_seconds=3),r.sample_id,r.split,ffmpeg)
    if r.split=='train':
        for k in range(2): views[f'rawboost{k}']=Audio(rawboost(views['clean'].samples,seed=evidence['sample_seed']+k),22050)
    else: views['gain_half']=Audio(views['clean'].samples*.5,22050)
    return views,evidence

def atomic_npz(path,**arrays):
    temporary=path.with_suffix('.tmp.npz'); np.savez_compressed(temporary,**arrays); os.replace(temporary,path)

def build_cache(root,config,*,benchmark_sources=None):
    from .c2_student import initialize_tensorflow,build_student,peak_process_mib
    from .preprocessing import Preprocessor
    contract,records=cache_contract(root,config); out=safe_path(root,str(BASE/'caches'/config['cache_id']))
    if (out/'cache_metadata.json').exists(): raise FileExistsError('Sealed cache cannot be overwritten')
    work=out.with_name(out.name+'.incomplete'); work.mkdir(parents=True,exist_ok=True)
    identity=work/'contract.json'
    if identity.exists() and json.loads(identity.read_text())!=json.loads(json.dumps(contract)): raise ValueError('Interrupted cache contract incompatible')
    write_json(identity,contract)
    if shutil.disk_usage(root).free<config['disk_reserve_gib']*1024**3+256*1024**2: raise ValueError('Disk reserve')
    tf=initialize_tensorflow(); full,backbone,head,extractor=build_student(seed=6107); ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg: raise ValueError('ffmpeg required')
    if benchmark_sources:
        records=[r for role in ('train','development') for r in [x for x in records if x.split==role][:benchmark_sources]]
    started=perf_counter(); reused=0
    try:
        for ordinal,r in enumerate(records):
            if perf_counter()-started>config['maximum_seconds']: raise TimeoutError('Cache budget exceeded')
            if file_hash(root/r.path)!=r.checksum: raise ValueError('Changed source')
            key=object_hash(r.sample_id)[:24]; archive=work/(key+'.npz'); meta=work/(key+'.json')
            if meta.exists():
                saved=json.loads(meta.read_text())
                if saved['source_checksum']!=r.checksum or file_hash(archive)!=saved['archive_sha256']: raise ValueError('Interrupted shard mismatch')
                reused+=1; continue
            views,evidence=waveform_views(root,r,ffmpeg); conditions=TRAIN_CONDITIONS if r.split=='train' else DEV_CONDITIONS
            tensors=[Preprocessor('P0').extract(views[c].samples) for c in conditions]; vectors=[]
            for j in range(0,len(tensors),2): vectors.extend(extractor(np.stack(tensors[j:j+2]),training=False).numpy())
            atomic_npz(archive,embeddings=np.stack(vectors))
            diagnostics=[]
            for c in conditions:
                wave=views[c].samples
                diagnostics.append(dict(condition=c,waveform_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
                    rms=float(np.sqrt(np.mean(wave.astype('float64')**2))),peak=float(np.max(np.abs(wave))),
                    silence_fraction=float(np.mean(np.abs(wave)<1e-4)),clipped_fraction=float(np.mean(np.abs(wave)>1)),
                    transform_seed=evidence['sample_seed']+(int(c[-1]) if c.startswith('rawboost') else 0)))
            write_json(meta,dict(sample_id=r.sample_id,source_checksum=r.checksum,archive_sha256=file_hash(archive),views=diagnostics))
            if (ordinal+1)%16==0: print(f'Phase7 cache {ordinal+1}/{len(records)} {perf_counter()-started:.1f}s',flush=True)
        arrays=defaultdict(list); rows=[]
        for r in records:
            key=object_hash(r.sample_id)[:24]; info=json.loads((work/(key+'.json')).read_text())
            with np.load(work/(key+'.npz'),allow_pickle=False) as z: vectors=z['embeddings']
            for diagnostic,vector in zip(info['views'],vectors):
                k=r.split+'_P0'; index=len(arrays[k]); arrays[k].append(vector)
                rows.append(dict(sample_id=r.sample_id,split=r.split,speaker_id=r.speaker_id,attack_id=r.attack_id,label=r.label,policy='P0',
                    array_key=k,array_index=index,source_checksum=r.checksum,**diagnostic))
        atomic_npz(work/'embeddings.npz',**{k:np.stack(v) for k,v in arrays.items()}); write_csv(work/'view_manifest.csv',rows)
        elapsed=perf_counter()-started; state=dict(status='SEALED_PHASE7_CACHE',contract=contract,contract_hash=object_hash(contract),elapsed_seconds=elapsed,
            benchmark_only=bool(benchmark_sources),sources=len(records),views=len(rows),arrays={k:list(np.stack(v).shape) for k,v in arrays.items()},
            resumed_shards=reused,peak_ram_mib=peak_process_mib(),gpu_available=bool(tf.config.list_physical_devices('GPU')),
            output_hashes={name:file_hash(work/name) for name in ('embeddings.npz','view_manifest.csv','contract.json')})
        write_json(work/'cache_metadata.json',state)
        # Entire completed directory published by one same-volume rename.
        if out.exists(): raise FileExistsError(out)
        os.replace(work,out); return state
    except BaseException as error:
        write_json(work/'failure.json',dict(status='FAILED_OR_INTERRUPTED',error=repr(error),elapsed_seconds=perf_counter()-started)); raise

def load_cache(root,config):
    from .phase3 import verify_seal
    contract,records=cache_contract(root,config)
    directory=safe_path(root,str(BASE/'caches'/config['cache_id'])); m=json.loads((directory/'cache_metadata.json').read_text())
    if m['status']!='SEALED_PHASE7_CACHE' or m['benchmark_only'] or object_hash(contract)!=m['contract_hash']: raise ValueError('Stale/incomplete cache')
    verify_seal(directory,m)
    return directory,m,records
