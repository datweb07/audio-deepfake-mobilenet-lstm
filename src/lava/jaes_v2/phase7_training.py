"""Explicit Phase7 seed blocks and genuine tensor-to-backbone training lifecycle."""
from dataclasses import dataclass,asdict
from pathlib import Path
from time import perf_counter
import hashlib,json,os,re
import numpy as np
from .evidence import read_csv,write_csv,write_json,file_hash,object_hash
from .phase7_data import BASE,safe_path,selection,load_cache
from .phase6_training import paired_objective,reliability_metrics

BANKS={'H0':['clean'],'H1':['clean','gain_half','noise10','opus24'],
       'H2':['clean','gain_half','noise10','opus24'],'H3':['clean','gain_half','noise10','opus24'],
       'H4':['clean','rawboost0','rawboost1'],'H5':['clean','rawboost_online','rawboost_online']}

@dataclass(frozen=True)
class Config:
    experiment_id:str
    source_manifest:str
    manifest_sha256:str
    selection_id:str
    selection_seed:int
    cache_config:str
    training_seed:int
    augmentation_policy:str
    phase:int=7
    scope:str='EXPLORATORY_DEVELOPMENT'
    dataset_id:str='ASVspoof2019_LA'
    preprocessing_policy:str='P0'
    training_policy:str='HEAD_ONLY'
    model_architecture:str='existing-MobileNetV3Small-LSTM128'
    initialization:str='imagenet_fresh_head'
    optimizer:str='Adam'
    learning_rate:float=.001
    schedule:str='linear_half_over_epochs'
    epochs:int=6
    optimizer_updates:int=240
    batch_size:int=16
    gradient_accumulation:int=1
    checkpoint_selection:str='DEVELOPMENT_A_CLEAN_BCE'
    operating_threshold:float=.5
    maximum_seconds:float=240
    consistency_weight:float=.1
    protocol_path:str='outputs/jaes_v2/phase7/protocol_v1.json'
    resume_policy:str='RESUME_NOT_SUPPORTED'
    def __post_init__(self):
        if not re.fullmatch(r'p7_[a-z0-9_]+',self.experiment_id) or self.phase!=7 or self.scope!='EXPLORATORY_DEVELOPMENT' or self.augmentation_policy not in BANKS: raise ValueError('Unsupported experiment scope/method')
        if self.training_policy not in ('HEAD_ONLY','PARTIAL_BACKBONE_FINETUNE') or self.preprocessing_policy!='P0' or self.dataset_id!='ASVspoof2019_LA' or self.initialization!='imagenet_fresh_head' or self.model_architecture!='existing-MobileNetV3Small-LSTM128': raise ValueError('Architecture contract')
        if self.optimizer!='Adam' or self.schedule!='linear_half_over_epochs' or self.checkpoint_selection!='DEVELOPMENT_A_CLEAN_BCE' or self.operating_threshold!=.5 or self.resume_policy!='RESUME_NOT_SUPPORTED': raise ValueError('Lifecycle contract')
        if not 1<=self.epochs<=6 or self.batch_size not in (1,16) or not 1<=self.gradient_accumulation<=16 or not 0<self.learning_rate<=.001 or not 0<self.maximum_seconds<=3000 or not 0<=self.consistency_weight<=1: raise ValueError('Bounded resource config')
        for value in (self.source_manifest,self.cache_config,self.protocol_path):
            if Path(value).is_absolute() or '..' in Path(value).parts: raise ValueError('Unsafe path')
        if not re.fullmatch('[a-z0-9_]+',self.selection_id) or not re.fullmatch('[a-f0-9]{64}',self.manifest_sha256): raise ValueError('Selection identity')

def seeds(seed):
    return dict(initialization=seed,dropout=seed,batch_order=seed+100000,view_choice=seed+200000,online_augmentation=seed+300000)

def weights_hash(weights):
    return hashlib.sha256(b''.join(v.numpy().tobytes() for v in weights)).hexdigest()

def named_weights_hash(weights):
    """Trainability can reorder Keras weights; compare canonical named tensors."""
    digest=hashlib.sha256()
    for v in sorted(weights,key=lambda x:x.name):
        digest.update(v.name.encode());digest.update(str(v.shape.as_list()).encode());digest.update(v.numpy().tobytes())
    return digest.hexdigest()

def online_rawboost(samples,*,training_seed,epoch,sample_id,occurrence=0,sample_rate=22050):
    from .rawboost import rawboost
    if sample_rate!=22050 or epoch<0 or occurrence<0: raise ValueError('Reviewed online adapter requires22,050Hz')
    seed=int(object_hash(dict(seed=training_seed,epoch=epoch,sample_id=sample_id,occurrence=occurrence))[:16],16)
    result=rawboost(samples,sample_rate=sample_rate,seed=seed,algorithm=5)
    if np.max(np.abs(result))>1.00001: raise ValueError('RawBoost bounded-amplitude contract failed; no post-normalization')
    return result,seed

def atomic_save(model,path):
    temporary=path.with_name(path.stem+'.temporary.keras'); model.save(temporary); os.replace(temporary,path)

def score_metrics(rows):
    from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
    y=np.array([int(r['label']) for r in rows]); p=np.array([float(r['raw_score']) for r in rows]); pred=p>=.5
    if not len(y) or not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Missing/invalid predictions')
    n0=int(sum(y==0)); n1=int(sum(y==1)); fp=int(sum(pred&(y==0))); fn=int(sum(~pred&(y==1)))
    result=dict(n=len(y),n_REAL=n0,n_FAKE=n1,REAL_false_alarm=fp/n0 if n0 else None,FAKE_miss=fn/n1 if n1 else None,
        fp=fp,fn=fn,mean_score=float(np.mean(p)),score_p10=float(np.quantile(p,.1)),score_p90=float(np.quantile(p,.9)),
        roc_auc=None,pr_auc=None,eer=None,two_class_status='DEFINED' if n0 and n1 else 'ONE_CLASS_NOT_DEFINED',**reliability_metrics(y,p))
    if n0 and n1:
        fpr,tpr,_=roc_curve(y,p); k=np.argmin(np.abs(fpr-(1-tpr)))
        result.update(roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),eer=float((fpr[k]+1-tpr[k])/2))
    return result

def train(root,config,*,preflight_only=False):
    from .c2_student import initialize_tensorflow,build_student,peak_process_mib
    from .runtime import source_fingerprint
    from .phase6_data import DEV_CONDITIONS
    from .preprocessing import Preprocessor
    from .transforms import load_audio
    from .controls import prepare_headroom
    c=asdict(config); cache_cfg=json.loads(safe_path(root,config.cache_config).read_text()); cache,cache_meta,records=load_cache(root,cache_cfg)
    cache_audit=json.loads((root/BASE/'cache_partition_audit_v1.json').read_text())
    if cache_audit['status']!='PASS' or cache_audit['cache_metadata_sha256']!=file_hash(cache/'cache_metadata.json') or cache_audit['view_manifest_sha256']!=file_hash(cache/'view_manifest.csv'):raise ValueError('Post-transform cache audit required')
    selection_dir,selection_meta,_=selection(root,cache_cfg['selection_path'])
    if config.manifest_sha256!=selection_meta['manifest_sha256'] or config.source_manifest!=(Path(cache_cfg['selection_path'])/'manifest.csv').as_posix() or config.selection_id!=selection_meta['selection_id'] or config.selection_seed!=selection_meta['seed']: raise ValueError('Config/selection mismatch')
    protocol=json.loads(safe_path(root,config.protocol_path).read_text())
    if protocol['configuration_hashes'].get(config.experiment_id+'.json')!=object_hash_config_file(c,root): raise ValueError('Configuration not registered in locked protocol')
    if protocol['confirmation_enabled'] or config.training_seed not in protocol['training_seeds'] or config.maximum_seconds>protocol['maximum_run_seconds'] or config.manifest_sha256!=protocol['selection_hash']: raise ValueError('Locked protocol mismatch')
    if config.epochs!=protocol['epochs'] and config.epochs!=protocol['representation_epochs']: raise ValueError('Unregistered epoch schedule')
    tr=sorted([r for r in records if r.split=='train'],key=lambda r:r.sample_id); dv=sorted([r for r in records if r.split=='development'],key=lambda r:r.sample_id)
    b_ids={r['sample_id'] for r in read_csv(selection_dir/'selection_rows.csv') if r['stratum']=='B_REAL_ONLY'}
    dev_a=[r for r in dv if r.sample_id not in b_ids]
    effective=config.batch_size*config.gradient_accumulation
    if config.optimizer_updates!=int(np.ceil(len(tr)/effective))*config.epochs: raise ValueError('Update count mismatch')
    tf=initialize_tensorflow(); full,backbone,head,extractor=build_student(seed=config.training_seed)
    partial=config.training_policy=='PARTIAL_BACKBONE_FINETUNE'; online=config.augmentation_policy=='H5'; paired=config.augmentation_policy in ('H2','H3')
    if not preflight_only and partial:
        gate=json.loads((root/BASE/'representation_resource_gate_v1.json').read_text())
        if gate['status']!='PASS' or gate['config_hash']!=object_hash(c): raise ValueError('Representation resource gate mismatch')
    if config.training_seed!=6107:
        gate=json.loads((root/BASE/'multi_seed_gate_v1.json').read_text())
        if gate['status']!='PASS': raise ValueError('Multi-seed gate failed')
    initial=weights_hash(full.weights);initial_named=named_weights_hash(full.weights)
    if partial:
        from src.model import unfreeze_backbone_for_finetuning
        unfreeze_backbone_for_finetuning(backbone)
    if named_weights_hash(full.weights)!=initial_named:raise ValueError('Trainability transition changed initialized tensors')
    variables=list(full.trainable_weights if partial else head.trainable_weights)
    frozen=[v for v in full.weights if id(v) not in {id(x) for x in variables}]; frozen_before=weights_hash(frozen)
    kernels=[v for v in backbone.trainable_weights if 'kernel' in v.name]; kernel_before={v.name:v.numpy().copy() for v in kernels}
    bn=[l for l in backbone.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
    optimizer=tf.keras.optimizers.Adam(config.learning_rate); sd=seeds(config.training_seed)
    order_rng=np.random.default_rng(sd['batch_order']); view_rng=np.random.default_rng(sd['view_choice'])
    stored=np.load(cache/'embeddings.npz',allow_pickle=False); arrays={k:stored[k] for k in stored.files}; stored.close()
    entries=read_csv(cache/'view_manifest.csv'); index={(r['sample_id'],r['condition']):r for r in entries}
    def embedding(record,condition):
        row=index[(record.sample_id,condition)]; return arrays[row['array_key']][int(row['array_index'])]
    def tensor(record):
        if file_hash(root/record.path)!=record.checksum: raise ValueError('Source changed during training')
        audio=load_audio(root/record.path,sample_rate=22050,max_seconds=3); clean,_=prepare_headroom(audio,.25)
        digest=hashlib.sha256(clean.samples.astype('<f4').tobytes()).hexdigest()
        if digest!=index[(record.sample_id,'clean')]['waveform_sha256']:raise ValueError('Prepared clean waveform differs from shared cache')
        return Preprocessor('P0').extract(clean.samples)
    # Fine-tuning does not consume detached embeddings for training or inference.
    if partial and config.augmentation_policy!='H0': raise ValueError('Only budgeted ERM representation pilot enabled')
    out=root/BASE/('preflight' if preflight_only else 'training')/config.experiment_id; out.mkdir(parents=True,exist_ok=False)
    code=source_fingerprint(root); state=dict(status='RUNNING',phase=7,config=c,config_hash=object_hash(c),code=code,seed_ledger=sd,
        protocol_sha256=file_hash(safe_path(root,config.protocol_path)),cache_metadata_sha256=file_hash(cache/'cache_metadata.json'),
        checkpoint_sha256=None,initial_model_weights_sha256=initial,initial_named_weights_sha256=initial_named,calibration_accessed=False,evaluation_accessed=False,teacher_used=False)
    write_json(out/'resolved_config.json',c); write_json(out/'initialization_hash.json',dict(full_weights_sha256=initial,canonical_named_weights_sha256=initial_named,seeds=sd,pretraining_weight_sha256=cache_meta['contract']['pretraining_weight_sha256']))
    write_json(out/'data_hashes.json',dict(manifest=config.manifest_sha256,cache=file_hash(cache/'cache_metadata.json')))
    import platform,sys
    write_json(out/'environment.json',dict(python=sys.version,tensorflow=tf.__version__,platform=platform.platform(),gpu_visible=[str(x) for x in tf.config.list_physical_devices('GPU')],TEMP=os.getenv('TEMP'),TMP=os.getenv('TMP'),MPLCONFIGDIR=os.getenv('MPLCONFIGDIR')))
    if partial:
        write_json(out/'trainable_layer_manifest.json',dict(trainable=[dict(name=v.name,shape=v.shape.as_list()) for v in variables],frozen=[v.name for v in frozen]))
        write_json(out/'BatchNorm_policy.json',dict(policy='ALL_BN_FROZEN',layers=[l.name for l in bn],all_frozen=all(not l.trainable for l in bn)))
    write_json(out/'run_metadata.json',state); started=perf_counter(); updates=0; history=[]; log=[]; best=float('inf'); gradients_by_name={}; timing=[]
    @tf.function(reduce_retracing=True)
    def gradient(x,y,clean):
        with tf.GradientTape() as tape:
            network=full if partial else head
            p=network(x,training=True); q=network(clean,training=True) if paired else None
            value=paired_objective(y,p,q,consistency=config.augmentation_policy=='H3',weight=config.consistency_weight)
        g=tape.gradient(value,variables)
        for a in g:
            if a is None: raise ValueError('Detached trainable variable')
            tf.debugging.assert_all_finite(a,'Nonfinite gradient')
        return value,g
    def infer(model,group,condition):
        if not partial:
            return model(np.stack([embedding(r,condition) for r in group]),training=False).numpy().reshape(-1)
        from .phase7_data import waveform_views
        # No recurrent optimizer/tensor cache is shared with the frozen-feature path.
        probabilities=[]
        for r in group:
            if condition=='clean': x=tensor(r)
            else:
                views,_=waveform_views(root,r,shutil_which_ffmpeg()); x=Preprocessor('P0').extract(views[condition].samples)
            probabilities.append(float(model(x[None],training=False).numpy()[0,0]))
        return np.array(probabilities)
    try:
        for epoch in range(1 if preflight_only else config.epochs):
            optimizer.learning_rate.assign(config.learning_rate*(1-.5*epoch/config.epochs)); losses=[]
            order=order_rng.permutation(len(tr))
            limit=min(effective,len(tr)) if preflight_only else len(tr)
            for begin in range(0,limit,effective):
                if perf_counter()-started>config.maximum_seconds: raise TimeoutError('Training lifecycle wall-clock budget')
                chosen=[tr[i] for i in order[begin:begin+effective]]; conditions=view_rng.choice(BANKS[config.augmentation_policy],len(chosen)).tolist()
                accumulated=[tf.zeros_like(v) for v in variables]; loss_sum=0.; t=perf_counter()
                for j in range(0,len(chosen),config.batch_size):
                    group=chosen[j:j+config.batch_size]; cs=conditions[j:j+config.batch_size]; x=[]
                    for r,condition in zip(group,cs):
                        aug_seed=None
                        if partial: value=tensor(r)
                        elif condition=='rawboost_online':
                            from .transforms import load_audio
                            a=load_audio(root/r.path,sample_rate=22050,max_seconds=3); a,_=prepare_headroom(a,.25)
                            wave,aug_seed=online_rawboost(a.samples,training_seed=sd['online_augmentation'],epoch=epoch,sample_id=r.sample_id)
                            value=extractor(Preprocessor('P0').extract(wave)[None],training=False).numpy()[0]
                        else: value=embedding(r,condition)
                        x.append(value); log.append(dict(epoch=epoch+1,update=updates+1,sample_id=r.sample_id,view=condition,second='clean' if paired else 'NONE',augmentation_seed=aug_seed if aug_seed is not None else 'FINITE_BANK'))
                    x=np.stack(x); y=np.array([r.label for r in group],dtype='float32'); clean=np.stack([embedding(r,'clean') for r in group]) if paired else x
                    loss,g=gradient(x,y,clean); factor=len(group)/len(chosen)
                    accumulated=[a+factor*b for a,b in zip(accumulated,g)]; loss_sum+=float(loss)*factor
                optimizer.apply_gradients(zip(accumulated,variables)); updates+=1; losses.append(loss_sum); timing.append(perf_counter()-t)
                gradients_by_name={v.name:float(tf.linalg.norm(g)) for v,g in zip(variables,accumulated)}
            if preflight_only: break
            clean_p=infer(full if partial else head,dev_a,'clean'); y=np.array([r.label for r in dev_a]); p=np.clip(clean_p,1e-7,1-1e-7)
            bce=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),development_clean_bce=bce,optimizer_updates=updates,learning_rate=float(optimizer.learning_rate.numpy())))
            write_csv(out/'training_history.csv',history)
            if bce<best:
                best=bce; reference_tensor=tensor(dev_a[0])[None]; reference_score=float(full(reference_tensor,training=False).numpy()[0,0])
                atomic_save(full,out/'best.keras'); write_json(out/'checkpoint_selection.json',dict(epoch=epoch+1,criterion=config.checkpoint_selection,value=bce,checkpoint_sha256=file_hash(out/'best.keras'),parity_sample_id=dev_a[0].sample_id,reference_prediction=reference_score))
            print(f'{config.experiment_id} epoch{epoch+1} loss{np.mean(losses):.4f} dev{bce:.4f} {perf_counter()-started:.1f}s',flush=True)
        if weights_hash(frozen)!=frozen_before: raise ValueError('Frozen/BN state modified')
        changed=[v.name for v in kernels if not np.array_equal(kernel_before[v.name],v.numpy())]
        if partial and (not changed or not all(not l.trainable for l in bn)): raise ValueError('Backbone/BN gradient gate failed')
        if preflight_only:
            resource=dict(status='PASS',update_seconds=timing,peak_ram_mib=peak_process_mib(),changed_kernels=changed,gradients=gradients_by_name,
                trainable_parameters=int(sum(np.prod(v.shape) for v in variables)),total_parameters=full.count_params(),batchnorm_frozen=sum(not l.trainable for l in bn),
                projected_training_seconds=float(np.median(timing)*config.optimizer_updates),projected_clean_validation_seconds=len(dev_a)*.35*config.epochs,
                projected_full_evaluation_seconds=len(dv)*7*.6,activation_memory='Included in measured OS process peak; tensor allocator component NOT_MEASURED',
                microbatch=config.batch_size,gradient_accumulation=config.gradient_accumulation,source_records=len(chosen),
                actual_train_sources=[dict(sample_id=r.sample_id,label=r.label,attack_id=r.attack_id,speaker_id=r.speaker_id) for r in chosen])
            write_json(out/'resource_measurements.json',resource); state.update(status='SMOKE_TESTED',elapsed_seconds=perf_counter()-started)
        else:
            loaded=tf.keras.models.load_model(out/'best.keras',compile=False)
            checkpoint_info=json.loads((out/'checkpoint_selection.json').read_text())
            residual=abs(float(loaded(tensor(dev_a[0])[None],training=False).numpy()[0,0])-checkpoint_info['reference_prediction'])
            if residual>1e-6: raise ValueError('Checkpoint reload parity failed')
            if partial: selected=loaded
            else:
                inputs=tf.keras.Input((6,576)); z=inputs
                for name in ('temporal_lstm','classifier_dense','classifier_dropout','probability_fake'): z=loaded.get_layer(name)(z)
                selected=tf.keras.Model(inputs,z)
            cache_parity=None
            if not partial:
                cache_parity=abs(float(selected(embedding(dev_a[0],'clean')[None],training=False).numpy()[0,0])-float(loaded(tensor(dev_a[0])[None],training=False).numpy()[0,0]))
                if cache_parity>1e-5: raise ValueError('Frozen cache versus full-student inference parity failed')
            scores=[]; metrics=[]; partial_predictions={condition:[] for condition in DEV_CONDITIONS}
            if partial:
                from .phase7_data import waveform_views
                # Generate each immutable waveform bank once per source; predict all
                # conditions together in small batches, without detached embeddings.
                for ordinal,r in enumerate(dv):
                    if perf_counter()-started>config.maximum_seconds: raise TimeoutError('Full-backbone evaluation budget')
                    views,_=waveform_views(root,r,shutil_which_ffmpeg())
                    for condition in DEV_CONDITIONS:
                        digest=hashlib.sha256(views[condition].samples.astype('<f4').tobytes()).hexdigest()
                        if digest!=index[(r.sample_id,condition)]['waveform_sha256']:raise ValueError('Fine-tuning evaluation waveform differs from frozen-cache comparison')
                    tensors=[Preprocessor('P0').extract(views[c].samples) for c in DEV_CONDITIONS]; values=[]
                    for j in range(0,len(tensors),2): values.extend(selected(np.stack(tensors[j:j+2]),training=False).numpy().reshape(-1))
                    for condition,value in zip(DEV_CONDITIONS,values): partial_predictions[condition].append(value)
                    if (ordinal+1)%32==0: print(f'{config.experiment_id} full-backbone evaluation {ordinal+1}/{len(dv)} {perf_counter()-started:.1f}s',flush=True)
            for condition in DEV_CONDITIONS:
                if perf_counter()-started>config.maximum_seconds: raise TimeoutError('Evaluation-inclusive lifecycle budget')
                p=partial_predictions[condition] if partial else infer(selected,dv,condition)
                for r,value in zip(dv,p): scores.append(dict(sample_id=r.sample_id,speaker_id=r.speaker_id,attack_id=r.attack_id,label=r.label,raw_score=float(value),predicted_label=int(value>=.5),threshold=.5,condition_id=condition,stratum='B_REAL_ONLY' if r.sample_id in b_ids else 'A_PAIRED'))
                for stratum in ('A_PAIRED','B_REAL_ONLY'):
                    metrics.append(dict(condition=condition,stratum=stratum,**score_metrics([r for r in scores if r['condition_id']==condition and r['stratum']==stratum])))
                print(f'{config.experiment_id} scored {condition} {perf_counter()-started:.1f}s',flush=True)
            write_csv(out/'scores.csv',scores); write_json(out/'metrics.json',metrics)
            # Optimizer state retained as evidence; no exact-resume claim.
            atomic_npz_optimizer(out/'optimizer_state.npz',optimizer)
            write_json(out/'gradient_diagnostics.json',dict(last_gradient_norms=gradients_by_name,changed_backbone_kernels=changed,
                frozen_variables_unchanged=len(frozen),all_batchnorm_frozen=all(not l.trainable for l in bn),optimizer_created_after_trainability=True))
            write_json(out/'checkpoint_reload.json',dict(max_residual=residual,frozen_cache_full_student_residual=cache_parity,sample_id=dev_a[0].sample_id,optimizer_state='Last-update evidence, not selected-best optimizer continuation',exact_resume='RESUME_NOT_SUPPORTED'))
            write_csv(out/'training_views.csv',log)
            resource=dict(elapsed_seconds=perf_counter()-started,update_seconds_median=float(np.median(timing)),update_seconds_total=sum(timing),
                trainable_parameters=int(sum(np.prod(v.shape) for v in variables)),total_parameters=full.count_params(),batchnorm_frozen=sum(not l.trainable for l in bn),
                peak_ram_mib=peak_process_mib(),gpu_used=False,optimizer_updates=updates,unique_training_sources=len(tr),source_examples_seen=len(tr)*config.epochs,
                supervised_views_seen=len(tr)*config.epochs*(2 if paired else 1),model_training_forwards=len(log)*(2 if paired else 1)//config.batch_size,
                backward_calls=int(np.ceil(len(tr)/config.batch_size))*config.epochs,checkpoint_bytes=(out/'best.keras').stat().st_size)
            write_json(out/'resource_measurements.json',resource)
            state.update(status='PILOT_EVALUATED',checkpoint_sha256=file_hash(out/'best.keras'),elapsed_seconds=resource['elapsed_seconds'],optimizer_updates=updates)
        write_json(out/'status.json',dict(status=state['status'],resume='RESUME_NOT_SUPPORTED'))
        state['output_hashes']={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='run_metadata.json'}; write_json(out/'run_metadata.json',state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=repr(error),optimizer_updates=updates,elapsed_seconds=perf_counter()-started)
        write_json(out/'status.json',dict(status=state['status'],error=repr(error))); write_json(out/'run_metadata.json',state); raise

def shutil_which_ffmpeg():
    import shutil
    value=shutil.which('ffmpeg')
    if not value: raise ValueError('ffmpeg unavailable')
    return value

def atomic_npz_optimizer(path,optimizer):
    from .phase7_data import atomic_npz
    atomic_npz(path,**{f'variable_{i}':v.numpy() for i,v in enumerate(optimizer.variables())})

def object_hash_config_file(config,root):
    path=root/'configs/jaes_v2/phase7'/(config['experiment_id']+'.json')
    if json.loads(path.read_text())!=config: raise ValueError('Config content mismatch')
    return file_hash(path)
