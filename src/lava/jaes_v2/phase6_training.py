"""Explicit Phase6 head-only baselines and a separate real partial-backbone sanity path."""
from dataclasses import dataclass,asdict
from pathlib import Path
from time import perf_counter
import json
import re
import hashlib
import numpy as np
from .evidence import read_csv,write_csv,write_json,file_hash,object_hash
from .phase3 import verify_seal
from .c2_student import build_student,initialize_tensorflow,peak_process_mib
from .c2_training import validate_views,evaluate
from .phase6_data import BASE,DEV_CONDITIONS


@dataclass(frozen=True)
class Config:
    experiment_id: str
    policy: str='P0'
    baseline: str='B0'
    seed: int=5806
    epochs: int=6
    batch_size: int=16
    learning_rate: float=.001
    consistency_weight: float=.1
    maximum_seconds: float=180
    scope: str='CONTROLLED_PILOT'
    training_policy: str='HEAD_ONLY'
    def __post_init__(self):
        if not re.fullmatch(r'p6_[a-z0-9_]+',self.experiment_id) or self.policy not in ('P0','P1') or self.baseline not in ('B0','B1','B2','B4_CONTROL','B4','RB'): raise ValueError('Unsupported baseline/config')
        if self.scope!='CONTROLLED_PILOT' or self.training_policy!='HEAD_ONLY': raise ValueError('Confirmation/fine-tuning cannot be enabled through a head-only config')
        if not 1<=self.epochs<=6 or self.batch_size!=16 or not 0<self.maximum_seconds<=300 or not 0<self.learning_rate<=.01 or not 0<=self.consistency_weight<=1: raise ValueError('Bounded pilot contract')


def paired_objective(labels,first,second=None,*,consistency=False,weight=.1):
    import tensorflow as tf
    y=tf.reshape(tf.cast(labels,tf.float32),(-1,1)); p=tf.reshape(first,(-1,1))
    loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,p))
    if second is None:
        if consistency: raise ValueError('Consistency requires paired views')
        return loss
    q=tf.reshape(second,(-1,1))
    if not 0<=weight<=1: raise ValueError('Invalid consistency weight')
    return .5*(loss+tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,q)))+(weight*tf.reduce_mean(tf.square(p-q)) if consistency else 0.)


def reliability_metrics(labels,scores,*,bins=10):
    y=np.asarray(labels,dtype=int); p=np.asarray(scores,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not set(y)<= {0,1} or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or bins<2: raise ValueError('Invalid bounded score reliability inputs')
    clipped=np.clip(p,1e-7,1-1e-7); cells=np.minimum((p*bins).astype(int),bins-1)
    ece=sum(np.mean(cells==i)*abs(float(p[cells==i].mean())-float(y[cells==i].mean())) for i in range(bins) if np.any(cells==i))
    return dict(nll=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),brier=float(np.mean((p-y)**2)),ece=float(ece),
                bins=bins,classwise={str(c):dict(n=int(sum(y==c)),mean_score=float(p[y==c].mean()) if np.any(y==c) else None,
                    brier=float(np.mean((p[y==c]-c)**2)) if np.any(y==c) else None) for c in (0,1)},calibration_fitted=False)


def train(root,config):
    from .runtime import source_fingerprint
    cache=root/BASE/'feature_cache_v2'; metadata=json.loads((cache/'cache_metadata.json').read_text())
    if metadata['status']!='SEALED_PHASE6_FEATURE_CACHE': raise ValueError('Complete cache gate required')
    verify_seal(cache,metadata)
    gate=json.loads((root/BASE/'protocol_v1.json').read_text())
    if gate['confirmation_enabled'] or gate['maximum_training_seconds']<config.maximum_seconds: raise ValueError('Protocol budget/scope mismatch')
    tf=initialize_tensorflow(); full,backbone,head,_=build_student(seed=config.seed)
    if file_hash(Path.home()/'.keras/models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5')!=metadata['pretraining_weight_sha256']: raise ValueError('Pretraining mismatch')
    entries=read_csv(cache/'view_manifest.csv'); validate_views(entries); entries=[r for r in entries if r['policy']==config.policy]
    with np.load(cache/'embeddings.npz',allow_pickle=False) as compressed: arrays={k:compressed[k] for k in compressed.files}
    index={(r['sample_id'],r['condition']):r for r in entries}
    tr=sorted([r for r in entries if r['split']=='train' and r['condition']=='clean'],key=lambda r:r['sample_id'])
    dev=sorted([r for r in entries if r['split']=='development' and r['condition']=='clean'],key=lambda r:r['sample_id'])
    def batch(records,conditions):
        return np.stack([arrays[index[(r['sample_id'],c)]['array_key']][int(index[(r['sample_id'],c)]['array_index'])] for r,c in zip(records,conditions)])
    output=root/BASE/'training'/config.experiment_id; output.mkdir(parents=True,exist_ok=False)
    state=dict(status='TRAINING',phase=6,scope=config.scope,config=asdict(config),config_hash=object_hash(asdict(config)),code=source_fingerprint(root),
        initial_model_weights_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in full.weights)).hexdigest(),
        protocol_sha256=file_hash(root/BASE/'protocol_v1.json'),cache_metadata_sha256=file_hash(cache/'cache_metadata.json'),
        pretraining_weight_sha256=metadata['pretraining_weight_sha256'],training_policy='HEAD_ONLY',teacher_used=False,threshold=.5,
        research_model_id='jaes_v2_mobilenet_student',calibration_accessed=False,evaluation_accessed=False)
    write_json(output/'resolved_config.json',asdict(config)); write_json(output/'run_metadata.json',state)
    write_json(output/'environment.json',{**metadata['environment'],'execution_policy':'Phase6 CPU head-only research; reserved data unopened'})
    write_json(output/'data_manifest_hashes.json',dict(manifest=file_hash(cache/'manifest.csv'),views=file_hash(cache/'view_manifest.csv')))
    optimizer=tf.keras.optimizers.Adam(config.learning_rate); shuffle=np.random.default_rng(config.seed); views=np.random.default_rng(config.seed)
    paired=config.baseline in ('B4_CONTROL','B4'); bank={'B0':['clean'],'B1':['clean','gain_half'],'B2':['clean','gain_half','noise10','opus24'],
        'B4_CONTROL':['clean','gain_half','noise10','opus24'],'B4':['clean','gain_half','noise10','opus24'],'RB':['clean','rawboost0','rawboost1']}[config.baseline]
    @tf.function
    def step(x,y,clean):
        with tf.GradientTape() as tape:
            p=head(x,training=True); q=head(clean,training=True) if paired else None
            value=paired_objective(y,p,q,consistency=config.baseline=='B4',weight=config.consistency_weight)
        gradients=tape.gradient(value,head.trainable_weights); tf.debugging.assert_all_finite(value,'Nonfinite loss')
        optimizer.apply_gradients(zip(gradients,head.trainable_weights)); return value,tf.linalg.global_norm(gradients)
    started=perf_counter(); history=[]; best=float('inf'); steps=0; pair_log=[]
    try:
        for epoch in range(config.epochs):
            if perf_counter()-started>config.maximum_seconds: raise TimeoutError('Incomplete budget; run cannot be promoted')
            optimizer.learning_rate.assign(config.learning_rate*(1-.5*epoch/config.epochs)); order=shuffle.permutation(len(tr)); losses=[]; norms=[]
            for begin in range(0,len(tr),config.batch_size):
                selected=[tr[i] for i in order[begin:begin+config.batch_size]]; conditions=views.choice(bank,len(selected)).tolist()
                y=np.array([int(r['label']) for r in selected],dtype='float32')
                value,norm=step(batch(selected,conditions),y,batch(selected,['clean']*len(selected))); losses.append(float(value)); norms.append(float(norm)); steps+=1
                pair_log.extend(dict(epoch=epoch+1,step=steps,sample_id=r['sample_id'],view=c,second='clean' if paired else 'NONE') for r,c in zip(selected,conditions))
            clean_scores=head(batch(dev,['clean']*len(dev)),training=False).numpy().reshape(-1)
            y=np.array([int(r['label']) for r in dev]); dev_loss=float(np.mean(-(y*np.log(np.clip(clean_scores,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-clean_scores,1e-7,1)))))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),gradient_norm=float(np.mean(norms)),development_clean_bce=dev_loss,optimizer_steps=steps))
            write_csv(output/'training_history.csv',history)
            if dev_loss<best:
                best=dev_loss; full.save(output/'best.keras'); write_json(output/'checkpoint_selection.json',dict(epoch=epoch+1,criterion='DEVELOPMENT_CLEAN_BCE',value=best,checkpoint_sha256=file_hash(output/'best.keras')))
            print(f'{config.experiment_id}: epoch {epoch+1}, loss {np.mean(losses):.4f}, dev BCE {dev_loss:.4f}',flush=True)
        loaded=tf.keras.models.load_model(output/'best.keras',compile=False); inputs=tf.keras.Input((6,576)); x=inputs
        for name in ('temporal_lstm','classifier_dense','classifier_dropout','probability_fake'): x=loaded.get_layer(name)(x)
        selected_head=tf.keras.Model(inputs,x); scores=[]; results=[]
        for condition in DEV_CONDITIONS:
            rows,result=evaluate(selected_head,batch(dev,[condition]*len(dev)),np.array([int(r['label']) for r in dev]),dev,condition=condition)
            result['reliability']=reliability_metrics([r['label'] for r in rows],[r['raw_score'] for r in rows]); scores+=rows; results.append(result)
        write_csv(output/'validation_scores.csv',scores); write_json(output/'metrics.json',results); write_csv(output/'training_views.csv',pair_log)
        write_json(output/'diagnostics.json',dict(nonzero_gradients=all(h['gradient_norm']>0 for h in history),frozen_pretrained_backbone=True,paired_supervision=paired))
        state.update(status='PILOT_EVALUATED',optimizer_steps=steps,source_examples=len(tr)*config.epochs,
            supervised_views=len(tr)*config.epochs*(2 if paired else 1),training_head_forwards=steps*(2 if paired else 1),backward_passes=steps,
            elapsed_seconds=perf_counter()-started,peak_ram_mib=peak_process_mib(),parameters=full.count_params(),checkpoint_sha256=file_hash(output/'best.keras'),
            limitation='Single-seed finite-view frozen-backbone development; no novel method or calibrated probability claim')
        state['output_hashes']={p.name:file_hash(p) for p in output.iterdir() if p.is_file() and p.name!='run_metadata.json'}; write_json(output/'run_metadata.json',state)
        return output
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error),optimizer_steps=steps)
        write_json(output/'run_metadata.json',state); raise


def finetune_sanity(root,*,steps=5,device='cpu',output_name='partial_finetune_sanity_v1'):
    if device not in ('cpu','gpu') or not re.fullmatch('[a-z0-9_]+',output_name) or not 1<=steps<=5: raise ValueError('Bounded fine-tuning resource scope')
    if device=='cpu': tf=initialize_tensorflow()
    else:
        import tensorflow as tf
        if not tf.config.list_physical_devices('GPU'): raise ValueError('GPU runtime unavailable; no CPU fallback disguised as GPU')
    output=root/BASE/output_name; output.mkdir(parents=True,exist_ok=False)
    from .transforms import load_audio
    from .controls import prepare_headroom
    from .preprocessing import Preprocessor
    sources=read_csv(root/BASE/'data_selection_v1/manifest.csv'); records=[next(r for r in sources if r['split']=='train' and r['label']==str(c)) for c in (0,1)]
    tensors=[]; prep=perf_counter()
    for r in records:
        a=load_audio(root/r['path'],sample_rate=22050,max_seconds=3); a,_=prepare_headroom(a,.25); tensors.append(Preprocessor('P0').extract(a.samples))
    preparation=perf_counter()-prep; x=tf.convert_to_tensor(np.stack(tensors)); y=tf.constant([[0.],[1.]])
    if device=='cpu': full,backbone,head,_=build_student(seed=5806)
    else:
        tf.keras.utils.set_random_seed(5806)
        from src.model import build_hybrid_model
        full,backbone=build_hybrid_model(weights='imagenet')
    # Reuse production's existing, tested unfreeze policy; no production edit.
    from src.model import unfreeze_backbone_for_finetuning
    write_json(output/'progress.json',dict(stage='HEAD_PREFLIGHT_BATCH1',device=device))
    head_started=perf_counter()
    with tf.GradientTape() as tape: head_value=paired_objective(y[:1],full(x[:1],training=False))
    head_forward=perf_counter()-head_started; head_started=perf_counter(); hg=tape.gradient(head_value,full.trainable_weights); head_backward=perf_counter()-head_started
    unfreeze_backbone_for_finetuning(backbone)
    variables=list(full.trainable_weights); optimizer=tf.keras.optimizers.Adam(.0001)
    frozen=[v for v in full.weights if id(v) not in {id(w) for w in variables}]; before=[v.numpy().copy() for v in frozen]
    selected=[v for v in backbone.trainable_weights if 'kernel' in v.name]; selected_before=[v.numpy().copy() for v in selected]
    if not selected: raise ValueError('No trainable backbone kernels')
    bn=[l for l in backbone.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
    history=[]; started=perf_counter(); gradients_by_name={}
    for i in range(steps):
        write_json(output/'progress.json',dict(stage='PARTIAL_FINETUNE_MICROBATCH1',step=i+1))
        gradients=[tf.zeros_like(v) for v in variables]; ft=0.; bt=0.; values=[]
        for j in range(2):
            f=perf_counter()
            with tf.GradientTape() as tape: value=paired_objective(y[j:j+1],full(x[j:j+1],training=False))*.5
            ft+=perf_counter()-f; b=perf_counter(); micro=tape.gradient(value,variables)
            if any(g is None or not np.isfinite(g.numpy()).all() for g in micro): raise ValueError('Missing/nonfinite fine-tuning gradient')
            gradients=[a+g for a,g in zip(gradients,micro)]; bt+=perf_counter()-b; values.append(float(value))
        b=perf_counter(); optimizer.apply_gradients(zip(gradients,variables)); bt+=perf_counter()-b
        gradients_by_name={v.name:float(tf.linalg.norm(g)) for v,g in zip(variables,gradients)}
        history.append(dict(step=i+1,loss=sum(values),forward_seconds=ft,backward_seconds=bt))
        if perf_counter()-started>90: raise TimeoutError('Fine-tuning sanity budget90s')
    if not all(np.array_equal(a,v.numpy()) for a,v in zip(before,frozen)): raise ValueError('Frozen weights/BN state changed')
    prediction=np.concatenate([full(x[j:j+1],training=False).numpy() for j in range(2)])
    full.save(output/'sanity.keras'); loaded=tf.keras.models.load_model(output/'sanity.keras',compile=False)
    restored=np.concatenate([loaded(x[j:j+1],training=False).numpy() for j in range(2)])
    delta=float(np.max(np.abs(restored-prediction)))
    changed=[v.name for a,v in zip(selected_before,selected) if not np.array_equal(a,v.numpy())]
    if not changed or delta>1e-6 or not all(not l.trainable for l in bn): raise ValueError('Fine-tuning verification failed')
    write_csv(output/'training_history.csv',history)
    report=dict(status='SMOKE_TESTED',scope='OVERFIT_SANITY_RESOURCE_PREFLIGHT',training_policy='PARTIAL_BACKBONE_FINETUNE',seed=5806,
        sources=[r['sample_id'] for r in records],data_role='TRAIN_ONLY',steps=steps,batch=1,gradient_accumulation=2,effective_batch=2,preprocessing_seconds=preparation,
        head_only_forward_seconds=head_forward,head_only_backward_seconds=head_backward,head_only_gradient_norm=float(tf.linalg.global_norm(hg)),
        elapsed_seconds=perf_counter()-started,trainable_parameters=int(sum(np.prod(v.shape) for v in variables)),total_parameters=full.count_params(),
        selected_backbone_kernels=[v.name for v in selected],changed_backbone_kernels=changed,gradients=gradients_by_name,
        frozen_variables_unchanged=len(frozen),batchnorm_frozen=len(bn),save_load_max_residual=delta,peak_ram_mib=peak_process_mib(),
        gpu_available=bool(tf.config.list_physical_devices('GPU')),checkpoint_bytes=(output/'sanity.keras').stat().st_size,
        initial_loss=history[0]['loss'],final_loss=history[-1]['loss'],limitations='Two train recordings / five updates; no development quality comparison; prefix/middle layers intentionally frozen')
    report['output_hashes']={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}; write_json(output/'run_metadata.json',report); return report
