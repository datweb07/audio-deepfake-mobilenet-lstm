"""Budgeted baseline lifecycle on a frozen ImageNet backbone; no evaluation access."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score

from .c2_student import build_student,initialize_tensorflow,objective
from .evidence import file_hash,object_hash,read_csv,write_json,write_csv
from .phase3 import verify_seal
from .public_analysis import metrics
from .runtime import source_fingerprint


@dataclass(frozen=True)
class TrainingConfig:
    experiment_id: str
    cache: str='outputs/jaes_v2/c2_feature_cache_v1'
    policy: str='P0'
    baseline: str='B0'
    initialization: str='imagenet'
    trainable_scope: str='temporal_head_only'
    seed: int=5804
    epochs: int=20
    batch_size: int=16
    learning_rate: float=.001
    consistency_weight: float=.1
    maximum_seconds: float=180
    phase: str='TIER2_RESEARCH_PILOT'
    threshold: float=.5

    def __post_init__(self):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,70}',self.experiment_id) or self.policy not in {'P0','P1'} or self.baseline not in {'B0','B1','B2','B3','B4'}:
            raise ValueError('Invalid separately versioned research baseline identity')
        if self.initialization!='imagenet' or self.trainable_scope!='temporal_head_only':
            raise ValueError('Frozen feature cache cannot train a scratch or finetuned backbone')
        if self.phase not in {'TIER1_OVERFIT_SANITY','TIER2_RESEARCH_PILOT'} or not 1<=self.epochs<=100 or not 1<=self.batch_size<=32:
            raise ValueError('Only bounded Phase 5 sanity/pilot training is permitted')
        if not 0<self.learning_rate<=.01 or not 0<=self.consistency_weight<=1 or not 0<self.maximum_seconds<=300 or self.threshold!=.5:
            raise ValueError('Invalid fixed pilot optimizer/budget/threshold policy')


def validate_views(rows: list[dict]) -> None:
    partitions={}
    seen=set()
    labels={}
    pair_identity={}; source_identity={}
    for row in rows:
        if row['split'] not in {'train','development'}:
            raise ValueError('No calibration/evaluation/test rows in training or selection')
        key=(row['sample_id'],row['policy'],row['condition'])
        if key in seen:
            raise ValueError('Duplicate paired sample/view')
        seen.add(key)
        if row['sample_id'] in partitions and partitions[row['sample_id']]!=row['split']:
            raise ValueError('Cross-split paired sample leakage')
        partitions[row['sample_id']]=row['split']
        value=int(row['label'])
        if value not in (0,1) or (row['sample_id'] in labels and labels[row['sample_id']]!=value):
            raise ValueError('Inconsistent REAL/FAKE labels')
        labels[row['sample_id']]=value
        identity=tuple(row.get(field) for field in ('source_checksum','transform_seed','waveform_sha256'))
        pair=(row['sample_id'],row['condition'])
        if pair in pair_identity and pair_identity[pair]!=identity:
            raise ValueError('Preprocessor policies do not share the same source/seed/waveform view')
        pair_identity[pair]=identity
        source=(row.get('source_checksum'),row.get('speaker_id'),row.get('attack_id'))
        if row['sample_id'] in source_identity and source_identity[row['sample_id']]!=source:
            raise ValueError('Source identity changed across paired views')
        source_identity[row['sample_id']]=source


def evaluate(head,values: np.ndarray,labels: np.ndarray,records: list[dict],*,condition: str,threshold: float=.5) -> tuple[list[dict],dict]:
    probabilities=head.predict(values,batch_size=16,verbose=0).reshape(-1)
    if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Invalid FAKE-oriented student scores')
    rows=[{'sample_id':r['sample_id'],'speaker_id':r['speaker_id'],'attack_id':r['attack_id'],'label':int(y),
           'raw_score':float(p),'predicted_label':int(p>=threshold),'threshold':threshold,'condition_id':condition,
           'calibrated_score':'NOT_FITTED'} for r,y,p in zip(records,labels,probabilities)]
    result=metrics(rows)
    predicted=probabilities>=threshold
    result.update(accuracy=float(accuracy_score(labels,predicted)),precision=float(precision_score(labels,predicted,zero_division=0)),
                  recall=float(recall_score(labels,predicted,zero_division=0)),specificity=1-result['false_alarm_rate'],
                  macro_f1=float(f1_score(labels,predicted,average='macro',zero_division=0)),
                  pr_auc=float(average_precision_score(labels,probabilities)),condition=condition)
    return rows,result


def train(root: Path,config: TrainingConfig) -> Path:
    cache=root/config.cache; metadata=json.loads((cache/'cache_metadata.json').read_text())
    if metadata['status']!='SEALED_C2_FEATURE_CACHE':
        raise ValueError('Complete data/resource gate required')
    verify_seal(cache,metadata)
    if file_hash(Path.home()/'.keras/models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5')!=metadata['pretraining_weight_sha256']:
        raise ValueError('Cached features and student ImageNet initialization differ')
    resource=root/'outputs/jaes_v2/audit/phase5_training_preflight_v2.json'
    if not resource.is_file() or json.loads(resource.read_text())['status']!='PASS':
        raise ValueError('Training resource forward/backward preflight required')
    if config.phase=='TIER2_RESEARCH_PILOT':
        sanity=root/'outputs/jaes_v2/c2/c2_overfit_sanity_v1/diagnostics.json'
        if not sanity.is_file() or not json.loads(sanity.read_text())['overfit_sanity_passed']:
            raise ValueError('Tiny real-data overfit gate has not passed')
    entries=read_csv(cache/'view_manifest.csv'); validate_views(entries)
    entries=[r for r in entries if r['policy']==config.policy]
    stored=np.load(cache/'embeddings.npz',allow_pickle=False)
    index={(r['sample_id'],r['condition']):r for r in entries}
    train_records=sorted((r for r in entries if r['split']=='train' and r['condition']=='clean'),key=lambda r:r['sample_id'])
    dev_records=sorted((r for r in entries if r['split']=='development' and r['condition']=='clean'),key=lambda r:r['sample_id'])
    if config.phase=='TIER1_OVERFIT_SANITY':
        train_records=[r for label in ('0','1') for r in [r for r in train_records if r['label']==label][:8]]
    train_labels=np.array([int(r['label']) for r in train_records],dtype='float32')
    dev_labels=np.array([int(r['label']) for r in dev_records],dtype='int32')
    def batch(records,conditions):
        return np.stack([stored[index[(r['sample_id'],condition)]['array_key']][int(index[(r['sample_id'],condition)]['array_index'])]
                         for r,condition in zip(records,conditions)])
    output=root/'outputs/jaes_v2/c2'/config.experiment_id; output.mkdir(parents=True,exist_ok=False)
    write_json(output/'resolved_config.json',asdict(config))
    environment={**metadata['environment'],'execution_policy':'Separately versioned CPU research head training; no conference retraining, calibration fitting or evaluation access'}
    write_json(output/'environment.json',environment)
    write_json(output/'data_manifest_hashes.json',{'cache_metadata_sha256':file_hash(cache/'cache_metadata.json'),'selection_sha256':metadata['selection_hash'],
                                                'calibration_accessed':False,'evaluation_accessed':False})
    full,backbone,head,_=build_student(seed=config.seed,initialization=config.initialization)
    tf=initialize_tensorflow(); optimizer=tf.keras.optimizers.Adam(config.learning_rate)
    rng=np.random.default_rng(config.seed); view_rng=np.random.default_rng(config.seed)
    started=perf_counter(); history=[]; best=float('inf'); optimizer_steps=0
    code=source_fingerprint(root)
    state={'status':'TRAINING','scope':config.phase,'config_hash':object_hash(asdict(config)),'code':code,
           'research_model_id':'jaes_v2_mobilenet_student','architecture_version':'existing-MobilenetV3Small-6xLSTM128',
           'pretraining_weight_sha256':metadata['pretraining_weight_sha256'],'preprocessing_version':'research-mel-floor-1',
           'augmentation_version':'c2-view-1','optimizer':'Adam','optimizer_version':tf.__version__,'weight_decay':0.,
           'loss':'BCE probabilities; standard MSE pair consistency for B4 only',
           'baseline_alias':'B3 merged into B2, same single-view bank' if config.baseline=='B3' else None,
           'frozen_backbone':not backbone.trainable,'inference_extra_modules':0,'teacher_used':False}
    write_json(output/'run_metadata.json',state)
    @tf.function
    def step(x,y,other):
        with tf.GradientTape() as tape:
            predictions=head(x,training=True)
            changed=head(other,training=True) if config.baseline=='B4' else None
            value=objective(y,predictions,changed,baseline=config.baseline,consistency_weight=config.consistency_weight)
        gradients=tape.gradient(value,head.trainable_weights)
        tf.debugging.assert_all_finite(value,'Nonfinite loss')
        optimizer.apply_gradients(zip(gradients,head.trainable_weights))
        return value,tf.linalg.global_norm(gradients)
    try:
        for epoch in range(config.epochs):
            if perf_counter()-started>config.maximum_seconds:
                state['budget_stopped_early']=True; break
            # Same declared linear LR schedule and updates across all four pilot cells.
            optimizer.learning_rate.assign(config.learning_rate*(1-.5*epoch/config.epochs))
            losses=[]; norms=[]
            for start in range(0,len(train_records),config.batch_size):
                if start==0:
                    order=rng.permutation(len(train_records))
                selected=[train_records[i] for i in order[start:start+config.batch_size]]
                labels=train_labels[order[start:start+config.batch_size]]
                if config.baseline=='B0': choices=['clean']*len(selected)
                elif config.baseline=='B1': choices=view_rng.choice(['clean','gain_half'],len(selected)).tolist()
                else: choices=view_rng.choice(['clean','gain_half','noise10','opus24'],len(selected)).tolist()
                x=batch(selected,choices); clean=batch(selected,['clean']*len(selected))
                value,norm=step(x,labels,clean); losses.append(float(value)); norms.append(float(norm)); optimizer_steps+=1
            clean_train=batch(train_records,['clean']*len(train_records))
            train_scores=head.predict(clean_train,batch_size=16,verbose=0).reshape(-1)
            accuracy=float(np.mean((train_scores>=.5)==train_labels))
            clean_dev=batch(dev_records,['clean']*len(dev_records)); dev_scores=head.predict(clean_dev,batch_size=16,verbose=0).reshape(-1)
            dev_loss=float(tf.keras.losses.binary_crossentropy(dev_labels.astype('float32'),dev_scores).numpy().mean())
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'train_accuracy':accuracy,'development_clean_bce':dev_loss,
                            'gradient_norm_mean':float(np.mean(norms)),'learning_rate':float(optimizer.learning_rate.numpy()),'optimizer_steps':optimizer_steps})
            write_csv(output/'training_history.csv',history)
            criterion=-accuracy if config.phase=='TIER1_OVERFIT_SANITY' else dev_loss
            if criterion<best:
                best=criterion; full.save(output/'best.keras')
                write_json(output/'checkpoint_selection.json',{'selected_epoch':epoch+1,'criterion':'TRAIN_ACCURACY_SANITY_ONLY' if config.phase=='TIER1_OVERFIT_SANITY' else 'DEVELOPMENT_CLEAN_BCE',
                    'value':criterion,'evaluation_labels_used':False,'checkpoint_sha256':file_hash(output/'best.keras')})
            print(f'{config.experiment_id}: epoch {epoch+1}, train acc {accuracy:.3f}, dev BCE {dev_loss:.4f}',flush=True)
            if config.phase=='TIER1_OVERFIT_SANITY' and accuracy>=.9375:
                break
        if not (output/'best.keras').exists():
            raise RuntimeError('No complete epoch/checkpoint; training cannot publish')
        # Reload the exact selected full model; head is reconstructed from its loaded layers.
        loaded=tf.keras.models.load_model(output/'best.keras',compile=False)
        inputs=tf.keras.Input((6,576)); x=inputs
        for name in ('temporal_lstm','classifier_dense','classifier_dropout','probability_fake'):
            x=loaded.get_layer(name)(x)
        selected_head=tf.keras.Model(inputs,x)
        scores=[]; results=[]
        for condition in ('clean','gain_quarter','noise5','opus64','A_then_B','B_then_A'):
            members,metric=evaluate(selected_head,batch(dev_records,[condition]*len(dev_records)),dev_labels,dev_records,condition=condition)
            scores.extend(members); results.append(metric)
        write_csv(output/'validation_scores.csv',scores); write_json(output/'metrics.json',results)
        sanity_accuracy=float(np.mean((selected_head.predict(clean_train,batch_size=16,verbose=0).reshape(-1)>=.5)==train_labels))
        write_json(output/'diagnostics.json',{'overfit_sanity_passed':config.phase=='TIER1_OVERFIT_SANITY' and sanity_accuracy>=.9375,
            'selected_train_accuracy':sanity_accuracy,'nonzero_gradients':any(row['gradient_norm_mean']>0 for row in history),
            'optimizer_steps':optimizer_steps,'backbone_frozen_pretrained':True,'no_test_scores':'No evaluation was run',
            'limitations':'Single-seed tiny frozen-ImageNet-backbone pilot, not full finetuning or a new learned method'})
        from .c2_student import peak_process_mib
        state.update(status='SMOKE_TESTED' if config.phase=='TIER1_OVERFIT_SANITY' else 'PILOT_EVALUATED',
                     elapsed_seconds=perf_counter()-started,checkpoint_sha256=file_hash(output/'best.keras'),parameters=full.count_params(),
                     model_bytes=(output/'best.keras').stat().st_size,optimizer_steps=optimizer_steps,os_peak_working_set_mib=peak_process_mib())
        state['output_hashes']={p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='run_metadata.json'}
        write_json(output/'run_metadata.json',state); return output
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=f'{type(error).__name__}: {error}',optimizer_steps=optimizer_steps)
        write_json(output/'run_metadata.json',state); raise


def aggregate(run: Path) -> list[dict]:
    metadata=json.loads((run/'run_metadata.json').read_text()); verify_seal(run,metadata)
    rows=read_csv(run/'validation_scores.csv'); results=[]
    for condition in sorted({r['condition_id'] for r in rows}):
        selected=[r for r in rows if r['condition_id']==condition]
        results.append({'condition':condition,**metrics(selected)})
    return results
