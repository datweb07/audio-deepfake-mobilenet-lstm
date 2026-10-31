"""Register explicit baseline families before training; never launches jobs."""
from pathlib import Path
from dataclasses import asdict
from src.lava.jaes_v2.evidence import file_hash,write_json,object_hash
from src.lava.jaes_v2.phase8_training import ExperimentConfig,MEDIUM,FOUNDATION,BASE

def main():
    root=Path(__file__).resolve().parents[2];folder=root/'configs/jaes_v2/phase8';folder.mkdir(parents=True,exist_ok=True)
    protocol=root/BASE/'protocol/baselines_v1.json'
    if protocol.exists():raise FileExistsError(protocol)
    hashes={};configs={}
    for scope,source in [('tiny',MEDIUM),('medium',MEDIUM),('foundation',FOUNDATION)]:
        for method in ('erm','weighted','dro','channel','equal','consistency','rawboost'):
            for seed in ((6107,) if scope=='tiny' else (6107,6207,6307)):
                for policy in (('head','partial') if scope!='foundation' else ('partial',)):
                    identifier=f'p8_{scope}_{method}_{policy}_s{seed}_v1'
                    development=source if source==MEDIUM else source.replace('train_manifest','development_manifest')
                    values=dict(experiment_id=identifier,research_phase=8,dataset_version='ASVspoof2019_LA',source_manifest=source,
                        manifest_sha256=file_hash(root/source),development_manifest=development,development_sha256=file_hash(root/development),
                        training_scope='TINY_LIFECYCLE' if scope=='tiny' else 'DEVELOPMENT_RESEARCH',architecture='MobileNetV3Small-LSTM128',
                        pretrained_weights='imagenet_fresh_head',preprocessing_policy='P0',augmentation_policy='clean' if method=='erm' else 'rawboost_finite' if method=='rawboost' else 'channel',
                        training_objective={'weighted':'WEIGHTED_BCE','dro':'GROUP_DRO','equal':'EQUAL_VIEW','consistency':'CONSISTENCY'}.get(method,'ERM'),
                        objective_parameters=dict(class_weights=[1.,1.5],dro_rate=.01,dro_floor=.05,consistency_weight=.1),
                        trainable_backbone_policy='HEAD_ONLY' if policy=='head' else 'WARMUP_PARTIAL',BatchNorm_policy='ALL_FROZEN',
                        training_seed=seed,augmentation_seed=5804,sampling='group_balanced' if method=='dro' else 'natural',
                        learning_rate=.001,finetune_learning_rate=.0001,scheduler='linear_half_per_stage',batch_size=8 if scope=='tiny' else 16,
                        microbatch_size=1 if policy=='partial' else 4,gradient_accumulation=(8 if scope=='tiny' else 16) if policy=='partial' else (2 if scope=='tiny' else 4),
                        warmup_epochs=1 if policy=='partial' else 2 if scope=='tiny' else 20,
                        finetune_epochs=(1 if scope=='tiny' else 20) if policy=='partial' else 0,
                        maximum_updates=100 if scope=='tiny' else 100000,checkpoint_policy='CLEAN_BCE',checkpoint_every_updates=1 if scope=='tiny' else 500,
                        validation_policy='A_EVERY_EPOCH_ALL_CONDITIONS_FINAL_ALL_DEV',early_stopping_patience=5,
                        clean_REAL_FA_cap=.10,clean_FAKE_miss_cap=.15,
                        resource_limits=dict(seconds_per_invocation=900 if scope=='tiny' else 86400,disk_reserve_gib=15,max_process_mib=4000,tensor_lru_items=8),
                        runtime='cpu',protocol_path=protocol.relative_to(root).as_posix(),tiny_per_attack=1 if scope=='tiny' else 0)
                    config=asdict(ExperimentConfig(**values));configs[identifier]=config
    for variant in ('balanced','guarded'):
        c=dict(configs['p8_foundation_channel_partial_s6107_v1']);c['experiment_id']=c['experiment_id'].replace('channel_',f'channel_{variant}_')
        if variant=='balanced':c['sampling']='class_balanced'
        else:c['checkpoint_policy']='CLEAN_RISK_CONSTRAINED_ROBUSTNESS'
        ExperimentConfig(**c);configs[c['experiment_id']]=c
    # Independent resume counterpart uses same seed/data/objective, a separate run identity.
    for policy in ('head','partial'):
        c=dict(configs[f'p8_tiny_erm_{policy}_s6107_v1']);c['experiment_id']=c['experiment_id'].replace('_v1','_resume_v1');configs[c['experiment_id']]=c
    for identifier,c in configs.items():
        path=folder/(identifier+'.json')
        if path.exists():raise FileExistsError(path)
        write_json(path,c);hashes[identifier]=object_hash(c)
    protocol.parent.mkdir(parents=True,exist_ok=True)
    write_json(protocol,dict(config_hashes=hashes,calibration_enabled=False,evaluation_enabled=False,
        scope='Development baselines only; no M3; no automatic full training',selection_threshold=.5,
        primary_endpoint='Both classwise stressed errors; paired seed/speaker effects descriptive',
        guardrails='Absolute caps only in guarded config; comparative clean deltas require matched baseline',
        augmentation='Fixed waveform bank5804; view RNG is separate. No online RawBoost claim.'))
    print(f'{len(hashes)} registered configurations; no training launched')

if __name__=='__main__':main()
