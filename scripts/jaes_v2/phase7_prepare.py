"""Create fresh registered Phase7 configurations before observing training outcomes."""
import json
from dataclasses import asdict
from pathlib import Path
from src.lava.jaes_v2.evidence import write_json,file_hash
from src.lava.jaes_v2.phase7_training import Config

def main():
    root=Path(__file__).resolve().parents[2]; base=root/'outputs/jaes_v2/phase7'; configs=root/'configs/jaes_v2/phase7'; configs.mkdir(exist_ok=False)
    selection='outputs/jaes_v2/phase7/selections/medium_all_attack_v1'; meta=json.loads((root/selection/'selection.json').read_text())
    cache=dict(selection_path=selection,selection_sha256=meta['manifest_sha256'],cache_id='medium_p0_v1',policies=['P0'],waveform_version='c2-view-1-rb5-finite2',maximum_seconds=2400,disk_reserve_gib=15)
    write_json(configs/'cache_medium_v1.json',cache); write_json(configs/'cache_benchmark_v1.json',{**cache,'cache_id':'benchmark_p0_v1','maximum_seconds':120})
    for seed in (6107,6207,6307):
        for method in ('H0','H1','H2','H3','H4'):
            c=Config(experiment_id=f'p7_{method.lower()}_s{seed}_v1',source_manifest=selection+'/manifest.csv',manifest_sha256=meta['manifest_sha256'],selection_id=meta['selection_id'],selection_seed=meta['seed'],cache_config='configs/jaes_v2/phase7/cache_medium_v1.json',training_seed=seed,augmentation_policy=method)
            write_json(configs/(c.experiment_id+'.json'),asdict(c))
    for name,policy in [('f0','HEAD_ONLY'),('f1','PARTIAL_BACKBONE_FINETUNE')]:
        c=Config(experiment_id=f'p7_{name}_s6107_v1',source_manifest=selection+'/manifest.csv',manifest_sha256=meta['manifest_sha256'],selection_id=meta['selection_id'],selection_seed=meta['seed'],cache_config='configs/jaes_v2/phase7/cache_medium_v1.json',training_seed=6107,augmentation_policy='H0',training_policy=policy,epochs=1,optimizer_updates=40,batch_size=1,gradient_accumulation=16,learning_rate=.0001,maximum_seconds=3000)
        write_json(configs/(c.experiment_id+'.json'),asdict(c))
    write_json(base/'protocol_v1.json',dict(status='LOCKED_BEFORE_PHASE7_TRAINING',date='2026-10-31 Asia/Novosibirsk',selection_hash=meta['manifest_sha256'],
        training_seeds=[6107,6207,6307],data_selection_seed=7107,primary_comparison=['H3','H2'],primary_metric='Equal-speaker mean FAKE miss over noise5,A_then_B,B_then_A; stratumA only',
        clean_REAL_FA_margin=.02,clean_FAKE_miss_margin=.02,margin_rationale='Development screen tolerates at most2 additional errors in each112-record class (2/112<.02); a tighter2pp bound than Phase6 1/28; NOT population risk control',
        epochs=6,representation_epochs=1,head_updates=240,representation_updates=40,maximum_run_seconds=3000,maximum_head_seconds=240,
        fixed_threshold=.5,ranking_metrics=['ROC_AUC','PR_AUC','EER'],development_threshold_fit='NOT_EXECUTED',
        checkpoint_selection='A_clean_BCE; B_REAL_ONLY not used for checkpoint selection',confirmation_enabled=False,calibration_enabled=False,new_method_enabled=False,
        multi_seed_gate='After all five6107 runs pass seals, initialization/view parity, full prediction coverage and measured aggregate training budget<1200seconds',
        representation_gate='Measured microbatch1accum16 preflight including projected evaluation fits3000seconds; F0/F1 same1epoch/LR/order/initialization; three-seedFT NOT_AUTHORIZED_BY_GATE',
        online_RawBoost='Adapter tested; main H5 quality run only if measured additional extraction/training budget fits; finite views explicitly distinguished',
        inference='All exploratory; seven speaker clusters LIMITED_CLUSTER_PRECISION; paired seed std separate from descriptive speaker bootstrap; no significance claim',
        seed_ledger='Model/dropout seed=s; order=s+100000; view=s+200000; online=s+300000; finite waveforms c2-view-1 uses5804+sampleID (shared across blocks)',
        configuration_hashes={p.name:file_hash(p) for p in sorted(configs.glob('p7_*.json'))}))
    print('Created',len(list(configs.glob('*.json'))),'configs; protocol',file_hash(base/'protocol_v1.json'))
if __name__=='__main__': main()
