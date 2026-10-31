"""Bounded independent export, warmup/BN and initialization checks on real pilots."""
import json,os,tempfile,argparse
from pathlib import Path
import numpy as np
from src.lava.jaes_v2.evidence import write_json,file_hash
from src.lava.jaes_v2.phase8_training import BASE,initialize,load_config,dataset,Inputs,verify_tree

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',choices=('v2','v3'),default='v3');args=parser.parse_args();version=args.version
    root=Path(__file__).resolve().parents[2];temp=root/'.research_tmp';os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp);tempfile.tempdir=str(temp)
    from scripts.jaes_v2.phase6_experiment import resource_lock
    with resource_lock(root):
        tf=initialize('cpu');results={}
        for policy in ('head','partial'):
            run=root/BASE/'training'/f'p8_tiny_erm_{policy}_s6107_{version}';verify_tree(run)
            c=load_config(root,run/'resolved_config.json');tr,dv,b,_=dataset(root,c)
            best=json.loads((run/'best.json').read_text());loaded=tf.keras.models.load_model(run/best['path'],compile=False)
            from src.model import build_hybrid_model,unfreeze_backbone_for_finetuning
            tf.keras.utils.set_random_seed(6107);fresh,backbone=build_hybrid_model(weights='imagenet')
            initialized={v.name:v.numpy().copy() for v in backbone.weights}
            # Inspect first warmup export, which exists in this tiny pilot.
            warm=tf.keras.models.load_model(run/'best_epoch_001.keras',compile=False)
            warm_backbone=warm.get_layer('time_distributed_mobilenetv3small').layer
            warmup_unchanged=all(np.array_equal(initialized[v.name],v.numpy()) for v in warm_backbone.weights)
            selected_backbone=loaded.get_layer('time_distributed_mobilenetv3small').layer
            changed=[v.name for v in selected_backbone.weights if not np.array_equal(initialized[v.name],v.numpy())]
            bn=[l for l in selected_backbone.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
            bn_unchanged=all(np.array_equal(initialized[v.name],v.numpy()) for layer in bn for v in layer.weights)
            r=next(x for x in dv if x.sample_id not in b)
            from src.lava.jaes_v2.transforms import load_audio
            from src.lava.jaes_v2.controls import prepare_headroom
            from src.lava.jaes_v2.preprocessing import Preprocessor
            from src.lava.jaes_v2.phase8_training import source_path
            audio=load_audio(source_path(root,r.path),sample_rate=22050,max_seconds=3);audio,_=prepare_headroom(audio,.25)
            tensor=Preprocessor('P0').extract(audio.samples)[None]
            eager=float(loaded(tensor,training=False).numpy()[0,0])
            compiled=float(tf.function(lambda x:loaded(x,training=False))(tensor).numpy()[0,0])
            from src.lava.jaes_v2.evidence import read_csv
            logged=next(row for row in read_csv(run/'scores.csv') if row['sample_id']==r.sample_id and row['condition_id']=='clean')
            residual=abs(eager-float(logged['raw_score']))
            results[policy]=dict(warmup_backbone_unchanged=warmup_unchanged,BN_unchanged=bn_unchanged,BN_layers=len(bn),
                changed_backbone_variables=changed,full_export_vs_logged_score_residual=residual,eager_compiled_residual=abs(eager-compiled),
                export_sha256=file_hash(run/best['path']))
            if not warmup_unchanged or not bn_unchanged or residual>1e-6 or abs(eager-compiled)>1e-6 or (policy=='partial' and not changed):raise ValueError('Representation/export gate failed')
        seeds={}
        for seed in (6107,6207,6307):
            run=root/BASE/'training'/f'p8_tiny_erm_head_s{seed}_{version}';verify_tree(run)
            seeds[str(seed)]=json.loads((run/'trainable_layers.json').read_text())['initialization_hash']
        if len(set(seeds.values()))!=3:raise ValueError('Independent seed initialization failed')
        out=root/BASE/'audit'/f'representation_export_seed_checks_{version}.json'
        if out.exists():raise FileExistsError(out)
        write_json(out,dict(status='PASS',representation=results,seed_initialization_hashes=seeds))
        print(json.dumps(results,indent=2))

if __name__=='__main__':main()
