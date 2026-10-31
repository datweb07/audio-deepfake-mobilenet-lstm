"""Only tiny lifecycle pilots; never executes a medium/foundation config."""
from pathlib import Path
import argparse,json,subprocess,sys
from src.lava.jaes_v2.evidence import write_json,object_hash,file_hash
from src.lava.jaes_v2.phase8_training import ExperimentConfig,BASE,verify_tree
VERSION='v2'

def register(root):
    folder=root/'configs/jaes_v2/phase8';protocol=root/BASE/'protocol'/f'tiny_{VERSION}.json';configs=[]
    for name in ('erm_head','weighted_head','dro_head','erm_partial'):
        original=json.loads((folder/f'p8_tiny_{name}_s6107_v1.json').read_text())
        for suffix,seed in ([('',6107),('_resume',6107),('',6207),('',6307)] if name=='erm_head' else [('',6107),('_resume',6107)] if name=='erm_partial' else [('',6107)]):
            c=dict(original);c['experiment_id']=f'p8_tiny_{name}_s{seed}{suffix}_{VERSION}';c['training_seed']=seed;c['protocol_path']=protocol.relative_to(root).as_posix()
            ExperimentConfig(**c);configs.append(c)
    if protocol.exists():return json.loads(protocol.read_text())['config_hashes']
    for c in configs:write_json(folder/(c['experiment_id']+'.json'),c)
    write_json(protocol,dict(config_hashes={c['experiment_id']:object_hash(c) for c in configs},calibration_enabled=False,evaluation_enabled=False,
        scope='12train/14development tiny lifecycle; no statistical quality claim',resume_test='fresh process interruption after1update vs continuous4updates'))
    return {c['experiment_id']:object_hash(c) for c in configs}

def run(root, identifiers):
    logs=root/BASE/'pilot_logs';logs.mkdir(exist_ok=True)
    for identifier in identifiers:
        out=root/BASE/'training'/identifier
        if (out/'seal.json').exists():verify_tree(out);continue
        def command(mode,extra=()):
            cli=[sys.executable,'-X','utf8','-m','scripts.jaes_v2.phase8_experiment',mode]
            cli+=['--run',str(out)] if mode=='resume' else ['--config',str(root/'configs/jaes_v2/phase8'/(identifier+'.json'))]
            path=logs/(identifier+'_'+mode+'.txt')
            with path.open('x',encoding='utf-8') as log:
                result=subprocess.run(cli+list(extra),cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{identifier} failed; inspect {path}')
        if '_resume_' in identifier:
            command('train',('--stop-after-updates','1'));command('resume')
        else:command('train')
        verify_tree(out);print(identifier+' SEALED',flush=True)

def compare(root):
    results={}
    for policy in ('head','partial'):
        runs=[root/BASE/'training'/f'p8_tiny_erm_{policy}_s6107{suffix}_{VERSION}' for suffix in ('','_resume')]
        for p in runs:verify_tree(p)
        resources=[json.loads((p/'resource_measurements.json').read_text()) for p in runs]
        states=[json.loads((p/'checkpoints'/json.loads((p/'latest.json').read_text())['checkpoint']/'controller.json').read_text()) for p in runs]
        journals=[[json.loads(q.read_text()) for q in sorted((p/'updates').glob('*.json'))] for p in runs]
        checks=dict(final_weights=resources[0]['final_weights_hash']==resources[1]['final_weights_hash'],
            optimizer=resources[0]['optimizer_hash']==resources[1]['optimizer_hash'],
            sample_order_views_losses_lr=journals[0]==journals[1],step=states[0]['step']==states[1]['step'],
            counter=states[0]['micro_counter']==states[1]['micro_counter'],history=states[0]['history']==states[1]['history'],
            export_scores=file_hash(runs[0]/'scores.csv')==file_hash(runs[1]/'scores.csv'))
        results[policy]=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
            numerical_tolerance=0,scope='CPU TF2.15 same environment; exact optimizer boundaries; fresh processes; warmup/partial transition included' if policy=='partial' else 'CPU TF2.15 same environment; fresh processes',
            final_weights_hash=resources[0]['final_weights_hash'],optimizer_hash=resources[0]['optimizer_hash'],steps=states[0]['step'])
    path=root/BASE/'audit'/f'resume_integration_{VERSION}.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,results)
    if any(x['status']!='PASS' for x in results.values()):raise ValueError('Resume parity failed')
    print(json.dumps(results,indent=2))

def main():
    global VERSION
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','head','partial','compare'));parser.add_argument('--version',choices=('v2','v3'),default='v2');args=parser.parse_args();VERSION=args.version
    root=Path(__file__).resolve().parents[2]
    ids=register(root)
    if args.mode in ('head','partial'):run(root,[name for name in ids if f'_{args.mode}_' in name])
    elif args.mode=='compare':compare(root)
    else:print(json.dumps(list(ids),indent=2))

if __name__=='__main__':main()
