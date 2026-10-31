"""Phase8 training, recovery and read-only verification CLI."""
import argparse, json, os, tempfile, shutil, platform, math
from pathlib import Path
from src.lava.jaes_v2.evidence import write_json, object_hash
from src.lava.jaes_v2.phase8_training import load_config, dataset, train, verify_tree, BASE

def preflight(root, c):
    from src.lava.jaes_v2.phase8_training import initialize, code_contract, epoch_order
    from src.lava.jaes_v2.runtime import preflight as historical_preflight
    historical = historical_preflight(root)
    tr, dv, b, audit = dataset(root,c); tf = initialize(c.runtime)
    updates = sum(math.ceil(len(epoch_order(tr,c,e))/c.batch_size) for e in range(c.warmup_epochs+c.finetune_epochs))
    count = math.ceil(updates/c.checkpoint_every_updates)+2*(c.warmup_epochs+c.finetune_epochs)
    projected = count*24*1024**2+(c.warmup_epochs+c.finetune_epochs)*8*1024**2+256*1024**2
    free = shutil.disk_usage(root).free
    result = dict(config_hash=object_hash(c.__dict__),source=code_contract(root),historical_baseline=historical['baseline_freeze_verification'],
        train_sources=len(tr),development_sources=len(dv),REAL_only_development=len(b),partition_audit=audit,
        optimizer_updates=updates,maximum_updates_sufficient=c.maximum_updates>=updates,projected_retained_bytes=projected,
        projected_storage_status='CONSERVATIVE_EXTRAPOLATION_NOT_MEASUREMENT',free_D_bytes=free,free_C_bytes=shutil.disk_usage(Path.home().anchor).free,
        disk_gate=free>=c.resource_limits['disk_reserve_gib']*1024**3+projected,tensorflow=tf.__version__,
        tensorflow_GPU=[str(d) for d in tf.config.list_physical_devices('GPU')],
        GPU_runtime_status='GPU_RUNTIME_NOT_VERIFIED',calibration_accessed=False,evaluation_accessed=False)
    result['status']='PASS' if result['disk_gate'] and result['maximum_updates_sufficient'] else 'NO_GO'
    return result

def main():
    root=Path(__file__).resolve().parents[2]; temp=root/'.research_tmp';temp.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp);tempfile.tempdir=str(temp)
    os.environ['MPLCONFIGDIR']=str(temp/'matplotlib')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','prepare-data','train','resume','verify','monitor','aggregate','validate'))
    parser.add_argument('--config',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--stop-after-updates',type=int)
    args=parser.parse_args()
    # Atomic JSON pointers can be read during training; monitoring must never
    # wait for the exclusive training mutex held by the process being monitored.
    if args.mode=='monitor':
        if not args.run:parser.error('--run required')
        run=args.run.resolve()
        if not run.is_relative_to((root/BASE/'training').resolve()):raise ValueError('Phase8 run only')
        result={p.name:json.loads(p.read_text()) for p in (run/'status.json',run/'latest.json',run/'best.json') if p.exists()}
        journals=sorted((run/'updates').glob('*.json'))[-10:]
        if len(journals)>1:
            import statistics
            intervals=[b.stat().st_mtime-a.stat().st_mtime for a,b in zip(journals,journals[1:])]
            result['observed_wall_seconds_per_journal_update']=statistics.median(intervals)
            c=json.loads((run/'resolved_config.json').read_text());n=len(json.loads((run/'dataset_audit.json').read_text())['source_ids'])
            planned=math.ceil(n/c['batch_size'])*(c['warmup_epochs']+c['finetune_epochs'])
            step=json.loads(journals[-1].read_text())['step']
            result['rough_remaining_update_wall_seconds']=max(0,planned-step)*statistics.median(intervals)
            result['ETA_limit']='Filesystem wall intervals include validation/pauses; exclude future final evaluation; not a completion promise'
        print(json.dumps(result,indent=2));return
    from scripts.jaes_v2.phase6_experiment import resource_lock
    with resource_lock(root):
        if args.mode=='validate':
            if not args.output:parser.error('--output required')
            from scripts.jaes_v2 import validate
            import sys
            sys.argv=['validate','--output',str(args.output)];validate.main();return
        if args.mode in ('verify','monitor'):
            if not args.run:parser.error('--run required')
            run=args.run.resolve()
            if not run.is_relative_to((root/BASE/'training').resolve()):raise ValueError('Phase8 run only')
            if args.mode=='verify':result=verify_tree(run)
            else:result={p.name:json.loads(p.read_text()) for p in (run/'status.json',run/'latest.json',run/'best.json') if p.exists()}
        elif args.mode=='aggregate':
            from scripts.jaes_v2.phase8_report import aggregate
            if not args.output:parser.error('--output required')
            result=aggregate(root,args.output)
        else:
            if args.mode=='resume':
                if not args.run:parser.error('--run required')
                args.config=args.run/'resolved_config.json'
            if not args.config:parser.error('--config required')
            c=load_config(root,args.config)
            if args.mode in ('preflight','prepare-data'):
                if not args.output:parser.error('--output required')
                result=preflight(root,c);args.output.parent.mkdir(parents=True,exist_ok=True)
                if args.output.exists():raise FileExistsError(args.output)
                write_json(args.output,result)
            else:
                if args.mode=='train' and preflight(root,c)['status']!='PASS':raise RuntimeError('Resource preflight NO_GO')
                result=train(root,c,resume=args.mode=='resume',stop_after=args.stop_after_updates)
                result={k:result[k] for k in ('step','epoch','best','finished')}
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
