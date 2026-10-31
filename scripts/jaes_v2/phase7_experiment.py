"""Sequential Phase7 research stages with explicit versioned configs."""
import argparse,json,os,tempfile
from datetime import datetime,timezone
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2]; temporary=root/'.research_tmp'; temporary.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temporary); os.environ['TMP']=str(temporary); tempfile.tempdir=str(temporary)
    mpl=temporary/'matplotlib'; mpl.mkdir(exist_ok=True); os.environ['MPLCONFIGDIR']=str(mpl)
    from scripts.jaes_v2.phase6_experiment import resource_lock
    from src.lava.jaes_v2.phase7_data import select,build_cache
    from src.lava.jaes_v2.phase7_training import Config,train
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=('select','benchmark','cache','train','finetune-preflight'))
    parser.add_argument('--config',type=Path); args=parser.parse_args()
    with resource_lock(root):
        # OS byte-range lock is released by the kernel on process death. This
        # owner record is audit information, never a lock-deletion heuristic.
        from src.lava.jaes_v2.evidence import write_json
        jobs=root/'outputs/jaes_v2/phase7/job_owners';jobs.mkdir(exist_ok=True)
        owner=jobs/(str(os.getpid())+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
        write_json(owner,dict(pid=os.getpid(),mode=args.mode,config=str(args.config),started_utc=datetime.now(timezone.utc).isoformat(),lock='Existing Phase6 OS-enforced byte-range mutex; process exit releases ownership; no manual stale-lock deletion'))
        if args.mode=='select': result=select(root)
        else:
            if not args.config: parser.error('--config required')
            config=json.loads(args.config.read_text())
            if args.mode in ('benchmark','cache'):
                if args.mode=='cache':
                    gate=json.loads((root/'outputs/jaes_v2/phase7/cache_resource_gate_v1.json').read_text())
                    if gate['status']!='PASS' or gate['cache_config_sha256']!=__import__('src.lava.jaes_v2.evidence',fromlist=['file_hash']).file_hash(args.config): raise ValueError('Measured cache gate mismatch')
                result=build_cache(root,config,benchmark_sources=2 if args.mode=='benchmark' else None)
                if args.mode=='cache':
                    from src.lava.jaes_v2.phase7_cache_audit import audit
                    result['post_transform_audit']=audit(root,config)
            else: result=train(root,Config(**config),preflight_only=args.mode=='finetune-preflight')
    print(json.dumps({k:v for k,v in result.items() if k not in ('code','output_hashes')},indent=2))
if __name__=='__main__': main()
