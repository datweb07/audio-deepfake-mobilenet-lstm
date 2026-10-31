"""Create a separately registered config identity without editing prior protocols."""
import argparse,json,re
from pathlib import Path
from src.lava.jaes_v2.phase8_training import ExperimentConfig,BASE
from src.lava.jaes_v2.evidence import object_hash,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--experiment-id',required=True);p.add_argument('--runtime',choices=('cpu','gpu'),required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    if not re.fullmatch(r'p8_[a-z0-9_]+',args.experiment_id):raise ValueError('Phase8 identity required')
    config=json.loads(args.source.read_text());config['experiment_id']=args.experiment_id;config['runtime']=args.runtime
    protocol=root/BASE/'protocol'/(args.experiment_id+'.json');config['protocol_path']=protocol.relative_to(root).as_posix()
    out=root/'configs/jaes_v2/phase8'/(args.experiment_id+'.json')
    if out.exists() or protocol.exists():raise FileExistsError('Existing identity/protocol preserved')
    ExperimentConfig(**config);write_json(out,config)
    write_json(protocol,dict(config_hashes={args.experiment_id:object_hash(config)},calibration_enabled=False,evaluation_enabled=False,
        reason='Explicit new identity/runtime port; does not inherit measured runtime verification'))
    print(out)

if __name__=='__main__':main()
