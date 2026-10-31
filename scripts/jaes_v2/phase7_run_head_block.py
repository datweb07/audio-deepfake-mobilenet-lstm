"""Run one registered seed block sequentially; verify first-block feasibility before expansion."""
import argparse,json,subprocess,sys
from pathlib import Path
from time import perf_counter
from src.lava.jaes_v2.evidence import file_hash,read_csv,write_json
from src.lava.jaes_v2.phase7_data import BASE
from src.lava.jaes_v2.phase3 import verify_seal

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=(6107,6207,6307),required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/BASE/'execution_logs';out.mkdir(exist_ok=True)
    if args.seed!=6107:
        gate=json.loads((root/BASE/'multi_seed_gate_v1.json').read_text())
        if gate['status']!='PASS':raise ValueError('First-block gate required')
    timings=[]
    for method in ('h0','h1','h2','h3','h4'):
        identifier=f'p7_{method}_s{args.seed}_v1'; config='configs/jaes_v2/phase7/'+identifier+'.json';log=out/(identifier+'.txt')
        print('Starting',identifier,flush=True);started=perf_counter()
        with log.open('x',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.jaes_v2.phase7_experiment','train','--config',config],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        elapsed=perf_counter()-started;timings.append(dict(experiment_id=identifier,process_wall_seconds=elapsed,returncode=result.returncode,log_sha256=file_hash(log)))
        write_json(out/f'seed{args.seed}_process_measurements.json',timings)
        if result.returncode:raise RuntimeError('Training failed; preserved log: '+str(log))
        print('Completed',identifier,f'{elapsed:.1f}s including startup',flush=True)
    if args.seed==6107:
        runs=[]
        for method in ('h0','h1','h2','h3','h4'):
            folder=root/BASE/'training'/f'p7_{method}_s6107_v1';m=json.loads((folder/'run_metadata.json').read_text());verify_seal(folder,m)
            if m['status']!='PILOT_EVALUATED' or m['optimizer_updates']!=240 or len(read_csv(folder/'scores.csv'))!=368*7:raise ValueError('Single-block completeness gate')
            runs.append(m)
        a=root/BASE/'training/p7_h2_s6107_v1';b=root/BASE/'training/p7_h3_s6107_v1'
        if file_hash(a/'training_views.csv')!=file_hash(b/'training_views.csv') or len({m['initial_model_weights_sha256'] for m in runs})!=1:raise ValueError('Initialization/view fairness gate')
        total=sum(r['elapsed_seconds'] for r in runs)
        write_json(root/BASE/'multi_seed_gate_v1.json',dict(status='PASS' if total*3<1200 else 'FAIL',first_seed=6107,single_block_lifecycle_seconds=total,
            projected_three_block_lifecycle_seconds=total*3,budget_seconds=1200,initialization_parity=True,equal_view_parity=True,scores_per_run=368*7,
            gate_scope='Feasibility/fairness only; not selected based on favorable scientific outcome',source_metadata_hashes={m['config']['experiment_id']:file_hash(root/BASE/'training'/m['config']['experiment_id']/'run_metadata.json') for m in runs}))
    print('Seed block complete',args.seed)
if __name__=='__main__':main()
