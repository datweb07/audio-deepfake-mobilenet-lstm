"""Sequential bounded verification. Includes no medium/foundation train command."""
from pathlib import Path
import subprocess,sys

def main():
    root=Path(__file__).resolve().parents[2];out=root/'outputs/jaes_v2/phase8/verification_logs';out.mkdir(exist_ok=True)
    commands=[
        ('resume_comparison',['scripts.jaes_v2.phase8_pilots','compare','--version','v3']),
        ('cpu_benchmark',['scripts.jaes_v2.phase8_benchmark']),
        ('guarded_negative',['scripts.jaes_v2.phase8_experiment','train','--config','configs/jaes_v2/phase8/p8_tiny_guarded_infeasible_head_s6107_v3.json']),
        ('medium_preflight',['scripts.jaes_v2.phase8_experiment','preflight','--config','configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json','--output','outputs/jaes_v2/phase8/preflight/medium_erm_partial_v1.json']),
        ('foundation_preflight',['scripts.jaes_v2.phase8_experiment','prepare-data','--config','configs/jaes_v2/phase8/p8_foundation_erm_partial_s6107_v1.json','--output','outputs/jaes_v2/phase8/preflight/foundation_erm_partial_v1.json']),
        ('representation_export',['scripts.jaes_v2.phase8_checks','--version','v3']),
        ('regression',['scripts.jaes_v2.phase8_experiment','validate','--output','outputs/jaes_v2/phase8/validation_final'])]
    for name,args in commands:
        path=out/(name+'.txt')
        with path.open('x',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-X','utf8','-m']+args,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        print(name+': '+str(result.returncode),flush=True)
        if result.returncode:raise RuntimeError('Verification failure: '+str(path))

if __name__=='__main__':main()
