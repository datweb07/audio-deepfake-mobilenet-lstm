"""Actual matched full-student inference timing for F0/F1, outside cached-score timing."""
import json,os,tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
from src.lava.jaes_v2.evidence import read_csv,file_hash,write_json
from src.lava.jaes_v2.phase7_data import BASE

def main():
    root=Path(__file__).resolve().parents[2];temporary=root/'.research_tmp';temporary.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temporary);os.environ['TMP']=str(temporary);tempfile.tempdir=str(temporary)
    from scripts.jaes_v2.phase6_experiment import resource_lock
    with resource_lock(root):execute(root)

def execute(root):
    from src.lava.jaes_v2.c2_student import initialize_tensorflow,peak_process_mib
    from src.lava.jaes_v2.transforms import load_audio
    from src.lava.jaes_v2.controls import prepare_headroom
    from src.lava.jaes_v2.preprocessing import Preprocessor
    from src.lava.jaes_v2.runtime import source_fingerprint
    output=root/BASE/'representation_efficiency_v1.json'
    if output.exists():raise FileExistsError(output)
    tf=initialize_tensorflow();records=read_csv(root/BASE/'selections/medium_all_attack_v1/manifest.csv')
    r=next(r for r in records if r['split']=='development' and r['label']=='0')
    if file_hash(root/r['path'])!=r['checksum']:raise ValueError('Timing source changed')
    results=[];started=perf_counter()
    for identifier in ('p7_f0_s6107_v1','p7_f1_s6107_v1'):
        path=root/BASE/'training'/identifier/'best.keras';model=tf.keras.models.load_model(path,compile=False);times=[]
        for i in range(6):
            t=perf_counter();audio=load_audio(root/r['path'],sample_rate=22050,max_seconds=3);audio,_=prepare_headroom(audio,.25);d=perf_counter();x=Preprocessor('P0').extract(audio.samples)[None];p=perf_counter();score=float(model(x,training=False).numpy()[0,0]);end=perf_counter()
            if i:times.append(dict(decode=d-t,preprocessing=p-d,model=end-p,end_to_end=end-t,score=score))
        observed=min(float(r['duration']),3);median={k:float(np.median([v[k] for v in times])) for k in ('decode','preprocessing','model','end_to_end')}
        results.append(dict(experiment_id=identifier,sample_id=r['sample_id'],source_checksum=r['checksum'],observed_seconds=observed,parameters=model.count_params(),checkpoint_sha256=file_hash(path),median_seconds=median,offline_rtf=median['end_to_end']/observed,raw_timings=times,process_lifetime_peak_mib=peak_process_mib(),gpu_used=False))
        if perf_counter()-started>120:raise TimeoutError('Representation inference timing budget120seconds')
    write_json(output,dict(status='MEASURED_OFFLINE_CPU',repetitions=5,warmup_excluded=1,runs=results,code=source_fingerprint(root),limitations='One source and warm sequential process; lifetime peaks not isolated; no streaming/energy/timing-superiority claim'))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
