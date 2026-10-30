"""New full-student offline CPU timing; no quality table or calibration is refitted."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from src.lava.jaes_v2.c2_student import initialize_tensorflow,peak_process_mib
from src.lava.jaes_v2.controls import prepare_headroom
from src.lava.jaes_v2.transforms import load_audio
from src.lava.jaes_v2.preprocessing import Preprocessor
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.evidence import read_csv,write_json,file_hash
from src.lava.jaes_v2.phase6_data import BASE


def main():
    root=Path(__file__).resolve().parents[2]
    from scripts.jaes_v2.phase6_experiment import resource_lock
    with resource_lock(root): measure(root)


def measure(root):
    output=root/BASE/'efficiency_v1.json'
    if output.exists(): raise FileExistsError(output)
    tf=initialize_tensorflow(); records=read_csv(root/BASE/'data_selection_v1/manifest.csv'); record=next(r for r in records if r['split']=='development')
    results=[]
    for name in ('p6_p0_b2_v1','p6_p0_b4_control_v1','p6_p0_b4_v1','p6_p0_rb_v1'):
        run=root/BASE/'training'/name; metadata=json.loads((run/'run_metadata.json').read_text()); verify_seal(run,metadata)
        model=tf.keras.models.load_model(run/'best.keras',compile=False); policy=Preprocessor(metadata['config']['policy'])
        timings=[]
        for i in range(6):
            start=perf_counter(); audio=load_audio(root/record['path'],sample_rate=22050,max_seconds=3); audio,_=prepare_headroom(audio,.25); decoded=perf_counter()
            features=policy.extract(audio.samples); prepared=perf_counter(); model(features[None],training=False).numpy(); done=perf_counter()
            if i: timings.append(dict(decode_seconds=decoded-start,preprocessing_seconds=prepared-decoded,model_seconds=done-prepared,end_to_end_seconds=done-start))
        duration=len(audio.samples)/audio.sample_rate
        results.append(dict(run=name,sample_id=record['sample_id'],source_sha256=record['checksum'],observed_seconds=duration,checkpoint_sha256=file_hash(run/'best.keras'),parameters=model.count_params(),
            checkpoint_bytes=(run/'best.keras').stat().st_size,median_seconds={k:float(np.median([r[k] for r in timings])) for k in timings[0]},
            offline_rtf=float(np.median([r['end_to_end_seconds']/duration for r in timings])),process_lifetime_peak_mib=peak_process_mib(),raw_timings=timings))
        tf.keras.backend.clear_session()
    write_json(output,dict(status='MEASURED_OFFLINE_CPU',repetitions=5,warmup_excluded=1,results=results,
        limitations='One declared dev recording; sequential process lifetime peaks, not isolated per-model RAM; no streaming/energy or timing-significance claim'))
    print([(r['run'],r['median_seconds']['end_to_end_seconds'],r['offline_rtf']) for r in results])


if __name__=='__main__': main()
