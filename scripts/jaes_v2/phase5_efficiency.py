"""Bounded measured offline end-to-end cost for the four selected pilot students."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from src.lava.jaes_v2.c2_student import initialize_tensorflow,peak_process_mib
from src.lava.jaes_v2.evidence import read_csv,file_hash,write_json
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.transforms import load_audio
from src.lava.jaes_v2.controls import prepare_headroom
from src.lava.jaes_v2.preprocessing import Preprocessor


def main():
    root=Path(__file__).resolve().parents[2]; output=root/'outputs/jaes_v2/audit/phase5_efficiency.json'
    if output.exists(): raise FileExistsError('Measured cost evidence is immutable')
    tf=initialize_tensorflow()
    config=json.loads((root/'configs/jaes_v2/c2_p0_erm_pilot_v2.json').read_text())
    records=read_csv(root/config['cache']/'manifest.csv')
    sample=next(r for r in records if r['split']=='development')
    results=[]
    for run in sorted((root/'outputs/jaes_v2/c2').glob('*_pilot_v2')):
        metadata=json.loads((run/'run_metadata.json').read_text()); verify_seal(run,metadata)
        configuration=json.loads((run/'resolved_config.json').read_text()); policy=Preprocessor(configuration['policy'])
        model=tf.keras.models.load_model(run/'best.keras',compile=False)
        audio=load_audio(root/sample['path'],sample_rate=22050,max_seconds=3); prepared,_=prepare_headroom(audio,.25)
        model(policy.extract(prepared.samples)[None],training=False).numpy()  # excluded warmup
        timing=[]
        for _ in range(5):
            start=perf_counter(); audio=load_audio(root/sample['path'],sample_rate=22050,max_seconds=3); prepared,_=prepare_headroom(audio,.25)
            decoded=perf_counter(); tensor=policy.extract(prepared.samples); processed=perf_counter()
            model(tensor[None],training=False).numpy(); finished=perf_counter()
            timing.append(dict(decode_headroom_seconds=decoded-start,preprocessing_seconds=processed-decoded,
                               model_seconds=finished-processed,end_to_end_seconds=finished-start))
        observed_seconds=len(audio.samples)/audio.sample_rate
        results.append(dict(run=run.name,policy=configuration['policy'],parameters=model.count_params(),model_bytes=(run/'best.keras').stat().st_size,
            checkpoint_sha256=file_hash(run/'best.keras'),sample_id=sample['sample_id'],source_checksum=sample['checksum'],
            observed_input_seconds=observed_seconds,median_seconds={key:float(np.median([r[key] for r in timing])) for key in timing[0]},
            offline_rtf=float(np.median([r['end_to_end_seconds']/observed_seconds for r in timing])),raw_timings=timing,
            os_process_lifetime_peak_mib=peak_process_mib()))
        tf.keras.backend.clear_session()
    write_json(output,dict(status='MEASURED_OFFLINE_CPU_COST',repetitions=5,warmup_excluded=1,results=results,
        scope='One declared development recording, warm in-process decode/preprocessing/full-model calls; sequential process lifetime peak is not per-model isolated memory',
        limitations='Small timing diagnostic; not real-time streaming certification or target-device energy; teacher cost unmeasured'))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
