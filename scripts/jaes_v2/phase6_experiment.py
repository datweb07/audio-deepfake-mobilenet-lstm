"""One command per Phase6 bounded stage; no confirmation or reserved-data access."""
import argparse
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import os
from contextlib import contextmanager
from src.lava.jaes_v2.phase6_data import BASE,select,build_cache
from src.lava.jaes_v2.phase6_training import Config,train,finetune_sanity
from src.lava.jaes_v2.evidence import read_csv,write_json,file_hash


@contextmanager
def resource_lock(root):
    """Reject overlapping Phase6 jobs; this mutex does not reserve system RAM."""
    directory=root/BASE; directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.tensorflow_job.lock').open('a+b') as stream:
        if stream.tell()==0: stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as error: raise RuntimeError('Another Phase6 heavy job is active; use sequential execution') from error
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def preflight(root):
    from src.lava.jaes_v2.c2_student import build_student,initialize_tensorflow,peak_process_mib
    from src.lava.jaes_v2.transforms import load_audio
    from src.lava.jaes_v2.controls import prepare_headroom
    from src.lava.jaes_v2.preprocessing import Preprocessor
    from src.lava.jaes_v2.rawboost import rawboost
    from src.lava.jaes_v2.phase6_training import paired_objective
    output=root/BASE/'resource_preflight_v1.json'
    if output.exists(): raise FileExistsError(output)
    records=read_csv(root/BASE/'data_selection_v1/manifest.csv'); tf=initialize_tensorflow(); full,backbone,head,extractor=build_student(seed=5806)
    record=next(r for r in records if r['split']=='train'); start=perf_counter()
    audio=load_audio(root/record['path'],sample_rate=22050,max_seconds=3); audio,_=prepare_headroom(audio,.25)
    tensors=np.stack([Preprocessor(p).extract(audio.samples) for p in ('P0','P1')]); prep=perf_counter()-start
    warm=[]
    for _ in range(4):
        start=perf_counter(); extractor(tensors,training=False).numpy(); warm.append(perf_counter()-start)
    start=perf_counter(); rawboost(audio.samples,seed=5806); rb=perf_counter()-start
    x=tf.constant(np.zeros((16,6,576),dtype='float32')); y=tf.constant(np.tile([0.,1.],8)); start=perf_counter()
    with tf.GradientTape() as tape: loss=paired_objective(y,head(x,training=True))
    forward=perf_counter()-start; start=perf_counter(); gradients=tape.gradient(loss,head.trainable_weights); tf.keras.optimizers.Adam(.001).apply_gradients(zip(gradients,head.trainable_weights)); backward=perf_counter()-start
    ntrain=sum(r['split']=='train' for r in records); ndev=len(records)-ntrain; cells=ntrain*6+ndev*7
    estimate=(float(np.mean(warm[1:]))+prep)*cells+rb*ntrain*2+100
    report=dict(status='PASS' if estimate<1100 and forward+backward<15 else 'AWAITING_COMPUTE',
        measured_encoder_batch2_seconds=float(np.mean(warm[1:])),decode_two_preprocessors_seconds=prep,rawboost_seconds=rb,
        head_forward_batch16_seconds=forward,head_backward_batch16_seconds=backward,peak_ram_mib=peak_process_mib(),gpu_available=bool(tf.config.list_physical_devices('GPU')),
        projected_cache_seconds=estimate,cache_budget_seconds=1100,embeddings=cells*2,train_sources=ntrain,dev_sources=ndev,
        head_data_loading='Materialize each NPZ array once; Phase5 per-sample decompression eliminated in new path')
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=('select','preflight','cache','train','finetune'))
    parser.add_argument('--config',type=Path); parser.add_argument('--device',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--output-name',default='partial_finetune_sanity_v1'); args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    with resource_lock(root): execute(root,args,parser)


def execute(root,args,parser):
    if args.mode=='select': print(select(root))
    elif args.mode=='preflight': print(json.dumps(preflight(root),indent=2))
    elif args.mode=='cache':
        gate=json.loads((root/BASE/'resource_preflight_v1.json').read_text())
        if gate['status']!='PASS': raise ValueError('Resource gate failed')
        print(build_cache(root))
    elif args.mode=='finetune': print(json.dumps(finetune_sanity(root,device=args.device,output_name=args.output_name),indent=2))
    else:
        if not args.config: parser.error('--config required')
        print(train(root,Config(**json.loads(args.config.read_text()))))


if __name__=='__main__': main()
