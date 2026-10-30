"""One public command per bounded C2 stage, with explicit data/resource gates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.lava.jaes_v2.c2_student import initialize_tensorflow,build_student,objective,peak_process_mib
from src.lava.jaes_v2.c2_training import TrainingConfig,train,aggregate
from src.lava.jaes_v2.c2_data import build_cache
from src.lava.jaes_v2.evidence import write_json,file_hash


def resource_preflight(root: Path,output: Path) -> dict:
    if output.exists():
        raise FileExistsError('New resource report required')
    tf=initialize_tensorflow(); full,backbone,head,extractor=build_student(seed=5804)
    from benchmark.efficiency import _rss_mib
    x=tf.convert_to_tensor(np.random.default_rng(5804).uniform(0,255,(2,6,224,224,3)).astype('float32')); y=tf.constant([[0.],[1.]])
    optimizer=tf.keras.optimizers.Adam(.001)
    start=perf_counter()
    with tf.GradientTape() as tape:
        prediction=full(x,training=True); value=objective(y,prediction)
    forward=perf_counter()-start; start=perf_counter()
    gradients=tape.gradient(value,full.trainable_weights)
    norm=float(tf.linalg.global_norm(gradients)); optimizer.apply_gradients(zip(gradients,full.trainable_weights)); backward=perf_counter()-start
    warm=[]
    for _ in range(5):
        start=perf_counter(); extractor(x,training=False).numpy(); warm.append(perf_counter()-start)
    from src.lava.jaes_v2.evidence import read_csv
    from src.preprocessing import load_audio
    from src.lava.jaes_v2.preprocessing import Preprocessor
    record=read_csv(root/'outputs/jaes_v2/datasets/asv2019_foundation_v1/train_manifest.csv')[0]
    start=perf_counter(); waveform=load_audio(record['path']); Preprocessor('P0').extract(waveform); Preprocessor('P1').extract(waveform)
    preprocessing_seconds=perf_counter()-start
    params=full.count_params(); temp=output.with_suffix('.keras'); full.save(temp)
    report={'status':'PASS' if norm>0 and np.isfinite(norm) and forward+backward<15 else 'AWAITING_COMPUTE',
            'forward_seconds_batch2':forward,'backward_optimizer_seconds_batch2':backward,'gradient_norm':norm,
            'batch_input_bytes':int(np.prod(x.shape))*4,'process_rss_after_batch_mib':_rss_mib(),'os_peak_working_set_mib':peak_process_mib(),
            'gpu_available':bool(tf.config.list_physical_devices('GPU')),'gpu_allocated_memory':None,
            'parameters':params,'checkpoint_bytes':temp.stat().st_size,'checkpoint_sha256':file_hash(temp),
            'warm_encoder_seconds_batch2':float(np.mean(warm[1:])),'warmup_encoder_seconds':warm[0],
            'actual_audio_decode_two_preprocessors_seconds':preprocessing_seconds,
            'cache_projection_upper_seconds':(float(np.mean(warm[1:]))/2)*1792+preprocessing_seconds*896+100,'budget_cache_seconds':600,
            'backbone':'frozen pretrained ImageNet; scratch freezing disallowed','training_budget_seconds_per_run':180,
            'batch_scope':'Resource synthetic fixture, not a scientific result'}
    if report['cache_projection_upper_seconds']>600:
        report['status']='AWAITING_COMPUTE'
    write_json(output,report); return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','cache','train','aggregate'))
    parser.add_argument('--config',type=Path); parser.add_argument('--run',type=Path)
    parser.add_argument('--cache-output',type=Path,default=Path('outputs/jaes_v2/c2_feature_cache_v1'))
    parser.add_argument('--report',type=Path,default=Path('outputs/jaes_v2/audit/phase5_training_preflight_v2.json'))
    args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    if args.mode=='preflight': print(json.dumps(resource_preflight(root,root/args.report),indent=2))
    elif args.mode=='cache':
        report=json.loads((root/'outputs/jaes_v2/audit/phase5_training_preflight_v2.json').read_text())
        if report['status']!='PASS': raise RuntimeError('AWAITING_COMPUTE')
        print(build_cache(root,root/args.cache_output)['status'])
    elif args.mode=='train':
        if not args.config: parser.error('--config required')
        print(train(root,TrainingConfig(**json.loads(args.config.read_text()))))
    else:
        if not args.run: parser.error('--run required')
        print(json.dumps(aggregate(args.run),indent=2))


if __name__=='__main__':
    main()
