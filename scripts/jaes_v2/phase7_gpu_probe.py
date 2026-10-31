"""Portable isolated-runtime GPU sanity only; refuses CPU fallback and never installs dependencies."""
import argparse,json,os,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from src.lava.jaes_v2.evidence import write_json,file_hash

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import tensorflow as tf
    gpus=tf.config.list_physical_devices('GPU')
    if not gpus:
        write_json(args.output/'probe.json',dict(status='AWAITING_COMPATIBLE_GPU_RUNTIME',tensorflow=tf.__version__,platform=platform.platform(),gpu_used=False));raise RuntimeError('No TensorFlowGPU visible; refusing CPU fallback')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    tf.keras.utils.set_random_seed(6107)
    from src.model import build_hybrid_model,unfreeze_backbone_for_finetuning
    full,backbone=build_hybrid_model(weights='imagenet');unfreeze_backbone_for_finetuning(backbone);optimizer=tf.keras.optimizers.Adam(.0001)
    x=tf.zeros((1,6,224,224,3));y=tf.ones((1,1));start=perf_counter()
    with tf.GradientTape() as tape:loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,full(x,training=True)))
    gradients=tape.gradient(loss,full.trainable_weights);optimizer.apply_gradients(zip(gradients,full.trainable_weights));elapsed=perf_counter()-start
    p=full(x,training=False).numpy();full.save(args.output/'probe.keras');loaded=tf.keras.models.load_model(args.output/'probe.keras',compile=False);residual=float(np.max(np.abs(p-loaded(x,training=False).numpy())))
    write_json(args.output/'probe.json',dict(status='SMOKE_TESTED' if residual<1e-6 else 'FAIL',tensorflow=tf.__version__,build_info=tf.sysconfig.get_build_info(),devices=[str(g) for g in gpus],
        elapsed_seconds=elapsed,save_load_max_residual=residual,checkpoint_sha256=file_hash(args.output/'probe.keras'),trainable_parameters=int(sum(np.prod(v.shape) for v in full.trainable_weights)),
        gpu_used=True,scope='Synthetic portability sanity only, no detection-quality evidence; fullPhase7CLI currently validatedCPUonly'))
if __name__=='__main__':main()
