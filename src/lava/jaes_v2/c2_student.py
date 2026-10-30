"""Separate research stratum using the existing MobileNet-LSTM architecture builder."""
from __future__ import annotations

import os
import numpy as np


def initialize_tensorflow():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'; os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def build_student(*,seed: int,initialization: str='imagenet'):
    tf=initialize_tensorflow(); tf.keras.utils.set_random_seed(seed)
    from src.model import build_hybrid_model
    if initialization not in {'imagenet','scratch'}:
        raise ValueError('Explicit pretrained/scratch policy required; no conference-weight warm start')
    full,backbone=build_hybrid_model(weights='imagenet' if initialization=='imagenet' else None)
    if initialization=='scratch':
        backbone.trainable=True  # never freeze a randomly initialized feature extractor
    features=full.get_layer('time_distributed_mobilenetv3small').output
    extractor=tf.keras.Model(full.inputs,features,name='jaes_v2_mobilenet_feature_cache')
    inputs=tf.keras.Input(shape=(6,int(backbone.output_shape[-1])))
    x=inputs
    for name in ('temporal_lstm','classifier_dense','classifier_dropout','probability_fake'):
        x=full.get_layer(name)(x)
    head=tf.keras.Model(inputs,x,name='jaes_v2_mobilenet_student_head')
    return full,backbone,head,extractor


def objective(labels,clean_scores,changed_scores=None,*,baseline: str='B0',consistency_weight: float=.1):
    # Runtime configuration belongs outside a traced loss. Re-entering the
    # environment initializer here makes AutoGraph inspect process settings.
    import tensorflow as tf
    if baseline not in {'B0','B1','B2','B3','B4'} or not 0<=consistency_weight<=1:
        raise ValueError('Only reviewed standard baselines; no claimed novel loss')
    labels=tf.cast(tf.reshape(labels,(-1,1)),tf.float32)
    clean_scores=tf.reshape(clean_scores,(-1,1))
    bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)
    if baseline=='B4':
        if changed_scores is None:
            raise ValueError('Paired consistency needs both views')
        return .5*(bce(labels,clean_scores)+bce(labels,changed_scores))+consistency_weight*tf.reduce_mean(tf.square(clean_scores-changed_scores))
    return bce(labels,clean_scores)


def fake_probabilities(two_class_logits: np.ndarray,*,fake_index: int) -> np.ndarray:
    values=np.asarray(two_class_logits,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=2 or fake_index not in (0,1) or not np.isfinite(values).all():
        raise ValueError('Explicit two-class logits and documented FAKE index required')
    exp=np.exp(values-values.max(axis=1,keepdims=True))
    return (exp/exp.sum(axis=1,keepdims=True))[:,fake_index]


def peak_process_mib() -> float | None:
    """OS lifetime peak working set; separate from instantaneous sampled RSS."""
    if os.name!='nt':
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        except ImportError:
            return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
                    ('peak_working','working','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile')]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
    library=ctypes.WinDLL('psapi'); library.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    return counters.peak_working/1024**2 if library.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb) else None
