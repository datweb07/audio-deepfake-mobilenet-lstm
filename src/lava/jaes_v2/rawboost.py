"""Seed-isolated adapter of Tak et al.'s MIT RawBoost reference, not online reproduction."""
from threading import Lock
import numpy as np
from .vendor.rawboost import RawBoost as reference

_RNG_LOCK=Lock()
PARAMETERS=dict(N_f=5,nBands=5,minF=20,maxF=8000,minBW=100,maxBW=1000,minCoeff=10,maxCoeff=100,
                minG=0,maxG=0,minBiasLinNonLin=5,maxBiasLinNonLin=20)


def rawboost(samples,*,sample_rate=22050,seed=5806,algorithm=5):
    values=np.asarray(samples,dtype=np.float64)
    if values.ndim!=1 or not values.size or not np.isfinite(values).all() or sample_rate<16000 or algorithm not in (0,1,2,3,4,5):
        raise ValueError('Finite mono waveform, fs>=16k and reviewed algorithm required')
    # Upstream normalization divides by max amplitude: define silence safely.
    if not np.any(values) or algorithm==0: return values.astype('float32').copy()
    with _RNG_LOCK:
        state=np.random.get_state(); np.random.seed(seed%(2**32))
        try:
            y=values.copy()
            if algorithm in (1,4,5): y=reference.LnL_convolutive_noise(y,**PARAMETERS,fs=sample_rate)
            if algorithm in (2,4,5): y=reference.ISD_additive_noise(y,10,2)
            if algorithm in (3,4):
                p={k:v for k,v in PARAMETERS.items() if k not in ('N_f','minBiasLinNonLin','maxBiasLinNonLin')}
                y=reference.SSI_additive_noise(y,10,40,**p,fs=sample_rate)
        finally: np.random.set_state(state)
    y=np.asarray(y,dtype='float32')
    if y.shape!=values.shape or not np.isfinite(y).all(): raise ValueError('Invalid reference RawBoost output')
    return y
