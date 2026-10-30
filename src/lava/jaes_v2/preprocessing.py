"""Versioned research preprocessors; production implementation is never replaced."""
from __future__ import annotations

from dataclasses import dataclass,asdict
import cv2
import numpy as np
from scipy.signal import stft

import config
from src.preprocessing import process_audio_data,segment_audio,_mel_filter_bank,normalize_audio_length
from .evidence import object_hash


@dataclass(frozen=True)
class Preprocessor:
    policy: str='P0'
    version: str='research-mel-floor-1'
    relative_floor: float=1e-8
    rms_target: float=.1

    def __post_init__(self):
        if self.policy not in {'P0','P1','P2'} or self.version!='research-mel-floor-1' or not 0<self.relative_floor<1 or not 0<self.rms_target<=1:
            raise ValueError('Invalid research preprocessor contract')

    @property
    def fingerprint(self) -> str:
        return object_hash(asdict(self))

    def extract(self,samples: np.ndarray) -> np.ndarray:
        values=np.asarray(samples,dtype=np.float32)
        if values.ndim!=1 or not values.size or not np.isfinite(values).all():
            raise ValueError('Finite nonempty mono samples required')
        if self.policy=='P0':
            return process_audio_data(values)
        values=normalize_audio_length(values)
        if self.policy=='P2':
            rms=float(np.sqrt(np.mean(values.astype(np.float64)**2)))
            # Conventional observed-window RMS normalization, not an oracle gain.
            return process_audio_data((values*(self.rms_target/rms)).astype('float32') if rms else values)
        images=[]
        for segment in segment_audio(values):
            _,_,spectrum=stft(segment,fs=config.SAMPLE_RATE,window='hann',nperseg=config.N_FFT,
                              noverlap=config.N_FFT-config.HOP_LENGTH,nfft=config.N_FFT,boundary='zeros',padded=True)
            power=_mel_filter_bank().astype(np.float64)@(np.abs(spectrum.astype(np.complex128))**2)
            reference=float(power.max())
            if reference==0:
                db=np.full(power.shape,-config.TOP_DB,dtype=np.float64)
            else:
                db=np.maximum(10*np.log10(np.maximum(power/reference,self.relative_floor)),-config.TOP_DB)
            pixels=np.clip((db+config.TOP_DB)*(255/config.TOP_DB),config.INPUT_VALUE_MIN,config.INPUT_VALUE_MAX)
            image=cv2.resize(pixels,config.IMAGE_SIZE,interpolation=cv2.INTER_LINEAR).astype('float32')
            images.append(np.repeat(image[...,None],config.CHANNELS,axis=-1))
        return np.stack(images)
