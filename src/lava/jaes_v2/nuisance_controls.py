"""Explicit signal interventions on the already observed detector window, not source repair."""
from __future__ import annotations

import numpy as np

from .transforms import Audio

POLICIES=('global_half_v1','rms_anchor_v1','edge_repack_v1')


def signal_features(audio: Audio, *, activity_threshold: float=1e-4) -> dict:
    x=audio.samples.astype(np.float64)
    active=np.flatnonzero(np.abs(x)>activity_threshold)
    frame=max(1,round(.02*audio.sample_rate))
    frame_rms=[np.sqrt(np.mean(x[i:i+frame]**2)) for i in range(0,len(x),frame)]
    power=np.abs(np.fft.rfft(x))**2
    frequencies=np.fft.rfftfreq(len(x),1/audio.sample_rate)
    total=float(power.sum())
    result={'duration':len(x)/audio.sample_rate,'sample_rate':audio.sample_rate,'rms':float(np.sqrt(np.mean(x*x))),
            'peak':float(np.max(np.abs(x))),'silence_fraction_below_1e-4':float(np.mean(np.abs(x)<=activity_threshold)),
            'leading_below_threshold_seconds':float(active[0]/audio.sample_rate) if active.size else len(x)/audio.sample_rate,
            'trailing_below_threshold_seconds':float((len(x)-1-active[-1])/audio.sample_rate) if active.size else len(x)/audio.sample_rate,
            'above_threshold_sample_seconds':float(active.size/audio.sample_rate),
            'frame_activity_proxy_seconds':float(sum(value>1e-3 for value in frame_rms)*frame/audio.sample_rate),
            'frame_activity_definition':'20ms RMS>1e-3; signal proxy, not VAD/speech ground truth',
            'rolloff95_hz':float(frequencies[min(len(frequencies)-1,np.searchsorted(np.cumsum(power),.95*total))]) if total else None,
            'outside_unit_fraction':float(np.mean(np.abs(x)>1))}
    for low,high,name in ((0,300,'0_300'),(300,3400,'300_3400'),(3400,8000,'3400_8000'),(8000,float('inf'),'above_8000')):
        result[f'energy_fraction_{name}hz']=float(power[(frequencies>=low)&(frequencies<high)].sum()/total) if total else None
    return result


def intervene(audio: Audio, policy: str) -> tuple[Audio,dict]:
    if policy not in POLICIES:
        raise ValueError('Unsupported versioned nuisance policy')
    x=audio.samples.copy()
    evidence={'policy':policy,'input_samples':len(x),'post_channel_normalization':False}
    if policy=='global_half_v1':
        x*=.5
        evidence.update(gain=.5,information_risk='Changes amplitude/floor interaction; negative-control sensitivity, not new forensic method')
    elif policy=='rms_anchor_v1':
        rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
        peak=float(np.max(np.abs(x)))
        if not rms:
            raise ValueError('Cannot RMS-anchor a silent input')
        gain=min(.03/rms,.25/peak)
        x*=gain
        evidence.update(gain=gain,target_rms=.03,peak_cap=.25,achieved_rms=float(np.sqrt(np.mean(x.astype(np.float64)**2))),
                        information_risk='RMS is not perceptual loudness; cap may prevent equal RMS; amplitude cues may be forensic evidence')
    else:
        active=np.flatnonzero(np.abs(x)>1e-4)
        if not active.size:
            raise ValueError('No above-threshold content; edge intervention undefined')
        start,end=int(active[0]),int(active[-1])+1
        kept=x[start:end].copy()
        x=np.pad(kept,(0,len(x)-len(kept)))
        evidence.update(start_sample=start,end_sample=end,retained_samples=len(kept),
                        retained_content_exact=True,activity_threshold=1e-4,
                        information_risk='Low-energy tails removed and content shifted across Mel segment boundaries; semantic preservation not independently verified')
    return Audio(x,audio.sample_rate),evidence
