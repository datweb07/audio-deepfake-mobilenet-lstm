"""Measure exact production Mel tensors and absolute spectral-floor exposure, model-free."""
import argparse
from pathlib import Path
import json

import numpy as np
import cv2
from scipy.signal import stft

import config
from src.preprocessing import load_audio,process_audio_data,segment_audio,_mel_filter_bank
from src.lava.jaes_v2.evidence import file_hash,read_csv,write_csv,write_json
from src.lava.jaes_v2.phase3 import verify_seal


def floor_fraction(samples: np.ndarray) -> float:
    counts=[]
    for segment in segment_audio(samples):
        _,_,spectrum=stft(segment,fs=config.SAMPLE_RATE,window='hann',nperseg=config.N_FFT,
                          noverlap=config.N_FFT-config.HOP_LENGTH,nfft=config.N_FFT,boundary='zeros',padded=True)
        power=_mel_filter_bank()@(np.abs(spectrum)**2)
        counts.append(float(np.mean(power<1e-10)))
    return float(np.mean(counts))


def diagnostic_features(samples: np.ndarray, *, mel_power_floor: float) -> np.ndarray:
    """Model-free counterfactual only; never installed as detector preprocessing."""
    if mel_power_floor<=0:
        raise ValueError('Positive diagnostic floor required')
    tensors=[]
    for segment in segment_audio(samples):
        _,_,spectrum=stft(segment,fs=config.SAMPLE_RATE,window='hann',nperseg=config.N_FFT,
                          noverlap=config.N_FFT-config.HOP_LENGTH,nfft=config.N_FFT,boundary='zeros',padded=True)
        power=_mel_filter_bank()@(np.abs(spectrum)**2)
        db=10*np.log10(np.maximum(power,mel_power_floor)); db-=float(np.max(db))
        db=np.maximum(db,-config.TOP_DB)
        image=np.clip((db+config.TOP_DB)*(255/config.TOP_DB),config.INPUT_VALUE_MIN,config.INPUT_VALUE_MAX)
        resized=cv2.resize(image,config.IMAGE_SIZE,interpolation=cv2.INTER_LINEAR)
        tensors.append(np.repeat(resized[...,None],config.CHANNELS,axis=-1))
    return np.stack(tensors).astype('float32')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New diagnostic directory required')
    base=Path('outputs/jaes_v2/c1_mlaad_tiny_dev_v1_waveforms')
    new=Path('outputs/jaes_v2/c1_nuisance_controls_dev_v1_waveforms')
    for directory in (base,new):
        verify_seal(directory,json.loads((directory/'cache_metadata.json').read_text()))
    originals={(r['sample_id'],r['condition_id']):r for r in read_csv(base/'conditions.csv')}
    rows=[]
    for row in read_csv(new/'conditions.csv'):
        if row['policy']!='global_half_v1' or row['condition_id']!='clean':
            continue
        old_path=base/originals[(row['sample_id'],'clean')]['path']; new_path=new/row['path']
        old_wave,new_wave=load_audio(old_path),load_audio(new_path)
        if not np.array_equal(new_wave,old_wave*.5):
            raise ValueError('Global attenuation is not exactly half of observed production waveform')
        old_features,new_features=process_audio_data(old_wave),process_audio_data(new_wave)
        delta=np.abs(new_features-old_features)
        replica=diagnostic_features(old_wave,mel_power_floor=1e-10)
        if not np.array_equal(replica,old_features):
            raise ValueError('Diagnostic reference differs from production Mel implementation')
        counterfactual=diagnostic_features(new_wave,mel_power_floor=1e-10*.5**2)
        residual=np.abs(counterfactual-old_features)
        rows.append({'sample_id':row['sample_id'],'old_wave_sha256':file_hash(old_path),'new_wave_sha256':file_hash(new_path),
                     'tensor_mean_abs_difference':float(delta.mean()),'tensor_max_abs_difference':float(delta.max()),
                     'tensor_fraction_difference_gt_1e_3':float(np.mean(delta>1e-3)),
                     'baseline_mel_cells_below_absolute_floor':floor_fraction(old_wave),
                     'half_mel_cells_below_absolute_floor':floor_fraction(new_wave),
                     'diagnostic_scaled_floor_tensor_max_difference':float(residual.max()),
                     'diagnostic_scaled_floor_tensor_mean_difference':float(residual.mean())})
    args.output.mkdir(parents=True)
    write_csv(args.output/'feature_differences.csv',rows)
    report={'scope':'MODEL_FREE_EXACT_PRODUCTION_INPUT_DIAGNOSTIC','sources':len(rows),
            'production_preprocessor_sha256':file_hash(Path('src/preprocessing.py')),
            'global_half_exact_waveform_parity':True,'maximum_tensor_difference':max(r['tensor_max_abs_difference'] for r in rows),
            'mean_tensor_absolute_difference':float(np.mean([r['tensor_mean_abs_difference'] for r in rows])),
            'absolute_floor':1e-10,'per_segment_reference':'subtract per-segment maximum dB after fixed absolute Mel-power floor',
            'counterfactual_floor':2.5e-11,'counterfactual_max_tensor_residual':max(r['diagnostic_scaled_floor_tensor_max_difference'] for r in rows),
            'counterfactual_mean_tensor_residual':float(np.mean([r['diagnostic_scaled_floor_tensor_mean_difference'] for r in rows])),
            'counterfactual_predictor_calls':0,'production_preprocessing_modified':False,
            'interpretation':'Uniform waveform attenuation is not guaranteed feature-invariant because of absolute floor and floating arithmetic. Exposure/feature differences do not isolate their causal share of detector decisions.'}
    write_json(args.output/'diagnostic.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
