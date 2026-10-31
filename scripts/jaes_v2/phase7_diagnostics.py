"""Bounded proxy controls, online augmentation cost and full-student timing."""
import argparse,io,json,os,tempfile,shutil
from pathlib import Path
from time import perf_counter
import numpy as np
import soundfile as sf
from src.lava.jaes_v2.evidence import read_csv,write_csv,write_json,file_hash
from src.lava.jaes_v2.phase7_data import BASE,load_cache,waveform_views
from src.lava.jaes_v2.phase7_training import online_rawboost

def cosine(a,b):
    a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1); denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip((1+np.dot(a,b)/denominator)/2,0,1)) if denominator else None

def correlation(x,y):
    from scipy.stats import spearmanr
    if np.std(x)<1e-10 or np.std(y)<1e-10: return None
    value=float(spearmanr(x,y).statistic); return value if np.isfinite(value) else None

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]; temp=root/'.research_tmp';temp.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp);tempfile.tempdir=str(temp)
    from scripts.jaes_v2.phase6_experiment import resource_lock
    with resource_lock(root): execute(root,args.output)

def execute(root,out):
    from src.lava.jaes_v2.c2_student import initialize_tensorflow,build_student,peak_process_mib
    from src.lava.jaes_v2.preprocessing import Preprocessor
    from src.lava.jaes_v2.transforms import Audio,Transform,load_audio
    from src.lava.jaes_v2.controls import NoisePlan,prepare_headroom
    config=json.loads((root/'configs/jaes_v2/phase7/cache_medium_v1.json').read_text()); cache,meta,records=load_cache(root,config)
    out.mkdir(parents=True,exist_ok=False); tf=initialize_tensorflow(); full,backbone,head,extractor=build_student(seed=6107)
    table=read_csv(cache/'view_manifest.csv'); z=np.load(cache/'embeddings.npz'); arrays={k:z[k] for k in z.files};z.close(); idx={(r['sample_id'],r['condition']):r for r in table}
    def vector(identifier,condition):
        r=idx[(identifier,condition)];return arrays[r['array_key']][int(r['array_index'])]
    run=root/BASE/'training/p7_h0_s6107_v1'; scores=read_csv(run/'scores.csv'); predictions={(r['sample_id'],r['condition_id']):r for r in scores}
    rows=[]
    for row in table:
        identifier=row['sample_id']; condition=row['condition']
        if row['split']!='development' or condition=='clean' or predictions[(identifier,condition)]['stratum']!='A_PAIRED':continue
        p=predictions[(identifier,condition)];clean=predictions[(identifier,'clean')];label=int(row['label'])
        rows.append(dict(sample_id=identifier,speaker_id=row['speaker_id'],attack_id=row['attack_id'],label=label,condition=condition,
            similarity=cosine(vector(identifier,'clean'),vector(identifier,condition)),error_change=int(int(p['predicted_label'])!=label)-int(int(clean['predicted_label'])!=label),
            rms=float(row['rms']),silence_fraction=float(row['silence_fraction']),severity=condition))
    associations=[]; rng=np.random.default_rng(7107)
    for label in (0,1):
        group=[r for r in rows if r['label']==label]; x=np.array([r['similarity'] for r in group]);y=np.array([r['error_change'] for r in group],dtype=float)
        # Categorical nuisance adjustment is descriptive residualization, not causality.
        columns=[np.ones(len(group)),np.log(np.maximum([r['rms'] for r in group],1e-12)),np.array([r['silence_fraction'] for r in group])]
        for key in ('speaker_id','attack_id','condition'):
            for value in sorted({r[key] for r in group})[1:]:columns.append(np.array([float(r[key]==value) for r in group]))
        design=np.column_stack(columns); rx=x-design@np.linalg.lstsq(design,x,rcond=None)[0];ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        strata={}
        for i,r in enumerate(group):strata.setdefault((r['speaker_id'],r['attack_id'],r['condition']),[]).append(i)
        shuffled=[]
        for _ in range(100):
            sx=x.copy()
            for indices in strata.values():sx[indices]=rng.permutation(x[indices])
            shuffled.append(correlation(sx,y))
        associations.append(dict(label=label,view_rows=len(group),speakers=len({r['speaker_id'] for r in group}),raw_spearman=correlation(x,y),adjusted_residual_spearman=correlation(rx,ry),
            shuffled_associations=shuffled,shuffle_strata='speaker x attack x condition within class',
            adjustment='speaker,attack,condition severity,logRMS,silence fraction',causal_interpretation='NONE',trained_weight_ablation='NOT_EXECUTED; no weighted objective justified'))
    write_csv(out/'proxy_development_rows.csv',rows)
    # Positive serialization control and challenging temporal/noise/codec controls.
    chosen=[next(r for r in records if r.split=='development' and r.label==0)]
    for a in ('A01','A02','A03','A04','A05','A06'):chosen.append(next(r for r in records if r.split=='development' and r.attack_id==a))
    controls=[]; online_times=[]; online_checks=[]; started=perf_counter()
    for r in chosen:
        audio=load_audio(root/r.path,sample_rate=22050,max_seconds=3);clean,_=prepare_headroom(audio,.25)
        buffer=io.BytesIO();sf.write(buffer,clean.samples,22050,format='WAV',subtype='FLOAT');buffer.seek(0);restored,sr=sf.read(buffer,dtype='float32')
        if not np.array_equal(restored,clean.samples):raise ValueError('Float serialization parity failed')
        plan=NoisePlan.build(clean,target_snr_db=0,seed=7107);noisy,_=plan.insert(clean,'fixed_absolute')
        codec=Transform('opus8','codec',dict(encoder='libopus',bitrate='8k',extension='.ogg'));coded,_=codec.apply(clean,seed=0,ffmpeg=shutil.which('ffmpeg'))
        variants={'identity':clean.samples,'float_lossless':restored,'small_gain_0_98':clean.samples*.98,'gain_half':clean.samples*.5,'noise0':noisy.samples,'opus8':coded.samples,'boundary_roll5ms':np.roll(clean.samples,110)}
        tensors=[Preprocessor('P0').extract(w) for w in variants.values()]; vectors=[]
        for j in range(0,len(tensors),2):vectors.extend(extractor(np.stack(tensors[j:j+2]),training=False).numpy())
        for condition,vector in zip(variants,vectors):controls.append(dict(sample_id=r.sample_id,label=r.label,attack_id=r.attack_id,condition=condition,embedding_similarity=cosine(vectors[0],vector),
            target_evidence='Identity/float equality known; gain preserves ideal relative structure but P0 numerical invariance not assumed; forensic evidence under challenging transforms UNKNOWN'))
        t=perf_counter();wave,seed=online_rawboost(clean.samples,training_seed=306107,epoch=0,sample_id=r.sample_id);prep=Preprocessor('P0').extract(wave);extractor(prep[None],training=False).numpy();online_times.append(perf_counter()-t)
        online_checks.append(dict(sample_id=r.sample_id,seed=seed,peak=float(np.max(np.abs(wave))),label_unchanged=r.label,rate=22050))
        if perf_counter()-started>120:raise TimeoutError('Bounded diagnostics budget120seconds')
    write_csv(out/'proxy_controls.csv',controls)
    write_json(out/'proxy_decision.json',dict(status='NON_SPECIFIC',formula='r=(1+cosine(flattened frozen ImageNet features))/2',associations=associations,
        controls=controls,positive_control='Identity and FLOAT serialization preserve exact waveform and embeddings',
        rejection='No ground-truth forensic cue retention. ImageNet similarity and error association do not establish mechanism, including after nuisance adjustment. Challenging severity is not a cue-removal oracle.',
        weighted_loss_implemented=False,new_method_gate='FAIL',neural_inference='Only seven-source control embeddings; association aggregation model-free'))
    mean=float(np.mean(online_times)); projected=640*6*(2/3)*mean
    write_json(out/'online_rawboost_preflight.json',dict(status='AWAITING_COMPUTE',implemented=True,tested=True,quality_experiment='NOT_EXECUTED',
        algorithm=5,convolution_parameters=__import__('src.lava.jaes_v2.rawboost',fromlist=['PARAMETERS']).PARAMETERS,ISD_parameters=dict(P=10,g_sd=2),sampling_rate=22050,max_filter_hz=8000,resampling='NONE; original16k setting adapted directly to22050Hz frequency-calibrated filters',
        amplitude='Reference internal processing; reject peak>1.00001; no hidden post-channel normalization',
        per_source_extract_seconds=online_times,mean_seconds=mean,projected_six_epoch_extra_extraction_seconds=projected,
        head_run_budget_seconds=240,decision='Additional extraction alone exceeds locked head-only budget; do not substitute finite-view results',checks=online_checks))
    # Actual full-student inference, same selected P0 source and all three chosen models.
    timing=[]; record=chosen[0]
    for identifier in ('p7_h0_s6107_v1','p7_h3_s6107_v1','p7_h4_s6107_v1'):
        path=root/BASE/'training'/identifier/'best.keras'; model=tf.keras.models.load_model(path,compile=False);times=[]
        for iteration in range(6):
            t=perf_counter();a=load_audio(root/record.path,sample_rate=22050,max_seconds=3);a,_=prepare_headroom(a,.25);decoded=perf_counter();tensor=Preprocessor('P0').extract(a.samples)[None];processed=perf_counter();model(tensor,training=False).numpy();done=perf_counter()
            if iteration:times.append(dict(decode=decoded-t,preprocessing=processed-decoded,model=done-processed,end_to_end=done-t))
        timing.append(dict(experiment_id=identifier,sample_id=record.sample_id,checkpoint_sha256=file_hash(path),parameters=model.count_params(),checkpoint_bytes=path.stat().st_size,
            observed_audio_seconds=min(record.duration,3),raw_timings=times,median_seconds={k:float(np.median([v[k] for v in times])) for k in times[0]},
            offline_rtf=float(np.median([v['end_to_end'] for v in times]))/min(record.duration,3),peak_process_lifetime_mib=peak_process_mib(),gpu_used=False))
    write_json(out/'efficiency.json',dict(status='MEASURED_OFFLINE_CPU',warmup_excluded=1,repetitions=5,runs=timing,limitation='One source; sequential process-lifetime peaks; no streaming,energy or timing-superiority claim'))
    write_json(out/'seal.json',dict(output_hashes={p.name:file_hash(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(proxy='NON_SPECIFIC',online_extra_seconds=projected,inference_rtf=[r['offline_rtf'] for r in timing]),indent=2))
if __name__=='__main__':main()
