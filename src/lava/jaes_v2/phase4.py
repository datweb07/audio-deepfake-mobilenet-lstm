"""Bounded paired nuisance pilot after the source-matching gate; old scores are reused."""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import re
from time import perf_counter

from .evidence import file_hash,object_hash,read_csv,write_csv,write_json
from .nuisance_controls import POLICIES,intervene,signal_features
from .phase2 import ControlConfig,make_variants
from .phase3 import verify_seal
from .phase4_analysis import NEW_CHANNELS,summarize
from .public_analysis import PUBLIC_CONDITIONS
from .runtime import LegacyMelPredictor,preflight,source_fingerprint
from .runner import timestamp
from .source_matching import MATCHING_VERSION
from .transforms import Audio,load_audio,write_audio


@dataclass(frozen=True)
class NuisanceConfig:
    experiment_id: str
    selection_path: str
    selection_sha256: str
    phase3_cache: str
    phase3_cache_metadata_sha256: str
    source_matching_report: str
    source_matching_sha256: str
    scope: str='EXPLORATORY_WITHIN_RECORD_DEVELOPMENT'
    seed: int=4804
    transform_seed: int=3804
    policies: tuple=POLICIES
    replicates: int=2000

    def __post_init__(self):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,60}',self.experiment_id) or self.scope!='EXPLORATORY_WITHIN_RECORD_DEVELOPMENT':
            raise ValueError('Bounded development identity required; no confirmation scope')
        if tuple(self.policies)!=POLICIES or self.seed<0 or self.replicates<100 or self.transform_seed!=3804:
            raise ValueError('Prespecified three policies and original Phase 3 noise seed required')
        if any(not re.fullmatch('[a-f0-9]{64}',value) for value in (self.selection_sha256,self.phase3_cache_metadata_sha256,self.source_matching_sha256)):
            raise ValueError('All input seals required')


def check_inputs(root: Path,config: NuisanceConfig) -> tuple[Path,list[dict]]:
    for name,digest in ((config.selection_path,config.selection_sha256),(config.source_matching_report,config.source_matching_sha256)):
        if file_hash(root/name)!=digest:
            raise ValueError('Locked input changed: '+name)
    cache=root/config.phase3_cache
    if file_hash(cache/'cache_metadata.json')!=config.phase3_cache_metadata_sha256:
        raise ValueError('Phase 3 cache seal changed')
    verify_seal(cache,json.loads((cache/'cache_metadata.json').read_text()))
    selection=json.loads((root/config.selection_path).read_text())
    if selection['scope']!='PUBLIC_DEVELOPMENT' or selection['matching_status']!='SOURCE_MATCHED_EXPERIMENT_BLOCKED':
        raise ValueError('Only explicitly labelled within-record fallback is implemented')
    originals={row['sample_id']:row for row in read_csv(cache/'manifest.csv')}
    records=selection['records']
    if len(records)!=32 or len({r['sample_id'] for r in records})!=32 or any(r!=originals.get(r['sample_id']) for r in records):
        raise ValueError('Exact 32-record metadata-only selection required')
    if any(sum(int(r['label'])==label for r in records)!=16 for label in (0,1)):
        raise ValueError('Balanced class coverage required')
    return cache,records


def prepare(root: Path,config: NuisanceConfig) -> Path:
    environment=preflight(root)
    estimate=32*len(POLICIES)*len(NEW_CHANNELS)*3*22050*4
    if environment['free_disk_bytes']<estimate+1024**3:
        raise ValueError('Insufficient disk reserve')
    cache,records=check_inputs(root,config)
    output=root/'outputs/jaes_v2'/(config.experiment_id+'_waveforms')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',asdict(config)); write_csv(output/'manifest.csv',records)
    write_json(output/'resource_preflight.json',{'free_disk_bytes':environment['free_disk_bytes'],'new_waveform_bytes_estimate':estimate,
        'new_waveforms':288,'new_predictions_two_artifacts':576,'baseline_predictions_reused':448,
        'runtime_estimate_seconds':400,'additional_audio_download_bytes':0})
    original_conditions={(r['sample_id'],r['condition_id']):r for r in read_csv(cache/'conditions.csv')}
    conditions=[]; features=[]; started=perf_counter()
    from src.preprocessing import load_audio as production_load
    for source in records:
        identity=source['sample_id']
        import soundfile as sf
        full,rate=sf.read(root/source['path'],dtype='float32',always_2d=True)
        features.append({'sample_id':identity,'input_definition':'original_full_source','policy':'original_source','condition':'native',
                         **signal_features(Audio(full.mean(axis=1),rate)),'original_duration':source['duration'],
                         'original_fraction_not_observed':max(0,1-3/float(source['duration']))})
        baseline=original_conditions[(identity,'clean')]
        prepared=load_audio(cache/baseline['path'],sample_rate=22050,max_seconds=3)
        # Measure the actual padded 3-second production input, separately from serialized/full source.
        exact=Audio(production_load(cache/baseline['path']),22050)
        features.append({'sample_id':identity,'input_definition':'phase3_prepared_detector_input','policy':'phase3_baseline','condition':'clean',
                         **signal_features(exact),'original_duration':source['duration'],
                         'original_fraction_not_observed':max(0,1-3/float(source['duration']))})
        for condition in ('clean_native','fixed_A_then_B','fixed_B_then_A'):
            item=original_conditions[(identity,condition)]
            features.append({'sample_id':identity,'input_definition':'phase3_transformed_detector_input','policy':'phase3_baseline','condition':condition,
                             **signal_features(Audio(production_load(cache/item['path']),22050)),'original_duration':source['duration'],
                             'original_fraction_not_observed':max(0,1-3/float(source['duration']))})
        for policy in POLICIES:
            controlled,evidence=intervene(prepared,policy)
            variants,channel_evidence=make_variants(controlled,ControlConfig(config.experiment_id,seed=config.transform_seed),
                                                   sample_id=identity,ffmpeg=environment['ffmpeg'])
            for condition in NEW_CHANNELS:
                relative=Path('audio')/policy/condition/(identity+'.wav')
                target=output/relative; target.parent.mkdir(parents=True,exist_ok=True)
                write_audio(target,variants[condition])
                conditions.append({'sample_id':identity,'policy':policy,'condition_id':condition,'path':relative.as_posix(),
                                   'sha256':file_hash(target),'intervention':json.dumps(evidence,sort_keys=True),
                                   'channel_evidence':json.dumps(channel_evidence[condition],sort_keys=True)})
                features.append({'sample_id':identity,'input_definition':'new_transformed_detector_input','policy':policy,'condition':condition,
                                 **signal_features(Audio(production_load(target),22050)),'original_duration':source['duration'],
                                 'original_fraction_not_observed':max(0,1-3/float(source['duration']))})
        write_csv(output/'conditions.csv',conditions)
        print(f'Prepared {len(conditions)}/288 nuisance waveforms',flush=True)
    write_csv(output/'signal_features.csv',features)
    metadata={'status':'SEALED_WITHIN_RECORD_DEVELOPMENT','config_hash':object_hash(asdict(config)),
              'source_matching_version':MATCHING_VERSION,'environment':environment,'code':source_fingerprint(root),
              'elapsed_seconds':perf_counter()-started,'created':timestamp(),
              'output_hashes':{p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file()}}
    write_json(output/'cache_metadata.json',metadata)
    return output


def aggregate(run: Path) -> dict:
    meta=json.loads((run/'run_metadata.json').read_text())
    for name,digest in meta['analysis_inputs'].items():
        if file_hash(run/name)!=digest:
            raise ValueError('Analysis input changed')
    shared=Path(meta['shared_cache'])
    if file_hash(shared/'cache_metadata.json')!=meta['shared_cache_sha256']:
        raise ValueError('Shared cache seal changed')
    verify_seal(shared,json.loads((shared/'cache_metadata.json').read_text()))
    config=NuisanceConfig(**json.loads((run/'config.json').read_text()))
    if object_hash(asdict(config))!=json.loads((shared/'cache_metadata.json').read_text())['config_hash']:
        raise ValueError('Analysis configuration differs from prepared protocol')
    report=summarize(read_csv(run/'scores.csv'),read_csv(run/'manifest.csv'),seed=config.seed,replicates=config.replicates)
    write_json(run/'statistics.json',report)
    write_csv(run/'metrics.csv',[{**row,'class_score_distributions':json.dumps(row['class_score_distributions'],sort_keys=True)} for row in report['metrics']])
    write_json(run/'precision.json',report['precision_plans'])
    from .phase4_figures import draw
    draw(run,report)
    return report


def infer(root: Path,config: NuisanceConfig,model: str) -> Path:
    if model not in {'mobilenetv3_lstm','shufflenetv2_lstm'}:
        raise ValueError('Unsupported existing artifact')
    environment=preflight(root); previous,records=check_inputs(root,config)
    cache=root/'outputs/jaes_v2'/(config.experiment_id+'_waveforms')
    shared=json.loads((cache/'cache_metadata.json').read_text()); verify_seal(cache,shared)
    if shared['config_hash']!=object_hash(asdict(config)):
        raise ValueError('Nuisance configuration mismatch')
    old=root/'outputs/jaes_v2'/('c1_mlaad_tiny_dev_v1_'+model)
    oldmeta=json.loads((old/'run_metadata.json').read_text()); verify_seal(old,oldmeta)
    ids={r['sample_id'] for r in records}
    reused=[{**row,'policy':'phase3_baseline','score_origin':'SEALED_PHASE3_REUSE'} for row in read_csv(old/'scores.csv') if row['sample_id'] in ids]
    output=root/'outputs/jaes_v2'/(config.experiment_id+'_'+model); output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'manifest.csv',records); write_json(output/'config.json',asdict(config))
    metadata={'status':'RUNNING','scope':config.scope,'environment':environment,'code':source_fingerprint(root),
              'shared_cache':str(cache.resolve()),'shared_cache_sha256':file_hash(cache/'cache_metadata.json'),
              'phase3_scores_sha256':file_hash(old/'scores.csv'),'matching_version':MATCHING_VERSION,'started':timestamp()}
    write_json(output/'run_metadata.json',metadata)
    started=perf_counter(); scores=reused[:]
    try:
        predictor=LegacyMelPredictor(model)
        metadata.update(artifact_hashes=predictor.artifact_hashes,threshold=predictor.threshold,warmup_calls=1)
        sources={r['sample_id']:r for r in records}
        for item in read_csv(cache/'conditions.csv'):
            source=sources[item['sample_id']]
            row={**source,'model_id':model,'condition_id':item['condition_id'],'policy':item['policy'],
                 'threshold':predictor.threshold,'status':'OK','score_origin':'PHASE4_NEW_INFERENCE'}
            path=cache/item['path']
            if len(scores)==len(reused):
                predictor.predict(path)
            try:
                value,timing=predictor.predict(path)
                row.update(raw_score=value,predicted_label=int(value>=predictor.threshold),**timing)
            except Exception as error:
                row.update(raw_score='',predicted_label='',status='FAILED',error=f'{type(error).__name__}: {error}')
            scores.append(row)
            # Serialize a stable compact score schema, preserving every failure.
            fields=['sample_id','parent_recording_id','dataset_id','generator_id','label','model_id','condition_id','policy',
                    'threshold','status','score_origin','raw_score','predicted_label','error']
            write_csv(output/'scores.csv',[{key:r.get(key,'') for key in fields} for r in scores],fields)
            if (len(scores)-len(reused))%9==0:
                print(f'{model}: {len(scores)-len(reused)}/288 new scores; {len(reused)} reused',flush=True)
        metadata['counts']={'new_predictions':len(scores)-len(reused),'successful_new':sum(r['status']=='OK' for r in scores[len(reused):]),
                            'failed_new':sum(r['status']!='OK' for r in scores[len(reused):]),'reused_scores':len(reused),'excluded':0}
        metadata['analysis_inputs']={name:file_hash(output/name) for name in ('scores.csv','manifest.csv','config.json')}
        write_json(output/'run_metadata.json',metadata); aggregate(output)
        metadata.update(status='WITHIN_RECORD_DEVELOPMENT_EVALUATED',elapsed_seconds=perf_counter()-started,finished=timestamp())
        metadata['output_hashes']={p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='run_metadata.json'}
        write_json(output/'run_metadata.json',metadata)
    except Exception as error:
        metadata.update(status='FAILED',error=f'{type(error).__name__}: {error}'); write_json(output/'run_metadata.json',metadata)
        raise
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','infer','aggregate','verify'))
    parser.add_argument('--config',type=Path); parser.add_argument('--model',choices=('mobilenetv3_lstm','shufflenetv2_lstm'))
    parser.add_argument('--run',type=Path); parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.mode in {'prepare','infer'}:
        if not args.config or (args.mode=='infer' and not args.model):
            parser.error('--config and, for inference, --model required')
        config=NuisanceConfig(**json.loads(args.config.read_text())); root=Path(__file__).resolve().parents[3]
        print(prepare(root,config) if args.mode=='prepare' else infer(root,config,args.model))
    elif args.mode=='aggregate':
        print(aggregate(args.run)['scope'])
    else:
        if not args.run or not args.report or args.report.exists():
            parser.error('--run and new --report required')
        metadata=json.loads((args.run/'run_metadata.json').read_text())
        count=verify_seal(args.run,metadata)
        digest=file_hash(args.run/'statistics.json'); aggregate(args.run)
        if file_hash(args.run/'statistics.json')!=digest:
            raise ValueError('Statistics parity failed')
        verify_seal(args.run,metadata)
        import sys
        write_json(args.report,{'status':'PASS','sealed_run_files':count,'statistics_parity':True,
                               'neural_framework_imported':any(n in sys.modules for n in ('tensorflow','torch'))})


if __name__=='__main__':
    main()
