"""Public-only development: shared waveform cache, frozen detector artifacts, sealed analysis."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import perf_counter
import sys

from .analysis import SCORE_FIELDS
from .controls import codec_delay_probe
from .data_audit import audit_sources
from .datasets import load_public_manifest
from .evidence import file_hash, object_hash, read_csv, write_csv, write_json
from .mlaad import confounding_audit
from .overlap import near_duplicate_candidates, overlap_dimensions
from .phase2 import ControlConfig, make_variants
from .provenance import UNKNOWN
from .public_analysis import PUBLIC_CONDITIONS, summarize_public
from .runtime import LegacyMelPredictor, preflight, source_fingerprint
from .runner import timestamp
from .transforms import diagnostics, load_audio, write_audio


@dataclass(frozen=True)
class PublicConfig:
    experiment_id: str
    manifest: str
    manifest_sha256: str
    selection_sha256: str
    per_class: int = 96
    seed: int = 3804
    bootstrap_replicates: int = 2000
    scope: str = 'PUBLIC_DEVELOPMENT'
    conditions: tuple = PUBLIC_CONDITIONS

    def __post_init__(self):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,60}',self.experiment_id) or self.scope != 'PUBLIC_DEVELOPMENT':
            raise ValueError('Explicit public development identity required; confirmation unavailable')
        if tuple(self.conditions) != PUBLIC_CONDITIONS or not 8 <= self.per_class <= 200 or self.seed < 0 or self.bootstrap_replicates < 100:
            raise ValueError('Invalid bounded public condition/cohort configuration')
        if any(not re.fullmatch('[a-f0-9]{64}',value) for value in (self.manifest_sha256,self.selection_sha256)):
            raise ValueError('Locked manifest and selection hashes required')


def verify_seal(directory: Path, metadata: dict) -> int:
    for name, digest in metadata['output_hashes'].items():
        if file_hash(directory/name) != digest:
            raise ValueError('Sealed artifact changed: ' + name)
    return len(metadata['output_hashes'])


def prepare(root: Path, config: PublicConfig) -> Path:
    environment = preflight(root)
    manifest = root/config.manifest
    selection = manifest.parent/'selection.json'
    if file_hash(manifest) != config.manifest_sha256 or file_hash(selection) != config.selection_sha256:
        raise ValueError('Locked public selection/manifest changed')
    sources = load_public_manifest(manifest)
    if any(row.split != 'development' or row.research_split != 'development' or row.dataset_id != 'MLAAD_TINY'
           for row in sources) or any(sum(row.label == label for row in sources) != config.per_class for label in (0,1)):
        raise ValueError('Only the exact balanced imported MLAAD development cohort is allowed')
    source_metadata = json.loads((manifest.parent/'source_metadata.json').read_text())
    if source_metadata['status'] != 'IMPORTED_REAL_PUBLIC_AUDIO' or source_metadata['manifest_sha256'] != file_hash(manifest):
        raise ValueError('Real-public import gate failed')
    estimate = len(sources)*len(PUBLIC_CONDITIONS)*3*22050*4
    if environment['free_disk_bytes'] < estimate + 1024**3 or not environment['ffmpeg']:
        raise ValueError('Insufficient disk reserve or FFmpeg unavailable')
    output = root/'outputs/jaes_v2'/(config.experiment_id+'_waveforms')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',asdict(config))
    write_csv(output/'manifest.csv',[row.to_dict() for row in sources])
    audit, features = audit_sources(root,sources)
    near = near_duplicate_candidates({row.sample_id:load_audio(root/row.path,sample_rate=22050,max_seconds=3) for row in sources},max_pairs=80000)
    write_json(output/'source_audit.json',audit)
    write_csv(output/'source_features.csv',features)
    write_json(output/'near_duplicates.json',near)
    write_json(output/'overlap_dimensions.json',overlap_dimensions(sources))
    confounding = confounding_audit(sources)
    write_json(output/'confounding_audit.json',confounding)
    if confounding['language_label_complete_separation'] or confounding['source_corpus_label_complete_separation']:
        raise ValueError('CONFOUNDED_FACTOR: class separated by language/source corpus')
    if audit['decoded_duplicate_groups'] or near:
        raise ValueError('Duplicate/candidate review required before inference; no automatic exclusion')
    legacy = read_csv(root/'data/manifests/split_manifest.csv')
    overlap = sorted({row.checksum for row in sources} & {row['sha256'] for row in legacy})
    write_json(output/'legacy_overlap.json',{'exact_byte_overlap':overlap,'legacy_manifest_sha256':file_hash(root/'data/manifests/split_manifest.csv'),
        'speaker_generator_original_overlap':UNKNOWN,'decoded_legacy_overlap':'NOT_EXHAUSTIVELY_SCREENED',
        'independence_certificate':False})
    if overlap:
        raise ValueError('Public/legacy exact byte overlap; review required')
    write_json(output/'codec_delay_probe.json',codec_delay_probe(environment['ffmpeg']))
    configuration = ControlConfig(config.experiment_id,per_class=min(config.per_class,100),seed=config.seed)
    conditions = []
    started = perf_counter()
    for source in sources:
        audio = load_audio(root/source.path,sample_rate=22050,max_seconds=3)
        variants,evidence = make_variants(audio,configuration,sample_id=source.sample_id,ffmpeg=environment['ffmpeg'])
        for condition in PUBLIC_CONDITIONS:
            relative = Path('audio')/condition/(source.sample_id+'.wav')
            target = output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            write_audio(target,variants[condition])
            conditions.append({'sample_id':source.sample_id,'condition_id':condition,'path':relative.as_posix(),
                               'sha256':file_hash(target),'settings':json.dumps(evidence[condition],sort_keys=True),
                               'diagnostics':json.dumps(diagnostics(variants[condition]),sort_keys=True)})
        write_csv(output/'conditions.csv',conditions)
        print(f'Prepared {len(conditions)}/{len(sources)*len(PUBLIC_CONDITIONS)} shared waveforms',flush=True)
    metadata = {'status':'PUBLIC_DEVELOPMENT_WAVEFORMS_SEALED','created':timestamp(),'environment':environment,
                'config_hash':object_hash(asdict(config)), 'code':source_fingerprint(root),'elapsed_seconds':perf_counter()-started,
                'input_source_hashes':{str((root/row.path).resolve()):row.checksum for row in sources},
                'conditions':list(PUBLIC_CONDITIONS),'independent_population_units':UNKNOWN}
    metadata['output_hashes'] = {p.relative_to(output).as_posix():file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'cache_metadata.json',metadata)
    return output


def aggregate(run: Path) -> dict:
    metadata = json.loads((run/'run_metadata.json').read_text())
    for name,digest in metadata['analysis_input_hashes'].items():
        if file_hash(run/name) != digest:
            raise ValueError('Analysis input changed: '+name)
    cache = Path(metadata['cache'])
    if file_hash(cache/'cache_metadata.json') != metadata['cache_metadata_sha256']:
        raise ValueError('Cache metadata changed')
    shared = json.loads((cache/'cache_metadata.json').read_text())
    verify_seal(cache,shared)
    config = PublicConfig(**json.loads((run/'config.json').read_text()))
    if object_hash(asdict(config)) != metadata['config_hash']:
        raise ValueError('Run configuration changed')
    report = summarize_public(read_csv(run/'scores.csv'),read_csv(run/'manifest.csv'),replicates=config.bootstrap_replicates,seed=config.seed)
    write_json(run/'statistics.json',report)
    write_csv(run/'metrics.csv',report['metric_table'])
    write_json(run/'interactions.json',report['generator_channel_interactions'])
    from .public_figures import make_figures
    make_figures(run,report)
    return report


def infer(root: Path,config: PublicConfig,model_id: str) -> Path:
    if model_id not in {'mobilenetv3_lstm','shufflenetv2_lstm'}:
        raise ValueError('Unapproved detector')
    environment = preflight(root)
    cache = root/'outputs/jaes_v2'/(config.experiment_id+'_waveforms')
    shared = json.loads((cache/'cache_metadata.json').read_text())
    verify_seal(cache,shared)
    if shared['status'] != 'PUBLIC_DEVELOPMENT_WAVEFORMS_SEALED' or shared['config_hash'] != object_hash(asdict(config)):
        raise ValueError('Cache/config provenance mismatch')
    for path,digest in shared['input_source_hashes'].items():
        if file_hash(Path(path)) != digest:
            raise ValueError('Original public audio changed')
    output = root/'outputs/jaes_v2'/(config.experiment_id+'_'+model_id)
    output.mkdir(parents=True,exist_ok=False)
    manifest = read_csv(cache/'manifest.csv')
    write_csv(output/'manifest.csv',manifest)
    write_json(output/'config.json',asdict(config))
    metadata = {'status':'RUNNING','scope':'EXPLORATORY_PUBLIC_DEVELOPMENT','started':timestamp(),
                'cache':str(cache.resolve()),'cache_metadata_sha256':file_hash(cache/'cache_metadata.json'),
                'config_hash':object_hash(asdict(config)), 'code':source_fingerprint(root),'environment':environment,
                'model_id':model_id,'training':'NONE','calibration':'NONE','threshold_policy':'INHERITED_FIXED',
                'independent_population_units':UNKNOWN}
    write_json(output/'run_metadata.json',metadata)
    started = perf_counter()
    try:
        predictor = LegacyMelPredictor(model_id)
        metadata.update(threshold=predictor.threshold,artifact_hashes=predictor.artifact_hashes,model_load_seconds=predictor.load_seconds)
        sources = {row['sample_id']:row for row in manifest}
        scores = []
        for item in read_csv(cache/'conditions.csv'):
            source = sources[item['sample_id']]
            path = cache/item['path']
            if not scores:
                predictor.predict(path)
            row = dict.fromkeys(SCORE_FIELDS,'')
            row.update(run_id=output.name,model_id=model_id,condition_id=item['condition_id'],transform_order=item['condition_id'],
                       seed=config.seed,threshold=predictor.threshold,calibrated_score='NOT_FITTED',status='OK',abstain=False,stage_used=1)
            row.update({key:source[key] for key in ('sample_id','parent_recording_id','dataset_id','generator_id','speaker_id','label')})
            try:
                score,timing = predictor.predict(path)
                predicted = int(score >= predictor.threshold)
                row.update(raw_score=score,predicted_label=predicted,correct=predicted==int(source['label']),**timing)
            except Exception as error:
                row.update(status='FAILED',error=f'{type(error).__name__}: {error}')
            scores.append(row)
            if len(scores) % len(PUBLIC_CONDITIONS) == 0:
                write_csv(output/'scores.csv',scores,SCORE_FIELDS)
                print(f'{model_id}: {len(scores)}/{len(manifest)*len(PUBLIC_CONDITIONS)} scores persisted',flush=True)
        metadata['counts'] = {'attempted':len(scores),'successful':sum(row['status']=='OK' for row in scores),
                              'failed':sum(row['status']!='OK' for row in scores),'excluded':0}
        metadata['analysis_input_hashes'] = {name:file_hash(output/name) for name in ('scores.csv','manifest.csv','config.json')}
        write_json(output/'run_metadata.json',metadata)
        aggregate(output)
        metadata.update(status='EXTERNAL_DEVELOPMENT_EVALUATED',finished=timestamp(),elapsed_seconds=perf_counter()-started)
        metadata['output_hashes'] = {p.relative_to(output).as_posix():file_hash(p) for p in sorted(output.rglob('*'))
                                     if p.is_file() and p.name!='run_metadata.json'}
        write_json(output/'run_metadata.json',metadata)
    except Exception as error:
        metadata.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_metadata.json',metadata)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','infer','aggregate','verify'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--model',choices=('mobilenetv3_lstm','shufflenetv2_lstm'),default='mobilenetv3_lstm')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    if args.mode in {'prepare','infer'}:
        if not args.config:
            parser.error('--config required')
        config = PublicConfig(**json.loads(args.config.read_text()))
        root = Path(__file__).resolve().parents[3]
        print(prepare(root,config) if args.mode=='prepare' else infer(root,config,args.model))
    elif args.mode == 'aggregate':
        if not args.run:
            parser.error('--run required')
        print(aggregate(args.run)['scope'])
    else:
        if not args.run or not args.report or args.report.exists():
            parser.error('--run and new --report required')
        metadata = json.loads((args.run/'run_metadata.json').read_text())
        count = verify_seal(args.run,metadata)
        prior = object_hash(json.loads((args.run/'statistics.json').read_text()))
        aggregate(args.run)
        if object_hash(json.loads((args.run/'statistics.json').read_text())) != prior:
            raise ValueError('Statistics reproduction mismatch')
        write_json(args.report,{'status':'PASS','sealed_run_files':count,'shared_cache_verified':True,
                               'model_free_statistics_parity':True,'framework_imported':any(n in sys.modules for n in ('tensorflow','torch'))})


if __name__ == '__main__':
    main()
