"""Verify immutable Phase 1–5 evidence and snapshot reviewable research sources."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from src.lava.jaes_v2.evidence import file_hash,read_csv,write_json
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.runtime import preflight,source_fingerprint


def main():
    root=Path(__file__).resolve().parents[2]; audit=root/'outputs/jaes_v2/audit'; target=audit/'phase5_final_integrity.json'
    if target.exists(): raise FileExistsError('Final integrity evidence already exists')
    before=json.loads((audit/'phase5_before.json').read_text())
    changed=[name for name,digest in before['historical_files'].items() if file_hash(root/name)!=digest]
    if changed: raise ValueError('Historical artifacts changed: '+str(changed[:10]))
    config=json.loads((root/'configs/jaes_v2/c2_p0_erm_pilot_v2.json').read_text()); cache=root/config['cache']
    cache_seal=verify_seal(cache,json.loads((cache/'cache_metadata.json').read_text()))
    pp=root/'outputs/jaes_v2/phase5_preprocessing_study_v1'; pp_seal=verify_seal(pp,json.loads((pp/'run_metadata.json').read_text()))
    runs={}
    for name in ['c2_overfit_sanity_v1']+[f'c2_{p}_{b}_pilot_v2' for p in ('p0','p1') for b in ('erm','aug')]:
        run=root/'outputs/jaes_v2/c2'/name; metadata=json.loads((run/'run_metadata.json').read_text())
        expected='SMOKE_TESTED' if name=='c2_overfit_sanity_v1' else 'PILOT_EVALUATED'
        if metadata['status']!=expected: raise ValueError('Incomplete research run: '+name)
        if expected=='PILOT_EVALUATED' and (metadata['optimizer_steps']!=48 or len(read_csv(run/'training_history.csv'))!=6):
            raise ValueError('Main factorial cells must complete identical six-epoch / 48-update budgets')
        runs[name]={'verified_files':verify_seal(run,metadata),'checkpoint_sha256':file_hash(run/'best.keras'),'status':metadata['status']}
    validation=json.loads((audit/'phase5_validation_final_v3/validation.json').read_text())
    if not all(s['successful'] for s in validation['suites'].values()): raise ValueError('Regression failed')
    first=audit/'phase5_report_v2'; second=audit/'phase5_report_reproduced_v2'
    verify_seal(first,json.loads((first/'report_metadata.json').read_text())); verify_seal(second,json.loads((second/'report_metadata.json').read_text()))
    comparable=[p.name for p in first.iterdir() if p.is_file() and p.name!='report_metadata.json']
    if any(file_hash(first/name)!=file_hash(second/name) for name in comparable): raise ValueError('Model-free reproduction mismatch')
    foundation=root/'outputs/jaes_v2/datasets/asv2019_foundation_v1'; data=json.loads((foundation/'foundation.json').read_text())
    for role,digest in data['manifest_hashes'].items():
        if file_hash(foundation/(role+'_manifest.csv'))!=digest: raise ValueError('Foundation manifest changed')
    if file_hash(root/'data/downloads/LA.zip')!=data['archive_sha256']: raise ValueError('User archive changed')
    if shutil.disk_usage(root).free<15*1024**3: raise ValueError('Disk reserve violated')
    claims=root/'papers/jaes_v2/CLAIM_EVIDENCE_MAP.csv'; rows=read_csv(claims)
    additions=[
        dict(claim_id='P501',claim='50,224 official ASV CM records inventoried with speaker-separated calibration',scope='LOCAL_DATA_FOUNDATION',evidence='outputs/jaes_v2/datasets/asv2019_foundation_v1/foundation.json',status='SUPPORTED_LOCAL_INTEGRITY',limitation='Remote digest and original-source lineage not certified; evaluation unopened'),
        dict(claim_id='P502',claim='P1/P2 remove half-gain flips in the 32-record diagnostic; P1 worsens ShuffleNet discrimination',scope='ALTERED_PREPROCESSOR_DEVELOPMENT',evidence='outputs/jaes_v2/audit/phase5_report_v2/summary.json',status='SUPPORTED_DIAGNOSTIC',limitation='Frozen old models; changed input distribution; not novel trained-detector quality'),
        dict(claim_id='P503',claim='Four controlled single-seed frozen-backbone baseline pilots completed',scope='ASV_TINY_DEVELOPMENT',evidence='outputs/jaes_v2/audit/phase5_report_v2/pilot_metrics.csv',status='PILOT_EVALUATED',limitation='Known preprocessing and augmentation; no multi-seed, strong-baseline or independent-confirmation claim')]
    for row in additions:
        if row['claim_id'] not in {r['claim_id'] for r in rows}: rows.append(row)
    with claims.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths=[]
    for name in ('src/lava/jaes_v2','scripts/jaes_v2','tests/jaes_v2','configs/jaes_v2','docs/jaes_v2','papers/jaes_v2'):
        paths.extend(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    snapshot=audit/'phase5_source_snapshot.zip'
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths): archive.write(path,path.relative_to(root).as_posix())
        for path in sorted((audit/'phase5_source_versions').glob('*.py')):
            archive.write(path,path.relative_to(root).as_posix())
    (audit/'phase5_working_tree.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=root))
    environment=preflight(root); environment['execution_policy']='New CPU research head training only; frozen conference artifacts; no evaluation or calibration fitting'
    report={'status':'PASS','environment_and_baseline':environment,'historical_run_files_unchanged':len(before['historical_files']),
        'cache_files_verified':cache_seal,'preprocessing_files_verified':pp_seal,'research_runs':runs,'model_free_reproduction_files':len(comparable),
        'source_snapshot_files':len(paths),'source_snapshot_sha256':file_hash(snapshot),'code':source_fingerprint(root),'regression':validation['suites'],
        'executed_source_variants':{p.name:file_hash(p) for p in (audit/'phase5_source_versions').glob('*.py')},
        'new_download_bytes':0,'archive_sha256_reverified':data['archive_sha256'],'teacher_loaded':False,'new_research_training':True,
        'C1':'SUPPORTING_VALIDATION_ENGINEERING','C2':'STANDARD_BASELINES_PILOT_EVALUATED; NOVELTY_NOT_ESTABLISHED','C3':'DEFERRED','confirmation':'NOT_LOCKED'}
    write_json(target,report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'environment_and_baseline','code','regression'}},indent=2))


if __name__=='__main__': main()
