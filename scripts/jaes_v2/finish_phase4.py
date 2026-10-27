"""Final integrity, source snapshot and claim map for executed Phase 4 evidence."""
import csv
import json
from pathlib import Path
import subprocess
import zipfile

from src.lava.jaes_v2.evidence import file_hash,read_csv,write_json
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.runtime import preflight,source_fingerprint


def main() -> None:
    root=Path(__file__).resolve().parents[2]; audit=root/'outputs/jaes_v2/audit'
    target=audit/'phase4_final_integrity.json'
    if target.exists():
        raise FileExistsError('New finalization identity required')
    before=json.loads((audit/'phase4_before.json').read_text())
    changed=[name for name,digest in before['historical_files'].items() if file_hash(root/name)!=digest]
    if changed:
        raise ValueError('Historical output changed: '+str(changed))
    cache=root/'outputs/jaes_v2/c1_nuisance_controls_dev_v1_waveforms'
    shared=json.loads((cache/'cache_metadata.json').read_text()); count=verify_seal(cache,shared)
    seals={}
    for model in ('mobilenetv3_lstm','shufflenetv2_lstm'):
        run=root/'outputs/jaes_v2'/('c1_nuisance_controls_dev_v1_'+model)
        meta=json.loads((run/'run_metadata.json').read_text())
        if meta['counts']['new_predictions']!=288 or meta['counts']['failed_new']:
            raise ValueError('Incomplete actual experiment')
        seals[model]=verify_seal(run,meta)
    validation=json.loads((audit/'phase4_validation_final/validation.json').read_text())
    if not all(result['successful'] for result in validation['suites'].values()):
        raise ValueError('Final regression did not pass')
    claims=root/'papers/jaes_v2/CLAIM_EVIDENCE_MAP.csv'; rows=read_csv(claims)
    additions=[
        dict(claim_id='P401',claim='Phase3 selected systems have zero original-source overlap',scope='METADATA_AUDIT',
             evidence='outputs/jaes_v2/audit/phase4_source_matching/matching_coverage.json',status='SUPPORTED_AUDIT',
             limitation='Sparse full-release author matches exist; no level4 or adequate fixed-pair cohort'),
        dict(claim_id='P402',claim='576 new nuisance predictions and 448 existing scores analyzed',scope='WITHIN_RECORD_DEVELOPMENT',
             evidence='outputs/jaes_v2/audit/phase4_report_v1/summary.json',status='SUPPORTED_EXECUTION',
             limitation='Four original source-text clusters; not matched generator effect or confirmation'),
        dict(claim_id='P403',claim='Absolute Mel floor explains observed feature gain non-invariance up to floating precision',scope='MODEL_FREE_DIAGNOSTIC',
             evidence='outputs/jaes_v2/audit/phase4_feature_diagnostics_v2/diagnostic.json',status='SUPPORTED_ENGINEERING_DIAGNOSIS',
             limitation='No changed-preprocessor detector inference or primary scientific novelty')]
    for entry in additions:
        if entry['claim_id'] not in {row['claim_id'] for row in rows}:
            rows.append(entry)
    with claims.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths=[]
    for name in ('src/lava/jaes_v2','scripts/jaes_v2','tests/jaes_v2','configs/jaes_v2','docs/jaes_v2','papers/jaes_v2'):
        paths.extend(path for path in (root/name).rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    snapshot=audit/'phase4_source_snapshot.zip'
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path,path.relative_to(root).as_posix())
    (audit/'phase4_working_tree.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=root))
    result={'status':'PASS','environment_and_baseline':preflight(root),'historical_run_files_unchanged':len(before['historical_files']),
            'new_shared_files_verified':count,'new_run_files_verified':seals,'source_snapshot_files':len(paths),
            'source_snapshot_sha256':file_hash(snapshot),'code':source_fingerprint(root),
            'regression':validation['suites'],'source_matched_experiment':'BLOCKED',
            'C1':'SUPPORTING_VALIDATION_ENGINEERING; NOVELTY_NOT_ESTABLISHED','confirmation':'NOT_LOCKED',
            'new_audio_downloaded_bytes':0,'teacher_loaded':False,'training_executed':False}
    write_json(target,result)
    print(json.dumps({key:value for key,value in result.items() if key not in {'environment_and_baseline','code','regression'}},indent=2))


if __name__=='__main__':
    main()
