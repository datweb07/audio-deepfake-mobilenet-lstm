"""Seal final integrity and a portable source snapshot without touching historical runs."""
import csv
import json
from pathlib import Path
import subprocess
import zipfile

from src.lava.jaes_v2.evidence import file_hash, read_csv, write_json
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.runtime import preflight, source_fingerprint


def main():
    root=Path(__file__).resolve().parents[2]
    audit=root/'outputs/jaes_v2/audit'
    target=audit/'phase3_final_integrity.json'
    if target.exists():
        raise FileExistsError('Final integrity report already exists')
    before=json.loads((audit/'phase3_before.json').read_text())
    changed=[name for name,digest in before['historical_files'].items() if file_hash(root/name)!=digest]
    if changed:
        raise ValueError('Historical output changed: '+str(changed))
    manifest=read_csv(root/'outputs/jaes_v2/datasets/mlaad_tiny_dev_v1/provenance_manifest.csv')
    for row in manifest:
        if file_hash(Path(row['path'])) != row['checksum']:
            raise ValueError('Acquired original audio changed')
    seals={}
    for model in ('mobilenetv3_lstm','shufflenetv2_lstm'):
        run=root/'outputs/jaes_v2'/('c1_mlaad_tiny_dev_v1_'+model)
        metadata=json.loads((run/'run_metadata.json').read_text())
        seals[model]=verify_seal(run,metadata)
    cache=root/'outputs/jaes_v2/c1_mlaad_tiny_dev_v1_waveforms'
    cache_count=verify_seal(cache,json.loads((cache/'cache_metadata.json').read_text()))
    claims=root/'papers/jaes_v2/CLAIM_EVIDENCE_MAP.csv'
    rows=read_csv(claims)
    for entry in [
        dict(claim_id='P301',claim='192 real public MLAAD-tiny audio imported with upstream checksums',scope='PUBLIC_DEVELOPMENT',
             evidence='outputs/jaes_v2/datasets/mlaad_tiny_dev_v1/source_metadata.json',status='SUPPORTED_IMPLEMENTATION',limitation='REAL speaker/parent lineage UNKNOWN'),
        dict(claim_id='P302',claim='2688 executed predictions on shared public channel waveforms',scope='PUBLIC_DEVELOPMENT',
             evidence='outputs/jaes_v2/audit/phase3_report_v1/summary.json',status='SUPPORTED_EXECUTION',limitation='Existing artifacts and thresholds; no improvement or independence claim'),
        dict(claim_id='P303',claim='Annotated-system channel contrasts computed with source-text clusters',scope='EXPLORATORY_PUBLIC_DEVELOPMENT',
             evidence='outputs/jaes_v2/audit/phase3_report_v1/table4_interactions.csv',status='DESCRIPTIVE_ONLY',limitation='Four source-text speakers; observational; no novelty or confirmation')]:
        if entry['claim_id'] not in {row['claim_id'] for row in rows}:
            rows.append(entry)
    with claims.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    paths=[]
    for name in ('src/lava/jaes_v2','scripts/jaes_v2','tests/jaes_v2','configs/jaes_v2','docs/jaes_v2','papers/jaes_v2'):
        paths.extend(path for path in (root/name).rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    archive=audit/'phase3_source_snapshot.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(paths):
            stream.write(path,path.relative_to(root).as_posix())
    patch=subprocess.check_output(['git','diff','--binary'],cwd=root)
    (audit/'phase3_working_tree.patch').write_bytes(patch)
    result={'status':'PASS','environment_and_baseline':preflight(root),
            'historical_run_files_unchanged':len(before['historical_files']),
            'actual_public_original_hashes_verified':len(manifest),'shared_cache_files_verified':cache_count,
            'run_files_verified_after_reproduction':seals,'code':source_fingerprint(root),
            'snapshot_files':len(paths),'snapshot_sha256':file_hash(archive),
            'no_inference_rerun_for_finalization':True,'novelty':'NOT_ESTABLISHED','confirmation':'CONFIRMATION_NOT_LOCKED'}
    write_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'environment_and_baseline','code'}},indent=2))


if __name__=='__main__':
    main()
