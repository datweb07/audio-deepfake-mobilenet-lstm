"""Final Phase6 source bundle, historical integrity and completed research evidence checks."""
from pathlib import Path
import json
import shutil
import subprocess
import zipfile
from src.lava.jaes_v2.evidence import file_hash,read_csv,write_csv,write_json
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.runtime import preflight,source_fingerprint
from src.lava.jaes_v2.phase6_data import BASE


def main():
    root=Path(__file__).resolve().parents[2]; output=root/'outputs/jaes_v2/audit/phase6_final_integrity.json'
    if output.exists(): raise FileExistsError(output)
    before=json.loads((root/'outputs/jaes_v2/audit/phase6_before.json').read_text())
    changed=[p for p,h in before['historical_files'].items() if file_hash(root/p)!=h]
    if changed: raise ValueError('Historical output changed '+str(changed[:5]))
    validation=json.loads((root/'outputs/jaes_v2/audit/phase6_validation_final_v2/validation.json').read_text())
    if not all(r['successful'] for r in validation['suites'].values()): raise ValueError('Regression failed')
    verified={}
    for run in sorted((root/BASE/'training').iterdir()):
        if not run.is_dir(): continue
        m=json.loads((run/'run_metadata.json').read_text())
        if m['status']!='PILOT_EVALUATED' or m['optimizer_steps']!=60: raise ValueError('Incomplete baseline')
        verified[run.name]=verify_seal(run,m)
    if len(verified)!=9: raise ValueError('Expected registered baseline matrix incomplete')
    for folder,meta in [('feature_cache_v2','cache_metadata.json'),('partial_finetune_sanity_v2','run_metadata.json')]:
        d=root/BASE/folder; verified[folder]=verify_seal(d,json.loads((d/meta).read_text()))
    for first,second in [('reports_v1','reports_reproduced_v1'),('resource_figures_v1','resource_figures_reproduced_v1')]:
        a=root/BASE/first; b=root/BASE/second
        if any(file_hash(p)!=file_hash(b/p.name) for p in a.iterdir() if p.is_file()): raise ValueError('Figure/summary reproduction differs')
    audit=json.loads((root/BASE/'master_audit_v2/validation.json').read_text()); reproduced=json.loads((root/BASE/'master_audit_reproduced_v2/validation.json').read_text())
    if audit['status']!='PASS' or any(audit[k]!=reproduced[k] for k in ('markdown_sha256','inventory_sha256')): raise ValueError('Master report failed reproduction')
    if file_hash(root/'docs/jaes_v2/LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT.md')!=audit['markdown_sha256']: raise ValueError('Master report edited outside generator')
    if file_hash(root/'docs/jaes_v2/LAVA_JOURNAL_V2_RESEARCH_INVENTORY.json')!=audit['inventory_sha256']: raise ValueError('Inventory edited outside generator')
    claims=root/'papers/jaes_v2/CLAIM_EVIDENCE_MAP.csv'; rows=read_csv(claims)
    additions=[dict(claim_id='P601',claim='Speaker-balanced160/56 cohort and9 standard60-update baselines evaluated',scope='CONTROLLED_SINGLE_SEED_DEVELOPMENT',evidence='outputs/jaes_v2/phase6/reports_v1/summary.json',status='PILOT_EVALUATED',limitation='A01-A04 only; unknown source lineage; no independent confirmation'),
        dict(claim_id='P602',claim='Paired standard consistency improves prespecified equal-view development contrast',scope='STANDARD_BASELINE_NOT_NOVEL_METHOD',evidence='outputs/jaes_v2/phase6/reports_v1/summary.json',status='EXPLORATORY_SCREEN',limitation='One seed,7 official speakers, descriptive interval; stressed false alarms remain'),
        dict(claim_id='P603',claim='Partial-backbone gradients, frozen BN/variables and save-load parity verified',scope='TRAIN_ONLY_IMPLEMENTATION_SANITY',evidence='outputs/jaes_v2/phase6/partial_finetune_sanity_v2/run_metadata.json',status='SMOKE_TESTED',limitation='Two train recordings,5 updates; no learned representation-quality claim')]
    for row in additions:
        if row['claim_id'] not in {r['claim_id'] for r in rows}:rows.append(row)
    write_csv(claims,rows)
    paths=[]
    for folder in ('src/lava/jaes_v2','scripts/jaes_v2','configs/jaes_v2','tests/jaes_v2','docs/jaes_v2','papers/jaes_v2'):
        paths += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    snapshot=root/'outputs/jaes_v2/audit/phase6_source_snapshot.zip'
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(paths): stream.write(p,p.relative_to(root).as_posix())
        for p in (root/'outputs/jaes_v2/audit/phase6_prior_art').glob('*executed_v1.py'):stream.write(p,p.relative_to(root).as_posix())
    patch=root/'outputs/jaes_v2/audit/phase6_working_tree.patch';patch.write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=root))
    current=preflight(root)
    if current['free_disk_bytes']<15*1024**3: raise ValueError('Disk reserve violated')
    result=dict(status='PASS',historical_files_unchanged=len(before['historical_files']),protected_baseline=current['baseline_freeze_verification'],
        accepted_manuscript_sha256=current['accepted_paper_sha256'],verified_files=verified,regression={k:{a:b for a,b in v.items() if a!='failure_details'} for k,v in validation['suites'].items()},
        master_report_consistency='PASS',master_markdown_sha256=audit['markdown_sha256'],inventory_sha256=audit['inventory_sha256'],
        model_free_reproduction='BYTE_IDENTICAL_STATISTICS_FIGURES_MASTER_AND_INVENTORY',source_files=len(paths),source_snapshot_sha256=file_hash(snapshot),code=source_fingerprint(root),
        remaining_D_bytes=shutil.disk_usage(root).free,remaining_C_bytes=shutil.disk_usage('C:/').free,calibration_accessed=False,evaluation_accessed=False,
        C1='SUPPORTING_VALIDATION_ENGINEERING',C2='STRONG_BASELINES_PILOT_EVALUATED; NOVELTY_NOT_ESTABLISHED',C3='DEFERRED',confirmation='NOT_LOCKED')
    write_json(output,result); print(json.dumps({k:v for k,v in result.items() if k!='code'},indent=2))


if __name__=='__main__': main()
