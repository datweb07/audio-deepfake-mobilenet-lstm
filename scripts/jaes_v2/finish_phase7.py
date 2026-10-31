"""Final immutable-history, actual regression, run seals and model-free reproduction audit."""
from pathlib import Path
import json,shutil,subprocess,zipfile
from src.lava.jaes_v2.evidence import file_hash,write_json,read_csv
from src.lava.jaes_v2.runtime import preflight,source_fingerprint
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.phase7_data import BASE

def main():
    root=Path(__file__).resolve().parents[2];b=root/BASE;out=root/'outputs/jaes_v2/audit/phase7_final_integrity.json'
    if out.exists():raise FileExistsError(out)
    initial=json.loads((root/'outputs/jaes_v2/audit/phase7_before.json').read_text());checks={}
    changed=[name for name,h in initial['historical_files'].items() if not (root/name).is_file() or file_hash(root/name)!=h]
    checks['Phase1_6_history']='FAIL' if changed else 'PASS'
    if changed:raise ValueError('Historical change: '+str(changed[:8]))
    environment=preflight(root);checks['protected_baseline']='PASS'
    regression=json.loads((root/'outputs/jaes_v2/audit/phase7_validation_final/validation.json').read_text());checks['regression']='PASS' if all(x['successful'] for x in regression['suites'].values()) else 'FAIL'
    if checks['regression']!='PASS':raise ValueError('Regression failed')
    sealed={};run_hashes={};checkpoint_hashes={}
    for path in sorted((b/'training').glob('*/run_metadata.json')):
        m=json.loads(path.read_text());name=path.parent.name
        if m['status']!='PILOT_EVALUATED':raise ValueError('Incomplete registered training run: '+name)
        sealed[name]=verify_seal(path.parent,m);run_hashes[name]=file_hash(path);checkpoint_hashes[name]=file_hash(path.parent/'best.keras')
    if len(sealed)!=17:raise ValueError('Expected15head+2matchedrepresentation runs; do not silently promote incomplete matrix')
    head=[json.loads((b/'training'/name/'run_metadata.json').read_text()) for name in sealed if name.startswith('p7_h')]
    blocks={seed:{m['initial_model_weights_sha256'] for m in head if m['config']['training_seed']==seed} for seed in (6107,6207,6307)}
    if any(len(v)!=1 for v in blocks.values()) or len(set.union(*blocks.values()))!=3 or len({m['code']['hash'] for m in head})!=1:raise ValueError('Seed independence/source parity failed')
    for seed in blocks:
        if file_hash(b/'training'/f'p7_h2_s{seed}_v1/training_views.csv')!=file_hash(b/'training'/f'p7_h3_s{seed}_v1/training_views.csv'):raise ValueError('Paired supervised exposure differs')
    a=json.loads((b/'training/p7_f0_s6107_v1/run_metadata.json').read_text());c=json.loads((b/'training/p7_f1_s6107_v1/run_metadata.json').read_text())
    if a['initial_named_weights_sha256']!=c['initial_named_weights_sha256'] or a['code']['hash']!=c['code']['hash'] or file_hash(b/'training/p7_f0_s6107_v1/training_views.csv')!=file_hash(b/'training/p7_f1_s6107_v1/training_views.csv'):raise ValueError('Representation pilot initialization/source/view parity failed')
    checks['training_seed_independence']='PASS';checks['equal_view_fairness']='PASS';checks['representation_initialization_and_exposure']='PASS'
    for label in ('medium_p0_v1','benchmark_p0_v1'):
        cache=b/'caches'/label;sealed[label]=verify_seal(cache,json.loads((cache/'cache_metadata.json').read_text()))
    for path in (b/'preflight').glob('*/run_metadata.json'):
        metadata=json.loads(path.read_text())
        if metadata['status']!='SMOKE_TESTED':raise ValueError('Incomplete resource preflight')
        sealed[path.parent.name+'_resource_preflight']=verify_seal(path.parent,metadata)
    verify_seal(b/'diagnostics_v1',json.loads((b/'diagnostics_v1/seal.json').read_text()))
    for first,second in [('statistics','statistics_reproduced')]:
        a=b/first;c=b/second
        files=[p for p in a.rglob('*') if p.is_file()]
        differences=[p.relative_to(a).as_posix() for p in files if not (c/p.relative_to(a)).is_file() or file_hash(p)!=file_hash(c/p.relative_to(a))]
        if differences:raise ValueError('Model-free reproduction mismatch: '+str(differences[:6]))
    m1=json.loads((b/'master_final/validation.json').read_text());m2=json.loads((b/'master_reproduced/validation.json').read_text())
    for key in ('markdown_sha256','inventory_sha256'):
        if m1[key]!=m2[key]:raise ValueError('Master report not byte-identical')
    docs=root/'docs/jaes_v2'
    if file_hash(docs/'LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE7.md')!=m1['markdown_sha256'] or file_hash(docs/'LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE7.json')!=m1['inventory_sha256']:raise ValueError('Published report differs from generator')
    checks['new_run_seals']='PASS';checks['model_free_reproduction']='PASS';checks['master_reproduction']='PASS'
    source=[]
    for folder in ('src/lava/jaes_v2','scripts/jaes_v2','configs/jaes_v2','tests/jaes_v2','docs/jaes_v2','papers/jaes_v2'):
        source.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    fingerprint=source_fingerprint(root)
    source=sorted(set(source)|{root/name for name in fingerprint['files']})
    snapshot=root/'outputs/jaes_v2/audit/phase7_source_snapshot.zip'
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in source:z.write(p,p.relative_to(root).as_posix())
    executed_bundle=root/'outputs/jaes_v2/audit/phase7_training_source_v1.zip'
    executed_meta=json.loads(executed_bundle.with_suffix('.json').read_text())
    if file_hash(executed_bundle)!=executed_meta['sha256']:raise ValueError('Executed head source bundle changed')
    artifact_hashes={p.relative_to(root).as_posix():file_hash(p) for p in b.rglob('*') if p.is_file()}
    write_json(root/'outputs/jaes_v2/audit/phase7_all_artifact_hashes.json',artifact_hashes)
    result=dict(status='PASS',audit_scope='Historical integrity, implementation/regression, seals and reproduction; not a claim all research gates passed',checks=checks,
        research_gates=dict(A_DATA='PASS',B_CACHE='PASS',C_BASELINE_FAIRNESS='PASS',D_MULTI_SEED='PASS',E_REAL_BACKBONE_GRADIENT_AND_BUDGET='PASS',F_METHOD_NOVELTY='FAIL_NOT_ESTABLISHED',G_EXTERNAL_CONFIRMATION='NOT_LOCKED'),
        not_measured=dict(activation_memory='NOT_ISOLATED; included in measured OS lifetime peak',GPU_timing='NOT_MEASURED',deployment_risk_guarantees='NOT_ESTABLISHED'),
        not_run=dict(independent_confirmation='NOT_RUN:NOT_LOCKED',reserved_calibration='NOT_RUN:NOT_LOCKED',GPU_training='NOT_RUN',online_RawBoost_quality='NOT_RUN:AWAITING_COMPUTE',novel_weighted_loss='NOT_RUN:FAILED_NOVELTY_GATE',three_seed_finetuning='NOT_RUN:boundedone-seedpilotonly'),
        accepted_manuscript_sha256=environment['accepted_paper_sha256'],protected_baseline_verification=environment['baseline_freeze_verification'],
        historical_files_unchanged=len(initial['historical_files']),phase6_historical_reference_count=initial['phase6_historical_verified'],new_run_seals=sealed,new_run_hashes=run_hashes,checkpoint_hashes=checkpoint_hashes,
        manifest_sha256=file_hash(b/'selections/medium_all_attack_v1/manifest.csv'),source_fingerprint=fingerprint,source_files=len(source),source_snapshot_sha256=file_hash(snapshot),executed_head_source_snapshot_sha256=file_hash(executed_bundle),
        all_artifact_hashes_sha256=file_hash(root/'outputs/jaes_v2/audit/phase7_all_artifact_hashes.json'),regression={k:{a:v for a,v in x.items() if a!='failure_details'} for k,x in regression['suites'].items()},
        master_report_sha256=m1['markdown_sha256'],research_inventory_sha256=m1['inventory_sha256'],model_free_reproduction='BYTE_IDENTICAL_STATISTICS_SVG_PNG_MASTER_JSON',
        remaining_D_bytes=shutil.disk_usage(root).free,remaining_C_bytes=shutil.disk_usage('C:/').free,calibration_accessed=False,evaluation_accessed=False,
        C1='SUPPORTING_VALIDATION_ENGINEERING',C2='MULTI_SEED_STANDARD_BASELINES; NOVELTY_NOT_ESTABLISHED',C3='DEFERRED',confirmation='NOT_LOCKED')
    if result['remaining_D_bytes']<15*1024**3:raise ValueError('Disk reserve violated')
    write_json(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_fingerprint','new_run_hashes','checkpoint_hashes')},indent=2))
if __name__=='__main__':main()
