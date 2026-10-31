"""Read-only historical verification and an immutable pre-Phase7 snapshot."""
from pathlib import Path
import json, os, shutil, subprocess, tempfile
from src.lava.jaes_v2.evidence import file_hash, write_json
from src.lava.jaes_v2.runtime import preflight

def main():
    root=Path(__file__).resolve().parents[2]
    destination=root/'outputs/jaes_v2/audit/phase7_before.json'
    if destination.exists(): raise FileExistsError(destination)
    environment=preflight(root)
    old=json.loads((root/'outputs/jaes_v2/audit/phase6_before.json').read_text())
    for name,digest in old['historical_files'].items():
        if file_hash(root/name)!=digest: raise ValueError('Historical change: '+name)
    final=json.loads((root/'outputs/jaes_v2/audit/phase6_final_integrity.json').read_text())
    for name,key in [('LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT.md','master_markdown_sha256'),('LAVA_JOURNAL_V2_RESEARCH_INVENTORY.json','inventory_sha256')]:
        if file_hash(root/'docs/jaes_v2'/name)!=final[key]: raise ValueError('Phase6 handoff changed')
    paths=[p for p in (root/'outputs/jaes_v2').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for folder in ('docs/jaes_v2','papers/jaes_v2','src/lava/jaes_v2','scripts/jaes_v2','configs/jaes_v2','tests/jaes_v2') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'phase7' not in p.name.lower()]
    write_json(destination,dict(status='PASS',environment=environment,phase6_historical_verified=len(old['historical_files']),
        historical_files={p.relative_to(root).as_posix():file_hash(p) for p in sorted(set(paths))},
        disk={volume:shutil.disk_usage(volume).free for volume in ('C:/','D:/')},
        temp_before=dict(TEMP=os.getenv('TEMP'),TMP=os.getenv('TMP'),python_temp=tempfile.gettempdir()),
        git_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True)))
    print('Historical verification PASS; captured',len(set(paths)),'files')
if __name__=='__main__': main()
