"""Seal the metadata-only fallback selection after a blocked matched-source audit."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from src.lava.jaes_v2.evidence import file_hash,read_csv,write_json
from src.lava.jaes_v2.phase4 import NuisanceConfig


def main() -> None:
    selection_path=Path('configs/jaes_v2/phase4_nuisance_selection_v1.json')
    config_path=Path('configs/jaes_v2/phase4_nuisance_dev_v1.json')
    if selection_path.exists() or config_path.exists():
        raise FileExistsError('New selection/config identity required')
    cache=Path('outputs/jaes_v2/c1_mlaad_tiny_dev_v1_waveforms')
    matching=Path('outputs/jaes_v2/audit/phase4_source_matching/matching_coverage.json')
    audit=json.loads(matching.read_text())
    if audit['status']!='SOURCE_MATCHED_EXPERIMENT_BLOCKED':
        raise ValueError('Fallback requires explicit source-matching decision')
    records=read_csv(cache/'manifest.csv'); rng=np.random.default_rng(4804)
    cells=defaultdict(list)
    for row in records:
        if int(row['label'])==1:
            cells[(row['generator_id'],row['source_speaker_id'])].append(row)
    chosen=[]
    for key,members in sorted(cells.items()):
        members=sorted(members,key=lambda row:row['sample_id'])
        chosen.append(members[int(rng.integers(len(members)))])
    real=sorted((r for r in records if int(r['label'])==0),key=lambda r:r['sample_id'])
    chosen.extend(real[i] for i in sorted(rng.choice(len(real),16,replace=False)))
    selection={'scope':'PUBLIC_DEVELOPMENT','matching_status':audit['status'],'seed':4804,
               'selection_policy':'One seeded recording per annotated system/source-text-speaker cell, plus 16 English REAL; no score input',
               'records':sorted(chosen,key=lambda row:row['sample_id'])}
    write_json(selection_path,selection)
    config=NuisanceConfig(experiment_id='c1_nuisance_controls_dev_v1',selection_path=selection_path.as_posix(),
        selection_sha256=file_hash(selection_path),phase3_cache=cache.as_posix(),
        phase3_cache_metadata_sha256=file_hash(cache/'cache_metadata.json'),
        source_matching_report=matching.as_posix(),source_matching_sha256=file_hash(matching))
    write_json(config_path,asdict(config)); print(json.dumps(asdict(config),indent=2))


if __name__=='__main__':
    main()
