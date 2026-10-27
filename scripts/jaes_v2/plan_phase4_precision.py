"""Replan FAKE paired nuisance precision from sealed development scores only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from src.lava.jaes_v2.analysis import loss
from src.lava.jaes_v2.evidence import read_csv,write_json,file_hash
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.phase4_analysis import summarize
from src.lava.jaes_v2.precision import plan_precision
from src.lava.jaes_v2.nuisance_controls import POLICIES


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',choices=POLICIES,default='edge_repack_v1')
    parser.add_argument('--target-effect',type=float,default=.10)
    parser.add_argument('--half-width',type=float,default=.05)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New planning report filename required')
    metadata=json.loads((args.run/'run_metadata.json').read_text())
    if metadata['status']!='WITHIN_RECORD_DEVELOPMENT_EVALUATED' or metadata['scope']!='EXPLORATORY_WITHIN_RECORD_DEVELOPMENT':
        raise ValueError('Only sealed development nuisance runs can inform this planner')
    verify_seal(args.run,metadata)
    rows=read_csv(args.run/'scores.csv'); manifest=read_csv(args.run/'manifest.csv')
    summarize(rows,manifest,replicates=100)  # validate complete paired score/label/threshold contracts
    index={(r['sample_id'],r['policy'],r['condition_id']):r for r in rows}
    groups=defaultdict(list)
    for source in manifest:
        if int(source['label'])!=1:
            continue
        def order(policy):
            return loss(index[(source['sample_id'],policy,'fixed_A_then_B')],'error')-loss(index[(source['sample_id'],policy,'fixed_B_then_A')],'error')
        groups[source['source_speaker_id']].append(order(args.policy)-order('phase3_baseline'))
    result={'run_metadata_sha256':file_hash(args.run/'run_metadata.json'),'policy':args.policy,
            'confirmation_data_read':False,**plan_precision(groups,target_effect=args.target_effect,half_width=args.half_width)}
    write_json(args.output,result); print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
