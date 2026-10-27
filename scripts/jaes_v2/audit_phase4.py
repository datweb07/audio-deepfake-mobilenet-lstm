"""Reproduce source matching, coverage figures and a metadata-only confirmation draft."""
import argparse
from collections import Counter
import json
from pathlib import Path

from src.lava.jaes_v2.evidence import read_csv,write_json
from src.lava.jaes_v2.source_matching import audit_matching
from src.lava.jaes_v2.confirmation_gate import assess_confirmation


def figures(directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='source-matching-phase4'
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads((directory/'matching_coverage.json').read_text())
    output=directory/'figures'; output.mkdir(exist_ok=True)
    pairs=report['phase3_pairwise_overlap']; groups=sorted({r['G0'] for r in pairs}|{r['G1'] for r in pairs})
    matrix=np.zeros((len(groups),len(groups)),dtype=int)
    for row in pairs:
        a,b=groups.index(row['G0']),groups.index(row['G1']); matrix[a,b]=matrix[b,a]=row['source_overlap']
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    multiplicity=report['same_language_source_multiplicity']
    axes[0].bar(list(multiplicity),list(multiplicity.values()))
    axes[0].set(yscale='log',xlabel='Systems sharing an annotated source (same language)',ylabel='Source count, log scale')
    for i,(key,value) in enumerate(multiplicity.items()):
        axes[0].text(i,value,str(value),ha='center',va='bottom')
    axes[1].imshow(matrix,vmin=0,vmax=1,cmap='Blues')
    for i in range(len(groups)):
        for j in range(len(groups)):
            axes[1].text(j,i,'—' if i==j else str(matrix[i,j]),ha='center',va='center')
    axes[1].set(xticks=range(len(groups)),yticks=range(len(groups)),xticklabels=groups,yticklabels=groups,title='Phase 3 selected source-ID overlaps')
    axes[1].tick_params(axis='x',rotation=60,labelsize=8)
    fig.suptitle('METADATA AUDIT — selected systems share speakers, not source recordings')
    fig.savefig(output/'source_matching_coverage.png',dpi=300)
    fig.savefig(output/'source_matching_coverage.svg',metadata={'Date':None}); plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--existing',type=Path)
    parser.add_argument('--discovery',type=Path,default=Path('outputs/jaes_v2/datasets/mlaad_tiny_metadata_v1'))
    parser.add_argument('--manifest',type=Path,default=Path('outputs/jaes_v2/datasets/mlaad_tiny_dev_v1/provenance_manifest.csv'))
    args=parser.parse_args()
    if args.output:
        audit_matching(args.discovery,args.manifest,args.output); directory=args.output
    elif args.existing:
        directory=args.existing
    else:
        parser.error('--output NEW_DIRECTORY or --existing MATCHING_AUDIT_DIRECTORY required')
    figures(directory)
    draft={'validation_kind':'A_SOURCE_MATCHED_HYPOTHESIS','confirmation_labels_opened':False,
           'primary_hypothesis':'Prespecified source-matched order contrast differs across documented systems',
           'primary_estimand':'Mean source-matched difference of paired AB-BA FAKE misses, equal independent source-group weight',
           'matching_rule':'author-full-source-id-and-text-1','matching_level':3,
           'threshold_policy':'Inherited artifacts; any new calibration separate from confirmation',
           'conditions':['clean','fixed_noise_10dB','opus24','noise_then_codec','codec_then_noise'],
           'multiplicity_policy':'One selected development-informed primary contrast; Holm for prespecified secondary inferential family',
           'success_failure_criteria':'Prespecified practical effect and uncertainty; retain null/reversal/confounder explanation',
           'sample_ids':[],'verified_independent_clusters':0}
    write_json(directory/'confirmation_draft.json',draft)
    write_json(directory/'confirmation_readiness.json',assess_confirmation(draft,development_ids={r['sample_id'] for r in read_csv(args.manifest)}))
    print(json.dumps(assess_confirmation(draft,development_ids=set()),indent=2))


if __name__=='__main__':
    main()
