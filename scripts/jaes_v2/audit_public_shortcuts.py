"""Compare original-recording and actual detector-input metadata shortcuts."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.evidence import read_csv, write_csv, write_json, file_hash
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.shortcut import grouped_shortcuts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    metadata=json.loads((args.cache/'cache_metadata.json').read_text())
    verify_seal(args.cache,metadata)
    manifest=read_csv(args.cache/'manifest.csv')
    original=read_csv(args.cache/'source_features.csv')
    prepared=[]
    for row in read_csv(args.cache/'conditions.csv'):
        if row['condition_id'] != 'clean':
            continue
        values,rate=sf.read(args.cache/row['path'],dtype='float32',always_2d=True)
        mono=values.mean(axis=1)
        prepared.append({'sample_id':row['sample_id'],'duration':len(mono)/rate,'sample_rate':rate,
                         'rms':float(np.sqrt(np.mean(mono.astype('float64')**2))),'peak':float(np.max(np.abs(mono))),
                         'silence_fraction_below_1e-4':float(np.mean(np.abs(mono)<1e-4))})
    result={'cache_metadata_sha256':file_hash(args.cache/'cache_metadata.json'),
            'original':grouped_shortcuts(original,manifest),'detector_input':grouped_shortcuts(prepared,manifest),
            'note':'Original Phase 2 LOPO sample-rate accuracy may be zero with constant inputs due to majority inversion. Constant features here are explicitly noninformative.'}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'shortcut_report.json',result)
    write_csv(args.output/'detector_input_features.csv',prepared)
    table=[{'input':key,**row} for key in ('original','detector_input') for row in result[key]['results']]
    write_csv(args.output/'shortcuts.csv',table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    valid=[row for row in table if row['accuracy'] is not None]
    ax.barh([row['input']+'/'+row['baseline'] for row in valid],[row['accuracy'] for row in valid])
    ax.set(xlim=(0,1),xlabel='Held-out grouped development accuracy',title='EXPLORATORY PUBLIC DEVELOPMENT — metadata shortcuts\nKnown source components grouped; REAL lineage UNKNOWN')
    fig.savefig(args.output/'shortcuts.png',dpi=300)
    fig.savefig(args.output/'shortcuts.svg',metadata={'Date':None})
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
