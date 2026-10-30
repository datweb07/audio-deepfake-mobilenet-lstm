"""Regenerate fine-tuning and efficiency figures from sealed measurement metadata."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.lava.jaes_v2.evidence import read_csv,write_json,file_hash
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.phase6_data import BASE


def generate(root,output):
    output.mkdir(parents=True,exist_ok=False); base=root/BASE
    f=json.loads((base/'partial_finetune_sanity_v2/run_metadata.json').read_text()); verify_seal(base/'partial_finetune_sanity_v2',f)
    hist=read_csv(base/'partial_finetune_sanity_v2/training_history.csv'); fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot([int(r['step']) for r in hist],[float(r['loss']) for r in hist],'o-')
    axes[0].set(xlabel='Actual optimizer update',ylabel='Two-train-record mean BCE',title='Partial-backbone sanity; not development quality')
    g=f['gradients']; names=list(g); axes[1].bar(range(len(g)),list(g.values()))
    axes[1].set_xticks(range(len(g)),[n.replace('/kernel:0','').replace('/bias:0',' bias') for n in names],rotation=80,ha='right',fontsize=6)
    axes[1].set_title('Actual final-step trainable-variable gradient norms'); fig.tight_layout(); fig.savefig(output/'partial_finetune_loss_gradients.png',dpi=180); plt.close(fig)
    measured=json.loads((base/'efficiency_v1.json').read_text()); fig,axes=plt.subplots(1,2,figsize=(12,4))
    for p in sorted((base/'training').glob('*/run_metadata.json')):
        m=json.loads(p.read_text()); verify_seal(p.parent,m); axes[0].bar(p.parent.name,m['elapsed_seconds'])
    axes[0].set_ylabel('Train + selection/evaluation seconds'); axes[0].tick_params(axis='x',rotation=80,labelsize=7)
    for r in measured['results']: axes[1].bar(r['run'],1000*r['median_seconds']['end_to_end_seconds'])
    axes[1].set_ylabel('Warm decode + preprocessing + full-student ms'); axes[1].tick_params(axis='x',rotation=60,labelsize=7)
    fig.suptitle('Measured CPU costs: head-only training vs full inference, distinct boundaries'); fig.tight_layout(); fig.savefig(output/'training_full_inference_cost.png',dpi=180); plt.close(fig)
    write_json(output/'figure_metadata.json',dict(scope='MODEL_FREE_FIGURES_FROM_EXECUTED_MEASUREMENTS',output_hashes={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    generate(Path(__file__).resolve().parents[2],args.output)
