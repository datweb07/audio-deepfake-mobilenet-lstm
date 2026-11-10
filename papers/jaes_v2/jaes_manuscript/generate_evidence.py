"""Generate manuscript tables and vector figures from audited development evidence."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = 'outputs/jaes_v2/phase10/empirical_execution_v1/post9207_consolidated_v1.json'
EVIDENCE = ROOT/REL
METHODS = ('erm','channel')
ENDPOINTS = {
    'A_clean_FA':'A clean REAL false alarms', 'A_clean_miss':'A clean FAKE misses',
    'A_stress_FA':'A stressed REAL false alarms', 'A_stress_miss':'A stressed FAKE misses',
    'B_clean_FA':'B clean REAL false alarms',
}
ATTACKS = {f'A{i:02}_stress_miss':f'A{i:02} stressed FAKE misses' for i in range(1,7)}


def save_table(name, label, caption, columns, rows):
    content = ['\\begin{table*}[t]',f'\\caption{{{caption}}}\\label{{tab:{label}}}',
               '\\centering', '\\begin{tabular}{@{}l'+'r'*(len(columns)-1)+'@{}}',
               '\\toprule',' & '.join(columns)+r' \\', '\\colrule']
    content.extend(' & '.join(map(str,row))+r' \\' for row in rows)
    content.extend(['\\botrule','\\end{tabular}','\\end{table*}'])
    (HERE/'tables'/f'{name}.tex').write_text('\n'.join(content)+'\n',encoding='utf-8')


def error_count(data,key):
    value = data['miss' if 'miss' in key else 'false_alarm']
    return f"{value['errors']}/{value['n']}"


def run():
    evidence = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    seeds = evidence['seeds']
    for directory in ('tables','figures'):
        (HERE/directory).mkdir(exist_ok=True)
    endpoints = [s['endpoints'][m] for s in seeds.values() for m in METHODS]
    columns = ('Development endpoint','9107 ERM','9107 Channel','9207 ERM','9207 Channel')
    rows = [
        ('Selected update',*(int(e['selected_checkpoint'][1:7]) for e in endpoints)),
        ('Clean A ROC-AUC',*(f"{e['clean']['auc']:.4f}" for e in endpoints)),
        ('Clean A EER (percent)',*(f"{100*e['clean']['eer']:.2f}" for e in endpoints)),
        ('A macro stressed ROC-AUC',*(f"{e['macro_stress_auc']:.4f}" for e in endpoints)),
        *((label,*(error_count(s['paired'][k][m],k) for s in seeds.values() for m in METHODS))
          for k,label in ENDPOINTS.items()),
        ('Optimizer minutes',*(f"{e['train_seconds']/60:.1f}" for e in endpoints)),
        ('Validation minutes',*(f"{e['validation_seconds']/60:.1f}" for e in endpoints)),
    ]
    save_table('development','development','Development results. Stress counts use repeated views. Timing is from CPU logs.',columns,rows)
    save_table('attack','attack','Attack-specific stressed FAKE misses over Noise5 and both ordered chains.',
               columns,[(label,*(error_count(s['paired'][k][m],k) for s in seeds.values() for m in METHODS))
                        for k,label in ATTACKS.items()])
    rows = []
    for key,label in {'macro_stress_auc':'A macro stress ROC-AUC',**ENDPOINTS,
                      **{k:ATTACKS[k] for k in ('A05_stress_miss','A06_stress_miss')}}.items():
        auc = key=='macro_stress_auc'
        scale = 1 if auc else 100
        fmt = lambda x: f'{x*scale:+.4f}' if auc else f'{x*scale:+.2f}'
        seed_values = []
        for s in seeds.values():
            interval = s['macro_stress_auc_interval'] if auc else s['paired'][key]['bootstrap']
            lo,hi = interval['percentile95']
            seed_values.append(f"{fmt(s['deltas'][key])} [{fmt(lo)}, {fmt(hi)}]")
        mean,limits = (evidence['summary'][key][field] for field in ('mean','range'))
        rows.append((label,*seed_values,fmt(mean),f'{fmt(limits[0])} to {fmt(limits[1])}'))
    save_table('uncertainty','uncertainty','Channel minus ERM. Brackets give within-seed 95 percent speaker-cluster intervals. Error differences are percentage points.',
               ('Endpoint','9107 [interval]','9207 [interval]','Mean','Range'),rows)
    rows = []
    for seed, attacks in evidence['attack_details'].items():
        for attack,a in attacks.items():
            left,right = (a['misses'][m] for m in METHODS)
            cells = [f'{left[c]} / {right[c]}' for c in ('clean','noise5','A_then_B','B_then_A')]
            all_missed = a['all_stress_missed']
            rows.append((f'{seed} / {attack}',*cells,f"{all_missed['erm']} / {all_missed['channel']}",
                         f"{a['harmed_sources']} / {a['recovered_sources']}"))
    save_table('attack_conditions','attackconditions','A05 and A06 source-level diagnostics. Each condition has 18 sources. Error cells are ERM / channel. Harm and recovery can overlap across views.',
               ('Seed / attack','Clean','Noise5','Noise to codec','Codec to noise','All stress','Harmed / recovered'),rows)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'axes.labelsize':10})
    colors = ('#17608a','#a94d11')
    figure, axes = plt.subplots(1,2,figsize=(6.7,2.8),layout='constrained')
    for ax,key,title in zip(axes,('macro_stress_auc','A_stress_miss'),
                            ('Macro stress ROC-AUC','A stressed FAKE misses (percent)')):
        for j,(seed,s) in enumerate(seeds.items()):
            values = ([s['endpoints'][m][key] for m in METHODS] if key=='macro_stress_auc' else
                      [100*s['paired'][key][m]['miss']['errors']/s['paired'][key][m]['miss']['n'] for m in METHODS])
            ax.plot((0,1),values,'o-' if j==0 else 's--',color=colors[j],label=seed)
        ax.set(xticks=(0,1),xticklabels=('ERM','Channel'),ylabel=title,xlim=(-.18,1.18))
        ax.legend(title='Seed',frameon=False)
    figure.savefig(HERE/'figures'/'aggregate.pdf')
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(5.9,3.1),layout='constrained')
    xs = np.arange(len(ATTACKS))
    for j,(seed,s) in enumerate(seeds.items()):
        effects = [100*s['deltas'][key] for key in ATTACKS]
        bounds = np.array([s['paired'][key]['bootstrap']['percentile95'] for key in ATTACKS])*100
        center = xs+(j-.5)*.22
        ax.vlines(center,bounds[:,0],bounds[:,1],color=colors[j])
        ax.scatter(center,effects,label=seed,marker=('o','s')[j],color=colors[j],zorder=3)
    ax.axhline(0,color='.4',linewidth=.8)
    ax.set(xticks=xs,xticklabels=[key[:3] for key in ATTACKS],
           xlabel='Training-represented attack ID',ylabel='Change in stressed FAKE misses (pp)')
    ax.legend(title='Seed',frameon=False)
    figure.savefig(HERE/'figures'/'attack_changes.pdf')
    plt.close(figure)
    figure, axes = plt.subplots(1,2,figsize=(6.7,3.15),sharey=True,layout='constrained')
    for ax,(seed,s) in zip(axes,seeds.items()):
        by_model = [s['endpoints'][m]['speaker_errors'] for m in METHODS]
        keys = sorted(key for key in by_model[0] if key.startswith('B_REAL_ONLY/') and key.endswith('/clean'))
        for j,(m,data) in enumerate(zip(METHODS,by_model)):
            counts = [data[k]['false_alarm']['errors'] for k in keys]
            ax.barh(np.arange(len(keys))+(j-.5)*.36,counts,height=.34,color=colors[j],label=m.upper())
        ax.set(title=f'Seed {seed}',xlabel='Clean REAL false alarms (count)',
               yticks=np.arange(len(keys)),yticklabels=[k.split('/')[1].replace('LA_','') for k in keys])
        ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('Development B speaker ID (LA prefix omitted)')
    axes[0].invert_yaxis()
    figure.savefig(HERE/'figures'/'speaker_counts.pdf')
    plt.close(figure)
    figure, axes = plt.subplots(1,2,figsize=(6.7,2.85),sharex=True,sharey=True,layout='constrained')
    groups = ('A_PAIRED/0','A_PAIRED/1','B_REAL_ONLY/0')
    for ax,(seed,s) in zip(axes,seeds.items()):
        for j,method in enumerate(METHODS):
            summaries = [s['endpoints'][method]['order'][group]['scores'] for group in groups]
            y = np.arange(len(groups))+(j-.5)*.25
            q = np.array([row['quantiles'] for row in summaries])
            ax.hlines(y,q[:,1],q[:,5],color=colors[j],linewidth=1)
            ax.hlines(y,q[:,2],q[:,4],color=colors[j],linewidth=3)
            ax.scatter(q[:,3],y,marker=('o','s')[j],color=colors[j],label=method.upper(),zorder=3)
        ax.axvline(0,color='.5',linewidth=.7)
        ax.set(title=f'Seed {seed}',xlabel='Noise-to-codec minus codec-to-noise score',
               yticks=np.arange(len(groups)),yticklabels=('A REAL','A FAKE','B REAL'))
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False,fontsize=8)
    figure.savefig(HERE/'figures'/'order_scores.pdf')
    plt.close(figure)

    claims = [
        ('Methods','Four matched runs, 840 updates, 21 epochs and 2,576 predictions each','four run seals','development','9107/9207','both','budget and score grid',REL,'VERIFIED_DEVELOPMENT','shared cohort'),
        ('Results','Stress macro AUC improves in both paired blocks','four selected checkpoints','development A','9107/9207','both','macro stress AUC',REL,'VERIFIED_DEVELOPMENT','two initializations'),
        ('Results','FAKE stress misses decline by 67 and 9 views','four selected checkpoints','development A','9107/9207','both','stress FAKE misses',REL,'VERIFIED_DEVELOPMENT','9207 interval includes zero'),
        ('Results','Clean B false alarms move in opposite directions','four selected checkpoints','development B','9107/9207','both','REAL false alarms',REL,'VERIFIED_DEVELOPMENT','nine speakers'),
        ('Results','A05 and A06 misses rise in both blocks','four selected checkpoints','development A','9107/9207','both','attack stress FAKE misses',REL,'VERIFIED_DEVELOPMENT','18 sources per attack'),
        ('Results','Order-only errors differ by class','paired channel views','development A/B','9107/9207','both','decision transitions',REL,'VERIFIED_DEVELOPMENT','fixed threshold and chains'),
        ('Results','Nuisance screen leaves directional AUC at 0.538 and 0.432','matched components','development','9107','both','order contrast','outputs/jaes_v2/phase10/final_diagnostic_v1/scientific_evidence.json','VERIFIED_DEVELOPMENT','retrospective one seed'),
        ('Results','Timing from logged optimizer and validation work','four run resource logs','development','9107/9207','both','seconds',REL,'VERIFIED_DEVELOPMENT','includes replay where logged'),
        ('Introduction','Conference benchmark uses six heterogeneous detector artifacts','conference manuscript','historical','historical','six','benchmark scope','papers/PaperID_804_final.docx','VERIFIED_HISTORICAL','venue and rights pending'),
        ('Limitations','No verified independent forensic target','matched components','development','9107','both','mechanism','','NOT_ESTABLISHED','classifier score is a proxy'),
        ('Limitations','Official held-out evidence is pending','prelock','evaluation','9107/9207','both','none','outputs/jaes_v2/phase10/empirical_execution_v1/heldout_prelock_9107_9207_v1.json','HELDOUT_PENDING','separate author approvals'),
    ]
    with (HERE/'claim_evidence_map.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('section','claim','experiment','partition','seed','model','metric','artifact','status','limitation'))
        writer.writerows(claims)


if __name__ == '__main__':
    run()
