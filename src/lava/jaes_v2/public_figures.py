"""Figures derived only from executed public-development scores and provenance."""
from pathlib import Path

import numpy as np

from .evidence import read_csv, write_csv, write_json


def make_figures(run: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt':'lava-public-dev-v1','font.size':10})
    import matplotlib.pyplot as plt
    output = run/'figures'
    output.mkdir(exist_ok=True)
    def save(figure,name):
        figure.suptitle('EXPLORATORY PUBLIC DEVELOPMENT — '+run.name.split('_')[-2],fontsize=11)
        figure.savefig(output/(name+'.png'),dpi=300,bbox_inches='tight')
        figure.savefig(output/(name+'.svg'),metadata={'Date':None},bbox_inches='tight')
        plt.close(figure)
    manifest = read_csv(run/'manifest.csv')
    groups = sorted({row['generator_id'] for row in manifest if row['label']=='1'})
    fig,ax = plt.subplots(figsize=(11,3),layout='constrained')
    ax.axis('off')
    real_count = sum(row['label']=='0' for row in manifest)
    fake_count = len(manifest)-real_count
    languages = ','.join(sorted({row['language'] for row in manifest}))
    boxes = ['Pinned official\nMLAAD-tiny revision','Author annotations\n+ dual license notices',
             'SHA-256 + PCM +\noverlap / confounding audit',f"{len(manifest)} {languages} development\n{real_count} REAL + {fake_count} FAKE"]
    for index,text in enumerate(boxes):
        x=.12+index*.25
        ax.text(x,.5,text,ha='center',va='center',transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.7',facecolor='#edf4fa',edgecolor='#2166ac'),fontsize=9)
        if index<3:
            ax.annotate('',xy=(x+.15,.5),xytext=(x+.10,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.set_title('Verified acquisition flow; original REAL lineage remains UNKNOWN')
    save(fig,'dataset_provenance')
    fig,ax = plt.subplots(figsize=(9,4),layout='constrained')
    ax.barh(['REAL']+groups,[sum(row['label']=='0' for row in manifest)]+[sum(row['generator_id']==g for row in manifest) for g in groups])
    ax.set(xlabel='Selected English recordings',title='Author-annotated system coverage; REAL identity UNKNOWN')
    save(fig,'group_coverage')
    table = report['metric_table']
    conditions = [row['condition'] for row in table if row['group']=='ALL']
    matrix = np.array([[next(row['miss_rate'] for row in table if row['group']==g and row['condition']==c) for c in conditions] for g in groups])
    fig,ax = plt.subplots(figsize=(11,4),layout='constrained')
    image = ax.imshow(matrix,vmin=0,vmax=1,cmap='magma',aspect='auto')
    ax.set(yticks=range(len(groups)),yticklabels=groups,xticks=range(len(conditions)),xticklabels=conditions,title='FAKE miss fraction at inherited threshold')
    ax.tick_params(axis='x',labelrotation=35)
    fig.colorbar(image,ax=ax,label='Miss fraction')
    save(fig,'group_channel_heatmap')
    effects = [row for row in report['order_contrasts'] if row.get('risk')=='error' and row['cluster_field']=='source_speaker_id']
    if effects:
        fig,ax = plt.subplots(figsize=(10,4),layout='constrained')
        for position,row in enumerate(effects):
            value = row['estimate']; low,high = row['ci95'] or [value,value]
            ax.errorbar(value,position,xerr=[[max(0,value-low)],[max(0,high-value)]],fmt='o',capsize=4)
        ax.axvline(0,color='gray',linestyle='--')
        ax.set(yticks=range(len(effects)),yticklabels=[row['group'] for row in effects],xlabel='Noise→codec minus codec→noise miss difference',
               title='Descriptive source-text-speaker bootstrap 95% intervals')
        save(fig,'order_effects')
    overall = [row for row in table if row['group']=='ALL']
    fig,ax = plt.subplots(figsize=(10,4),layout='constrained')
    for key in ('error_rate','false_alarm_rate','miss_rate','auc','eer'):
        ax.plot(conditions,[row[key] for row in overall],marker='o',label=key)
    ax.tick_params(axis='x',labelrotation=35)
    ax.set(ylabel='Fraction / ROC-AUC',ylim=(-.03,1.03),title='Frozen operating point and diagnostic ranking; EER is not a new threshold')
    ax.legend(ncol=3)
    save(fig,'threshold_vs_ranking')
    (run/'tables').mkdir(exist_ok=True)
    write_csv(run/'tables/clean_channel_results.csv',table)
    write_json(run/'tables/interaction_estimates.json',report['generator_channel_interactions'])
    write_json(run/'tables/order_effects.json',report['order_contrasts'])
