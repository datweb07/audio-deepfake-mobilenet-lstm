"""Build Phase 3 tables and status documents from completed, executed evidence."""
from collections import Counter
import json
from pathlib import Path

from src.lava.jaes_v2.evidence import read_csv, write_csv, write_json, file_hash


def main():
    root=Path(__file__).resolve().parents[2]
    base=root/'outputs/jaes_v2'
    output=base/'audit/phase3_report_v1'
    if output.exists():
        raise FileExistsError('Research report directory is immutable')
    manifest=read_csv(base/'datasets/mlaad_tiny_dev_v1/provenance_manifest.csv')
    sources=json.loads((base/'datasets/mlaad_tiny_dev_v1/source_metadata.json').read_text())
    shortcuts=json.loads((base/'audit/phase3_shortcuts_v1/shortcut_report.json').read_text())
    config=json.loads((root/'configs/jaes_v2/c1_mlaad_tiny_dev_v1.json').read_text())
    models={}
    metrics=[]
    interactions=[]
    for model in ('mobilenetv3_lstm','shufflenetv2_lstm'):
        run=base/('c1_mlaad_tiny_dev_v1_'+model)
        metadata=json.loads((run/'run_metadata.json').read_text())
        if metadata['status']!='EXTERNAL_DEVELOPMENT_EVALUATED':
            raise ValueError('Report requires completed actual inference: '+model)
        statistics=json.loads((run/'statistics.json').read_text())
        models[model]={'metadata':metadata,'statistics':statistics}
        metrics.extend({'model':model,**row} for row in statistics['metric_table'])
        for row in statistics['generator_channel_interactions']:
            if row['status'].startswith('DESCRIPTIVE_IDENTIFIABLE'):
                estimate=row['contrasts']['generator_channel_interaction']
                interactions.append({'model':model,'G0':row['G0'],'G1':row['G1'],'channel':row['channel'],
                                     'risk':'FAKE miss fraction','estimate':estimate['estimate'],'ci95':json.dumps(estimate['ci95']),
                                     'cluster_field':row['cluster_field'],'cells':json.dumps(row['cells'],sort_keys=True),
                                     'status':row['status']})
            else:
                interactions.append({'model':model,'G0':row['G0'],'G1':row['G1'],'channel':row['channel'],
                                     'risk':'FAKE miss fraction','estimate':None,'ci95':None,'cluster_field':'source_speaker_id',
                                     'cells':None,'status':row['status']})
    output.mkdir(parents=True)
    inventory=[{'dataset':'MLAAD-tiny','revision':sources['revision'],
                'release_real':sources['class_counts']['0'],'release_fake':sources['class_counts']['1'],
                'selected_real':sum(r['label']=='0' for r in manifest),'selected_fake':sum(r['label']=='1' for r in manifest),
                'languages_selected':','.join(sorted({r['language'] for r in manifest})),
                'author_model_names_release':sources['fake_model_names'],
                'author_model_names_selected':len({r['generator_id'] for r in manifest if r['label']=='1'}),
                'real_speaker_identity':'UNKNOWN','fake_source_text_speakers':len({r['source_speaker_id'] for r in manifest if r['label']=='1'}),
                'license':'FAKE CC-BY-NC-4.0; REAL M-AILABS original notice','role':'DEVELOPMENT'}]
    write_csv(output/'table1_dataset_inventory.csv',inventory)
    cohort=[]
    for group in ['REAL']+config_groups(manifest):
        selected=[r for r in manifest if (r['label']=='0' if group=='REAL' else r['label']=='1' and r['generator_id']==group)]
        cohort.append({'group':group,'research_split':'development','n':len(selected),
                       'real':sum(r['label']=='0' for r in selected),'fake':sum(r['label']=='1' for r in selected),
                       'local_or_annotated_parent_keys':len({r['parent_recording_id'] for r in selected}),
                       'known_source_text_speakers':len({r['source_speaker_id'] for r in selected}-{'UNKNOWN'}),
                       'independent_population_units':'UNKNOWN','excluded':0,'exclusion_reason':'NONE; full prespecified selection executed'})
    write_csv(output/'table2_cohort.csv',cohort)
    write_csv(output/'table3_channel_results.csv',metrics)
    write_csv(output/'table4_interactions.csv',interactions)
    confounders=[
        {'potential_confounder':'language','method':'Author annotation/audiofolder counts','evidence':'English REAL and FAKE',
         'mitigation':'Same-language selection before inference','residual':'No multilingual generalization'},
        {'potential_confounder':'source corpus','method':'original_file and author card','evidence':'M-AILABS original/source-text association',
         'mitigation':'Common source origin; source-text speaker balanced for FAKE','residual':'Synthesis/recording environment and REAL lineage not equalized'},
        {'potential_confounder':'sample rate','method':'Decoded actual WAV rates','evidence':json.dumps(dict(Counter(r['sample_rate'] for r in manifest))),
         'mitigation':'Same source rate and common model resampling','residual':'Other container/amplitude/silence cues remain'},
        {'potential_confounder':'metadata shortcut','method':'Grouped development logistic baselines on original and prepared input',
         'evidence':json.dumps(shortcuts['detector_input']['results']), 'mitigation':'Report all baselines; constant features marked noninformative',
         'residual':'Does not identify which feature the detector uses'},
        {'potential_confounder':'speaker/parent dependence','method':'Manifest and known source component grouping',
         'evidence':'Four annotated FAKE source-text speakers; REAL source identity UNKNOWN',
         'mitigation':'Joint cluster bootstrap for FAKE group contrasts; paired waveform reuse', 'residual':'No population-independence certificate'},
        {'potential_confounder':'threshold transfer','method':'Fixed inherited threshold plus pooled AUC/EER',
         'evidence':'No fitting on public labels','mitigation':'Report operating and ranking diagnostics separately','residual':'Neither score calibration nor causal mechanism is identified'}]
    write_csv(output/'table5_confounders.csv',confounders)
    novelty=[{'dimension':key,'nearest_prior':prior,'conference_LAVA':conference,'C1_Phase3':current,'status':'NOVELTY_NOT_ESTABLISHED'} for key,prior,conference,current in [
        ('transformations','RADAR sequential media pipeline','100-record diagnostic','Explicit fixed-power paired controls'),
        ('order sensitivity','RADAR ordered pipeline; reversal estimator not identified in overview','Limited channel diagnostic','Within-record order reversal'),
        ('generator identity','MLAAD/source tracing annotated models','UNKNOWN','Four author system names; checkpoint hash UNKNOWN'),
        ('estimator','Established difference-in-differences/cluster bootstrap','Not established','Application to annotated FAKE source groups'),
        ('confounders','DSPA fixed-threshold/class-asymmetric noise diagnostics','Limited','Headroom/PCM/noise/source/shortcut audit'),
        ('confirmation','RADAR separate evaluation','None for new claim','Not performed'),
        ('insight','Known robustness/generalization problem','Conference benchmark conclusions','Exploratory observations, not mechanism proof')]]
    write_csv(output/'table6_novelty.csv',novelty)
    summary={'status':'EXTERNAL_DEVELOPMENT_EVALUATED','novelty':'NOT_ESTABLISHED','confirmation':'CONFIRMATION_NOT_LOCKED',
             'dataset':inventory[0],'cohort':cohort,'models':{m:{'counts':v['metadata']['counts'],'threshold':v['metadata']['threshold'],
             'elapsed_seconds':v['metadata']['elapsed_seconds'],'overall_metrics':[r for r in v['statistics']['metric_table'] if r['group']=='ALL'],
             'order_contrasts':v['statistics']['order_contrasts']} for m,v in models.items()},
             'shortcut_report_sha256':file_hash(base/'audit/phase3_shortcuts_v1/shortcut_report.json')}
    write_json(output/'summary.json',summary)
    lines=['# Phase 3 executed research findings','', 'Updated 27 October 2026. Scope: **EXPLORATORY PUBLIC DEVELOPMENT**. Novelty **NOT_ESTABLISHED**; confirmation **NOT_LOCKED**.', '',
           'Actual MLAAD-tiny cohort: 192 audio / 96 REAL / 96 FAKE, four author-annotated English system groups, 24 FAKE each. Source revision and license evidence are in PHASE3_DATASET_DECISION.md and PHASE3_DATA_ACQUISITION.md.', '',
           'Four known original source-text speakers contribute to FAKE, with balanced six recordings per speaker/system. REAL speaker/original identity is UNKNOWN. Local or annotated parent keys and source-text speakers are not a certified number of independent population units.', '',
           'Seven conditions share 1,344 immutable waveform files between detectors. They use identical sample IDs, noise instances and codec outputs. Scores and thresholds remain separate per artifact. No retraining, calibration, confirmation data or performance-based sample selection was used.', '',
           '| Detector | Threshold | Successful / attempted | Failures | Elapsed seconds |','|---|---:|---:|---:|---:|']
    for model,value in models.items():
        meta=value['metadata']; counts=meta['counts']
        lines.append(f"| {model} | {meta['threshold']:.4f} | {counts['successful']}/{counts['attempted']} | {counts['failed']} | {meta['elapsed_seconds']:.2f} |")
    lines += ['', 'Timing includes artifact loading and report generation in two concurrent CPU processes; this is not a latency/fair-speed benchmark. Complete paired scores are persisted; failed predictions are never removed.', '',
              '| Detector | Condition | Error | REAL false alarm | FAKE miss | ROC-AUC | Diagnostic EER |','|---|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        if row['group']=='ALL':
            lines.append(f"| {row['model']} | {row['condition']} | {row['error_rate']:.4f} | {row['false_alarm_rate']:.4f} | {row['miss_rate']:.4f} | {row['auc']:.4f} | {row['eer']:.4f} |")
    lines += ['', 'EER is a ranking diagnostic calculated after execution, not a replacement decision threshold. Binary F1/AUC/EER are missing for FAKE-only system groups; miss-rate denominators are 24 for those groups and 96 for pooled FAKE.', '',
              '## Interpretation supported by these executions', '',
              'Both artifacts lose detection quality under noisy chains. Their inherited thresholds also show large class-asymmetric shifts: more FAKE misses can coincide with fewer REAL false alarms. On this balanced cohort both artifacts show only a 2/192 increase in overall error for noise→codec versus codec→noise, while MobileNet has nine additional FAKE misses and ShuffleNet thirteen. The aggregate difference conceals cancellation between the two classes. This is an exploratory artifact observation, not proof of a new general mechanism.', '',
              'Headroom preparation changes the native-baseline results; therefore it cannot be treated as neutral transport. The separate native/prepared/PCM conditions must stay in the report. Ranking and operating-point behavior differ across treatments, so the result does not justify fitting a new threshold on this cohort and claiming robustness.', '',
              'The two artifacts are not matched training replicas. Their cross-artifact agreement does not establish architectural superiority, global generator ordering or independence from legacy training. Four source-text clusters give limited support for interaction uncertainty. No follow-up condition was selected by these observed scores.', '',
              '| Detector | FAKE system | Order miss contrast | Descriptive source-text-speaker 95% interval | Clusters |', '|---|---|---:|---|---:|']
    for model,value in models.items():
        for row in value['statistics']['order_contrasts']:
            if row.get('risk')=='error' and row['cluster_field']=='source_speaker_id':
                lines.append(f"| {model} | {row['group']} | {row['estimate']:.4f} | {row['ci95']} | {row['parents']} |")
    lines += ['', 'Order means noise→codec minus codec→noise. Positive means more FAKE misses in the first order. Intervals resample four annotated source-text speakers, not four independent synthesized identities. They are descriptive, non-simultaneous and cannot support significance or population equivalence. Every pairwise group/channel result is included in table4_interactions.csv; no favorable subset is selected.', '',
              'Generator×channel contrasts are observational FAKE-miss difference-in-differences with common English/original-source support and joint source-text-speaker bootstrap. Checkpoint identity, reference/synthesized voice composition and model-relative familiarity remain unresolved. Numerical identifiability of these annotation-defined cells is not causal identification.', '',
              'Shortcut results, including the actual prepared model inputs, are in phase3_shortcuts_v1/shortcuts.csv and table5_confounders.csv. Constant sample rate has no predictive content and is explicitly missing. The original LOPO constant-feature artifact is not treated as a result. Strong silence/numeric diagnostic performance means balanced language and rate do not remove dataset fingerprints.', '',
              'The supported result is external execution plus measured transfer/order/threshold behavior for these two artifacts on this cohort. No new forensic mechanism, generator superiority, detector improvement, SOTA, three international contributions or JAES readiness follows. See NEAREST_PRIOR_ART.md and PHASE3_CONFIRMATION_GATE.md.', '',
              'Programmatic tables: outputs/jaes_v2/audit/phase3_report_v1/table1_…table6_*.csv. Executed figures: figures/ in each detector run (SVG and 300 DPI PNG). No journal Results section or accepted manuscript was altered.']
    (root/'docs/jaes_v2/PHASE3_RESEARCH_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({model:value['metadata']['counts'] for model,value in models.items()},indent=2))


def config_groups(manifest):
    return sorted({row['generator_id'] for row in manifest if row['label']=='1'})


if __name__=='__main__':
    main()
