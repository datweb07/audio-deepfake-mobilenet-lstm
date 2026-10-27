"""Build executed Phase 4 tables and findings, keeping blocked estimands missing."""
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from src.lava.jaes_v2.evidence import read_csv,write_csv,write_json,file_hash


def main() -> None:
    root=Path(__file__).resolve().parents[2]; base=root/'outputs/jaes_v2'
    output=base/'audit/phase4_report_v1'
    if output.exists():
        raise FileExistsError('New research-report directory required')
    matching=base/'audit/phase4_source_matching'
    coverage=json.loads((matching/'matching_coverage.json').read_text())
    cache=base/'c1_nuisance_controls_dev_v1_waveforms'
    manifest=read_csv(cache/'manifest.csv'); features=read_csv(cache/'signal_features.csv')
    feature_diag=json.loads((base/'audit/phase4_feature_diagnostics_v2/diagnostic.json').read_text())
    models={}; performance=[]; contrasts=[]; planning=[]
    for model in ('mobilenetv3_lstm','shufflenetv2_lstm'):
        directory=base/('c1_nuisance_controls_dev_v1_'+model)
        meta=json.loads((directory/'run_metadata.json').read_text())
        if meta['status']!='WITHIN_RECORD_DEVELOPMENT_EVALUATED':
            raise ValueError('Executed complete inference required')
        report=json.loads((directory/'statistics.json').read_text())
        models[model]={'metadata':meta,'report':report}
        performance.extend({'model':model,**row,'class_score_distributions':json.dumps(row['class_score_distributions'],sort_keys=True)} for row in report['metrics'])
        contrasts.extend({'model':model,**row,'ci95':json.dumps(row['ci95']),'controlled_order':json.dumps(row['controlled_order'],sort_keys=True)} for row in report['contrasts'])
        planning.extend({'model':model,**row} for row in report['precision_plans'])
    output.mkdir(parents=True)
    shutil.copyfile(matching/'table1_source_feasibility.csv',output/'table1_source_feasibility.csv')
    blocked=[{'contrast':row['G0']+' minus '+row['G1'],'common_author_source_ids':row['source_overlap'],
              'language':'en','stage':'METADATA_CANDIDATE_NOT_EXECUTED','effect':None,'ci95':None,
              'independent_clusters':'NOT_ESTABLISHED','status':'INSUFFICIENT_SOURCE_MATCHES'} for row in coverage['best_english_pairs']]
    write_csv(output/'table2_matched_candidate_feasibility.csv',blocked)
    write_csv(output/'table5_source_matched_interactions.csv',blocked)
    write_csv(output/'table4_classwise_performance.csv',performance)
    write_csv(output/'within_record_contrasts.csv',contrasts)
    write_json(output/'precision_planning.json',planning)
    write_csv(output/'nuisance_cohort.csv',[{'sample_id':r['sample_id'],'label':r['label'],'system':r['generator_id'],
        'source_recording_id':r['source_recording_id'],'source_speaker_id':r['source_speaker_id'],'scope':'WITHIN_RECORD_DEVELOPMENT_ONLY'} for r in manifest])
    labels={r['sample_id']:r['label'] for r in manifest}
    factors=('rms','peak','silence_fraction_below_1e-4','leading_below_threshold_seconds',
             'trailing_below_threshold_seconds','above_threshold_sample_seconds','rolloff95_hz','original_fraction_not_observed')
    nuisance=[]
    for policy in ('global_half_v1','rms_anchor_v1','edge_repack_v1'):
        for label in ('0','1'):
            before=[r for r in features if r['policy']=='phase3_baseline' and r['condition']=='clean' and labels[r['sample_id']]==label]
            after=[r for r in features if r['policy']==policy and r['condition']=='clean' and labels[r['sample_id']]==label]
            for factor in factors:
                a=float(np.mean([float(r[factor]) for r in before])); b=float(np.mean([float(r[factor]) for r in after]))
                nuisance.append({'factor':factor,'label':label,'baseline':'prepared padded 3-second detector input','policy':policy,
                                 'before_mean':a,'after_mean':b,'difference':b-a,'n':len(after),
                                 'limitation':'Signal sensitivity, not semantic or causal shortcut identification'})
    write_csv(output/'table3_nuisance_diagnostics.csv',nuisance)
    novelty=[
        {'candidate_claim':'Source-controlled generator/order interaction','closest_prior':'RADAR; ASVspoof5 codec pairing; MLAAD source tracing',
         'new_evidence':'Sparse source-ID overlap quantified; fixed-pair study blocked','uncertainty':'No matched-source experiment or independent replication',
         'status':'NOT_ESTABLISHED; NOT_PRIMARY_NOVELTY'},
        {'candidate_claim':'Silence/amplitude shortcut controls','closest_prior':'ASVspoof5 shortcut-reduction pipeline; CVPR2025 Circumventing Shortcuts',
         'new_evidence':'Within-record controls and exact Mel-floor counterfactual on existing artifacts',
         'uncertainty':'No demonstrated better detector or novel general mechanism','status':'ENGINEERING_VALIDATION_CONTRIBUTION'},
        {'candidate_claim':'Class-asymmetric order failures','closest_prior':'RADAR; FP-NAC; fixed-threshold noise diagnostics',
         'new_evidence':'Actual classwise nuisance sensitivity, four original source clusters',
         'uncertainty':'No population inference or three-contribution claim','status':'SUPPORTING_VALIDATION'}]
    write_csv(output/'table6_novelty_evidence.csv',novelty)
    summary={'status':'WITHIN_RECORD_DEVELOPMENT_EVALUATED','matched_experiment':'SOURCE_MATCHED_EXPERIMENT_BLOCKED',
             'C1_recommendation':'SUPPORTING_VALIDATION_AND_ENGINEERING; NOT_A_PRIMARY_INTERNATIONAL_NOVELTY',
             'source_matching':coverage,'feature_diagnostic':feature_diag,
             'primary':{model:next(r for r in value['report']['contrasts'] if r['primary']) for model,value in models.items()},
             'counts':{model:value['metadata']['counts'] for model,value in models.items()},
             'independent_population_units':'UNKNOWN','confirmation':'CONFIRMATION_NOT_LOCKED'}
    write_json(output/'summary.json',summary)
    lines=['# Phase 4 executed research findings','', '27 October 2026. **WITHIN_RECORD_DEVELOPMENT_EVALUATED; SOURCE_MATCHED_EXPERIMENT_BLOCKED.**', '',
           'The six Phase 3 system pairs share zero selected original source IDs. Full tiny metadata contains 290 same-language multi-system source groups, but the largest English fixed-system-pair overlap is three sources. Author-source/text matches exist at level 3; none was certified level 4, and no REAL mapping was verified. A sparse incomplete-block analysis would require a different estimand/design; these findings do not prove all conceivable matched development designs impossible.', '',
           'The executed fallback is c1_nuisance_controls_dev_v1: 32 existing English public audio files (16 REAL, 16 FAKE). FAKE has four systems × four original source-text speakers, one seeded recording per cell. Those source recordings are different across systems. REAL original identity remains UNKNOWN. There are 20 connected known/local components, not 20 certified independent speakers. No additional audio was downloaded.', '',
           'Three versioned interventions (global half-amplitude, RMS anchor with cap, observed-window edge repacking) each precede clean/noise→codec/codec→noise. Their fixed-absolute noise is anchored once to each intervention’s clean reference, with the original sample-derived unit-noise seed. No post-channel normalization occurs. Existing native/prepared/PCM/codec/noise/order scores are reused, not predicted again.', '',
           '| Artifact | New predictions | Reused scores | New failures | Runtime seconds |','|---|---:|---:|---:|---:|']
    for model,value in models.items():
        meta=value['metadata']; c=meta['counts']
        lines.append(f"| {model} | {c['successful_new']}/{c['new_predictions']} | {c['reused_scores']} | {c['failed_new']} | {meta['elapsed_seconds']:.2f} |")
    lines+=['', 'Both artifact jobs reuse the same 288 new waveforms. Two separate warm-up forwards are recorded outside scored counts. Concurrent CPU timing includes loading/reporting and is not an efficiency comparison.', '',
            '| Artifact | Primary change in FAKE order contrast | Descriptive 95% interval | Source-text clusters |', '|---|---:|---|---:|']
    for model,value in models.items():
        r=next(r for r in value['report']['contrasts'] if r['primary'])
        lines.append(f"| {model} | {r['estimate']:.4f} | {r['ci95']} | {r['parents']} |")
    lines+=['', 'Primary = [(noise→codec miss − codec→noise miss) after edge repacking] minus the same difference at prepared baseline, averaged equally across four original source-text-speaker clusters. This is a within-record preprocessing sensitivity, not a generator effect. All 18 descriptive contrasts per artifact are retained; no p-values/significance. Edge repacking shifts segment alignment and can remove low-energy phonetic tails, so it is not a certified semantic intervention.', '',
            '| Artifact | Policy | Condition | REAL false alarm | FAKE miss | ROC-AUC | Decision flips vs same-channel baseline |', '|---|---|---|---:|---:|---:|---:|']
    for r in performance:
        if r['condition'] in ('clean','fixed_A_then_B','fixed_B_then_A'):
            lines.append(f"| {r['model']} | {r['policy']} | {r['condition']} | {r['false_alarm_rate']:.4f} | {r['miss_rate']:.4f} | {r['auc']:.4f} | {r['decision_flips_vs_same_channel_baseline']} |")
    lines+=['', 'Severe noisy-chain misses remain after these controls. They do not establish that the detector uses silence as a shortcut or that a generator-specific mechanism survives content matching. Thresholds remain .82/.12; AUC/EER are diagnostics, not fitted operating points.', '',
            '## Measured feature-level explanation', '',
            f"On all {feature_diag['sources']} exact prepared inputs, half-amplitude waveform parity is exact. Fixed production Mel floor gives maximum tensor change {feature_diag['maximum_tensor_difference']:.8f} on the 0–255 input scale and mean absolute change {feature_diag['mean_tensor_absolute_difference']:.8f}. In a model-free diagnostic only, scaling the floor from 1e-10 to 2.5e-11 reduces the maximum tensor residual to {feature_diag['counterfactual_max_tensor_residual']:.10f}. This isolates the absolute-floor contribution to feature non-invariance up to floating precision; it does not measure detector performance under an altered preprocessor. Production preprocessing and accepted results are unchanged.", '',
            'The pure-amplitude clean sensitivity caused three MobileNet and two ShuffleNet decisions to flip on these 32 records. This cautions against attributing all channel shifts to lost synthesis artifacts. The model-free floor counterfactual is an engineering diagnosis, not a trained method or world-first contribution.', '',
            '## Decision', '', 'C1 should currently be a **supporting validation / engineering contribution**, not one of three established primary international novelties. The source-matched scientific claim remains untested; confirmation is NOT_LOCKED. C2 needs a useful, licensed and compatible teacher or justified teacher-free alternative, controlled training data and a mechanism distinct from augmentation/KD/consistency. C3 remains deferred.', '',
            'Programmatic tables: outputs/jaes_v2/audit/phase4_report_v1/. Source audit: outputs/jaes_v2/audit/phase4_source_matching/. Figures: source audit figures/ and both nuisance artifact figures/. Precision assumptions and next-stage requirements are documented separately.']
    (root/'docs/jaes_v2/PHASE4_RESEARCH_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary['primary'],indent=2))


if __name__=='__main__':
    main()
