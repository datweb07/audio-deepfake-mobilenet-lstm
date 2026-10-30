"""Build evidence-linked Phase1–6 handoff without neural inference or hard-coded metrics."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import re
import shutil
import statistics
import zipfile
import xml.etree.ElementTree as ET
from src.lava.jaes_v2.evidence import file_hash,read_csv,write_json
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.public_analysis import metrics
from src.lava.jaes_v2.runtime import preflight

UNKNOWN='UNKNOWN'


def verify_metric_rows(rows,declared):
    """Reject stale or edited scalar summaries, rather than copying reported numbers."""
    for item in declared:
        selected=[r for r in rows if r['condition_id']==item['condition']]
        if not selected: raise ValueError('Declared condition has no executed scores')
        computed=metrics(selected)
        for key in ('n','error_rate','false_alarm_rate','miss_rate','auc','eer','f1'):
            if key in item and item[key] is not None and (computed[key] is None or abs(float(item[key])-computed[key])>1e-10):
                raise ValueError('Executed-score/metric mismatch: '+str((item['condition'],key)))


def md_table(headers,rows):
    def cell(x): return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+
                     ['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])


def phase_of(path):
    s=path.as_posix()
    if '/phase6/' in s: return 6
    if '/c2/' in s or 'phase5_preprocessing' in s: return 5
    for text,phase in [('nuisance',4),('mlaad',3),('channel_controls',2),('noise_opus',1)]:
        if text in s:return phase
    return UNKNOWN


def collect(root):
    evidence={}; gaps=[]
    def read(relative):
        path=root/relative
        if not path.is_file(): gaps.append(str(relative)); return None
        evidence[str(relative).replace('\\','/')]=file_hash(path)
        return json.loads(path.read_text(encoding='utf8'))
    before=read('outputs/jaes_v2/audit/phase6_before.json')
    if before is None: raise ValueError('Missing initial integrity audit')
    changed=[name for name,h in before['historical_files'].items() if not (root/name).is_file() or file_hash(root/name)!=h]
    if changed: raise ValueError('Historical evidence changed: '+str(changed[:5]))
    live_environment=preflight(root)
    environment_path=root/'outputs/jaes_v2/phase6/report_environment_v1.json'
    if not environment_path.exists(): write_json(environment_path,live_environment)
    env=read(environment_path.relative_to(root))
    manuscript=root/'papers/PaperID_804_final.docx'; evidence['papers/PaperID_804_final.docx']=file_hash(manuscript)
    with zipfile.ZipFile(manuscript) as z: tree=ET.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paper='\n'.join(''.join(p.itertext()) for p in tree.findall('.//w:t',ns))
    questions=[]
    for i in (1,2,3):
        match=re.search(r'RQ'+str(i)+r':\s*(.*?)(?=RQ\d:|The study makes)',paper,re.S)
        questions.append(' '.join(match.group(1).split()) if match else UNKNOWN)
    split=root/'data/manifests/split_manifest.csv'; legacy=read_csv(split); evidence[split.relative_to(root).as_posix()]=file_hash(split)
    datasets=[dict(name='LAVA_LEGACY_INTERNAL',status='EXECUTED',counts=dict(Counter(r['split'] for r in legacy)),
                   label_counts=dict(Counter(r.get('label',UNKNOWN) for r in legacy)),manifest=split.relative_to(root).as_posix(),source=UNKNOWN,license=UNKNOWN,
                   lineage='Byte-group disjoint; speaker/source/generator UNKNOWN')]
    foundation=read('outputs/jaes_v2/datasets/asv2019_foundation_v1/foundation.json')
    if foundation:
        archive=read('outputs/jaes_v2/datasets/asv2019_archive_v1/archive_inventory.json')
        observed=[]
        for role in ('train','development'):
            path=root/f'outputs/jaes_v2/datasets/asv2019_foundation_v1/{role}_manifest.csv'
            observed+=read_csv(path)
        datasets.append(dict(name='ASVspoof2019_LA',status='TRAIN_DEV_EXTRACTED_EVAL_UNOPENED',counts=foundation['counts'],
            source='https://datashare.ed.ac.uk/handle/10283/3336',license='ODC-By database; upstream content caveats',archive_sha256=foundation['archive_sha256'],
            manifests={role:dict(path=f'outputs/jaes_v2/datasets/asv2019_foundation_v1/{role}_manifest.csv',sha256=h) for role,h in foundation['manifest_hashes'].items()},
            lineage=foundation['limitations'],storage=archive,
            observed_train_dev_profile=dict(sample_rates=sorted({int(r['sample_rate']) for r in observed}),
                duration_seconds=dict(min=min(float(r['duration']) for r in observed),median=statistics.median(float(r['duration']) for r in observed),max=max(float(r['duration']) for r in observed)),
                calibration_profile='NOT_REOPENED; foundation counts/hashes retained')))
    for name,path,source,status in [('MLAAD_TINY','outputs/jaes_v2/datasets/mlaad_tiny_dev_v1/provenance_manifest.csv','DERIVE_FROM_MANIFEST','EXPLORATORY_DEVELOPMENT_EXECUTED')]:
        if (root/path).exists():
            rows=read_csv(root/path); evidence[path]=file_hash(root/path)
            datasets.append(dict(name=name,status=status,count=len(rows),class_counts=dict(Counter(r['label'] for r in rows)),
                source=sorted({r.get('source_url_or_record',UNKNOWN) for r in rows}),manifest=path,license=sorted({r['license'] for r in rows}),source_revision=sorted({r.get('source_revision',UNKNOWN) for r in rows}),
                sample_rates=sorted({int(r['sample_rate']) for r in rows}),duration_seconds=dict(min=min(float(r['duration']) for r in rows),median=statistics.median(float(r['duration']) for r in rows),max=max(float(r['duration']) for r in rows)),
                known_source_speakers=sorted({r.get('source_speaker_id',UNKNOWN) for r in rows}-{UNKNOWN}),
                lineage='Four selected author systems; source-text clusters are not synthesized-voice independence; REAL lineage UNKNOWN'))
    datasets += [dict(name='RADAR2026',status='METADATA_ONLY_NO_AUDIO',source='https://zenodo.org/records/22110815',license='CC-BY-4.0 release declaration',counts='See Phase3 release inventory; no locally executed audio'),
                 dict(name='In-the-Wild',status='NOT_ACQUIRED',source='https://huggingface.co/datasets/mueller91/In-The-Wild',license='Repository Apache-2.0 declaration; upstream conditions',counts='NOT_VERIFIED_LOCALLY'),
                 dict(name='DeepFakeVox-HQ / ANC-Spoof',status='PRIOR_ART_ONLY_NOT_ACQUIRED',source='F-SAT / PCL-NET papers',license='NOT_VERIFIED_FOR_LOCAL_USE',counts='NOT_MEASURED_LOCALLY')]
    from src.lava.registry import get_spec
    models=[]; clean=[]; benchmark_diagnostics=[]
    model_ids=['mobilenetv3_lstm','shufflenetv2_lstm','mnasnet_lstm','efficientnet_b0_lstm','rawnet2','aasist']
    for name in model_ids:
        source=f'outputs/lava_6/clean/{name}/summary.json' if name=='shufflenetv2_lstm' else f'outputs/lava_5/clean/{name}/summary.json'
        summary=read(source)
        if summary:
            clean.append(dict(source=source,**summary)); spec=get_spec(summary.get('registry_name',name)); record=json.loads(json.dumps(asdict(spec),default=str))
            artifact=Path(spec.model_artifact); record.update(role='FROZEN_CONFERENCE_ARTIFACT',training_status='External reference checkpoint' if name in ('rawnet2','aasist') else 'Historical local training',
                artifact_sha256=file_hash(artifact if artifact.is_absolute() else root/artifact) if artifact.is_file() else 'NOT_VERIFIED',benchmark_source=source)
            efficiency=f'outputs/lava_6/efficiency/{name}/summary.json' if name=='shufflenetv2_lstm' else f'outputs/lava_5/efficiency/{name}/summary.json'
            record['efficiency']=read(efficiency)
            meta_path=Path(spec.metadata_artifact)
            if meta_path.is_file(): record['training_metadata']=read(meta_path.relative_to(root) if meta_path.is_absolute() else meta_path)
            scorefile=(root/source).parent/'scores.csv'
            if scorefile.is_file():
                scored=read_csv(scorefile); transformed=[dict(label=r['true_label'],raw_score=r['p_fake'],predicted_label=r['predicted_label'],condition_id='clean') for r in scored]
                verify_metric_rows(transformed,[dict(condition='clean',n=summary['samples'],auc=summary['roc_auc'],eer=summary['eer'],f1=summary['f1'])])
                if summary.get('scores_sha256') and file_hash(scorefile)!=summary['scores_sha256']: raise ValueError('Benchmark score checksum changed')
            models.append(record)
    for folder in ('outputs/lava_5/diagnostic_100','outputs/lava_6/diagnostic_100'):
        for p in sorted((root/folder).rglob('summary.json')):
            d=read(p.relative_to(root))
            if d and 'eer' in d and 'condition' in d: benchmark_diagnostics.append(dict(source=p.relative_to(root).as_posix(),**d))
    experiments=[]; logical={}; journal_metrics=[]
    for p in sorted((root/'outputs/jaes_v2').rglob('run_metadata.json')):
        rel=p.relative_to(root).as_posix(); meta=read(rel); directory=p.parent; phase=phase_of(p)
        if phase==UNKNOWN: continue
        cfg=read(directory.relative_to(root)/'resolved_config.json') if (directory/'resolved_config.json').exists() else {}
        cfg=cfg or meta.get('config',{})
        score_path=next((directory/f for f in ('validation_scores.csv','scores.csv') if (directory/f).exists()),None)
        rows=read_csv(score_path) if score_path else []
        declared=None
        if rows and (directory/'metrics.json').is_file():
            declared=read(directory.relative_to(root)/'metrics.json')
            if isinstance(declared,list): verify_metric_rows(rows,declared)
        if score_path:evidence[score_path.relative_to(root).as_posix()]=file_hash(score_path)
        model=meta.get('model_id',meta.get('research_model_id',UNKNOWN)); model_values=sorted({r.get('model_id',model) for r in rows})
        identity=meta.get('experiment_id',cfg.get('experiment_id',directory.name)); key=(phase,identity,tuple(model_values))
        if key in logical: raise ValueError('Duplicate/inconsistent experiment identity '+str(key))
        logical[key]=rel
        seal=verify_seal(directory,meta) if meta.get('output_hashes') else 0
        train=bool(phase>=5 and ('/c2/' in rel or '/training/' in rel or 'finetune' in rel))
        item=dict(experiment_id=identity,phase=phase,model=model_values,training=train,status=meta.get('status',UNKNOWN),scope=meta.get('scope',UNKNOWN),
            source=rel,output=directory.relative_to(root).as_posix(),metadata_sha256=file_hash(p),config=cfg,
            seed=cfg.get('seed',meta.get('seed',meta.get('training_seed',UNKNOWN))),threshold=meta.get('threshold',cfg.get('threshold',UNKNOWN)),
            recordings=len({r['sample_id'] for r in rows}) if rows and 'sample_id' in rows[0] else UNKNOWN,
            conditions=len({r.get('condition_id',UNKNOWN) for r in rows}) if rows else UNKNOWN,score_rows=len(rows),verified_sealed_files=seal,
            counts=meta.get('counts',{}),optimizer_steps=meta.get('optimizer_steps',UNKNOWN),elapsed_seconds=meta.get('elapsed_seconds','NOT_MEASURED'),
            checkpoint_sha256=meta.get('checkpoint_sha256',UNKNOWN),declared_metrics=declared,limitations=meta.get('limitation',meta.get('limitations',UNKNOWN)))
        experiments.append(item)
        completed=meta.get('status') not in ('FAILED','INTERRUPTED','TRAINING','FAILED_PROCESS_TERMINATION')
        if rows and completed and all(k in rows[0] for k in ('label','raw_score','predicted_label','condition_id')):
            for condition in sorted({r['condition_id'] for r in rows}):
                # Keep multiple model/preprocessor strata separate in the frozen-model diagnostic.
                strata=sorted({(r.get('model_id',model),r.get('preprocessor_id','NONE')) for r in rows if r['condition_id']==condition})
                for m,policy in strata:
                    selected=[r for r in rows if r['condition_id']==condition and r.get('model_id',model)==m and r.get('preprocessor_id','NONE')==policy]
                    journal_metrics.append(dict(phase=phase,experiment_id=identity,model=m,policy=policy,condition=condition,source=score_path.relative_to(root).as_posix(),**metrics(selected)))
        if train:
            checkpoint=directory/'best.keras'
            if not checkpoint.exists(): checkpoint=directory/'sanity.keras'
            models.append(dict(name=identity,framework='TensorFlow',role='PHASE6_SANITY' if 'finetune' in rel else 'JOURNAL_RESEARCH_STUDENT',
                architecture='MobileNetV3Small + LSTM128 + Dense64 + sigmoid',input='six224x224x3 Mel images / three-second observed window',
                initialization='ImageNet; fresh head',pretraining_hash=meta.get('pretraining_weight_sha256',UNKNOWN),
                trainable_scope=meta.get('training_policy','HEAD_ONLY' if '/c2/' in rel else UNKNOWN),artifact=str(checkpoint.relative_to(root)) if checkpoint.exists() else 'NOT_AVAILABLE',
                artifact_sha256=file_hash(checkpoint) if checkpoint.exists() else 'NOT_AVAILABLE',training_status=meta.get('status',UNKNOWN)))
    phase6_selection=read('outputs/jaes_v2/phase6/data_selection_v1/selection.json'); phase6_report=read('outputs/jaes_v2/phase6/reports_v1/summary.json')
    fine=read('outputs/jaes_v2/phase6/partial_finetune_sanity_v2/run_metadata.json')
    regression=read('outputs/jaes_v2/audit/phase6_validation_final_v2/validation.json')
    scaling=read('outputs/jaes_v2/phase6/scaling_plan_v1.json'); resource=read('outputs/jaes_v2/phase6/resource_preflight_v1.json')
    protocol=read('outputs/jaes_v2/phase6/protocol_v1.json')
    system=read('outputs/jaes_v2/phase6/resource_system_final.json')
    efficiency=read('outputs/jaes_v2/phase6/efficiency_v1.json')
    cache=read('outputs/jaes_v2/phase6/feature_cache_v2/cache_metadata.json')
    historical_integrity={}
    for phase in (3,4,5): historical_integrity[str(phase)]=read(f'outputs/jaes_v2/audit/phase{phase}_final_integrity.json')
    inventory=dict(project='LAVA Journal V2',current_phase=6,updated='2026-10-30 Asia/Novosibirsk',conference_baseline=dict(manuscript='papers/PaperID_804_final.docx',sha256=file_hash(manuscript),research_questions=questions,clean=clean,diagnostic_benchmarks=benchmark_diagnostics),
        datasets=datasets,models=models,experiments=experiments,journal_metrics=journal_metrics,
        contributions=[dict(id='C1',status='SUPPORTING_VALIDATION_ENGINEERING'),dict(id='C2',status='STRONG_BASELINES_PILOT_EVALUATED; NOVELTY_NOT_ESTABLISHED'),dict(id='C3',status='DEFERRED')],
        reproducibility=dict(environment=env,historical_files_unchanged=len(before['historical_files']),historical_integrity=historical_integrity,regression=regression,
            evidence_sha256=evidence,evidence_gaps=gaps,confirmation_locked=False,neural_inference=False),
        phase6=dict(selection=phase6_selection,analysis=phase6_report,finetuning=fine,scaling=scaling,resource=resource,protocol=protocol,cache=cache,system=system,efficiency=efficiency),
        blockers=[dict(category=k,missing=v,resolution=r,user_action=a) for k,v,r,a in [
            ('DATA','Small balanced subset; limited attacks/speakers; no unused external corpus','Scale controlled cohorts and freeze external role before scores','No immediate download required'),
            ('METHOD','No validated forensic evidence-retention estimator or distinct objective','Compare stronger/equal-view controls; falsify proxy and prior-art distinction','None yet'),
            ('COMPUTE','Production runtimes CPU-only; desktop commit pressure','Use sequential jobs; budget compatible isolated GPU full fine-tuning','Provide validated compute only for next large study'),
            ('VALIDATION','No independent confirmation / operating point certification','Lock unused external/calibration protocol before use','Choose publication/evaluation scope'),
            ('STATISTICS','One initialization seed','At least three independently trained seeds; preserve speaker clustering','Approve measured larger compute budget later'),
            ('PROVENANCE','Original utterance / exact generators UNKNOWN; remote archive digest not certified','Acquire/verify upstream identity and publisher digest','Upstream metadata if available'),
            ('JOURNAL ELIGIBILITY','Conference venue/DOI/copyright and current Clarivate record not verified here','Check AES extension rules, disclose/cite accepted version, verify official MJL','Supply venue, DOI/publication date and copyright terms')]],
        next_steps=['Finish strong online augmentation and balanced all-attack scaling','Validate partial fine-tuning under measured compatible runtime','Predeclare one technically distinct mechanism or reject it','At least three seeds and locked external evaluation; reserve calibration until approved'])
    return inventory


def render(inv):
    out=['# LAVA JOURNAL V2 — MASTER RESEARCH REPORT','',f"Updated {inv['updated']}. Evidence-generated handoff for independent scientific review. Phase1–5 evidence is preserved; this report supersedes outdated current-status prose without rewriting its historical files.",'']
    def section(title,text): out.extend(['## '+title,'',text,''])
    experiment_rows=inv['experiments']; metrics_rows=inv['journal_metrics']; p6=inv['phase6']; env=inv['reproducibility']['environment']
    section('1. Executive Research Summary',
        'Project: LAVA, a lightweight benchmarking framework for robust and efficient deepfake voice detection. The accepted conference work compares deployed artifacts with heterogeneous initialization and thresholds. The journal extension asks which measured failure mechanisms survive source/threshold controls and whether a lightweight learned detector can improve robustness at an explicit clean-risk cost. '
        'Current maturity: controlled development engineering and standard-baseline pilots. There is no independently confirmed new methodological contribution, no claim of three international novelties and no journal-ready conclusion. New research artifacts are separate from frozen production models. '
        'Phase6 adds balanced speaker/class sampling, amplitude and equal-view consistency controls, finite RawBoost integration, reliability diagnostics, a real partial-backbone gradient sanity path, and this cross-phase inventory. Limits include small cohorts, one seed, unknown original-source lineage, bounded CPU resources, and protected calibration/evaluation that remain unused.')
    section('2. Research Questions',
        '### Accepted conference questions\n\n'+'\n\n'.join(f'- RQ{i+1}: {q}' for i,q in enumerate(inv['conference_baseline']['research_questions']))+
        '\n\nThese are artifact comparison, diagnostic stress and scenario-specific efficiency questions; training and pretraining are heterogeneous, so the answers concern artifacts rather than intrinsic architecture superiority.\n\n'
        '### Journal questions\n\nJ-RQ1: Does channel order create practically important classwise risk changes after insertion-power, source, preprocessing and operating-point controls? Exploratory evidence exists; the fixed-generator source-matched confirmation lacks adequate support. '
        'J-RQ2: Can ordinary amplitude/channel/paired training or a genuinely distinct evidence-preserving mechanism improve a lightweight detector subject to bounded clean REAL false alarms? Standard controls are being tested; a distinct mechanism is not established. '
        'J-RQ3: Can selective inference improve matched-cost risk/coverage with independent calibration and a validated second stage? Unanswered; C3 is deferred.')
    narratives=[(1,'Ordered noise/codec feasibility','Shared source variants, strict scores and parent bootstrap','Observed reversal/interaction is descriptive; insertion-point noise scaling is a confound'),
        (2,'Confounder-aware development','Fixed versus relative noise, headroom/PCM/delay controls and metadata audit','Channel order cannot be isolated from amplitude, codec gain and preparation by one average metric'),
        (3,'External MLAAD-tiny development','Pinned public acquisition, source provenance, common cache and two frozen artifacts','Class-asymmetric noisy-channel failures persist; author system labels are not verified unseen checkpoints'),
        (4,'Source matching and nuisance diagnosis','Source-ID/text matching levels, confirmation feasibility and within-record silence/gain controls','Sparse source support blocks the chosen matched-generator design; absolute Mel floor explains feature non-invariance'),
        (5,'Controlled data and standard learner','ASV foundation, reserved calibration, P0/P1/P2 and separately trained head-only students','Gain invariance can coexist with worse discrimination; clean total error hides opposite REAL/FAKE changes'),
        (6,'Stronger controlled comparisons and consolidation','Speaker-balanced cohort, B1/B4/control/finite RawBoost, partial-backbone sanity and auditable master report','Read actual Phase6 results below; implementation and learning success do not establish novelty')]
    section('3. Complete Phase Timeline',md_table(['Phase','Objective','Implementation','Executed run IDs','Finding / status'],[[p,o,i,', '.join(r['experiment_id'] for r in experiment_rows if r['phase']==p),f] for p,o,i,f in narratives])+
        '\n\nEach phase is incremental. Phase1–4 use frozen conference predictors; Phase5 introduces fresh research students. The matched-generator hypothesis is not silently renamed as a successful experiment. Phase6 uses a new cohort and therefore is not a direct continuation of Phase5 accuracy values.')
    section('4. Dataset Inventory',md_table(['Dataset','Local status','Source','License','Counts / scope'],[[d['name'],d['status'],d['source'],d['license'],d.get('counts',d.get('count',UNKNOWN))] for d in inv['datasets']])+
        '\n\nRADAR and In-the-Wild are investigated options, not executed external evidence. Literature datasets DeepFakeVox-HQ and ANC-Spoof were not acquired. A cited dataset or code repository is not installed training data. No Phase6 multi-GB download or ASV re-extraction occurred. The existing user archive has local SHA/CRC checks; authenticated remote byte identity remains unverified.')
    profiles=[]
    for d in inv['datasets']:
        profile=d.get('observed_train_dev_profile',d)
        if 'sample_rates' in profile: profiles.append([d['name'],profile['sample_rates'],profile.get('duration_seconds',UNKNOWN),d.get('known_source_speakers',d.get('lineage',UNKNOWN))])
    out.extend([md_table(['Observed dataset profile','Rates / Hz','Duration seconds min/median/max','Known speaker/lineage limits'],profiles),''])
    archive=next(d for d in inv['datasets'] if d['name']=='ASVspoof2019_LA').get('storage') or {}
    out.extend(['ASV archive/extraction bytes from the retained central-directory audit: '+str({k:archive.get(k,UNKNOWN) for k in ('archive_bytes','total_expanded_bytes','extracted_bytes','extracted_files')})+'. Calibration header profiles were not reopened in Phase6.',''])
    asv=next(d for d in inv['datasets'] if d['name']=='ASVspoof2019_LA')
    section('5. Dataset Split Protocol',
        'The canonical legacy split groups byte-identical files; it does not certify speaker, generator, original utterance or corpus independence. MLAAD-tiny has a custom exploratory development selection with its official split UNKNOWN; all earlier design use prevents relabeling it as untouched confirmation. '
        'ASV official train stays train; official development is divided by known official speakers into model-selection development and reserved calibration. Evaluation audio/protocol labels remain unopened. The new selector reads only the two train/model-selection manifests, never the reserved calibration file. It excludes development speakers lacking both classes and records the resulting target population. Byte, known-parent, known-speaker and selected decoded-PCM conflicts are checked. Re-encoding, perceptual reuse and unknown original lineage remain unresolved.\n\n'+
        md_table(['ASV research role','Actual counts','Manifest','SHA256'],[[role,asv['counts'][role],item['path'],item['sha256']] for role,item in asv['manifests'].items()])+
        '\n\nPhase6 selection:\n\n```json\n'+json.dumps(p6['selection'],indent=2)+'\n```')
    section('6. Complete Model Inventory',md_table(['Artifact/model','Framework','Role / training','Input / scope','Checkpoint path','SHA256'],[[m.get('name',UNKNOWN),m.get('framework',UNKNOWN),str(m.get('role'))+' / '+str(m.get('training_status')),m.get('input',m.get('input_type',UNKNOWN))+' / '+str(m.get('trainable_scope',m.get('training_policy',UNKNOWN))),m.get('artifact',m.get('model_artifact',UNKNOWN)),m.get('artifact_sha256',UNKNOWN)] for m in inv['models']])+
        '\n\nMobileNet and EfficientNet use declared ImageNet transfer initialization; ShuffleNet and MnasNet conference artifacts use their recorded scratch-training stratum, with pretraining not verified. RawNet2/AASIST are external reference checkpoints, not freshly trained Journal V2 models. Mel models use the retained temporal-image contract; raw references retain their own waveform rate/length contract and score orientation. Research students reuse the existing MobileNetV3Small/LSTM/Dense architecture with fresh head and pinned ImageNet weights. Exact registry metadata, input rates, initialization classes and artifact fields are retained in the accompanying JSON. Failed/interrupted and sanity checkpoints remain labeled by scope.')
    section('7. Complete Experiment Inventory',md_table(['Phase','ID','Scope/status','Models','Train?','Originals / conditions / score rows','Seed / threshold','Output'],[[r['phase'],r['experiment_id'],str(r['scope'])+' / '+r['status'],r['model'],r['training'],f"{r['recordings']} / {r['conditions']} / {r['score_rows']}",f"{r['seed']} / {r['threshold']}",r['output']] for r in experiment_rows])+
        '\n\nWarmups, repeated views, predictor calls, reused scores and source originals have separate denominators. Unknown counts are printed UNKNOWN; missing scores are never promoted into performance. No confirmation experiment exists. The JSON inventory includes verified seal counts, metadata hashes, source count, conditions, update counts and resource measurements for every discovered run, including resource-stopped and interrupted predecessors.')
    section('8. Reproducibility Status',
        f"Initial and current checks preserve {inv['reproducibility']['historical_files_unchanged']} historical Journal output files. Protected baseline result: {env['baseline_freeze_verification']}. Accepted manuscript SHA256: `{inv['conference_baseline']['sha256']}`. "
        'Run seals are verified against every declared output hash; unsealed/incomplete experiments are labeled rather than repaired with invented evidence. Model-free score aggregation recomputes per-condition metrics and figures, not neural inference. RawBoost source bytes/license, selected waveform hashes, transform seeds, pretraining hash and all new checkpoints are traceable. Exact waveform byte hashes certify the generated local realization, not full original-source independence.\n\nRegression evidence:\n\n```json\n'+json.dumps((inv['reproducibility']['regression'] or {}).get('suites','NOT_EXECUTED'),indent=2)+'\n```')
    old=[r for r in metrics_rows if r['phase'] in (1,2,3,4) and (r['phase']<3 or r['condition'] in ('clean','clean_native','fixed_noise','fixed_A_then_B','fixed_B_then_A'))]
    section('9. Phase1–4 Scientific Findings',
        'The ordered pilot demonstrates a reproducible analysis contract, not generator causation. A relative-SNR insertion rule changes absolute noise across arms; fixed-power and codec/PCM/headroom/delay controls are necessary. MLAAD adds real public transfer observations, but its few source-text clusters and UNKNOWN REAL lineage limit inference. FAKE misses can increase while REAL false alarms decrease, concealing failures in total error. '
        'Source matching found no adequate selected fixed-system English pair; author source/text agreement does not become verified acoustic source matching. Within-record silence/gain controls expose nuisance sensitivity. An absolute Mel floor violates ideal gain cancellation; scaling that floor in a counterfactual is a mechanistic diagnostic, not a deployable oracle or a learned contribution. C1 remains supporting validation/engineering.\n\nExecuted condition metrics (all variants retained in JSON):\n\n'+
        md_table(['Phase / experiment','Model','Condition','n','Error','REAL FA','FAKE miss','AUC','EER','Source'],[[str(r['phase'])+' / '+r['experiment_id'],r['model'],r['condition'],r['n'],r['error_rate'],r['false_alarm_rate'],r['miss_rate'],r['auc'],r['eer'],r['source']] for r in old]))
    five=[r for r in metrics_rows if r['phase']==5 and r['experiment_id'].endswith('pilot_v2')]
    section('10. Phase5 Findings',
        'The controlled foundation enabled real separately versioned learning while preserving the accepted paper. The frozen-backbone student pilot and changed-preprocessor diagnostic answer different questions. P1 removes half-gain feature/embedding/score differences on the tested frozen-model cohort yet harms ShuffleNet discrimination. Fresh P1 ERM improves clean ranking/error on its cohort while worsening noisy FAKE misses. Ordinary augmentation changes the clean operating point and therefore must retain classwise risks. '
        'Phase5 development contains unequal REAL/FAKE speaker support; min-size worst-group filtering excludes small REAL-only groups and cannot be reported without denominators. One seed and a small feature-cache cohort are exploratory. Preliminary time-capped/interrupted runs stay distinct from the main fixed-update cells.\n\n'+md_table(['Main run','Condition','Originals','Error','REAL FA','FAKE miss','AUC','EER'],[[r['experiment_id'],r['condition'],r['n'],r['error_rate'],r['false_alarm_rate'],r['miss_rate'],r['auc'],r['eer']] for r in five])+
        '\n\nExisting CPU timing is an offline warm-process diagnostic, not streaming/edge-device certification. NII teacher runtime prerequisites and checkpoint quality remain unresolved; corpus training overlap is known at the ASV/MLAAD level. A synthetic class-orientation helper does not certify real teacher quality.')
    six=[r for r in metrics_rows if r['phase']==6]
    section('11. Phase6 Findings',
        'Speaker-aware selection balances both classes within each retained speaker. Attack/category coverage and excluded speakers are explicit; source utterances remain unknown. B1 uses discrete clean/half-gain exposure with held-out quarter-gain evaluation. B2 uses the existing noise/Opus bank. B4_CONTROL and B4 share the same two supervised views, initialization, order, source IDs and updates; the sole objective difference is probability MSE consistency. '
        'RawBoost uses unchanged MIT author functions behind a seed-isolating wrapper, with two finite algorithm5 realizations at the repo sample rate. It is not online stochastic training or an exact original-paper benchmark reproduction. Larger cohorts are planned from measured costs rather than automatically executed.\n\n'+
        md_table(['Run','Condition','n','Error','REAL FA','FAKE miss','AUC','EER'],[[r['experiment_id'],r['condition'],r['n'],r['error_rate'],r['false_alarm_rate'],r['miss_rate'],r['auc'],r['eer']] for r in six])+
        '\n\nPrimary matched comparison and post-hoc retention-proxy investigation:\n\n```json\n'+json.dumps((p6['analysis'] or {}).get('primary','NOT_EXECUTED'),indent=2)+'\n```\n\n'
        'The embedding-similarity proxy is available without labels; labels assess only its development association afterward. It is not a validated measure of preserved spoof evidence and cannot justify a new weighted loss by itself. Partial-backbone sanity metadata follows; failed predecessors are retained in the experiment inventory.\n\n```json\n'+json.dumps(p6['finetuning'] or 'NOT_MEASURED',indent=2)+'\n```')
    out.extend(['Post-hoc retention-proxy associations (repeated development views; undefined constant-class association is null):','',
        md_table(['Policy','Label','View records','Spearman rho','Interpretation'],[[r['policy'],r['label'],r['n'],r['spearman_rho'],r['interpretation']] for r in (p6['analysis'] or {}).get('retention_proxy',[])]),''])
    risks=[]
    for name,result in (p6['analysis'] or {}).get('results',{}).items():
        c=next(m for m in result['metrics'] if m['condition']=='clean')
        stress=[m for m in result['metrics'] if m['condition'] in ('noise5','A_then_B','B_then_A')]
        risks.append([name,c['false_alarm_rate'],c['miss_rate'],statistics.mean(m['false_alarm_rate'] for m in stress),statistics.mean(m['miss_rate'] for m in stress)])
    out.extend(['Classwise aggregate interpretation: amplitude-only training does not resolve noise/order failures; ordinary channel/finite RawBoost training reduces FAKE misses with stressed REAL false alarms. The primary consistency improvement is a standard-baseline result; P1 gain augmentation is redundant on identical gain-invariant features. No method is declared universally best.','',
        md_table(['Run','Clean REAL FA','Clean FAKE miss','Mean3-stress REAL FA','Mean3-stress FAKE miss'],risks),''])
    section('12. Scientific Novelty Assessment',md_table(['Candidate','Nearest prior art','Distinct mechanism','Evidence','Limitation','Status'],[
        ['C1','RADAR / RawBoost / standard interaction estimands','No proven new general mechanism','Paired controls and public development','Source-matched confirmation blocked','SUPPORTING_VALIDATION_ENGINEERING'],
        ['C2','F-SAT / PCL-NET / DK-CAST / RawBoost','Evidence-retention weighting is not sufficiently distinguished','Stronger baseline pilot and real gradient path','Proxy validity, seed variability, external evaluation missing','NOVELTY_NOT_ESTABLISHED'],
        ['C3','Selective prediction / matched-cost cascades','NOT_IMPLEMENTED','Prerequisites and reserved calibration','No validated second stage / risk protocol','DEFERRED']])+
        '\n\nInternational methodological novelty, a new local empirical observation, independent confirmation, engineering implementation and reproducibility are different claims. F-SAT constrains adversarial spectral perturbations; PCL-NET applies feature-level paired consistency; DK-CAST already weights/distills distortion-aware supervision. Changing backbone or summing known terms does not establish novelty. The detailed primary-source matrix and method rejection/design gate are PHASE6_NEAREST_PRIOR_ART_MATRIX.md and PHASE6_C2_METHOD_PROPOSAL.md. No new loss was implemented after the novelty-plausibility gate failed.')
    failed=[r for r in experiment_rows if any(s in r['status'] for s in ('FAIL','INTERRUPT'))]
    section('13. Research Failures and Negative Results',
        'The fixed-generator source-matched design lacked support; unknown lineage limits causal claims. Gain-invariant preprocessing can suppress useful cues or shift input distributions. Augmentation may improve misses while increasing clean false alarms. Small data, threshold sensitivity and frozen-backbone restrictions remain material. Teacher runtime/checkpoint-quality gates prevent KD. '
        'Phase6 initially ran cache and fine-tuning TensorFlow processes concurrently; cache reported an allocation failure and the fine-tuning process terminated without a completed result. Sequential execution and microbatching were introduced; native termination without a traceback is not falsely diagnosed as a measured gradient failure. Failed outputs are preserved, never reused as complete caches or published quality evidence.\n\n'+md_table(['Run','Phase','Status','Output','Counts/steps retained'],[[r['experiment_id'],r['phase'],r['status'],r['output'],r['optimizer_steps']] for r in failed])+
        '\n\nResource-stopped but sealed exploratory runs and superseded figure/report versions remain in their historical directories. Their presence does not add independent replication or favorable evidence.')
    section('14. Model and Training Architecture',
        'Source bytes → deterministic official-speaker/class selection → shared seeded waveform views → P0/P1 preprocessing → MobileNetV3Small → LSTM128 → Dense64/dropout → sigmoid FAKE score. Source labels are REAL0/FAKE1. P0 calls production preprocessing; P1 uses per-segment relative power flooring with explicit silent-segment behavior. Both retain the temporal image dimensions. '
        'Head-only training materializes detached frozen-backbone embeddings once in RAM, reusing the same source/condition cells across objectives. BCE supervises each sampled view; equal-two-view control averages two BCE terms; B4 adds0.1 mean squared probability discrepancy. ImageNet weight hash, seed, LR decay, Adam updates, view logs, clean-development-BCE checkpoint selection and reload evaluation are recorded. '
        'Partial fine-tuning uses raw/preprocessed train tensors, calls the backbone inside the tape, reuses the existing tail-unfreeze policy, freezes BN statistics, creates a fresh optimizer after trainability changes, and audits actual variable gradients/updates and checkpoint parity. Frozen embeddings cannot support CNN updates. Microbatching bounds memory; its tiny sanity loss is not development performance.')
    section('15. Statistical Methodology',
        'Each source contributes paired conditions; conditions are not independent recordings. Known official speakers or author source-text clusters define descriptive bootstrap units. Uniform-cluster and recording-weighted estimands differ when group sizes or class support differ. Parent-byte grouping handles only known local variants; it does not recover unknown original utterance ancestry. '
        'Training initialization uncertainty needs independently trained seeds and is not replaced by bootstrap replication. Phase6 is one seed with prospective primary comparison and clean classwise guardrails; secondary comparisons are descriptive, without significance claims or selective unadjusted p-values. The future confirmatory family requires multiplicity control and at least three seeds. '
        'A sigmoid is a bounded score. NLL/Brier/ECE/classwise reliability diagnostics are implemented without fitting calibration; small-bin ECE is unstable at this sample size. A diagnostic EER threshold is not used as an operating policy. Reserved calibration cannot be consumed by a config flag or a proposal alone; confirmation remains disabled.\n\nLocked Phase6 protocol:\n\n```json\n'+json.dumps(p6['protocol'],indent=2)+'\n```')
    reliability=[]
    for r in experiment_rows:
        if r['phase']==6 and isinstance(r.get('declared_metrics'),list):
            clean=next((m for m in r['declared_metrics'] if m.get('condition')=='clean'),{})
            if clean.get('reliability'): reliability.append([r['experiment_id']]+[clean['reliability'][k] for k in ('nll','brier','ece')])
    out.extend(['Clean bounded-score reliability, no calibration fitted:','',md_table(['Run','NLL','Brier','ECE'],reliability),''])
    section('16. Resource and Hardware Inventory',
        'Current measurements and historical timing have different execution boundaries; the laptop host is not automatically the conference paper machine. Physical GPU inventory is not a working CUDA runtime. Existing TensorFlow and Torch CPU environments remain unchanged. Training-state caches and teacher load costs must be counted separately from deployed student inference.\n\n'+
        md_table(['Measured field','Value'],[[k,env[k]] for k in ('platform','cpu_count','gpu_inventory_not_used_by_c1','packages','free_disk_bytes','ffmpeg_version')])+
        '\n\nPhase6 resource gate:\n\n```json\n'+json.dumps(p6['resource'],indent=2)+'\n```\n\nMeasured/extrapolated size alternatives (estimates are not runtime promises):\n\n```json\n'+json.dumps(p6['scaling'],indent=2)+'\n```\n\n'
        'The current cache and each run bind actual elapsed seconds, bytes, process peak RAM and updates. Memory-pressure failure shows why independent jobs must run sequentially on this desktop. No multi-day job, full dataset fine-tuning, teacher install or extra large dataset acquisition is automatic.')
    out.extend(['Current-volume constraints and new full-student warm CPU timings:','',
        '```json\n'+json.dumps(dict(system=p6['system'],efficiency=p6['efficiency']),indent=2)+'\n```',''])
    section('17. Complete File and Module Map',md_table(['Directory / key modules','Responsibility'],[
        ['src/lava/jaes_v2/provenance.py, datasets.py, asv_archive.py','Source identity, UNKNOWN semantics, official protocols and selective extraction'],
        ['transforms.py, controls.py, nuisance_controls.py, source_matching.py','Waveforms, fixed/relative noise, nuisance controls and source-match levels'],
        ['preprocessing.py, c2_student.py, c2_training.py','Versioned policies, reused architecture and preserved Phase5 lifecycle'],
        ['phase6_data.py, rawboost.py, vendor/rawboost/','New speaker-aware selection, finite-view cache, pinned licensed reference adapter'],
        ['phase6_training.py','New baseline objectives, reliability, bounded head training and real partial fine-tuning'],
        ['scripts/jaes_v2/phase6_experiment.py, phase6_report.py','Executable gates, training commands and score-only statistics/figures'],
        ['scripts/jaes_v2/build_master_research_report.py','Cross-phase inventory, metric reconciliation, report and evidence audit'],
        ['configs/jaes_v2/, tests/jaes_v2/','Versioned experiment contracts and regression/unit tests'],
        ['outputs/jaes_v2/phase6/','Selections, caches, failed/completed runs, primary protocol, reports and resources'],
        ['docs/jaes_v2/, papers/jaes_v2/','Research decisions, current master, claim restrictions and paper evidence mapping']]))
    section('18. Evidence Traceability Matrix',md_table(['Claim / numerical source','Cohort/model','Metric','Status','Verified evidence'],
        [['Conference clean performance',r.get('model',UNKNOWN),{k:r.get(k) for k in ('samples','eer','f1','roc_auc')},r['status'],r['source']] for r in inv['conference_baseline']['clean']])+
        '\n\nAll Journal per-condition numerical rows above name or resolve to an executed score CSV in `journal_metrics` in the inventory. Decisive file hashes:\n\n'+
        md_table(['Evidence path','SHA256'],[[p,h] for p,h in inv['reproducibility']['evidence_sha256'].items() if ('phase6/' in p or 'final_integrity' in p or 'foundation.json' in p or 'clean/' in p)]))
    # Retain all executed diagnostic benchmark rows in the self-contained report,
    # not just the conference clean headline figures.
    diagnostic=inv['conference_baseline']['diagnostic_benchmarks']
    out.extend(['### Historical benchmark diagnostic metrics (source-linked)',
        '',md_table(['Model','Condition','n','F1','AUC','EER','REAL FA','FAKE miss','Source'],[
            [r.get('model'),r.get('condition'),r.get('samples'),r.get('f1'),r.get('roc_auc'),r.get('eer'),
             r['fp']/(r['fp']+r['tn']) if r.get('fp',0)+r.get('tn',0) else UNKNOWN,
             r['fn']/(r['fn']+r['tp']) if r.get('fn',0)+r.get('tp',0) else UNKNOWN,r['source']] for r in diagnostic]),''])
    section('19. Remaining Scientific Blockers',md_table(['Category','Missing','Why / resolution','User action'],[[b['category'],b['missing'],b['resolution'],b['user_action']] for b in inv['blockers']]))
    section('20. International Journal Readiness',
        'NOT_JOURNAL_READY as a new methodological paper. The project has stronger software traceability and controlled baseline evidence, but tests, more perturbations or a working gradient path cannot establish international novelty. A suitable submission needs a defensible technical distinction or an appropriately scoped empirical audio-engineering contribution, strong matched controls, external validity, ablations, seed variability and clear operational trade-offs. '
        'The AES official journal site and author guidelines govern scope and publication policy. Current Clarivate indexing/JCR rank and the exact conference extension/copyright eligibility are not independently certified by this code audit. The accepted conference venue, DOI/publication date, transferred rights and all-author extension/disclosure status remain required metadata. An accepted-but-unreleased paper cannot automatically be treated as having no prior-publication obligations. No submission or publication action is performed.')
    section('21. Phase7 Research Roadmap',md_table(['Priority / objective','Prerequisite','Code/data work','Compute basis','Evidence / acceptance gate'],[
        ['1 All-attack controlled scaling','Source/attack support audited; larger budget locked','Increase per-speaker cell counts; preserve train/dev/calibration roles','Actual small-cache/run throughput; scaling estimates above','Both classes/speakers balanced; all declared attacks; no hidden overlap'],
        ['2 Strong online vs finite augmentation','Faithful finite adapter already tested','Online waveform/spectrogram data path; strong RawBoost and equal-view controls','Measure preprocessing and backbone training; no cached-embedding finetune claims','Matched source exposure and disclosed compute'],
        ['3 Partial-backbone learning study','Tiny gradient/resource sanity must pass; compatible isolated runtime','At least three seeds, matched initial state/stage schedule, locked BN policy','Extrapolate real microbatch forward/backward; budget before training','Representation updates, clean-risk tolerance and ablations'],
        ['4 Retention mechanism or rejection','Compare F-SAT/PCL-NET/DK-CAST and validate observable proxy','Specify inputs/estimator/degeneracy guard and falsification','No new-loss experiment until plausibility gate passes','Benefit beyond equal-view standard controls; negative results retained'],
        ['5 Unused external validation and calibration','Freeze corpus role/overlap/primary tests before labels','Choose storage-safe external source; lock calibration separately','Recheck reserve, expanded archive and inference cost','No design use of final confirmation; classwise/seed/group uncertainty'],
        ['6 Journal extension eligibility','Actual conference/publication/copyright facts','Disclosure, citations, substantive contribution mapping','No compute needed','Publisher rules and author rights verified; no promised acceptance']]))
    section('22. Exact Next Commands',
        '### Executed stages\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment select\n.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment preflight\n.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment cache\n.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment train --config configs/jaes_v2/p6_p0_b4_v1.json\n.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment finetune --output-name partial_finetune_sanity_v2\n```\n\n'
        'Existing output identities refuse overwrite. The full registered run list is in the experiment table; these sample stage commands do not authorize rerunning completed training.\n\n### Safe score-only regeneration, fresh destination\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.phase6_report --output outputs/jaes_v2/phase6/reports_another_reproduction\n.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.build_master_research_report --audit-output outputs/jaes_v2/phase6/master_audit_another\n```\n\n'
        '### External runtime preparation / requires compatible GPU\n\n```powershell\npython -m scripts.jaes_v2.phase6_experiment finetune --device gpu --output-name partial_finetune_gpu_sanity_v1\n```\n\n'
        'The GPU command requires an independently provisioned TensorFlow GPU environment, pinned dependencies/data/weights and the same repository contracts. It fails if no GPU is visible; it is a bounded sanity command, not an implemented full-scale GPU training pipeline. Longer fine-tuning and seed schedules require new validated configurations and budgets. No unsupported command-line flag is proposed.\n\n'
        '### Evidence gaps\n\n'+('\n'.join('- '+p for p in inv['reproducibility']['evidence_gaps']) or 'No required structured evidence file is missing from this generated handoff. UNKNOWN provenance and NOT_MEASURED scientific claims remain explicitly unresolved.'))
    return '\n'.join(out)+'\n'


def build(root,audit_output):
    audit_output.mkdir(parents=True,exist_ok=False); inv=collect(root); text=render(inv)
    if len(re.findall(r'^## \d+\.',text,re.M))!=22: raise ValueError('Master report structure incomplete')
    docs=root/'docs/jaes_v2'; target=docs/'LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT.md'; inventory=docs/'LAVA_JOURNAL_V2_RESEARCH_INVENTORY.json'
    target.write_text(text,encoding='utf8'); write_json(inventory,inv)
    report=dict(status='PASS' if not inv['reproducibility']['evidence_gaps'] else 'EVIDENCE_GAPS',sections=22,experiments=len(inv['experiments']),models=len(inv['models']),
        numerical_condition_rows=len(inv['journal_metrics']),benchmark_rows=len(inv['conference_baseline']['clean'])+len(inv['conference_baseline']['diagnostic_benchmarks']),
        evidence_hashes=len(inv['reproducibility']['evidence_sha256']),historical_files_unchanged=inv['reproducibility']['historical_files_unchanged'],
        markdown_sha256=file_hash(target),inventory_sha256=file_hash(inventory),missing_evidence=inv['reproducibility']['evidence_gaps'],neural_inference=False,
        cross_phase_checks=['Dataset/cohort denominators retained','Model strata and checkpoint hashes separate','Fixed score thresholds not called calibrated','Single seed distinct from bootstrap','Head-only distinct from partial fine-tuning','Incomplete runs excluded from quality tables'])
    write_json(audit_output/'validation.json',report); print(json.dumps(report,indent=2)); return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--audit-output',type=Path,required=True); args=parser.parse_args()
    build(Path(__file__).resolve().parents[2],args.audit_output)
