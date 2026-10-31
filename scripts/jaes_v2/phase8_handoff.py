"""Generate readiness from actual tests/integrations, then seal an immutable handoff."""
from pathlib import Path
import argparse,json,shutil,zipfile
from src.lava.jaes_v2.evidence import file_hash,write_json,object_hash,read_csv
from src.lava.jaes_v2.phase8_training import BASE,load_config,verify_tree
from scripts.jaes_v2.phase8_report import collect,table


def evidence(root):
    b=root/BASE;docs=root/'docs/jaes_v2'
    before=json.loads((b/'audit/historical_before.json').read_text())['files']
    changed=[p for p,h in before.items() if not (root/p).is_file() or file_hash(root/p)!=h]
    if changed:raise ValueError('Historical changes: '+str(changed[:5]))
    write_json(b/'audit/historical_after.json',dict(status='PASS',verified_files=len(before),changed=[]))
    tests=json.loads((b/'validation_final/validation.json').read_text())['suites']
    integration=json.loads((b/'audit/resume_integration_v3.json').read_text())
    representation=json.loads((b/'audit/representation_export_seed_checks_v3.json').read_text())
    configs={}
    for p in sorted((root/'configs/jaes_v2/phase8').glob('*.json')):
        c=load_config(root,p);configs[c.experiment_id]=dict(path=p.relative_to(root).as_posix(),hash=object_hash(c.__dict__))
    preflights={p.name:json.loads(p.read_text()) for p in sorted((b/'preflight').glob('*.json'))}
    negative=b/'training/p8_tiny_guarded_infeasible_head_s6107_v3'
    status=json.loads((negative/'status.json').read_text());state=json.loads((negative/'checkpoints'/json.loads((negative/'latest.json').read_text())['checkpoint']/'controller.json').read_text())
    negative_pass=status['status']=='NO_FEASIBLE_CHECKPOINT' and state['finished'] and state['epoch']==1 and not (negative/'best.json').exists()
    checks=dict(regression=all(v['successful'] and not v['skipped'] for v in tests.values()),
        exact_resume=all(x['status']=='PASS' for x in integration.values()),representation=representation['status']=='PASS',
        historical_integrity=not changed,registered_configs=bool(configs),resource_preflight=len(preflights)>=2 and all(p['status']=='PASS' for p in preflights.values()),
        infeasible_guarded_early_stop=negative_pass)
    head_initializations={name:json.loads((b/'training'/f'p8_tiny_{name}_head_s6107_v3/trainable_layers.json').read_text())['initialization_hash'] for name in ('erm','weighted','dro')}
    checks['matched_baseline_initialization']=len(set(head_initializations.values()))==1
    runs=collect(root);resources={r['experiment_id']:r['resources'] for r in runs if 'resources' in r}
    benchmark_dir=b/'benchmark/partial_cpu_v1';verify_tree(benchmark_dir)
    benchmark=json.loads((benchmark_dir/'measurements.json').read_text())
    estimates={}
    for name,p in preflights.items():
        exposure=p['train_sources']*21
        update_projection=benchmark['steady_update_median']*2*p['optimizer_updates']
        preparation_projection=benchmark['preparation_seconds']/8*exposure
        views=(p['development_sources']-p['REAL_only_development'])*21+p['development_sources']*7
        forward_projection=benchmark['steady_forward_median']*views
        transformed=resources['p8_tiny_erm_partial_s6107_v3']['input_timing']['transform_seconds']/14*p['development_sources']
        estimates[name]=dict(status='EXTRAPOLATED_NOT_FULL_RUN_MEASUREMENT',training_update_seconds=update_projection,
            uncached_train_preparation_seconds=preparation_projection,validation_forward_seconds=forward_projection,
            development_transform_seconds=transformed,total_seconds=update_projection+preparation_projection+forward_projection+transformed,
            assumptions='ERM partial only; microbatch1; measured8source steady update scaled to16; warmup approximated by partial backward cost; no thermal/IO/long-run guarantee')
    # The validation traversal changed; match predictions by source/condition.
    parity={}
    for name in ('erm_head_s6107','erm_partial_s6107','weighted_head_s6107','dro_head_s6107'):
        a=b/'training'/f'p8_tiny_{name}_v2';z=b/'training'/f'p8_tiny_{name}_v3'
        old={(r['sample_id'],r['condition_id']):float(r['raw_score']) for r in read_csv(a/'scores.csv')}
        new={(r['sample_id'],r['condition_id']):float(r['raw_score']) for r in read_csv(z/'scores.csv')}
        if set(old)!=set(new):raise ValueError('Validation prediction coverage differs')
        residual=max(abs(old[k]-new[k]) for k in old)
        parity[name]=dict(max_score_residual=residual,v2_seconds=resources[a.name]['elapsed_seconds'],v3_seconds=resources[z.name]['elapsed_seconds'],
            scope='Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication')
        if residual>1e-6:raise ValueError('Validation optimization changed scores')
    checks['validation_optimization_parity']=True
    ready='YES' if all(checks.values()) else 'NO'
    command='.\\.venv\\Scripts\\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json'
    result=dict(FULL_TRAINING_CODE_READY=ready,NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT='NO',checks=checks,
        tests={k:{a:v for a,v in x.items() if a!='failure_details'} for k,x in tests.items()},integration=integration,representation=representation,
        historical_files_unchanged=len(before),registered_configurations=len(configs),actual_run_entries=len(runs),CPU_benchmark=benchmark,CPU_time_extrapolation=estimates,
        training_config_hashes=configs,validation_optimization=parity,resources=dict(measured=resources,preflight=preflights,
            free_D_bytes=shutil.disk_usage(root).free,free_C_bytes=shutil.disk_usage(Path.home().anchor).free,
            GPU='NOT_MEASURED_NOT_VERIFIED',long_training='NOT_EXECUTED',disk_reserve_gib=15),
        full_training_commands=[command],known_blockers=['M3 entry gate failed: weighted augmentation/worst-of-two reductions; no justified forensic target',
            'GPU runtime unverified','Longer full-data/three-seed representation quality not executed','Uniform-q matched group-sampling control not registered',
            'Original source/generator checkpoint lineage unknown','External/calibration protocols not locked','Conference venue/rights and authenticated current journal indexing unresolved'])
    target=b/'audit/readiness_evidence.json'
    if target.exists():raise FileExistsError(target)
    write_json(target,result)
    readiness=f'''# Full training readiness — evidence generated

FULL_TRAINING_CODE_READY: **{ready}** (CPU, tested TF2.15 environment).

NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: **NO**.

Gate1 IMPLEMENTATION_READY: {'YES' if ready=='YES' else 'see failures'}. Gate2 FULL_TRAINING_CODE_READY: {ready}. Gate3 SCIENTIFIC_EXPERIMENT_READY for a novel C2: NO. No full-data run or GPU training is implied.

```json
{json.dumps(checks,indent=2)}
```

Actual regression:

{table(result['tests'].values(),['run','failures','errors','skipped','successful'])}

CPU fresh-process resume compares final tensor bytes, Adam bytes, source/view order, loss/LR journals, counters, history and selected-model scores. Both head and warmup→partial schedules require exact hashes with tolerance0. Checkpoints occur only at optimizer boundaries; no claim of restoring half an accumulated microbatch. The full CNN export was checked against recorded scores within1e-6, warmup backbone/BN values were unchanged and partial kernels changed. Three seeds have distinct initialization hashes. A guarded tiny run actually early-stopped with no feasible export.

Frozen BN stays unchanged. Warmup masks backbone gradients while pre-registering tail slots; this is explicitly different from toggling trainable flags each epoch or resetting Adam at the transition. Both stages use one common scalar LR; no discriminative-LR implementation is claimed. Head-only can use detached embeddings; partial always runs the differentiable CNN on P0 tensors. Finite banks are not online augmentation.

Resource preflight verifies data byte hashes/known partition separation, complete lifecycle update capacity and conservative retained-checkpoint storage against15GiB reserve. Runtime guards check disk, peak process RAM, elapsed invocation time, finite gradients and predictions. Model selection and fixed threshold are explicit. Pipeline supports medium and full-foundation data without constructing an unbudgeted Mel cache. GPU source portability is prepared but its runtime, mixed precision, long-run convergence and arbitrary-shift risk remain unverified.

Verified historical files: {len(before)}; accepted DOCX remains SHA91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073. {len(configs)} configuration identities are registered, not trained models. Actual run entries include first-version pilots, optimization repeats and interrupted counterparts; they must not inflate independent-seed counts. Missing metrics are null/absent, never zero.

Detailed machine-readable evidence: outputs/jaes_v2/phase8/audit/readiness_evidence.json. Standard baseline launch commands are in FULL_TRAINING_LAUNCH_PACKAGE; scientific blockers remain in NOVELTY_DECISION and METHOD_COMPARISON_PLAN.
'''
    (docs/'PHASE8_FULL_TRAINING_READINESS.md').write_text(readiness,encoding='utf-8')
    measured=[]
    for name,r in resources.items():
        measured.append(dict(experiment_id=name,invocation_seconds=r['elapsed_seconds'],updates_seconds=r['timing']['update_seconds'],validation_seconds=r['timing']['validation_seconds'],peak_MiB=r['peak_process_mib'],checkpoint_bytes=r['retained_checkpoint_bytes']))
    resource=f'''# Full-training resource plan

Observed host:12logicalCPUs, RTX3050Laptop4GiB/driver595.71, native Windows TF2.15CPU. Current D free at evidence generation={result['resources']['free_D_bytes']}bytes; C={result['resources']['free_C_bytes']}bytes. The CLI redirects process temp and plotting cache to D, preserves15GiB reserve and uses the existing single-heavy-job OS lock. No dependency/environment installation or historical cleanup occurred.

## Measured tiny invocations

{table(measured,['experiment_id','invocation_seconds','updates_seconds','validation_seconds','peak_MiB','checkpoint_bytes'])}

Resume rows measure the resumed invocation only, excluding earlier process time. Source-version repeats and counterpart pairs are not independent quality replications. OS lifetime peak includes allocator/activation effects but does not isolate activations. GPU peak VRAM/time is NOT_MEASURED; nvidia-smi free memory is inventory only. Step timing includes gradient/optimizer and input preparation; validation timing includes its transforms/model/metrics. Input decode, transforms, Mel, compiled forwards, metrics, checkpoint and startup components are separately retained where instrumented. Components overlap and must not be summed twice. Figure generation is in the separate model-free report process, never inside training timing.

Original extracted audio uses3,178,404,021bytes; user LA.zip7,646,913,333bytes is retained, not copied or re-extracted. The existing Phase7 cache contains6,416embeddings /88,694,784rawfloat bytes and cost1776.57s to prepare. Head times exclude that shared historical cost. New partial training holds at most8P0 tensors plus one source waveform bank in RAM; it does not create a multi-GB image cache. Each full P0 tensor is6×224×224×3float32 (~3.45MiB);8entries about27.6MiB before framework activations. Final model exports are approximately6MB each, while Adam checkpoints are larger; measured files, not model size, define actual storage.

Validation optimization comparisons:

```json
{json.dumps(parity,indent=2)}
```

Source-first evaluation reuses one deterministic waveform bank instead of regenerating all codec chains for each requested condition. This is fixed-view caching within a traversal, not fresh online augmentation or a performance result for a new model. Prediction parity is checked by source/condition, independent of CSV row order.

## Warmed CPU benchmark and illustrative time projection

```json
{json.dumps(benchmark,indent=2)}
```

```json
{json.dumps(estimates,indent=2)}
```

The benchmark performs6effective8-source partial-CNN updates; first tracing update is separated from5steady repetitions. Full forward timing similarly excludes its first trace. Time estimates apply only to the specified ERM21epoch schedule, separately scaling training, preparation, clean-epoch/final-view forwards and development transforms. Reused tiny inputs, cache behavior, thermal throttling and large-corpus I/O limit the extrapolation; these are not measured full epoch/run times. Channel/DRO/RawBoost require their own resource calibration and view accounting. Invocation guards remain authoritative.

## Preflight projections (not full-job measurements)

{table([dict(config=p['config_hash'],train=p['train_sources'],development=p['development_sources'],updates=p['optimizer_updates'],retained_bytes=p['projected_retained_bytes'],status=p['status']) for p in preflights.values()],['config','train','development','updates','retained_bytes','status'])}

Retained-size projection assumes24MiB per boundary snapshot plus8MiB per possible epoch export and256MiB fixed overhead; checkpoint cadence500updates plus epoch boundaries. Each job's preflight is independent; sum projections for the intended seed/method matrix and recheck remaining disk before each launch. Do not multiply this single-run PASS into a guarantee that all families fit. No existing checkpoint is automatically deleted. Full-foundation21epochs have approximately33,327updates at effective batch16, before early stopping; medium21epochs840updates. Natural final partial batches use their actual denominator. Group-balanced/oversampling changes source exposure but not the unique recording count.

For a coarse CPU planning estimate, scale the measured tiny **update-path seconds / actual source exposures** by the planned source exposures, and separately add clean-epoch validation plus full final evaluation. This includes first-trace overhead and small-cohort cache effects and is not a reliable long-run ETA. Full-foundation final evaluation has17,503×7=122,521views, whereas a tiny final has14×7=98; epoch primary validation is an additional cost. Long-run epoch time, complete multi-seed wall time, GPU speedup and sustained memory remain NOT_MEASURED. Use the first medium run and journal-based monitor to refine estimates. Jobs may need repeated24h invocations with resume; final evaluation restarts if interrupted. Three-seed storage must be budgeted explicitly and jobs run sequentially.

Do not copy data into WSL or install CUDA/WSL in this phase. See GPU_TRAINING_GUIDE for the isolated portability path. Disk/RAM/NaN/power recovery and author commands are specified in FULL_TRAINING_LAUNCH_PACKAGE.
'''
    (docs/'PHASE8_FULL_TRAINING_RESOURCE_PLAN.md').write_text(resource,encoding='utf-8')
    experiments='# Phase8 actual experiment results\n\n'+table([dict(experiment_id=r['experiment_id'],status=r['status'],scope=r['config']['training_scope'],steps=r.get('steps','UNKNOWN')) for r in runs],['experiment_id','status','scope','steps'])
    experiments+='\n\nAll completed quality rows use12train and14development sources (12primary A,2REAL-only B), A01–A06 seen attacks. These are lifecycle/objective pilots, not a generalization study. Original implementation, exact-resume pairs and optimized-validation reruns are not independent scientific replications. M1/M2 optimize standard losses; no M3 run exists. Full medium/foundation training, independent calibration/confirmation, GPU and cascade are NOT_EXECUTED.\n\n'
    experiments+='Guarded negative integration: NO_FEASIBLE_CHECKPOINT, epoch1 of a declared5epoch maximum, patience1; no export fallback. Historical ordinary consistency and RawBoost trade-offs remain unchanged.\n\n'
    experiments+='CPU exact-resume evidence:\n\n```json\n'+json.dumps(integration,indent=2)+'\n```\n\nSelected-model classwise values and measured costs are in the model-free per_run.csv/classwise_conditions.json; each run also retains all seen-attack metrics with one-class AUC/REAL-FA nulls. No posthoc threshold tuning or claim of superiority from tiny scores.\n'
    (docs/'PHASE8_EXPERIMENT_RESULTS.md').write_text(experiments,encoding='utf-8')
    print(json.dumps(dict(ready=ready,checks=checks,actual_entries=len(runs)),indent=2))


def finish(root):
    b=root/BASE;docs=root/'docs/jaes_v2'
    initial=json.loads((b/'audit/historical_before.json').read_text())['files']
    changed=[p for p,h in initial.items() if file_hash(root/p)!=h]
    if changed:raise ValueError(changed[:5])
    a=json.loads((b/'master_final/validation.json').read_text());z=json.loads((b/'master_reproduced/validation.json').read_text())
    if a!=z:raise ValueError('Master reproduction differs')
    for name in ('LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE8.md','LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE8.json'):
        target=docs/name
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(b/'master_final'/name,target)
    differences=[]
    for p in (b/'statistics_final').rglob('*'):
        if p.is_file() and file_hash(p)!=file_hash(b/'statistics_final_reproduced'/p.relative_to(b/'statistics_final')):differences.append(str(p))
    if differences:raise ValueError('Statistics/figures reproduction differs')
    seals={r['experiment_id']:file_hash(root/r['output']/'seal.json') for r in collect(root) if 'seal_sha256' in r}
    files=[]
    for directory in ('src/lava/jaes_v2','scripts/jaes_v2','configs/jaes_v2','tests/jaes_v2','docs/jaes_v2'):
        files.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    snapshot=b/'audit/source_snapshot_final.zip'
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):archive.write(p,p.relative_to(root).as_posix())
    hashes={p.relative_to(root).as_posix():file_hash(p) for p in b.rglob('*') if p.is_file() and p.name not in ('all_artifact_hashes.json','final_integrity.json')}
    write_json(b/'audit/all_artifact_hashes.json',hashes)
    result=dict(status='PASS',historical_files_unchanged=len(initial),accepted_manuscript_sha256=file_hash(root/'papers/PaperID_804_final.docx'),
        report=a,completed_run_seals=seals,source_snapshot_sha256=file_hash(snapshot),all_artifact_hashes_sha256=file_hash(b/'audit/all_artifact_hashes.json'),
        model_free_reproduction='BYTE_IDENTICAL_TABLES_JSON_PNG_SVG_MASTER',FULL_TRAINING_CODE_READY=json.loads((b/'audit/readiness_evidence.json').read_text())['FULL_TRAINING_CODE_READY'],
        NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT='NO',calibration_accessed=False,evaluation_accessed=False,full_training_executed=False,GPU_runtime='NOT_VERIFIED',
        free_D_bytes=shutil.disk_usage(root).free,free_C_bytes=shutil.disk_usage(Path.home().anchor).free)
    if result['free_D_bytes']<15*1024**3:raise ValueError('Disk reserve')
    out=b/'audit/final_integrity.json'
    if out.exists():raise FileExistsError(out)
    write_json(out,result);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('evidence','finish'));args=p.parse_args();root=Path(__file__).resolve().parents[2]
    (evidence if args.mode=='evidence' else finish)(root)

if __name__=='__main__':main()
