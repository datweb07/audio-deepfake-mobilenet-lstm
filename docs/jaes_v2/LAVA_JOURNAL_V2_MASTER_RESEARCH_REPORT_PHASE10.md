# LAVA Journal V2 — Master research report, Phase 10

08–09/11/2026, Asia/Novosibirsk. Deadline 20/12/2026. **CPU medium qualification passed. Medium-plus and foundation remain NO-GO.**

The 22:48 recheck passed preflight with 4.33 GiB available RAM and 7.36 GiB available commit. The real ASVspoof numerical probe then completed all nine fixtures in 43.23 s, with 804 MiB peak RSS. Both warm-up fixtures and five of seven partial fine-tuning fixtures passed. The REAL noise5 case exceeded the limits for gradients and Adam weight updates. The mixed REAL clean and FAKE noise5 case also exceeded both limits. Across the run, maximum absolute gradient difference was 1.3512e-4, maximum absolute Adam update difference was 8.0310e-5, maximum absolute score difference was 1.1921e-7, and frozen tensors matched. The optimizer-slot checks passed their separate tolerances. The result rejects interchangeability of the current segmented gradient with the full-sequence reference under the locked criteria. A targeted trace on the failing REAL noise5 clip found one Conv_1 activation entering different hard-swish derivative branches when batching all six frames versus processing one at a time. The preactivation values were −2.9999988 and −3.0000014. This establishes a concrete branch-change mechanism for the batching comparison. Attribution of all per-tensor errors to that one crossing remains open. [Resource snapshot](../../outputs/jaes_v2/phase10/resources_ram_recheck_v1.json) · [Per-tensor evidence](../../outputs/jaes_v2/phase10/numeric_parity.json) · [Boundary diagnostic](../../outputs/jaes_v2/phase10/numeric_boundary_diagnostic_v1.json)

Resource recovery rerun on 08/11. The author recovered C to roughly 6.71 GiB. Phase 10 CLI now pins OpenBLAS and OMP to one thread before NumPy import and includes that policy in the runtime contract. Observed CLI private commit fell from about 782 to 76 MiB. An early rerun snapshot measured 2.10 GiB free RAM and 2.77 GiB free commit. A subsequent numerical attempt passed startup resource checks and loaded TensorFlow and MobileNetV3, then stopped inside the fixture loop at the unchanged 3 GiB commit floor. No complete numerical acceptance report was produced. The original regression suites passed again (164 JAES, 40 selected legacy), followed by all six focused Phase 10 tests after the startup change. [Recovery sample](../../outputs/jaes_v2/phase10/resources_recovery_pinned_v2.json) · [Recovery regression](../../outputs/jaes_v2/phase10/regression_recovery_v1/validation.json)

The v2 backend `FULL_SHAPE_RECOMPUTE` retains the reference CNN batch shape in fine-tuning, rematerializing it for backpropagation and leaving BatchNorm frozen. It uses a fresh qualification ID. The unchanged acceptance limits passed on all nine real fixtures in 38.14 s. The v2 lifecycle passed all 17 logical comparisons over nine optimizer updates with a verified restart at update five. The two exported `model.weights.h5` members matched byte for byte. Archive metadata contains different save timestamps, and the serialized config contains three process-specific Keras initializer IDs. These differences explain the observed member-level serialization differences. The tiny cohort triggered a prediction-collapse warning, so lifecycle reproducibility provides engineering evidence only. [Gradient v2](../../outputs/jaes_v2/phase10/numeric_parity_v2.json) · [Exact resume v2](../../outputs/jaes_v2/phase10/lifecycle_resume_v2.json)

## Conference and historical evidence

The accepted conference manuscript benchmarked MobileNetV3, ShuffleNetV2, MnasNet, EfficientNet, RawNet2 and AASIST. The six artifacts differ in initialization, input contract and provenance. ShuffleNetV2 had clean ROC-AUC 0.9929 and EER 1.46% on 2,737 checksum-group-disjoint recordings. MobileNetV3 measured 43.81 ms end to end on the conference desktop workload. The 100-recording stress diagnostic is exploratory. Conference results are separate from the ASVspoof development experiments. [Accepted manuscript](../../papers/PaperID_804_final.docx) · [Phase 8 historical inventory](LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE8.json)

| Phase | Completed scope |
| --- | --- |
| 1 | Ordered noise/Opus paired-source exploratory pilots. |
| 2 | Noise, headroom, PCM and timing controls. |
| 3 | MLAAD-tiny development pilot with two frozen detectors and limited provenance. |
| 4 | Matching and nuisance audit. Verified parallel generator lineage remained insufficient. |
| 5 | ASVspoof 2019 LA foundation and preprocessing pilots. Calibration set aside. |
| 6 | Matched baselines, equal-view controls, finite RawBoost and partial fine-tune sanity. |
| 7 | A/B and all-attack cohorts. Three seeds on historical frozen protocol. Consistency missed the guard in all three blocks. |
| 8 | Weighted BCE and Group-DRO control implementations, tiny pilots, and two sealed 840-update medium runs. |
| 9 | Two-case gradient spot check, four-view inference spot check, medium-plus manifest and draft. |

Both medium models used the same 640 training sources and 368 development sources. ERM kept standard gradients. Channel recovered from OOM and changed arithmetic at steps 426 and 430. Both completed 21 epochs and used fixed 0.5. A stress ROC-AUC .62112→.80841, B clean FA 19/144→30/144, A05 miss 40/54→47/54 and A06 19/54→37/54. This is an exploratory paired description. [Phase 9 master](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE9.md) · [Phase 8 medium review](PHASE8_MEDIUM_TRAINING_SCIENTIFIC_REVIEW.md)

## Engine qualification and evidence classification

| Component | Status | Evidence and limits |
| --- | --- | --- |
| Fixed engine | INTEGRATION_TESTED v2 | Full-shape recompute during partial fine-tuning, head-only warmup, fixed Adam lifecycle, single-thread deterministic CPU, stateless masks and seeded order. Reuses dataset, P0 inputs, model, objective and checkpoint primitives. |
| Numerical parity | INTEGRATION_TESTED, PASS v2 | 9/9 real fixtures passed the registered limits. The two v1 failures remain archived. |
| Exact resume | INTEGRATION_TESTED, PASS v2 | Exact model, optimizer, cursor, journal, history and scores for 9 updates with restart at 5. Atomic committed pointer and strict contract reject corruption and config, model, dataset, backend, RNG, batch, preprocessing or source mismatches. The archive differs in serialization metadata. |
| Validation and selection | INTEGRATION_TESTED on sealed historical scores | A=224 clean rows, B=1,040 rows with A clean/stress and B clean, C=2,576 rows across seven conditions. All metadata joined to manifest before scoring diagnostics. Matched reference, exact counts, fixed threshold .5, absolute A clean caps and registered risk deltas. The historical channel fails B clean, A05 and A06 in retrospective selection. A scientifically matched ERM reference still needs to be sealed from prospective training. |
| Full validation numerical parity | INTEGRATION_TESTED, PASS v2 | 2,576 matched score pairs on 368 sources with the sealed ERM model. Max score drift 6.2585e-6, mean 1.9577e-7, one near-threshold case and zero decision flips. ROC-AUC, PR-AUC, EER and group counts matched. Measured 789.08 s for paired traversal, 224.08 s standard forward and 243.60 s segmentwise forward. The segmented inference path was slower on this workload. |
| Monitoring and safety | IMPLEMENTED, UNIT_TESTED for gates | Atomic status reports current update, losses, norm, LR, epoch clean metrics, Tier B counts, current/best checkpoint, resources and measured train-rate estimate. ETA excludes future validation and restart overhead. Rejects nonfinite gradients/weights and zero or exploding updates. Two collapse epochs with at least three completed evaluation epochs raise warning. Update longer than 300 s pauses at a checkpoint if resource-safe. Disk/commit/RAM guard stops at the previous committed pointer if a new write would be unsafe. Heavy end-to-end failure injection and export remain unqualified. |
| 100-update pilot | BENCHMARKED, PASS | Profile `p10_engine_100_channel_s9107_v2`, original 640 train, split at 55. Both final states and 2,576 selected-model score rows matched exactly. All 17 logical comparisons passed. |

[Engine](../../src/lava/jaes_v2/phase10_training.py) · [Acceptance config](../../configs/jaes_v2/phase10/qualification_v2.json) · [Validation v2](../../outputs/jaes_v2/phase10/validation_parity_v2.json) · [Score replay](../../outputs/jaes_v2/phase10/historical_scores_final_v1.json) · [Regression v2](../../outputs/jaes_v2/phase10/regression_ram_recheck_v2/validation.json)

## Resources, order and launch decision

RTX 3050 Laptop has 4 GiB VRAM. Native Python 3.11.9, TF/Keras 2.15.0, NumPy 1.24.3 and SoundFile 0.12.1. The measured training path is CPU. Uninterrupted pilot took 698.51 s for training updates, 891.88 s for validation, 10.33 s for startup, 2.76 s for checkpoint writes and 0.44 s for export. Peak RSS was 724.37 MiB and checkpoint bytes were 116,185,874. The restart leg recorded 11.82 s startup and restore together. Minimum sampled system commit headroom across the three pilot processes was approximately 5.14 GiB. These are measured lifecycle costs for the bounded workload. Medium is GO on CPU subject to current preflight. Medium-plus and foundation need separate evidence. [Pilot comparison](../../outputs/jaes_v2/phase10/benchmark_resume_v2.json) · [Resource evidence and owner action](PHASE10_WINDOWS_STORAGE_ACTION.md)

Scientific launch adds a narrow config dispatch and a different completion label to the pilot engine. The gradient kernel, data preparation, update loop, checkpoint serialization and selection remain the same. The launcher verifies the two pilot seals, the benchmark hash, the qualified dependencies and the released adapter hash. Each scientific checkpoint contract includes the registry hash. This release was checked through the prospective preflight. The scientific runs themselves remain for the author. [Registry](../../configs/jaes_v2/phase10/prospective_medium_v1.json) · [Author commands](PHASE10_AUTHOR_TRAINING_COMMANDS.md)

| Priority | Data and configuration | Gate |
| --- | --- | --- |
| 1 | MobileNetV3Small–LSTM128 CPU, 12-source lifecycle train and real ASVspoof fixture views | Completed. Numerical parity, exact resume and full development parity passed. |
| 2 | 100-update medium channel qualification with uninterrupted and resumed comparator | Completed. Resource instrumentation retained per invocation. |
| 3 | Medium ERM/channel, 640 train, same 368 development, new seeds 9107/9207/9307 | Requires qualification PASS. Lock source order, initialization, preprocessing, optimizer, batch, budget, backend and selection. Only augmentation varies. Keep each failed guard result. |
| 4 | Medium-plus 2,560 balanced train, same 368 development | At least one technically valid matched pair plus meaningful scaling question and measured cost. |
| 5 | Foundation 25,380 train, 17,503 allowed development | Separate resource and scientific decision. 7,341 calibration entries reserved in historical metadata. |
| Conditional | ShuffleNet transfer, then provenance-audited RawNet2/AASIST reference | After replication and overlap audit. |

No new datasets acquired. Known speaker disjointness and byte identity do not establish global parent or generator independence. [Dataset inventory](PHASE9_DATASET_ACQUISITION_DECISION.md) · [Audit commands](PHASE10_AUTHOR_TRAINING_COMMANDS.md)

## C2 research lock and contribution audit

Precommit the following screen. A unit is an aligned source and generator-version pair with a verified lineage. Fix a forensic descriptor independent of classifier score, channel seeds, common source, gain, alignment and bitrate. Use Phase 9 projected cue survival only for pairs whose cue norm exceeds a preregistered measurement floor. Report the paired difference between noise→Opus and Opus→noise, with uncertainty grouped by source and generator. Require identity/lossless stability, explicit cue knockout, matched-band/energy sham and scrambled-pair controls before any detector loss. Blind the detector readout to cue selection.

| Explanation | Discriminating outcome |
| --- | --- |
| Evidence destruction | Cue survival and aligned intervention response both deteriorate under order after nuisance controls. |
| Preprocessing/score drift | Cue remains measurable while ranking or scores change across fixed preprocessing controls. Threshold stays .5 in historical and prospective primary reporting. |
| Corpus/attack shortcut | Apparent cue specificity falls away under matched nuisance controls or held-out source/generator. |
| Training trade-off | Paired methods differ in A/B risk even at matched cue survival and fixed numerical protocol. |

The current corpus lacks verified parallel lineage for this mechanism. `C2_CUE_IDENTIFICATION_BLOCKED_ON_DATA`. Cap this screening to documentation and data feasibility until 20/11. If lineage and a mechanism-distinguishing minimum experiment are still absent then, stop C2 method development for the deadline. F-SAT, paired codec consistency, DK-CAST, SONAR, TFCL and prior shortcut interventions cover nearby mechanisms. The candidate distinction remains a falsifiable hypothesis. [Prior-art assessment](PHASE9_NOVELTY_COMPARISON.md)

| Candidate | New beyond conference | New beyond nearest international work | Evidence | Remaining experiment | Status |
| --- | --- | --- | --- | --- | --- |
| C1 | Ordered condition and class/attack trade-off on speaker-disjoint ASV development | Distinct cohort and diagnostic detail. General novelty pending | Two sealed single-seed runs and order controls | Matched replication and source-aware uncertainty | SUPPORTING DEVELOPMENT |
| C2 | Source-controlled cue survival | Generator-specific intervention response independent of model is hypothesized | No verified parallel lineage | Identity, sham, knockout, nuisance and held-out tests | NOT_ESTABLISHED / BLOCKED |
| C3 | Selective risk vs coverage and cost | Standard cascades and abstention cover the base design | No locked calibration or evaluation | Risk coverage and matched-cost controls | DEFERRED |

## Deadline and author actions

The English Phase 9 draft remains the active evidence-limited manuscript. Phase 10 engineering evidence can enter reproducibility after qualification. By 20/11 decide whether resources support prospective replication. By 02/12 assess which matched comparisons completed. Use 03–15/12 for analysis and manuscript completion within actual evidence, then 16–20/12 for QA, overlap disclosure and rights review. Three established contributions by the deadline are at risk. A paper centered on one strong method plus protocol and deployment evidence is a potential framing only if the method passes its scientific gate. Venue, prior publication and copyright still require confirmation before submission. [Draft](../../papers/jaes_v2/journal_draft/PHASE9_JOURNAL_DRAFT.md) · [Bàn giao](PHASE10_EXECUTIVE_HANDOFF.md) · [Hashes and decisions](LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE10.json)
