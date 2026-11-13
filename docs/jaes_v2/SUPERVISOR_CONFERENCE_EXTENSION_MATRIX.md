# Conference-to-journal extension audit

The accepted manuscript and final reviewer response are the historical sources. The journal estimates a paired channel-view intervention using six development checkpoints and the original four-model locked evaluation. The same architecture family and standard mathematical definitions appear in both studies.

| Dimension | Accepted LAVA conference study | Journal V4 | Added evidence | Remaining limitation |
| --- | --- | --- | --- | --- |
| Objective | Benchmark six available artifacts for offline deployment trade-offs | Estimate the effect of the training-view policy | ERM versus channel in paired blocks | One detector architecture in the intervention |
| Research questions | Comparative accuracy, stress and efficiency | RQ1 operating risk, RQ2 heterogeneity, RQ3 order | Explicit endpoint and interpretation per question | Within-release scope |
| Data | Internal collection with incomplete source lineage | ASVspoof 2019 LA with official attack and speaker IDs | Audited train, development and held-out identities | Original-utterance lineage remains incomplete |
| Training | Mixed ImageNet, scratch, external pretraining and warm-up artifact | Same model, optimizer, source order, budget and checkpoint rule within seed | Policy is the registered intervention | Realized clean exposure and selected prefixes differ |
| Architectures | MobileNetV3Small, ShuffleNetV2, MnasNet and EfficientNet-B0 with LSTM, plus RawNet2 and AASIST | MobileNetV3Small–LSTM128 across eight development fits | Eliminates backbone variation from the matched comparison | Architecture transfer unevaluated |
| Benchmark | Clean 2,737 and stratified 100-source stress diagnostic | Development A/B and locked 1,088-source held-out | Dedicated cohorts and support | Class prevalence set by cohort construction |
| Channels | Nine noise, codec and simulated replay conditions | Registered noise, gain and codec conditions plus two ordered chains | Direct reversal with nominal settings and shared noise | Limited codec and noise family |
| Source linkage | Paired clean correctness, common 100-source stress panel | Per-recording arm and order transitions | Shared errors, corrected and newly harmed sources | Views of a source are dependent |
| Attack diagnostics | No reliable generator identity in the internal benchmark | A05/A06 and A17–A19 errors by condition with paired recovery | Attack-ID-specific evidence | IDs do not establish independent generator families |
| Statistics | McNemar with Holm, stratified bootstrap | Speaker-cluster contrasts conditional on each seed | Cluster unit reflects speaker/view dependence | Pointwise attack intervals, no multiplicity claim |
| Seed replication | One artifact per detector | Three paired development initializations | Repeated direction of stress AUC | No population stability estimate |
| Held-out | Checksum-disjoint internal test split | Locked, speaker-disjoint, same-release evaluation | 17,408 recorded predictions and replayed statistics | Cross-corpus confirmation open |
| Efficiency | Historical desktop timing across six artifacts | Training workload and bounded CPU inference for the selected architecture | Parameter partition and stage-separated latency | No mobile, batch-throughput or streaming benchmark |
| Reproducibility | Preprocessing and provenance caveats in text and supplement | Contract, seals, score grid, verification, claim map | Recalculable score-level diagnostics | Source audio governed by its license |

## Overlap review

- Paragraphs. Both texts describe log-Mel inputs, lightweight CNN–LSTM and threshold errors. V4 methods describe the registered ASVspoof intervention and are rewritten around that design. The study framing, estimand and source-level results are distinct.
- Tables and results. Conference tables compare six historical artifacts. V4 tables report independent run artifacts and cohort-specific endpoints. The small comparison table uses historical counts to identify the prior scope.
- Figures and captions. V4 figures derive from the registered score grids, journals and diagnostic outputs. No conference figure is inserted.
- Equations. Shared definitions of binary classification, ROC-AUC and error rates are standard. V4 adds paired order and policy estimands. Equations are not asserted as inventions.
- Claims. The conference already reports robustness and paired inference. The journal extends these to a controlled training policy, paired source transitions, speaker resampling and locked within-release evaluation. A page-count difference is not a quantitative reuse estimate.

## Draft cover-letter statement

This submission extends our accepted LAVA conference study, which compared six heterogeneous detector artifacts in offline clean and stress benchmarks. We add three matched MobileNetV3Small–LSTM development comparisons, a locked within-release evaluation, and source-linked analyses of classwise and attack-level errors under ordered noise and Opus processing. We disclose the related manuscript and its publication status separately and will provide its full bibliographic and rights information when verified.

Author to confirm the proceedings and publication status, citation and DOI if assigned, copyright agreement, and any permission requirements before submission. [Historical extension checklist](../../papers/jaes_v2/jaes_manuscript/CONFERENCE_EXTENSION_DISCLOSURE.md)

V7 adds a fixed-epoch size comparison on the shared development cohort. The larger seed-9207 pair doubles the source count and optimizer budget. The original locked evaluation retains four 640-recording models. The new comparisons use journal evidence and the conference already included aggregate robustness on the fixed 100-recording stress panel.
