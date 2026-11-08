# LAVA Journal V2 — Master research report, Phase 9

08/11/2026, Asia/Novosibirsk. Deadline nghiên cứu 20/12/2026. Bản hội nghị được kiểm tra ở SHA-256 `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`. Master Phase 8 SHA-256 `633572c15f3acd8df50d393e1691c4b905bb7222d165344144a9d5370cc39773`. [Conference](../../papers/PaperID_804_final.docx) · [Phase 8 sealed report](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE8.md)

## Research trajectory

| Phase | Evidence and unresolved scope |
| --- | --- |
| Conference | Six heterogeneous offline artifacts, 2,737 checksum-group-disjoint test recordings, 100 fixed stress diagnostics. ShuffleNetV2 clean AUC 0.9929, EER 1.46%. MobileNetV3 desktop end-to-end 43.81 ms. Speaker, source and generator identities in assembled corpus unknown. |
| 1 | Three paired-source noise/Opus order pilots and parent bootstrap, nuisance remains. |
| 2 | Relative/fixed noise, headroom, PCM and delay controls separate plausible mechanisms. |
| 3 | Frozen artifacts on an MLAAD-tiny development subset, unknown checkpoint exposure. |
| 4 | Source matching audit on two backbones, insufficient verified generator pairs. |
| 5 | ASVspoof 2019 LA foundation and head-only preprocessing pilots, gain invariance/accuracy trade-off. Calibration reserved. |
| 6 | Matched baseline and equal-view controls, finite RawBoost and partial fine-tuning sanity. |
| 7 | All-attack A/B cohorts, three seeds for frozen baselines. Consistency minus channel stress miss mean −0.00397, SD 0.08174, 0/3 guard passes. In one 40-update frozen/partial comparison clean AUC 0.7508→0.8104, stress miss 0.6756→0.7173. |
| 8 | Weighted BCE and Group-DRO controls, 18 tiny attempts, 17 evaluated and one infeasible. Two 840-update medium runs sealed. |
| 9 | Two-case gradient audit, four-view timing and score parity, relative guard unit tests, 2,560-source medium-plus manifest and read-only full data preflight. No registered training. |

[Historical master and exact artifact links](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_POST_TRAINING.md) · [Machine-readable inventory](LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE9.json)

## Phase 8 re-audit and interpretation

The re-audit checked 993 ERM and 1,181 channel seal entries, 43/104 checkpoint journals, matched train/development audio and fingerprints, and recomputed outputs from 2,576 scores per run. Both runs used 640 training sources, 224 A sources (112/class) and 144 B REAL. ERM selected epoch 17 (clean BCE .12535) and channel epoch 16 (.16297), from 21 epochs/840 updates each. Channel reached patience 5 at the epoch cap. ERM reached cap with four bad epochs. Both updated six backbone tensors and zero of 136 BN tensors. Channel switched arithmetic at steps 426 and 430 after historical OOM. ERM kept the standard gradient path. Final score inference was segmented in both.

| Development metric | ERM | Channel |
| --- | ---: | ---: |
| A clean FA / miss at 0.5 | 4/112, 8/112 | 4/112, 8/112 |
| A clean ROC-AUC / EER | .98868 / 5.36% | .97911 / 5.36% |
| A stress FA / miss at 0.5 | 137/336, 162/336 | 72/336, 128/336 |
| A stress ROC-AUC / EER | .62112 / 41.82% | .80841 / 26.79% |
| B clean FA | 19/144 | 30/144 |
| A05 stress miss | 40/54 | 47/54 |
| A06 stress miss | 19/54 | 37/54 |

Stress A aggregates noise5 and both ordered noise/codec views with equal condition support. ROC-AUC change shows ranking differences within stress conditions beyond a threshold-only shift. This remains descriptive of one seed, repeated dev selection and different training arithmetic. Different Phase 7 cohorts and budgets preclude attributing cross-phase differences to fine-tuning. [Detailed Phase 8 review](PHASE8_MEDIUM_TRAINING_SCIENTIFIC_REVIEW.md) · [Re-audit JSON](../../outputs/jaes_v2/phase9/medium_reaudit_v1.json)

## Engine, data and readiness

Three Phase 9 guard tests passed, as did 155 JAES and 40 selected legacy tests. Existing historical validation exercises interruption and selection. Prospective fixed-engine exact resume, full validation score parity and GPU execution remain unqualified. CPU probe peaked at 687 MiB for two gradient cases. Foundation preflight reported 33,327 projected updates and 3,187,671,040 projected checkpoint bytes, and 0 TF-visible GPUs. The storage gate alone passed. Full foundation remains **NO-GO**. [Gradient](PHASE9_GRADIENT_NUMERICAL_PARITY.md) · [Validation](PHASE9_VALIDATION_OPTIMIZATION.md) · [Resources](PHASE9_FULL_TRAINING_GO_NO_GO.md) · [Dataset inventory](PHASE9_DATASET_ACQUISITION_DECISION.md)

C1 remains supporting order/group evidence. C2 is `DATA_BLOCKER / NOVELTY_NOT_ESTABLISHED`. C3 is deferred. The immediate evidence sequence is engine qualification, 100 updates with resume after resource review, and fresh paired ERM/channel seeds if feasible. Medium-plus, full foundation, ShuffleNet transfer and external evaluation are subsequent conditional jobs. [Novelty](PHASE9_NOVELTY_COMPARISON.md) · [Experiments](PHASE9_TRAINING_PRIORITY_MATRIX.md) · [Verified commands](PHASE9_AUTHOR_TRAINING_COMMANDS.md)

## Deadline and manuscript

| Date | Gate and action |
| --- | --- |
| 08–12 Nov | Archive re-audit, arithmetic and data evidence. Clear C/pagefile headroom and design fixed-engine acceptance. |
| 13–20 Nov | Prove exact same-engine resume, full validation parity and bounded cost before launching paired seed blocks. |
| 21 Nov–02 Dec | Complete matched medium as feasible. Decide whether results justify medium-plus. Consider controlled cue study only with verified pairs. |
| 03–10 Dec | External and transfer work only with frozen, lawful data and enough budget. Otherwise retain development scope. |
| 11–15 Dec | Complete manuscript tables, figures, uncertainty and contribution audit. |
| 16–20 Dec | QA references, reproducibility, rights, conference disclosure and formatting. Submit only after author-side eligibility and scientific review. |

The current English Markdown is an evidence-limited working draft. The journal extension changes the research question, cohort and experimental focus. The authors still need conference venue, proceedings and copyright details and confirmation of how to disclose the accepted work. [Draft](../../papers/jaes_v2/journal_draft/PHASE9_JOURNAL_DRAFT.md) · [AES author guidelines](https://aes.org/publications/journal-author-guidelines/)
