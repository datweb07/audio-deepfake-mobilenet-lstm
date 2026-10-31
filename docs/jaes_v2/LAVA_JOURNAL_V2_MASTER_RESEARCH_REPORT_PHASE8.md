# LAVA JOURNAL V2 — MASTER RESEARCH REPORT — PHASE8 EDITION

Evidence-generated; historical Phase7 edition preserved. New results are explicitly separated from inherited evidence.

## 01. Executive Summary

FULL_TRAINING_CODE_READY: YES. NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: NO.

# Phase8 research plan — 31 October 2026

The starting observation is a conflict between REAL false alarms and FAKE misses, not established success of consistency. Phase7 H3−H2 stressed miss effects were −.095238, +.062500, +.020833; mean −.003968, seed SD .081741. All three blocks failed the paired clean guardrails. Its one-epoch F1 improved clean AUC but worsened stressed misses. These results motivate measuring both classes separately.

C1 remains SUPPORTING_EMPIRICAL_AND_ENGINEERING_CONTRIBUTION. Ordered-channel controls contribute an auditable empirical protocol, but additional bitrates and plots do not distinguish it from RADAR/AT-ADD. C2 is the central research question: can a genuinely distinct mechanism improve both classwise stressed risks with acceptable clean risk? C3 remains DEFERRED until a credible C2 and independent calibration exist.

Phase8 first reviews nearby objectives, then specifies M1 weighted BCE, M2 group robustness, and proposed order-aware M3 variants. Algebraic reduction is a rejection criterion. No M3 loss is implemented without TECHNICALLY_DISTINCT_CANDIDATE status. The current candidates do not pass; see PHASE8_NOVELTY_DECISION.md and PHASE8_METHOD_REDIRECTION.md.

Engineering proceeds regardless: a separate trainer supports fresh ImageNet initialization, deterministic sampling/dropout, multiple epochs/seeds, head training and warmup plus actual CNN fine-tuning, Adam checkpointing, bounded resource use, declared checkpoint selection and exported-model evaluation. Tiny pilots are implementation evidence. The author launches medium/full research runs explicitly; Phase8 does not automatically execute them.

Use the existing medium cohort and foundation train/model-selection-development manifests. Never parse the reserved 7,341 calibration records or official evaluation. A01–A06 are seen attack families; generator checkpoint and original source lineage remain unknown. MLAAD-tiny and all development channel compositions are already exposed research data, not untouched confirmation.

Success criteria are separate: implementation readiness, full-training lifecycle readiness, scientific readiness. A successful trainer or a favorable tiny score cannot establish three international contributions. The final evidence-generated master report records the actual gates and blockers.

## 02. Conference Baseline

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

Accepted manuscript `papers/PaperID_804_final.docx` remains byte-identical: `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`. Original six deployed detector artifacts retain their historical training/threshold strata; none was retrained. The protected2376file baseline and3authorized amendments are verified separately from Journal evidence.

| model | samples | roc_auc | eer | f1 | threshold | source |
| --- | --- | --- | --- | --- | --- | --- |
| mobilenetv3_lstm | 2737 | 0.991147 | 0.024957 | 0.972414 | 0.820000 | outputs/lava_5/clean/mobilenetv3_lstm/summary.json |
| shufflenetv2_lstm | 2737 | 0.992883 | 0.014630 | 0.982411 | 0.120000 | outputs/lava_6/clean/shufflenetv2_lstm/summary.json |
| mnasnet_lstm | 2737 | 0.988628 | 0.030981 | 0.964454 | 0.900000 | outputs/lava_5/clean/mnasnet_lstm/summary.json |
| efficientnet_b0_lstm | 2737 | 0.987702 | 0.040000 | 0.956255 | 0.900000 | outputs/lava_5/clean/efficientnet_b0_lstm/summary.json |
| rawnet2 | 2737 | 0.517837 | 0.481270 | 0.533019 | 0.500000 | outputs/lava_5/clean/rawnet2/summary.json |
| aasist | 2737 | 0.559664 | 0.446349 | 0.513856 | 0.500000 | outputs/lava_5/clean/aasist/summary.json |

These are heterogeneous deployed artifacts, not a controlled architecture ranking. Conference venue, DOI/publication state and rights remain UNKNOWN.

Retained source-linked conference diagnostic condition inventory (not an independent Journal confirmation):

| model | condition | samples | f1 | roc_auc | eer | source |
| --- | --- | --- | --- | --- | --- | --- |
| aasist | clean | 100 | 0.506024 | 0.619458 | 0.380952 | outputs/lava_5/diagnostic_100/clean/aasist/summary.json |
| efficientnet_b0_lstm | clean | 100 | 0.962963 | 0.974959 | 0.047619 | outputs/lava_5/diagnostic_100/clean/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | clean | 100 | 0.975610 | 0.968801 | 0.047619 | outputs/lava_5/diagnostic_100/clean/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | clean | 100 | 0.975610 | 0.963465 | 0.047619 | outputs/lava_5/diagnostic_100/clean/mobilenetv3_lstm/summary.json |
| rawnet2 | clean | 100 | 0.566038 | 0.637521 | 0.413793 | outputs/lava_5/diagnostic_100/clean/rawnet2/summary.json |
| aasist | aac_96k | 100 | 0.523810 | 0.622742 | 0.380952 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/aasist/summary.json |
| efficientnet_b0_lstm | aac_96k | 100 | 0.962963 | 0.973727 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | aac_96k | 100 | 0.975610 | 0.969212 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | aac_96k | 100 | 0.962963 | 0.961002 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/mobilenetv3_lstm/summary.json |
| rawnet2 | aac_96k | 100 | 0.566038 | 0.639573 | 0.428571 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/rawnet2/summary.json |
| aasist | mp3_128k | 100 | 0.506024 | 0.617816 | 0.380952 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/aasist/summary.json |
| efficientnet_b0_lstm | mp3_128k | 100 | 0.962963 | 0.971675 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | mp3_128k | 100 | 0.941176 | 0.963875 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | mp3_128k | 100 | 0.962963 | 0.961002 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/mobilenetv3_lstm/summary.json |
| rawnet2 | mp3_128k | 100 | 0.555556 | 0.621511 | 0.428571 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/rawnet2/summary.json |
| aasist | mp3_64k | 100 | 0.506024 | 0.619048 | 0.380952 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/aasist/summary.json |
| efficientnet_b0_lstm | mp3_64k | 100 | 0.962963 | 0.971264 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | mp3_64k | 100 | 0.941176 | 0.963875 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | mp3_64k | 100 | 0.962963 | 0.964286 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/mobilenetv3_lstm/summary.json |
| rawnet2 | mp3_64k | 100 | 0.568807 | 0.619869 | 0.428571 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/rawnet2/summary.json |
| aasist | opus_64k | 100 | 0.517647 | 0.616995 | 0.380952 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/aasist/summary.json |
| efficientnet_b0_lstm | opus_64k | 100 | 0.962963 | 0.974138 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | opus_64k | 100 | 0.951220 | 0.962233 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | opus_64k | 100 | 0.975610 | 0.966338 | 0.047619 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/mobilenetv3_lstm/summary.json |
| rawnet2 | opus_64k | 100 | 0.576577 | 0.631773 | 0.431034 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/rawnet2/summary.json |
| aasist | snr_0 | 100 | 0.609375 | 0.573481 | 0.452381 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/aasist/summary.json |
| efficientnet_b0_lstm | snr_0 | 100 | 0.000000 | 0.627258 | 0.380952 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_0 | 100 | 0.000000 | 0.539409 | 0.452381 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_0 | 100 | 0.046512 | 0.616174 | 0.404762 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_0 | 100 | 0.000000 | 0.527094 | 0.476190 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/rawnet2/summary.json |
| aasist | snr_10 | 100 | 0.574257 | 0.532841 | 0.431034 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/aasist/summary.json |
| efficientnet_b0_lstm | snr_10 | 100 | 0.090909 | 0.683498 | 0.357143 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_10 | 100 | 0.567901 | 0.665846 | 0.380952 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_10 | 100 | 0.290909 | 0.719212 | 0.379310 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_10 | 100 | 0.000000 | 0.376847 | 0.595238 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/rawnet2/summary.json |
| aasist | snr_20 | 100 | 0.494382 | 0.584154 | 0.448276 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/aasist/summary.json |
| efficientnet_b0_lstm | snr_20 | 100 | 0.539683 | 0.727422 | 0.285714 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_20 | 100 | 0.796117 | 0.909278 | 0.155172 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_20 | 100 | 0.705882 | 0.834154 | 0.241379 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_20 | 100 | 0.233333 | 0.531199 | 0.500000 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/rawnet2/summary.json |
| aasist | snr_5 | 100 | 0.610169 | 0.544335 | 0.476190 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/aasist/summary.json |
| efficientnet_b0_lstm | snr_5 | 100 | 0.000000 | 0.657225 | 0.380952 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_5 | 100 | 0.090909 | 0.529557 | 0.500000 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_5 | 100 | 0.170213 | 0.649425 | 0.428571 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_5 | 100 | 0.000000 | 0.428982 | 0.586207 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/rawnet2/summary.json |
| aasist | synthetic_channel | 100 | 0.574257 | 0.630952 | 0.428571 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/aasist/summary.json |
| efficientnet_b0_lstm | synthetic_channel | 100 | 0.898876 | 0.969622 | 0.071429 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | synthetic_channel | 100 | 0.826667 | 0.943760 | 0.142857 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | synthetic_channel | 100 | 0.913580 | 0.965107 | 0.095238 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/mobilenetv3_lstm/summary.json |
| rawnet2 | synthetic_channel | 100 | 0.576923 | 0.557471 | 0.431034 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/rawnet2/summary.json |
| shufflenetv2_lstm | clean | 100 | 0.975610 | 0.964286 | 0.047619 | outputs/lava_6/diagnostic_100/clean/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | aac_96k | 100 | 0.962963 | 0.963875 | 0.047619 | outputs/lava_6/diagnostic_100/robustness/compression/aac_96k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | mp3_128k | 100 | 0.975610 | 0.963054 | 0.047619 | outputs/lava_6/diagnostic_100/robustness/compression/mp3_128k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | mp3_64k | 100 | 0.975610 | 0.964696 | 0.047619 | outputs/lava_6/diagnostic_100/robustness/compression/mp3_64k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | opus_64k | 100 | 0.975610 | 0.963054 | 0.047619 | outputs/lava_6/diagnostic_100/robustness/compression/opus_64k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_0 | 100 | 0.274510 | 0.696634 | 0.344828 | outputs/lava_6/diagnostic_100/robustness/noise/snr_0/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_10 | 100 | 0.692308 | 0.855501 | 0.258621 | outputs/lava_6/diagnostic_100/robustness/noise/snr_10/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_20 | 100 | 0.913580 | 0.950739 | 0.095238 | outputs/lava_6/diagnostic_100/robustness/noise/snr_20/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_5 | 100 | 0.600000 | 0.754926 | 0.310345 | outputs/lava_6/diagnostic_100/robustness/noise/snr_5/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | synthetic_channel | 100 | 0.950000 | 0.973317 | 0.071429 | outputs/lava_6/diagnostic_100/robustness/replay/synthetic_channel/shufflenetv2_lstm/summary.json |

## 03. Journal Research Objectives

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

### Accepted conference questions

- RQ1: How do deployed artifacts compare on clean AUC and EER, and how should threshold-dependent metrics be interpreted under heterogeneous training and calibration?

- RQ2: How does absolute performance change under the executed noise, codec, and simulated-replay conditions on the fixed diagnostic subset?

- RQ3: Which artifacts are non-dominated across diagnostic-subset EER, mean stressed F1, and end-to-end RTF?

These are artifact comparison, diagnostic stress and scenario-specific efficiency questions; training and pretraining are heterogeneous, so the answers concern artifacts rather than intrinsic architecture superiority.

### Journal questions

J-RQ1: Does channel order create practically important classwise risk changes after insertion-power, source, preprocessing and operating-point controls? Exploratory evidence exists; the fixed-generator source-matched confirmation lacks adequate support. J-RQ2: Can ordinary amplitude/channel/paired training or a genuinely distinct evidence-preserving mechanism improve a lightweight detector subject to bounded clean REAL false alarms? Standard controls are being tested; a distinct mechanism is not established. J-RQ3: Can selective inference improve matched-cost risk/coverage with independent calibration and a validated second stage? Unanswered; C3 is deferred.

Phase7: RQ7.1 asks whether stronger all-attack coverage changes the pilot findings; RQ7.2 asks whether H3–H2 survives independent training seeds; RQ7.3 asks whether selected CNN adaptation improves classwise detection under a matched schedule; RQ7.4 asks whether a falsifiable technical distinction survives current prior art.

## 04. Complete Phase Timeline

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

| Phase | Objective | Implementation | Executed run IDs | Finding / status |
|---|---|---|---|---|
| 1 | Ordered noise/codec feasibility | Shared source variants, strict scores and parent bootstrap | c1_noise_opus_pilot_v1, c1_noise_opus_pilot_v2, c1_noise_opus_pilot_v3 | Observed reversal/interaction is descriptive; insertion-point noise scaling is a confound |
| 2 | Confounder-aware development | Fixed versus relative noise, headroom/PCM/delay controls and metadata audit | c1_channel_controls_dev_v1 | Channel order cannot be isolated from amplitude, codec gain and preparation by one average metric |
| 3 | External MLAAD-tiny development | Pinned public acquisition, source provenance, common cache and two frozen artifacts | c1_mlaad_tiny_dev_v1_mobilenetv3_lstm, c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | Class-asymmetric noisy-channel failures persist; author system labels are not verified unseen checkpoints |
| 4 | Source matching and nuisance diagnosis | Source-ID/text matching levels, confirmation feasibility and within-record silence/gain controls | c1_nuisance_controls_dev_v1_mobilenetv3_lstm, c1_nuisance_controls_dev_v1_shufflenetv2_lstm | Sparse source support blocks the chosen matched-generator design; absolute Mel floor explains feature non-invariance |
| 5 | Controlled data and standard learner | ASV foundation, reserved calibration, P0/P1/P2 and separately trained head-only students | c2_overfit_sanity_v1, c2_p0_aug_pilot_v2, c2_p0_erm_pilot_v1, c2_p0_erm_pilot_v2, c2_p1_aug_pilot_v2, c2_p1_erm_pilot_v1, c2_p1_erm_pilot_v2, phase5_preprocessing_study_v1 | Gain invariance can coexist with worse discrimination; clean total error hides opposite REAL/FAKE changes |
| 6 | Stronger controlled comparisons and consolidation | Speaker-balanced cohort, B1/B4/control/finite RawBoost, partial-backbone sanity and auditable master report | partial_finetune_sanity_v1, partial_finetune_sanity_v2, p6_p0_b0_v1, p6_p0_b1_v1, p6_p0_b2_v1, p6_p0_b4_control_v1, p6_p0_b4_v1, p6_p0_rb_v1, p6_p1_b0_v1, p6_p1_b1_v1, p6_p1_b2_v1 | Read actual Phase6 results below; implementation and learning success do not establish novelty |

Each phase is incremental. Phase1–4 use frozen conference predictors; Phase5 introduces fresh research students. The matched-generator hypothesis is not silently renamed as a successful experiment. Phase6 uses a new cohort and therefore is not a direct continuation of Phase5 accuracy values.

Phase7 adds the all-attack selector, REAL-only sensitivity stratum, versioned resumable atomic P0 cache, independent training blocks, standard equal-view controls, genuine fine-tuning lifecycle, proxy falsification, primary-source refresh and this separate handoff edition. Training quality and implementation preflight remain separate inventory entries.

Phase8 adds standard M1/M2, genuine multi-epoch warmup/partial training, exact-boundary resume tests and an explicit M3 rejection; no full-data experiment was automatically launched.

## 05. Dataset Inventory

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

| Dataset | Local status | Source | License | Counts / scope |
|---|---|---|---|---|
| LAVA_LEGACY_INTERNAL | EXECUTED | UNKNOWN | UNKNOWN | {'test': 2737, 'train': 12762, 'validation': 2733} |
| ASVspoof2019_LA | TRAIN_DEV_EXTRACTED_EVAL_UNOPENED | https://datashare.ed.ac.uk/handle/10283/3336 | ODC-By database; upstream content caveats | {'train': {'recordings': 25380, 'class_counts': {'0': 2580, '1': 22800}, 'speakers': 20}, 'development': {'recordings': 17503, 'class_counts': {'0': 2023, '1': 15480}, 'speakers': 16}, 'calibration': {'recordings': 7341, 'class_counts': {'0': 525, '1': 6816}, 'speakers': 4}} |
| MLAAD_TINY | EXPLORATORY_DEVELOPMENT_EXECUTED | ['https://huggingface.co/datasets/mueller91/MLAAD-tiny/tree/9143e5ea709575ebab6bec52840a1043aada7bb1'] | ['CC-BY-NC-4.0', 'M-AILABS original notice'] | 192 |
| RADAR2026 | METADATA_ONLY_NO_AUDIO | https://zenodo.org/records/22110815 | CC-BY-4.0 release declaration | See Phase3 release inventory; no locally executed audio |
| In-the-Wild | NOT_ACQUIRED | https://huggingface.co/datasets/mueller91/In-The-Wild | Repository Apache-2.0 declaration; upstream conditions | NOT_VERIFIED_LOCALLY |
| DeepFakeVox-HQ / ANC-Spoof | PRIOR_ART_ONLY_NOT_ACQUIRED | F-SAT / PCL-NET papers | NOT_VERIFIED_FOR_LOCAL_USE | NOT_MEASURED_LOCALLY |

RADAR and In-the-Wild are investigated options, not executed external evidence. Literature datasets DeepFakeVox-HQ and ANC-Spoof were not acquired. A cited dataset or code repository is not installed training data. No Phase6 multi-GB download or ASV re-extraction occurred. The existing user archive has local SHA/CRC checks; authenticated remote byte identity remains unverified.

| Observed dataset profile | Rates / Hz | Duration seconds min/median/max | Known speaker/lineage limits |
|---|---|---|---|
| ASVspoof2019_LA | [16000] | {'min': 0.6524375, 'median': 3.23175, 'max': 13.1879375} | ['Source/parent content and exact generator identity UNKNOWN', 'Official speaker IDs are not full source-utterance lineage', 'SHA256 and CRC establish local integrity; official remote archive checksum not verified', 'Decoded/perceptual duplicate screening not exhaustive'] |
| MLAAD_TINY | [16000] | {'min': 1.32, 'median': 6.874875, 'max': 18.349875} | ['M-AILABS:en_UK/by_book/female/elizabeth_klett', 'M-AILABS:en_US/by_book/female/judy_bieber', 'M-AILABS:en_US/by_book/female/mary_ann', 'M-AILABS:en_US/by_book/male/elliot_miller'] |

ASV archive/extraction bytes from the retained central-directory audit: {'archive_bytes': 7646913333, 'total_expanded_bytes': 7655650274, 'extracted_bytes': 3178404021, 'extracted_files': 50371}. Calibration header profiles were not reopened in Phase6.

Phase7 selected cohort:

| role | n | real | fake | speakers | attacks |
| --- | --- | --- | --- | --- | --- |
| train | 640 | 320 | 320 | 20 | {"A01": 53, "A02": 54, "A03": 54, "A04": 54, "A05": 53, "A06": 52} |
| development_A | 224 | 112 | 112 | 7 | {"A01": 19, "A02": 19, "A03": 19, "A04": 19, "A05": 18, "A06": 18} |
| development_B | 144 | 144 | 0 | 9 | {} |

Selection manifest SHA256: `029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319`. No new dataset/audio download. REAL-only speakers may have prior exploratory Phase5 exposure. MLAAD-tiny remains already-used development. RADAR audio and In-the-Wild remain unacquired.

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

The canonical legacy split groups byte-identical files; it does not certify speaker, generator, original utterance or corpus independence. MLAAD-tiny has a custom exploratory development selection with its official split UNKNOWN; all earlier design use prevents relabeling it as untouched confirmation. ASV official train stays train; official development is divided by known official speakers into model-selection development and reserved calibration. Evaluation audio/protocol labels remain unopened. The new selector reads only the two train/model-selection manifests, never the reserved calibration file. It excludes development speakers lacking both classes and records the resulting target population. Byte, known-parent, known-speaker and selected decoded-PCM conflicts are checked. Re-encoding, perceptual reuse and unknown original lineage remain unresolved.

| ASV research role | Actual counts | Manifest | SHA256 |
|---|---|---|---|
| train | {'recordings': 25380, 'class_counts': {'0': 2580, '1': 22800}, 'speakers': 20} | outputs/jaes_v2/datasets/asv2019_foundation_v1/train_manifest.csv | 7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72 |
| development | {'recordings': 17503, 'class_counts': {'0': 2023, '1': 15480}, 'speakers': 16} | outputs/jaes_v2/datasets/asv2019_foundation_v1/development_manifest.csv | dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a |
| calibration | {'recordings': 7341, 'class_counts': {'0': 525, '1': 6816}, 'speakers': 4} | outputs/jaes_v2/datasets/asv2019_foundation_v1/calibration_manifest.csv | 4fb256823a83d313e2ddab938c4271aa2e72bc28f8cb6f1cb0e6e3b35f25ca74 |

Phase6 selection:

```json
{
  "status": "LOCKED_SPEAKER_BALANCED_DEVELOPMENT",
  "seed": 5806,
  "per_speaker_per_class": 4,
  "foundation_hashes": {
    "train": "7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72",
    "development": "dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a"
  },
  "counts": {
    "train": {
      "n": 160,
      "speakers": 20,
      "real": 80,
      "fake": 80,
      "attacks": [
        "A01",
        "A02",
        "A03",
        "A04"
      ]
    },
    "development": {
      "n": 56,
      "speakers": 7,
      "real": 28,
      "fake": 28,
      "attacks": [
        "A01",
        "A02",
        "A03",
        "A04"
      ]
    }
  },
  "exclusions": "Development speakers without FAKE support excluded prospectively; not representative full official dev",
  "unknown_lineage": "Original utterance/generator checkpoint UNKNOWN",
  "decoded_duplicate_groups": 0,
  "calibration_accessed": false,
  "evaluation_accessed": false,
  "manifest_sha256": "45ff123a734e11935e9e7f7a1ee9259f950cc82f7441bf06b6ca96cd56d87965"
}
```

Phase7 preserves official train/model-selection-development roles. Reserved7341calibration and official evaluation were not loaded. Byte checks, selected exact decoded PCM, known speakers and local-byte parents show no selected overlap; original utterance ancestry remains ORIGINAL_SOURCE_LINEAGE_UNKNOWN. The selector rotates the16FAKE remainder across six attacks per speaker instead of always privileging the first four. Selection and cache contracts reject stale hashes and reserved roles. `selection_rows.csv` records stratum and explicit provenance fields; manifest hash is stored in its selection metadata to avoid an impossible self-referential CSV hash.

Verified historical exposure: all9REAL-only speakers were used in Phase5development; 3Phase7B recordings also match those earlier selected IDs. This is broader-speaker sensitivity inside development, never independent confirmation. Reserved manifests may be byte-hashed for historical integrity; their records/labels are not parsed or used for training, selection, calibration or evaluation.

## 06. Model Inventory

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

| Artifact/model | Framework | Role / training | Input / scope | Checkpoint path | SHA256 |
|---|---|---|---|---|---|
| mobilenetv3_lstm | tensorflow | FROZEN_CONFERENCE_ARTIFACT / Historical local training | mel_sequence / transfer_learning | D:\audio-deepfake-mobilenet-lstm\models\lava_mobilenetv3_lstm.keras | 7345bdeb9542adb7686e3cefbf070ad30f2b0de0da77ef5e1d644dd743763c84 |
| shufflenetv2_lstm | tensorflow | FROZEN_CONFERENCE_ARTIFACT / Historical local training | mel_sequence / full_end_to_end_from_epoch_1 | D:\audio-deepfake-mobilenet-lstm\models\shufflenetv2_lstm\model.keras | 0a7dc43f6731e7c5c0379c2c9a341e7358a81bc37a7bedf4786e006942ec9442 |
| mnasnet_lstm | tensorflow | FROZEN_CONFERENCE_ARTIFACT / Historical local training | mel_sequence / full_end_to_end_from_epoch_1 | D:\audio-deepfake-mobilenet-lstm\models\mnasnet_lstm\model.keras | 2c5301b2c3dc3e3faeb2e8ff11409c0b2dd637568a407c5a7b4eefba02055b02 |
| efficientnet_b0_lstm | tensorflow | FROZEN_CONFERENCE_ARTIFACT / Historical local training | mel_sequence / transfer_learning | D:\audio-deepfake-mobilenet-lstm\models\efficientnet_b0_lstm\model.keras | 853a428ad0280faa644aaad0b4d3d6e25d9130b70b4167dfed9a7193a1b053f2 |
| rawnet2_pretrained | onnxruntime (PyTorch export) | FROZEN_CONFERENCE_ARTIFACT / External reference checkpoint | waveform / native_reference | D:\audio-deepfake-mobilenet-lstm\models\rawnet2_pretrained\model.onnx | afdaf27fa4e010b8a6d3051b37ea8d6ffb27dbaa7678faf22fef53285d515671 |
| aasist_pretrained | onnxruntime (PyTorch export) | FROZEN_CONFERENCE_ARTIFACT / External reference checkpoint | waveform / native_reference | D:\audio-deepfake-mobilenet-lstm\models\aasist_pretrained\model.onnx | 7e7bbe8baf2dd8f6a09dcb5236133c81ea7c9a74b12929d685b39fb0026d776c |
| c2_overfit_sanity_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / SMOKE_TESTED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\c2\c2_overfit_sanity_v1\best.keras | 536e5436bdad8e8c0942ab1e8721f50d9d6dc13ef2b63380ea960f68aca1d02b |
| c2_p0_aug_pilot_v2 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\c2\c2_p0_aug_pilot_v2\best.keras | bf739431bb5b617c62fafc82c8d260b48686d7cd99577fc539cf0900ff77e215 |
| c2_p0_erm_pilot_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\c2\c2_p0_erm_pilot_v1\best.keras | b080c4cb19c896dda56084012e12bee1e89e450799acd150ec7b3b42ab95b8c6 |
| c2_p0_erm_pilot_v2 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\c2\c2_p0_erm_pilot_v2\best.keras | cb1b305eb27c320069a4d66d0cf1a2d195f2e140b11108143d5c6bbb74aa5794 |
| c2_p1_aug_pilot_v2 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\c2\c2_p1_aug_pilot_v2\best.keras | dfc7dfc51112a43a38fe6f7d639fc877655ce70eab8bd509e912f4f7c991e02f |
| c2_p1_erm_pilot_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / INTERRUPTED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\c2\c2_p1_erm_pilot_v1\best.keras | 02b56b6ed46b42a5d94434681daf7043ce5b835cef61ce9fa38d65898d42046b |
| c2_p1_erm_pilot_v2 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\c2\c2_p1_erm_pilot_v2\best.keras | 806cc3964ceac90e55b79f431ad3020401c87fd3607c4c96f17a6ea42fa7c2c5 |
| partial_finetune_sanity_v1 | TensorFlow | PHASE6_SANITY / FAILED_PROCESS_TERMINATION | six224x224x3 Mel images / three-second observed window / UNKNOWN | NOT_AVAILABLE | NOT_AVAILABLE |
| partial_finetune_sanity_v2 | TensorFlow | PHASE6_SANITY / SMOKE_TESTED | six224x224x3 Mel images / three-second observed window / PARTIAL_BACKBONE_FINETUNE | outputs\jaes_v2\phase6\partial_finetune_sanity_v2\sanity.keras | 0e36448923fd41248844feb886a2858a3222aa8233c755ba8cdeef4aef2d2ebb |
| p6_p0_b0_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p0_b0_v1\best.keras | 154759156da658d316bbf536fbeb86076e51e81157b0acac1bff3251eaf86800 |
| p6_p0_b1_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p0_b1_v1\best.keras | 1cf0989b932f243f275a6237ae028e455288c455af85d61d89e0cbd3d2aac740 |
| p6_p0_b2_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p0_b2_v1\best.keras | 573f16fa0d17b188a47e6a52dbe661ab97b5e07fb5f9511bbb3af99704818ba3 |
| p6_p0_b4_control_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p0_b4_control_v1\best.keras | 6283e1c421246c5fcaf3ca533b9a7d3496483634f6957a8838f891fde9c184b4 |
| p6_p0_b4_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p0_b4_v1\best.keras | dabb61807bdc7d10ff79e28cbd83d7b27dad712b79bb62b4715a62f45b9e0841 |
| p6_p0_rb_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p0_rb_v1\best.keras | cb74cd9a73986585b2e1f1a6091c487585f800350ec1cfcd34509f1b722224d6 |
| p6_p1_b0_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p1_b0_v1\best.keras | 0b57b6b8c975fef57e072ded4732a605b61543e5d0d703313e208c61cab6c612 |
| p6_p1_b1_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p1_b1_v1\best.keras | e44d9ae4fe53b70b912c701169e24cd8dce849283b1f94e9720ec918f09b24fc |
| p6_p1_b2_v1 | TensorFlow | JOURNAL_RESEARCH_STUDENT / PILOT_EVALUATED | six224x224x3 Mel images / three-second observed window / HEAD_ONLY | outputs\jaes_v2\phase6\training\p6_p1_b2_v1\best.keras | 8b99f8709ab3896a7355b150788cb6898881b5e3cdcab0495af930202c3d643f |

MobileNet and EfficientNet use declared ImageNet transfer initialization; ShuffleNet and MnasNet conference artifacts use their recorded scratch-training stratum, with pretraining not verified. RawNet2/AASIST are external reference checkpoints, not freshly trained Journal V2 models. Mel models use the retained temporal-image contract; raw references retain their own waveform rate/length contract and score orientation. Research students reuse the existing MobileNetV3Small/LSTM/Dense architecture with fresh head and pinned ImageNet weights. Exact registry metadata, input rates, initialization classes and artifact fields are retained in the accompanying JSON. Failed/interrupted and sanity checkpoints remain labeled by scope.

Phase7 fresh research checkpoints:

| model | policy | seed | checkpoint | sha256 |
| --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | HEAD_ONLY | 6107 | outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/best.keras | b17f0084b2f25f5767274dde3fe82ffcdad14c534a5a68e765170da57f9821b6 |
| p7_f1_s6107_v1 | PARTIAL_BACKBONE_FINETUNE | 6107 | outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/best.keras | af2004818fe084579d1162bd291b36f42cf226e02b0e606ff89833b4f79945d1 |
| p7_h0_s6107_v1 | HEAD_ONLY | 6107 | outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/best.keras | 86c1dd14097eab4f4d5ff5fc807a54fb54653be377c951e0154c6b0632456766 |
| p7_h0_s6207_v1 | HEAD_ONLY | 6207 | outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/best.keras | 8fb10e7d341e5440669e764b4ed7bfffe81a948be320bdb99c6575c219cf52e2 |
| p7_h0_s6307_v1 | HEAD_ONLY | 6307 | outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/best.keras | bb9d31074cfe6138b71d0225c625e23828eaa3486974bc08e1cbff43f696d427 |
| p7_h1_s6107_v1 | HEAD_ONLY | 6107 | outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/best.keras | 0763cf3a9d9c7ab8b5cc562c80285d4947342ec74b79e075dc1b847f93105406 |
| p7_h1_s6207_v1 | HEAD_ONLY | 6207 | outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/best.keras | 0dd78c6d6ff235b727afef2377e68cfd8cf9dc30a79899386b6f9fc2baa84d6b |
| p7_h1_s6307_v1 | HEAD_ONLY | 6307 | outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/best.keras | ddfe8801c0c3d30b7b1af76ff2aab887e592975e552528a4123f5050ee00e4cd |
| p7_h2_s6107_v1 | HEAD_ONLY | 6107 | outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/best.keras | 5344a41441d5e39c00d638aa4af918192d1184bd8ec5b546d70184e7ceae53e5 |
| p7_h2_s6207_v1 | HEAD_ONLY | 6207 | outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/best.keras | 1dc7a0a10f7c45e9ac6bedc24586add15eadea23e32df1d7bb28fb0641001ac9 |
| p7_h2_s6307_v1 | HEAD_ONLY | 6307 | outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/best.keras | c530cfd2a0a4bdb751b39fda60757b59f489becc17d87f5660ca74bc2cbc3589 |
| p7_h3_s6107_v1 | HEAD_ONLY | 6107 | outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/best.keras | 4066ad204682742456775cd2fbf3407acac458199702e90175659db0a77dcbc3 |
| p7_h3_s6207_v1 | HEAD_ONLY | 6207 | outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/best.keras | 50ef09c423b906b400f6f3cdaa20d39a748537694c482d4512b375af8726fb84 |
| p7_h3_s6307_v1 | HEAD_ONLY | 6307 | outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/best.keras | 54fd584f23a8db721f215308e23c34bcf29f40978c31a1065cafd7e6b7ae0d22 |
| p7_h4_s6107_v1 | HEAD_ONLY | 6107 | outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/best.keras | e79c1e5bb0ef542359c847f0f7adf256464ef409cbbd064369bbe543ec2dd361 |
| p7_h4_s6207_v1 | HEAD_ONLY | 6207 | outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/best.keras | 1e2209c28eb7e1843f68bd86469a445eabc6445db369e0ffaa5b266a1fb0a905 |
| p7_h4_s6307_v1 | HEAD_ONLY | 6307 | outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/best.keras | b52e5fa3cdf3ae426b129051202134f325ce31c3f1f28b7821dd2f935f07eb68 |

Phase8 model rows: 17. Fresh-run and interrupted counterpart pairs are lifecycle checks, not additional independent seeds.

## 07. Experiment Inventory

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

| Phase | ID | Scope/status | Models | Train? | Originals / conditions / score rows | Seed / threshold | Output |
|---|---|---|---|---|---|---|---|
| 2 | c1_channel_controls_dev_v1 | LEGACY_DEVELOPMENT / DEVELOPMENT_EVALUATED | ['mobilenetv3_lstm'] | False | 64 / 14 / 896 | UNKNOWN / 0.82 | outputs/jaes_v2/c1_channel_controls_dev_v1 |
| 3 | c1_mlaad_tiny_dev_v1_mobilenetv3_lstm | EXPLORATORY_PUBLIC_DEVELOPMENT / EXTERNAL_DEVELOPMENT_EVALUATED | ['mobilenetv3_lstm'] | False | 192 / 7 / 1344 | UNKNOWN / 0.82 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm |
| 3 | c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | EXPLORATORY_PUBLIC_DEVELOPMENT / EXTERNAL_DEVELOPMENT_EVALUATED | ['shufflenetv2_lstm'] | False | 192 / 7 / 1344 | UNKNOWN / 0.12 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm |
| 1 | c1_noise_opus_pilot_v1 | PILOT / PILOT_EVALUATED | ['mobilenetv3_lstm'] | False | 12 / 5 / 60 | LEGACY_CHECKPOINT_METADATA / 0.82 | outputs/jaes_v2/c1_noise_opus_pilot_v1 |
| 1 | c1_noise_opus_pilot_v2 | PILOT / PILOT_EVALUATED | ['mobilenetv3_lstm'] | False | 12 / 5 / 60 | LEGACY_CHECKPOINT_METADATA / 0.82 | outputs/jaes_v2/c1_noise_opus_pilot_v2 |
| 1 | c1_noise_opus_pilot_v3 | PILOT / PILOT_EVALUATED | ['mobilenetv3_lstm'] | False | 12 / 5 / 60 | LEGACY_CHECKPOINT_METADATA / 0.82 | outputs/jaes_v2/c1_noise_opus_pilot_v3 |
| 4 | c1_nuisance_controls_dev_v1_mobilenetv3_lstm | EXPLORATORY_WITHIN_RECORD_DEVELOPMENT / WITHIN_RECORD_DEVELOPMENT_EVALUATED | ['mobilenetv3_lstm'] | False | 32 / 7 / 512 | UNKNOWN / 0.82 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm |
| 4 | c1_nuisance_controls_dev_v1_shufflenetv2_lstm | EXPLORATORY_WITHIN_RECORD_DEVELOPMENT / WITHIN_RECORD_DEVELOPMENT_EVALUATED | ['shufflenetv2_lstm'] | False | 32 / 7 / 512 | UNKNOWN / 0.12 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_shufflenetv2_lstm |
| 5 | c2_overfit_sanity_v1 | TIER1_OVERFIT_SANITY / SMOKE_TESTED | ['jaes_v2_mobilenet_student'] | True | 64 / 6 / 384 | 5804 / 0.5 | outputs/jaes_v2/c2/c2_overfit_sanity_v1 |
| 5 | c2_p0_aug_pilot_v2 | TIER2_RESEARCH_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 64 / 6 / 384 | 5804 / 0.5 | outputs/jaes_v2/c2/c2_p0_aug_pilot_v2 |
| 5 | c2_p0_erm_pilot_v1 | TIER2_RESEARCH_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 64 / 6 / 384 | 5804 / 0.5 | outputs/jaes_v2/c2/c2_p0_erm_pilot_v1 |
| 5 | c2_p0_erm_pilot_v2 | TIER2_RESEARCH_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 64 / 6 / 384 | 5804 / 0.5 | outputs/jaes_v2/c2/c2_p0_erm_pilot_v2 |
| 5 | c2_p1_aug_pilot_v2 | TIER2_RESEARCH_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 64 / 6 / 384 | 5804 / 0.5 | outputs/jaes_v2/c2/c2_p1_aug_pilot_v2 |
| 5 | c2_p1_erm_pilot_v1 | TIER2_RESEARCH_PILOT / INTERRUPTED | [] | True | UNKNOWN / UNKNOWN / 0 | 5804 / 0.5 | outputs/jaes_v2/c2/c2_p1_erm_pilot_v1 |
| 5 | c2_p1_erm_pilot_v2 | TIER2_RESEARCH_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 64 / 6 / 384 | 5804 / 0.5 | outputs/jaes_v2/c2/c2_p1_erm_pilot_v2 |
| 5 | phase5_preprocessing_study_v1 | UNKNOWN / DEVELOPMENT_DIAGNOSTIC_EVALUATED | ['mobilenetv3_lstm', 'shufflenetv2_lstm'] | False | 32 / 2 / 384 | UNKNOWN / UNKNOWN | outputs/jaes_v2/phase5_preprocessing_study_v1 |
| 6 | partial_finetune_sanity_v1 | UNKNOWN / FAILED_PROCESS_TERMINATION | [] | True | UNKNOWN / UNKNOWN / 0 | UNKNOWN / UNKNOWN | outputs/jaes_v2/phase6/partial_finetune_sanity_v1 |
| 6 | partial_finetune_sanity_v2 | OVERFIT_SANITY_RESOURCE_PREFLIGHT / SMOKE_TESTED | [] | True | UNKNOWN / UNKNOWN / 0 | 5806 / UNKNOWN | outputs/jaes_v2/phase6/partial_finetune_sanity_v2 |
| 6 | p6_p0_b0_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p0_b0_v1 |
| 6 | p6_p0_b1_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p0_b1_v1 |
| 6 | p6_p0_b2_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p0_b2_v1 |
| 6 | p6_p0_b4_control_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p0_b4_control_v1 |
| 6 | p6_p0_b4_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p0_b4_v1 |
| 6 | p6_p0_rb_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p0_rb_v1 |
| 6 | p6_p1_b0_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p1_b0_v1 |
| 6 | p6_p1_b1_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p1_b1_v1 |
| 6 | p6_p1_b2_v1 | CONTROLLED_PILOT / PILOT_EVALUATED | ['jaes_v2_mobilenet_student'] | True | 56 / 7 / 392 | 5806 / 0.5 | outputs/jaes_v2/phase6/training/p6_p1_b2_v1 |

Warmups, repeated views, predictor calls, reused scores and source originals have separate denominators. Unknown counts are printed UNKNOWN; missing scores are never promoted into performance. No confirmation experiment exists. The JSON inventory includes verified seal counts, metadata hashes, source count, conditions, update counts and resource measurements for every discovered run, including resource-stopped and interrupted predecessors.

Phase7 exact inventory:

| experiment_id | status | seed | training_policy | augmentation_policy | conditions | checkpoint_sha256 | artifact_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | PILOT_EVALUATED | 6107 | HEAD_ONLY | H0 | 7 | b17f0084b2f25f5767274dde3fe82ffcdad14c534a5a68e765170da57f9821b6 | outputs/jaes_v2/phase7/training/p7_f0_s6107_v1 |
| p7_f1_s6107_v1 | PILOT_EVALUATED | 6107 | PARTIAL_BACKBONE_FINETUNE | H0 | 7 | af2004818fe084579d1162bd291b36f42cf226e02b0e606ff89833b4f79945d1 | outputs/jaes_v2/phase7/training/p7_f1_s6107_v1 |
| p7_h0_s6107_v1 | PILOT_EVALUATED | 6107 | HEAD_ONLY | H0 | 7 | 86c1dd14097eab4f4d5ff5fc807a54fb54653be377c951e0154c6b0632456766 | outputs/jaes_v2/phase7/training/p7_h0_s6107_v1 |
| p7_h0_s6207_v1 | PILOT_EVALUATED | 6207 | HEAD_ONLY | H0 | 7 | 8fb10e7d341e5440669e764b4ed7bfffe81a948be320bdb99c6575c219cf52e2 | outputs/jaes_v2/phase7/training/p7_h0_s6207_v1 |
| p7_h0_s6307_v1 | PILOT_EVALUATED | 6307 | HEAD_ONLY | H0 | 7 | bb9d31074cfe6138b71d0225c625e23828eaa3486974bc08e1cbff43f696d427 | outputs/jaes_v2/phase7/training/p7_h0_s6307_v1 |
| p7_h1_s6107_v1 | PILOT_EVALUATED | 6107 | HEAD_ONLY | H1 | 7 | 0763cf3a9d9c7ab8b5cc562c80285d4947342ec74b79e075dc1b847f93105406 | outputs/jaes_v2/phase7/training/p7_h1_s6107_v1 |
| p7_h1_s6207_v1 | PILOT_EVALUATED | 6207 | HEAD_ONLY | H1 | 7 | 0dd78c6d6ff235b727afef2377e68cfd8cf9dc30a79899386b6f9fc2baa84d6b | outputs/jaes_v2/phase7/training/p7_h1_s6207_v1 |
| p7_h1_s6307_v1 | PILOT_EVALUATED | 6307 | HEAD_ONLY | H1 | 7 | ddfe8801c0c3d30b7b1af76ff2aab887e592975e552528a4123f5050ee00e4cd | outputs/jaes_v2/phase7/training/p7_h1_s6307_v1 |
| p7_h2_s6107_v1 | PILOT_EVALUATED | 6107 | HEAD_ONLY | H2 | 7 | 5344a41441d5e39c00d638aa4af918192d1184bd8ec5b546d70184e7ceae53e5 | outputs/jaes_v2/phase7/training/p7_h2_s6107_v1 |
| p7_h2_s6207_v1 | PILOT_EVALUATED | 6207 | HEAD_ONLY | H2 | 7 | 1dc7a0a10f7c45e9ac6bedc24586add15eadea23e32df1d7bb28fb0641001ac9 | outputs/jaes_v2/phase7/training/p7_h2_s6207_v1 |
| p7_h2_s6307_v1 | PILOT_EVALUATED | 6307 | HEAD_ONLY | H2 | 7 | c530cfd2a0a4bdb751b39fda60757b59f489becc17d87f5660ca74bc2cbc3589 | outputs/jaes_v2/phase7/training/p7_h2_s6307_v1 |
| p7_h3_s6107_v1 | PILOT_EVALUATED | 6107 | HEAD_ONLY | H3 | 7 | 4066ad204682742456775cd2fbf3407acac458199702e90175659db0a77dcbc3 | outputs/jaes_v2/phase7/training/p7_h3_s6107_v1 |
| p7_h3_s6207_v1 | PILOT_EVALUATED | 6207 | HEAD_ONLY | H3 | 7 | 50ef09c423b906b400f6f3cdaa20d39a748537694c482d4512b375af8726fb84 | outputs/jaes_v2/phase7/training/p7_h3_s6207_v1 |
| p7_h3_s6307_v1 | PILOT_EVALUATED | 6307 | HEAD_ONLY | H3 | 7 | 54fd584f23a8db721f215308e23c34bcf29f40978c31a1065cafd7e6b7ae0d22 | outputs/jaes_v2/phase7/training/p7_h3_s6307_v1 |
| p7_h4_s6107_v1 | PILOT_EVALUATED | 6107 | HEAD_ONLY | H4 | 7 | e79c1e5bb0ef542359c847f0f7adf256464ef409cbbd064369bbe543ec2dd361 | outputs/jaes_v2/phase7/training/p7_h4_s6107_v1 |
| p7_h4_s6207_v1 | PILOT_EVALUATED | 6207 | HEAD_ONLY | H4 | 7 | 1e2209c28eb7e1843f68bd86469a445eabc6445db369e0ffaa5b266a1fb0a905 | outputs/jaes_v2/phase7/training/p7_h4_s6207_v1 |
| p7_h4_s6307_v1 | PILOT_EVALUATED | 6307 | HEAD_ONLY | H4 | 7 | b52e5fa3cdf3ae426b129051202134f325ce31c3f1f28b7821dd2f935f07eb68 | outputs/jaes_v2/phase7/training/p7_h4_s6307_v1 |
| p7_f1_s6107_v1_resource_preflight | SMOKE_TESTED | 6107 | PARTIAL_BACKBONE_FINETUNE | H0 | 0 | NOT_DEFINED | outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1 |
| p7_cache_benchmark_v1 | SEALED_PHASE7_CACHE | {"encoder_initialization": 6107, "waveform_base": 5804} | FROZEN_IMAGE_NET_ENCODER | Declared in source artifact | See source views | 002803a73df2ab65425267ee2082d1ec213b6aa9716f19ac1a1ed5361239c5f6 | outputs/jaes_v2/phase7/caches/benchmark_p0_v1 |
| p7_medium_feature_cache_v1 | SEALED_PHASE7_CACHE | {"encoder_initialization": 6107, "waveform_base": 5804} | FROZEN_IMAGE_NET_ENCODER | Declared in source artifact | See source views | 002803a73df2ab65425267ee2082d1ec213b6aa9716f19ac1a1ed5361239c5f6 | outputs/jaes_v2/phase7/caches/medium_p0_v1 |
| p7_retention_proxy_controls_v1 | NON_SPECIFIC | {"cached_waveform_base": 5804, "encoder_initialization": 6107, "noise_control": 7107, "shuffle": 7107} | NO_TRAINING | Declared in source artifact | See source views | NOT_DEFINED | outputs/jaes_v2/phase7/diagnostics_v1 |
| p7_online_rawboost_preflight_v1 | AWAITING_COMPUTE | {"encoder_initialization": 6107, "epoch": 0, "online_base": 306107, "per_example": "SHA-derived, exact values in checks"} | NO_TRAINING | Declared in source artifact | See source views | NOT_DEFINED | outputs/jaes_v2/phase7/diagnostics_v1 |
| p7_full_student_efficiency_v1 | MEASURED_OFFLINE_CPU | {"no_new_training": true, "reused_training_seed": 6107} | NO_TRAINING | Declared in source artifact | See source views | NOT_DEFINED | outputs/jaes_v2/phase7/diagnostics_v1 |
| p7_representation_efficiency_v1 | MEASURED_OFFLINE_CPU | {"no_new_training": true, "reused_training_seed": 6107} | NO_TRAINING | Declared in source artifact | See source views | NOT_DEFINED | outputs/jaes_v2/phase7 |

All full experiment/config/source/manifest/checkpoint fields appear in the accompanying JSON. A group of three completed head runs is MULTI_SEED_DEVELOPMENT_EVALUATED evidence; each run remains PILOT_EVALUATED. Cache generation, proxy diagnostics, timing, failures and preflights are separate inventory records and never counted as additional trained model replicates. Phase7 trained exactly17quality-evaluated models, not one model per auxiliary record.

Phase8 run inventory:

| experiment_id | objective | policy | scope | steps |
| --- | --- | --- | --- | --- |
| p8_tiny_dro_head_s6107_v2 | GROUP_DRO | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_dro_head_s6107_v3 | GROUP_DRO | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_resume_v2 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_resume_v3 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v1 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v2 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v3 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6207_v2 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6207_v3 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6307_v2 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6307_v3 | ERM | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_resume_v2 | ERM | WARMUP_PARTIAL | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_resume_v3 | ERM | WARMUP_PARTIAL | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_v2 | ERM | WARMUP_PARTIAL | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_v3 | ERM | WARMUP_PARTIAL | TINY_LIFECYCLE | 4 |
| p8_tiny_weighted_head_s6107_v2 | WEIGHTED_BCE | HEAD_ONLY | TINY_LIFECYCLE | 4 |
| p8_tiny_weighted_head_s6107_v3 | WEIGHTED_BCE | HEAD_ONLY | TINY_LIFECYCLE | 4 |

Auxiliary records (not additional quality models):

| experiment_id | kind | output |
| --- | --- | --- |
| p8_aux_resource_preflight_preflight_foundation_erm_partial_v1 | RESOURCE_PREFLIGHT | outputs/jaes_v2/phase8/preflight/foundation_erm_partial_v1.json |
| p8_aux_resource_preflight_preflight_medium_erm_partial_v1 | RESOURCE_PREFLIGHT | outputs/jaes_v2/phase8/preflight/medium_erm_partial_v1.json |
| p8_aux_resource_benchmark_partial_cpu_v1_measurements | RESOURCE_BENCHMARK | outputs/jaes_v2/phase8/benchmark/partial_cpu_v1/measurements.json |
| p8_aux_resume_integration_audit_resume_integration_v2 | RESUME_INTEGRATION | outputs/jaes_v2/phase8/audit/resume_integration_v2.json |
| p8_aux_resume_integration_audit_resume_integration_v3 | RESUME_INTEGRATION | outputs/jaes_v2/phase8/audit/resume_integration_v3.json |
| p8_aux_export_seed_bn_integration_audit_representation_export_seed_checks_v3 | EXPORT_SEED_BN_INTEGRATION | outputs/jaes_v2/phase8/audit/representation_export_seed_checks_v3.json |
| p8_aux_regression_validation_final_validation | REGRESSION | outputs/jaes_v2/phase8/validation_final/validation.json |

## 08. Phase1–4 Findings

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

The ordered pilot demonstrates a reproducible analysis contract, not generator causation. A relative-SNR insertion rule changes absolute noise across arms; fixed-power and codec/PCM/headroom/delay controls are necessary. MLAAD adds real public transfer observations, but its few source-text clusters and UNKNOWN REAL lineage limit inference. FAKE misses can increase while REAL false alarms decrease, concealing failures in total error. Source matching found no adequate selected fixed-system English pair; author source/text agreement does not become verified acoustic source matching. Within-record silence/gain controls expose nuisance sensitivity. An absolute Mel floor violates ideal gain cancellation; scaling that floor in a counterfactual is a mechanistic diagnostic, not a deployable oracle or a learned contribution. C1 remains supporting validation/engineering.

Executed condition metrics (all variants retained in JSON):

| Phase / experiment | Model | Condition | n | Error | REAL FA | FAKE miss | AUC | EER | Source |
|---|---|---|---|---|---|---|---|---|---|
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | clean | 64 | 0.015625 | 0.03125 | 0.0 | 0.9990234375 | 0.03125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | clean_native | 64 | 0.03125 | 0.03125 | 0.03125 | 0.998046875 | 0.03125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | codec | 64 | 0.03125 | 0.0 | 0.0625 | 0.998046875 | 0.03125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | codec_aligned | 64 | 0.03125 | 0.0 | 0.0625 | 0.998046875 | 0.03125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | fixed_A_then_B | 64 | 0.46875 | 0.0 | 0.9375 | 0.728515625 | 0.34375 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | fixed_B_then_A | 64 | 0.34375 | 0.03125 | 0.65625 | 0.7431640625 | 0.28125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | fixed_noise | 64 | 0.359375 | 0.03125 | 0.6875 | 0.7275390625 | 0.28125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | headroom_low | 64 | 0.015625 | 0.03125 | 0.0 | 1.0 | 0.0 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | identity | 64 | 0.015625 | 0.03125 | 0.0 | 0.9990234375 | 0.03125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | local_A_then_B | 64 | 0.46875 | 0.0 | 0.9375 | 0.728515625 | 0.34375 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | local_B_then_A | 64 | 0.34375 | 0.03125 | 0.65625 | 0.7421875 | 0.28125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | local_noise | 64 | 0.359375 | 0.03125 | 0.6875 | 0.7275390625 | 0.28125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | pcm16 | 64 | 0.015625 | 0.03125 | 0.0 | 0.9990234375 | 0.03125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 2 / c1_channel_controls_dev_v1 | mobilenetv3_lstm | resample | 64 | 0.015625 | 0.03125 | 0.0 | 0.9990234375 | 0.03125 | outputs/jaes_v2/c1_channel_controls_dev_v1/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | clean | 192 | 0.28125 | 0.5 | 0.0625 | 0.8195529513888888 | 0.23958333333333334 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | clean_native | 192 | 0.2552083333333333 | 0.3854166666666667 | 0.125 | 0.8266059027777778 | 0.19791666666666666 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | fixed_A_then_B | 192 | 0.4895833333333333 | 0.020833333333333332 | 0.9583333333333334 | 0.677951388888889 | 0.35416666666666663 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | fixed_B_then_A | 192 | 0.4791666666666667 | 0.09375 | 0.8645833333333334 | 0.6831597222222223 | 0.3645833333333333 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | fixed_noise | 192 | 0.4635416666666667 | 0.08333333333333333 | 0.84375 | 0.6786024305555556 | 0.3541666666666667 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | clean | 192 | 0.24479166666666666 | 0.25 | 0.23958333333333334 | 0.8467881944444444 | 0.23958333333333337 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | clean_native | 192 | 0.22395833333333334 | 0.20833333333333334 | 0.23958333333333334 | 0.870334201388889 | 0.22916666666666663 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | fixed_A_then_B | 192 | 0.3802083333333333 | 0.11458333333333333 | 0.6458333333333334 | 0.6005859375 | 0.4375 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | fixed_B_then_A | 192 | 0.3697916666666667 | 0.22916666666666666 | 0.5104166666666666 | 0.6447482638888888 | 0.3854166666666667 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm/scores.csv |
| 3 / c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | fixed_noise | 192 | 0.3854166666666667 | 0.25 | 0.5208333333333334 | 0.6570095486111112 | 0.3645833333333333 | outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm/scores.csv |
| 1 / c1_noise_opus_pilot_v1 | mobilenetv3_lstm | A | 12 | 0.25 | 0.0 | 0.5 | 0.9166666666666667 | 0.16666666666666666 | outputs/jaes_v2/c1_noise_opus_pilot_v1/scores.csv |
| 1 / c1_noise_opus_pilot_v1 | mobilenetv3_lstm | A_then_B | 12 | 0.5 | 0.0 | 1.0 | 0.888888888888889 | 0.16666666666666666 | outputs/jaes_v2/c1_noise_opus_pilot_v1/scores.csv |
| 1 / c1_noise_opus_pilot_v1 | mobilenetv3_lstm | B | 12 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | outputs/jaes_v2/c1_noise_opus_pilot_v1/scores.csv |
| 1 / c1_noise_opus_pilot_v1 | mobilenetv3_lstm | B_then_A | 12 | 0.3333333333333333 | 0.0 | 0.6666666666666666 | 0.8055555555555556 | 0.33333333333333337 | outputs/jaes_v2/c1_noise_opus_pilot_v1/scores.csv |
| 1 / c1_noise_opus_pilot_v1 | mobilenetv3_lstm | clean | 12 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | outputs/jaes_v2/c1_noise_opus_pilot_v1/scores.csv |
| 1 / c1_noise_opus_pilot_v2 | mobilenetv3_lstm | A | 12 | 0.25 | 0.0 | 0.5 | 0.9166666666666667 | 0.16666666666666666 | outputs/jaes_v2/c1_noise_opus_pilot_v2/scores.csv |
| 1 / c1_noise_opus_pilot_v2 | mobilenetv3_lstm | A_then_B | 12 | 0.5 | 0.0 | 1.0 | 0.888888888888889 | 0.16666666666666666 | outputs/jaes_v2/c1_noise_opus_pilot_v2/scores.csv |
| 1 / c1_noise_opus_pilot_v2 | mobilenetv3_lstm | B | 12 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | outputs/jaes_v2/c1_noise_opus_pilot_v2/scores.csv |
| 1 / c1_noise_opus_pilot_v2 | mobilenetv3_lstm | B_then_A | 12 | 0.3333333333333333 | 0.0 | 0.6666666666666666 | 0.8055555555555556 | 0.33333333333333337 | outputs/jaes_v2/c1_noise_opus_pilot_v2/scores.csv |
| 1 / c1_noise_opus_pilot_v2 | mobilenetv3_lstm | clean | 12 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | outputs/jaes_v2/c1_noise_opus_pilot_v2/scores.csv |
| 1 / c1_noise_opus_pilot_v3 | mobilenetv3_lstm | A | 12 | 0.25 | 0.0 | 0.5 | 0.9166666666666667 | 0.16666666666666666 | outputs/jaes_v2/c1_noise_opus_pilot_v3/scores.csv |
| 1 / c1_noise_opus_pilot_v3 | mobilenetv3_lstm | A_then_B | 12 | 0.5 | 0.0 | 1.0 | 0.888888888888889 | 0.16666666666666666 | outputs/jaes_v2/c1_noise_opus_pilot_v3/scores.csv |
| 1 / c1_noise_opus_pilot_v3 | mobilenetv3_lstm | B | 12 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | outputs/jaes_v2/c1_noise_opus_pilot_v3/scores.csv |
| 1 / c1_noise_opus_pilot_v3 | mobilenetv3_lstm | B_then_A | 12 | 0.3333333333333333 | 0.0 | 0.6666666666666666 | 0.8055555555555556 | 0.33333333333333337 | outputs/jaes_v2/c1_noise_opus_pilot_v3/scores.csv |
| 1 / c1_noise_opus_pilot_v3 | mobilenetv3_lstm | clean | 12 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | outputs/jaes_v2/c1_noise_opus_pilot_v3/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | clean | 128 | 0.2734375 | 0.46875 | 0.078125 | 0.825927734375 | 0.234375 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | clean_native | 32 | 0.25 | 0.375 | 0.125 | 0.875 | 0.125 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | fixed_A_then_B | 128 | 0.5 | 0.015625 | 0.984375 | 0.621826171875 | 0.359375 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | fixed_B_then_A | 128 | 0.5 | 0.09375 | 0.90625 | 0.714599609375 | 0.3125 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_mobilenetv3_lstm | mobilenetv3_lstm | fixed_noise | 32 | 0.46875 | 0.0625 | 0.875 | 0.7578125 | 0.25 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_mobilenetv3_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | clean | 128 | 0.2109375 | 0.296875 | 0.125 | 0.86962890625 | 0.203125 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_shufflenetv2_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | clean_native | 32 | 0.1875 | 0.25 | 0.125 | 0.9140625 | 0.1875 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_shufflenetv2_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | fixed_A_then_B | 128 | 0.5078125 | 0.25 | 0.765625 | 0.634521484375 | 0.359375 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_shufflenetv2_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | fixed_B_then_A | 128 | 0.390625 | 0.265625 | 0.515625 | 0.677734375 | 0.34375 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_shufflenetv2_lstm/scores.csv |
| 4 / c1_nuisance_controls_dev_v1_shufflenetv2_lstm | shufflenetv2_lstm | fixed_noise | 32 | 0.40625 | 0.25 | 0.5625 | 0.69140625 | 0.3125 | outputs/jaes_v2/c1_nuisance_controls_dev_v1_shufflenetv2_lstm/scores.csv |

## 09. Phase5 Findings

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

The controlled foundation enabled real separately versioned learning while preserving the accepted paper. The frozen-backbone student pilot and changed-preprocessor diagnostic answer different questions. P1 removes half-gain feature/embedding/score differences on the tested frozen-model cohort yet harms ShuffleNet discrimination. Fresh P1 ERM improves clean ranking/error on its cohort while worsening noisy FAKE misses. Ordinary augmentation changes the clean operating point and therefore must retain classwise risks. Phase5 development contains unequal REAL/FAKE speaker support; min-size worst-group filtering excludes small REAL-only groups and cannot be reported without denominators. One seed and a small feature-cache cohort are exploratory. Preliminary time-capped/interrupted runs stay distinct from the main fixed-update cells.

| Main run | Condition | Originals | Error | REAL FA | FAKE miss | AUC | EER |
|---|---|---|---|---|---|---|---|
| c2_p0_aug_pilot_v2 | A_then_B | 64 | 0.328125 | 0.25 | 0.40625 | 0.7802734375 | 0.28125 |
| c2_p0_aug_pilot_v2 | B_then_A | 64 | 0.390625 | 0.25 | 0.53125 | 0.775390625 | 0.25 |
| c2_p0_aug_pilot_v2 | clean | 64 | 0.234375 | 0.28125 | 0.1875 | 0.8994140625 | 0.21875 |
| c2_p0_aug_pilot_v2 | gain_quarter | 64 | 0.15625 | 0.09375 | 0.21875 | 0.869140625 | 0.21875 |
| c2_p0_aug_pilot_v2 | noise5 | 64 | 0.40625 | 0.21875 | 0.59375 | 0.7626953125 | 0.28125 |
| c2_p0_aug_pilot_v2 | opus64 | 64 | 0.203125 | 0.25 | 0.15625 | 0.900390625 | 0.21875 |
| c2_p0_erm_pilot_v2 | A_then_B | 64 | 0.390625 | 0.125 | 0.65625 | 0.6884765625 | 0.34375 |
| c2_p0_erm_pilot_v2 | B_then_A | 64 | 0.40625 | 0.09375 | 0.71875 | 0.685546875 | 0.34375 |
| c2_p0_erm_pilot_v2 | clean | 64 | 0.1875 | 0.0625 | 0.3125 | 0.9296875 | 0.1875 |
| c2_p0_erm_pilot_v2 | gain_quarter | 64 | 0.171875 | 0.0625 | 0.28125 | 0.8623046875 | 0.21875 |
| c2_p0_erm_pilot_v2 | noise5 | 64 | 0.40625 | 0.09375 | 0.71875 | 0.6376953125 | 0.34375 |
| c2_p0_erm_pilot_v2 | opus64 | 64 | 0.1875 | 0.09375 | 0.28125 | 0.9306640625 | 0.1875 |
| c2_p1_aug_pilot_v2 | A_then_B | 64 | 0.359375 | 0.125 | 0.59375 | 0.755859375 | 0.34375 |
| c2_p1_aug_pilot_v2 | B_then_A | 64 | 0.390625 | 0.1875 | 0.59375 | 0.7578125 | 0.3125 |
| c2_p1_aug_pilot_v2 | clean | 64 | 0.1875 | 0.3125 | 0.0625 | 0.9150390625 | 0.15625 |
| c2_p1_aug_pilot_v2 | gain_quarter | 64 | 0.1875 | 0.3125 | 0.0625 | 0.9150390625 | 0.15625 |
| c2_p1_aug_pilot_v2 | noise5 | 64 | 0.390625 | 0.125 | 0.65625 | 0.7490234375 | 0.34375 |
| c2_p1_aug_pilot_v2 | opus64 | 64 | 0.140625 | 0.21875 | 0.0625 | 0.923828125 | 0.15625 |
| c2_p1_erm_pilot_v2 | A_then_B | 64 | 0.4375 | 0.03125 | 0.84375 | 0.71484375 | 0.34375 |
| c2_p1_erm_pilot_v2 | B_then_A | 64 | 0.46875 | 0.03125 | 0.90625 | 0.70703125 | 0.34375 |
| c2_p1_erm_pilot_v2 | clean | 64 | 0.15625 | 0.28125 | 0.03125 | 0.9580078125 | 0.0625 |
| c2_p1_erm_pilot_v2 | gain_quarter | 64 | 0.15625 | 0.28125 | 0.03125 | 0.9580078125 | 0.0625 |
| c2_p1_erm_pilot_v2 | noise5 | 64 | 0.46875 | 0.03125 | 0.90625 | 0.669921875 | 0.34375 |
| c2_p1_erm_pilot_v2 | opus64 | 64 | 0.09375 | 0.125 | 0.0625 | 0.9482421875 | 0.09375 |

Existing CPU timing is an offline warm-process diagnostic, not streaming/edge-device certification. NII teacher runtime prerequisites and checkpoint quality remain unresolved; corpus training overlap is known at the ASV/MLAAD level. A synthetic class-orientation helper does not certify real teacher quality.

## 10. Phase6 Findings

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

Speaker-aware selection balances both classes within each retained speaker. Attack/category coverage and excluded speakers are explicit; source utterances remain unknown. B1 uses discrete clean/half-gain exposure with held-out quarter-gain evaluation. B2 uses the existing noise/Opus bank. B4_CONTROL and B4 share the same two supervised views, initialization, order, source IDs and updates; the sole objective difference is probability MSE consistency. RawBoost uses unchanged MIT author functions behind a seed-isolating wrapper, with two finite algorithm5 realizations at the repo sample rate. It is not online stochastic training or an exact original-paper benchmark reproduction. Larger cohorts are planned from measured costs rather than automatically executed.

| Run | Condition | n | Error | REAL FA | FAKE miss | AUC | EER |
|---|---|---|---|---|---|---|---|
| p6_p0_b0_v1 | A_then_B | 56 | 0.42857142857142855 | 0.0 | 0.8571428571428571 | 0.7984693877551021 | 0.25 |
| p6_p0_b0_v1 | B_then_A | 56 | 0.4642857142857143 | 0.0 | 0.9285714285714286 | 0.7436224489795918 | 0.32142857142857145 |
| p6_p0_b0_v1 | clean | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b0_v1 | gain_half | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b0_v1 | gain_quarter | 56 | 0.08928571428571429 | 0.14285714285714285 | 0.03571428571428571 | 0.9846938775510203 | 0.0357142857142857 |
| p6_p0_b0_v1 | noise5 | 56 | 0.5 | 0.0 | 1.0 | 0.6645408163265306 | 0.4642857142857143 |
| p6_p0_b0_v1 | opus64 | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b1_v1 | A_then_B | 56 | 0.44642857142857145 | 0.0 | 0.8928571428571429 | 0.8214285714285714 | 0.2142857142857143 |
| p6_p0_b1_v1 | B_then_A | 56 | 0.48214285714285715 | 0.0 | 0.9642857142857143 | 0.7704081632653061 | 0.3214285714285714 |
| p6_p0_b1_v1 | clean | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b1_v1 | gain_half | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b1_v1 | gain_quarter | 56 | 0.03571428571428571 | 0.03571428571428571 | 0.03571428571428571 | 0.9961734693877551 | 0.0357142857142857 |
| p6_p0_b1_v1 | noise5 | 56 | 0.5 | 0.0 | 1.0 | 0.7053571428571429 | 0.32142857142857145 |
| p6_p0_b1_v1 | opus64 | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b2_v1 | A_then_B | 56 | 0.25 | 0.2857142857142857 | 0.21428571428571427 | 0.8584183673469389 | 0.2142857142857143 |
| p6_p0_b2_v1 | B_then_A | 56 | 0.26785714285714285 | 0.32142857142857145 | 0.21428571428571427 | 0.860969387755102 | 0.2142857142857143 |
| p6_p0_b2_v1 | clean | 56 | 0.017857142857142856 | 0.03571428571428571 | 0.0 | 1.0 | 0.0 |
| p6_p0_b2_v1 | gain_half | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b2_v1 | gain_quarter | 56 | 0.07142857142857142 | 0.10714285714285714 | 0.03571428571428571 | 0.9948979591836735 | 0.0357142857142857 |
| p6_p0_b2_v1 | noise5 | 56 | 0.19642857142857142 | 0.17857142857142858 | 0.21428571428571427 | 0.8660714285714286 | 0.2142857142857143 |
| p6_p0_b2_v1 | opus64 | 56 | 0.017857142857142856 | 0.03571428571428571 | 0.0 | 1.0 | 0.0 |
| p6_p0_b4_control_v1 | A_then_B | 56 | 0.30357142857142855 | 0.03571428571428571 | 0.5714285714285714 | 0.8520408163265306 | 0.1785714285714286 |
| p6_p0_b4_control_v1 | B_then_A | 56 | 0.3392857142857143 | 0.07142857142857142 | 0.6071428571428571 | 0.8392857142857143 | 0.21428571428571427 |
| p6_p0_b4_control_v1 | clean | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b4_control_v1 | gain_half | 56 | 0.017857142857142856 | 0.0 | 0.03571428571428571 | 1.0 | 0.0 |
| p6_p0_b4_control_v1 | gain_quarter | 56 | 0.125 | 0.21428571428571427 | 0.03571428571428571 | 0.9579081632653061 | 0.07142857142857142 |
| p6_p0_b4_control_v1 | noise5 | 56 | 0.35714285714285715 | 0.07142857142857142 | 0.6428571428571429 | 0.8239795918367346 | 0.21428571428571427 |
| p6_p0_b4_control_v1 | opus64 | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b4_v1 | A_then_B | 56 | 0.26785714285714285 | 0.07142857142857142 | 0.4642857142857143 | 0.8533163265306122 | 0.1785714285714286 |
| p6_p0_b4_v1 | B_then_A | 56 | 0.2857142857142857 | 0.07142857142857142 | 0.5 | 0.840561224489796 | 0.1785714285714286 |
| p6_p0_b4_v1 | clean | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_b4_v1 | gain_half | 56 | 0.017857142857142856 | 0.0 | 0.03571428571428571 | 1.0 | 0.0 |
| p6_p0_b4_v1 | gain_quarter | 56 | 0.125 | 0.21428571428571427 | 0.03571428571428571 | 0.9591836734693877 | 0.07142857142857142 |
| p6_p0_b4_v1 | noise5 | 56 | 0.35714285714285715 | 0.07142857142857142 | 0.6428571428571429 | 0.8214285714285714 | 0.2142857142857143 |
| p6_p0_b4_v1 | opus64 | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_rb_v1 | A_then_B | 56 | 0.23214285714285715 | 0.25 | 0.21428571428571427 | 0.8354591836734694 | 0.2142857142857143 |
| p6_p0_rb_v1 | B_then_A | 56 | 0.3392857142857143 | 0.35714285714285715 | 0.32142857142857145 | 0.7793367346938777 | 0.3571428571428571 |
| p6_p0_rb_v1 | clean | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p0_rb_v1 | gain_half | 56 | 0.017857142857142856 | 0.0 | 0.03571428571428571 | 1.0 | 0.0 |
| p6_p0_rb_v1 | gain_quarter | 56 | 0.07142857142857142 | 0.10714285714285714 | 0.03571428571428571 | 0.9821428571428572 | 0.03571428571428571 |
| p6_p0_rb_v1 | noise5 | 56 | 0.30357142857142855 | 0.10714285714285714 | 0.5 | 0.7321428571428572 | 0.3571428571428571 |
| p6_p0_rb_v1 | opus64 | 56 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| p6_p1_b0_v1 | A_then_B | 56 | 0.5 | 0.0 | 1.0 | 0.8061224489795917 | 0.2857142857142857 |
| p6_p1_b0_v1 | B_then_A | 56 | 0.5 | 0.0 | 1.0 | 0.7442602040816326 | 0.30357142857142855 |
| p6_p1_b0_v1 | clean | 56 | 0.03571428571428571 | 0.0 | 0.07142857142857142 | 0.9961734693877551 | 0.03571428571428571 |
| p6_p1_b0_v1 | gain_half | 56 | 0.03571428571428571 | 0.0 | 0.07142857142857142 | 0.9961734693877551 | 0.03571428571428571 |
| p6_p1_b0_v1 | gain_quarter | 56 | 0.03571428571428571 | 0.0 | 0.07142857142857142 | 0.9961734693877551 | 0.03571428571428571 |
| p6_p1_b0_v1 | noise5 | 56 | 0.5 | 0.0 | 1.0 | 0.6989795918367347 | 0.3571428571428571 |
| p6_p1_b0_v1 | opus64 | 56 | 0.05357142857142857 | 0.0 | 0.10714285714285714 | 0.9987244897959184 | 0.0357142857142857 |
| p6_p1_b1_v1 | A_then_B | 56 | 0.5 | 0.0 | 1.0 | 0.8061224489795917 | 0.2857142857142857 |
| p6_p1_b1_v1 | B_then_A | 56 | 0.5 | 0.0 | 1.0 | 0.7442602040816326 | 0.30357142857142855 |
| p6_p1_b1_v1 | clean | 56 | 0.03571428571428571 | 0.0 | 0.07142857142857142 | 0.9961734693877551 | 0.03571428571428571 |
| p6_p1_b1_v1 | gain_half | 56 | 0.03571428571428571 | 0.0 | 0.07142857142857142 | 0.9961734693877551 | 0.03571428571428571 |
| p6_p1_b1_v1 | gain_quarter | 56 | 0.03571428571428571 | 0.0 | 0.07142857142857142 | 0.9961734693877551 | 0.03571428571428571 |
| p6_p1_b1_v1 | noise5 | 56 | 0.5 | 0.0 | 1.0 | 0.6989795918367347 | 0.3571428571428571 |
| p6_p1_b1_v1 | opus64 | 56 | 0.05357142857142857 | 0.0 | 0.10714285714285714 | 0.9987244897959184 | 0.0357142857142857 |
| p6_p1_b2_v1 | A_then_B | 56 | 0.16071428571428573 | 0.0 | 0.32142857142857145 | 0.8698979591836735 | 0.2142857142857143 |
| p6_p1_b2_v1 | B_then_A | 56 | 0.21428571428571427 | 0.14285714285714285 | 0.2857142857142857 | 0.8482142857142858 | 0.25 |
| p6_p1_b2_v1 | clean | 56 | 0.05357142857142857 | 0.0 | 0.10714285714285714 | 0.9961734693877551 | 0.0357142857142857 |
| p6_p1_b2_v1 | gain_half | 56 | 0.05357142857142857 | 0.0 | 0.10714285714285714 | 0.9961734693877551 | 0.0357142857142857 |
| p6_p1_b2_v1 | gain_quarter | 56 | 0.05357142857142857 | 0.0 | 0.10714285714285714 | 0.9961734693877551 | 0.0357142857142857 |
| p6_p1_b2_v1 | noise5 | 56 | 0.19642857142857142 | 0.03571428571428571 | 0.35714285714285715 | 0.8545918367346939 | 0.1785714285714286 |
| p6_p1_b2_v1 | opus64 | 56 | 0.08928571428571429 | 0.0 | 0.17857142857142858 | 1.0 | 0.0 |

Primary matched comparison and post-hoc retention-proxy investigation:

```json
{
  "comparison": [
    "p6_p0_b4_v1",
    "p6_p0_b4_control_v1"
  ],
  "equal_view_fairness": "PASS",
  "contrast": {
    "estimate": -0.07142857142857142,
    "parents": 7,
    "ci95": [
      -0.13095238095238096,
      -0.023809523809523808
    ],
    "ci_status": "DESCRIPTIVE_PERCENTILE"
  },
  "clean_guardrail_passes": true,
  "clean_far_change": 0.0,
  "clean_miss_change": 0.0,
  "screening_improvement": true,
  "scope": "Single-seed exploratory screen; no significance or confirmed novelty"
}
```

The embedding-similarity proxy is available without labels; labels assess only its development association afterward. It is not a validated measure of preserved spoof evidence and cannot justify a new weighted loss by itself. Partial-backbone sanity metadata follows; failed predecessors are retained in the experiment inventory.

```json
{
  "status": "SMOKE_TESTED",
  "scope": "OVERFIT_SANITY_RESOURCE_PREFLIGHT",
  "training_policy": "PARTIAL_BACKBONE_FINETUNE",
  "seed": 5806,
  "sources": [
    "ASVspoof2019_LA:LA_T_1075018",
    "ASVspoof2019_LA:LA_T_1113457"
  ],
  "data_role": "TRAIN_ONLY",
  "steps": 5,
  "batch": 1,
  "gradient_accumulation": 2,
  "effective_batch": 2,
  "preprocessing_seconds": 0.04903580000063812,
  "head_only_forward_seconds": 0.6505297000003338,
  "head_only_backward_seconds": 0.36144469999999274,
  "head_only_gradient_norm": 4.490053653717041,
  "elapsed_seconds": 9.81548440000006,
  "trainable_parameters": 646481,
  "total_parameters": 1308401,
  "selected_backbone_kernels": [
    "expanded_conv_10/squeeze_excite/Conv/kernel:0",
    "expanded_conv_10/squeeze_excite/Conv_1/kernel:0",
    "expanded_conv_10/project/kernel:0",
    "Conv_1/kernel:0"
  ],
  "changed_backbone_kernels": [
    "expanded_conv_10/squeeze_excite/Conv/kernel:0",
    "expanded_conv_10/squeeze_excite/Conv_1/kernel:0",
    "expanded_conv_10/project/kernel:0",
    "Conv_1/kernel:0"
  ],
  "gradients": {
    "expanded_conv_10/squeeze_excite/Conv/kernel:0": 0.2547600269317627,
    "expanded_conv_10/squeeze_excite/Conv/bias:0": 0.020035885274410248,
    "expanded_conv_10/squeeze_excite/Conv_1/kernel:0": 0.23635178804397583,
    "expanded_conv_10/squeeze_excite/Conv_1/bias:0": 0.02518070861697197,
    "expanded_conv_10/project/kernel:0": 0.32761093974113464,
    "Conv_1/kernel:0": 0.45991718769073486,
    "temporal_lstm/lstm_cell/kernel:0": 1.2389395236968994,
    "temporal_lstm/lstm_cell/recurrent_kernel:0": 0.22725650668144226,
    "temporal_lstm/lstm_cell/bias:0": 0.07371043413877487,
    "classifier_dense/kernel:0": 0.3783886134624481,
    "classifier_dense/bias:0": 0.07067957520484924,
    "probability_fake/kernel:0": 0.1936441957950592,
    "probability_fake/bias:0": 0.007385313510894775
  },
  "frozen_variables_unchanged": 200,
  "batchnorm_frozen": 34,
  "save_load_max_residual": 0.0,
  "peak_ram_mib": 837.37890625,
  "gpu_available": false,
  "checkpoint_bytes": 5916699,
  "initial_loss": 0.677266925573349,
  "final_loss": 0.5458562970161438,
  "limitations": "Two train recordings / five updates; no development quality comparison; prefix/middle layers intentionally frozen",
  "output_hashes": {
    "progress.json": "dd00d770c969f32246c0608670def2d88f700ffbf42e1758848880276d3abf57",
    "sanity.keras": "0e36448923fd41248844feb886a2858a3222aa8233c755ba8cdeef4aef2d2ebb",
    "training_history.csv": "fdd71d5e279b6fdb990d118f86db38757d155c1b12d6b707501c7ef460ede7b6"
  }
}
```

Post-hoc retention-proxy associations (repeated development views; undefined constant-class association is null):

| Policy | Label | View records | Spearman rho | Interpretation |
|---|---|---|---|---|
| P0 | 0 | 168 | 0.047499509735996974 | Post-hoc association; repeated views, no p-value or forensic-retention validation |
| P0 | 1 | 168 | -0.8522095356590662 | Post-hoc association; repeated views, no p-value or forensic-retention validation |
| P1 | 0 | 168 | None | Post-hoc association; repeated views, no p-value or forensic-retention validation |
| P1 | 1 | 168 | -0.8031899230373969 | Post-hoc association; repeated views, no p-value or forensic-retention validation |

Classwise aggregate interpretation: amplitude-only training does not resolve noise/order failures; ordinary channel/finite RawBoost training reduces FAKE misses with stressed REAL false alarms. The primary consistency improvement is a standard-baseline result; P1 gain augmentation is redundant on identical gain-invariant features. No method is declared universally best.

| Run | Clean REAL FA | Clean FAKE miss | Mean3-stress REAL FA | Mean3-stress FAKE miss |
|---|---|---|---|---|
| p6_p0_b0_v1 | 0.0 | 0.0 | 0.0 | 0.9285714285714286 |
| p6_p0_b1_v1 | 0.0 | 0.0 | 0.0 | 0.9523809523809524 |
| p6_p0_b2_v1 | 0.03571428571428571 | 0.0 | 0.2619047619047619 | 0.21428571428571427 |
| p6_p0_b4_control_v1 | 0.0 | 0.0 | 0.05952380952380952 | 0.6071428571428571 |
| p6_p0_b4_v1 | 0.0 | 0.0 | 0.07142857142857142 | 0.5357142857142857 |
| p6_p0_rb_v1 | 0.0 | 0.0 | 0.23809523809523808 | 0.34523809523809523 |
| p6_p1_b0_v1 | 0.0 | 0.07142857142857142 | 0.0 | 1.0 |
| p6_p1_b1_v1 | 0.0 | 0.07142857142857142 | 0.0 | 1.0 |
| p6_p1_b2_v1 | 0.0 | 0.10714285714285714 | 0.05952380952380952 | 0.32142857142857145 |

## 11. Phase7 Findings

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

| experiment_id | seed | training_policy | epochs | clean_REAL_FA | clean_FAKE_miss | stress_REAL_FA | stress_FAKE_miss | B_clean_REAL_FA | B_stress_REAL_FA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 6107 | HEAD_ONLY | 1 | 0.232143 | 0.446429 | 0.258929 | 0.675595 | 0.111111 | 0.127315 |
| p7_f1_s6107_v1 | 6107 | PARTIAL_BACKBONE_FINETUNE | 1 | 0.258929 | 0.267857 | 0.235119 | 0.717262 | 0.145833 | 0.094907 |
| p7_h0_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.017857 | 0.107143 | 0.636905 | 0.267857 | 0.118056 | 0.537037 |
| p7_h0_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.026786 | 0.071429 | 0.389881 | 0.613095 | 0.104167 | 0.298611 |
| p7_h0_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.071429 | 0.071429 | 0.779762 | 0.321429 | 0.250000 | 0.724537 |
| p7_h1_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.008929 | 0.098214 | 0.392857 | 0.226190 | 0.145833 | 0.534722 |
| p7_h1_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.000000 | 0.133929 | 0.000000 | 0.627976 | 0.034722 | 0.009259 |
| p7_h1_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.044643 | 0.089286 | 0.550595 | 0.184524 | 0.263889 | 0.567130 |
| p7_h2_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.035714 | 0.080357 | 0.113095 | 0.410714 | 0.180556 | 0.182870 |
| p7_h2_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.008929 | 0.071429 | 0.175595 | 0.398810 | 0.118056 | 0.152778 |
| p7_h2_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.044643 | 0.062500 | 0.261905 | 0.354167 | 0.201389 | 0.275463 |
| p7_h3_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.000000 | 0.116071 | 0.366071 | 0.315476 | 0.083333 | 0.416667 |
| p7_h3_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.000000 | 0.098214 | 0.080357 | 0.461310 | 0.097222 | 0.097222 |
| p7_h3_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.026786 | 0.089286 | 0.199405 | 0.375000 | 0.180556 | 0.245370 |
| p7_h4_s6107_v1 | 6107 | HEAD_ONLY | 6 | 0.044643 | 0.107143 | 0.919643 | 0.077381 | 0.083333 | 0.930556 |
| p7_h4_s6207_v1 | 6207 | HEAD_ONLY | 6 | 0.017857 | 0.080357 | 0.824405 | 0.354167 | 0.125000 | 0.837963 |
| p7_h4_s6307_v1 | 6307 | HEAD_ONLY | 6 | 0.017857 | 0.098214 | 0.895833 | 0.125000 | 0.138889 | 0.849537 |

Primary H3–H2 mean signed miss contrast = -0.003968; independent-seed SD = 0.081741. Individual effects: {"6107": -0.09523809523809525, "6207": 0.06249999999999999, "6307": 0.020833333333333322}. Only1/3blocks favor consistency; 0/3pass both registered clean guardrails. The Phase6 favorable single-seed conclusion does not establish a stable accepted improvement here. Standard consistency is not a new method; these are descriptive development results.

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

```json
{
  "clean_guardrails_by_seed": {
    "6107": {
      "FAKE_miss_delta": 0.03571428571428571,
      "REAL_FA_delta": -0.03571428571428571,
      "passes": false
    },
    "6207": {
      "FAKE_miss_delta": 0.026785714285714288,
      "REAL_FA_delta": -0.008928571428571428,
      "passes": false
    },
    "6307": {
      "FAKE_miss_delta": 0.026785714285714288,
      "REAL_FA_delta": -0.01785714285714286,
      "passes": false
    }
  },
  "first": "H3",
  "group_support": "LIMITED_CLUSTER_PRECISION",
  "mean": -0.003968253968253978,
  "p_value": null,
  "paired_seed_effects": {
    "6107": -0.09523809523809525,
    "6207": 0.06249999999999999,
    "6307": 0.020833333333333322
  },
  "primary": true,
  "second": "H2",
  "seed_bootstrap_ci95": [
    -0.09523809523809525,
    0.06249999999999999
  ],
  "seed_count": 3,
  "seed_std": 0.08174145656791475,
  "speaker_bootstrap_ci95": [
    -0.01884920634920636,
    0.01190476190476189
  ],
  "speaker_count": 7,
  "uncertainty": "DESCRIPTIVE_ONLY; speaker interval conditional on trained seeds; seed interval conditional on same7speakers; only3seed blocks and7clusters"
}
```

Seed std is computed across independently trained blocks, not bootstrap replicates. Speaker bootstrap resamples7known speaker effects while retaining all paired conditions and trained seeds. Seed bootstrap separately resamples3paired blocks on the same speakers. Both intervals are descriptive with LIMITED_CLUSTER_PRECISION; no strong population significance or automatic hierarchical-bootstrap guarantee. Secondary comparisons are exploratory, no selective p-values.

## 12. Phase8 Research Direction

# Phase8 research plan — 31 October 2026

The starting observation is a conflict between REAL false alarms and FAKE misses, not established success of consistency. Phase7 H3−H2 stressed miss effects were −.095238, +.062500, +.020833; mean −.003968, seed SD .081741. All three blocks failed the paired clean guardrails. Its one-epoch F1 improved clean AUC but worsened stressed misses. These results motivate measuring both classes separately.

C1 remains SUPPORTING_EMPIRICAL_AND_ENGINEERING_CONTRIBUTION. Ordered-channel controls contribute an auditable empirical protocol, but additional bitrates and plots do not distinguish it from RADAR/AT-ADD. C2 is the central research question: can a genuinely distinct mechanism improve both classwise stressed risks with acceptable clean risk? C3 remains DEFERRED until a credible C2 and independent calibration exist.

Phase8 first reviews nearby objectives, then specifies M1 weighted BCE, M2 group robustness, and proposed order-aware M3 variants. Algebraic reduction is a rejection criterion. No M3 loss is implemented without TECHNICALLY_DISTINCT_CANDIDATE status. The current candidates do not pass; see PHASE8_NOVELTY_DECISION.md and PHASE8_METHOD_REDIRECTION.md.

Engineering proceeds regardless: a separate trainer supports fresh ImageNet initialization, deterministic sampling/dropout, multiple epochs/seeds, head training and warmup plus actual CNN fine-tuning, Adam checkpointing, bounded resource use, declared checkpoint selection and exported-model evaluation. Tiny pilots are implementation evidence. The author launches medium/full research runs explicitly; Phase8 does not automatically execute them.

Use the existing medium cohort and foundation train/model-selection-development manifests. Never parse the reserved 7,341 calibration records or official evaluation. A01–A06 are seen attack families; generator checkpoint and original source lineage remain unknown. MLAAD-tiny and all development channel compositions are already exposed research data, not untouched confirmation.

Success criteria are separate: implementation readiness, full-training lifecycle readiness, scientific readiness. A successful trainer or a favorable tiny score cannot establish three international contributions. The final evidence-generated master report records the actual gates and blockers.

## 13. Mathematical Method

# C2 method specification

## Problem and assumptions

Let y=0 REAL and y=1 FAKE, pθ(x) the sigmoid FAKE score, and ℓ(y,p)=−y log p−(1−y)log(1−p). Clip probabilities only for numerical stability. Labels are inherited from official ASV protocols; post-generation channel transformations preserve the source label, but may erase detectable evidence. Paired views share a recording, not an independent sample. Official speakers, source byte hashes and A01–A06 are verified fields. They are not verified generator checkpoints or original-utterance lineage. Development examples/conditions and repeated training seeds are dependent observations.

Define R0(c)=E[ℓ(0,pθ(c(x)))|y=0] and R1(a,c)=E[ℓ(1,pθ(c(x)))|y=1,a]. Decision metrics are P(p≥.5|REAL,c) and P(p<.5|FAKE,a,c); BCE caps do not guarantee those rates. The research direction contains clean, noise, codec, noise→codec and codec→noise. Current training controls use clean/gain/noise10/Opus24; ordered chains remain already-used development diagnostics and are not secretly inserted into training.

## M1 — standard baseline

Implemented WEIGHTED_BCE uses L1=(1/N)Σi wyᵢ ℓi, with explicit weights [1,1.5]. It does not divide by the sum of weights. Thus both relative class emphasis and overall gradient scale are declared. This choice is a baseline setting, not optimized or established optimal for the imbalanced foundation. Natural and class-balanced sampling are separate registered protocols. Weighted sampling plus class weights changes the target distribution twice and must be disclosed.

An alternative mathematical specification is minθ Eℓ subject to R0(clean)≤b0 and R1(clean)≤b1. Its standard squared-hinge relaxation adds λΣy[max(0,Ry−by)]². `clean_constraint` has independent gradient tests but is NOT wired into a training family. No claim of a trained primal-dual/constrained optimizer is made. The implemented guarded checkpoint selector is a different operation, described below.

## M2 — standard group-robust control

Groups are REAL×condition plus A01…A06 FAKE×condition; REAL never inherits an UNKNOWN generator label. For the channel bank there are 28 groups. Every effective update draws sources from all seven class/attack groups and expands every sampled source through all four training conditions. Repeated sources remain repeated exposure. Group support and counts are checked before taking a gradient.

L2(θ,q)=Σg qg (1/ng)Σi∈g ℓi. Initialize q uniformly. Train θ using the previous q, then compute pre-update losses on every group and set q′∝q exp(η R̂g); qnew=(1−ε)q′+ε/G. η=.01, ε=.05. This is an explicit smoothed group-DRO baseline, not an exact reproduction of Sagawa Algorithm1 or an unrestricted simplex-max guarantee. The floor prevents zero weights; it does not guarantee optimization or population robustness. There is no additional ERM term. Each microbatch receives coefficients qg/ng computed over the entire effective update, so accumulation cannot accidentally normalize every microbatch separately.

M2 changes sampling as well as weighting. A matched uniform-group control is required before attributing an effect solely to adversarial q. All-view exposure also costs more than a single-view ERM update. Compare supervised views, forward/backward calls and runtime; equal update counts are not equal compute.

## M3 — proposed mechanisms and rejection proofs

Let u=C(N(x)), v=N(C(x)), with the same fixed absolute noise realization and insertion reference. Noncommutativity u≠v is a signal-chain fact; it does not establish a new learning principle.

M3a proposed allocating supervision by a deterministic signal discrepancy s(u,v): L=α(s,y)ℓ(y,p(u))+(1−α(s,y))ℓ(y,p(v)). For fixed α this is importance-weighted augmentation over the expanded source/view distribution. Naming α an order coefficient does not distinguish the optimizer from weighted ERM. If α is learned from losses, its comparator becomes adaptive weighting/DRO, not a claim of automatic novelty. REJECTED for the presently specified version.

M3b proposed prioritizing the harder order with max(ℓu,ℓv). The identity max(a,b)=(a+b+|a−b|)/2 shows that the apparent order disagreement term is exactly worst-of-two robust loss. Relabeling the two conditions inside a group-DRO/CVaR objective does not add a mechanism. REJECTED.

M3c proposed matching the signed prediction commutator p(u)−p(v) to an acoustic discrepancy. No verified source-specific relation maps waveform discrepancy to surviving FAKE evidence. It could encode loudness, clipping or codec delay and suppress difficult supervision. The Phase7 ImageNet-cosine proxy is NON_SPECIFIC and cannot supply this missing target. PROPOSED, NOT_TECHNICALLY_JUSTIFIED, NOT_IMPLEMENTED. No equality of embeddings/predictions across orders is imposed.

## Optimization, selection, falsification and costs

Adam uses a common LR per stage (no discriminative parameter LR claim), linear half-decay inside each stage, and a declared warmup/fine-tune transition. Warmup masks backbone gradients; the tail variables/slots are registered in advance, frozen BN stays unchanged, then real CNN gradients are applied during fine-tuning. This retains optimizer state across stages; it differs from resetting Adam after warmup. Head-only uses the sealed frozen-feature bank. Fine-tuning streams P0 tensors through the differentiable ImageNet CNN.

Selection CLEAN_BCE uses only primary A clean BCE. CLEAN_RISK_CONSTRAINED_ROBUSTNESS first measures both clean errors at .5, rejects checkpoints above absolute caps, then minimizes the mean balanced class error across noise5 and both orders. A rejected epoch is INFEASIBLE_UNDER_REGISTERED_CONSTRAINT. No feasible checkpoint means no selected export; no fallback to an infeasible model. Guardrails .10 REAL FA and .15 FAKE miss are exploratory absolute caps, distinct from Phase7 paired 2pp margins and from training constraints. Final selected-model evaluation includes all seven conditions, REAL-only B and per-attack FAKE groups.

Potential failures: overfit codec artifacts; increased REAL FA; missed hard attacks; q focusing on noisy small groups; underfit warmup; incompatible narrow-band inputs; unknown-source leakage. Falsify a claimed improvement with fresh matched seeds, both clean guardrails, both stressed class errors, matched group sampling, uniform q, equal-view supervision, consistency and finite/online RawBoost distinctions. M3-specific ablations are not fabricated because M3 is not selected. See METHOD_COMPARISON_PLAN and INTERNATIONAL_NOVELTY_MATRIX.

ERM has one supervised view/source/update; equal-view and M2 have four on the channel bank, consistency has two forwards per sampled view. Inference architecture remains the same student. Additional training overhead is measured in actual tiny runs and is not extrapolated as a GPU latency result. No full method convergence, independent confirmation or calibrated FAR guarantee is established.

## 14. Implementation

New modules: phase8_training.py, phase8_objectives.py; CLI phase8_experiment.py; standard helpers/source data remain inherited and unchanged.

# Full training readiness — evidence generated

FULL_TRAINING_CODE_READY: **YES** (CPU, tested TF2.15 environment).

NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: **NO**.

Gate1 IMPLEMENTATION_READY: YES. Gate2 FULL_TRAINING_CODE_READY: YES. Gate3 SCIENTIFIC_EXPERIMENT_READY for a novel C2: NO. No full-data run or GPU training is implied.

```json
{
  "regression": true,
  "exact_resume": true,
  "representation": true,
  "historical_integrity": true,
  "registered_configs": true,
  "resource_preflight": true,
  "infeasible_guarded_early_stop": true,
  "matched_baseline_initialization": true,
  "validation_optimization_parity": true
}
```

Actual regression:

| run | failures | errors | skipped | successful |
| --- | --- | --- | --- | --- |
| 149 | 0 | 0 | 0 | True |
| 40 | 0 | 0 | 0 | True |

CPU fresh-process resume compares final tensor bytes, Adam bytes, source/view order, loss/LR journals, counters, history and selected-model scores. Both head and warmup→partial schedules require exact hashes with tolerance0. Checkpoints occur only at optimizer boundaries; no claim of restoring half an accumulated microbatch. The full CNN export was checked against recorded scores within1e-6, warmup backbone/BN values were unchanged and partial kernels changed. Three seeds have distinct initialization hashes. A guarded tiny run actually early-stopped with no feasible export.

Frozen BN stays unchanged. Warmup masks backbone gradients while pre-registering tail slots; this is explicitly different from toggling trainable flags each epoch or resetting Adam at the transition. Both stages use one common scalar LR; no discriminative-LR implementation is claimed. Head-only can use detached embeddings; partial always runs the differentiable CNN on P0 tensors. Finite banks are not online augmentation.

Resource preflight verifies data byte hashes/known partition separation, complete lifecycle update capacity and conservative retained-checkpoint storage against15GiB reserve. Runtime guards check disk, peak process RAM, elapsed invocation time, finite gradients and predictions. Model selection and fixed threshold are explicit. Pipeline supports medium and full-foundation data without constructing an unbudgeted Mel cache. GPU source portability is prepared but its runtime, mixed precision, long-run convergence and arbitrary-shift risk remain unverified.

Verified historical files: 6243; accepted DOCX remains SHA91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073. 98 configuration identities are registered, not trained models. Actual run entries include first-version pilots, optimization repeats and interrupted counterparts; they must not inflate independent-seed counts. Missing metrics are null/absent, never zero.

Detailed machine-readable evidence: outputs/jaes_v2/phase8/audit/readiness_evidence.json. Standard baseline launch commands are in FULL_TRAINING_LAUNCH_PACKAGE; scientific blockers remain in NOVELTY_DECISION and METHOD_COMPARISON_PLAN.

## 15. Novelty Comparison

Detailed primary-source matrix: [PHASE8_INTERNATIONAL_NOVELTY_MATRIX.md](PHASE8_INTERNATIONAL_NOVELTY_MATRIX.md). SONAR is now FULL_TEXT_VERIFIED from the official PDF. M1/M2 are standard, M3a/b rejected by algebra; M3c lacks a justified evidence target. No world-first assertion.

## 16. Falsification

# Method redirection after the M3 entry gate

## Direction1: source-specific evidence survival under known generation

Acquire a small licensed paired corpus with original speech, generator version/checkpoint, exact source alignment, and the generated counterpart. Define an observable generator residual or a controlled injected synthesis cue before channel transformations; do not substitute ImageNet cosine. Test whether an estimator tracks the survival of that specified cue after each ordered chain, controlling RMS, delay, bandwidth and silence. Split by original utterance and speaker, then hold out generators before labels are opened.

The gap is a justified evidence target rather than another robust aggregation. Closest methods are forensic source similarity, SONAR frequency coupling and codec-aware DK-CAST. A genuinely different identifiable target and estimator must be demonstrated before weighting training. Required falsification: exact identity/lossless positives, cue-removal negatives, nuisance-only changes, shuffled estimator targets, severity-only baseline, held-out generator prediction. Fail if the target is not identifiable after alignment or nuisance controls explain it. Data lineage, license and cue semantics are currently missing. PROPOSED; no data acquired or algorithm implemented.

## Direction2: order-conditioned uncertainty from an explicit channel model

Study whether a measured codec/noise generative channel supplies a predictive uncertainty interval for source evidence that is not recoverable from one distorted view. Start with a tractable synthetic signal/known codec experiment and derive identifiability or impossibility conditions; separate uncertain evidence from confident REAL. Compare channel likelihood modeling with ordinary augmentation, heteroscedastic weighting, worst-order DRO, adversarial training and selective classification. A new theorem, estimator or identifiable interaction is needed; a penalty called a commutator is insufficient.

Falsification: channel mismatch, swapped order metadata, no-source-pair baseline, equal compute DRO and channel-quality prediction controls. Failure occurs if order labels alone explain performance, uncertainty is uncalibrated, or the mechanism reduces to standard weighted risk. C3 decisions remain deferred; no abstention model is trained before independent calibration is locked. Known channel parameters and held-out source experiments are currently missing. PROPOSED, not claimed technically distinct.

These directions are bounded research proposals, not substitute contributions or promised journal acceptance. First run the prepared standard baselines to determine whether a difficult unexplained residual remains worth pursuing.

## 17. Baseline Matrix

# Matched comparison plan

Registered configuration hashes live in `outputs/jaes_v2/phase8/protocol/baselines_v1.json`; tiny lifecycle configs use `protocol/tiny_v2.json`. Neither enables calibration/evaluation. No full run was automatically authorized by generating configs.

Main architecture: MobileNetV3Small-LSTM128, fresh ImageNet backbone and independent head seeds6107/6207/6307. Medium train640, development224A+144B; foundation train25,380 (2,580REAL/22,800FAKE), development17,503 excludes reserved speakers. Never interpret oversampled REAL views as new recordings. Natural and balanced protocols have different target distributions and separate IDs.

| Control | Objective/exposure | Research role |
|---|---|---|
| A0 | ERM, clean | ordinary representation baseline |
| A1 | one sampled clean/gain/noise10/Opus24 | channel augmentation |
| M1 | same bank, weighted BCE[1,1.5] | ordinary class emphasis |
| A2 | equal supervision of all four views | all-view exposure control |
| M2/A3 | seven source groups, all four views, adaptive q | smoothed group robustness |
| Consistency | two supervised forwards, .1 probability MSE | ordinary invariance |
| A4 | clean plus two fixed RawBoost realizations | finite bank; not original online reproduction |
| B0 / M3 ablations | NOT_CREATED | entry gate failed |

All three-seed medium/foundation families are prepared; longer jobs remain NOT_EXECUTED. Two-stage partial schedules use1warmup+20finetune epochs; head-only20epochs. These are neither equal-update nor compute-matched. Compare the actual journal, updates, views, repeated-source frequencies, trainable parameters, wall time, validation cost and checkpoints. A fair M2 effect additionally requires uniform q **with the same group-balanced source sampler**; that dedicated new experiment is not yet registered. The natural-sampling equal-view family alone cannot isolate q. Do not promote a DRO advantage to a mechanism claim without that control.

Selection uses primary A development only. Report fixed-threshold .5 REAL FA, FAKE miss, AUC/EER, probability quality, all conditions, all seen attacks and B one-class nulls. The guarded selector uses prospective absolute caps, not the retrospective choice of whichever class looks favorable. Primary scientific contrasts require paired initialization/order within seed and paired known speakers. Report separate seed SD and speaker bootstrap uncertainty; views and seed×recording products are not independent samples. Tiny lifecycle cohorts cannot support population significance.

To escalate a future M3: precise non-equivalent mechanism; locked matched controls including sampling; support across independent training seeds with both clean errors acceptable; independent source/license/overlap and calibration protocol locked before any confirmation. Existing order compositions have been inspected since Phase1 and cannot become untouched tests through renaming. Architecture transfer to ShuffleNet comes only after a credible MobileNet effect; no six-detector retraining or cascade is launched.

## 18. Actual Phase8 Experiments

# Phase8 actual experiment results

| experiment_id | status | scope | steps |
| --- | --- | --- | --- |
| p8_tiny_dro_head_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_dro_head_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_resume_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_resume_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v1 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6207_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6207_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6307_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_head_s6307_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_resume_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_resume_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_erm_partial_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_guarded_infeasible_head_s6107_v3 | NO_FEASIBLE_CHECKPOINT | TINY_LIFECYCLE | 2 |
| p8_tiny_weighted_head_s6107_v2 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |
| p8_tiny_weighted_head_s6107_v3 | PILOT_EVALUATED | TINY_LIFECYCLE | 4 |

All completed quality rows use12train and14development sources (12primary A,2REAL-only B), A01–A06 seen attacks. These are lifecycle/objective pilots, not a generalization study. Original implementation, exact-resume pairs and optimized-validation reruns are not independent scientific replications. M1/M2 optimize standard losses; no M3 run exists. Full medium/foundation training, independent calibration/confirmation, GPU and cascade are NOT_EXECUTED.

Guarded negative integration: NO_FEASIBLE_CHECKPOINT, epoch1 of a declared5epoch maximum, patience1; no export fallback. Historical ordinary consistency and RawBoost trade-offs remain unchanged.

CPU exact-resume evidence:

```json
{
  "head": {
    "status": "PASS",
    "checks": {
      "final_weights": true,
      "optimizer": true,
      "sample_order_views_losses_lr": true,
      "step": true,
      "counter": true,
      "history": true,
      "export_scores": true
    },
    "numerical_tolerance": 0,
    "scope": "CPU TF2.15 same environment; fresh processes",
    "final_weights_hash": "5b72bf0e6ff9251662ebe0dce9944004267dd3bb7fb3c8693c97bb76d08760bb",
    "optimizer_hash": "a95699f9c07e107d731f9b09934637854839e197a8000ef57a48191f2a8d7827",
    "steps": 4
  },
  "partial": {
    "status": "PASS",
    "checks": {
      "final_weights": true,
      "optimizer": true,
      "sample_order_views_losses_lr": true,
      "step": true,
      "counter": true,
      "history": true,
      "export_scores": true
    },
    "numerical_tolerance": 0,
    "scope": "CPU TF2.15 same environment; exact optimizer boundaries; fresh processes; warmup/partial transition included",
    "final_weights_hash": "294eb9a9177a8a3c9637dc6c2167b8c69bc43d3dc1ed4540f197eae0c5f6a520",
    "optimizer_hash": "2057f8b1483bdb01ca1c073a599c6ca7d906af0c04d38e8b46ae4ca7725b3954",
    "steps": 4
  }
}
```

Selected-model classwise values and measured costs are in the model-free per_run.csv/classwise_conditions.json; each run also retains all seen-attack metrics with one-class AUC/REAL-FA nulls. No posthoc threshold tuning or claim of superiority from tiny scores.

## 19. Classwise Results

| experiment_id | clean_REAL_FA | clean_FAKE_miss | stress_REAL_FA | stress_FAKE_miss | clean_AUC |
| --- | --- | --- | --- | --- | --- |
| p8_tiny_dro_head_s6107_v2 | 1.0 | 0.0 | 1.0 | 0.0 | 0.3611111111111111 |
| p8_tiny_dro_head_s6107_v3 | 1.0 | 0.0 | 1.0 | 0.0 | 0.3611111111111111 |
| p8_tiny_erm_head_s6107_resume_v2 | 0.5 | 0.5 | 0.3333333333333333 | 0.611111111111111 | 0.5833333333333334 |
| p8_tiny_erm_head_s6107_resume_v3 | 0.5 | 0.5 | 0.3333333333333333 | 0.611111111111111 | 0.5833333333333334 |
| p8_tiny_erm_head_s6107_v1 | 0.5 | 0.5 | 0.3333333333333333 | 0.611111111111111 | 0.5833333333333334 |
| p8_tiny_erm_head_s6107_v2 | 0.5 | 0.5 | 0.3333333333333333 | 0.611111111111111 | 0.5833333333333334 |
| p8_tiny_erm_head_s6107_v3 | 0.5 | 0.5 | 0.3333333333333333 | 0.611111111111111 | 0.5833333333333334 |
| p8_tiny_erm_head_s6207_v2 | 0.3333333333333333 | 0.5 | 0.8333333333333334 | 0.2222222222222222 | 0.6111111111111112 |
| p8_tiny_erm_head_s6207_v3 | 0.3333333333333333 | 0.5 | 0.8333333333333334 | 0.2222222222222222 | 0.6111111111111112 |
| p8_tiny_erm_head_s6307_v2 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 | 0.6666666666666666 | 0.611111111111111 |
| p8_tiny_erm_head_s6307_v3 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 | 0.6666666666666666 | 0.611111111111111 |
| p8_tiny_erm_partial_s6107_resume_v2 | 0.6666666666666666 | 0.16666666666666666 | 1.0 | 0.0 | 0.6388888888888888 |
| p8_tiny_erm_partial_s6107_resume_v3 | 0.6666666666666666 | 0.16666666666666666 | 1.0 | 0.0 | 0.6388888888888888 |
| p8_tiny_erm_partial_s6107_v2 | 0.6666666666666666 | 0.16666666666666666 | 1.0 | 0.0 | 0.6388888888888888 |
| p8_tiny_erm_partial_s6107_v3 | 0.6666666666666666 | 0.16666666666666666 | 1.0 | 0.0 | 0.6388888888888888 |
| p8_tiny_weighted_head_s6107_v2 | 0.8333333333333334 | 0.0 | 1.0 | 0.0 | 0.6666666666666667 |
| p8_tiny_weighted_head_s6107_v3 | 0.8333333333333334 | 0.0 | 1.0 | 0.0 | 0.6666666666666667 |

These tiny selected cohorts are for lifecycle testing; stress means here are recording-weighted condition means. Full per-condition/per-attack/null-denominator values remain in each sealed run and model-free tables. They do not replace Phase7 equal-speaker scientific estimands.

## 20. Stability

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

```json
{
  "clean_guardrails_by_seed": {
    "6107": {
      "FAKE_miss_delta": 0.03571428571428571,
      "REAL_FA_delta": -0.03571428571428571,
      "passes": false
    },
    "6207": {
      "FAKE_miss_delta": 0.026785714285714288,
      "REAL_FA_delta": -0.008928571428571428,
      "passes": false
    },
    "6307": {
      "FAKE_miss_delta": 0.026785714285714288,
      "REAL_FA_delta": -0.01785714285714286,
      "passes": false
    }
  },
  "first": "H3",
  "group_support": "LIMITED_CLUSTER_PRECISION",
  "mean": -0.003968253968253978,
  "p_value": null,
  "paired_seed_effects": {
    "6107": -0.09523809523809525,
    "6207": 0.06249999999999999,
    "6307": 0.020833333333333322
  },
  "primary": true,
  "second": "H2",
  "seed_bootstrap_ci95": [
    -0.09523809523809525,
    0.06249999999999999
  ],
  "seed_count": 3,
  "seed_std": 0.08174145656791475,
  "speaker_bootstrap_ci95": [
    -0.01884920634920636,
    0.01190476190476189
  ],
  "speaker_count": 7,
  "uncertainty": "DESCRIPTIVE_ONLY; speaker interval conditional on trained seeds; seed interval conditional on same7speakers; only3seed blocks and7clusters"
}
```

Seed std is computed across independently trained blocks, not bootstrap replicates. Speaker bootstrap resamples7known speaker effects while retaining all paired conditions and trained seeds. Seed bootstrap separately resamples3paired blocks on the same speakers. Both intervals are descriptive with LIMITED_CLUSTER_PRECISION; no strong population significance or automatic hierarchical-bootstrap guarantee. Secondary comparisons are exploratory, no selective p-values.

New seed initialization/resume checks:

```json
{
  "head": {
    "status": "PASS",
    "checks": {
      "final_weights": true,
      "optimizer": true,
      "sample_order_views_losses_lr": true,
      "step": true,
      "counter": true,
      "history": true,
      "export_scores": true
    },
    "numerical_tolerance": 0,
    "scope": "CPU TF2.15 same environment; fresh processes",
    "final_weights_hash": "5b72bf0e6ff9251662ebe0dce9944004267dd3bb7fb3c8693c97bb76d08760bb",
    "optimizer_hash": "a95699f9c07e107d731f9b09934637854839e197a8000ef57a48191f2a8d7827",
    "steps": 4
  },
  "partial": {
    "status": "PASS",
    "checks": {
      "final_weights": true,
      "optimizer": true,
      "sample_order_views_losses_lr": true,
      "step": true,
      "counter": true,
      "history": true,
      "export_scores": true
    },
    "numerical_tolerance": 0,
    "scope": "CPU TF2.15 same environment; exact optimizer boundaries; fresh processes; warmup/partial transition included",
    "final_weights_hash": "294eb9a9177a8a3c9637dc6c2167b8c69bc43d3dc1ed4540f197eae0c5f6a520",
    "optimizer_hash": "2057f8b1483bdb01ca1c073a599c6ca7d906af0c04d38e8b46ae4ca7725b3954",
    "steps": 4
  }
}
```

Tiny independent ERM seed scores do not establish a stable method effect. Interrupted counterpart shares its seed and is not independent.

## 21. Representation Training

Actual warmup/partial gradients, unchanged BN and exported model checks:

```json
{
  "status": "PASS",
  "representation": {
    "head": {
      "warmup_backbone_unchanged": true,
      "BN_unchanged": true,
      "BN_layers": 34,
      "changed_backbone_variables": [],
      "full_export_vs_logged_score_residual": 8.940696716308594e-08,
      "eager_compiled_residual": 8.940696716308594e-08,
      "export_sha256": "03e73253ff7d1565a95b7dfc188828304c8b2401ffe335575e5938ea809b5964"
    },
    "partial": {
      "warmup_backbone_unchanged": true,
      "BN_unchanged": true,
      "BN_layers": 34,
      "changed_backbone_variables": [
        "expanded_conv_10/squeeze_excite/Conv/kernel:0",
        "expanded_conv_10/squeeze_excite/Conv/bias:0",
        "expanded_conv_10/squeeze_excite/Conv_1/kernel:0",
        "expanded_conv_10/squeeze_excite/Conv_1/bias:0",
        "expanded_conv_10/project/kernel:0",
        "Conv_1/kernel:0"
      ],
      "full_export_vs_logged_score_residual": 5.960464477539063e-08,
      "eager_compiled_residual": 5.960464477539063e-08,
      "export_sha256": "afc4db5d96a67a82fcf4a19d491608b312a72838e83bb642d2c58fdb54a8a899"
    }
  },
  "seed_initialization_hashes": {
    "6107": "7ebff9028f0fef3803e54d48fe06c11c08d9b2d17b8b03f17d016f7f4e3b877f",
    "6207": "4945de38ce0f4293a59dc7353c4b3943ea0ddd7651f6785667540c63bb2cc958",
    "6307": "41cb934c008191d828e6a18169ab91b3ef299ffd3635170bf031349863eae831"
  }
}
```

## 22. Full Training Infrastructure

# Full training readiness — evidence generated

FULL_TRAINING_CODE_READY: **YES** (CPU, tested TF2.15 environment).

NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: **NO**.

Gate1 IMPLEMENTATION_READY: YES. Gate2 FULL_TRAINING_CODE_READY: YES. Gate3 SCIENTIFIC_EXPERIMENT_READY for a novel C2: NO. No full-data run or GPU training is implied.

```json
{
  "regression": true,
  "exact_resume": true,
  "representation": true,
  "historical_integrity": true,
  "registered_configs": true,
  "resource_preflight": true,
  "infeasible_guarded_early_stop": true,
  "matched_baseline_initialization": true,
  "validation_optimization_parity": true
}
```

Actual regression:

| run | failures | errors | skipped | successful |
| --- | --- | --- | --- | --- |
| 149 | 0 | 0 | 0 | True |
| 40 | 0 | 0 | 0 | True |

CPU fresh-process resume compares final tensor bytes, Adam bytes, source/view order, loss/LR journals, counters, history and selected-model scores. Both head and warmup→partial schedules require exact hashes with tolerance0. Checkpoints occur only at optimizer boundaries; no claim of restoring half an accumulated microbatch. The full CNN export was checked against recorded scores within1e-6, warmup backbone/BN values were unchanged and partial kernels changed. Three seeds have distinct initialization hashes. A guarded tiny run actually early-stopped with no feasible export.

Frozen BN stays unchanged. Warmup masks backbone gradients while pre-registering tail slots; this is explicitly different from toggling trainable flags each epoch or resetting Adam at the transition. Both stages use one common scalar LR; no discriminative-LR implementation is claimed. Head-only can use detached embeddings; partial always runs the differentiable CNN on P0 tensors. Finite banks are not online augmentation.

Resource preflight verifies data byte hashes/known partition separation, complete lifecycle update capacity and conservative retained-checkpoint storage against15GiB reserve. Runtime guards check disk, peak process RAM, elapsed invocation time, finite gradients and predictions. Model selection and fixed threshold are explicit. Pipeline supports medium and full-foundation data without constructing an unbudgeted Mel cache. GPU source portability is prepared but its runtime, mixed precision, long-run convergence and arbitrary-shift risk remain unverified.

Verified historical files: 6243; accepted DOCX remains SHA91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073. 98 configuration identities are registered, not trained models. Actual run entries include first-version pilots, optimization repeats and interrupted counterparts; they must not inflate independent-seed counts. Missing metrics are null/absent, never zero.

Detailed machine-readable evidence: outputs/jaes_v2/phase8/audit/readiness_evidence.json. Standard baseline launch commands are in FULL_TRAINING_LAUNCH_PACKAGE; scientific blockers remain in NOVELTY_DECISION and METHOD_COMPARISON_PLAN.

## 23. GPU Readiness

# GPU portability guide

Actual host inventory: NVIDIA GeForce RTX3050 Laptop,4096MiB,driver595.71; an observed free reading was3203MiB. These are `nvidia-smi` inventory readings, not TensorFlow allocation measurements. Current `.venv` runs TensorFlow2.15 on native Windows CPU. **GPU_RUNTIME_NOT_VERIFIED**. No GPU forward/backward, GPU checkpoint or GPU peak memory is claimed. Prior WSL enumeration returned access denied; WSL configuration is unknown.

The [official TensorFlow install guide](https://www.tensorflow.org/install/pip) says native Windows GPU support ends at2.10; newer versions require a compatible Linux/WSL2 route. Do not upgrade/downgrade the production `.venv` to try to force CUDA. Use a separate Linux Python3.11 environment and a pinned TF2.15-compatible dependency set if pursuing portability. The [official tested-build table](https://www.tensorflow.org/install/source#gpu) specifies TF2.15 with Python3.9–3.11, CUDA12.2/cuDNN8.9. Validate the chosen package stack rather than installing an arbitrary latest CUDA toolkit.

The [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) requires a compatible Windows driver and WSL2; do not install a Linux display driver inside WSL. System installation, reboot and large environment downloads have not been performed. C free space is limited, so measure environment/container size and storage location first. No WSL/CUDA installation is implied by this handoff.

Keep data on D initially, mounted at `/mnt/d/audio-deepfake-mobilenet-lstm`; do not duplicate3.18GB extracted audio plus7.65GB archive without a new disk budget. NTFS/mount I/O performance must be benchmarked. Phase8 maps sealed absolute Windows source paths to this same repository root on Linux, preserving manifest bytes; Phase7 code is unchanged. The Linux runtime still needs local ImageNet weights with the same SHA, ffmpeg/libopus and pinned preprocessing dependencies. Any differing transform bytes require a new data contract and cannot be silently called exact reproduction.

For a GPU experiment, create a new registered config identity with `runtime="gpu"` using `scripts.jaes_v2.phase8_register`; never edit a hashed CPU config in place. First use a tiny partial config. `initialize('gpu')` requires TF to see a GPU and enables memory growth; it does not force CPU. After a tiny run, inspect device placement, actual backward execution, model reload and `GPU_memory` in resource measurements. Only after all pass may that runtime be marked verified. Device visibility alone remains insufficient.

Use float32 initially, microbatch1 and accumulation8/tiny or16/research. Mixed precision is not supported by this release's schema and must not be enabled by changing a global policy without versioning, loss scaling, finite-gradient and score-parity tests. Resume contracts include TF/Python/runtime and code/data hashes. CPU→GPU or different-version continuation is **not exact resume**; use a newly specified transfer experiment, not `resume` on an incompatible run. Same-environment GPU determinism also requires its own interruption comparison; CPU bitwise parity is not a GPU guarantee.

All heavy jobs share the existing OS lock. Do not run a second training process to monitor the first; `monitor` reads atomic pointers without waiting on that lock. Source/config portability is prepared; actual GPU runtime readiness remains NO.

## 24. Resources

# Full-training resource plan

Observed host:12logicalCPUs, RTX3050Laptop4GiB/driver595.71, native Windows TF2.15CPU. Current D free at evidence generation=25162997760bytes; C=3371966464bytes. The CLI redirects process temp and plotting cache to D, preserves15GiB reserve and uses the existing single-heavy-job OS lock. No dependency/environment installation or historical cleanup occurred.

## Measured tiny invocations

| experiment_id | invocation_seconds | updates_seconds | validation_seconds | peak_MiB | checkpoint_bytes |
| --- | --- | --- | --- | --- | --- |
| p8_tiny_dro_head_s6107_v2 | 17.430957799999305 | 1.7973471999939648 | 2.5677473999967333 | 594.1328125 | 49531554 |
| p8_tiny_dro_head_s6107_v3 | 16.435690300000715 | 1.7677022999996552 | 1.922864500000287 | 594.9453125 | 49531554 |
| p8_tiny_erm_head_s6107_resume_v2 | 17.642306599998847 | 1.135863999996218 | 2.24513239999942 | 593.77734375 | 49525461 |
| p8_tiny_erm_head_s6107_resume_v3 | 16.947843900001317 | 1.0145000999982585 | 2.0570679000011296 | 596.5703125 | 49525461 |
| p8_tiny_erm_head_s6107_v1 | 23.066664599999058 | 6.379174699999567 | 2.302599899998313 | 671.62109375 | 49538621 |
| p8_tiny_erm_head_s6107_v2 | 16.92577920000258 | 1.4337328000001435 | 2.0069834000023548 | 591.5234375 | 49525461 |
| p8_tiny_erm_head_s6107_v3 | 16.48634889999812 | 1.3439178000007814 | 1.7161124999984168 | 594.984375 | 49525461 |
| p8_tiny_erm_head_s6207_v2 | 18.01922530000229 | 1.4623159000002488 | 2.5548464999992575 | 592.51171875 | 49525481 |
| p8_tiny_erm_head_s6207_v3 | 17.50609700000132 | 1.3573063999974693 | 2.3128409999953874 | 593.30859375 | 49525481 |
| p8_tiny_erm_head_s6307_v2 | 17.845343399996636 | 1.2716087999979209 | 2.9973333999987517 | 592.09375 | 49525585 |
| p8_tiny_erm_head_s6307_v3 | 18.84009789999982 | 1.686547100005555 | 2.9735313999954087 | 592.359375 | 49525585 |
| p8_tiny_erm_partial_s6107_resume_v2 | 129.93436350000047 | 6.786689700002171 | 113.72618629999852 | 593.23046875 | 62844760 |
| p8_tiny_erm_partial_s6107_resume_v3 | 73.83351319999929 | 6.037354800002504 | 57.79424740000104 | 593.48046875 | 62844760 |
| p8_tiny_erm_partial_s6107_v2 | 134.23009629999797 | 8.500418499999796 | 115.08173810000153 | 592.9296875 | 62844760 |
| p8_tiny_erm_partial_s6107_v3 | 76.1902973000033 | 9.538251900001342 | 56.27032290000352 | 598.1171875 | 62844760 |
| p8_tiny_weighted_head_s6107_v2 | 16.792202099997667 | 1.477954399997543 | 1.8238217999969493 | 593.53515625 | 49531525 |
| p8_tiny_weighted_head_s6107_v3 | 17.958572800002003 | 1.4712383999940357 | 1.6652536000037799 | 593.64453125 | 49531525 |

Resume rows measure the resumed invocation only, excluding earlier process time. Source-version repeats and counterpart pairs are not independent quality replications. OS lifetime peak includes allocator/activation effects but does not isolate activations. GPU peak VRAM/time is NOT_MEASURED; nvidia-smi free memory is inventory only. Step timing includes gradient/optimizer and input preparation; validation timing includes its transforms/model/metrics. Input decode, transforms, Mel, compiled forwards, metrics, checkpoint and startup components are separately retained where instrumented. Components overlap and must not be summed twice. Figure generation is in the separate model-free report process, never inside training timing.

Original extracted audio uses3,178,404,021bytes; user LA.zip7,646,913,333bytes is retained, not copied or re-extracted. The existing Phase7 cache contains6,416embeddings /88,694,784rawfloat bytes and cost1776.57s to prepare. Head times exclude that shared historical cost. New partial training holds at most8P0 tensors plus one source waveform bank in RAM; it does not create a multi-GB image cache. Each full P0 tensor is6×224×224×3float32 (~3.45MiB);8entries about27.6MiB before framework activations. Final model exports are approximately6MB each, while Adam checkpoints are larger; measured files, not model size, define actual storage.

Validation optimization comparisons:

```json
{
  "erm_head_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 16.92577920000258,
    "v3_seconds": 16.48634889999812,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  },
  "erm_partial_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 134.23009629999797,
    "v3_seconds": 76.1902973000033,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  },
  "weighted_head_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 16.792202099997667,
    "v3_seconds": 17.958572800002003,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  },
  "dro_head_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 17.430957799999305,
    "v3_seconds": 16.435690300000715,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  }
}
```

Source-first evaluation reuses one deterministic waveform bank instead of regenerating all codec chains for each requested condition. This is fixed-view caching within a traversal, not fresh online augmentation or a performance result for a new model. Prediction parity is checked by source/condition, independent of CSV row order.

## Warmed CPU benchmark and illustrative time projection

```json
{
  "scope": "RESOURCE_BENCHMARK_ONLY; no quality comparison",
  "runtime": "CPU",
  "microbatch": 1,
  "effective_batch": 8,
  "sources": [
    "ASVspoof2019_LA:LA_T_1021790",
    "ASVspoof2019_LA:LA_T_1064484",
    "ASVspoof2019_LA:LA_T_1076943",
    "ASVspoof2019_LA:LA_T_1087973",
    "ASVspoof2019_LA:LA_T_1137942",
    "ASVspoof2019_LA:LA_T_1152394",
    "ASVspoof2019_LA:LA_T_1161711",
    "ASVspoof2019_LA:LA_T_1199930"
  ],
  "preparation_seconds": 0.14199619999999413,
  "input_components": {
    "decode_seconds": 0.02979820000109612,
    "transform_seconds": 0.0004695000025094487,
    "mel_seconds": 0.11145949999627192
  },
  "warmup_update_seconds": 4.625684800001181,
  "steady_update_seconds": [
    0.9374950000019453,
    1.496697900001891,
    1.0480618000001414,
    0.7952495999998064,
    0.8152121000020998
  ],
  "steady_update_median": 0.9374950000019453,
  "forward_warmup_seconds": 2.145351200000732,
  "steady_forward_median": 0.09477520000291406,
  "checkpoint_save_seconds": 0.3031520000004093,
  "checkpoint_bytes": 10470234,
  "peak_process_mib": 556.83203125,
  "total_parameters": 1308401,
  "trainable_parameters": 646481,
  "BN_unchanged": true,
  "GPU_memory": null
}
```

```json
{
  "foundation_erm_partial_v1.json": {
    "status": "EXTRAPOLATED_NOT_FULL_RUN_MEASUREMENT",
    "training_update_seconds": 62487.79173012966,
    "uncached_train_preparation_seconds": 9460.14183449961,
    "validation_forward_seconds": 44483.40354776774,
    "development_transform_seconds": 25479.76933679566,
    "total_seconds": 141911.10644919268,
    "assumptions": "ERM partial only; microbatch1; measured8source steady update scaled to16; warmup approximated by partial backward cost; no thermal/IO/long-run guarantee"
  },
  "medium_erm_partial_v1.json": {
    "status": "EXTRAPOLATED_NOT_FULL_RUN_MEASUREMENT",
    "training_update_seconds": 1574.991600003268,
    "uncached_train_preparation_seconds": 238.55361599999014,
    "validation_forward_seconds": 689.9634560212144,
    "development_transform_seconds": 535.7113132572017,
    "total_seconds": 3039.2199852816743,
    "assumptions": "ERM partial only; microbatch1; measured8source steady update scaled to16; warmup approximated by partial backward cost; no thermal/IO/long-run guarantee"
  }
}
```

The benchmark performs6effective8-source partial-CNN updates; first tracing update is separated from5steady repetitions. Full forward timing similarly excludes its first trace. Time estimates apply only to the specified ERM21epoch schedule, separately scaling training, preparation, clean-epoch/final-view forwards and development transforms. Reused tiny inputs, cache behavior, thermal throttling and large-corpus I/O limit the extrapolation; these are not measured full epoch/run times. Channel/DRO/RawBoost require their own resource calibration and view accounting. Invocation guards remain authoritative.

## Preflight projections (not full-job measurements)

| config | train | development | updates | retained_bytes | status |
| --- | --- | --- | --- | --- | --- |
| 1777ebe9283657b532681bfd816b1f4a451bee6f3a4d7b212c868fd15fc0d5d0 | 25380 | 17503 | 33327 | 3187671040 | PASS |
| 330c06b03fd4f650249f2d3f5ba08c1a766ac61576fc2d64838fa1253643ff01 | 640 | 368 | 840 | 1551892480 | PASS |

Retained-size projection assumes24MiB per boundary snapshot plus8MiB per possible epoch export and256MiB fixed overhead; checkpoint cadence500updates plus epoch boundaries. Each job's preflight is independent; sum projections for the intended seed/method matrix and recheck remaining disk before each launch. Do not multiply this single-run PASS into a guarantee that all families fit. No existing checkpoint is automatically deleted. Full-foundation21epochs have approximately33,327updates at effective batch16, before early stopping; medium21epochs840updates. Natural final partial batches use their actual denominator. Group-balanced/oversampling changes source exposure but not the unique recording count.

For a coarse CPU planning estimate, scale the measured tiny **update-path seconds / actual source exposures** by the planned source exposures, and separately add clean-epoch validation plus full final evaluation. This includes first-trace overhead and small-cohort cache effects and is not a reliable long-run ETA. Full-foundation final evaluation has17,503×7=122,521views, whereas a tiny final has14×7=98; epoch primary validation is an additional cost. Long-run epoch time, complete multi-seed wall time, GPU speedup and sustained memory remain NOT_MEASURED. Use the first medium run and journal-based monitor to refine estimates. Jobs may need repeated24h invocations with resume; final evaluation restarts if interrupted. Three-seed storage must be budgeted explicitly and jobs run sequentially.

Do not copy data into WSL or install CUDA/WSL in this phase. See GPU_TRAINING_GUIDE for the isolated portability path. Disk/RAM/NaN/power recovery and author commands are specified in FULL_TRAINING_LAUNCH_PACKAGE.

## 25. External Readiness

Reserved7,341 calibration records and official evaluation remain unused. External confirmation NOT_LOCKED. MLAAD-tiny is prior development; RADAR/ITW audio not acquired. Selection, hyperparameters, model/seed family, source/license/overlap and calibration/primary tests must lock before confirmation.

## 26. Limits

Tiny pilots, CPU-only runtime validation, finite augmentation, already-used development compositions, unknown original utterance/generator lineage, unmeasured long-run convergence and absent independent confirmation limit interpretation. Full long jobs and GPU tests are not executed. Early-stopping and guardrails are engineering selection policies, not deployment risk bounds.

## 27. Contribution Matrix

| Candidate | Status | Evidence |
|---|---|---|
| C1 | supporting empirical/engineering | historical controlled channels |
| C2 | novelty NOT_ESTABLISHED | standard baselines and rejected M3 specifications |
| C3 | DEFERRED | no cascade/calibration run |

## 28. Negative Results

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

The fixed-generator source-matched design lacked support; unknown lineage limits causal claims. Gain-invariant preprocessing can suppress useful cues or shift input distributions. Augmentation may improve misses while increasing clean false alarms. Small data, threshold sensitivity and frozen-backbone restrictions remain material. Teacher runtime/checkpoint-quality gates prevent KD. Phase6 initially ran cache and fine-tuning TensorFlow processes concurrently; cache reported an allocation failure and the fine-tuning process terminated without a completed result. Sequential execution and microbatching were introduced; native termination without a traceback is not falsely diagnosed as a measured gradient failure. Failed outputs are preserved, never reused as complete caches or published quality evidence.

| Run | Phase | Status | Output | Counts/steps retained |
|---|---|---|---|---|
| c2_p1_erm_pilot_v1 | 5 | INTERRUPTED | outputs/jaes_v2/c2/c2_p1_erm_pilot_v1 | UNKNOWN |
| partial_finetune_sanity_v1 | 6 | FAILED_PROCESS_TERMINATION | outputs/jaes_v2/phase6/partial_finetune_sanity_v1 | UNKNOWN |

Resource-stopped but sealed exploratory runs and superseded figure/report versions remain in their historical directories. Their presence does not add independent replication or favorable evidence.

Phase7 deliberately retains a failed proxy/novelty gate and any resource-preflight/run failures. Standard consistency can be unstable across training blocks or violate clean constraints; all effects are retained. RawBoost online quality is not substituted by finite-view scores. A short F0/F1 schedule cannot diagnose converged representation quality; a negative one-epoch result must not be generalized to all fine-tuning.

Engineering audit: Keras can reorder the full weights list when backbone trainability changes. The first representation resource preflight records an order-dependent concatenation SHA, which cannot be compared directly to the head-only hash. Before F0/F1 quality training, initialization was hashed before the transition and independently by sorted variable name/shape/value; the transition is required to preserve the canonical hash. The old preflight evidence remains unchanged. Scientific gradients, objective and resource measurements are unaffected.

# Phase8 novelty decision

NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: **NO**.

M1: standard weighted BCE, IMPLEMENTED; no novelty claim. M2: smoothed group-DRO control, IMPLEMENTED; no novelty claim. The source and actual smoke/pilot status are traced in PHASE8_EXPERIMENT_RESULTS.md.

M3a deterministic order-dependent supervision weights reduce to importance-weighted augmentation: PRIOR_ART_OVERLAP → REJECTED. M3b max over the two orders reduces exactly to worst-of-two loss: PRIOR_ART_OVERLAP → REJECTED. M3c acoustic-to-forensic signed commutator supervision lacks a justified target: PROPOSED only; fails the entry gate. No `NovelLoss`, M3 model, M3 training config or successful M3 ablation is created.

Full SONAR now verifies that label-conditioned representation alignment is already occupied prior art; this corrects the Phase7 ABSTRACT_ONLY evidence limitation without changing the historical Phase7 report. No inference of algebraic equivalence is made between SONAR and the unimplemented M3c.

C1 remains a supporting empirical/engineering contribution; C2 is a research question with stronger standard controls; C3 is deferred. There are not three established international novelties. Absence of a verified matching paper would not itself prove novelty. The targeted review is not exhaustive, and abstract-only additional work remains explicitly limited.

The separate FULL_TRAINING_CODE_READY decision is evidence-generated in PHASE8_FULL_TRAINING_READINESS.md. Code maturity and scientific novelty must never share a single green status.


First Phase8 preflight rejected absolute in-workspace audio paths; the source resolver was corrected before a training output existed. First successful head pilot exposed tf.function retracing. Executed source versions and pilots are retained, not overwritten. Codec preparation dominates the measured first partial pilot; subsequent resource changes are versioned and compared.

## 29. Evidence Traceability

Every completed run has config/data/code contracts, initialization hash, source/view update journals, Adam checkpoints and hashes, selected export, scores, classwise metrics, resources and a recursive artifact seal. Historical snapshot and accepted manuscript are verified separately. See outputs/jaes_v2/phase8/audit/readiness_evidence.json and the final integrity audit. A statistical result has to trace to a sealed run, not an illustrative method diagram.

## 30. Reproducibility

# Next commands

Use the full launch package for long training and recovery. Start with the medium ERM partial baseline, then matched channel partial, before committing to full-foundation or three-seed expansion. No novel M3 training is ready. All commands below exist in the tested parsers; use a fresh output name when repeating a report/test operation.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment --help
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment preflight --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json --output outputs/jaes_v2/phase8/preflight/author_readiness_review.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment monitor --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment resume --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment verify --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_report aggregate --output outputs/jaes_v2/phase8/reproduction_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_report master --statistics outputs/jaes_v2/phase8/statistics --output outputs/jaes_v2/phase8/master_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment validate --output outputs/jaes_v2/phase8/validation_review
```

`resume` is for an interrupted unsealed run; do not execute it on a sealed completed run. `train` rejects existing IDs. The stored final master is a fixed handoff edition, so adding future full runs requires a new edition/readiness evidence rather than rewriting historical metrics. Historical Phase7 reproduction remains available unchanged through its own scripts.


```json
{
  "jaes_v2": {
    "run": 149,
    "failures": 0,
    "errors": 0,
    "skipped": 0,
    "successful": true
  },
  "legacy_selected": {
    "run": 40,
    "failures": 0,
    "errors": 0,
    "skipped": 0,
    "successful": true
  }
}
```

Historical Phase7 master/inventory hashes are retained in the new inventory. The model-free builder produces byte-identical report/JSON from the same fixed evidence. The final source snapshot archives new implementation/config/test/document bytes.

## 31. Journal Readiness

Historical Phase7 edition, retained as Phase1–7 evidence (not new Phase8 measurements):

NOT_READY_FOR_JOURNAL_SUBMISSION. Target JAES covers audio research, including relevant forensic/ML work, but successful software tests and standard-baseline improvements alone do not certify a notable original contribution. Consult the current official AES instructions and disclose/cite the accepted conference work. The exact venue/DOI/publication/rights facts are not supplied; no arbitrary extension percentage or guaranteed acceptance is asserted. Current Clarivate/JCR listing is NOT_VERIFIED in this implementation phase. No submission was made.

The [official AES journal site](https://aes.org/publications/journal/) remains active and provides a submission link. The [official masthead](https://aes.org/publications/journal/journal-masthead/) identifies ISSN1549-4950 and2026publication information. This establishes the publisher's current publication/submission presence, not a current Clarivate indexing certificate.

The [author guidelines, Convention & Conference Papers section](https://aes.org/publications/journal-author-guidelines/) permit consideration of substantially revised conference work: extend the treatment and literature review, cite the event paper, and use a different title. Acceptance remains editorial; an accepted but unreleased manuscript does not resolve rights, disclosure or simultaneous-submission questions. The exact conference venue, publication/DOI status and rights terms are still required. The publisher does not specify a universal extension percentage or a minimum of three international contributions on this page.

Current SCIE/JCR status remains NOT_VERIFIED from Clarivate: the [official MJL lookup](https://mjl.clarivate.com/search-results?issn=1549-4950) was inaccessible through the research web tool. Third-party directories are not substituted for an authenticated current record, and no current JIF/quartile is certified. This limitation does not mean the journal is invalid or delisted.

LAVA is NOT_READY_FOR_SUBMISSION as a newly established methodological paper. Phase7 provides controlled development evidence and software traceability; it does not establish three distinct international novelties, independent external confirmation or calibrated operational risk. The conference manuscript and all conference models remain unchanged. No submission, author communication, rights transfer or publication action occurred.

Phase8 does not resolve conference venue, disclosure/rights or authenticated Clarivate indexing. A new trainer does not make the paper submission-ready.

## 32. GO / NO-GO

# Full training readiness — evidence generated

FULL_TRAINING_CODE_READY: **YES** (CPU, tested TF2.15 environment).

NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: **NO**.

Gate1 IMPLEMENTATION_READY: YES. Gate2 FULL_TRAINING_CODE_READY: YES. Gate3 SCIENTIFIC_EXPERIMENT_READY for a novel C2: NO. No full-data run or GPU training is implied.

```json
{
  "regression": true,
  "exact_resume": true,
  "representation": true,
  "historical_integrity": true,
  "registered_configs": true,
  "resource_preflight": true,
  "infeasible_guarded_early_stop": true,
  "matched_baseline_initialization": true,
  "validation_optimization_parity": true
}
```

Actual regression:

| run | failures | errors | skipped | successful |
| --- | --- | --- | --- | --- |
| 149 | 0 | 0 | 0 | True |
| 40 | 0 | 0 | 0 | True |

CPU fresh-process resume compares final tensor bytes, Adam bytes, source/view order, loss/LR journals, counters, history and selected-model scores. Both head and warmup→partial schedules require exact hashes with tolerance0. Checkpoints occur only at optimizer boundaries; no claim of restoring half an accumulated microbatch. The full CNN export was checked against recorded scores within1e-6, warmup backbone/BN values were unchanged and partial kernels changed. Three seeds have distinct initialization hashes. A guarded tiny run actually early-stopped with no feasible export.

Frozen BN stays unchanged. Warmup masks backbone gradients while pre-registering tail slots; this is explicitly different from toggling trainable flags each epoch or resetting Adam at the transition. Both stages use one common scalar LR; no discriminative-LR implementation is claimed. Head-only can use detached embeddings; partial always runs the differentiable CNN on P0 tensors. Finite banks are not online augmentation.

Resource preflight verifies data byte hashes/known partition separation, complete lifecycle update capacity and conservative retained-checkpoint storage against15GiB reserve. Runtime guards check disk, peak process RAM, elapsed invocation time, finite gradients and predictions. Model selection and fixed threshold are explicit. Pipeline supports medium and full-foundation data without constructing an unbudgeted Mel cache. GPU source portability is prepared but its runtime, mixed precision, long-run convergence and arbitrary-shift risk remain unverified.

Verified historical files: 6243; accepted DOCX remains SHA91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073. 98 configuration identities are registered, not trained models. Actual run entries include first-version pilots, optimization repeats and interrupted counterparts; they must not inflate independent-seed counts. Missing metrics are null/absent, never zero.

Detailed machine-readable evidence: outputs/jaes_v2/phase8/audit/readiness_evidence.json. Standard baseline launch commands are in FULL_TRAINING_LAUNCH_PACKAGE; scientific blockers remain in NOVELTY_DECISION and METHOD_COMPARISON_PLAN.


# Phase8 novelty decision

NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: **NO**.

M1: standard weighted BCE, IMPLEMENTED; no novelty claim. M2: smoothed group-DRO control, IMPLEMENTED; no novelty claim. The source and actual smoke/pilot status are traced in PHASE8_EXPERIMENT_RESULTS.md.

M3a deterministic order-dependent supervision weights reduce to importance-weighted augmentation: PRIOR_ART_OVERLAP → REJECTED. M3b max over the two orders reduces exactly to worst-of-two loss: PRIOR_ART_OVERLAP → REJECTED. M3c acoustic-to-forensic signed commutator supervision lacks a justified target: PROPOSED only; fails the entry gate. No `NovelLoss`, M3 model, M3 training config or successful M3 ablation is created.

Full SONAR now verifies that label-conditioned representation alignment is already occupied prior art; this corrects the Phase7 ABSTRACT_ONLY evidence limitation without changing the historical Phase7 report. No inference of algebraic equivalence is made between SONAR and the unimplemented M3c.

C1 remains a supporting empirical/engineering contribution; C2 is a research question with stronger standard controls; C3 is deferred. There are not three established international novelties. Absence of a verified matching paper would not itself prove novelty. The targeted review is not exhaustive, and abstract-only additional work remains explicitly limited.

The separate FULL_TRAINING_CODE_READY decision is evidence-generated in PHASE8_FULL_TRAINING_READINESS.md. Code maturity and scientific novelty must never share a single green status.

## 33. Exact Training Commands

# Full-training launch package

This package authorizes no automatic long job. The author explicitly runs the commands below after reading FULL_TRAINING_READINESS. Novel-method training is NO-GO; standard baseline research can proceed only when the measured code/resource gates say YES. All commands start in `D:/audio-deepfake-mobilenet-lstm` using the existing `.venv`. No production environment or conference checkpoint is replaced.

## Prerequisites and first order

Keep extracted ASV train/development audio, ImageNet weight cache and ffmpeg/libopus available. Verify the accepted manuscript, old evidence, source contracts and new tests; leave reserved calibration/evaluation closed. Keep at least15GiB D free after the projected outputs, and additional C space for the OS. TEMP/TMP and plotting cache are redirected to `.research_tmp` on D by the CLI. One heavy job at a time uses the inherited OS file mutex; the kernel releases it on process death.

Start with **medium A0 ERM partial**:640train,224A+144Bdevelopment,1warmup+20partial epochs, microbatch1/accum16, fresh seed6107, Adam.001 warmup/.0001 partial, frozen BN. Then run channel augmentation under the same staged schedule. These are development baselines, not confirmatory experiments. The first preflight commands create new JSON files and fail rather than overwrite an existing file. If rerunning preflight, choose a new output name.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment preflight --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json --output outputs/jaes_v2/phase8/preflight/author_medium_erm_6107.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment verify --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_medium_channel_partial_s6107_v1.json
```

After inspecting both classes and resources, run seeds6207/6307 in sequence using their existing configs. Weighted, equal-view, consistency, smoothed-DRO and finite-RawBoost controls have separate existing configs for each seed. Do not silently change LR, threshold, sampling or a config's hash. A new setting requires a new ID/protocol; `phase8_register` can copy an existing setting to a new CPU/GPU identity, not change scientific parameters invisibly. No B0/M3 family exists because the novelty entry gate failed.

## Foundation family, only after medium results and renewed storage gate

Foundation uses all25,380 training sources, natural prevalence (2,580REAL/22,800FAKE), and17,503 model-selection development sources. It still excludes7,341 reserved calibration and all official evaluation. Final evaluation measures every allowed development source under seven conditions. This is substantially more costly than the medium cohort and is NOT_EXECUTED in this phase.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment prepare-data --config configs/jaes_v2/phase8/p8_foundation_erm_partial_s6107_v1.json --output outputs/jaes_v2/phase8/preflight/author_foundation_erm_6107.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_foundation_erm_partial_s6107_v1.json
```

`prepare-data` verifies manifests, source byte hashes and known partition/speaker separation and computes a conservative storage projection. It does not create a giant feature/Mel cache, copy the archive, extract evaluation or certify unknown original lineage. Fine-tuning streams P0 tensors through the real CNN. Head-only foundation training is explicitly rejected because no corresponding frozen cache is prepared. Medium head-only uses the sealed Phase7 cache.

## Outputs and monitoring

Each ID creates `outputs/jaes_v2/phase8/training/<experiment_id>/` once: resolved config; source/code/environment contract; dataset audit; initialization/trainable manifest; per-update source/view/loss/LR journals; atomic optimizer checkpoints; latest pointer; history; selected `best_epoch_*.keras` and its SHA; selected-model scores; condition/attack classwise metrics; resources; terminal status; recursive seal. Checkpoint names include epoch/step/cursor/pending-validation state. Head/partial initialization is fresh; Phase7 training checkpoints are never used as a warm start.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment monitor --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
```

Monitor reads atomic state while the job owns the lock. Its rough ETA uses recent journal filesystem wall intervals; pauses/validation may distort it and final evaluation is excluded. Use the measured resource table for a planning range, not a promised completion time. Default limits are4,000MiB lifetime process peak,15GiB disk reserve and24h per invocation. A budget stop is a recoverable interruption, not a completed experiment. GPU memory/timing is unverified; no CUDA speedup is assumed.

## Resume and failure recovery

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment resume --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
```

Resume reconstructs the same architecture/trainable-variable set, verifies config/data/code/environment, validates checkpoint hashes and the committed update-journal hash chain, restores all Adam variables/iteration and explicit controller cursor/dropout counter, then recreates deterministic epoch/view sampling. Checkpoints occur at complete optimizer boundaries; accumulators are zero there. Interrupted uncommitted microbatches are replayed from that boundary. This is not weight-only restart. CPU bitwise verification is limited to the tested TF2.15/Python/runtime; switching GPU/version is not exact resume.

For power loss/Ctrl+C, preserve every file and resume the last published pointer. Unique `.pending.*` directories remain unpublished evidence and are never loaded as checkpoints or automatically deleted. A committed orphan may be reused only when its sealed controller matches the replayed state. If failure occurs before the first committed checkpoint, exact continuation is unavailable: preserve the failed run and register a fresh ID.

For OOM, close other heavy jobs and restart from a committed boundary only if unchanged config/runtime can meet the budget. Reducing microbatch changes dropout grouping and the config, so it requires a new experiment identity; no false exact-resume claim. For disk stops, make user-approved space available outside protected evidence, or plan a new experiment; do not delete Phase1–7 or conference artifacts. For NaN/Inf, inspect the preserved failure and journal; fail-closed gradients and metrics prevent publication as success. Changing LR/data requires a new protocol. For corrupt checkpoint/hash failure, preserve the corrupt files. An earlier sealed boundary can be selected only through an explicitly audited pointer-recovery procedure; no silent fallback or hash rewrite is provided in this release.

Early stopping stops when the declared criterion fails to improve for5epochs. If guarded selection has no feasible epoch, status is NO_FEASIBLE_CHECKPOINT; there is no fallback winner. Final selected-model development evaluation follows completion or early stop. If interrupted during final evaluation, resume redoes evaluation from the committed finished training state. There is no separate advertised evaluate-development CLI operation.

## Reload verification, aggregation and handback

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment verify --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment aggregate --output outputs/jaes_v2/phase8/author_statistics_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment validate --output outputs/jaes_v2/phase8/author_validation_review
```

Verify checks sealed bytes; selected models have actual reload evaluation during training and separate tiny full-model/export-parity checks. The standalone verify operation does not claim an additional neural forward. Aggregation is model-free and retains incomplete/negative runs separately. Keep all configs, logs, failures, contracts, checkpoint/export hashes, history, scores, classwise/per-attack metrics, resources and seals for research handback. Report independent seed identities and both error classes, not just a best score. A new final journal table must use these real outputs; no expected score is supplied here.

## 34. Phase9 Roadmap

First run matched medium standard representation baselines under the handed-off commands. Inspect both class errors and instability; use results to find a residual problem that survives ordinary controls. Pursue one of the two redirection proposals only after a justified evidence target/technical distinction is established. Then lock a new scientific protocol; independent confirmation and C3 follow later.

