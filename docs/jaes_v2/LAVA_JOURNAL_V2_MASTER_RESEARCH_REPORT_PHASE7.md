# LAVA JOURNAL V2 — MASTER RESEARCH REPORT — PHASE7 EDITION

Evidence-generated on31October2026 Asia/Novosibirsk. Separate Phase7 edition; historical Phase6 files preserved.

## 01. Executive Research Summary

Standard baselines now have all-attack, three-seed development evidence. No novel C2 algorithm or independent confirmation is established.

| method | seeds | clean_REAL_FA_mean | clean_FAKE_miss_mean | stress_REAL_FA_mean | stress_FAKE_miss_mean | stress_FAKE_miss_std |
| --- | --- | --- | --- | --- | --- | --- |
| H0 | 3 | 0.038690 | 0.083333 | 0.602183 | 0.400794 | 0.185799 |
| H1 | 3 | 0.017857 | 0.107143 | 0.314484 | 0.346230 | 0.244887 |
| H2 | 3 | 0.029762 | 0.071429 | 0.183532 | 0.387897 | 0.029811 |
| H3 | 3 | 0.008929 | 0.101190 | 0.215278 | 0.383929 | 0.073326 |
| H4 | 3 | 0.026786 | 0.095238 | 0.879960 | 0.185516 | 0.147984 |

The same seven primary development speakers recur across all seeds and transforms. A01–A06 are training-seen conditions, not unseen generators. The one-epoch F0/F1 pilot is a separate learning-duration stratum. Three international contributions are NOT_ESTABLISHED.

## 02. Accepted Conference Baseline

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

## 03. Original and Journal Research Questions

### Accepted conference questions

- RQ1: How do deployed artifacts compare on clean AUC and EER, and how should threshold-dependent metrics be interpreted under heterogeneous training and calibration?

- RQ2: How does absolute performance change under the executed noise, codec, and simulated-replay conditions on the fixed diagnostic subset?

- RQ3: Which artifacts are non-dominated across diagnostic-subset EER, mean stressed F1, and end-to-end RTF?

These are artifact comparison, diagnostic stress and scenario-specific efficiency questions; training and pretraining are heterogeneous, so the answers concern artifacts rather than intrinsic architecture superiority.

### Journal questions

J-RQ1: Does channel order create practically important classwise risk changes after insertion-power, source, preprocessing and operating-point controls? Exploratory evidence exists; the fixed-generator source-matched confirmation lacks adequate support. J-RQ2: Can ordinary amplitude/channel/paired training or a genuinely distinct evidence-preserving mechanism improve a lightweight detector subject to bounded clean REAL false alarms? Standard controls are being tested; a distinct mechanism is not established. J-RQ3: Can selective inference improve matched-cost risk/coverage with independent calibration and a validated second stage? Unanswered; C3 is deferred.

Phase7: RQ7.1 asks whether stronger all-attack coverage changes the pilot findings; RQ7.2 asks whether H3–H2 survives independent training seeds; RQ7.3 asks whether selected CNN adaptation improves classwise detection under a matched schedule; RQ7.4 asks whether a falsifiable technical distinction survives current prior art.

## 04. Complete Phase1–7 Timeline

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

## 05. Dataset Inventory and Provenance

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

## 06. Dataset Split and Integrity Protocol

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

## 07. Model and Checkpoint Inventory

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

## 08. Full Experiment Inventory

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

## 09. Phase1–4 Findings

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

## 10. Phase5 Findings

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

## 11. Phase6 Findings

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

## 12. Phase7 Findings

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

## 13. All-Attack Experimental Design

Selection and protocol are locked independently of scores.

```json
{
  "calibration_accessed": false,
  "counts": {
    "development_A": {
      "attacks": {
        "A01": 19,
        "A02": 19,
        "A03": 19,
        "A04": 19,
        "A05": 18,
        "A06": 18
      },
      "fake": 112,
      "n": 224,
      "real": 112,
      "speakers": 7
    },
    "development_B": {
      "attacks": {},
      "fake": 0,
      "n": 144,
      "real": 144,
      "speakers": 9
    },
    "train": {
      "attacks": {
        "A01": 53,
        "A02": 54,
        "A03": 54,
        "A04": 54,
        "A05": 53,
        "A06": 52
      },
      "fake": 320,
      "n": 640,
      "real": 320,
      "speakers": 20
    }
  },
  "decoded_duplicates": 0,
  "evaluation_accessed": false,
  "foundation_hashes": {
    "development": "dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a",
    "train": "7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72"
  },
  "known_partition_audit": "PASS",
  "manifest_sha256": "029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319",
  "original_source_lineage": "ORIGINAL_SOURCE_LINEAGE_UNKNOWN",
  "per_cell": 16,
  "scope": "EXPLORATORY_DEVELOPMENT; A01-A06 SEEN IN TRAIN; B speakers may have prior Phase5 exposure",
  "seed": 7107,
  "selection_id": "medium_all_attack_v1",
  "selection_rows_sha256": "52b65bc3cac4cd5b415a353dd400969058cb2774223c80c0f26476d05794d9bb",
  "status": "LOCKED_ALL_ATTACK_SELECTION"
}
```

| split | speaker_id | REAL | FAKE | A01 | A02 | A03 | A04 | A05 | A06 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | LA_0069 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| development | LA_0070 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| development | LA_0071 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| development | LA_0072 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| development | LA_0075 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| development | LA_0076 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| development | LA_0077 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| development | LA_0099 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0100 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0102 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0103 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0104 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0105 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0106 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0107 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0108 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | LA_0079 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0080 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0081 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0082 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0083 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0084 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0085 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0086 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0087 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0088 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0089 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0090 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0091 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0092 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0093 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0094 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0095 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0096 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0097 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0098 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |

StratumA defines a matched-support controlled estimand. StratumB tests false-alarm sensitivity on9REAL-only speakers. No fake records are imputed. Per-attack FAKE-only ROC-AUC/EER/PR-AUC are undefined and stored null; the full112REAL reference is not silently replicated per attack.

## 14. Training and Fine-Tuning Architecture

Source bytes → deterministic official-speaker/class selection → shared seeded waveform views → P0/P1 preprocessing → MobileNetV3Small → LSTM128 → Dense64/dropout → sigmoid FAKE score. Source labels are REAL0/FAKE1. P0 calls production preprocessing; P1 uses per-segment relative power flooring with explicit silent-segment behavior. Both retain the temporal image dimensions. Head-only training materializes detached frozen-backbone embeddings once in RAM, reusing the same source/condition cells across objectives. BCE supervises each sampled view; equal-two-view control averages two BCE terms; B4 adds0.1 mean squared probability discrepancy. ImageNet weight hash, seed, LR decay, Adam updates, view logs, clean-development-BCE checkpoint selection and reload evaluation are recorded. Partial fine-tuning uses raw/preprocessed train tensors, calls the backbone inside the tape, reuses the existing tail-unfreeze policy, freezes BN statistics, creates a fresh optimizer after trainability changes, and audits actual variable gradients/updates and checkpoint parity. Frozen embeddings cannot support CNN updates. Microbatching bounds memory; its tiny sanity loss is not development performance.

Phase7 head-only uses pinned ImageNet embeddings and a freshly seeded shared architecture. F0/F1 instead share1epoch,40updates,Adam0.0001,microbatch1/accum16, identical training source order and initialization. F1 reuses the existing tail-unfreeze policy, keeps all BN frozen, creates the optimizer after trainability changes, backpropagates through actual CNN operations and verifies frozen bytes and changed kernels. Versioned run configs reject unknown keys/unsafe paths and check their hashes against the locked protocol. Checkpoints publish atomically and are reloaded for all evaluation. Last-update optimizer arrays are retained as evidence; exact continuation is RESUME_NOT_SUPPORTED. Train/view/development and resource budgets are enforced without reusing protected conference weights.

## 15. Standard Baseline Comparisons

| method | seeds | clean_REAL_FA_mean | clean_FAKE_miss_mean | clean_AUC_mean | stress_REAL_FA_mean | stress_FAKE_miss_mean | B_clean_REAL_FA_mean | B_stress_REAL_FA_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | 3 | 0.038690 | 0.083333 | 0.976004 | 0.602183 | 0.400794 | 0.157407 | 0.520062 |
| H1 | 3 | 0.017857 | 0.107143 | 0.976190 | 0.314484 | 0.346230 | 0.148148 | 0.370370 |
| H2 | 3 | 0.029762 | 0.071429 | 0.977280 | 0.183532 | 0.387897 | 0.166667 | 0.203704 |
| H3 | 3 | 0.008929 | 0.101190 | 0.972630 | 0.215278 | 0.383929 | 0.120370 | 0.253086 |
| H4 | 3 | 0.026786 | 0.095238 | 0.971354 | 0.879960 | 0.185516 | 0.115741 | 0.872685 |

H0ERM; H1channel; H2two-view BCE; H3same pairs plus0.1probabilityMSE; H4finite RawBoost. Each source has one sampled first view per epoch. H2/H3 also supervise clean and therefore have7680supervised views versus3840for one-view baselines. They are equal-update comparisons, not equal-compute comparisons.

Representation pilot:

| experiment_id | epochs | training_policy | clean_REAL_FA | clean_FAKE_miss | stress_REAL_FA | stress_FAKE_miss | clean_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 1 | HEAD_ONLY | 0.232143 | 0.446429 | 0.258929 | 0.675595 | 0.750797 |
| p7_f1_s6107_v1 | 1 | PARTIAL_BACKBONE_FINETUNE | 0.258929 | 0.267857 | 0.235119 | 0.717262 | 0.810427 |

F1 minus matchedF0: {"clean_AUC": 0.05963010204081631, "clean_FAKE_miss": -0.1785714285714286, "clean_REAL_FA": 0.0267857142857143, "stress_FAKE_miss": 0.04166666666666674, "stress_REAL_FA": -0.023809523809523836}. Clean ranking improves in this short pilot while stressed FAKE miss worsens; there is no demonstrated robustness improvement or convergence claim. F2channel/F3equal-view/F4consistency representation ablations and three-seed fine-tuning are NOT_EXECUTED under the boundedF0/F1priority.

## 16. Multi-Seed Statistical Evidence

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

## 17. Classwise Robustness Analysis

| method | attack | unique_fake_per_seed_condition | speakers | clean_miss_mean | stress_miss_mean |
| --- | --- | --- | --- | --- | --- |
| H0 | A01 | 19 | 7 | 0.000000 | 0.286550 |
| H0 | A02 | 19 | 7 | 0.000000 | 0.374269 |
| H0 | A03 | 19 | 7 | 0.035088 | 0.614035 |
| H0 | A04 | 19 | 7 | 0.070175 | 0.333333 |
| H0 | A05 | 18 | 7 | 0.000000 | 0.561728 |
| H0 | A06 | 18 | 7 | 0.407407 | 0.234568 |
| H1 | A01 | 19 | 7 | 0.000000 | 0.070175 |
| H1 | A02 | 19 | 7 | 0.000000 | 0.561404 |
| H1 | A03 | 19 | 7 | 0.000000 | 0.175439 |
| H1 | A04 | 19 | 7 | 0.070175 | 0.070175 |
| H1 | A05 | 18 | 7 | 0.000000 | 0.728395 |
| H1 | A06 | 18 | 7 | 0.592593 | 0.500000 |
| H2 | A01 | 19 | 7 | 0.000000 | 0.011696 |
| H2 | A02 | 19 | 7 | 0.000000 | 0.842105 |
| H2 | A03 | 19 | 7 | 0.000000 | 0.087719 |
| H2 | A04 | 19 | 7 | 0.052632 | 0.005848 |
| H2 | A05 | 18 | 7 | 0.000000 | 0.839506 |
| H2 | A06 | 18 | 7 | 0.388889 | 0.574074 |
| H3 | A01 | 19 | 7 | 0.000000 | 0.017544 |
| H3 | A02 | 19 | 7 | 0.017544 | 0.766082 |
| H3 | A03 | 19 | 7 | 0.035088 | 0.140351 |
| H3 | A04 | 19 | 7 | 0.087719 | 0.017544 |
| H3 | A05 | 18 | 7 | 0.000000 | 0.833333 |
| H3 | A06 | 18 | 7 | 0.481481 | 0.561728 |
| H4 | A01 | 19 | 7 | 0.000000 | 0.210526 |
| H4 | A02 | 19 | 7 | 0.017544 | 0.105263 |
| H4 | A03 | 19 | 7 | 0.000000 | 0.345029 |
| H4 | A04 | 19 | 7 | 0.052632 | 0.228070 |
| H4 | A05 | 18 | 7 | 0.000000 | 0.160494 |
| H4 | A06 | 18 | 7 | 0.518519 | 0.055556 |

Each attack group has18–19FAKE recordings on7speakers, with only2–3per speaker. The table averages condition/seed miss rates within attack and is descriptive. Full `attack_metrics.csv` includes score quantiles and explicit null two-class metrics. `speaker_metrics.csv` retains every REAL-only speaker with missing FAKE denominator as null; small groups are not removed. Different cohort size, attack mixture, seed and update count prevent attributing cross-phase changes solely to the addition ofA05/A06. Within-Phase7 matched comparisons are the supported estimands.

## 18. Clean–Robustness Trade-Offs

| first | second | mean | seed_std | clean_guardrails_by_seed |
| --- | --- | --- | --- | --- |
| H3 | H2 | -0.003968 | 0.081741 | {"6107": {"FAKE_miss_delta": 0.03571428571428571, "REAL_FA_delta": -0.03571428571428571, "passes": false}, "6207": {"FAKE_miss_delta": 0.026785714285714288, "REAL_FA_delta": -0.008928571428571428, "passes": false}, "6307": {"FAKE_miss_delta": 0.026785714285714288, "REAL_FA_delta": -0.01785714285714286, "passes": false}} |
| H3 | H1 | 0.037698 | 0.184075 | {"6107": {"FAKE_miss_delta": 0.017857142857142863, "REAL_FA_delta": -0.008928571428571428, "passes": true}, "6207": {"FAKE_miss_delta": -0.03571428571428571, "REAL_FA_delta": 0.0, "passes": true}, "6307": {"FAKE_miss_delta": 0.0, "REAL_FA_delta": -0.01785714285714286, "passes": true}} |
| H3 | H4 | 0.198413 | 0.079266 | {"6107": {"FAKE_miss_delta": 0.008928571428571438, "REAL_FA_delta": -0.044642857142857144, "passes": true}, "6207": {"FAKE_miss_delta": 0.01785714285714285, "REAL_FA_delta": -0.017857142857142856, "passes": true}, "6307": {"FAKE_miss_delta": -0.008928571428571425, "REAL_FA_delta": 0.008928571428571428, "passes": true}} |
| H1 | H0 | -0.054563 | 0.076710 | {"6107": {"FAKE_miss_delta": -0.008928571428571425, "REAL_FA_delta": -0.008928571428571428, "passes": true}, "6207": {"FAKE_miss_delta": 0.0625, "REAL_FA_delta": -0.026785714285714284, "passes": false}, "6307": {"FAKE_miss_delta": 0.017857142857142863, "REAL_FA_delta": -0.02678571428571428, "passes": true}} |
| H4 | H0 | -0.215278 | 0.037920 | {"6107": {"FAKE_miss_delta": 0.0, "REAL_FA_delta": 0.026785714285714288, "passes": false}, "6207": {"FAKE_miss_delta": 0.008928571428571438, "REAL_FA_delta": -0.008928571428571428, "passes": true}, "6307": {"FAKE_miss_delta": 0.026785714285714288, "REAL_FA_delta": -0.05357142857142857, "passes": false}} |

The predeclared2pp clean delta limit applies within paired comparisons at fixed0.5. It is a sample-level screen, not a certified real-world bound. Lower FAKE miss accompanied by higher REAL false alarm is a trade-off, never universal superiority. Figures05/12 show both error classes. No new threshold was fitted after inspecting stress results.

Secondary descriptive Pareto screen relative to H0, explicitly posthoc and distinct from the registered H3/H2 test:

| method | clean_screen_vs_H0 | stress_REAL_FA | stress_FAKE_miss | lifecycle_seconds_mean | nondominated_stress_errors_within_screen |
| --- | --- | --- | --- | --- | --- |
| H0 | True | 0.602183 | 0.400794 | 19.561808 | False |
| H1 | False | 0.314484 | 0.346230 | 20.112182 | False |
| H2 | True | 0.183532 | 0.387897 | 21.862857 | True |
| H3 | False | 0.215278 | 0.383929 | 19.845208 | False |
| H4 | False | 0.879960 | 0.185516 | 20.573749 | False |

## 19. Preprocessing and Nuisance Findings

Phase4/5 show absolute Mel floor and silence/gain handling can change representation and class discrimination. Phase7 keeps P0 fixed rather than tuning preprocessing on new outcomes. The gain-half and gain-quarter evaluations remain in every run; those numerical results live in `condition_metrics.json`. The new proxy study adjusts post-hoc association for known speaker, attack, condition severity, logRMS and silence fraction. These adjustments cannot identify missing original-source lineage or causal forensic cue retention.

## 20. RawBoost and Consistency Controls

H2/H3 use the same initialization SHA and identical training-view log per seed, including independently stochastic dropout in paired forward calls. The sole objective difference is0.1MSE. Finite RawBoost algorithm5 supplies2fixed realizations at22050Hz/maxF8000, with original author functions/license unchanged. Finite realizations are shared across seeds; training seeds change head/order/view selection.

```json
{
  "ISD_parameters": {
    "P": 10,
    "g_sd": 2
  },
  "algorithm": 5,
  "amplitude": "Reference internal processing; reject peak>1.00001; no hidden post-channel normalization",
  "convolution_parameters": {
    "N_f": 5,
    "maxBW": 1000,
    "maxBiasLinNonLin": 20,
    "maxCoeff": 100,
    "maxF": 8000,
    "maxG": 0,
    "minBW": 100,
    "minBiasLinNonLin": 5,
    "minCoeff": 10,
    "minF": 20,
    "minG": 0,
    "nBands": 5
  },
  "decision": "Additional extraction alone exceeds locked head-only budget; do not substitute finite-view results",
  "head_run_budget_seconds": 240,
  "implemented": true,
  "max_filter_hz": 8000,
  "mean_seconds": 0.3652749857142875,
  "projected_six_epoch_extra_extraction_seconds": 935.103963428576,
  "quality_experiment": "NOT_EXECUTED",
  "resampling": "NONE; original16k setting adapted directly to22050Hz frequency-calibrated filters",
  "sampling_rate": 22050,
  "status": "AWAITING_COMPUTE",
  "tested": true
}
```

The stateless online adapter changes per-example realization across epochs, isolates global numpy state and preserves labels/length. Frequency parameters remainHz at22,050Hz without resampling. It is not exact16k original reproduction. Online quality is NOT_EXECUTED when extra extraction violates the locked budget; Figure09 is deliberately not generated.

## 21. Proposed C2 Method and Falsification

Candidate weighted consistency remains NOT_IMPLEMENTED. r=(1+cosine)/2 is a generic frozen ImageNet representation statistic, not an observable forensic label. Identity and lossless FLOAT controls preserve the waveform exactly. Half-gain, noise0,Opus8 and5msboundary-roll challenge nuisance behavior; none supplies ground truth that a precise spoof cue was destroyed.

```json
{
  "formula": "r=(1+cosine(flattened frozen ImageNet features))/2",
  "neural_inference": "Only seven-source control embeddings; association aggregation model-free",
  "new_method_gate": "FAIL",
  "positive_control": "Identity and FLOAT serialization preserve exact waveform and embeddings",
  "rejection": "No ground-truth forensic cue retention. ImageNet similarity and error association do not establish mechanism, including after nuisance adjustment. Challenging severity is not a cue-removal oracle.",
  "status": "NON_SPECIFIC",
  "weighted_loss_implemented": false
}
```

| label | view_rows | speakers | raw_spearman | adjusted_residual_spearman | adjustment | causal_interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 672 | 7 | -0.470960 | 0.158744 | speaker,attack,condition severity,logRMS,silence fraction | NONE |
| 1 | 672 | 7 | -0.240576 | -0.203047 | speaker,attack,condition severity,logRMS,silence fraction | NONE |

100within-speaker/attack/condition proxy permutations are retained in JSON. This is a negative control for association, not a trained shuffled-weight loss ablation. The latter is NOT_EXECUTED because the estimator failed the semantic gate. Reject weighting if ordinary equal-view consistency, severity alone or shuffled weights explains its claimed mechanism; never drop supervised hard views.
RMS/silence covariates describe observed transformed waveforms before fixed-length input padding. They do not remove every nuisance, duration or preprocessing effect. Residualized correlation on already-used development is not held-out predictive validation or causal cue identification. Small0.98gain and stronger0.5gain are both retained as controls; P0 exact gain invariance is not assumed.

## 22. International Novelty Assessment

| candidate | mechanism | nearest | experiment | independent_confirmation | status |
| --- | --- | --- | --- | --- | --- |
| C1 | No distinct new general mechanism | RADAR; established signal controls | Phases1–4controls | NO | ENGINEERING_CONTRIBUTION / supporting validation |
| C2 | Weighted retention objective unjustified | F-SAT;Mel-BAP;PCL-NET;DK-CAST;ForensicSimilarity | Phase7standard baselines and proxy controls | NO | EMPIRICAL_OBSERVATION; METHODOLOGICAL_CONTRIBUTION_NOT_ESTABLISHED |
| C3 | NOT_IMPLEMENTED | Selective prediction;AT-ADDrouting;ICLAD | NOT_EXECUTED | NO | DEFERRED |


Primary sources were inspected through publisher/proceedings and arXiv pages. Search included selective invariance, audio forensic similarity, corruption-aware learning, and classwise robustness. This is a targeted review, not exhaustive novelty clearance. None of the following published scores is a reproduced LAVA result.

| Work | Verification | Technical overlap / difference |
|---|---|---|
| [AT-ADD v2](https://arxiv.org/html/2608.23437v2), 15 October2026 | FULL_TEXT_VERIFIED; preprint challenge overview; SectionsVI–VIII | Studies unseen synthesizers, channels, replay and audio types; leading systems combine augmentation, multi-crop and routing. More channel grids or a cascade are insufficient novelty. This repository has not reproduced the challenge. |
| [Adaptive Compression and Targeted Perturbation](https://arxiv.org/html/2610.16049v1) | FULL_TEXT_VERIFIED Sections3.2–3.3; [official NeurIPS2026 listing](https://neurips.cc/Downloads/2026) confirms accepted title; poster page unavailable | Whisper features route through heterogeneous compression branches; band-masked log-Mel PGD trains robust boundaries. This differs from F-SAT waveform reconstruction and directly blocks calling Mel-selective perturbation a new LAVA mechanism. Code not verified. |
| [I Can Hear You / F-SAT](https://proceedings.iclr.cc/paper_files/paper/2025/file/1ab925f5f4ac0670256c3d335868c996-Paper-Conference.pdf) | PEER_REVIEWED ICLR2025; FULL_TEXT_VERIFIED; Phase6 audit retained | Band-constrained STFT-magnitude adversarial training reconstructs waveform with original phase. Established selective robust training; LAVA probability consistency has no adversarial inner loop. |
| [PCL-NET](https://arxiv.org/html/2610.07216v1) | PREPRINT; FULL_TEXT_VERIFIED architecture/loss; not locally reproduced | XLS-R pairs original and neural-codec representations using feature consistency and classification. Ordinary paired invariance already exists; conventional Opus is not neural-codec resynthesis. |
| [DK-CAST](https://link.springer.com/article/10.1007/s10791-025-09746-4) | Publisher primary method; Phase6 full-method audit retained; CODE_NOT_VERIFIED | Teacher/student phoneme/logit/feature supervision and codec-confidence weighting overlap generic reliability weighting. A smaller backbone is no technical distinction. |
| [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing) | CODE_VERIFIED pinned MIT source; [paper](https://arxiv.org/abs/2111.04433) | Nonlinear convolutive and additive recipes are reused. LAVA algorithm5 at22050Hz/maxF8000 and finite or stateless online exposure are adaptations, not exact16k original reproduction. |
| [Forensic Similarity for Speech Deepfakes](https://arxiv.org/html/2510.02864v2) | FULL_TEXT_VERIFIED Section3; [ACM2026 publication](https://doi.org/10.1145/3785353.3815078) | Source-tracing-trained embeddings feed a Siamese similarity model for shared generator traces. Generic ImageNet cosine cannot be relabeled forensic retention. Its same-generator objective differs from preservation of class evidence after corruption. |
| [SONAR](https://proceedings.mlr.press/v306/hidekel26a.html) | PEER_REVIEWED ICML2026 proceedings; ABSTRACT_ONLY; PDF access failed | Dual content/HF paths with class-dependent spectral consistency already occupy selective forensic representation learning. Full method not verified here; no equivalence or reproduction claim. |
| [ICLAD](https://aclanthology.org/2026.findings-acl.450/) | PEER_REVIEWED ACLFindings2026; ABSTRACT_ONLY | Detector-to-audio-language-model routing and comparative reasoning further limit generic cascade novelty claims. No local routing experiment. |

Candidate L=.5[BCEclean+BCEdistorted]+lambda*w(r)*MSE remains PRIOR_ART_OVERLAP. The candidate r=(1+cosine)/2 lacks forensic ground truth. Before any methodological claim it must beat severity-only, label-free shuffled and equal-view controls without discarding difficult supervision. A successful standard baseline or fine-tuning pilot is empirical evidence, not a distinct international algorithm.

[AES author guidelines](https://aes.org/publications/journal-author-guidelines/) require original research and supported experimental claims. Current Clarivate/JCR status, conference venue/rights and extension eligibility remain unverified in this phase. No three-novelty requirement was inferred from the publisher. No manuscript was submitted or rewritten.
An INDEPENDENTLY_CONFIRMED_FINDING requires an untouched and locked evaluation. None exists. No three international novelties or journal acceptance is claimed.

## 23. Negative Results and Rejected Hypotheses

The fixed-generator source-matched design lacked support; unknown lineage limits causal claims. Gain-invariant preprocessing can suppress useful cues or shift input distributions. Augmentation may improve misses while increasing clean false alarms. Small data, threshold sensitivity and frozen-backbone restrictions remain material. Teacher runtime/checkpoint-quality gates prevent KD. Phase6 initially ran cache and fine-tuning TensorFlow processes concurrently; cache reported an allocation failure and the fine-tuning process terminated without a completed result. Sequential execution and microbatching were introduced; native termination without a traceback is not falsely diagnosed as a measured gradient failure. Failed outputs are preserved, never reused as complete caches or published quality evidence.

| Run | Phase | Status | Output | Counts/steps retained |
|---|---|---|---|---|
| c2_p1_erm_pilot_v1 | 5 | INTERRUPTED | outputs/jaes_v2/c2/c2_p1_erm_pilot_v1 | UNKNOWN |
| partial_finetune_sanity_v1 | 6 | FAILED_PROCESS_TERMINATION | outputs/jaes_v2/phase6/partial_finetune_sanity_v1 | UNKNOWN |

Resource-stopped but sealed exploratory runs and superseded figure/report versions remain in their historical directories. Their presence does not add independent replication or favorable evidence.

Phase7 deliberately retains a failed proxy/novelty gate and any resource-preflight/run failures. Standard consistency can be unstable across training blocks or violate clean constraints; all effects are retained. RawBoost online quality is not substituted by finite-view scores. A short F0/F1 schedule cannot diagnose converged representation quality; a negative one-epoch result must not be generalized to all fine-tuning.

Engineering audit: Keras can reorder the full weights list when backbone trainability changes. The first representation resource preflight records an order-dependent concatenation SHA, which cannot be compared directly to the head-only hash. Before F0/F1 quality training, initialization was hashed before the transition and independently by sorted variable name/shape/value; the transition is required to preserve the canonical hash. The old preflight evidence remains unchanged. Scientific gradients, objective and resource measurements are unaffected.

## 24. Computational and Storage Resources

```json
{
  "CIM_probe": "Read denied by host; physical/commit memory measured with Windows GlobalMemoryStatusEx",
  "concurrency": "Existing OS file mutex; one heavyTensorFlow job; PID ownership audit in job_owners",
  "cpu_count": 12,
  "date": "2026-10-31 Asia/Novosibirsk",
  "disk": {
    "C:/": {
      "free_bytes": 1744568320,
      "total_bytes": 325344104448
    },
    "D:/": {
      "free_bytes": 26325270528,
      "total_bytes": 157286395904
    }
  },
  "gpu_inventory": {
    "returncode": 0,
    "stderr": "",
    "stdout": "NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 595.71"
  },
  "gpu_tensorflow": "No GPU visible in executed current TensorFlowCPU cache/training",
  "memory": {
    "available_commit": 4586565632,
    "available_physical": 6092017664,
    "available_virtual": 140731997741056,
    "extended_virtual": 0,
    "length": 64,
    "load_percent": 62,
    "total_commit": 27264434176,
    "total_physical": 16452980736,
    "total_virtual": 140737488224256
  },
  "new_dependencies_installed": false,
  "pip_cache_actual": {
    "returncode": 0,
    "stderr": "",
    "stdout": "c:\\users\\thanh dat\\appdata\\local\\packages\\sandbox.{3424b7d1-ae6a-402f-8ef2-d7d24d2d9471}\\ac\\pip\\cache"
  },
  "production_environments_modified": false,
  "python": "D:\\audio-deepfake-mobilenet-lstm\\.venv\\Scripts\\python.exe",
  "temp": {
    "TEMP": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp",
    "TMP": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp",
    "checkpoints": "D:\\audio-deepfake-mobilenet-lstm\\outputs\\jaes_v2\\phase7\\training",
    "matplotlib_cache": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp\\matplotlib",
    "pip_cache_default": "C:\\Users\\THANH DAT\\AppData\\Local\\pip\\Cache",
    "python_temp": "D:\\audio-deepfake-mobilenet-lstm\\.research_tmp",
    "tensorflow_temporary": "Process TEMP/TMP onD; no system environment change"
  },
  "tensorflow": "2.15.0",
  "wsl_inventory": {
    "returncode": 4294967295,
    "stderr": "",
    "stdout": "Access is denied. \n\nError code: Wsl/E_ACCESSDENIED"
  }
}
```

Cache actual:

```json
{
  "arrays": {
    "development_P0": [
      2576,
      6,
      576
    ],
    "train_P0": [
      3840,
      6,
      576
    ]
  },
  "benchmark_only": false,
  "contract_hash": "7411d09b8d9efedf5fd5b69f0129652335f1c5b034ca7a25a33bddaf83b28d12",
  "elapsed_seconds": 1776.5667477000002,
  "gpu_available": false,
  "peak_ram_mib": 959.32421875,
  "resumed_shards": 0,
  "sources": 1008,
  "status": "SEALED_PHASE7_CACHE",
  "views": 6416
}
```

Training resources:

| experiment_id | elapsed_seconds | update_seconds_total | peak_ram_mib | optimizer_updates | source_examples_seen | supervised_views_seen | model_training_forwards | backward_calls | checkpoint_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 12.506894 | 6.477249 | 626.136719 | 40 | 640 | 640 | 640 | 640 | 5916718 |
| p7_f1_s6107_v1 | 1472.984895 | 48.669298 | 883.273438 | 40 | 640 | 640 | 640 | 640 | 5916699 |
| p7_h0_s6107_v1 | 17.454715 | 7.225866 | 631.628906 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h0_s6207_v1 | 18.022321 | 7.495766 | 633.785156 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h0_s6307_v1 | 23.208389 | 10.856521 | 633.832031 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h1_s6107_v1 | 20.377338 | 7.607627 | 635.992188 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h1_s6207_v1 | 19.773671 | 7.199822 | 638.687500 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h1_s6307_v1 | 20.185537 | 9.552592 | 633.992188 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h2_s6107_v1 | 22.733548 | 10.222911 | 650.046875 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h2_s6207_v1 | 16.987935 | 8.666293 | 647.390625 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h2_s6307_v1 | 25.867089 | 13.359128 | 651.187500 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h3_s6107_v1 | 19.772342 | 9.057026 | 650.335938 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h3_s6207_v1 | 17.604633 | 9.544557 | 650.816406 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h3_s6307_v1 | 22.158650 | 10.731365 | 647.113281 | 240 | 3840 | 7680 | 480 | 240 | 5916718 |
| p7_h4_s6107_v1 | 17.654819 | 7.361591 | 632.394531 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h4_s6207_v1 | 25.623703 | 9.261843 | 629.824219 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |
| p7_h4_s6307_v1 | 18.442723 | 8.462098 | 635.988281 | 240 | 3840 | 3840 | 240 | 240 | 5916718 |

Full-student inference:

| experiment_id | parameters | checkpoint_bytes | median_seconds | offline_rtf | peak_process_lifetime_mib |
| --- | --- | --- | --- | --- | --- |
| p7_h0_s6107_v1 | 1308401 | 5916718 | {"decode": 0.002616699999634875, "end_to_end": 0.24510800000007293, "model": 0.23322110000026441, "preprocessing": 0.009560999999848718} | 0.106609 | 656.878906 |
| p7_h3_s6107_v1 | 1308401 | 5916718 | {"decode": 0.0023830999998608604, "end_to_end": 0.2480652000003829, "model": 0.23644430000058492, "preprocessing": 0.009863700000096287} | 0.107895 | 679.839844 |
| p7_h4_s6107_v1 | 1308401 | 5916718 | {"decode": 0.0024597999999969034, "end_to_end": 0.2576280000002953, "model": 0.24508510000032402, "preprocessing": 0.010029600000052596} | 0.112055 | 689.761719 |

Cache cost is shared preprocessing, not hidden inside fast head-only training. Timing excludes process startup/model construction and separates full-student inference from embedding-head updates. CPU runtime, physical GPU and actual GPU execution are distinct. Process lifetime peaks include allocator/activation effects but do not isolate activation memory. No system CUDA stack or production environment was changed.
Matched full-student F0/F1 timing:

| experiment_id | parameters | median_seconds | offline_rtf | process_lifetime_peak_mib |
| --- | --- | --- | --- | --- |
| p7_f0_s6107_v1 | 1308401 | {"decode": 0.002626599999530299, "end_to_end": 0.278780700000425, "model": 0.2663632999992842, "preprocessing": 0.010294499999872642} | 0.121255 | 467.316406 |
| p7_f1_s6107_v1 | 1308401 | {"decode": 0.0025995000005423208, "end_to_end": 0.2883762999999817, "model": 0.27558030000000144, "preprocessing": 0.010318200000256184} | 0.125429 | 487.542969 |

The F1 lifecycle includes complete raw-view CNN evaluation; its optimizer-update seconds are reported separately. Comparing that lifecycle directly with cached-head scoring does not measure training overhead. Future work should profile/compile full inference with parity checks before longer experiments; no unmeasured speedup is claimed.

## 25. Reproducibility and Regression Evidence

Phase6 historical master and JSON retain exact verified hashes; all protected historical entries are checked against `phase7_before.json`. The expanded suite runs actual tests rather than hard-coded counts.

```json
{
  "jaes_v2": {
    "errors": 0,
    "failures": 0,
    "run": 131,
    "skipped": 0,
    "successful": true
  },
  "legacy_selected": {
    "errors": 0,
    "failures": 0,
    "run": 40,
    "skipped": 0,
    "successful": true
  }
}
```

Statistics, figures, master and JSON regenerate from immutable structured artifacts without TensorFlow/Torch inference. Final byte comparison and file hashes appear in `outputs/jaes_v2/audit/phase7_final_integrity.json`; that final audit is downstream of this generator to avoid a circular report hash. Unchanged inputs must produce identical bytes.

## 26. Code Architecture and Module Map

| Directory / key modules | Responsibility |
|---|---|
| src/lava/jaes_v2/provenance.py, datasets.py, asv_archive.py | Source identity, UNKNOWN semantics, official protocols and selective extraction |
| transforms.py, controls.py, nuisance_controls.py, source_matching.py | Waveforms, fixed/relative noise, nuisance controls and source-match levels |
| preprocessing.py, c2_student.py, c2_training.py | Versioned policies, reused architecture and preserved Phase5 lifecycle |
| phase6_data.py, rawboost.py, vendor/rawboost/ | New speaker-aware selection, finite-view cache, pinned licensed reference adapter |
| phase6_training.py | New baseline objectives, reliability, bounded head training and real partial fine-tuning |
| scripts/jaes_v2/phase6_experiment.py, phase6_report.py | Executable gates, training commands and score-only statistics/figures |
| scripts/jaes_v2/build_master_research_report.py | Cross-phase inventory, metric reconciliation, report and evidence audit |
| configs/jaes_v2/, tests/jaes_v2/ | Versioned experiment contracts and regression/unit tests |
| outputs/jaes_v2/phase6/ | Selections, caches, failed/completed runs, primary protocol, reports and resources |
| docs/jaes_v2/, papers/jaes_v2/ | Research decisions, current master, claim restrictions and paper evidence mapping |

| module | responsibility |
| --- | --- |
| phase7_data.py | All-attack selector; explicit selection/cache identities; source compatibility; per-source resumable shards; atomic directory publication |
| phase7_training.py | Independent seed streams; fair objective; head/real-CNN training; checkpoint parity; optimizer evidence; failure/budget lifecycle |
| phase7_statistics.py | Full denominator/metadata checks; score-derived classwise/attack/speaker metrics; paired seed and cluster uncertainty |
| phase7_readiness.py | Fail-closed confirmation/calibration contracts; no reserved-label fitting |
| phase7_experiment.py / phase7_diagnostics.py | Sequential CLI; process-scoped Dtemp; measured resource gates; proxy controls and actual inference timing |
| phase7_report.py / build_master_research_report_phase7.py | Model-free vector figures, tables and comprehensive versioned handoff |

## 27. Claims-to-Evidence Traceability Matrix

| claim | evidence | cohort | model | verified | limitation |
| --- | --- | --- | --- | --- | --- |
| All-attack cohort | outputs/jaes_v2/phase7/selections/medium_all_attack_v1/selection.json | 640/224A/144B | N/A | YES | Original lineage unknown |
| Seed-level standard baselines | outputs/jaes_v2/phase7/statistics/per_seed.csv | 224A/144B | H0-H4 | YES | 3seeds;7primary speakers |
| Matched consistency effect | outputs/jaes_v2/phase7/statistics/paired_comparisons.json | 224A | H3/H2 | YES | Descriptive development |
| Representation pilot | outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/gradient_diagnostics.json | 640train/224A/144B | F0/F1 | YES | 1epoch;1seed |
| Proxy not validated forensic retention | outputs/jaes_v2/phase7/diagnostics_v1/proxy_decision.json | Development only | FrozenImageNet/H0 | NON_SPECIFIC | No known forensic cue labels |

Decisive Phase7 evidence hashes:

| path | sha256 |
| --- | --- |
| outputs/jaes_v2/audit/phase7_validation_final/validation.json | 89bf76a0c85a7b801289f6d716c049f7dadac03efbd9618bc5f86326cd6adf15 |
| outputs/jaes_v2/phase7/cache_partition_audit_v1.json | 4e9d5d66d327d290c4c0390636502dda0890e00ef64fd681ab047d80fd0edc18 |
| outputs/jaes_v2/phase7/cache_resource_gate_v1.json | 102bea94c4d74bde8f7f3238e69c50144db903fab547efe349e01f4c631c5a78 |
| outputs/jaes_v2/phase7/caches/benchmark_p0_v1/cache_metadata.json | 59490196cb3c846fe6d2784f18b1f31ab50d7293199852cc4820fa9b7eee0031 |
| outputs/jaes_v2/phase7/caches/medium_p0_v1/cache_metadata.json | 92d06f11728cc83e41bce1e2ca69ceee168407ece30bb3cecb2389080ece4ea5 |
| outputs/jaes_v2/phase7/diagnostics_v1/efficiency.json | 1f1f0062cea8284c84aafb30b6b862cd47abe802881874ea6b460681f8022c11 |
| outputs/jaes_v2/phase7/diagnostics_v1/online_rawboost_preflight.json | b9eb580b3c9637c6deee2fc926b64f2ccbc011263fee8bbed8b5b411a61c247a |
| outputs/jaes_v2/phase7/diagnostics_v1/proxy_controls.csv | 49d8b14fcb93088474412c9d0c25b03657be33cb7b8de7ef095ba8716b8f66d4 |
| outputs/jaes_v2/phase7/diagnostics_v1/proxy_decision.json | 7fbfa2bd82cdada5a23bc5412717f9053921a03e165e8dc26bb8412a404f36b3 |
| outputs/jaes_v2/phase7/diagnostics_v1/proxy_development_rows.csv | 873389168f378e3412f07ffde7758eaaf80d66718080535af905fd7e8de3aa35 |
| outputs/jaes_v2/phase7/diagnostics_v1/seal.json | 680a1921e2903067463cbb123c1a9fb94d16bc86fd6bc18a250d81a65e279afb |
| outputs/jaes_v2/phase7/executed_source_versions_v1.json | 99d09105cbc06ff8c17f2a0600a9baba94f45053fcd523023e6de9df680d7180 |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/BatchNorm_policy.json | 482b3426233150968d149ff57e6d34ceb56796a182a2ddf4add917fc9160c7d6 |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/data_hashes.json | 0811f44b4e4b8a1133b503393ef5835f02f6c1c04db3dc90f486e03e69eea4f4 |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/environment.json | c0d3dbcd13661f603193f56c997c2f1a46e0a4e6ee97c501a72e98054d41c4f3 |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/initialization_hash.json | 22c1e584f4726e7bcee71b26983301724388ee9ab46b62d29c7d6f5797642afc |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/resolved_config.json | 8c5d2cb39958db7bb84106f9caef112edc7ec44d711df2b441f3b60dd4da0441 |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/resource_measurements.json | 774fd31431c0fb0b1743bf0c694e8ae188b066d6e219b33b5c7ee20e4414e7fc |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/run_metadata.json | f9d9e6e6db7738d0eedda79dc9d7ba7327fe2d43ca9cbd31e5c2c08f0775f8a2 |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/status.json | d7d25e009c694b853710755d9167575e045f7ba092091fac227a00d13c457544 |
| outputs/jaes_v2/phase7/preflight/p7_f1_s6107_v1/trainable_layer_manifest.json | 39306d7760c7497ebadd46fad0dfac055bee1d8845087504ca5e81d55f2da5bc |
| outputs/jaes_v2/phase7/protocol_v1.json | 3c81a85793a706c3fabc809dbc48b6a79ef08cdeda0608b8f5db3b388aaeab0f |
| outputs/jaes_v2/phase7/representation_efficiency_v1.json | 23e3b0fd4efde46a424241a3bad765700811b27464f9a98d862b480372824784 |
| outputs/jaes_v2/phase7/resource_inventory_final.json | a6d77a4f17667cd55cff0a094fe1fa4f1c2c933c6efadb38b217bd0c7338bf8e |
| outputs/jaes_v2/phase7/resource_inventory_v1.json | 46bb07acf4ae31ea0de219fd5a2a729756b9d33c1d7d135b697deac25a43f502 |
| outputs/jaes_v2/phase7/seed_initialization_audit_v1.json | c2639ceecdc060f31f295c4f3cc0b770abed92c64f87e4a19983cb7641a82065 |
| outputs/jaes_v2/phase7/selections/medium_all_attack_v1/manifest.csv | 029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319 |
| outputs/jaes_v2/phase7/selections/medium_all_attack_v1/prior_development_overlap.json | 63daaaf77c77e69fe9c23159b632e8fe4ace5a4bcb5e763bec1843f6bcdc9c6c |
| outputs/jaes_v2/phase7/selections/medium_all_attack_v1/selection.json | 282a86ca8f915591590ea6f6589c287c6d05a3f67bd06263d0fe4e5599fac07d |
| outputs/jaes_v2/phase7/stageB_statistical_audit_v1.json | 9d89d6e446a4b5d39eb35c711d9b307cb14b68ca43943aba1a10551cef046f9d |
| outputs/jaes_v2/phase7/statistics/attack_metrics.csv | 206f22cb39aec41f44e692bbda55eabbd8690bcf6bb825c68955f7557c65bffc |
| outputs/jaes_v2/phase7/statistics/paired_comparisons.json | a48e43069083f9cbb075ca51f2aba9aab3d1c9a28628ebcc38eb2cacdb44663d |
| outputs/jaes_v2/phase7/statistics/pareto_exploratory.json | 222589c3d8c64f4b0b4a43381ff16693f33e93b475449f8b69bb601d64769a6a |
| outputs/jaes_v2/phase7/statistics/per_seed.csv | 54455d08baeca10c15a075b1ad6d515eee0ba4091da258df8f586b18102b03aa |
| outputs/jaes_v2/phase7/statistics/speaker_metrics.csv | b2b834a395146084980d06d065a810583cbaf72839406979f1d2edc8b2400514 |
| outputs/jaes_v2/phase7/statistics/summary.json | 585356cfd004b122fb8eea9984a437bbdf178a356270dfbd6e6014516bdc48c9 |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/best.keras | b17f0084b2f25f5767274dde3fe82ffcdad14c534a5a68e765170da57f9821b6 |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/checkpoint_selection.json | 6c7e664bafbe9b5879e7ca3651d30aee9a020d40037446f3d7044ef1b75f5afc |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/initialization_hash.json | 46d24e7b41cc067be2f081577bb73aa7ddd46ee3ee3070bd06225099cf713d22 |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/metrics.json | 6174608a94271e0dcf9d8ef2ed0a6e9103a2d7fa537b068495072e9389e598ae |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/resource_measurements.json | 4f83213b7f42c3ea9b70c0005bb00f8c5707184c0024b4b95785c7504b0671bf |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/run_metadata.json | 6723be6e8b4e1fc847212e2a3d9d35a53c976283a545bb27d425b32182861e80 |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/scores.csv | 1521bcad8c20f2c82718c222c1a6896e861ead0edfa19e6e0c0ccf39388a7d19 |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/training_history.csv | 99a80c4bbc5281e6370f40356609bd4d814738e029909c3855dff2a5a34d11a9 |
| outputs/jaes_v2/phase7/training/p7_f0_s6107_v1/training_views.csv | 9afa989b448ca844fc7a3a5bf82c6bcf22bf825bf2e7d1c4f7fe65ad3930ac61 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/best.keras | af2004818fe084579d1162bd291b36f42cf226e02b0e606ff89833b4f79945d1 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/checkpoint_reload.json | c56da45819c6860d1a5542bb9b589f00b23de053b77db3c05085f3d27e194d00 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/checkpoint_selection.json | fdc70d2bf4e352356847c3b64e8996e388bd61ec67d81acb8dcb9a63234b8e23 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/initialization_hash.json | 46d24e7b41cc067be2f081577bb73aa7ddd46ee3ee3070bd06225099cf713d22 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/metrics.json | 5f879d543484ce536cb5a4c46ac60484562d9746a104d550200da412a4e78ab6 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/resource_measurements.json | f31b4b4b2e8217c72576a41548cedc0945eb5f8db251df9beeb84e6051688f96 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/run_metadata.json | 917cc478a1473381e96c10a5c98ea1f2bcaab77d8c5d23a4d4e374c253fdd1d5 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/scores.csv | 1ab19ceca1b3a4141d3cdbfe12733548f7b89716abd3f08b37d852e4ddfeafbf |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/training_history.csv | 93f6c569b71f84dc8a407383ffd224c9c74e76bdc3262b4ab22ce3c48aa43d88 |
| outputs/jaes_v2/phase7/training/p7_f1_s6107_v1/training_views.csv | 9afa989b448ca844fc7a3a5bf82c6bcf22bf825bf2e7d1c4f7fe65ad3930ac61 |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/best.keras | 86c1dd14097eab4f4d5ff5fc807a54fb54653be377c951e0154c6b0632456766 |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/checkpoint_selection.json | 82871d0e7f577279fa299b9450085326b5ca93d9729e50c6783a1c6917c33c14 |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/initialization_hash.json | a8489814896474a91c5a4475509980abd64fbe5ddac8a135cf3cc118f603f61c |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/metrics.json | e8fceb96a8ef634a32ea54e36f5d87cd5c4d5f08049949248d59d08002feb418 |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/resource_measurements.json | 941fa83e22a37584da616d80f4404d32ce18eb743ce6bab2373470203cc701d4 |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/run_metadata.json | a2a498d775e4c528069d93ddcd072d13b73fb2c79c5d88032649fffc9291842e |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/scores.csv | 56211e3c045a63448252dfa6b90c2254e426f6dc9d5c02de9a4b5e65cddfa34b |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/training_history.csv | 2674898cbdf5018119ff18ff921b35d7e74be256a1ef9a9458cbb7fea20c8df7 |
| outputs/jaes_v2/phase7/training/p7_h0_s6107_v1/training_views.csv | 8cb530d4ef8564d2ebc315c56e705b70b42b4ab527581961f7462139f97a8139 |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/best.keras | 8fb10e7d341e5440669e764b4ed7bfffe81a948be320bdb99c6575c219cf52e2 |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/checkpoint_selection.json | b2026b1b79dbd1ec07d48357dcc76e3da8807f9daf7d7d7278add6c9a06f7571 |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/initialization_hash.json | da906bbaec692707013bb375eb967a3f25e824edfe4376b3a7a5d9c8fc3e924c |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/metrics.json | 8d6a19ae1b420ab36488827b926e0429878169282314190ff3ec55ff84e77430 |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/resource_measurements.json | 7ecf3bf26135d7c3902a8941d01a807575f856876495c2654a9fbcd6ca2bb7d9 |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/run_metadata.json | f5958d711d7faa78a471b68a6d602a085f2dc8f721a98fb9f256cc63324de8a0 |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/scores.csv | 6854a66cfc59f7ba4dcf22babc976eab74e6eac3fc1f694168450f184f1aa79e |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/training_history.csv | 5eb92a56698ff871e195255807a62acff89adfa3e5752d03ea7d965b288860b5 |
| outputs/jaes_v2/phase7/training/p7_h0_s6207_v1/training_views.csv | 93fa7d28d6024c1e43fc04a85467abae4f4234aa6f9d9bde5e57d5775358c850 |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/best.keras | bb9d31074cfe6138b71d0225c625e23828eaa3486974bc08e1cbff43f696d427 |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/checkpoint_selection.json | f213798f05622d8c2292abc944108a757ec8394223cf23e3bf19b9c95d682df5 |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/initialization_hash.json | 87bb756c5bbf0ced331d3012dd1131a270da2000d240d0f3b6fda9a65604a9fa |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/metrics.json | 598cc46c045f6e1693c712f6e3779852e768348fde9a1e331a16890b57292243 |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/resource_measurements.json | df3f4b3e251eb8be14d004ac0a872d3cc54e9be50fb1d0beb1e578d4d66877ab |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/run_metadata.json | 0773f05be7bd6fc3c88489154a1e20988e82c7b059108512ff94f714980a8fc4 |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/scores.csv | 5571b649f1cf75bca032b6957244609e487fdac2c7223f990abb97312bcdc8af |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/training_history.csv | eb36707107c2399e2a8dd0505ab12a699ca35a3bf3337904ca77a3cee77f5cb7 |
| outputs/jaes_v2/phase7/training/p7_h0_s6307_v1/training_views.csv | 52a3fdc7c9c80ab19fbaacdcb8ccfba755ac059fb94228f4df9f635fad7173c0 |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/best.keras | 0763cf3a9d9c7ab8b5cc562c80285d4947342ec74b79e075dc1b847f93105406 |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/checkpoint_selection.json | 043474ec72d9de169eb25e6721ba2ddcc823a55275dbe16831fdc7f891762d78 |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/initialization_hash.json | a8489814896474a91c5a4475509980abd64fbe5ddac8a135cf3cc118f603f61c |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/metrics.json | 0e98515107153b78dbc963f510af18edfc5beae62ec37db9016d85e6f47b1f7f |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/resource_measurements.json | 9c57ad92777d1218f814b9e6cf64a863160fe93d8438f9bc514adaa925645a24 |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/run_metadata.json | bbf0202fe431256d7bab89b481e68c3efd4e44d2a75ebe3caded0f67afc46f31 |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/scores.csv | 403b5a76c7a7e6f1eab3d8058f56beb36b2ed3ddac41481365649d24095e31cb |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/training_history.csv | 1a4af4b459c5019c250dc53cf211edfb06fb5449d7f366feff9e7555263d79db |
| outputs/jaes_v2/phase7/training/p7_h1_s6107_v1/training_views.csv | 72bdde473914022f3ca9911d3cb7dcbe5f1a576fa6641abe5cb572b448614568 |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/best.keras | 0dd78c6d6ff235b727afef2377e68cfd8cf9dc30a79899386b6f9fc2baa84d6b |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/checkpoint_selection.json | e1772635be54f8b703729b40dc0471de966e8fea5c6b5532fd90fb89df6c0d18 |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/initialization_hash.json | da906bbaec692707013bb375eb967a3f25e824edfe4376b3a7a5d9c8fc3e924c |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/metrics.json | c4c21fb0858b62216895463d337ad715617b54ea2bae4c2586e6af5a19c2b78a |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/resource_measurements.json | ac19012f6394f4f84117c7b13a6d9373e843f42d0b6ee039a475c14aa93d985d |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/run_metadata.json | a0be8158c70f234d751bce25544224051fd5671913ccb52aa796ab84c242d502 |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/scores.csv | e87978b63fa6fdc2d5125a1b64eaf33944f92b1d6171e45eb944a82f6ecebc6f |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/training_history.csv | 8d2e700ab03fd8eeee299d06e3f4bd9c36df5f871a74dd30cf1e13b6bb2b885d |
| outputs/jaes_v2/phase7/training/p7_h1_s6207_v1/training_views.csv | 5da8bb282cbb89f69d48abf72f2a499ad0be05a0d05ffba32f5f03abd7912306 |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/best.keras | ddfe8801c0c3d30b7b1af76ff2aab887e592975e552528a4123f5050ee00e4cd |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/checkpoint_selection.json | 77f1cc979fd8ba0f2e3d62ebabb49739e914804a92af81ab1f51da6389ca147e |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/initialization_hash.json | 87bb756c5bbf0ced331d3012dd1131a270da2000d240d0f3b6fda9a65604a9fa |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/metrics.json | 416218a7c179711d8ea2a4d26f9b3cfef508b1f930f954f0efded7d7ecf7e284 |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/resource_measurements.json | 20551bc79920833277499cc44981487aa6ef15b5fa33dd6847fc5405f2ca0931 |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/run_metadata.json | 42edf92935617ed1562e741c03d4e38a938a7b506e093c160ae94d18b85c4a81 |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/scores.csv | 9c8fae2b3e8643248711a904246c5d3465fa7efd79ea787a2c2685e7e9650ddb |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/training_history.csv | 39af14247424a8ae8bf795baa4fe04557dbc41b796ffb12fe1f3dc59eea41861 |
| outputs/jaes_v2/phase7/training/p7_h1_s6307_v1/training_views.csv | b1b39298bde4bdf8cdad934cf8bbb7be2eae4925c115b1304e8d7099ca563c57 |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/best.keras | 5344a41441d5e39c00d638aa4af918192d1184bd8ec5b546d70184e7ceae53e5 |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/checkpoint_selection.json | 5f55e2bd137945a798c34c4287205f12809142da2bbf4fd5058e9160bdef07f9 |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/initialization_hash.json | a8489814896474a91c5a4475509980abd64fbe5ddac8a135cf3cc118f603f61c |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/metrics.json | cea413711f6bad84319b1e6b419958823be3ea4ba73b8ceb7bfe5849af98b413 |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/resource_measurements.json | 84c361104b04d2da342c4353b0cc104006e8870700b35119a78513dc809e696f |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/run_metadata.json | 027bdbc70b6ea4f354a3181ed1105c57c74605e28242e4c9cb40fbaaa9978fab |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/scores.csv | becf414ca4ad4170294c3e5e1cd95390de3782b9327d1b5c703d18c343192614 |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/training_history.csv | e5b7a85159770781e88771a75c1f28dc36f265fa357435c9c1a661c70115b638 |
| outputs/jaes_v2/phase7/training/p7_h2_s6107_v1/training_views.csv | cfe452598da307eb252775f218f74eabdc6e9862ebab0b6ee69495f9956b8fd6 |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/best.keras | 1dc7a0a10f7c45e9ac6bedc24586add15eadea23e32df1d7bb28fb0641001ac9 |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/checkpoint_selection.json | 22d60ee73241914a06293d89b3dbf6c00ce7b8186cf660a1dee1bffc6567f287 |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/initialization_hash.json | da906bbaec692707013bb375eb967a3f25e824edfe4376b3a7a5d9c8fc3e924c |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/metrics.json | fa2432bbbfecb4ccf8cfc14b6b27b521c5068abe1fd5edf9c187992e3f4de9ca |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/resource_measurements.json | 576b6072c3d0f96b5bd2a6c579607c76a7694d3884b79a49634cca6c10ef47ee |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/run_metadata.json | 416eb96674a10e3cc9be72b9e3e9b9839d62e4ed7e8692192828f57c21613828 |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/scores.csv | 47d69c333a4bc779adc2eb8e0052fb861dcfa41316f5aa9839565de37d531c98 |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/training_history.csv | bc02da19edd04f8099140f38deb4210904a11d3f2a17c283635a1468a6de9c1f |
| outputs/jaes_v2/phase7/training/p7_h2_s6207_v1/training_views.csv | f25c2dbbb8096c7cb27cf705a7fe40ea7c0bde8f6f27fd17ee7f93ee8d58c25e |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/best.keras | c530cfd2a0a4bdb751b39fda60757b59f489becc17d87f5660ca74bc2cbc3589 |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/checkpoint_selection.json | 626ced9e7ebebbb8e6b3fce0fae9bacd0e89c65792ee42a847a73db9f60fce33 |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/initialization_hash.json | 87bb756c5bbf0ced331d3012dd1131a270da2000d240d0f3b6fda9a65604a9fa |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/metrics.json | bc8a1eda105c9d46a5b453c6a3b8f2bf66b9fb84e5b87b1f5f86edf64e4a8187 |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/resource_measurements.json | b67e91bc27db6a6b101607b17a7fbdf7c8bf4313cdb0d2c5010276175dc40826 |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/run_metadata.json | ca6d5430e7c7086a42c4cb9956d1a63e70d37c6581f1e70346bca32523206e88 |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/scores.csv | bba449c89dcc4253eee1a08a5b5c9fd5170e80fb8580051807da67d8ca15223a |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/training_history.csv | f6b735b84337493679e2409e996d78bef7d2b58cfef81d1ff032990360a5bfe0 |
| outputs/jaes_v2/phase7/training/p7_h2_s6307_v1/training_views.csv | 2cf0fe0257ac358588204b198f15c4b43ba223a1b6239c9ac5090b990d1fcedb |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/best.keras | 4066ad204682742456775cd2fbf3407acac458199702e90175659db0a77dcbc3 |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/checkpoint_selection.json | 5792c76e55bf65f2608de1f2a951332fe7bdf89f302ad960756d5f5623b38d6d |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/initialization_hash.json | a8489814896474a91c5a4475509980abd64fbe5ddac8a135cf3cc118f603f61c |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/metrics.json | e7d0baf05fef91acf1a0cba7e9f07e3e46b7550da465a7c0bd31f5a07a6afd71 |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/resource_measurements.json | 57d4ad54ebfda4de08ff91d38ffe23ef4bb439ced4a15fdbfb777eee6dc572e0 |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/run_metadata.json | e0d86e6166cc8417397dd7043584ffbe21ecb77cdfcab14a60c2bc6b7cfeea54 |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/scores.csv | 310f26652564af9cb172fc4eab416537206f7766260686df770e0345d5543633 |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/training_history.csv | 1e5800a2eb9f5942269f48f14a02bf22a2037a9aec10ff867f27cb6e440d0ba9 |
| outputs/jaes_v2/phase7/training/p7_h3_s6107_v1/training_views.csv | cfe452598da307eb252775f218f74eabdc6e9862ebab0b6ee69495f9956b8fd6 |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/best.keras | 50ef09c423b906b400f6f3cdaa20d39a748537694c482d4512b375af8726fb84 |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/checkpoint_selection.json | ab275e6ff35ad7eaa7d0bad9329bd56cd54908299f8a719086a00b707be2cb0d |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/initialization_hash.json | da906bbaec692707013bb375eb967a3f25e824edfe4376b3a7a5d9c8fc3e924c |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/metrics.json | 4d1c3c474408b8cf5fe3b1fc849dc648fd29f348aa138a69daadc23a8f1a879d |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/resource_measurements.json | 88c3616be9c3a8511db0c5b001de0207369f3cce4bc2c052e2bc03148c7ab485 |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/run_metadata.json | 594ba7451396632781d3efabfa84cad68a0308e3613203b535b0cb5f853e8b44 |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/scores.csv | 174d195d99a0fcc1e37c1aa0b99bb50a4adc620651627f9bd28118703d94f886 |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/training_history.csv | d851cd9e7f3e8b12600be4ba75d9c1096e59948260f2d06dd17ed3d756ccca6b |
| outputs/jaes_v2/phase7/training/p7_h3_s6207_v1/training_views.csv | f25c2dbbb8096c7cb27cf705a7fe40ea7c0bde8f6f27fd17ee7f93ee8d58c25e |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/best.keras | 54fd584f23a8db721f215308e23c34bcf29f40978c31a1065cafd7e6b7ae0d22 |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/checkpoint_selection.json | b30c431153b38305260556221712f773d1885e64fcbbb54e428a46f4fbbf6a63 |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/initialization_hash.json | 87bb756c5bbf0ced331d3012dd1131a270da2000d240d0f3b6fda9a65604a9fa |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/metrics.json | 9f9875bcbe68e4e6822b4005985a2412420e8b5116ddd0e6ce2288e432de9746 |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/resource_measurements.json | e03bc921ce3cae7746f0af9fcfcdc935d975982672b513b3b9d5931582923462 |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/run_metadata.json | 64ed13ae7750593b8a12df81b462d0bfcce9d6eec27d95c3d7842287d7629f5c |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/scores.csv | 83ee6ceb3f916630ffcc59076e11035c691168fa19d8babbbf88408cc93c8aa3 |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/training_history.csv | c2c51dc419a17a1b9b4bf6c39bee819839350b0e6b5f227a23940ab37ab794a1 |
| outputs/jaes_v2/phase7/training/p7_h3_s6307_v1/training_views.csv | 2cf0fe0257ac358588204b198f15c4b43ba223a1b6239c9ac5090b990d1fcedb |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/best.keras | e79c1e5bb0ef542359c847f0f7adf256464ef409cbbd064369bbe543ec2dd361 |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/checkpoint_selection.json | f2f629c2d7cddc30db4e977285bfa117ae28ddfbe2a2a03b9e43bb6c95ff5489 |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/initialization_hash.json | a8489814896474a91c5a4475509980abd64fbe5ddac8a135cf3cc118f603f61c |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/metrics.json | 4bbbddf646efb102aa4986fb76404ef66f932e1f836ab766bffa10892a27aa7d |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/resource_measurements.json | f2c06a43d357d0aa2f5a154fd1741ee6ddf828b022cc440d622d760628fabbb7 |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/run_metadata.json | ce2732403fcfbaf71a82d5b7c380979c11627d1ba43ff2b6967fa69b0047fd21 |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/scores.csv | e77ff970a80c6bcc135c01191643dafc7b075e9609da07f516593d535c4e9ee0 |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/training_history.csv | 45fefe9f255b31809eed6f0d1bb86cb03f0457de8e735b61135ad50df0f63966 |
| outputs/jaes_v2/phase7/training/p7_h4_s6107_v1/training_views.csv | 5ec06a17848de02b32ddee8c2f9b044442a45d1bc03ad78641fb5f76b10bc4a3 |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/best.keras | 1e2209c28eb7e1843f68bd86469a445eabc6445db369e0ffaa5b266a1fb0a905 |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/checkpoint_selection.json | b6dcdcaf2c8bc9bd98b5e9e1e49f505a74a05f52fdb358ec22dc3de63c86ec7c |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/initialization_hash.json | da906bbaec692707013bb375eb967a3f25e824edfe4376b3a7a5d9c8fc3e924c |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/metrics.json | bd2bb408b6a74926ca02645ef626a6316acfbc6d36e86aee47dca63c8cd8edc5 |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/resource_measurements.json | 34769358abd6b34f5a43fd35eb402a34839db172c4efc1789332e9336d1f42d1 |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/run_metadata.json | fa7b580d0bfa34087c5df71073512e616dc232340cfd603de232e9d3b9cfd02c |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/scores.csv | 2da745c79f108d1976c7d072a07e4d7108946caf29076a5ffd4c54b58e399cd8 |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/training_history.csv | 5e9fc180e88b50dc64824f2de58cb91edf3fa54d05ce04bbd007723e01467170 |
| outputs/jaes_v2/phase7/training/p7_h4_s6207_v1/training_views.csv | 9aa0851b7d292694769d507c838fb7e37f1664261e923c131b1c13a4534490fe |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/best.keras | b52e5fa3cdf3ae426b129051202134f325ce31c3f1f28b7821dd2f935f07eb68 |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/checkpoint_reload.json | bd3cc9952df9b5c97734b3ba05964036c9e9f36f6315315d145686e0c3eba15e |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/checkpoint_selection.json | 4845de502c50feb875ba24d6e6679e90125c7504505839cab16fb776457856c9 |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/initialization_hash.json | 87bb756c5bbf0ced331d3012dd1131a270da2000d240d0f3b6fda9a65604a9fa |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/metrics.json | 184e773b8e187261637ef61b7e348fd10433cb4e242e0d878222ddf51b06f9d8 |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/resource_measurements.json | 8d686c2423013bdf4a8c2c4367ec76909e42433296e47805255734a5d5ac5241 |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/run_metadata.json | 6ef53f92b9f290141e66fc1152064459ea650b9181f53aaa59916dc0656a5184 |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/scores.csv | cf60607a2bcc8b6fee3084b0efb33daf3ea6b00d26d2f5b0e364a7cfd6db76fc |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/training_history.csv | 4493e6487c5178294af204712b83dddec1f1caf5cb8525d0ef7752f7aec1e08c |
| outputs/jaes_v2/phase7/training/p7_h4_s6307_v1/training_views.csv | 300a7febe623bc91e8d5c06c9d4445f467a25ccdd2e880ab5907f9370d91292a |

## 28. Remaining Research Blockers

Method: no validated forensic retention estimator or distinct objective. Data: only7primary speakers; attack IDs are seen conditions, exact generators/source lineage unknown. Statistics: only3training blocks on one selected corpus. Compute: CPU-only validated training; online/longer multi-seed representation training needs separate measured budgets. Validation: external and reserved calibration remain NOT_LOCKED. Journal: conference venue, publication/copyright details and current official index verification remain unresolved. No amount of repeated-view counts removes these blockers.

## 29. Journal Submission Readiness

NOT_READY_FOR_JOURNAL_SUBMISSION. Target JAES covers audio research, including relevant forensic/ML work, but successful software tests and standard-baseline improvements alone do not certify a notable original contribution. Consult the current official AES instructions and disclose/cite the accepted conference work. The exact venue/DOI/publication/rights facts are not supplied; no arbitrary extension percentage or guaranteed acceptance is asserted. Current Clarivate/JCR listing is NOT_VERIFIED in this implementation phase. No submission was made.

The [official AES journal site](https://aes.org/publications/journal/) remains active and provides a submission link. The [official masthead](https://aes.org/publications/journal/journal-masthead/) identifies ISSN1549-4950 and2026publication information. This establishes the publisher's current publication/submission presence, not a current Clarivate indexing certificate.

The [author guidelines, Convention & Conference Papers section](https://aes.org/publications/journal-author-guidelines/) permit consideration of substantially revised conference work: extend the treatment and literature review, cite the event paper, and use a different title. Acceptance remains editorial; an accepted but unreleased manuscript does not resolve rights, disclosure or simultaneous-submission questions. The exact conference venue, publication/DOI status and rights terms are still required. The publisher does not specify a universal extension percentage or a minimum of three international contributions on this page.

Current SCIE/JCR status remains NOT_VERIFIED from Clarivate: the [official MJL lookup](https://mjl.clarivate.com/search-results?issn=1549-4950) was inaccessible through the research web tool. Third-party directories are not substituted for an authenticated current record, and no current JIF/quartile is certified. This limitation does not mean the journal is invalid or delisted.

LAVA is NOT_READY_FOR_SUBMISSION as a newly established methodological paper. Phase7 provides controlled development evidence and software traceability; it does not establish three distinct international novelties, independent external confirmation or calibrated operational risk. The conference manuscript and all conference models remain unchanged. No submission, author communication, rights transfer or publication action occurred.

## 30. Phase8 Roadmap

1. Review the full per-seed classwise trade-offs and H3–H2 consistency, including guardrail failures and A05/A06.
2. If representation pilot is promising, lock a matched, longer schedule with warmup/checkpoint policy and measured CPU/GPU costs before3seed expansion; do not infer convergence from1epoch.
3. Validate a technically defined forensic cue estimator against positive, nuisance and mechanism-specific negative controls; abandon it if severity/shuffled controls explain results.
4. Compare appropriate representation/online/adversarial controls before claiming a method.
5. Freeze a selected model family and hyperparameters, then separately lock calibration/external source/license/overlap/operating point and primary statistical comparisons.
6. Resolve conference extension rights and current official journal indexing before submission. The large tier, cascade, final evaluation and new loss are not automatic Phase8 actions.

## 31. Exact Safe Next Commands

All commands run from `D:/audio-deepfake-mobilenet-lstm`; existing output identities reject overwrite. Reproduce reports, not training, to regenerate numerical tables.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase7_report --output outputs/jaes_v2/phase7/reproduction_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.build_master_research_report_phase7 --output outputs/jaes_v2/phase7/master_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase7_validate --output outputs/jaes_v2/audit/phase7_validation_review
```

Executed config-specific training commands and all hashes remain in run metadata. The GPU probe/package is a portability readiness step, not a validated GPU training result. New training requires new registered output IDs and applicable measured gates. Calibration/evaluation are fail-closed through `phase7_readiness.py`; these reproduction commands do not read their manifests/audio/labels.

