# LAVA JOURNAL V2 — MASTER RESEARCH REPORT

Updated 2026-10-30 Asia/Novosibirsk. Evidence-generated handoff for independent scientific review. Phase1–5 evidence is preserved; this report supersedes outdated current-status prose without rewriting its historical files.

## 1. Executive Research Summary

Project: LAVA, a lightweight benchmarking framework for robust and efficient deepfake voice detection. The accepted conference work compares deployed artifacts with heterogeneous initialization and thresholds. The journal extension asks which measured failure mechanisms survive source/threshold controls and whether a lightweight learned detector can improve robustness at an explicit clean-risk cost. Current maturity: controlled development engineering and standard-baseline pilots. There is no independently confirmed new methodological contribution, no claim of three international novelties and no journal-ready conclusion. New research artifacts are separate from frozen production models. Phase6 adds balanced speaker/class sampling, amplitude and equal-view consistency controls, finite RawBoost integration, reliability diagnostics, a real partial-backbone gradient sanity path, and this cross-phase inventory. Limits include small cohorts, one seed, unknown original-source lineage, bounded CPU resources, and protected calibration/evaluation that remain unused.

## 2. Research Questions

### Accepted conference questions

- RQ1: How do deployed artifacts compare on clean AUC and EER, and how should threshold-dependent metrics be interpreted under heterogeneous training and calibration?

- RQ2: How does absolute performance change under the executed noise, codec, and simulated-replay conditions on the fixed diagnostic subset?

- RQ3: Which artifacts are non-dominated across diagnostic-subset EER, mean stressed F1, and end-to-end RTF?

These are artifact comparison, diagnostic stress and scenario-specific efficiency questions; training and pretraining are heterogeneous, so the answers concern artifacts rather than intrinsic architecture superiority.

### Journal questions

J-RQ1: Does channel order create practically important classwise risk changes after insertion-power, source, preprocessing and operating-point controls? Exploratory evidence exists; the fixed-generator source-matched confirmation lacks adequate support. J-RQ2: Can ordinary amplitude/channel/paired training or a genuinely distinct evidence-preserving mechanism improve a lightweight detector subject to bounded clean REAL false alarms? Standard controls are being tested; a distinct mechanism is not established. J-RQ3: Can selective inference improve matched-cost risk/coverage with independent calibration and a validated second stage? Unanswered; C3 is deferred.

## 3. Complete Phase Timeline

| Phase | Objective | Implementation | Executed run IDs | Finding / status |
|---|---|---|---|---|
| 1 | Ordered noise/codec feasibility | Shared source variants, strict scores and parent bootstrap | c1_noise_opus_pilot_v1, c1_noise_opus_pilot_v2, c1_noise_opus_pilot_v3 | Observed reversal/interaction is descriptive; insertion-point noise scaling is a confound |
| 2 | Confounder-aware development | Fixed versus relative noise, headroom/PCM/delay controls and metadata audit | c1_channel_controls_dev_v1 | Channel order cannot be isolated from amplitude, codec gain and preparation by one average metric |
| 3 | External MLAAD-tiny development | Pinned public acquisition, source provenance, common cache and two frozen artifacts | c1_mlaad_tiny_dev_v1_mobilenetv3_lstm, c1_mlaad_tiny_dev_v1_shufflenetv2_lstm | Class-asymmetric noisy-channel failures persist; author system labels are not verified unseen checkpoints |
| 4 | Source matching and nuisance diagnosis | Source-ID/text matching levels, confirmation feasibility and within-record silence/gain controls | c1_nuisance_controls_dev_v1_mobilenetv3_lstm, c1_nuisance_controls_dev_v1_shufflenetv2_lstm | Sparse source support blocks the chosen matched-generator design; absolute Mel floor explains feature non-invariance |
| 5 | Controlled data and standard learner | ASV foundation, reserved calibration, P0/P1/P2 and separately trained head-only students | c2_overfit_sanity_v1, c2_p0_aug_pilot_v2, c2_p0_erm_pilot_v1, c2_p0_erm_pilot_v2, c2_p1_aug_pilot_v2, c2_p1_erm_pilot_v1, c2_p1_erm_pilot_v2, phase5_preprocessing_study_v1 | Gain invariance can coexist with worse discrimination; clean total error hides opposite REAL/FAKE changes |
| 6 | Stronger controlled comparisons and consolidation | Speaker-balanced cohort, B1/B4/control/finite RawBoost, partial-backbone sanity and auditable master report | partial_finetune_sanity_v1, partial_finetune_sanity_v2, p6_p0_b0_v1, p6_p0_b1_v1, p6_p0_b2_v1, p6_p0_b4_control_v1, p6_p0_b4_v1, p6_p0_rb_v1, p6_p1_b0_v1, p6_p1_b1_v1, p6_p1_b2_v1 | Read actual Phase6 results below; implementation and learning success do not establish novelty |

Each phase is incremental. Phase1–4 use frozen conference predictors; Phase5 introduces fresh research students. The matched-generator hypothesis is not silently renamed as a successful experiment. Phase6 uses a new cohort and therefore is not a direct continuation of Phase5 accuracy values.

## 4. Dataset Inventory

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

## 5. Dataset Split Protocol

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

## 6. Complete Model Inventory

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

## 7. Complete Experiment Inventory

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

## 8. Reproducibility Status

Initial and current checks preserve 3235 historical Journal output files. Protected baseline result: PASS (2376 files); 3 explicit Phase 2 source/test amendments; original snapshot retained. Accepted manuscript SHA256: `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`. Run seals are verified against every declared output hash; unsealed/incomplete experiments are labeled rather than repaired with invented evidence. Model-free score aggregation recomputes per-condition metrics and figures, not neural inference. RawBoost source bytes/license, selected waveform hashes, transform seeds, pretraining hash and all new checkpoints are traceable. Exact waveform byte hashes certify the generated local realization, not full original-source independence.

Regression evidence:

```json
{
  "jaes_v2": {
    "run": 104,
    "failures": 0,
    "errors": 0,
    "skipped": 0,
    "successful": true,
    "failure_details": []
  },
  "legacy_selected": {
    "run": 40,
    "failures": 0,
    "errors": 0,
    "skipped": 0,
    "successful": true,
    "failure_details": []
  }
}
```

## 9. Phase1–4 Scientific Findings

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

## 12. Scientific Novelty Assessment

| Candidate | Nearest prior art | Distinct mechanism | Evidence | Limitation | Status |
|---|---|---|---|---|---|
| C1 | RADAR / RawBoost / standard interaction estimands | No proven new general mechanism | Paired controls and public development | Source-matched confirmation blocked | SUPPORTING_VALIDATION_ENGINEERING |
| C2 | F-SAT / PCL-NET / DK-CAST / RawBoost | Evidence-retention weighting is not sufficiently distinguished | Stronger baseline pilot and real gradient path | Proxy validity, seed variability, external evaluation missing | NOVELTY_NOT_ESTABLISHED |
| C3 | Selective prediction / matched-cost cascades | NOT_IMPLEMENTED | Prerequisites and reserved calibration | No validated second stage / risk protocol | DEFERRED |

International methodological novelty, a new local empirical observation, independent confirmation, engineering implementation and reproducibility are different claims. F-SAT constrains adversarial spectral perturbations; PCL-NET applies feature-level paired consistency; DK-CAST already weights/distills distortion-aware supervision. Changing backbone or summing known terms does not establish novelty. The detailed primary-source matrix and method rejection/design gate are PHASE6_NEAREST_PRIOR_ART_MATRIX.md and PHASE6_C2_METHOD_PROPOSAL.md. No new loss was implemented after the novelty-plausibility gate failed.

## 13. Research Failures and Negative Results

The fixed-generator source-matched design lacked support; unknown lineage limits causal claims. Gain-invariant preprocessing can suppress useful cues or shift input distributions. Augmentation may improve misses while increasing clean false alarms. Small data, threshold sensitivity and frozen-backbone restrictions remain material. Teacher runtime/checkpoint-quality gates prevent KD. Phase6 initially ran cache and fine-tuning TensorFlow processes concurrently; cache reported an allocation failure and the fine-tuning process terminated without a completed result. Sequential execution and microbatching were introduced; native termination without a traceback is not falsely diagnosed as a measured gradient failure. Failed outputs are preserved, never reused as complete caches or published quality evidence.

| Run | Phase | Status | Output | Counts/steps retained |
|---|---|---|---|---|
| c2_p1_erm_pilot_v1 | 5 | INTERRUPTED | outputs/jaes_v2/c2/c2_p1_erm_pilot_v1 | UNKNOWN |
| partial_finetune_sanity_v1 | 6 | FAILED_PROCESS_TERMINATION | outputs/jaes_v2/phase6/partial_finetune_sanity_v1 | UNKNOWN |

Resource-stopped but sealed exploratory runs and superseded figure/report versions remain in their historical directories. Their presence does not add independent replication or favorable evidence.

## 14. Model and Training Architecture

Source bytes → deterministic official-speaker/class selection → shared seeded waveform views → P0/P1 preprocessing → MobileNetV3Small → LSTM128 → Dense64/dropout → sigmoid FAKE score. Source labels are REAL0/FAKE1. P0 calls production preprocessing; P1 uses per-segment relative power flooring with explicit silent-segment behavior. Both retain the temporal image dimensions. Head-only training materializes detached frozen-backbone embeddings once in RAM, reusing the same source/condition cells across objectives. BCE supervises each sampled view; equal-two-view control averages two BCE terms; B4 adds0.1 mean squared probability discrepancy. ImageNet weight hash, seed, LR decay, Adam updates, view logs, clean-development-BCE checkpoint selection and reload evaluation are recorded. Partial fine-tuning uses raw/preprocessed train tensors, calls the backbone inside the tape, reuses the existing tail-unfreeze policy, freezes BN statistics, creates a fresh optimizer after trainability changes, and audits actual variable gradients/updates and checkpoint parity. Frozen embeddings cannot support CNN updates. Microbatching bounds memory; its tiny sanity loss is not development performance.

## 15. Statistical Methodology

Each source contributes paired conditions; conditions are not independent recordings. Known official speakers or author source-text clusters define descriptive bootstrap units. Uniform-cluster and recording-weighted estimands differ when group sizes or class support differ. Parent-byte grouping handles only known local variants; it does not recover unknown original utterance ancestry. Training initialization uncertainty needs independently trained seeds and is not replaced by bootstrap replication. Phase6 is one seed with prospective primary comparison and clean classwise guardrails; secondary comparisons are descriptive, without significance claims or selective unadjusted p-values. The future confirmatory family requires multiplicity control and at least three seeds. A sigmoid is a bounded score. NLL/Brier/ECE/classwise reliability diagnostics are implemented without fitting calibration; small-bin ECE is unstable at this sample size. A diagnostic EER threshold is not used as an operating policy. Reserved calibration cannot be consumed by a config flag or a proposal alone; confirmation remains disabled.

Locked Phase6 protocol:

```json
{
  "status": "LOCKED_BEFORE_MAIN_TRAINING",
  "date": "2026-10-30",
  "confirmation_enabled": false,
  "selection_sha256": "45ff123a734e11935e9e7f7a1ee9259f950cc82f7441bf06b6ca96cd56d87965",
  "primary_comparison": [
    "p6_p0_b4_v1",
    "p6_p0_b4_control_v1"
  ],
  "primary_metric": "Equal-official-speaker mean FAKE miss over noise5,A_then_B,B_then_A",
  "clean_risk_margin": 0.03571428571428571,
  "margin_rationale": "Exploratory screen: at most one additional clean error in each 28-record class; not a deployment-certified tolerance",
  "operating_threshold": 0.5,
  "calibration_fitted": false,
  "uncertainty": "Paired known-speaker bootstrap; single seed; no discovery/significance claims",
  "maximum_training_seconds": 180,
  "epochs": 6,
  "optimizer_steps": 60,
  "seed": 5806,
  "finite_rawboost_realizations": 2,
  "bank": "All methods access same cache; B4 vs control same two supervised views and updates; one-view methods have half view exposure",
  "candidate_new_loss_enabled": false,
  "medium_large_training_enabled": false
}
```

Clean bounded-score reliability, no calibration fitted:

| Run | NLL | Brier | ECE |
|---|---|---|---|
| p6_p0_b0_v1 | 0.010258434928165791 | 0.001152762928903331 | 0.009586367140790204 |
| p6_p0_b1_v1 | 0.010015052232830137 | 0.001516326232567234 | 0.009087014967268024 |
| p6_p0_b2_v1 | 0.051123702580879914 | 0.013812425703467596 | 0.027201449515164948 |
| p6_p0_b4_control_v1 | 0.0197778874308154 | 0.0008870066300838399 | 0.01930717236661753 |
| p6_p0_b4_v1 | 0.020578203425197093 | 0.001005149191263211 | 0.020037868683825122 |
| p6_p0_rb_v1 | 0.021491808031863315 | 0.0021352295268495156 | 0.020237587300860976 |
| p6_p1_b0_v1 | 0.11446088475357839 | 0.03444716848395858 | 0.052311618658547686 |
| p6_p1_b1_v1 | 0.11446088475357839 | 0.03444716848395858 | 0.052311618658547686 |
| p6_p1_b2_v1 | 0.14814531963304706 | 0.04383923990089602 | 0.05740198970306663 |

## 16. Resource and Hardware Inventory

Current measurements and historical timing have different execution boundaries; the laptop host is not automatically the conference paper machine. Physical GPU inventory is not a working CUDA runtime. Existing TensorFlow and Torch CPU environments remain unchanged. Training-state caches and teacher load costs must be counted separately from deployed student inference.

| Measured field | Value |
|---|---|
| platform | Windows-10-10.0.26300-SP0 |
| cpu_count | 12 |
| gpu_inventory_not_used_by_c1 | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB |
| packages | {'numpy': '1.24.3', 'scipy': '1.11.4', 'soundfile': '0.12.1', 'matplotlib': '3.8.2', 'scikit-learn': '1.3.2', 'tensorflow': '2.15.0', 'onnxruntime': '1.17.3'} |
| free_disk_bytes | 26684940288 |
| ffmpeg_version | ffmpeg version 8.0.1-full_build-www.gyan.dev Copyright (c) 2000-2025 the FFmpeg developers |

Phase6 resource gate:

```json
{
  "status": "PASS",
  "measured_encoder_batch2_seconds": 0.33671840000048786,
  "decode_two_preprocessors_seconds": 0.08522129999983008,
  "rawboost_seconds": 0.030355599999893457,
  "head_forward_batch16_seconds": 0.08437579999917943,
  "head_backward_batch16_seconds": 0.09531229999993229,
  "peak_ram_mib": 495.44140625,
  "gpu_available": false,
  "projected_cache_seconds": 680.1762664003958,
  "cache_budget_seconds": 1100,
  "embeddings": 2704,
  "train_sources": 160,
  "dev_sources": 56,
  "head_data_loading": "Materialize each NPZ array once; Phase5 per-sample decompression eliminated in new path"
}
```

Measured/extrapolated size alternatives (estimates are not runtime promises):

```json
[
  {
    "tier": "small",
    "per_cell": 4,
    "counts": {
      "train": {
        "recordings": 160,
        "real": 80,
        "fake": 80,
        "speakers": 20,
        "attacks": 4
      },
      "development": {
        "recordings": 56,
        "real": 28,
        "fake": 28,
        "speakers": 7,
        "attacks": 4
      }
    },
    "embedding_count": 2704,
    "embedding_float32_bytes": 37380096,
    "projected_cache_seconds": 680.1762664003958,
    "training_estimate": "Measure actual optimized small-run throughput before extrapolation",
    "decision": "EXECUTE_SINGLE_SEED_BOUNDED",
    "independence": "More records do not create new original-source identity or more speakers"
  },
  {
    "tier": "medium",
    "per_cell": 16,
    "counts": {
      "train": {
        "recordings": 640,
        "real": 320,
        "fake": 320,
        "speakers": 20,
        "attacks": 6
      },
      "development": {
        "recordings": 224,
        "real": 112,
        "fake": 112,
        "speakers": 7,
        "attacks": 6
      }
    },
    "embedding_count": 10816,
    "embedding_float32_bytes": 149520384,
    "projected_cache_seconds": 2420.705065601583,
    "training_estimate": "Measure actual optimized small-run throughput before extrapolation",
    "decision": "NOT_EXECUTED_AWAITING_COMPUTE_BUDGET",
    "independence": "More records do not create new original-source identity or more speakers"
  },
  {
    "tier": "large",
    "per_cell": 64,
    "counts": {
      "train": {
        "recordings": 2560,
        "real": 1280,
        "fake": 1280,
        "speakers": 20,
        "attacks": 6
      },
      "development": {
        "recordings": 896,
        "real": 448,
        "fake": 448,
        "speakers": 7,
        "attacks": 6
      }
    },
    "embedding_count": 43264,
    "embedding_float32_bytes": 598081536,
    "projected_cache_seconds": 9382.820262406332,
    "training_estimate": "Measure actual optimized small-run throughput before extrapolation",
    "decision": "NOT_EXECUTED_AWAITING_COMPUTE_BUDGET",
    "independence": "More records do not create new original-source identity or more speakers"
  }
]
```

The current cache and each run bind actual elapsed seconds, bytes, process peak RAM and updates. Memory-pressure failure shows why independent jobs must run sequentially on this desktop. No multi-day job, full dataset fine-tuning, teacher install or extra large dataset acquisition is automatic.

Current-volume constraints and new full-student warm CPU timings:

```json
{
  "system": {
    "date": "2026-10-30",
    "disk_D_bytes": 26686615552,
    "disk_C_bytes": 446820352,
    "production_environments_unchanged": true,
    "concurrency_policy": "One Phase6 CLI job at a time, enforced by nonblocking file mutex"
  },
  "efficiency": {
    "status": "MEASURED_OFFLINE_CPU",
    "repetitions": 5,
    "warmup_excluded": 1,
    "results": [
      {
        "run": "p6_p0_b2_v1",
        "sample_id": "ASVspoof2019_LA:LA_D_1218000",
        "source_sha256": "a64038a5636bf3fac907233211c3b4e06aa4a46a5883c898eab03bdb2b4f33f4",
        "observed_seconds": 3.0,
        "checkpoint_sha256": "573f16fa0d17b188a47e6a52dbe661ab97b5e07fb5f9511bbb3af99704818ba3",
        "parameters": 1308401,
        "checkpoint_bytes": 5916718,
        "median_seconds": {
          "decode_seconds": 0.002763500000583008,
          "preprocessing_seconds": 0.011118699998405646,
          "model_seconds": 0.23471579999932146,
          "end_to_end_seconds": 0.24809979999918141
        },
        "offline_rtf": 0.08269993333306047,
        "process_lifetime_peak_mib": 464.1484375,
        "raw_timings": [
          {
            "decode_seconds": 0.003468799999609473,
            "preprocessing_seconds": 0.012479400000302121,
            "model_seconds": 0.22954380000010133,
            "end_to_end_seconds": 0.24549200000001292
          },
          {
            "decode_seconds": 0.0025272999992012046,
            "preprocessing_seconds": 0.01038460000017949,
            "model_seconds": 0.23518789999980072,
            "end_to_end_seconds": 0.24809979999918141
          },
          {
            "decode_seconds": 0.002763500000583008,
            "preprocessing_seconds": 0.011118699998405646,
            "model_seconds": 0.2433961000006093,
            "end_to_end_seconds": 0.25727829999959795
          },
          {
            "decode_seconds": 0.0029775000002700835,
            "preprocessing_seconds": 0.011297600000034436,
            "model_seconds": 0.23471579999932146,
            "end_to_end_seconds": 0.24899089999962598
          },
          {
            "decode_seconds": 0.002592799999547424,
            "preprocessing_seconds": 0.009676300000137417,
            "model_seconds": 0.22352590000082273,
            "end_to_end_seconds": 0.23579500000050757
          }
        ]
      },
      {
        "run": "p6_p0_b4_control_v1",
        "sample_id": "ASVspoof2019_LA:LA_D_1218000",
        "source_sha256": "a64038a5636bf3fac907233211c3b4e06aa4a46a5883c898eab03bdb2b4f33f4",
        "observed_seconds": 3.0,
        "checkpoint_sha256": "6283e1c421246c5fcaf3ca533b9a7d3496483634f6957a8838f891fde9c184b4",
        "parameters": 1308401,
        "checkpoint_bytes": 5916718,
        "median_seconds": {
          "decode_seconds": 0.002530300000216812,
          "preprocessing_seconds": 0.010412199999336735,
          "model_seconds": 0.2244602999999188,
          "end_to_end_seconds": 0.2376237000007677
        },
        "offline_rtf": 0.0792079000002559,
        "process_lifetime_peak_mib": 481.62109375,
        "raw_timings": [
          {
            "decode_seconds": 0.0025758000010682736,
            "preprocessing_seconds": 0.010148699999263044,
            "model_seconds": 0.24147050000101444,
            "end_to_end_seconds": 0.25419500000134576
          },
          {
            "decode_seconds": 0.002700699998968048,
            "preprocessing_seconds": 0.010905499999353196,
            "model_seconds": 0.23176540000167734,
            "end_to_end_seconds": 0.24537159999999858
          },
          {
            "decode_seconds": 0.002530300000216812,
            "preprocessing_seconds": 0.010633100000632112,
            "model_seconds": 0.2244602999999188,
            "end_to_end_seconds": 0.2376237000007677
          },
          {
            "decode_seconds": 0.0025153999995382037,
            "preprocessing_seconds": 0.010107500000231084,
            "model_seconds": 0.22403029999986757,
            "end_to_end_seconds": 0.23665319999963685
          },
          {
            "decode_seconds": 0.002490400000169757,
            "preprocessing_seconds": 0.010412199999336735,
            "model_seconds": 0.2210601000006136,
            "end_to_end_seconds": 0.2339627000001201
          }
        ]
      },
      {
        "run": "p6_p0_b4_v1",
        "sample_id": "ASVspoof2019_LA:LA_D_1218000",
        "source_sha256": "a64038a5636bf3fac907233211c3b4e06aa4a46a5883c898eab03bdb2b4f33f4",
        "observed_seconds": 3.0,
        "checkpoint_sha256": "dabb61807bdc7d10ff79e28cbd83d7b27dad712b79bb62b4715a62f45b9e0841",
        "parameters": 1308401,
        "checkpoint_bytes": 5916718,
        "median_seconds": {
          "decode_seconds": 0.0025599999989935895,
          "preprocessing_seconds": 0.010381199999756063,
          "model_seconds": 0.2306298000003153,
          "end_to_end_seconds": 0.2441610000005312
        },
        "offline_rtf": 0.08138700000017707,
        "process_lifetime_peak_mib": 503.08203125,
        "raw_timings": [
          {
            "decode_seconds": 0.002613700000438257,
            "preprocessing_seconds": 0.010917499999777647,
            "model_seconds": 0.2306298000003153,
            "end_to_end_seconds": 0.2441610000005312
          },
          {
            "decode_seconds": 0.0025740000000951113,
            "preprocessing_seconds": 0.00994399999945017,
            "model_seconds": 0.22754859999986365,
            "end_to_end_seconds": 0.24006659999940894
          },
          {
            "decode_seconds": 0.002491100000042934,
            "preprocessing_seconds": 0.010381199999756063,
            "model_seconds": 0.22579790000054345,
            "end_to_end_seconds": 0.23867020000034245
          },
          {
            "decode_seconds": 0.0025599999989935895,
            "preprocessing_seconds": 0.010398000000350294,
            "model_seconds": 0.23619090000102005,
            "end_to_end_seconds": 0.24914890000036394
          },
          {
            "decode_seconds": 0.002499599999282509,
            "preprocessing_seconds": 0.009403899999597343,
            "model_seconds": 0.23438619999978982,
            "end_to_end_seconds": 0.24628969999866968
          }
        ]
      },
      {
        "run": "p6_p0_rb_v1",
        "sample_id": "ASVspoof2019_LA:LA_D_1218000",
        "source_sha256": "a64038a5636bf3fac907233211c3b4e06aa4a46a5883c898eab03bdb2b4f33f4",
        "observed_seconds": 3.0,
        "checkpoint_sha256": "cb74cd9a73986585b2e1f1a6091c487585f800350ec1cfcd34509f1b722224d6",
        "parameters": 1308401,
        "checkpoint_bytes": 5916718,
        "median_seconds": {
          "decode_seconds": 0.002565400000094087,
          "preprocessing_seconds": 0.01053190000129689,
          "model_seconds": 0.23670769999989716,
          "end_to_end_seconds": 0.2499428999999509
        },
        "offline_rtf": 0.08331429999998363,
        "process_lifetime_peak_mib": 503.19921875,
        "raw_timings": [
          {
            "decode_seconds": 0.002565400000094087,
            "preprocessing_seconds": 0.010028000000602333,
            "model_seconds": 0.23642189999918628,
            "end_to_end_seconds": 0.2490152999998827
          },
          {
            "decode_seconds": 0.0025413000003027264,
            "preprocessing_seconds": 0.010967199999868171,
            "model_seconds": 0.24033679999956803,
            "end_to_end_seconds": 0.2538452999997389
          },
          {
            "decode_seconds": 0.002515500000299653,
            "preprocessing_seconds": 0.010719699999754084,
            "model_seconds": 0.23670769999989716,
            "end_to_end_seconds": 0.2499428999999509
          },
          {
            "decode_seconds": 0.0025838999990810407,
            "preprocessing_seconds": 0.01053190000129689,
            "model_seconds": 0.23553549999996903,
            "end_to_end_seconds": 0.24865130000034696
          },
          {
            "decode_seconds": 0.0034593000000313623,
            "preprocessing_seconds": 0.01015689999985625,
            "model_seconds": 0.24456070000087493,
            "end_to_end_seconds": 0.25817690000076254
          }
        ]
      }
    ],
    "limitations": "One declared dev recording; sequential process lifetime peaks, not isolated per-model RAM; no streaming/energy or timing-significance claim"
  }
}
```

## 17. Complete File and Module Map

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

## 18. Evidence Traceability Matrix

| Claim / numerical source | Cohort/model | Metric | Status | Verified evidence |
|---|---|---|---|---|
| Conference clean performance | mobilenetv3_lstm | {'samples': 2737, 'eer': 0.02495697074010328, 'f1': 0.9724137931034482, 'roc_auc': 0.9911466273256291} | BENCHMARKED | outputs/lava_5/clean/mobilenetv3_lstm/summary.json |
| Conference clean performance | shufflenetv2_lstm | {'samples': 2737, 'eer': 0.014629948364888179, 'f1': 0.9824109824109823, 'roc_auc': 0.9928828238122558} | BENCHMARKED | outputs/lava_6/clean/shufflenetv2_lstm/summary.json |
| Conference clean performance | mnasnet_lstm | {'samples': 2737, 'eer': 0.030981067125645412, 'f1': 0.9644539614561028, 'roc_auc': 0.9886282545146572} | BENCHMARKED | outputs/lava_5/clean/mnasnet_lstm/summary.json |
| Conference clean performance | efficientnet_b0_lstm | {'samples': 2737, 'eer': 0.04, 'f1': 0.9562554680664916, 'roc_auc': 0.9877021009206896} | BENCHMARKED | outputs/lava_5/clean/efficientnet_b0_lstm/summary.json |
| Conference clean performance | rawnet2 | {'samples': 2737, 'eer': 0.4812698412698413, 'f1': 0.5330188679245284, 'roc_auc': 0.5178370625358577} | BENCHMARKED | outputs/lava_5/clean/rawnet2/summary.json |
| Conference clean performance | aasist | {'samples': 2737, 'eer': 0.44634920634920633, 'f1': 0.5138558986539985, 'roc_auc': 0.5596636887686801} | BENCHMARKED | outputs/lava_5/clean/aasist/summary.json |

All Journal per-condition numerical rows above name or resolve to an executed score CSV in `journal_metrics` in the inventory. Decisive file hashes:

| Evidence path | SHA256 |
|---|---|
| outputs/jaes_v2/phase6/report_environment_v1.json | 1258c2f420bf5ed049cbd2e81104c82b4f4896a392f29b67c52404d4b5dde6a4 |
| outputs/jaes_v2/datasets/asv2019_foundation_v1/foundation.json | 1a47e4b81938d9efacf214e0efcf6b6ff524f9a549a7be8bef3d0d02b309cdfe |
| outputs/lava_5/clean/mobilenetv3_lstm/summary.json | fb6a0605247cbda6b4c1105dcbf954b8cd58264220a7df5f40ec6544434cc024 |
| outputs/lava_6/clean/shufflenetv2_lstm/summary.json | e12ac506bab702d4213e18e1af8842e25e445c08ad6f1ff4a807e45c2a389c42 |
| outputs/lava_5/clean/mnasnet_lstm/summary.json | ffd1d34c9bec084f4e5c26713ae15d686c8931b0e5dcbdca04e9c594ba1b5cfa |
| outputs/lava_5/clean/efficientnet_b0_lstm/summary.json | 230ce99ef5cccf6a56864709e255ea22352f9151d8345251d81df369b5cd0667 |
| outputs/lava_5/clean/rawnet2/summary.json | 12c4790ba2fe8e18f7ac215d4abbf8e10fa5bb9664df9fbf9eaa122f6a097e50 |
| outputs/lava_5/clean/aasist/summary.json | 349fef86c41817aa1c1ea5b7238e0027acb5c59e7601207b2c124b949411ab2e |
| outputs/lava_5/diagnostic_100/clean/aasist/summary.json | 264fc489d25e4de83b9c95651e72722e417eaa7f9a93cc67690e07946be4803b |
| outputs/lava_5/diagnostic_100/clean/efficientnet_b0_lstm/summary.json | e716ecf26595f805d988642a164a3f98b0d8c8e5b69ce2afb07389e078e72dfa |
| outputs/lava_5/diagnostic_100/clean/mnasnet_lstm/summary.json | 9a222165c56bda079313a2c0a12dffb03b05704881349c847ff59fb5ba09b8d5 |
| outputs/lava_5/diagnostic_100/clean/mobilenetv3_lstm/summary.json | f89d767576f4d7e369fac5cdd9622bd9f6da60cea76589722bb978abbabffd38 |
| outputs/lava_5/diagnostic_100/clean/rawnet2/summary.json | 308516a078640878a32e3a59244e43cbaf1fa71c7f02841ff30f4ecade1ebf38 |
| outputs/lava_6/diagnostic_100/clean/shufflenetv2_lstm/summary.json | 1bf952bd44d98544dc3c886c4b0d65878b5739f86bb35ed22699928a18d50e4d |
| outputs/jaes_v2/phase6/partial_finetune_sanity_v1/run_metadata.json | e95571c57207cf333f839fc08e5bfb42a2483038b5a62960dfff48a9236aefd9 |
| outputs/jaes_v2/phase6/partial_finetune_sanity_v2/run_metadata.json | 203e62aa7dbcdeb449cc28029178698a8aa73f459326d9ff3b2b4064bebced7b |
| outputs/jaes_v2/phase6/training/p6_p0_b0_v1/run_metadata.json | 1226ea1d2301cd1fc3719bf53020e1d99a0830d9fa4fd983ddeafe15292b87f6 |
| outputs/jaes_v2/phase6/training/p6_p0_b0_v1/resolved_config.json | 454d11e659ccf1988fa3de2d4cc097115b3d8361728b3c551b9b1c622e78aa48 |
| outputs/jaes_v2/phase6/training/p6_p0_b0_v1/metrics.json | 4e3a32786f72d1eb243b1d9603c9bade497139327c068c507c7c0bf971aec00c |
| outputs/jaes_v2/phase6/training/p6_p0_b0_v1/validation_scores.csv | 1e31c59c6a610f39548833ad4efb4d3d35cd22b8869c521a64290c9e56970e8d |
| outputs/jaes_v2/phase6/training/p6_p0_b1_v1/run_metadata.json | 23e7564e1f6f5107000dae1734e224a3ad121419e901f0173d216051677788ef |
| outputs/jaes_v2/phase6/training/p6_p0_b1_v1/resolved_config.json | 61e191c61970c2da8a6001eaeb61a77b6617aa58d75df354edc50f67d7443878 |
| outputs/jaes_v2/phase6/training/p6_p0_b1_v1/metrics.json | 53863e3fab4eb9b48712818d9ee3daa7308fb32a3c629e120a061f837ed22517 |
| outputs/jaes_v2/phase6/training/p6_p0_b1_v1/validation_scores.csv | 389a93595a17b1b54ff8cb797db1a3a857cd2ebe52e9b9cd8dd4861f5e1523c8 |
| outputs/jaes_v2/phase6/training/p6_p0_b2_v1/run_metadata.json | b1efcaadb67e3db64fcc5a54bcc59c012b2facbd026917af7e3f0d3059149ef8 |
| outputs/jaes_v2/phase6/training/p6_p0_b2_v1/resolved_config.json | 0d500a43fb147d68e6813f69ae65b047ce128b67ee56e2e5697424446e2dd9d1 |
| outputs/jaes_v2/phase6/training/p6_p0_b2_v1/metrics.json | 077ed99809e10a4cea638a6685ab77c073d1233fd15e7a78441e76c6b2bd5d91 |
| outputs/jaes_v2/phase6/training/p6_p0_b2_v1/validation_scores.csv | 14bd435367dfb9fab0e412aad43020bceac49856f20d3beaca78a9714eeb7777 |
| outputs/jaes_v2/phase6/training/p6_p0_b4_control_v1/run_metadata.json | 77211281624d2a19c61e62e5518a75aa0a83d1f31d703e055e8b820faa6d0ceb |
| outputs/jaes_v2/phase6/training/p6_p0_b4_control_v1/resolved_config.json | 978cf7bcf6a9612630538679f90d6b6f6c501e20649d161f472f68244728aba0 |
| outputs/jaes_v2/phase6/training/p6_p0_b4_control_v1/metrics.json | 2c2a3b43dee381273ba6d98e9508cc3bf54791d823d0bbe9286c27e9714ff7ef |
| outputs/jaes_v2/phase6/training/p6_p0_b4_control_v1/validation_scores.csv | 882af37b1310ee8a6e7761dd86fe057bd145bdb54f3469c433561601e9c082d7 |
| outputs/jaes_v2/phase6/training/p6_p0_b4_v1/run_metadata.json | 3ecffcf5d78d6f9c5aadf5003c1818e0d00f9906e6630cdca6b027e6bd28c76c |
| outputs/jaes_v2/phase6/training/p6_p0_b4_v1/resolved_config.json | 436c8f23d8280aadab17c05150603ed4d2cd25f281da86fb111bb7bd6af68b68 |
| outputs/jaes_v2/phase6/training/p6_p0_b4_v1/metrics.json | d9addbe4170d8475df62e32bb84c3b06c8afbd2493e710b0cf504f0bcb75f132 |
| outputs/jaes_v2/phase6/training/p6_p0_b4_v1/validation_scores.csv | 70e49b2d7acc22f93b5c2d9e6a32368f346aec7d500ee0be016ebe2591b80d38 |
| outputs/jaes_v2/phase6/training/p6_p0_rb_v1/run_metadata.json | e1c3f6b0f6692b68d22bc32f9b2b5cb7166431aea2ff961cb570a6a85e66879c |
| outputs/jaes_v2/phase6/training/p6_p0_rb_v1/resolved_config.json | 3d168c2838362621aef06aa7d53673d4b3534f6370f47a73ccf279b5a5b748d8 |
| outputs/jaes_v2/phase6/training/p6_p0_rb_v1/metrics.json | 893d58c299ed6da99efc899c47c84c9a012127ab528a8b442f2f69e8c2b4546a |
| outputs/jaes_v2/phase6/training/p6_p0_rb_v1/validation_scores.csv | a4b6900140df34fe81382af8c917ea7fbf5f7dfa238738b912d49e93fa4fa1d6 |
| outputs/jaes_v2/phase6/training/p6_p1_b0_v1/run_metadata.json | ec8258045a8a9f1695e4f123f0dfafce38a029b54e164c978d861cf39004d8b2 |
| outputs/jaes_v2/phase6/training/p6_p1_b0_v1/resolved_config.json | d58eee5ec63d4c52ce16459250ec87bc84e38c9df39d1b67ede14a885693d5fc |
| outputs/jaes_v2/phase6/training/p6_p1_b0_v1/metrics.json | 332219be1526e9b169b81c823a941924a3cfa720be2cdb32002af686aec19fd9 |
| outputs/jaes_v2/phase6/training/p6_p1_b0_v1/validation_scores.csv | f0960e6b44d8740aa04f8e3fb2f184e33ebf452017032a6badb6d9063f0bce96 |
| outputs/jaes_v2/phase6/training/p6_p1_b1_v1/run_metadata.json | 49e8317fbd68bf3e7351bcaff91fe6d1fd67696de25b94c777b10a87d3091141 |
| outputs/jaes_v2/phase6/training/p6_p1_b1_v1/resolved_config.json | 8fd5ed83517df580097889172645b55f6ca6f5323dff1ff934feed3d1424d414 |
| outputs/jaes_v2/phase6/training/p6_p1_b1_v1/metrics.json | 332219be1526e9b169b81c823a941924a3cfa720be2cdb32002af686aec19fd9 |
| outputs/jaes_v2/phase6/training/p6_p1_b1_v1/validation_scores.csv | f0960e6b44d8740aa04f8e3fb2f184e33ebf452017032a6badb6d9063f0bce96 |
| outputs/jaes_v2/phase6/training/p6_p1_b2_v1/run_metadata.json | 7da4f8353267f7657c98e2ad9b2255eb9a5cdde1191eed9adb1f9577f0191df8 |
| outputs/jaes_v2/phase6/training/p6_p1_b2_v1/resolved_config.json | b2cc05b700d209932f4ef53838a3a04dbd344da25accf019f32abe830e211cab |
| outputs/jaes_v2/phase6/training/p6_p1_b2_v1/metrics.json | 418e2b950d1bdf844ef2dd6f4dca8896fd95f5ddc8cdd3be086d7d91ceb0e956 |
| outputs/jaes_v2/phase6/training/p6_p1_b2_v1/validation_scores.csv | 1e9161a42cd822e8cac7253d39f13719a03ab8b850dbbb3062dd7875afe22bf6 |
| outputs/jaes_v2/phase6/data_selection_v1/selection.json | 6748a0a09804efa4b2b89ea3d85d8420d5a7b622ea36dc0100e448dce0a868f8 |
| outputs/jaes_v2/phase6/reports_v1/summary.json | 1cfc3d124364409f3c388f6afc113021b6cda9221e5daa79e213b0cb71218fb2 |
| outputs/jaes_v2/phase6/scaling_plan_v1.json | 445925642e776813498ae33729e47cbb21cb020464dc67124e783df2126e1b42 |
| outputs/jaes_v2/phase6/resource_preflight_v1.json | 8fb02c5a2123491ffa9597f722e8b47e8adfd300f4dd682d1021d563bbf9d654 |
| outputs/jaes_v2/phase6/protocol_v1.json | 84765b085adb1474dc452a6c831d634602f23a73db9f634290d0b8607c0e090b |
| outputs/jaes_v2/phase6/resource_system_final.json | 4ec3eee3f06fb20f2c0bcdb9749a267de61a53f23cde5cf9979ff04d0a8652bb |
| outputs/jaes_v2/phase6/efficiency_v1.json | 029e243750a1761d2f1473012ce0c8b131fa71a6804cfd57087c9274cecb4159 |
| outputs/jaes_v2/phase6/feature_cache_v2/cache_metadata.json | 7d68a391ead776bd5b4c8ed6955c73ab7bb34f716bce122a1c0e792ba3f25fc1 |
| outputs/jaes_v2/audit/phase3_final_integrity.json | 10d0f0e7a96e953c111881ab46cb8a56d7b1b66855358f111fd07c8ea9ee1c10 |
| outputs/jaes_v2/audit/phase4_final_integrity.json | d8be6641c695c490cc3cafc9953fa709a9f1910a2416d4d88592d18dda5f6627 |
| outputs/jaes_v2/audit/phase5_final_integrity.json | 2e4823962df2b67b69eaa3e20da7c2a4ebfd1cf50b2b4153f31d99fd2b0945cf |

### Historical benchmark diagnostic metrics (source-linked)

| Model | Condition | n | F1 | AUC | EER | REAL FA | FAKE miss | Source |
|---|---|---|---|---|---|---|---|---|
| aasist | clean | 100 | 0.5060240963855421 | 0.6194581280788177 | 0.38095238095238093 | 0.3448275862068966 | 0.5 | outputs/lava_5/diagnostic_100/clean/aasist/summary.json |
| efficientnet_b0_lstm | clean | 100 | 0.962962962962963 | 0.9749589490968801 | 0.04761904761904767 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/clean/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | clean | 100 | 0.975609756097561 | 0.9688013136288998 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_5/diagnostic_100/clean/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | clean | 100 | 0.975609756097561 | 0.9634646962233169 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_5/diagnostic_100/clean/mobilenetv3_lstm/summary.json |
| rawnet2 | clean | 100 | 0.5660377358490566 | 0.6375205254515599 | 0.41379310344827586 | 0.5862068965517241 | 0.2857142857142857 | outputs/lava_5/diagnostic_100/clean/rawnet2/summary.json |
| aasist | aac_96k | 100 | 0.5238095238095238 | 0.6227422003284073 | 0.38095238095238093 | 0.3448275862068966 | 0.47619047619047616 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/aasist/summary.json |
| efficientnet_b0_lstm | aac_96k | 100 | 0.962962962962963 | 0.973727422003284 | 0.04761904761904767 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | aac_96k | 100 | 0.975609756097561 | 0.9692118226600984 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | aac_96k | 100 | 0.962962962962963 | 0.9610016420361248 | 0.04761904761904767 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/mobilenetv3_lstm/summary.json |
| rawnet2 | aac_96k | 100 | 0.5660377358490566 | 0.6395730706075533 | 0.4285714285714286 | 0.5862068965517241 | 0.2857142857142857 | outputs/lava_5/diagnostic_100/robustness/compression/aac_96k/rawnet2/summary.json |
| aasist | mp3_128k | 100 | 0.5060240963855421 | 0.6178160919540231 | 0.38095238095238093 | 0.3448275862068966 | 0.5 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/aasist/summary.json |
| efficientnet_b0_lstm | mp3_128k | 100 | 0.962962962962963 | 0.9716748768472905 | 0.04761904761904767 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | mp3_128k | 100 | 0.9411764705882352 | 0.9638752052545155 | 0.04761904761904767 | 0.05172413793103448 | 0.047619047619047616 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | mp3_128k | 100 | 0.962962962962963 | 0.9610016420361248 | 0.047619047619047665 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/mobilenetv3_lstm/summary.json |
| rawnet2 | mp3_128k | 100 | 0.5555555555555556 | 0.6215106732348112 | 0.4285714285714286 | 0.6206896551724138 | 0.2857142857142857 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_128k/rawnet2/summary.json |
| aasist | mp3_64k | 100 | 0.5060240963855421 | 0.6190476190476191 | 0.38095238095238093 | 0.3448275862068966 | 0.5 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/aasist/summary.json |
| efficientnet_b0_lstm | mp3_64k | 100 | 0.962962962962963 | 0.9712643678160919 | 0.04761904761904767 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | mp3_64k | 100 | 0.9411764705882352 | 0.9638752052545156 | 0.04761904761904767 | 0.05172413793103448 | 0.047619047619047616 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | mp3_64k | 100 | 0.962962962962963 | 0.9642857142857142 | 0.04761904761904767 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/mobilenetv3_lstm/summary.json |
| rawnet2 | mp3_64k | 100 | 0.5688073394495413 | 0.6198686371100164 | 0.4285714285714286 | 0.6206896551724138 | 0.2619047619047619 | outputs/lava_5/diagnostic_100/robustness/compression/mp3_64k/rawnet2/summary.json |
| aasist | opus_64k | 100 | 0.5176470588235295 | 0.6169950738916257 | 0.38095238095238093 | 0.3620689655172414 | 0.47619047619047616 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/aasist/summary.json |
| efficientnet_b0_lstm | opus_64k | 100 | 0.962962962962963 | 0.9741379310344827 | 0.047619047619047665 | 0.0 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | opus_64k | 100 | 0.951219512195122 | 0.9622331691297208 | 0.04761904761904767 | 0.017241379310344827 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | opus_64k | 100 | 0.975609756097561 | 0.9663382594417077 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/mobilenetv3_lstm/summary.json |
| rawnet2 | opus_64k | 100 | 0.5765765765765766 | 0.6317733990147782 | 0.43103448275862066 | 0.6379310344827587 | 0.23809523809523808 | outputs/lava_5/diagnostic_100/robustness/compression/opus_64k/rawnet2/summary.json |
| aasist | snr_0 | 100 | 0.609375 | 0.5734811165845648 | 0.45238095238095233 | 0.8103448275862069 | 0.07142857142857142 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/aasist/summary.json |
| efficientnet_b0_lstm | snr_0 | 100 | 0.0 | 0.6272577996715928 | 0.38095238095238093 | 0.0 | 1.0 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_0 | 100 | 0.0 | 0.539408866995074 | 0.45238095238095233 | 0.017241379310344827 | 1.0 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_0 | 100 | 0.046511627906976744 | 0.6161740558292282 | 0.40476190476190477 | 0.0 | 0.9761904761904762 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_0 | 100 | 0.0 | 0.5270935960591133 | 0.47619047619047616 | 0.0 | 1.0 | outputs/lava_5/diagnostic_100/robustness/noise/snr_0/rawnet2/summary.json |
| aasist | snr_10 | 100 | 0.5742574257425742 | 0.5328407224958949 | 0.43103448275862066 | 0.5172413793103449 | 0.30952380952380953 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/aasist/summary.json |
| efficientnet_b0_lstm | snr_10 | 100 | 0.0909090909090909 | 0.6834975369458128 | 0.3571428571428571 | 0.0 | 0.9523809523809523 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_10 | 100 | 0.5679012345679013 | 0.6658456486042693 | 0.38095238095238093 | 0.27586206896551724 | 0.4523809523809524 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_10 | 100 | 0.2909090909090909 | 0.7192118226600985 | 0.3793103448275862 | 0.08620689655172414 | 0.8095238095238095 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_10 | 100 | 0.0 | 0.3768472906403941 | 0.5952380952380952 | 0.017241379310344827 | 1.0 | outputs/lava_5/diagnostic_100/robustness/noise/snr_10/rawnet2/summary.json |
| aasist | snr_20 | 100 | 0.49438202247191015 | 0.5841543513957308 | 0.4482758620689655 | 0.43103448275862066 | 0.47619047619047616 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/aasist/summary.json |
| efficientnet_b0_lstm | snr_20 | 100 | 0.5396825396825397 | 0.7274220032840724 | 0.2857142857142857 | 0.06896551724137931 | 0.5952380952380952 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_20 | 100 | 0.7961165048543689 | 0.9092775041050903 | 0.15517241379310345 | 0.3448275862068966 | 0.023809523809523808 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_20 | 100 | 0.7058823529411765 | 0.8341543513957307 | 0.2413793103448276 | 0.034482758620689655 | 0.42857142857142855 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_20 | 100 | 0.2333333333333333 | 0.5311986863711001 | 0.5 | 0.1896551724137931 | 0.8333333333333334 | outputs/lava_5/diagnostic_100/robustness/noise/snr_20/rawnet2/summary.json |
| aasist | snr_5 | 100 | 0.6101694915254237 | 0.5443349753694582 | 0.47619047619047616 | 0.6896551724137931 | 0.14285714285714285 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/aasist/summary.json |
| efficientnet_b0_lstm | snr_5 | 100 | 0.0 | 0.6572249589490968 | 0.38095238095238093 | 0.0 | 1.0 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | snr_5 | 100 | 0.0909090909090909 | 0.5295566502463054 | 0.5 | 0.0 | 0.9523809523809523 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | snr_5 | 100 | 0.1702127659574468 | 0.6494252873563219 | 0.4285714285714286 | 0.017241379310344827 | 0.9047619047619048 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/mobilenetv3_lstm/summary.json |
| rawnet2 | snr_5 | 100 | 0.0 | 0.42898193760262726 | 0.5862068965517241 | 0.0 | 1.0 | outputs/lava_5/diagnostic_100/robustness/noise/snr_5/rawnet2/summary.json |
| aasist | synthetic_channel | 100 | 0.5742574257425742 | 0.6309523809523809 | 0.4285714285714286 | 0.5172413793103449 | 0.30952380952380953 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/aasist/summary.json |
| efficientnet_b0_lstm | synthetic_channel | 100 | 0.898876404494382 | 0.9696223316912972 | 0.0714285714285714 | 0.1206896551724138 | 0.047619047619047616 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/efficientnet_b0_lstm/summary.json |
| mnasnet_lstm | synthetic_channel | 100 | 0.8266666666666667 | 0.94376026272578 | 0.1428571428571429 | 0.034482758620689655 | 0.2619047619047619 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/mnasnet_lstm/summary.json |
| mobilenetv3_lstm | synthetic_channel | 100 | 0.9135802469135802 | 0.9651067323481117 | 0.09523809523809523 | 0.034482758620689655 | 0.11904761904761904 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/mobilenetv3_lstm/summary.json |
| rawnet2 | synthetic_channel | 100 | 0.5769230769230769 | 0.557471264367816 | 0.43103448275862066 | 0.5517241379310345 | 0.2857142857142857 | outputs/lava_5/diagnostic_100/robustness/replay/synthetic_channel/rawnet2/summary.json |
| shufflenetv2_lstm | clean | 100 | 0.975609756097561 | 0.9642857142857142 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_6/diagnostic_100/clean/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | aac_96k | 100 | 0.962962962962963 | 0.9638752052545156 | 0.04761904761904767 | 0.0 | 0.07142857142857142 | outputs/lava_6/diagnostic_100/robustness/compression/aac_96k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | mp3_128k | 100 | 0.975609756097561 | 0.9630541871921182 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_6/diagnostic_100/robustness/compression/mp3_128k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | mp3_64k | 100 | 0.975609756097561 | 0.964696223316913 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_6/diagnostic_100/robustness/compression/mp3_64k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | opus_64k | 100 | 0.975609756097561 | 0.9630541871921182 | 0.04761904761904767 | 0.0 | 0.047619047619047616 | outputs/lava_6/diagnostic_100/robustness/compression/opus_64k/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_0 | 100 | 0.2745098039215686 | 0.6966338259441708 | 0.3448275862068966 | 0.034482758620689655 | 0.8333333333333334 | outputs/lava_6/diagnostic_100/robustness/noise/snr_0/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_10 | 100 | 0.6923076923076924 | 0.8555008210180625 | 0.25862068965517243 | 0.15517241379310345 | 0.35714285714285715 | outputs/lava_6/diagnostic_100/robustness/noise/snr_10/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_20 | 100 | 0.9135802469135802 | 0.9507389162561576 | 0.09523809523809523 | 0.034482758620689655 | 0.11904761904761904 | outputs/lava_6/diagnostic_100/robustness/noise/snr_20/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | snr_5 | 100 | 0.6 | 0.7549261083743843 | 0.3103448275862069 | 0.1206896551724138 | 0.5 | outputs/lava_6/diagnostic_100/robustness/noise/snr_5/shufflenetv2_lstm/summary.json |
| shufflenetv2_lstm | synthetic_channel | 100 | 0.9500000000000001 | 0.9733169129720853 | 0.0714285714285714 | 0.0 | 0.09523809523809523 | outputs/lava_6/diagnostic_100/robustness/replay/synthetic_channel/shufflenetv2_lstm/summary.json |

## 19. Remaining Scientific Blockers

| Category | Missing | Why / resolution | User action |
|---|---|---|---|
| DATA | Small balanced subset; limited attacks/speakers; no unused external corpus | Scale controlled cohorts and freeze external role before scores | No immediate download required |
| METHOD | No validated forensic evidence-retention estimator or distinct objective | Compare stronger/equal-view controls; falsify proxy and prior-art distinction | None yet |
| COMPUTE | Production runtimes CPU-only; desktop commit pressure | Use sequential jobs; budget compatible isolated GPU full fine-tuning | Provide validated compute only for next large study |
| VALIDATION | No independent confirmation / operating point certification | Lock unused external/calibration protocol before use | Choose publication/evaluation scope |
| STATISTICS | One initialization seed | At least three independently trained seeds; preserve speaker clustering | Approve measured larger compute budget later |
| PROVENANCE | Original utterance / exact generators UNKNOWN; remote archive digest not certified | Acquire/verify upstream identity and publisher digest | Upstream metadata if available |
| JOURNAL ELIGIBILITY | Conference venue/DOI/copyright and current Clarivate record not verified here | Check AES extension rules, disclose/cite accepted version, verify official MJL | Supply venue, DOI/publication date and copyright terms |

## 20. International Journal Readiness

NOT_JOURNAL_READY as a new methodological paper. The project has stronger software traceability and controlled baseline evidence, but tests, more perturbations or a working gradient path cannot establish international novelty. A suitable submission needs a defensible technical distinction or an appropriately scoped empirical audio-engineering contribution, strong matched controls, external validity, ablations, seed variability and clear operational trade-offs. The AES official journal site and author guidelines govern scope and publication policy. Current Clarivate indexing/JCR rank and the exact conference extension/copyright eligibility are not independently certified by this code audit. The accepted conference venue, DOI/publication date, transferred rights and all-author extension/disclosure status remain required metadata. An accepted-but-unreleased paper cannot automatically be treated as having no prior-publication obligations. No submission or publication action is performed.

## 21. Phase7 Research Roadmap

| Priority / objective | Prerequisite | Code/data work | Compute basis | Evidence / acceptance gate |
|---|---|---|---|---|
| 1 All-attack controlled scaling | Source/attack support audited; larger budget locked | Increase per-speaker cell counts; preserve train/dev/calibration roles | Actual small-cache/run throughput; scaling estimates above | Both classes/speakers balanced; all declared attacks; no hidden overlap |
| 2 Strong online vs finite augmentation | Faithful finite adapter already tested | Online waveform/spectrogram data path; strong RawBoost and equal-view controls | Measure preprocessing and backbone training; no cached-embedding finetune claims | Matched source exposure and disclosed compute |
| 3 Partial-backbone learning study | Tiny gradient/resource sanity must pass; compatible isolated runtime | At least three seeds, matched initial state/stage schedule, locked BN policy | Extrapolate real microbatch forward/backward; budget before training | Representation updates, clean-risk tolerance and ablations |
| 4 Retention mechanism or rejection | Compare F-SAT/PCL-NET/DK-CAST and validate observable proxy | Specify inputs/estimator/degeneracy guard and falsification | No new-loss experiment until plausibility gate passes | Benefit beyond equal-view standard controls; negative results retained |
| 5 Unused external validation and calibration | Freeze corpus role/overlap/primary tests before labels | Choose storage-safe external source; lock calibration separately | Recheck reserve, expanded archive and inference cost | No design use of final confirmation; classwise/seed/group uncertainty |
| 6 Journal extension eligibility | Actual conference/publication/copyright facts | Disclosure, citations, substantive contribution mapping | No compute needed | Publisher rules and author rights verified; no promised acceptance |

## 22. Exact Next Commands

### Executed stages

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment select
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment preflight
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment cache
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment train --config configs/jaes_v2/p6_p0_b4_v1.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_experiment finetune --output-name partial_finetune_sanity_v2
```

Existing output identities refuse overwrite. The full registered run list is in the experiment table; these sample stage commands do not authorize rerunning completed training.

### Safe score-only regeneration, fresh destination

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase6_report --output outputs/jaes_v2/phase6/reports_another_reproduction
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.build_master_research_report --audit-output outputs/jaes_v2/phase6/master_audit_another
```

### External runtime preparation / requires compatible GPU

```powershell
python -m scripts.jaes_v2.phase6_experiment finetune --device gpu --output-name partial_finetune_gpu_sanity_v1
```

The GPU command requires an independently provisioned TensorFlow GPU environment, pinned dependencies/data/weights and the same repository contracts. It fails if no GPU is visible; it is a bounded sanity command, not an implemented full-scale GPU training pipeline. Longer fine-tuning and seed schedules require new validated configurations and budgets. No unsupported command-line flag is proposed.

### Evidence gaps

No required structured evidence file is missing from this generated handoff. UNKNOWN provenance and NOT_MEASURED scientific claims remain explicitly unresolved.

