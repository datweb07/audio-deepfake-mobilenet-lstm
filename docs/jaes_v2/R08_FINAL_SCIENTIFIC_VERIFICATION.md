# R08 final scientific verification

**14 November 2026. CROSS_DATASET_DIAGNOSTIC_WITH_LINEAGE_LIMITATIONS.** The author-run R08 experiment is complete. This audit reads existing artifacts and performs no training, model inference, calibration or threshold tuning.

## Corpus and protocol

In-the-Wild (Müller et al., Interspeech 2022), release revision `eee168f92c367f8c82ff2cf42b6f61e362fd6211`. The recorded distributor clarification applies CC BY-SA 4.0 to WAV and metadata, permits local evaluation and aggregate reporting, requests citation and keeps source audio local. The existing access and scoring approvals validate against the unchanged files.

Deterministic selection used seed 20261112 and the primary quota. There are 512 distinct recordings, 256 REAL and 256 FAKE across 16 speaker groups, with 16 recordings per class per group. Known exclusions cover manifest recording IDs, canonical or supplied speaker identifiers and exact audio bytes. Byte and decoded-audio duplication within the selected cohort were checked in the acquisition/lock path. The protocol checks all selected WAV byte hashes. Parent videos, original utterances, related clips and near-duplicates remain incompletely traceable.

One native P0 view per recording was scored by each of six frozen MobileNetV3Small–LSTM128 models trained on 640 recordings. Outputs are FAKE-oriented scores with FAKE predicted at score ≥ 0.5. External results are separate from the 640 development, 1,280 development size comparison and original within-release held-out grids.

| Model | Selected checkpoint | Model SHA-256 |
| --- | --- | --- |
| erm_9107 | `u000560_e014_c000` | `0c787ec62ca715409439912e9bf387bd7dfed9b105c6716ee732445a1cbf2f78` |
| channel_9107 | `u000760_e019_c000` | `dd3e436aad035ab0a841dfc675d3230b77cdd5fe745d766561e3511286999b7d` |
| erm_9207 | `u000520_e013_c000` | `bf9c2c8b04a61521089f608968baec0c10cd91f7de4d938bd351ee70de46df24` |
| channel_9207 | `u000720_e018_c000` | `f6485f4de336f1c0a31c4f29f43f53fb99ff724b8126c324f23431b3d422f579` |
| erm_9307 | `u000360_e009_c000` | `72a05a07885a94355b8c809dc27cb2fe3fcb15ef41a0661800d2130f7133d5d0` |
| channel_9307 | `u000680_e017_c000` | `c079f7f5569c1eaf5b5c2ee40ff8e6aa7f21b19997a708946175239e659c3b4f` |

## Artifact and orientation checks

Selected rows SHA-256: `51cd5643dbf6df7c6f6cfe2d3dfa8e8ad0371fa0d5d6d900687adeb538ed3e38`.

- `configs/jaes_v2/r08_external_v1/protocol.json` → `f26fd9ae78eda7cd153a2d7973927b9de0465c206a04a1b9d016f3595db24423`.
- `outputs/jaes_v2/r08_external_v1/author_scoring_approval.json` → `f5d4a03e9e3b2ee8d46b5f337c2d8d0f1a18ddd29217c16227c575804479fe63`.
- `outputs/jaes_v2/r08_external_v1/author_data_access_approval.json` → `736f6aad6a273e944ffaed69c3b52a5a5b1f30e3f1a92864fa42d20efd5886de`.
- `outputs/jaes_v2/r08_external_v1/subset.json` → `2a123a4db4c8a1e91f4367f1687eee5ce671a36fb78862564736beaea307f82f`.
- `outputs/jaes_v2/r08_external_v1/run/seal.json` → `8c9cd7bab5f4d7ad24ef3bd73888a981b9f2bf38f8030186f55be9d7a58093ec`.
- `outputs/jaes_v2/r08_external_v1/run/scores.csv` → `e735b4ad4348a0a4871b0a4d5b631aaa56e99f90ca3cd6f464e256c20c037e7e`.
- `outputs/jaes_v2/r08_external_v1/run/metrics.json` → `1661af7760d337525e35a4298b231aad86212ce08cb6923e75ca816799a1042e`.

The seal lists 3,074 files (3,072 individual predictions, scores.csv and metrics.json). All entries were enumerated and independently hashed. The CSV contains exactly one row per locked recording and model, with matching shard content, protocol and checkpoint IDs, label, audio hash, native condition and P0. All scores are finite and within [0,1]. The checkpoint exports and associated seals match the lock.

The frozen parser and label fixtures map bonafide and bona-fide to REAL (0), and spoof to FAKE (1). The pinned ZIP metadata was hash-checked when the cohort was selected. The audit reuses those qualified rows and tests the archived parser mapping. FAKE is the positive ROC class and the decision rule reproduces every saved label. The 18 archived development replay anchors cover both labels across six models and match the frozen score path exactly. The model head is sigmoid with the same FAKE-oriented labels as the replay. The external path uses the registered P0 settings. These checks detected no data-label or software-orientation error.

## Independent recalculation from scores.csv

EER uses the empirical ROC point with minimum absolute FPR–FNR difference, averaging the two rates at that point. FPR uses 256 REAL, FNR uses 256 FAKE. The original score direction and threshold were retained.

| Model | AUC | EER | FP/256 (FPR) | FN/256 (FNR) | TN/FP/FN/TP | Median score REAL | Median score FAKE |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| erm_9107 | 0.3061 | 0.6523 | 234/256 (91.41%) | 94/256 (36.72%) | 22/234/94/162 | 0.99674 | 0.96467 |
| channel_9107 | 0.4997 | 0.5098 | 200/256 (78.12%) | 73/256 (28.52%) | 56/200/73/183 | 0.94130 | 0.93083 |
| erm_9207 | 0.3370 | 0.6289 | 238/256 (92.97%) | 66/256 (25.78%) | 18/238/66/190 | 0.99759 | 0.97432 |
| channel_9207 | 0.4364 | 0.5566 | 221/256 (86.33%) | 67/256 (26.17%) | 35/221/67/189 | 0.97311 | 0.93823 |
| erm_9307 | 0.3742 | 0.5996 | 230/256 (89.84%) | 84/256 (32.81%) | 26/230/84/172 | 0.98034 | 0.89944 |
| channel_9307 | 0.4635 | 0.5234 | 223/256 (87.11%) | 63/256 (24.61%) | 33/223/63/193 | 0.97227 | 0.95692 |

| Seed | ΔAUC | ΔFPR | ΔFNR | REAL recovered / introduced / shared error / shared correct | FAKE recovered / introduced / shared error / shared correct |
| --- | ---: | ---: | ---: | --- | --- |
| 9107 | +0.1936 | -0.1328 | -0.0820 | 36 / 2 / 198 / 20 | 27 / 6 / 67 / 156 |
| 9207 | +0.0994 | -0.0664 | +0.0039 | 22 / 5 / 216 / 13 | 12 / 13 / 54 / 177 |
| 9307 | +0.0892 | -0.0273 | -0.0820 | 11 / 4 / 219 / 22 | 24 / 3 / 60 / 169 |

| Paired endpoint | Mean | Sample SD | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| auc | +0.1274 | 0.0575 | +0.0892 | +0.1936 |
| fpr | -0.0755 | 0.0533 | -0.1328 | -0.0273 |
| fnr | -0.0534 | 0.0496 | -0.0820 | +0.0039 |

The six AUC point estimates are below 0.5. This is an unfavorable observed ranking, with no population-level significance inference. Channel gains relative to ERM occur in all three seeds and REAL false alarms fall, but the absolute FPR stays high (78.12% to 92.97%). FAKE misses improve in 9107 and 9307. In 9207 they rise from 66 to 67. The three paired contrasts are descriptive and share the same external cohort. No publication bootstrap intervals were calculated. The comparison with higher clean ASVspoof ranking is descriptive because the populations, recording conditions and attack provenance differ. Native external audio cannot replicate the registered noise/Opus order intervention.

## Historical evidence

Historical integrity passed for all eight completed training seals and 1,088 within-release shards corresponding to 17,408 original predictions. The 405 previously protected hashes, including scientific source and FIX01, match. All three FIX02 baseline artifacts match their recorded hashes. No completed experiment or author approval was rewritten.

Reproduce the sealed-result and historical checks with the existing runner modes `verify` and `integrity` from the repository root. The numerical audit was calculated separately from the CSV and compared to the archived metrics. The full aggregate numerical verification JSON is in `outputs/jaes_v2/supervisor_final_fix03/verification.json`.

**Evidence decision:** Suitable for manuscript integration as a bounded cross-dataset diagnostic, with parent-source independence and broader generalization still unresolved. [Interspeech reference](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.html) · [Existing R08 report](R08_CROSS_DATASET_EVALUATION_RESULTS.md)
