# Channel Order and Classwise Failure in Lightweight Synthetic Speech Detection

**Working research manuscript · 8 November 2026**

## Abstract

Channel augmentation is commonly evaluated with a single aggregate score. We examine how such an aggregate can conceal reversals across listener-relevant errors and synthesis attacks. Two MobileNetV3Small–LSTM detectors were trained on the same 640-source ASVspoof 2019 LA development-research selection and scored on a 368-source speaker-disjoint development cohort. Under noise and both orders of noise and codec processing, stress ROC-AUC averaged over conditions rose from 0.621 to 0.808 in the channel-augmented run. The same run incurred 30/144 clean false alarms in a REAL-only stratum versus 19/144 for clean ERM. Stress misses increased from 40/54 to 47/54 for A05 and from 19/54 to 37/54 for A06. These are observations from one seed and the training arithmetic changed after interruptions in the channel run. The results motivate preregistered groupwise checkpoint guards and a paired study of how generator traces survive channel processing.

## 1. Introduction

A detector for synthetic and converted speech may be applied after transmission, noise mixing, or codec compression. Transforms can change both class separation and the decision score at a fixed operating point. Real speech can be rejected more often even when average stressed discrimination improves. We investigate what classwise and attack-specific failures are visible when noise and codec operations are ordered and the same sources are scored for each transform. The immediate research question concerns development diagnostics in a compact detector.

The development evidence in this draft consists of a speaker-stratified ASVspoof protocol, selected-model diagnostics with exact support, and an engineering audit of arithmetic and validation costs. Controlled attribution of an augmentation effect, a novel training objective, and calibrated adaptive inference remain open.

For source x, transform T and detector score s(Tx), define the false-alarm indicator as 1[s(Tx) ≥ 0.5] for REAL and the miss indicator as 1[s(Tx) < 0.5] for FAKE. For each group and transform we estimate empirical risk by averaging the relevant indicator over its supported sources. Composition may alter s(Tx) differently for the two classes. We ask whether the two selected detectors differ in their score ranking and in their empirical risks under the same source and transform indices.

## 2. Related work

ASVspoof 2019 separates logical-access synthesis and voice conversion from physical replay [1]. ASVspoof 2021 adds transmission and compression artifacts [2]. RawBoost injects waveform perturbations during anti-spoofing training [3]. F-SAT applies frequency-targeted adversarial training [4], while PCL-NET and DK-CAST use paired codec representations and teacher signals [5,6]. SONAR combines speech and high-frequency information [7]. Group-DRO optimizes risk over predefined groups [8]. These methods motivate controls for the question studied here. Our diagnostics resolve order, operating-point risk and attack-specific failure within the same source cohort.

## 3. Data and protocol

The registered medium training manifest contains 320 bona fide and 320 spoof sources across 20 training speakers. Six attack IDs, A01–A06, are represented. Development stratum A includes 112 REAL and 112 FAKE sources across seven speakers. Development stratum B comprises 144 REAL sources from nine other speakers. Source bytes, manifest fingerprints and known speaker separation were checked. The designation of stratum A reflects support for both classes per speaker. Global parallel real–synthetic lineage is unresolved. The same sources are reused across transforms, so denominators count views and those views are dependent. The reserved calibration and official evaluation partitions are outside the analysis.

Seven development views were scored per selected model: clean, half gain, quarter gain, noise at 5 dB, Opus at 64 kb/s and two composed orders of noise and codec. The primary stress average contains noise and both composed orders, excluding single-codec and gain views. AUC and EER are condition averages over A, with 112 REAL and 112 FAKE per condition. B and individual attack subsets are single-class. Their reported errors use observed class counts.

## 4. Model and selection

The model consumes fixed-length Mel segment sequences using an ImageNet-initialized MobileNetV3Small and an LSTM with 128 units. Both medium runs used a head warmup, partial backbone fine-tuning with frozen BN, nominal batch 16, microbatch one and Adam, with 40 optimizer updates per epoch. Clean ERM uses the clean source view. The comparison samples registered channel views. Both selected by clean A binary cross-entropy and use the fixed rule p(FAKE) ≥ 0.5 for class errors. The historical trainer changed arithmetic in the augmented run after memory failures, from standard to segmentwise VJP at step 426 and to pointwise MatMul at 430. A new replication must start both methods under one versioned engine.

## 5. Results

Both runs completed 21 epochs and 840 updates. The best checkpoints were at epochs 17 (clean development BCE 0.12535) and 16 (0.16297), respectively. Six backbone tensors changed in each selected export and no BN tensors changed. Each model produced 2,576 final scores.

| Metric | Clean ERM | Channel augmentation |
| --- | ---: | ---: |
| Clean A false alarms / misses | 4/112, 8/112 | 4/112, 8/112 |
| Clean A ROC-AUC / EER | 0.98868 / 5.36% | 0.97911 / 5.36% |
| Stress A false alarms / misses | 137/336, 162/336 | 72/336, 128/336 |
| Stress A mean ROC-AUC / EER | 0.62112 / 41.82% | 0.80841 / 26.79% |
| Clean B false alarms | 19/144 | 30/144 |
| Stress B false alarms | 122/432 | 93/432 |
| Stress A01 / A02 misses | 17/57, 35/57 | 1/57, 35/57 |
| Stress A03 / A04 misses | 34/57, 17/57 | 7/57, 1/57 |
| Stress A05 / A06 misses | 40/54, 19/54 | 47/54, 37/54 |

On A, the AUC change under stress indicates altered ranking within stress conditions. Channel processing increased A-then-B misses from 13/112 to 29/112 while reducing false alarms from 94/112 to 38/112. Stress performance therefore depends on error type and transform order. The largest clean B speaker error in the channel run was 11/16. These support counts are small. The single observed seed provides no estimate of training-seed variability.

## 6. Discussion and limitations

The aggregate improvements coexist with worse errors for clean B, A05 and A06. A guard tied to the matched ERM and evaluated at a fixed score threshold can expose that trade-off before a checkpoint is selected for deployment-oriented study. A plausible mechanism is the loss or relocation of synthesis traces after sequential transforms. Testing this requires explicit source–generator pairs and matched nuisance controls. ASVspoof attack identifiers alone cannot identify the mechanism.

The results depend on repeated development use, synthetic channel interventions, attack support and different historical numerical paths. The final-invocation timings include restart and validation and do not establish lifetime training cost or deployment latency. Historical conference efficiency results apply to a separate workload and cohort. Real-world, cross-generator, calibrated, and selective-risk claims require independent evidence.

## Societal impact

False alarms can cause authentic speech to be rejected or treated as suspicious. Misses can allow synthetic or converted speech to pass an integrity screen. The reported development risks illustrate why evaluation should consider both errors and report whom the evaluation sampled. The scores are suitable for research diagnostics. Any decision affecting a person would require independent validation on the actual operating population and a review process.

## 7. Conclusion

In the observed development runs, stress ranking improved while clean REAL and selected attack-specific failure rates worsened. Per-class and per-attack evidence at the fixed threshold distinguishes the errors that an aggregate ranking metric cannot reveal. The next experiment is a controlled replication with prespecified per-group guards and identical numerical paths.

## References and reproducibility

1. [ASVspoof 2019 LA official release](https://datashare.ed.ac.uk/handle/10283/3336)
2. [ASVspoof 2021 overview and metadata](https://www.asvspoof.org/index2021.html)
3. [RawBoost](https://arxiv.org/abs/2111.04433)
4. [F-SAT](https://openreview.net/forum?id=2GcR9bO620)
5. [PCL-NET preprint](https://arxiv.org/abs/2610.07216)
6. [DK-CAST](https://link.springer.com/article/10.1007/s10791-025-09746-4)
7. [SONAR](https://proceedings.mlr.press/v306/hidekel26a.html)
8. [Group-DRO](https://arxiv.org/abs/1911.08731)

Run contracts, checkpoint selection and per-source score evidence are indexed in the Phase 9 inventory. See the prior research report for exact Phase 7 and historical Phase 8 boundaries. [Inventory](../../../docs/jaes_v2/LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE9.json) · [Medium review](../../../docs/jaes_v2/PHASE8_MEDIUM_TRAINING_SCIENTIFIC_REVIEW.md)

## Internal conference extension map

| Accepted PaperID 804 evidence | Journal draft extension |
| --- | --- |
| Six heterogeneous benchmark artifacts on an internally assembled collection with SHA-256 disjointness | Two training methods on an ASVspoof 2019 LA speaker-disjoint development selection with attack IDs and provenance auditing |
| One 100-item diagnostic covering noise, codecs and simulated replay | Repeated same-source views with composition order, classwise counts, group failures and explicit B REAL-only stratum |
| Exploratory desktop timing and Pareto analysis | Resource and gradient arithmetic audit with conditional path toward reproducible controlled replication |

The title differs from the accepted LAVA conference paper. Before submission, document venue, publication and copyright status, disclose the relationship to the editor, and cite the proceedings when an appropriate record exists. Review overlap and any rights needed to reuse content. Author-provided venue and rights details are pending. [AES author guidelines](https://aes.org/publications/journal-author-guidelines/) · [AES journal template and extension policy](https://www.overleaf.com/latex/templates/journal-of-the-audio-engineering-society-jaes-template/rqsctzghzcwx)
