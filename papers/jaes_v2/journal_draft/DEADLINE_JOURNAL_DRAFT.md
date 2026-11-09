# Order-Dependent Errors in Lightweight Speech Deepfake Detection

Working journal manuscript, 9 November 2026, Asia/Ho_Chi_Minh

## Abstract

We evaluate how ordinary channel augmentation changes ranking and operating-point errors in a lightweight audio deepfake detector. We trained paired MobileNetV3Small–LSTM models on 640 ASVspoof 2019 LA recordings with matched initialization, source order, numerical backend and checkpoint selection. On 368 speaker-disjoint development recordings, mean ROC-AUC over noise and two ordered channel chains increased from 0.6715 to 0.8026 on the paired-class stratum. At a fixed score threshold of 0.5, clean false alarms on a REAL-only stratum increased from 19/144 to 31/144 and A06 stressed misses increased from 41/54 to 45/54. The result identifies a trade-off within one seed and one development cohort.

## 1 Introduction and research gap

A detector may receive speech after both noise insertion and lossy coding. Aggregating those conditions can obscure operating-point errors and the difference between applying the same operators in opposite orders. This study asks which source-linked errors remain visible under a fixed decision rule when augmentation and baseline training share the numerical and data contract.

## 2 Related work

RawBoost adds waveform corruptions. F-SAT learns against frequency-selective adversarial perturbations. PCL-NET and TFCL align paired codec or acoustic front-end views, while DK-CAST uses clean-to-degraded distillation and codec-aware weighting. SONAR couples low- and high-frequency representations with class-asymmetric alignment. RADAR provides transformation-robustness development and blind-evaluation conditions. Proteus searches ordered transform chains and reports strong bona fide vulnerability. FP-NAC isolates class-asymmetric coding failures and adapts a neural codec with detector guidance. RealComm uses paired calls and staged acquisition-to-transmission augmentation. Our comparison measures a specific conventional Opus/noise order contrast under a matched CPU training intervention. [Proteus](https://arxiv.org/html/2606.29544) · [FP-NAC](https://arxiv.org/html/2609.39651) · [RealComm](https://arxiv.org/pdf/2610.27459) [RawBoost](https://arxiv.org/html/2111.04433) · [F-SAT](https://www.cs.columbia.edu/~junfeng/papers/fsat-iclr25.pdf) · [PCL-NET](https://arxiv.org/html/2610.07216v1) · [TFCL](https://arxiv.org/html/2607.17761v2) · [DK-CAST](https://link.springer.com/article/10.1007/s10791-025-09746-4) · [SONAR](https://proceedings.mlr.press/v306/hidekel26a.html) · [RADAR](https://arxiv.org/html/2605.09568v3)

## 3 Formulation and controlled setup

For source x with class y and transform T, score p(Tx) is classified as FAKE at p ≥ 0.5. The same source supplies two ordered channel views. Record which of the two alone causes a decision error and retain both errors when computing per-condition risks.

The two arms minimize binary cross entropy. ERM trains on clean speech. Channel training samples one view per source from clean, half-gain, noise10 and Opus24. Both use P0 Mel sequences, an ImageNet-initialized MobileNetV3Small and LSTM128, one head warm-up epoch and partial fine-tuning with frozen BN. The CPU execution contract uses deterministic initialization and full-shape recompute. Both runs complete 21 epochs and 840 updates, with clean A BCE checkpoint selection.

## 4 Data, metrics and statistics

ASVspoof 2019 LA medium contains 320 REAL and 320 FAKE training sources and attacks A01–A06. Development contains 224 A sources, 112 per class, and 144 REAL-only B sources. The two strata cover seven and nine development speakers respectively. Train and development speaker IDs are disjoint. The audit checked manifest and source fingerprints. Same-record channel views do not establish shared generator lineage. [Dataset release](https://datashare.ed.ac.uk/handle/10283/3336)

Both selected models score clean, two gain controls, noise5, Opus64 and two ordered noise/codec chains, yielding 2,576 scores each. The chains share a fixed absolute noise plan with 10 dB reference SNR and Opus24. The standalone noise and codec conditions use different settings. Stress macro AUC averages noise5 and the ordered chains on A. The corresponding error counts use 336 views per class across 112 sources. Report class errors at 0.5 and ROC-AUC/EER only where both classes are supported. Multiple views per source are dependent. The present table gives descriptive results from one seed. Paired speaker-cluster bootstrap resamples the seven A speakers and nine B speakers separately, retaining each source and all associated views. We use 10,000 draws and seed 91109. These descriptive intervals condition on the fitted models. Seed-to-seed uncertainty remains unmeasured.

## 5 Results

| Development measure | ERM | Channel |
| --- | ---: | ---: |
| Selected checkpoint epoch | 14 | 19 |
| A clean ROC-AUC / EER | 0.9872 / 4.46% | 0.9814 / 4.91% |
| A clean REAL false alarms / FAKE misses | 3/112 / 6/112 | 3/112 / 9/112 |
| A stress macro ROC-AUC | 0.6715 | 0.8026 |
| A stress REAL false alarms / FAKE misses | 54/336 / 222/336 | 45/336 / 155/336 |
| B clean REAL false alarms | 19/144 | 31/144 |
| A05 stress FAKE misses | 48/54 | 50/54 |
| A06 stress FAKE misses | 41/54 | 45/54 |

Among A REAL, the noise→codec-only false alarm count was 43/112 for ERM and 19/112 for channel, with zero reverse-only alarms. Among A FAKE, codec→noise alone caused 48/112 and 17/112 misses, respectively. The noise→codec-only miss counts were one and zero. Across the selected model pair, clean A misses, clean B false alarms and A06 stressed misses exceeded the registered relative guard margins. The guard was applied to the selected checkpoints as a separate analysis.

The paired speaker-bootstrap 95% interval for the stress macro AUC difference was +0.02 to +0.24 AUC units. The intervals were -25.31 to -14.58 percentage points for A stress FAKE misses and +1.39 to +17.36 percentage points for B clean false alarms. A05 and A06 each contributed 18 original sources and 54 repeated stress views. The counts of recordings missed under all three stress views were 14→16 for A05 and 6→12 for A06.

[Class and attack errors](../../../outputs/jaes_v2/phase10/final_diagnostic_v1/class_and_attack.png) · [Order and speaker diagnostics](../../../outputs/jaes_v2/phase10/final_diagnostic_v1/order_and_speaker.png)

### Matched component diagnostic

We scored Noise10 and Opus24 alone on all 368 development sources with both frozen models. The 1,472 new predictions were combined by source identity with the sealed clean and composed predictions. The component waveforms reconstructed both registered orders exactly. A three-source replay checked clean and composed scores for both models. The signed order contrast is p(noise→codec) minus p(codec→noise). The excess composition contrast is the mean composed score minus both component scores plus the clean score. The second quantity describes departure from score additivity and can arise from nonlinear coding or detector response.

For ERM, order-score AUC for class was 0.515 before and 0.538 after a leave-one-speaker-out nuisance regression on development. For CHANNEL, order-score AUC for class was 0.396 before and 0.432 after a leave-one-speaker-out nuisance regression on development. This descriptive analysis uses selected models on the same development cohort. It cannot identify an independent forensic cue or support a new training objective.

## 6 Ablations and comparison scope

The existing controlled factor is training augmentation. The gain and ordered conditions are evaluation diagnostics. An ablation of a proposed learning mechanism will require an independent target and equal-view controls. Cross-paper metric comparisons require matched corpus and operating policy. The conference benchmark used six heterogeneous artifacts on a different collection and a fixed stress diagnostic.

## 7 Efficiency

CPU logs record about 42.6 minutes of updates and 40.2 minutes of validation for ERM, and 82.9 and 39.1 minutes for channel. These measurements sum logged invocations and can include replayed work after interruption. They do not establish edge latency or full-foundation cost.

## 8 Discussion and limitations

Channel augmentation improved mean stressed ranking and reduced A stressed misses while increasing clean B false alarms and some attack-specific misses. A common order-dependent score shift can produce different class errors at 0.5. The observed changes leave open the contributions of preprocessing, score drift and forensic evidence degradation. The study uses one seed pair and repeated development, without official evaluation, calibration or verified generator lineage. Group support, especially for individual attacks, limits precision.

## 9 Conclusion

The selected detector pair exposes a robustness trade-off that average stressed AUC alone would miss. A matched multi-seed comparison and a locked held-out evaluation are needed before extending this conclusion beyond development.

## Reproducibility and conference extension

This work extends the accepted conference paper “LAVA: A Lightweight Benchmarking Framework for Robust and Efficient Deepfake Voice Detection.” The extension adds a controlled intervention with matched initialization and source order, class- and attack-specific order diagnostics, matched component inference and speaker-cluster uncertainty. Before submission, add the verified venue and bibliographic citation and disclose the prior version and overlap in the cover letter. The journal title is distinct. Check the conference publication agreement and rights for any reused text, figure or table. Clearance remains unverified. [AES Journal Author Guidelines](https://aes.org/publications/journal-author-guidelines/) · [AES Publications Policy](https://aes.org/publications/publications-policy/)

The repository retains the release registry, selected checkpoints, per-view scores, resource measurements, manifests, deterministic resume qualification and seals. This draft extends the accepted benchmark through a matched training intervention and within-source ordered class/attack diagnostics. Conference venue, publication status, copyright and author disclosure require verification before submission. [Final diagnostic](../../../docs/jaes_v2/FINAL_SCIENTIFIC_DIAGNOSTIC.md) · [Audit](../../../outputs/jaes_v2/phase10/final_diagnostic_v1/scientific_evidence.json) · [Registry](../../../configs/jaes_v2/phase10/prospective_medium_v1.json) · [AES author guidelines](https://aes.org/publications/journal-author-guidelines/)
