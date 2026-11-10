# Class-Asymmetric Robustness and Transformation-Order Effects in Lightweight Audio Deepfake Detection

Working journal manuscript, 10 November 2026

## Abstract

Speech deepfake detectors often encounter noise and lossy coding in succession. We examine whether channel augmentation improves stressed ranking while preserving decisions at a fixed threshold. Paired MobileNetV3Small–LSTM detectors trained on 640 ASVspoof 2019 LA recordings shared initialization and source order within each of two seed blocks. On a common 368-recording development cohort, mean stressed ROC-AUC on the mixed-class stratum increased from 0.6715 to 0.8026 and from 0.5946 to 0.7839. Stressed spoof misses fell from 222 to 155 and from 143 to 134 out of 336 transformed views per block. Clean bona fide false alarms on a separate stratum moved in opposite directions, while stressed misses increased for attacks A05 and A06 in both blocks. These two seed blocks on one development cohort show that aggregate ranking gains can coexist with attack-specific deterioration at the deployed decision threshold.

## 1 Introduction and research gap

The conference benchmark compared six heterogeneous detectors and documented stress failures on a fixed diagnostic subset. The journal extension tests a specific training intervention under matched initialization and source order. We ask whether channel augmentation changes aggregate ranking and classwise decisions (RQ1), whether the changes are concentrated by attack or speaker (RQ2), and whether reversing noise and codec order alters those decisions when the component settings are matched (RQ3). The present evidence addresses these questions in two paired initialization blocks on a common development cohort. Locked within-release evaluation is pending.

A detector may receive speech after both noise insertion and lossy coding. Aggregating those conditions can obscure operating-point errors and the difference between applying the same operators in opposite orders. This study asks which source-linked errors remain visible under a fixed decision rule when augmentation and baseline training share the numerical and data contract.

## 2 Related work

RawBoost adds waveform corruptions. F-SAT learns against frequency-selective adversarial perturbations. PCL-NET and TFCL align paired codec or acoustic front-end views, while DK-CAST uses clean-to-degraded distillation and codec-aware weighting. SONAR couples low- and high-frequency representations with class-asymmetric alignment. RADAR provides transformation-robustness development and blind-evaluation conditions. Proteus searches ordered transform chains and reports strong bona fide vulnerability. FP-NAC isolates class-asymmetric coding failures and adapts a neural codec with detector guidance. RealComm uses paired calls and staged acquisition-to-transmission augmentation. Our comparison measures a specific conventional Opus/noise order contrast under a matched CPU training intervention. [Proteus](https://arxiv.org/html/2606.29544) · [FP-NAC](https://arxiv.org/html/2609.39651) · [RealComm](https://arxiv.org/pdf/2610.27459) [RawBoost](https://arxiv.org/html/2111.04433) · [F-SAT](https://www.cs.columbia.edu/~junfeng/papers/fsat-iclr25.pdf) · [PCL-NET](https://arxiv.org/html/2610.07216v1) · [TFCL](https://arxiv.org/html/2607.17761v2) · [DK-CAST](https://link.springer.com/article/10.1007/s10791-025-09746-4) · [SONAR](https://proceedings.mlr.press/v306/hidekel26a.html) · [RADAR](https://arxiv.org/html/2605.09568v3)

## 3 Formulation and controlled setup

Prior robustness work also reports class asymmetry under a fixed clean-calibrated threshold, and cross-dataset experiments have documented codec- and vocoder-dependent failures. The contribution evaluated here is the combination of a matched augmentation intervention, conventional order reversal and same-record component controls, with error counts at a registered threshold and speaker-cluster uncertainty. [Noise-masked attack study](https://ieeexplore.ieee.org/abstract/document/11476736) · [Maltby et al.](https://repository.uwl.ac.uk/id/eprint/15273/)

For source x with class y and transform T, score p(Tx) is classified as FAKE at p ≥ 0.5. The same source supplies two ordered channel views. Record which of the two alone causes a decision error and retain both errors when computing per-condition risks.

The two arms minimize binary cross entropy. ERM trains on clean speech. Channel training samples one view per source from clean, half-gain, noise10 and Opus24. Both use P0 Mel sequences, an ImageNet-initialized MobileNetV3Small and LSTM128, one head warm-up epoch and partial fine-tuning with frozen BN. The CPU execution contract uses deterministic initialization and full-shape recompute. All four runs complete 21 epochs and 840 updates using the registered Adam schedule. We select the minimum clean A binary cross-entropy checkpoint. Seeds 9107 and 9207 define separate initialization blocks.

## 4 Data, metrics and statistics

ASVspoof 2019 LA medium contains 320 REAL and 320 FAKE training sources and attacks A01–A06. Development contains 224 A sources, 112 per class, and 144 REAL-only B sources. The two strata cover seven and nine development speakers respectively. Train and development speaker IDs are disjoint. The audit checked manifest and source fingerprints. Same-record channel views do not establish shared generator lineage. [Dataset release](https://datashare.ed.ac.uk/handle/10283/3336)

All four selected models score clean, two gain controls, noise5, Opus64 and two ordered noise/codec chains, yielding 2,576 scores each. The chains share a fixed absolute noise plan with 10 dB reference SNR and Opus24. The standalone noise and codec conditions use different settings. Stress macro AUC averages noise5 and the ordered chains on A. The corresponding error counts use 336 views per class across 112 sources. Report class errors at 0.5 and ROC-AUC/EER only where both classes are supported. Multiple views per source are dependent. Paired speaker-cluster bootstrap resamples the seven A speakers and nine B speakers separately, retaining each source and all associated views. We use 10,000 draws and seed 91109. These descriptive intervals condition on the fitted models. We summarize block differences by mean and range. Two blocks do not support inference about the distribution of training-seed effects.

## 5 Results

| Development endpoint | 9107 ERM | 9107 Channel | 9207 ERM | 9207 Channel |
| --- | ---: | ---: | ---: | ---: |
| Selected update | 560 | 760 | 520 | 720 |
| A clean ROC-AUC | 0.9872 | 0.9814 | 0.9851 | 0.9809 |
| A clean EER | 4.46% | 4.91% | 5.36% | 5.36% |
| A macro stressed ROC-AUC | 0.6715 | 0.8026 | 0.5946 | 0.7839 |
| A clean REAL false alarms | 3/112 | 3/112 | 4/112 | 4/112 |
| A clean FAKE misses | 6/112 | 9/112 | 9/112 | 9/112 |
| A stressed REAL false alarms | 54/336 | 45/336 | 167/336 | 70/336 |
| A stressed FAKE misses | 222/336 | 155/336 | 143/336 | 134/336 |
| B clean REAL false alarms | 19/144 | 31/144 | 26/144 | 20/144 |
| A01 stressed FAKE misses | 21/57 | 2/57 | 13/57 | 1/57 |
| A02 stressed FAKE misses | 47/57 | 46/57 | 27/57 | 45/57 |
| A03 stressed FAKE misses | 38/57 | 8/57 | 30/57 | 7/57 |
| A04 stressed FAKE misses | 27/57 | 4/57 | 19/57 | 3/57 |
| A05 stressed FAKE misses | 48/54 | 50/54 | 38/54 | 47/54 |
| A06 stressed FAKE misses | 41/54 | 45/54 | 16/54 | 31/54 |

| Channel − ERM | 9107 [95% cluster CI] | 9207 [95% cluster CI] | Mean | Range |
| --- | ---: | ---: | ---: | ---: |
| Stress ROC-AUC | +0.1312 [+0.0207, +0.2387] | +0.1893 [+0.0932, +0.2684] | +0.1602 | +0.1312 to +0.1893 |
| A clean REAL false alarms (pp) | +0.00 [-2.68, +2.68] | +0.00 [-2.68, +2.68] | +0.00 | +0.00 to +0.00 |
| A clean FAKE misses (pp) | +2.68 [+0.00, +6.25] | +0.00 [-3.57, +4.46] | +1.34 | +0.00 to +2.68 |
| A stressed REAL false alarms (pp) | -2.68 [-8.63, +4.76] | -28.87 [-37.20, -19.05] | -15.77 | -28.87 to -2.68 |
| A stressed FAKE misses (pp) | -19.94 [-25.31, -14.58] | -2.68 [-10.42, +3.57] | -11.31 | -19.94 to -2.68 |
| B clean REAL false alarms (pp) | +8.33 [+1.39, +17.36] | -4.17 [-6.94, -0.69] | +2.08 | -4.17 to +8.33 |
| A01 stressed FAKE misses (pp) | -33.33 [-40.00, -25.93] | -21.05 [-29.82, -11.67] | -27.19 | -33.33 to -21.05 |
| A02 stressed FAKE misses (pp) | -1.75 [-13.73, +10.00] | +31.58 [+22.22, +40.35] | +14.91 | -1.75 to +31.58 |
| A03 stressed FAKE misses (pp) | -52.63 [-71.93, -33.33] | -40.35 [-61.67, -20.00] | -46.49 | -52.63 to -40.35 |
| A04 stressed FAKE misses (pp) | -40.35 [-56.25, -25.00] | -28.07 [-49.02, -7.94] | -34.21 | -40.35 to -28.07 |
| A05 stressed FAKE misses (pp) | +3.70 [-3.70, +12.70] | +16.67 [+3.70, +29.82] | +10.19 | +3.70 to +16.67 |
| A06 stressed FAKE misses (pp) | +7.41 [-6.67, +22.92] | +27.78 [+11.76, +43.75] | +17.59 | +7.41 to +27.78 |

Both seed blocks show higher aggregate stressed ranking (+0.1312 and +0.1893 AUC units) and fewer A stressed FAKE misses (−19.94 and −2.68 percentage points). The 9207 miss interval crosses zero. A clean REAL false-alarm counts match within each block, A clean FAKE misses increase in 9107 and tie in 9207, and clean B REAL false alarms reverse direction (+8.33 and −4.17 points). A05 and A06 worsen in both seed blocks. Their 9107 intervals include zero. A02 also worsens in 9207. The registered selected-checkpoint comparison fails A clean miss, B clean false alarm and A06 stress miss in 9107, and A05/A06 stress miss in 9207. These guards describe the selected CLEAN_BCE checkpoints.

| Seed / attack | Clean | Noise5 | Noise→codec | Codec→noise | Missed in all three | Harmed / recovered sources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9107 / A05 | 0→0 | 17→17 | 14→16 | 17→17 | 14→16 | 3 / 1 |
| 9107 / A06 | 3→6 | 18→16 | 7→12 | 16→17 | 6→12 | 7 / 2 |
| 9207 / A05 | 0→0 | 17→17 | 7→13 | 14→17 | 7→13 | 8 / 0 |
| 9207 / A06 | 5→7 | 9→12 | 0→5 | 7→14 | 0→5 | 10 / 0 |

Each row contains 18 original recordings and 54 stressed views. Each condition has a denominator of 18. Harmed and recovered denote at least one cross-model threshold crossing on a stress view and may overlap. A05 clean errors were zero throughout. In both seeds its excess stress misses came from ordered processing, with Noise5 unchanged. A06 already incurred clean errors, and its largest excess came from Noise→codec in 9107 and Codec→noise in 9207. At 9207, 8/18 A05 and 10/18 A06 sources were harmed, with no recovered sources for those two attacks. At 9107, the counts were 3/18 and 7/18, with some recovery. Source-level transitions and per-condition mean FAKE scores are in the numerical artifact. These are fixed-threshold transitions on FAKE-only subsets. Attack-only ROC-AUC is undefined and no generator mechanism can be inferred. A05 and A06 were represented during training.

Among 112 A REAL sources, Noise→codec alone caused 43→19 errors in 9107 and 63→31 in 9207, with zero reverse-only errors. Among 112 A FAKE sources, Codec→noise alone caused 48→17 and 40→26 errors, respectively. Noise→codec-only FAKE errors were 1→0 and 0→0. Both models can shift scores in the same direction, producing different decision effects by class at 0.5. Only 9107 has matched standalone Noise10/Opus24 component scores and a nuisance-regression screen. Noise5 and Opus64 cannot substitute for those components. The 9107 adjusted order-score class AUC was 0.538 for ERM and 0.432 for channel in the declared direction. This does not identify an independent forensic cue. The 9207 result repeats order asymmetry at the decision level, with no matched-component or nuisance-screen replication.

[Paired effect intervals](../../../outputs/jaes_v2/phase10/empirical_execution_v1/paired_seed_evidence_9107_9207.png) · [Full numerical evidence](../../../outputs/jaes_v2/phase10/empirical_execution_v1/post9207_consolidated_v1.json)

### Matched component diagnostic

For seed 9107 only, we scored Noise10 and Opus24 alone on all 368 development sources with both frozen models. The 1,472 new predictions were combined by source identity with the sealed clean and composed predictions. The component waveforms reconstructed both registered orders exactly. A three-source replay checked clean and composed scores for both models. The signed order contrast is p(noise→codec) minus p(codec→noise). The excess composition contrast is the mean composed score minus both component scores plus the clean score. The second quantity describes departure from score additivity and can arise from nonlinear coding or detector response.

For ERM, order-score AUC for class was 0.515 before and 0.538 after a leave-one-speaker-out nuisance regression on development. For CHANNEL, order-score AUC for class was 0.396 before and 0.432 after a leave-one-speaker-out nuisance regression on development. This descriptive analysis uses selected models on the same development cohort. It cannot identify an independent forensic cue or support a new training objective.

## 6 Ablations and comparison scope

The existing controlled factor is training augmentation. The gain and ordered conditions are evaluation diagnostics. An ablation of a proposed learning mechanism will require an independent target and equal-view controls. Cross-paper metric comparisons require matched corpus and operating policy. The conference benchmark used six heterogeneous artifacts on a different collection and a fixed stress diagnostic.

## 7 Efficiency

CPU logs record optimizer/validation times in minutes of 42.6/40.2 and 82.9/39.1 for ERM and channel at seed 9107, and 41.5/41.3 and 82.4/39.5 at seed 9207. Sums can include replay after interruption. These are desktop training measurements. They provide no edge-latency or foundation-scale estimate.

## 8 Discussion and limitations

At a fixed threshold, channel augmentation can improve pooled ranking while worsening attacks or moving class errors in different directions. Differences between fixed-threshold error and ranking point to possible changes in score placement. The order contrasts alone cannot separate feature degradation, preprocessing effects and detector response. Same-record component evidence is retrospective and available for one block. This experiment establishes no new learning mechanism. The 640-recording training cohort, seven mixed-class development speakers and small attack groups limit inference. Both initialization blocks use the same development cohort. A locked subset from the same release would test within-release transport, and independently confirmed external generalization remains open.

## 9 Conclusion

Channel augmentation improved aggregate stressed ranking in both development blocks while worsening A05/A06 fixed-threshold misses. Source-linked class and attack counts reveal operational risks obscured by averages. The next test is the locked within-release evaluation.

## Claim-to-evidence ledger

| Claim | Artifact and scope |
| --- | --- |
| 640 training sources, 368 development sources, 840 updates, selected checkpoint and CPU resources | The paired seed evidence binds the four run seals, score hashes, checkpoints and logged resources. [Two-seed evidence](../../../outputs/jaes_v2/phase10/empirical_execution_v1/paired_seed_evidence_9107_9207.json) |
| Stress AUC, classwise trade-off, attack and speaker heterogeneity | Same-record development scores, two matched blocks, with speaker-cluster uncertainty. [Scientific evidence](../../../outputs/jaes_v2/phase10/empirical_execution_v1/post9207_consolidated_v1.json) |
| Matched Noise10/Opus24 component and order diagnostics | 1,472 new development predictions with registered waveform and score checks. [Component evidence](../../../outputs/jaes_v2/phase10/final_diagnostic_v1/matched_components.json) |
| Replication and official held-out | Protocol locked. Both required seed pairs are complete. Evaluation access remains closed. [Protocol](../../../docs/jaes_v2/EMPIRICAL_EXECUTION_PROTOCOL.md) |

## Reproducibility and conference extension

This work extends the accepted conference paper “LAVA: A Lightweight Benchmarking Framework for Robust and Efficient Deepfake Voice Detection.” The extension adds a controlled intervention with matched initialization and source order, a second matched seed block, class- and attack-specific order diagnostics, matched component inference and speaker-cluster uncertainty. Before submission, add the verified venue and bibliographic citation and disclose the prior version and overlap in the cover letter. The journal title is distinct. Check the conference publication agreement and rights for any reused text, figure or table. Clearance remains unverified. [AES Journal Author Guidelines](https://aes.org/publications/journal-author-guidelines/) · [AES Publications Policy](https://aes.org/publications/publications-policy/)

The repository retains the release registry, selected checkpoints, per-view scores, resource measurements, manifests, deterministic resume qualification and seals. This draft extends the accepted benchmark through a matched training intervention and within-source ordered class/attack diagnostics. Conference venue, publication status, copyright and author disclosure require verification before submission. [Final diagnostic](../../../docs/jaes_v2/FINAL_SCIENTIFIC_DIAGNOSTIC.md) · [Audit](../../../outputs/jaes_v2/phase10/final_diagnostic_v1/scientific_evidence.json) · [Registry](../../../configs/jaes_v2/phase10/prospective_medium_v1.json) · [AES author guidelines](https://aes.org/publications/journal-author-guidelines/)
