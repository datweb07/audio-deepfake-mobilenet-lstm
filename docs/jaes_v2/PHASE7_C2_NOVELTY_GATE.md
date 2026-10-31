# PHASE7 C2 NOVELTY GATE

## Proposed C2 Method and Falsification

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

## International Novelty Assessment

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

