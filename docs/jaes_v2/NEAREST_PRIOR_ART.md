# Nearest prior art — Phase 3 review, 27 October 2026

Novelty status: **NOT_ESTABLISHED**. The comparison separates literature evidence from this implementation; absence of a method in one paper does not establish first-in-world novelty.

## Direct comparison

| Dimension | Nearest prior work | Accepted conference LAVA | C1 Journal V2 Phase 3 |
|---|---|---|---|
| Channel transformations | [RADAR overview v3](https://arxiv.org/html/2605.09568v3), Section 4: probabilistic sequential media processing, including conventional codecs and noise | 100-record diagnostic stress slice | Explicit fixed-power noise, Opus and separate controls; revision-pinned public sources |
| Order sensitivity | RADAR uses an ordered pipeline; an A→B versus B→A reversal estimator was not identified in the inspected overview | Limited diagnostic, no verified generator factor | Matched within-record reversal; the fact of sequential transformations is not new |
| Attack/generator annotation | [MLAAD author source-tracing protocol](https://deepfake-demo.aisec.fraunhofer.de/sourcetracing) already partitions named synthesis systems | UNKNOWN for local trained artifacts/data | Four author model_name groups, no checkpoint-relative unseen assertion |
| Interaction estimator | RADAR reports challenge robustness; exact reversal-factor estimator not established from the inspected overview. [RawBoost](https://arxiv.org/abs/2111.04433) already combines nuisance processes for augmentation | Not established | Existing difference-in-differences/equal-cluster statistical technique applied to annotated FAKE groups; not a new statistical estimator |
| Confounder controls | [DSPA noise-masked attacks](https://ieeexplore.ieee.org/abstract/document/11476736) explicitly keeps a clean-calibrated threshold and reports class asymmetry | Limited controls | Fixed artifact operating point; noise amplitude, headroom, PCM, delay and source/language audits |
| Independent confirmation | RADAR has a separate multilingual evaluation phase | None for the proposed new claim | Not performed; Phase 3 is development with known metadata limitations |
| Scientific insight | Robustness under source/language/media changes is established as a problem | Conference performance/efficiency conclusions | Actual new external observations reported separately; no mechanism or novel contribution established from one cohort |

RADAR is particularly close: its public release enables robustness diagnostics and already includes sequential processing. Its released dev files are transformed inputs; the inspected label/language protocol does not provide an exact applied chain per utterance. A new local treatment on those inputs would have to name that baseline accurately. The release excludes one corrupt evaluation item relative to the overview's total; the release record, not the older overview count, governs acquisition. [Official release](https://zenodo.org/records/22110815).

## Other concrete overlap

- [RawBoost, ICASSP 2022](https://arxiv.org/abs/2111.04433): combined convolutive, impulsive and additive nuisance augmentation. Combination/augmentation itself cannot be claimed new here.
- [A New Approach to Voice Authenticity, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/muller24_interspeech.pdf): broad voice-edit taxonomy/toolkit. Named audio editing or a larger transform list is not sufficient contribution.
- [FP-NAC, 2026](https://arxiv.org/abs/2609.39651): mixed-pair class-specific codec effects and a detector-guided neural codec. Class asymmetry and forensic-preserving codec objectives already have direct precedent. LAVA's Opus treatment is conventional coding, not neural resynthesis.
- [PCL-NET, 2026](https://arxiv.org/abs/2610.07216): original/compressed-view representation consistency. Reusing consistency for C2 needs a distinct supported mechanism and matched baselines.
- [Source tracing/checkpoint attribution, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/stan25_interspeech.pdf): model identifiers can hide checkpoint overlap. Author model_name in Phase 3 is deliberately not relabelled as cryptographically verified checkpoint identity.

The direct DSPA source describes fixed-threshold noise diagnostics and class asymmetry, so those cannot justify novelty either. A plausible future distinction is a **replicated, practically important, generator-conditional order reversal that survives preprocessing/source/threshold controls**, followed by a method or audio-engineering insight it explains. The current software supports testing that candidate. Observational system-group differences alone do not establish it, and a null development outcome is retained rather than renamed.

This review inspected the RADAR overview and release plus the linked primary descriptions; it is not an exhaustive systematic novelty clearance. No journal manuscript is promoted to confirmatory results based on this review.
