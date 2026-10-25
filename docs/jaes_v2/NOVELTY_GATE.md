# Novelty gate and ranked decisions

Review date: 25 October 2026. Status has separate engineering and scientific dimensions.

| Priority | Candidate | Engineering status | Literature status | Scientific support |
|---|---|---|---|---|
| 1 | C1 generator–channel composition interaction | PILOT_EVALUATED for channel order only | Nearest works checked; difference not established | NOT_SUPPORTED yet |
| 2 | C2 lightweight robust forensic representation | DESIGNED; no proposed objective locked | Strong overlap with PCL-NET and DK-CAST | NOT_SUPPORTED |
| 3 | C3 budget-aware selective inference | DESIGNED | Broad principles established; closest audio-specific policy search incomplete | NOT_SUPPORTED |

## C1 questions and falsification

The specific open candidate is whether a prespecified generator-shift×channel-composition interaction remains after controlling corpus, speaker, bitrate, order, preprocessing and threshold. RADAR already uses sequential media transformations; ADD-C already benchmarks communication channels. Class asymmetry is explicitly studied by FP-NAC and the DSPA noise work. Therefore transformations, asymmetric errors and ordered chains alone cannot be called novel.

The current pilot tests implementation feasibility only. It has no known generator factor. To support C1, form all relevant factorial cells on verified metadata and retain the same parent comparisons for channel order. For different generator groups, use an appropriate group-level/factorial estimator rather than pretending unrelated utterances are paired. The current paired estimator must not be used for that cross-generator contrast without adaptation.

Falsification: interactions vanish within uncertainty, reverse across cohorts, or are explained by clipping/codec delay/threshold placement/source confounding. A null result is retained. Nearest-work review must specifically search for order reversal contrasts, not merely sequential augmentation.

## C2 questions and falsification

PCL-NET already applies pairwise representation consistency to original/neural-codec views. DK-CAST includes distortion-aware student learning, distillation and codec-conditioned losses. A sum of BCE/KD/consistency has no established distinction. Candidate next direction: investigate whether order-sensitive *class-conditional error direction*, rather than unconditional embedding invariance, warrants a different objective. This is a hypothesis, not a new algorithm.

Before coding a proposed loss: explain which C1 failure it addresses; compare to standard KD, consistency and same-view augmentation; specify teacher gradient stopping and temperature; verify checkpoint/license/development performance. A weak or mismatched teacher can make the hypothesis fail. No validated teacher has been acquired in this execution.

## C3 questions and falsification

Confidence rejection, entropy, temperature scaling and cascades are mandatory established baselines. A policy would need to improve risk at matched coverage and **measured total cost**, with all extra views/stages counted. Test-label policy fitting is prohibited. No guarantee under arbitrary shift is claimed. Temporal disagreement or channel sensitivity is only an untested input candidate.

C3 algorithm implementation is deferred according to the user's phase order: a validated C2 student and independent calibration cohort do not yet exist. Implementing an arbitrary cascade now would not satisfy the contribution gate.

No candidate has passed TECHNICALLY_DISTINCT, EXTERNALLY_VALIDATED or SUPPORTED. Negative literature overlap is a result of the audit, not a reason to rename an existing technique.
