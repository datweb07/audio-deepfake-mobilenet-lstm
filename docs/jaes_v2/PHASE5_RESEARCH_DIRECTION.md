# Phase 5 research direction — 28 October 2026

The primary investment is C2: a lightweight detector that retains useful forensic evidence under nuisance changes. C1 remains supporting validation and mechanistic diagnosis. C3 routing remains deferred. Three international methodological contributions are **NOT ESTABLISHED**.

The first falsifiable question is whether deployment-feasible relative Mel flooring improves gain stability without damaging discrimination after controlled training. A second question is whether ordinary channel augmentation explains the improvement. Neither relative normalization nor augmentation is claimed as new. The initial experiment is P0/P1 × ERM/channel augmentation, one initialization seed, a frozen ImageNet MobileNetV3Small backbone and a fresh LSTM classifier. This answers a bounded feasibility question, not full-network or cross-corpus performance.

Use teacher-free baselines now. The current teacher runtime does not satisfy the author loading recipe, and training overlap is known at corpus level. Do not invent a reliability-weighted loss before distinguishing it from codec-aware weighting, ordinary consistency and group-robust optimization.

The strongest subsequent hypothesis is **selective preservation of discriminative information under measurable distortion**, rather than unconditional feature equality. It remains PROPOSED: severity/reliability would need a label-free observable estimator, a specified range, equal-view baselines, controlled degradation sweeps, cue-destruction counterexamples and ablations. Discard it if matched standard augmentation or consistency explains the benefit, or clean classwise risk exceeds a prospectively fixed tolerance. No such new loss is implemented in Phase 5.

Mechanism alternatives remain separate: absolute-floor sensitivity, codec spectral loss, truncation/segmentation, silence/corpus shortcuts, destroyed forensic evidence, threshold effects and training-domain limitations. The present intervention addresses the first and documents, rather than resolves, the others. See PHASE5_RESEARCH_FINDINGS.md for executed evidence and PHASE5_C2_NOVELTY_GATE.md for exclusions.
