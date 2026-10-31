# Phase8 novelty decision

NOVEL_METHOD_READY_FOR_FULL_EXPERIMENT: **NO**.

M1: standard weighted BCE, IMPLEMENTED; no novelty claim. M2: smoothed group-DRO control, IMPLEMENTED; no novelty claim. The source and actual smoke/pilot status are traced in PHASE8_EXPERIMENT_RESULTS.md.

M3a deterministic order-dependent supervision weights reduce to importance-weighted augmentation: PRIOR_ART_OVERLAP → REJECTED. M3b max over the two orders reduces exactly to worst-of-two loss: PRIOR_ART_OVERLAP → REJECTED. M3c acoustic-to-forensic signed commutator supervision lacks a justified target: PROPOSED only; fails the entry gate. No `NovelLoss`, M3 model, M3 training config or successful M3 ablation is created.

Full SONAR now verifies that label-conditioned representation alignment is already occupied prior art; this corrects the Phase7 ABSTRACT_ONLY evidence limitation without changing the historical Phase7 report. No inference of algebraic equivalence is made between SONAR and the unimplemented M3c.

C1 remains a supporting empirical/engineering contribution; C2 is a research question with stronger standard controls; C3 is deferred. There are not three established international novelties. Absence of a verified matching paper would not itself prove novelty. The targeted review is not exhaustive, and abstract-only additional work remains explicitly limited.

The separate FULL_TRAINING_CODE_READY decision is evidence-generated in PHASE8_FULL_TRAINING_READINESS.md. Code maturity and scientific novelty must never share a single green status.
