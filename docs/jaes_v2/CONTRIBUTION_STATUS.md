# Phase 5 current update — 28 October 2026

**C1: SUPPORTING_VALIDATION / ENGINEERING. C2: STANDARD_BASELINES_PILOT_EVALUATED; NOVELTY_NOT_ESTABLISHED. C3: DEFERRED. INDEPENDENT_CONFIRMATION: NOT_LOCKED. THREE_INTERNATIONAL_CONTRIBUTIONS: NOT_ESTABLISHED.**

ASV controlled foundation:50,224 official CM records; train/dev extraction only; calibration reserved and evaluation unopened. Main pilot:128 train/64 dev, frozen ImageNet MobileNet-LSTM, four6-epoch/48-update cells, one seed. P1 removes gain flips but can worsen noise misses; augmentation changes clean classwise trade-offs. Development FAKE covers7 speakers versus16 for REAL, so support imbalance remains a scientific limitation. Teacher runtime/quality gates are unresolved; no KD/new loss/C3 routing was run.

Measured regression:92/92 JAES V2 and40/40 selected legacy, no failures/errors/skips. Current evidence and full metrics: **PHASE5_RESEARCH_FINDINGS.md**; commands and remaining gates: **PHASE5_NEXT_COMMANDS.md**. Standard controls and software correctness do not establish journal novelty. Earlier sections below are historical and preserved.

---

# Contribution status

## Current Phase 4 state

| Candidate | Actual evidence | Decision |
|---|---|---|
| C1 | Source audit; blocked fixed-pair matched design; 32-record paired nuisance pilot; feature-level counterfactual | Supporting validation/engineering; **not primary novelty**; confirmation absent |
| C2 | Two teacher provenance/license/contract audits; no compatible loaded/validated teacher | NOT_READY for full learner implementation/training |
| C3 | Validated student/independent calibration still absent | Deferred |

576 new predictions and reusable analysis infrastructure do not supply three international scientific contributions. The tested nuisance contrast remains descriptive with four source-text clusters. Full decision and strongest next direction: PHASE4_NOVELTY_DECISION.md. Prior phase tables below are historical.

## Current Phase 3 state — supersedes historical rows

| Candidate | Actual evidence | Scientific gate |
|---|---|---|
| C1 | Pinned public import; 192 recordings; shared seven-condition cache; 2,688 predictions; 24 descriptive group/channel contrasts | EXTERNAL_DEVELOPMENT_EVALUATED; NOVELTY_NOT_ESTABLISHED; independent confirmation pending |
| C2 | Real public diagnostic input now available; teacher still not loaded/validated | TEACHER_FEASIBILITY_ONLY; no new learner |
| C3 | Dependencies documented; no calibration/routing policy | PREREQUISITES_ONLY |

External execution does not establish three international contributions. Four FAKE source-text speakers, UNKNOWN REAL lineage and shortcut availability remain limitations. Regression: 58 V2 / 40 selected legacy, all passed. Full evidence: PHASE3_RESEARCH_FINDINGS.md.

| Contribution | Design | Code | Validation | Scientific gate |
|---|---|---|---|---|
| C1 ordered-channel slice | Defined | Implemented | 19 V2 tests; real 12×5 pilot | PROTOCOL EXTENSION; NOT_SUPPORTED as novelty |
| C1 generator×channel | Defined as next experiment | Hold-out validators only; factorial estimator pending | MISSING_DATA | No generator identities; independent confirmation absent |
| C2 robust lightweight learner | Prior-art-informed candidate | No proposed loss implemented | Teacher and controlled training pending | NOVELTY_BLOCKER |
| C3 selective inference | Baselines/constraints defined | No candidate policy implemented | Calibration/data/student pending | VALIDATION_BLOCKER and novelty search incomplete |

These statuses deliberately distinguish engineering progress from claims. Pilot class-asymmetric errors do not establish a mechanism or a method improvement. No state is SUPPORTED, EXTERNALLY_VALIDATED or journal-ready. The phase order prohibits implementing an arbitrary C3 policy before a validated student.

Phase 2 supersedes the engineering rows above: C1 channel controls are DEVELOPMENT_EVALUATED (64 recordings, 14 conditions); the public importer and factorial estimator are implemented/tested. Public audio integration and real generator-factor estimation remain EXTERNAL_DATA_PENDING. C2 has an audited teacher candidate, without a downloaded/loaded/validated checkpoint; C3 has no policy implementation. Scientific support for all three contributions remains NOT_ESTABLISHED.
