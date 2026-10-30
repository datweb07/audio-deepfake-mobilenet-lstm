# Phase 5 experiment matrix

| Experiment | Data / exposure | Purpose | Execution gate |
|---|---|---|---|
| P0/P1/P2 frozen old-model diagnostic | Same 32 MLAAD records, native/half gain, two old checkpoints | Feature/embedding/score/decision sensitivity, no retraining quality claim | Completed, 384 forwards including 128 explicit P0 bridge calls |
| Tier 0 | Synthetic contracts and official-format ZIP fixtures | Correctness, not scientific quality | Test reports |
| Tier 1 | 16 selected official train recordings, P0 clean | Verify tiny-data learning | >=.9375 training accuracy before Tier 2 |
| P0+B0 | 128 train, 64 dev, original P0 | Standard ERM control | Sealed cache + resource + overfit |
| P1+B0 | Identical IDs/seed/updates, P1 | Preprocessor main effect | Same gates |
| P0+B2 | Identical source IDs/updates, random one of four train views | Standard augmentation main effect | Same gates |
| P1+B2 | Identical bank/seed/updates, P1 | Interaction between known controls | Same gates |
| B1 amplitude-only | Clean/half-gain bank; configs supplied | Separate gain from channel augmentation | Planned, not automatically run |
| B3 same-view channel | Same operation as B2 | Avoid artificial extra baseline | Merged into B2 |
| B4 consistency | .5(BCE(view1)+BCE(clean)) + .1 MSE(probabilities) | Ordinary paired consistency control | Implemented/tests; planned only |
| KD / proposed loss / C3 | No valid teacher/new mechanism | Method development | Deferred |

All four main cells share the same Keras ImageNet weight file SHA, random seed5804, fresh head architecture, frozen backbone, train/development identities, batch16, 6 fixed epochs, optimizer/schedule, and fixed .5 threshold. B0 sees only clean; B2 sees one sampled precomputed view per source/update. B2 has broader condition exposure but no extra optimizer updates. B4 would add a second supervised head pass and must be compared with an equal-two-view baseline before an objective claim. No best-of-many post-hoc gain, threshold, loss weight or source selection is allowed.

Development condition grid: clean, quarter gain, AWGN5, Opus64, AWGN10→Opus24 and reverse. Six conditions per 64 originals are paired repeated measurements, **not 384 independent recordings**. Official attack groups and speaker groups retain denominators. Descriptive bootstrap uses official speaker clusters; it does not replace training seeds or certify original-utterance independence. EER is only a ranking diagnostic, never the chosen deployment threshold. Calibration remains reserved.

Formal Phase 6 design: lock at least three independent initialization seeds on one split; predeclare clean-risk tolerance and primary robustness contrast before external labels; use matched training exposure and compute; compare B1, strong RawBoost/channel baselines and equal-view consistency; ablate proposed components only if a technical distinction survives prior-art review. Separate seed variability from test-data cluster uncertainty and control multiplicity for formal claims. These Phase 5 cells are exploratory, with all conditions reported and no significance claims.
