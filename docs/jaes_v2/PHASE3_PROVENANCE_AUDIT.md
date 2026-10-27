# Phase 3 provenance and confounding audit

Scope: 192 real public-source **development** recordings; no independent confirmation.

Manifest: `outputs/jaes_v2/datasets/mlaad_tiny_dev_v1/provenance_manifest.csv`.
Revision: `9143e5ea709575ebab6bec52840a1043aada7bb1`.
Selection hash and audio hashes are persisted in the import/acquisition/config evidence before inference.

## What is established

- Each selected audio matches its upstream LFS SHA-256 and expected byte size.
- REAL/FAKE is the author audiofolder convention, not model predictions. Generator/system and architecture fields are taken from author per-file `meta.csv`. Changing an annotation or label causes import failure.
- 96 REAL and 96 FAKE are English; every selected source is 16 kHz. Four named FAKE groups each contribute 24 recordings.
- FAKE has four distinct source-text speakers, each contributing 24 recordings overall (six per system), parsed from the explicit structured `original_file` annotation. This is neither a synthesized-voice identity nor a target-speaker certificate.
- All transformed conditions retain sample and original/local parent identity. Different generator utterances are not artificially paired.
- Exact-byte overlap with the canonical legacy manifest is zero. Within this cohort exact decoded-PCM duplicate groups are zero. Prefix-correlation candidate screen returned zero pairs using >=.98 correlation, 8 kHz <=3 s prefixes and +/-20 ms lag.

## What remains unknown

REAL speaker, original recording lineage and true source parent are not provided by the tiny audiofolder. Local byte parents group variants only. A FAKE `original_file` supplies annotated original-source dependence, but does not identify every reference/synthesized voice. The author model_name is a system annotation, not an independently verified checkpoint hash. Relative familiarity to either existing detector's training data is UNKNOWN. Exhaustive perceptual/decoded overlap against all legacy audio was not performed. No operation certifies population independence from zero known overlap.

Both classes are associated with M-AILABS original/source-text material; that does not equalize microphones, mastering, synthesis, speaker distribution, silence or loudness. Language is controlled at English here and sample rate is equal, but the cohort is deliberately small and covers only four systems/four known source-text speakers. It cannot support a global ranking of generators.

## Shortcut evidence and grouping

The reused Phase 2 audit reports original-recording leave-one-parent-out features. Its constant sample-rate predictor can show zero accuracy on an exactly balanced leave-one-out cohort because training-majority inversion is repeated in every fold. That value is **not an acoustic signal**, and is retained as historical diagnostic output rather than promoted as a result. The separate Phase 3 grouped diagnostic marks constant features NONINFORMATIVE_CONSTANT_FEATURE.

`outputs/jaes_v2/audit/phase3_shortcuts_v1/` evaluates the original recordings and the actual prepared detector inputs. Known original source-speaker/parent components are kept together in four seeded group folds; missing REAL identities remain local groups, not verified independent speakers. Fold class counts and a training-only dummy comparator are reported, including an imbalanced/one-class held-out fold; no AUC is invented for those folds. No test/calibration labels or hyperparameter selection are used. These diagnostics measure shortcut availability and cannot prove that either deep detector actually uses those features.

Source and confounding reports are in the shared waveform directory. `source_audit.json` retains the reusable audit's generic near-duplicate-status field; the actual Phase 3 screen is separately executed and recorded in `near_duplicates.json`. Do not interpret the generic field as negating the separate screen.

## Factor estimands

Group contrasts are observational FAKE-miss contrasts, not binary EER per generator. The factorial estimator rejects missing cells, unknown factor/cluster identity, corpus or language support separation and mixed detector/threshold/seed strata. Shared original source-text speakers are jointly resampled across system groups. Four known source-text clusters give weak uncertainty support; descriptive percentile intervals are not confirmatory significance. Real identity remains unknown, so pooled local-parent intervals are explicitly limited. All pairwise descriptions are retained; there are no selected p-value discoveries or causal generator-effect claims.
