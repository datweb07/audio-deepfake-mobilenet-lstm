# Phase 4 identifiability assessment

| Estimand | Necessary support | Actual status |
|---|---|---|
| WITHIN_SOURCE_ORDER_EFFECT | Same audio under AB/BA; fixed artifact/threshold; complete paired cells | Executed; same-record preprocessing sensitivity, descriptively clustered |
| BETWEEN_SYSTEM_OBSERVATIONAL_CONTRAST | Documented groups and common language/corpus; different source content disclosed | Historical Phase 3 only; not a source-isolated generator effect |
| SOURCE_MATCHED_SYSTEM_INTERACTION | Identical full source across systems plus paired views and adequate independent grouping | Fixed-pair development design blocked; best English overlap three IDs |
| Population generalization | Appropriate independent source/voice/corpus support | INSUFFICIENT_INDEPENDENT_GROUPS; REAL and synthesized identity unresolved |

The six selected Phase 3 source intersections are zero, so those results cannot identify generator effects separately from source content. Four shared source-text speakers are not 96 independent FAKE sources or voices. Level-3 author matches in full metadata improve content evidence but do not identify synthesized speaker, model checkpoint or all generation settings. Level 4 was not established.

Known original speaker/parent connections are retained jointly; unknown IDs are not merged. The nuisance cohort has four known FAKE original-source-speaker components and sixteen locally grouped REAL records. The chosen bootstrap unit preserves known dependence but cannot certify absence of unknown shared voices/sources.

Primary nuisance contrasts are +.0625, descriptive 95% [.0,.1875] MobileNet; +.1875, [-.125,.4375] ShuffleNet. These are changes in FAKE order miss difference under edge repacking, not matched-system interactions. Four source-text clusters produce coarse, potentially unstable uncertainty. A zero/positive bootstrap endpoint does not confer significance; all 18 contrasts per artifact are reported without multiple-testing claims.

Language and original corpus support are common in the fallback, while source content across systems, actual voice, silence/energy, floor exposure and segment alignment remain potential explanations. Native/headroom/PCM baselines remain separate. Threshold transfer and ranking discrimination are reported separately. A lower REAL false-alarm rate cannot establish better robustness if FAKE misses rise.

The new estimator/API tests cover identical IDs, speaker-only and text-only links, basename collisions, conflicting text, missing cells, weak identities, language/corpus separation, unequal cluster weighting and confirmation exclusion. Blocked estimates remain missing, never zero. No Phase 3 output was changed to repair these limitations retrospectively.
