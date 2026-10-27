# Phase 2 metadata shortcut diagnostics

Actual cohort: 64 legacy validation recordings, 32 REAL / 32 FAKE, selected with seed 2804 before inference. All results here are **LEGACY DEVELOPMENT**, not external evaluation.

The diagnostic models use the original recording's metadata/acoustic summaries. The detector sees a <=3 s prefix, so original-duration association is not proof the detector uses duration. Five logistic baselines use fixed C=1, StandardScaler fitted within each fold and leave-one-local-parent-out validation; no model or feature tuning uses test data.

| Diagnostic | Held-out development accuracy |
|---|---:|
| Duration only | 0.515625 |
| Sample rate only | 0.531250 |
| Silence fraction only | 0.625000 |
| RMS level only | 0.593750 |
| Combined numeric metadata | 0.640625 |

These are small-cohort descriptive values, not population estimates or an indictment of the detector. No probability-calibration or causal claim is made. RMS is a signal-level proxy, not measured perceptual loudness/LUFS. Metadata access does not establish speaker/source independence.

Exact decoded PCM audit found zero duplicate groups in this cohort. The candidate near-duplicate screen found zero pairs at threshold 0.98, using <=3 s prefixes resampled to 8 kHz and a +/-20 ms correlation lag. This can miss different crops, long-range matches and nonlinear processing. No sample was automatically excluded. UNKNOWN speaker/generator/original-parent fields remain UNKNOWN; “zero known overlap” is not “verified disjointness.”

Evidence: `outputs/jaes_v2/audit/phase2_source_audit/source_audit.json`, `source_features.csv`, `near_duplicate_candidates.csv` and `overlap_dimensions.json`. The selected study ran all 64 records without exclusions.
