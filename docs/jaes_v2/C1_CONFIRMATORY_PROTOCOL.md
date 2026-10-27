# C1 confirmation protocol draft — NOT LOCKED

Status: **BLOCKED_ON_EXTERNAL_DATA**. No external confirmation split has been opened or evaluated in Phase 2. The legacy controls run is development only.

Required lock fields before execution:

| Field | Current decision / missing input |
|---|---|
| Dataset/source/version | ASVspoof 2019 LA is primary integration candidate; external corpus still to choose |
| Exact split hashes | PENDING actual official manifests and audio audit |
| Grouping | Verified speaker/source/parent as available; local checksum alone cannot prove original independence |
| Factor definition | Distinguish official attack partition from checkpoint-relative familiarity; legacy training overlap is UNKNOWN |
| Channels | Preselect noise policies, codec/bitrate, explicit order and component controls after development |
| Primary estimand | Difference of paired channel risk changes between documented factor groups; equal-cluster/corpus weights |
| Class strata | Generator/attack comparison primarily on FAKE misses unless both-class cells have a justified cohort definition |
| Threshold/calibration | Fixed development-selected rule; no test tuning; probability calibration requires a separate cohort |
| Uncertainty | Cluster resampling preserves all channel views and shared speaker dependence; reject missing/underidentified cells |
| Multiple comparisons | Prespecify one primary family and correction before confirmation; current intervals are descriptive only |
| Sample-size rationale | PENDING available verified independent clusters and precision/power calculation |
| Confirmation identity | PENDING untouched corpus/recording list; cannot reuse a cohort used to design controls |

`factorial.py` implements equal-cluster/corpus contrasts with paired channels within each recording. Different attack/generated utterances are not paired. Shared speaker clusters across factors are jointly resampled. The current stratified version rejects shared clusters across corpora, missing cells, insufficient clusters, unknown factor identities and corpus/factor confounding. This is an observational estimator, not a causal certificate.

The CLI deliberately does not expose a CONFIRMATORY flag. A future lock must contain concrete datasets, hashes, identities, thresholds and sample-size decisions, not just a renamed development config. Only then add and review a confirmation execution path.
