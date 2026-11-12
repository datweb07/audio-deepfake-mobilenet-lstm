# Supervisor major-revision tracker

This tracker covers the 12 requirements received on 11 November 2026. Scientific outputs are in [the V4 supplement](../../outputs/jaes_v2/supervisor_major_revision). The original V3 PDF and sealed experiments serve as the integrity baseline.

| ID | Requirement and evidence | Resolution and output | Status | Remaining limit or dependency |
| --- | --- | --- | --- | --- |
| R01 | V3 and closest published channel, order and classwise studies | Three precise questions and three empirical contributions in §1, tied to §§3–6 | COMPLETED | Novelty is the controlled empirical contrast |
| R02 | Accepted Word and final conference reviewer response | Fourteen-dimension matrix, overlap review, draft cover statement and condensed table in §1 | COMPLETED_WITH_LIMITATIONS | Confirm proceedings, DOI if assigned and rights |
| R03 | Medium/foundation/development/calibration manifests and locked list | Audited nested 1,280 cohort with matching class/attack proportions and verified hashes | NOT_FEASIBLE_BEFORE_DEADLINE | For the current released trainer, parsing and fixed-budget lifecycle for 1,280 remain unsupported. Reassess only with isolated qualification and review time |
| R04 | Frozen medium registry, exact-resume evidence and preflights | Third seed plan and 9307-specific smoke. See training readiness | QUALIFIED_FOR_AUTHOR_APPROVAL | Full pair requires separate approval |
| R05 | Four sealed development grids and locked held-out predictions | §6.2 and CSV confusion matrices, risk grid, empirical ROC/DET, score CDF and threshold windows | COMPLETED | Grid is retrospective and supports no threshold selection |
| R06 | Source-aligned A05/A06 and A17–A19 predictions | §6.3 and supplementary attack recovery, speaker intervals and repeated-view counts | COMPLETED_WITH_LIMITATIONS | Published attack relationships are partial. Generator-family attribution and multiplicity-adjusted confirmation remain unavailable |
| R07 | Source-linked compositions and standalone component runs | §6.4 and order-source rows, waveform contract, matched components and replay anchors | COMPLETED_WITH_LIMITATIONS | Independent cue mechanisms and equal achieved post-codec SNR are unestablished |
| R08 | Official DF release, provenance schema and local resource measurement | External and venue audit and metadata join, §6.5 and validity limits | BLOCKED_BY_RESOURCES | Full archive exceeds disk. A subset still needs license, member and overlap reconciliation |
| R09 | Hash-checked 9207 Keras exports and development timing | §6.6 and efficiency JSON, stage trainability, load, CPU latency and peak working set | COMPLETED_WITH_LIMITATIONS | First 12 development inputs and one timing arm. No device, streaming or system-throughput claim |
| R10 | Registered endpoint and subgroup evidence | Six RQ-oriented result subsections and reduced redundant Discussion | COMPLETED | Scope of each estimate retained |
| R11 | V3 bibliography and peer-reviewed nearest comparisons | Gap, added comparison context, updated audit and separate category/year-specific ranking review | COMPLETED_WITH_LIMITATIONS | Current institutional JCR/Scopus criterion needs confirmation |
| R12 | V3 figures and tables and V4 derived supplement | Preserved originals, one new operating-point figure, one extension table, and hash/compile QA | COMPLETED | Author metadata and final journal-format QA remain submission tasks |

The PDF is for supervisor review. The size experiment and external evaluation have no result rows in the manuscript. [Training readiness](SUPERVISOR_TRAINING_READINESS.md) · [Extension matrix](SUPERVISOR_CONFERENCE_EXTENSION_MATRIX.md) · [External and venue audit](SUPERVISOR_EXTERNAL_AND_VENUE_AUDIT.md)

## Supplement inventory

| File | Contents and provenance |
| --- | --- |
| threshold_risk.csv, confusion_at_050.csv | 792 model-partition-condition-threshold rows and the 72 rows at 0.5. Blank FNR/TPR for REAL-only B are intentional. |
| near_threshold.csv, score_and_order_diagnostics.json | Per-class counts, denominators and distribution summaries. ROC and DET coordinates derive from real scores. |
| attack_recovery.csv, attack_source_decisions.csv | Arm miss counts, distinct source identities, condition transitions, repeated errors, and pointwise percentile intervals from 10,000 speaker-cluster draws where supported. |
| order_source_decisions.csv, component_summary.json | Source-level order contrasts, all standalone component summaries, and descriptive score interaction. |
| components_9207/ | New development component scores, waveform diagnostics, 18 score anchors with zero error, and CPU timing for 36 forwards. It contains no raw audio or model checkpoint. |
| operating_points_v4.pdf, score_provenance.json, verification.json | Supplemental figure, registered score replay provenance, and revision integrity and build checks. |
| size_candidate.json, candidate_train1280.csv | Candidate cohort hash, exact quotas, byte audit and overlap. Keep the manifest internal if distributing derived evidence publicly because it contains absolute local paths. |
| df_metadata_audit.json | Official DF key/map join and 2019 overlap, with metadata license and package hashes. |
| seed9307_smoke.json, preflight_erm.txt, preflight_channel.txt, resources.json | Isolated numeric and resource admission evidence. No trained 9307 checkpoint exists. |

For the supplement, use the score-analysis and training-audit scripts shipped with the manuscript source. Raw audio, selected checkpoints and author-approval records stay in the internal workspace. The audit confirms the shared noise policy, seed, RMS and absolute power on all 368 development sources. Held-out uses the same source-keyed implementation under its original lock and was not rerun. The new CPU benchmark times decode/resampling/P0 and forward on clean inputs. Common headroom and channel processing are outside that timing path.

