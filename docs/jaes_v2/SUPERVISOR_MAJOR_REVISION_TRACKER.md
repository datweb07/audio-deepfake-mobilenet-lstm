# Supervisor major-revision tracker

This tracker covers the 12 requirements received on 11 November 2026. Scientific outputs are in [the V4 supplement](../../outputs/jaes_v2/supervisor_major_revision). V5 adds the verified 9307 development pair. The earlier readiness audit remains a historical snapshot. V4 and all sealed results remain the integrity baseline. [Three-seed review](THREE_SEED_REPLICATION_REVIEW.md)

| ID | Requirement and evidence | Resolution and output | Status | Remaining limit or dependency |
| --- | --- | --- | --- | --- |
| R01 | V3 and closest published channel, order and classwise studies | Three precise questions and three empirical contributions in §1, tied to §§3–6 | COMPLETED | Novelty is the controlled empirical contrast |
| R02 | Accepted Word and final conference reviewer response | Fourteen-dimension matrix, overlap review, draft cover statement and three-seed journal extension table in §1 | COMPLETED_WITH_LIMITATIONS | Confirm proceedings, DOI if assigned and rights |
| R03 | Sealed four-model size comparison | Both 1,280 arms verified. V7 reports the nested fixed-epoch contrast and conditional speaker intervals | EXPERIMENT_COMPLETED_WITH_EXPLICIT_LIMITATIONS | One seed, doubled optimizer updates, shared development |
| R04 | Frozen medium registry, exact-resume evidence and preflights | Both 9307 runs verified. Six development fits and three paired contrasts in §6 | COMPLETED_WITH_LIMITATIONS | Three fits per arm share development sources. Locked held-out has two pairs |
| R05 | Four sealed development grids and locked held-out predictions | §6.2 and CSV confusion matrices, risk grid, empirical ROC/DET, score CDF and threshold windows | COMPLETED | Grid is retrospective and supports no threshold selection |
| R06 | Source-aligned A05/A06 and A17–A19 predictions | §6.3, 9307 A05/A06 sign reversal and supplementary recovery, speaker intervals and repeated-view counts | COMPLETED_WITH_LIMITATIONS | Published attack relationships are partial. Generator-family attribution and multiplicity-adjusted confirmation remain unavailable |
| R07 | Source-linked compositions and standalone component runs | §6.4 and all three composed-order pairs. Standalone component diagnostics cover 9107 and 9207 | COMPLETED_WITH_LIMITATIONS | Independent cue mechanisms and equal achieved post-codec SNR are unestablished |
| R08 | Pinned archive and current local resource audit | Bounded retrieval fits in principle. Full DF exceeds disk. V7 discloses missing independent confirmation | CONDITIONAL_REQUIREMENT_ADDRESSED_WITH_EXPLICIT_LIMITATION | Audio-use terms, lineage, overlap and actual decode need verification |
| R09 | Hash-checked 9207 Keras exports and development timing | §6.6 and efficiency JSON, stage trainability, load, CPU latency and peak working set | COMPLETED_WITH_LIMITATIONS | First 12 development inputs and one timing arm. No device, streaming or system-throughput claim |
| R10 | Registered endpoint and subgroup evidence | Six RQ-oriented result subsections revised for three development pairs and two held-out pairs | COMPLETED | Scope of each estimate retained |
| R11 | V3 bibliography and peer-reviewed nearest comparisons | Gap, added comparison context, updated audit and separate category/year-specific ranking review | COMPLETED_WITH_LIMITATIONS | Current institutional JCR/Scopus criterion needs confirmation |
| R12 | V3 figures and tables and V4 derived supplement | Preserved V4 and updated three plots and three tables from the sealed V5 aggregate | COMPLETED | Author metadata and final journal-format QA remain submission tasks |

V7 incorporates the completed R03 size experiment. See the [final R01–R12 audit](SUPERVISOR_12_REQUIREMENTS_FINAL_AUDIT.md) for the current status of each requirement.

## V5 supplement

See [verification and statistics](../../outputs/jaes_v2/three_seed_revision/verification_and_statistics.json) and [the three-seed aggregate](../../outputs/jaes_v2/phase10/empirical_execution_v1/paired_seed_evidence_9107_9207_9307.json). The V4 PDF preserves its original figures and tables. The V7 size analysis is recorded separately. R08 remains experimentally open.

## Historical V4 supplement inventory

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
| seed9307_smoke.json, preflight_erm.txt, preflight_channel.txt, resources.json | Isolated numeric and resource admission evidence. These files predate the two subsequently verified 9307 training runs. |

For the supplement, use the score-analysis and training-audit scripts shipped with the manuscript source. Raw audio, selected checkpoints and author-approval records stay in the internal workspace. The audit confirms the shared noise policy, seed, RMS and absolute power on all 368 development sources. Held-out uses the same source-keyed implementation under its original lock and was not rerun. The new CPU benchmark times decode/resampling/P0 and forward on clean inputs. Common headroom and channel processing are outside that timing path.
