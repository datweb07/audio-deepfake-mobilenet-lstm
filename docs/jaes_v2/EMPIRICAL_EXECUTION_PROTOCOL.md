# Locked empirical protocol and execution handoff

09/11/2026. **BLOCKED_WITH_SPECIFIC_ACTIONS** for additional evidence. The empirical design is frozen before seed 9207. The held-out execution lock remains closed pending a complete 9207 pair, resource choice and separate author authorization. [Machine-readable protocol](../../configs/jaes_v2/empirical_protocol_v1.json)

## Scope and estimands

RQ1 tests whether augmentation improves stressed ranking and FAKE miss while worsening clean REAL false alarm. RQ2 resolves class, attack, speaker and threshold effects. RQ3 measures the same-record score and decision contrast for Noise10→Opus24 versus Opus24→Noise10. The sole architecture under controlled training is MobileNetV3Small–LSTM128. C2 is rejected and C3 deferred. Pretrained AASIST and RawNet2 require checkpoint and overlap audits before use.

Frozen training registry, initialization blocks and CLEAN_BCE checkpoint selection carry forward. FAKE is 1 and the score threshold is 0.5 with ties assigned to FAKE. Delta is channel minus ERM. The primary development endpoints are A macro ROC-AUC averaged over Noise5 and both chains, A stress FAKE miss, A clean REAL false alarm and A clean FAKE miss. B clean REAL false alarms, A stress REAL false alarms, attacks, speaker and order transitions are secondary. Gain conditions, Opus64, Noise10 and Opus24 are supplementary. Noise5 and Opus64 are not matched components of the two chains.

Draw 10,000 paired speaker-cluster bootstrap samples with seed 91109. In development draw A and B separately. Keep all recordings and views together for each sampled speaker. Report percentile 95% intervals and valid/invalid counts. Compare seeds as blocks, giving each seed separately, then mean and range. There is no reliable estimate of seed-level variability from one or two blocks. The 9107 AUC improvement and FAKE miss reduction coexist with a B clean false-alarm increase. LA_0100 and LA_0107 contributed 10 of 12 net new B alarms, and A05/A06 remained difficult. Component diagnostics show score drift and attack heterogeneity, with no established independent forensic mechanism. [Seed evidence](../../outputs/jaes_v2/phase10/empirical_execution_v1/paired_seed_evidence_9107.json) · [Detailed diagnostic](FINAL_SCIENTIFIC_DIAGNOSTIC.md)

## Gate outcomes and next training

| Gate | Status and action |
| --- | --- |
| Integrity and 9107 | VERIFIED. Both seals, source, data bytes, selected and final checkpoints and matched source order checked. |
| Held-out feasibility | Archive available and SHA-256 matches foundation. Official CM protocol and license members exist. Extraction is unnecessary for a bounded subset. |
| Protocol | Locked for replication. Evaluation authorization is false. |
| 9207 | Neither scientific run exists. Both preflights passed in the readiness snapshot. Author runs ERM, verifies it, then runs and verifies channel. |
| 9307 | SEED_9307_PENDING_AUTHOR_AUTHORIZATION. Decide after the full 9207 pair and only if both arms can finish by 16/11 without displacing held-out or manuscript QA. Decision may not depend on favorable effect direction. |
| Medium-plus and foundation | NO-GO. FULL_FOUNDATION_NO_GO_FOR_DECEMBER_MANUSCRIPT. |

Run the following from the repository root. Execute sequentially, continuing only on PASS. For an existing interrupted run, substitute `resume` for `train` after checkpoint validation. For a sealed completed run, use `verify`.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_erm_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_erm_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_erm_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_channel_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_channel_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_channel_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium compare --experiment p10_medium_erm_s9207_v2 --peer p10_medium_channel_s9207_v2 --output outputs/jaes_v2/phase10/medium_pair_s9207_v2.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.empirical_seed_summary
```

The current engine requires ≥2 GiB available RAM, ≥3 GiB available commit, ≥3 GiB on C and ≥15 GiB plus 128 MiB on D. Run one heavy job at a time. Stop for numerical, allocation, source, checkpoint or safety failure. The monitor and safe stop options are in the existing commands. The summary emits a new JSON named by complete seed blocks and refuses overwrites. The prior 9107 pair cost about 83 and 122 minutes of logged training plus validation. [Monitor and resume](DEADLINE_TRAINING_COMMANDS.md) · [Readiness](../../outputs/jaes_v2/phase10/empirical_execution_v1/readiness.json)

## Held-out plan and measured budget

The local LA archive is 7,646,913,333 bytes, with 71,933 FLAC members and 4,462,823,332 expanded audio bytes in the evaluation directory. The authors' official database paper reports 7,355 REAL and 63,882 spoof CM trials. Archive members and protocol trials are different inventory units. Counts and exact eligible IDs must be reconciled from the official CM protocol after authorization. The local train license is ODC Attribution. Review the evaluation license before release. Held-out is within the same ASVspoof release. The attack IDs A07–A19 do not themselves prove unseen generator lineage. [Release](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45) · [Database paper](https://arxiv.org/abs/1911.01601)

The 12-source development benchmark replayed 96 scores with zero drift. It took 21.74 s, with 0.119 s average forward time, 9.78 s total transformation time and 0.45 s Mel preparation. Four locked models and four conditions imply 17,408 predictions at 1,088 sources or 8,704 at 544. Using this small sample as a planning scenario gives roughly 50 or 25 minutes of decode/preprocessing/forward work before lock checks, ZIP access, output and restarts. These are planning estimates. Resource feasibility alone can select fallback, before evaluation access.

The hash-ranked speaker round robin uses official source IDs and fixed seed 91109, selecting 256 REAL and 64 per attack, with fallback 128 and 32. Shortages stay explicit. Enforce disjoint speaker, official IDs and bytes against registered train, development and reserved identity metadata. Full execution needs a separate authorized lock with approved subset, matched selected models, source hashes, archive hash and FFmpeg hash. A read-only ZIP path avoids extraction. Per-source committed score shards support technical resume. The runner was tested on fixtures and its waveform formula matched a real development source. Official metadata parsing, complete evaluation execution and performance remain unverified on real held-out data. [Pending lock](../../outputs/jaes_v2/phase10/empirical_execution_v1/heldout_lock_pending.json) · [Benchmark](../../outputs/jaes_v2/phase10/empirical_execution_v1/benchmark.json)

## Manuscript and deadline review

**JOURNAL_CONTRIBUTION_INSUFFICIENT for submission today.** Research Paper is the appropriate target if replication and held-out evidence support the research question. An Engineering Report can be reconsidered only if the validated procedure and practical measurements constitute a substantial engineering contribution. One-seed development evidence and the current overlap with prior robustness studies leave acceptance uncertain. The new value relative to conference is the matched training intervention, source-level conventional order/constituent controls and class, attack and speaker risk. This combination still needs replication. Exact venue, proceedings status, DOI and rights agreement remain author-side checks. [Draft](../../papers/jaes_v2/journal_draft/DEADLINE_JOURNAL_DRAFT.md) · [AES guidelines](https://aes.org/publications/journal-author-guidelines/)

09–10/11: audit, protocol and readiness complete. 10–13/11: author runs 9207 and returns both sealed runs. 14–16/11: decide 9307 separately. 17–20/11: freeze the model set and authorize the held-out lock. 21–30/11: execute held-out only if gates clear. 01–07/12: analysis and figures. 08–14/12: manuscript, related work and disclosure. 15–20/12: QA, copyright and submission package.
