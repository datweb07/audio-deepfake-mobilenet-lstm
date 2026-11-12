# Training readiness for the supervisor revision

## Seed 9307, registered medium pair

Hypothesis. Under the same 640-recording cohort and registered view-policy intervention, a third independent initialization provides an additional observation of the sign and magnitude of the stress AUC gain, clean and stressed class risks, and A05/A06 deterioration. Report all three fitted seed contrasts and descriptive mean and sample SD with ddof=1. Limit comparison to development A/B under the frozen contract. Previously opened held-out results stay the original four-model comparison.

Two resource/data preflights passed. Registry SHA-256 is c43a8de97462a77f393d58f17484de6321d7021cd765dd32debf998642631925 and the source manifest is 029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319. Exact-resume and numerical proof are inherited from the existing released engine and were checked by the registered preflight, including 100-update interruption/restart. The 9307-specific smoke exercises the real training input and both gradient stages without optimizer updates. It is separate from the full trial. The author must approve the pair separately.

Measured medium costs imply about 3.4 hours per pair before overhead. Allow about 4–5 hours plus scientific review time. Retention observed about 0.88 GiB per arm. Plan about 2.0 GiB above the existing D safety reserve for the pair and verify fresh resources before each arm. Do not start both arms in parallel. The 11 November snapshot had about 19.5 GiB D free, 8.9 GiB C free, 5.9 GiB available RAM and 9.3 GiB available commit. Per-run preflight and runtime monitors control current RAM, commit and disk safety. Set a review cutoff before 25 November for any newly authorized run.

After separate approval and smoke PASS, run sequentially from the repo root. If either preflight fails, address the reported resource or integrity blocker first.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_erm_s9307_v2
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_erm_s9307_v2
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_erm_s9307_v2
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_channel_s9307_v2
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_channel_s9307_v2
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_channel_s9307_v2
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium compare --experiment p10_medium_erm_s9307_v2 --peer p10_medium_channel_s9307_v2 --output outputs/jaes_v2/phase10/medium_pair_s9307_v2.json
```

Use the same registered ID with mode resume after interruption when a valid committed checkpoint exists. Use mode monitor with the same ID for read-only status. Stop on a failed resource gate, nonfinite or invalid gradient, incompatible resume contract, invalid seal or incomplete score grid. Preserve an unfavorable result. Send result.json, scores.csv, seal.json and contract.json from each scientific folder and the comparison JSON after both verifies. Expected budget is 840 optimizer updates and 21 epochs per arm, with 2,576 development scores per arm.

## 1,280-recording size sensitivity

The deterministic train-only candidate retains all 640 medium sources and adds 640. Both sizes are balanced by class, and each FAKE attack quota doubles exactly. Candidate checks found 20 speakers, verified 1,280 audio-byte hashes, and zero intersections by declared ID, speaker and byte hash against the foundation development and calibration manifests and the locked held-out subset. Original-utterance/generator lineage remains bounded by the available metadata. The exact cohort hash and attack counts are in size_candidate.json.

Scientific contrast. Compare Channel-minus-ERM at size 640 versus 1,280 at the same seed and fixed update and LR budget, using an identical validation/checkpoint schedule and the same A/B. This estimates sensitivity to expanding this particular nested training pool conditional on exposure budget. Use 9207 to compare with a completed 640 pair if an isolated fixed-budget profile can reproduce its update and validation schedule. At fixed 840 updates, 13,440 total visits are retained and average visits per source fall from 21 to 10.5. A 21-epoch larger run doubles updates to 1,680 and cannot isolate source diversity from compute and optimizer trajectory. Neither estimand represents every possible larger cohort.

The current loader only accepts the original medium or foundation manifest, and the released scientific preflight ties 840 updates to cohort size and epoch schedule. A 1,280 profile must be isolated and must pass parsing, warm-up/partial numeric checks, interrupted versus continuous exact resume, validation and export, and retention/peak-resource checks. Candidate status is data-qualified only. No train, resume or verify CLI for this profile is released. As a planning bound, the 21-epoch alternative would require about 3.5 GiB checkpoint retention across both arms and roughly 5.5–7 hours based on observed medium rates, before extra verification and changes to I/O. Budget and isolation design need qualification before requesting approval. No 2,560 or full-foundation runs are planned for this revision.

**Decision:** SEED_9307_PAIR_READY_FOR_AUTHOR_APPROVAL. Both configurations produced the same initial weight hash and first-epoch source-order hash. Initialization differs from 9107 and 9207. Warm-up and partial gradients were finite and nonzero on the smoke inputs. No optimizer update was taken. This is release qualification and not an experiment result.
