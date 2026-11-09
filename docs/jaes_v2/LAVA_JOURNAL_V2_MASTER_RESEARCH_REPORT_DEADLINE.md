# LAVA Journal V2 research sprint handoff

09/11/2026, Asia/Novosibirsk. Manuscript target 20/12/2026. `TRAINING_ENGINE_VERSION = FROZEN_PHASE10`. The completed seed 9107 pair supports a development finding about classwise and attack-specific trade-offs. The learning-method gate remains closed.

## Evidence from Phase 1 to Phase 10

| Stage | Scope and relevance |
| --- | --- |
| Conference | Six heterogeneous detectors on 2,737 checksum-disjoint recordings. ShuffleNetV2 led clean discrimination, MobileNetV3 was fastest on the measured desktop pipeline, and the fixed 100-recording stress diagnostic was exploratory. |
| Phases 1–2 | Source-paired order interventions and checks on noise insertion, headroom and delay. |
| Phases 3–4 | External development and provenance audit. Generator counterfactual lineage remained insufficient. |
| Phases 5–6 | ASVspoof 2019 LA foundation and bounded preprocessing, augmentation and fine-tuning pilots. |
| Phase 7 | All-attack A/B development, equal-view and consistency experiments. Ordinary consistency did not pass guards across the three historical blocks. |
| Phase 8 | Weighted BCE and Group-DRO controls, tiny lifecycle tests and two exploratory medium runs. Historical channel arithmetic changed midrun. |
| Phase 9 | Initial numerical tests, medium-plus selection and mechanism screen. |
| Phase 10 | Frozen CPU engine qualified on real data, with a new run contract and unchanged historical evidence. |

The audit reran historical integrity checks for 2,376 protected files with three approved amendments, 28 Phase 9 inventory entries and both historical medium seals. It verified passing artifacts for all nine numerical fixtures, the 9-update restart at 5, 2,576 matched validation scores with zero flips, and the 100-update pilot restart at 55. The pilot and lifecycle comparisons match 17 logical fields. Archive metadata hashes can differ. This exact-resume evidence is from the qualification jobs. The scientific ERM resumed successfully under its own contract, without an independent uninterrupted ERM replay. Conference SHA-256 remains `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`. [Phase 10 master](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE10.md) · [Sprint audit and gate hashes](../../outputs/jaes_v2/phase10/deadline_s9107_audit_v1.json)

## Prospective matched seed 9107

The checked CLI comparison passed identity and contract matching. Source IDs appeared in the same training order across all 840 updates. Both runs used P0, the same initial weights, fixed Adam and schedule, 640 sources, 21 epochs, the same selection policy and 368 development recordings. ERM uses clean. Channel draws from the registered single-view bank of clean, half-gain, noise10 and Opus24. No special order supervision was used. The two runs are complete, final checkpoints and their seals verified, and no resume is required.

| Evidence | ERM | Channel |
| --- | ---: | ---: |
| Sealed files | 264 | 262 |
| Selected epoch/update | 14 / 560 | 19 / 760 |
| Selected / final epoch clean NLL | 0.11856 / 0.15642 | 0.14766 / 0.15013 |
| Mean train loss, first → last epoch | 0.54803 → 0.01056 | 0.63532 → 0.17931 |
| Clean A AUC / EER | 0.98724 / 4.46% | 0.98143 / 4.91% |
| Clean A false alarm / miss | 3/112 / 6/112 | 3/112 / 9/112 |
| Clean B false alarm | 19/144 | 31/144 |
| Stress A macro AUC | 0.67148 | 0.80264 |
| Stress A REAL false alarm / FAKE miss | 54/336 / 222/336 | 45/336 / 155/336 |
| A05 / A06 stress miss | 48/54 / 41/54 | 50/54 / 45/54 |
| Logged train / validation time | 42.6 / 40.2 min | 82.9 / 39.1 min |
| Peak process RSS among recorded invocations | 645 MiB | 794 MiB |

Timings sum instrumentation across invocations and include update work that may have been replayed after interruption. They omit idle intervals. Clean NLL trajectories and lower within-run train loss indicate learning under each exposure regime. The different views prevent a direct comparison of train-loss magnitudes as if they were the same task. Late ERM clean NLL rises above its minimum. No independent generalization conclusion follows from the trajectory.

The guard compares **the selected endpoints** and reports `NO_FEASIBLE_CHECKPOINT` for that comparison. A clean miss rose +2.68 pp, B clean false alarm +8.33 pp and A06 stress miss +7.41 pp. This does not enumerate feasibility across every channel checkpoint. Stress views repeat source recordings, and a single seed does not establish uncertainty over training. [Compare JSON](../../outputs/jaes_v2/phase10/medium_pair_s9107_v2.json) · [ERM export evidence](../../outputs/jaes_v2/phase10/p10_medium_erm_s9107_v2/scientific/result.json) · [Channel export evidence](../../outputs/jaes_v2/phase10/p10_medium_channel_s9107_v2/scientific/result.json)

## Release scope and manuscript

No trainer changes were needed. Baseline seeds 9207 and 9307 remain registered, with 9207 first when live resources clear. Current preflight is blocked by commit pressure. Use the same checkpoints, audit policy and resource gate. Multi-seed empirical comparison has scientific value even if C2 stays closed. Evaluation and calibration remain closed until the method, comparison family, exact held-out cohort, provenance and policy are frozen.

The sprint draft separates the matched development result from the historical exploratory result. It adds order-discordant within-source counts, exact class/attack support, a guard interpretation and reproducibility artifacts. C1 is a development finding. C2 remains unestablished and C3 deferred. The accepted manuscript was preserved. The manuscript still needs matched seeds, held-out evidence if feasible, author disclosure, conference venue and copyright verification. [Sprint manuscript](../../papers/jaes_v2/journal_draft/DEADLINE_JOURNAL_DRAFT.md) · [Decision and schedule](DEADLINE_RESEARCH_DECISION.md) · [Protocol](DEADLINE_NOVELTY_AND_EXPERIMENT_PROTOCOL.md) · [Author commands](DEADLINE_TRAINING_COMMANDS.md) · [AES policy](https://aes.org/publications/journal-author-guidelines/)
