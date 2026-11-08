# Phase 9 — Validation cost and checkpoint policy

## Measurements

| Scope | Decode | Transform | Mel | Forward | Metric | Other |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Phase 9 four-view parity probe | 0.022 s | 0.876 s | 0.092 s | full 0.499 s, segmentwise 0.385 s | not timed | Includes cold tracing. Input timers include preceding gradient fixture preparation. |
| Historical ERM final invocation | 2.64 s | 406.57 s | 42.71 s | 415.90 s | 0.051 s | 868.91 s validation, 877.65 s invocation, 0 updates |
| Historical channel final invocation | 5.29 s | 243.92 s | 32.89 s | 305.40 s | 0.052 s | 577.52 s validation, 610.17 s invocation, 3 updates |

The final invocation counters cover all work recorded by that invocation, including any training input work. No lifetime validation or wall-clock totals were logged. Historical peak process RSS was 548/622 MiB for those invocations. The Phase 9 probe used matching input tensors and warmup weights across engines, maximum absolute score drift 4.172e-7 and zero flips out of four at 0.5. The spot check gives no cohort-level AUC or checkpoint-selection parity. [Timing and score JSON](../../outputs/jaes_v2/phase9/parity_v1.json) · [Historical reaudit](../../outputs/jaes_v2/phase9/medium_reaudit_v1.json)

## Prospective schedule

| Tier | Fixed role | Views and outputs |
| --- | --- | --- |
| A | Each epoch, 224 development A sources | Clean BCE, FA and miss at 0.5, 112/class. Log every epoch. Clean BCE can nominate a checkpoint for an exploratory baseline. |
| B | Every third epoch and final epoch, identical epochs for all matched methods | A clean, B clean (144 REAL), three stress views on A. Evaluate registered guards and stress objective only on these checkpoints. A05 and A06 each have 54 stress views of 18 sources. |
| C | Selected export once, fixed backend | Seven views on all 368 development sources, 2,576 scores. Include score hashes, class metrics, per-attack/speaker, flips and end-to-end runtime. |

For a prospective guarded comparison, seal the matching ERM reference, including comparison epoch IDs, score cohort and role hashes, before candidate inspection. For each guard g, require E_g/n_g − E_ref,g/n_g ≤ δ_g. Working protocol for prospective review uses +2 percentage points on A clean FA, A clean miss and B clean FA, and +5 pp on A05 and A06 stressed miss. These are absolute differences in risk against the matching reference. Clean A absolute caps remain 10% FA and 15% miss in the legacy contract. Both absolute and relative guard types must pass in the prospective integration. Among feasible scheduled checkpoints minimize (A stressed FA + A stressed miss)/2. Use clean A BCE then earliest epoch to break exact ties. Missing support or no feasible checkpoint halts selection. Keep threshold p(FAKE)≥0.5. Group counts come from scores with exact source-view keys and fixed conditions.

The tested Phase 9 helper implements the relative count comparisons and stress objective. Score-key binding, absolute caps, cadence and ranking across epochs still need integration in a separately versioned trainer. With the historical endpoints as an illustrative audit, channel fails three proposed relative constraints: B clean +7.64 pp, A05 +12.96 pp, A06 +33.33 pp. This is a retrospective application for engineering illustration.

`VALIDATION_PARITY_PASS=NO`. Next compare full and optimized forward paths on all 2,576 identical views at multiple weights, including near-threshold scores. Measure transform, decode, mel, cache, graph cold/warm, forward, metric, checkpoint and export. Compare ordered scores, binary flips, all guards and winner identity. Cache only after proving identical outputs under content-addressed source, transform, seed, preprocessing and environment keys and a bounded footprint. Keep full scoring once per selected model.
