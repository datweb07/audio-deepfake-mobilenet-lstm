# Phase6 executed results

New speaker-balanced exploratory cohort, one seed, fixed operating threshold. No direct quality comparison to the different Phase5 cohort is valid. Scores are bounded, uncalibrated; NLL/Brier/ECE/classwise diagnostics were computed without fitting or opening reserved calibration.

| Run | Clean REAL FA | Clean FAKE miss | Mean3-stress FAKE miss | Mean3-stress REAL FA |
|---|---|---|---|---|
| p6_p0_b0_v1 | 0.0 | 0.0 | 0.9285714285714285 | 0.0 |
| p6_p0_b1_v1 | 0.0 | 0.0 | 0.9523809523809524 | 0.0 |
| p6_p0_b2_v1 | 0.03571428571428571 | 0.0 | 0.21428571428571427 | 0.2619047619047619 |
| p6_p0_b4_control_v1 | 0.0 | 0.0 | 0.6071428571428571 | 0.059523809523809514 |
| p6_p0_b4_v1 | 0.0 | 0.0 | 0.5357142857142857 | 0.07142857142857142 |
| p6_p0_rb_v1 | 0.0 | 0.0 | 0.3452380952380952 | 0.2380952380952381 |
| p6_p1_b0_v1 | 0.0 | 0.07142857142857142 | 1.0 | 0.0 |
| p6_p1_b1_v1 | 0.0 | 0.07142857142857142 | 1.0 | 0.0 |
| p6_p1_b2_v1 | 0.0 | 0.10714285714285714 | 0.32142857142857145 | 0.059523809523809514 |

B1 does not solve noise/order failure. P1 B1 matches P1 ERM because half-gain produces identical P1 features; this is an informative invariance control, not a new gain-learning result. B2 and finite RawBoost reduce stressed misses but increase stressed false alarms. B4 improves the prespecified equal-view comparison on this cohort but remains ordinary consistency. Its single-seed speaker bootstrap is descriptive and does not quantify independent-training variability. Clean saturation here is especially limited by A01–A04-only coverage.

All per-condition classwise metrics, flips, ranking and descriptive group results: outputs/jaes_v2/phase6/reports_v1/summary.json. Retention-proxy correlations are post-hoc, repeated-view associations rather than evidence of cue preservation. Neural inference was not used during report regeneration.

The first concurrent cache/fine-tune attempt failed under observed memory pressure. V2 cache and microbatched sanity supersede it; original failure evidence is retained. The first regression run exposed a Phase5 fixture depending on the real temp-volume512MiB allowance: with C: below that allowance, the tiny fixture failed correctly at the production storage guard. The fixture now mocks free space for its byte-size ZIP; production storage enforcement is unchanged. Both regression attempts are retained.
