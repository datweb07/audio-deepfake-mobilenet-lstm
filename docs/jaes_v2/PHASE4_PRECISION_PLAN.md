# Phase 4 precision plan — development variance only

Target planning effect: **.10** absolute change in a paired class-error contrast; target 95% CI half-width **.05**. Neither was achieved as a population guarantee. The planner reads executed development contrasts only and does not open confirmation data.

`precision.plan_precision` averages recordings within each original source-text cluster, then computes variance across cluster means. It reports source/recording counts, an illustrative Student-t half-width, an unbalanced one-way method-of-moments ICC/design effect and variance uncertainty under explicit normal/iid-cluster assumptions. Cluster-mean variance, rather than file count, drives required cluster planning. Negative ICC is reported and only clipped for a nonnegative design-effect sensitivity.

For primary edge-repacking change, observed development has 16 FAKE recordings and four source-text clusters:

| Artifact | Cluster-mean variance | Illustrative current t95 half-width | Required clusters at pilot variance | At normal-theory variance upper bound |
|---|---:|---:|---:|---:|
| MobileNet | .015625 | .19890 | 27 | 337 |
| ShuffleNet | .0989583 | .50056 | 155 | 2,117 |

These are **conditional extrapolations**, not sample-size promises, independent-binomial calculations or achieved power. Four bounded/discrete cluster means do not validate normality, stable variance or independence, and the large upper-bound sensitivity shows the uncertainty. More utterances from the same four speakers cannot substitute for the planned independent groups.

Zero development variance yields no required-cluster estimate; it does not justify zero observations or population equivalence. The machine output marks INSUFFICIENT_INDEPENDENT_GROUPS. Confirmation would first require an estimand-specific source-matched variance pilot, sufficient independent source groups and a practical-effect/multiplicity lock. The current within-record nuisance variance cannot be recycled as proven variance for an unexecuted matched-generator effect.

Reproduction: `phase4 aggregate` rebuilds precision.json from sealed development scores; `scripts.jaes_v2.plan_phase4_precision` independently accepts a sealed development run, target effect and half-width and writes a new planning JSON. phase4_report_v1/precision_planning.json retains both artifacts/all policies. Exact commands are in PHASE4_NEXT_COMMANDS.md. No planning operation reads or changes confirmation labels.
