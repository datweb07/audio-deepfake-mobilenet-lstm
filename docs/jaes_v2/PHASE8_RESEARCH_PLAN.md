# Phase8 research plan — 31 October 2026

The starting observation is a conflict between REAL false alarms and FAKE misses, not established success of consistency. Phase7 H3−H2 stressed miss effects were −.095238, +.062500, +.020833; mean −.003968, seed SD .081741. All three blocks failed the paired clean guardrails. Its one-epoch F1 improved clean AUC but worsened stressed misses. These results motivate measuring both classes separately.

C1 remains SUPPORTING_EMPIRICAL_AND_ENGINEERING_CONTRIBUTION. Ordered-channel controls contribute an auditable empirical protocol, but additional bitrates and plots do not distinguish it from RADAR/AT-ADD. C2 is the central research question: can a genuinely distinct mechanism improve both classwise stressed risks with acceptable clean risk? C3 remains DEFERRED until a credible C2 and independent calibration exist.

Phase8 first reviews nearby objectives, then specifies M1 weighted BCE, M2 group robustness, and proposed order-aware M3 variants. Algebraic reduction is a rejection criterion. No M3 loss is implemented without TECHNICALLY_DISTINCT_CANDIDATE status. The current candidates do not pass; see PHASE8_NOVELTY_DECISION.md and PHASE8_METHOD_REDIRECTION.md.

Engineering proceeds regardless: a separate trainer supports fresh ImageNet initialization, deterministic sampling/dropout, multiple epochs/seeds, head training and warmup plus actual CNN fine-tuning, Adam checkpointing, bounded resource use, declared checkpoint selection and exported-model evaluation. Tiny pilots are implementation evidence. The author launches medium/full research runs explicitly; Phase8 does not automatically execute them.

Use the existing medium cohort and foundation train/model-selection-development manifests. Never parse the reserved 7,341 calibration records or official evaluation. A01–A06 are seen attack families; generator checkpoint and original source lineage remain unknown. MLAAD-tiny and all development channel compositions are already exposed research data, not untouched confirmation.

Success criteria are separate: implementation readiness, full-training lifecycle readiness, scientific readiness. A successful trainer or a favorable tiny score cannot establish three international contributions. The final evidence-generated master report records the actual gates and blockers.
