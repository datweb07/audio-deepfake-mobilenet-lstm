# Prior contributions and claim boundaries

The accepted Word manuscript already claims:

1. A checksum-group-disjoint and artifact-traceable clean evaluation contract, including score semantics and provenance disclosure.
2. Matched fixed-subset diagnostics for noise, codecs and simulated replay, with aggregate failure patterns.
3. Single-thread desktop timing plus exploratory three-objective Pareto selection.

It evaluates six heterogeneous artifacts, not six equivalently trained architectures. ShuffleNet clean F1/ROC-AUC/EER are 0.9824/0.9929/1.46%; MobileNet E2E latency is 43.81 ms on the reported conference host. These values must not be copied into a journal abstract as new results.

The accepted Pareto objective minimizes `(diagnostic EER, -mean stressed F1, E2E RTF)`; MobileNet and ShuffleNet are non-dominated. Statistics are conditional on one checkpoint per model. McNemar/Holm and test-sample bootstrap are already present and are not new journal contributions.

No accepted-paper evidence establishes unseen-generator/corpus performance, physical replay, causal streaming, edge-device performance, calibrated real-world probabilities or repeated training-seed stability. Future validation may fill a limitation without automatically becoming a new scientific contribution.
