# Phase7 research plan — 31 October 2026

Execute Stage A all-attack source audit, Stage B bounded single-seed then three-seed frozen-backbone comparisons, Stage C measured one-seed representation adaptation, Stage D proxy falsification and literature/readiness review, in that order. The Phase1–6 report, inventory, source, models and outputs remain immutable. Initial snapshot: `outputs/jaes_v2/audit/phase7_before.json`.

RQ7.1 asks whether the earlier A01–A04 observations survive A01–A06 coverage. RQ7.2 tests independent training variability in H3 versus H2. RQ7.3 compares real CNN adaptation against a matched head-only schedule. RQ7.4 asks whether a distinct, falsifiable C2 mechanism exists. A negative novelty decision is valid.

Selection seed7107 fixes640 train,224 paired-class development and144 REAL-only development recordings. Each retained paired speaker has16 REAL and16 FAKE, with2–3 examples per attack. Training seeds6107/6207/6307 change initialization/dropout, order and view-choice streams; finite waveform realizations are shared to isolate training randomness. Source lineage remains unknown. Neither additional recordings nor repeated views increase the seven primary development speakers.

Primary contrast H3 minus H2: equal-speaker FAKE miss averaged over noise5 and both order chains. Fixed threshold0.5 is uncalibrated. Two-point clean FAR and clean miss delta constraints are fixed before outcomes; at112 per class this permits at most2 extra observed errors. Descriptive cluster and seed uncertainty are separate; no population significance promise. Checkpoint selection uses only paired development clean BCE. REAL-only speakers are sensitivity analysis, never selection criteria.

Shared versioned P0 cache; no automatic P1 Cartesian expansion. H0ERM, H1channel, H2equal-view BCE, H3same pair plus0.1 probabilityMSE, H4finite RawBoost. All use240 updates over6epochs. H2/H3 must match initial bytes, source order and exact view-log hashes. Two-view methods see twice the supervised views; equal epochs do not imply equal compute.

Resource gates: one heavy TensorFlow job through the existing OS file mutex; project-scoped temporary files on D; preserve15GiB reserve. Benchmark before cache; single-seed measured run evidence before later seeds. F0/F1 use a separate one-epoch40update Adam0.0001 matched pilot, microbatch1,accum16, fresh initialization. Only actual preflight can enable F1. No automatic three-seed fine-tuning or large2560/896 tier. Failures remain identifiable; exact optimizer resume is unsupported.

No new weighted loss without proxy and prior-art gates. No calibration/evaluation labels, dataset download, production environment replacement, conference retraining or cascade. A new31section Phase7 master and JSON inventory will be generated from actual score artifacts, with deterministic model-free reproduction and final historical integrity audit.
