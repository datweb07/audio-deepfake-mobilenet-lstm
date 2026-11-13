# Conference-to-journal review

Internal author review, 10 November 2026. The path named in the request, `papers/PaperID 804 final.docx`, hashes to `ea9d0193f52064cbbae68c2f21b67d2405faa286fafb50fe71abf946e51509e8`. The expected protected hash `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073` belongs to `papers/PaperID_804_final.docx`. Both copies were inspected. The hash-matching copy is the evidentiary reference. The final reviewer response hashes to `584937afb09a80f1b83e7f9812bfb9a48cf1c9c1861612525122a1828fbaa911`. The document text, editable tables, formula objects or surrounding definitions, and the nine embedded panels of the protected copy were consulted. No review text or conference figure is reproduced in the journal.

## Original conference contributions

| Claimed contribution | Implementation and evidence | Boundary |
| --- | --- | --- |
| Artifact-traceable clean comparison | Exact SHA-256 grouping, cross-label conflict quarantine, common FAKE score orientation and per-artifact threshold. Four locally fitted Mel-sequence CNN–LSTM models and two external waveform references. Clean test contains 2,737 sources and reports class errors, ranking, EER and uncertainty. | Provenance gaps prevent source, speaker and generator independence claims. Training and calibration differ across artifacts. |
| Matched diagnostic stress benchmark | Same 100 recordings per artifact. Four AWGN levels, MP3/Opus/AAC and a simulated replay filter. Aggregate stress figures show low-SNR losses for the strong lightweight artifacts. | Neither full-test stress nor physical replay. Paired clean/stress scores were unavailable for individual error transitions. |
| Offline runtime and exploratory Pareto comparison | Isolated single-thread desktop CPU workloads and explicit timing boundaries. MobileNetV3 and ShuffleNetV2 were non-dominated for the reported diagnostic objectives. | One host and one workload. Pareto membership is artifact-specific. No causal streaming or edge verification. |

The visual sequence covers the registry, checksum-controlled data flow, lightweight architecture/provenance, ROC/DET, AWGN and condition-level failures, and an exploratory Pareto chart. It supports the conference benchmarking claims and does not identify a learning mechanism.

## Reviewer concern traceability

| Concern | Conference response | Adequacy | Journal evidence | Remaining limitation and action |
| --- | --- | --- | --- | --- |
| R1.1 Data provenance, counts, balance, preprocessing, conflict quarantine | Reported observed counts, documented exact checksum grouping and unknown provenance. | Integrity rule resolved. Provenance gap remained. | Registered ASVspoof LA partitions, source and speaker IDs, explicit view counts. | Describe benchmark provenance and lineage limits separately. |
| R1.2 Additional metrics and intervals | Added metrics and bootstrap intervals from stored test scores. | Adequate for fixed test. | Per-seed paired speaker bootstrap and condition/class counts. | Keep seven/nine speaker support and two seeds explicit. |
| R1.3 Architecture, weights, features, hyperparameters and calibration | Documented heterogeneous artifacts and external checkpoint provenance. | Documentation improved. Fair intervention remained unavailable. | Matched source order, initialization, architecture, numerical backend and clean checkpoint rule. | Avoid cross-architecture claims and keep 0.5 fixed. |
| R1.4 More stresses and unseen attacks | Reported executed SNR, codec and simulated replay on fixed subset. | Partial. | Per-source order interventions, class/attack/speaker transitions and component diagnostic. | Held-out scoring pending and same release. No physical replay. |
| R1.5 Multiple seeds and statistical tests | Added paired fixed-test tests and intervals, with no new fits. | Partial. | Two matched fit blocks and within-seed intervals. | No seed-population inference. |
| R1.6 External corpus, physical replay, streaming and edge resources | Narrowed deployment claims to desktop offline evidence. | Open scientifically. | Logged CPU work and prospective within-release lock. | Keep all four external/deployment claims out of results. |
| R2 general Evidence, prose, references and captions | Editorial check and cross-references. | Editorial response. | Reference audit, claim map and shared LaTeX source. | Review both rendered layouts before submission. |
| R2.1 Quantitative abstract | Added clean, stress and runtime figures with scope. | Addressed in conference. | Abstract reports two blocks and adverse subgroup effects. | Add held-out estimates only after verified scoring. |
| R2.2 Contribution list | Listed three benchmark contributions. | Addressed in conference. | Introduction distinguishes benchmark and new controlled evidence. | Avoid treating rewriting as novelty. |
| R2.3 Recent literature | Added cross-domain, calibration and replay literature. | Editorial coverage improved. | Active bibliography retains published evidence and excludes unverified preprints. | Refresh nearest published work before submission. |
| R2.4 Limitations | Dedicated subsection before concluding remarks. | Addressed editorially. | Dedicated internal/external validity and engineering limits. | Carry these limits into abstract and conclusion. |
| R2.5 Springer page target | Compressed the source and deferred final Word pagination. | Conference-specific. | Separate A4 lecturer layout and authentic JAES layout. | JAES publication page count is a separate measure. |
| R2.6 Image quality | Native source images and combined panels. | Conference response reports resolution. | Existing evidence-derived vector journal figures. | Inspect rendering and reuse no conference figures without clearance. |
| R2.7 Editable tables | Native Word scientific tables. | Addressed in source. | Shared editable TeX tables. | Retain underlying sealed values and denominators. |
| R2.8 Standard math editor | Native Office Math. | Addressed in source. | Shared numbered TeX equations. | Inspect typesetting in both builds. |
| R2.9 Contribution list repeated | Combined with R2.2. | Duplicate editorial request. | Same response as R2.2. | Avoid duplicate list. |
| R2.10 Practical and societal implications | Discussed defensive review and bounded scope. | Explanation, not deployment evidence. | Class-conditional costs and workflow implications. | Do not infer field workload, fairness or edge readiness. |

## Extension assessment

The journal materially extends the original questions through a controlled comparison of training-view policy in two matched initialization blocks. The new score grid links class, attack, speaker and condition to individual recordings. The two explicitly reversed processing chains and retrospective matched components support a more focused sensitivity diagnosis than the conference's aggregate stress table. The checkpoint and bootstrap analyses improve traceability, and the locked 1,088-source plan provides a path to independent partition testing within the release. The plan is pending evidence. It does not establish a new robust-learning objective, external-corpus transfer or an independent forensic cue.

The protected copy lists the conference authors and UEH affiliation. Do not reuse them as a confirmed journal author list. Acceptance is author-reported. ICAIN association, DOI, proceedings publication date, signed terms and reuse rights remain pending author confirmation. See [conference disclosure](CONFERENCE_EXTENSION_DISCLOSURE.md).

V7 adds a fixed-epoch size comparison on the shared development cohort. The larger seed-9207 pair doubles the source count and optimizer budget. The original locked evaluation retains four 640-recording models. The new comparisons use journal evidence and the conference already included aggregate robustness on the fixed 100-recording stress panel.
