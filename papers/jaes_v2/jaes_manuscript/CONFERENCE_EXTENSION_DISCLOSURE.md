# Conference-extension disclosure draft

The journal article extends the accepted conference manuscript *LAVA: A Lightweight Benchmarking Framework for Robust and Efficient Deepfake Voice Detection*. The conference compared six detector artifacts with different training and provenance, including external RawNet2 and AASIST references and a warm-up EfficientNet-B0. It evaluated clean discrimination with paired correctness tests and stratified bootstrap, a fixed stress diagnostic, and desktop latency. The journal adds three matched 640-recording initialization blocks and one nested 1,280-recording pair at seed 9207. Source-linked order, class, speaker and attack comparisons use paired speaker-cluster uncertainty. The locked within-release held-out evaluation is complete. The V3 paired error decomposition and selected attack highlights are post-hoc diagnostics.

The protected accepted file identifies Quoc Hung Nguyen, Khac Anh Tuan Phan, Thanh Dat Lai, Tan Khiem Nguyen, and Thanh Dat Truong, with University of Economics Ho Chi Minh City affiliation. The authors should separately confirm the journal author list and current affiliations. The author previously confirmed acceptance and that the paper had not been presented or released at that point.

The public ICAIN 2026 event page lists the International Conference on Artificial Intelligence and Networking, BITS Pilani Dubai Campus, 26–28 November 2026, and planned Springer LNNS proceedings. A public co-author listing links paper ID ICAIN-804 to the title. Treat the paper-to-venue match as provisional until the corresponding author confirms it against the acceptance notice. No chapter DOI, actual publication date, executed rights agreement, or permission to reuse text, figures, and tables was verified. [ICAIN publications](https://www.icain-conf.com/publications) · [Springer proceedings index](https://link.springer.com/conference/icain)

The LaTeX draft uses newly written prose and figures and tables generated from journal development and held-out evidence. Finalize the reference and disclosure after checking the acceptance letter, rights agreement, and proceedings entry. Draft cover correspondence remains pending verified publication facts.

## Substantive extension matrix

| Dimension | Accepted conference work | Journal extension |
| --- | --- | --- |
| Research question | Comparison of six deployed artifacts under a common evaluation and timing interface | Controlled effect of changing the training-view policy, including whether rank gains transfer to fixed-threshold subgroup decisions |
| Training | Four locally trained artifacts and two external references with heterogeneous provenance | Eight matched MobileNetV3Small–LSTM development fits, including three 640-recording seed blocks and one nested size pair at seed 9207 |
| Data and protocol | Checksum-disjoint test cohort and fixed stress diagnostic | Nested 640- and 1,280-source ASVspoof LA training cohorts, two speaker-stratified development groups, and source-matched order comparison |
| Statistical analysis | Aggregate benchmark outcomes and exploratory trade-off comparison | Paired speaker-cluster uncertainty, conditional attack and speaker errors, and explicit threshold transitions |
| Figures and tables | Clean discrimination, stress summary and desktop runtime comparisons | Newly generated rank/error contrasts, per-condition counts, attack deterioration, and speaker and order diagnostics |
| Findings | Relative behavior of heterogeneous detector artifacts | Three development blocks show stress rank gains with seed-dependent attack and classwise risks. The original two held-out pairs retain adverse A17–A19 results |
| Discussion and conclusions | Offline benchmark comparability and deployment-oriented trade-offs | Conditional evidence about augmentation and composition on the locked subset. Cross-corpus transport and independent forensic mechanisms remain open |

The extension is substantive in experimental design and evidence. The approximate 30 percent template guidance is an editorial criterion, and this matrix is not a publisher clearance or an acceptance decision. A quantitative novelty percentage cannot be inferred from rewritten word counts. [JAES template](https://www.overleaf.com/latex/templates/journal-of-the-audio-engineering-society-jaes-template/rqsctzghzcwx)

V7 adds a fixed-epoch size comparison on the shared development cohort. The larger seed-9207 pair doubles the source count and optimizer budget. The original locked evaluation retains four 640-recording models. The new comparisons use journal evidence and the conference already included aggregate robustness on the fixed 100-recording stress panel.
