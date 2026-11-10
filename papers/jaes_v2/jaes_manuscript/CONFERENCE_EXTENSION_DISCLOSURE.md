# Conference-extension disclosure draft

The journal article extends the accepted conference manuscript *LAVA: A Lightweight Benchmarking Framework for Robust and Efficient Deepfake Voice Detection*. The conference work compared six detector artifacts with different training and provenance, including external RawNet2 and AASIST references. It evaluated clean discrimination, a fixed stress diagnostic, and desktop latency. The present article retrains one architecture under a matched augmentation intervention, adds two matched initialization blocks, and reports source-linked order, class, speaker, and attack errors with paired speaker-cluster uncertainty. The official held-out evaluation is pending.

The protected accepted file identifies Quoc Hung Nguyen, Khac Anh Tuan Phan, Thanh Dat Lai, Tan Khiem Nguyen, and Thanh Dat Truong, with University of Economics Ho Chi Minh City affiliation. The authors should separately confirm the journal author list and current affiliations. The author previously confirmed acceptance and that the paper had not been presented or released at that point.

The public ICAIN 2026 event page lists the International Conference on Artificial Intelligence and Networking, BITS Pilani Dubai Campus, 26–28 November 2026, and planned Springer LNNS proceedings. A public co-author listing links paper ID ICAIN-804 to the title. Treat the paper-to-venue match as provisional until the corresponding author confirms it against the acceptance notice. No chapter DOI, actual publication date, executed rights agreement, or permission to reuse text, figures, and tables was verified. [ICAIN publications](https://www.icain-conf.com/publications) · [Springer proceedings index](https://link.springer.com/conference/icain)

The LaTeX draft uses newly written prose and figures/tables regenerated from the journal development evidence. The original DOCX remains protected. Finalize the reference and disclosure after checking the acceptance letter, rights agreement, and proceedings entry. Draft cover correspondence remains pending verified publication facts.

## Substantive extension matrix

| Dimension | Accepted conference work | Journal extension |
| --- | --- | --- |
| Research question | Comparison of six deployed artifacts under a common evaluation and timing interface | Controlled effect of changing the training-view policy, including whether rank gains transfer to fixed-threshold subgroup decisions |
| Training | Four locally trained artifacts and two external references with heterogeneous provenance | Four prospectively matched MobileNetV3Small–LSTM runs, two initialization blocks and identical checkpoint selection |
| Data and protocol | Checksum-disjoint test cohort and fixed stress diagnostic | 640-source ASVspoof LA training cohort, two speaker-stratified development groups, and source-matched order comparison |
| Statistical analysis | Aggregate benchmark outcomes and exploratory trade-off comparison | Paired speaker-cluster uncertainty, conditional attack and speaker errors, and explicit threshold transitions |
| Figures and tables | Clean discrimination, stress summary and desktop runtime comparisons | Newly generated rank/error contrasts, per-condition counts, attack deterioration, and speaker and order diagnostics |
| Findings | Relative behavior of heterogeneous detector artifacts | Two seed blocks show aggregate rank gains with adverse A05/A06 effects and opposite B false-alarm changes |
| Discussion and conclusions | Offline benchmark comparability and deployment-oriented trade-offs | Conditional evidence about augmentation and composition. Held-out transport and independent forensic mechanisms remain open |

The extension is substantive in experimental design and evidence. The approximate 30 percent template guidance is an editorial criterion, and this matrix is not a publisher clearance or an acceptance decision. A quantitative novelty percentage cannot be inferred from rewritten word counts. [JAES template](https://www.overleaf.com/latex/templates/journal-of-the-audio-engineering-society-jaes-template/rqsctzghzcwx)
