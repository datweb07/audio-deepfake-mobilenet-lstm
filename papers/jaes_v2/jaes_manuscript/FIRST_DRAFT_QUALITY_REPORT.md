# First-draft quality report

| Check | Result |
| --- | --- |
| Supervisor PDF | 27 A4 pages, single column, 11 pt |
| Official-layout PDF | 18 pages, JAES two-column class |
| Main-text length | Approximately 7,815 words from the section sources. The estimate excludes tables, figure captions, standalone equations, bibliography, and disclosures. |
| Shared content | 9 figures, 7 tables and 26 cited references |
| Build and rendering | Both entry points built with pdfLaTeX and BibTeX. No missing references or citations, no overfull boxes, and no blank pages. All pages rendered for visual review. |
| Review first page | Centered title, abstract, keywords, and introduction. No author block, affiliation, review label or date. |
| Scientific contributions | Matched augmentation comparison, class- and attack-specific analysis, and source-linked order diagnostics. These are empirical contributions. |

## Evidence and limitations

- Four sealed medium runs, two initialization blocks, and a shared development cohort. The figure generator verified the SHA-256 of all four score CSVs and all four result histories against their seals. It checked the 2,576-row grids and exact source identities across methods, repeated views and seeds.
- Recomputed clean and macro stressed ROC-AUC and checked paired error counts, including all A01–A06 denominators, against the consolidated results. The A05/A06 adverse outcomes are retained. Speaker-cluster intervals apply within seed and do not estimate variability across new initializations.
- Retained vector figures are the matched aggregate comparison, attack-level changes, order-score quantiles, and speaker false alarms. Added figures show classwise error rates, speaker-cluster uncertainty, empirical ROC curves, clean-validation BCE by epoch, and source-level correctness transitions. The BCE figure uses recorded values for all 21 epochs and marks the clean-selected checkpoint. Training loss is omitted because the two arms use different input-view distributions.
- The four-condition official LA evaluation remains unauthorized for scoring. Within-release transfer, independent forensic cue retention, cross-corpus transfer, and a new training algorithm remain unestablished. The prelock and proposed lock hashes were rechecked against the recorded values. No training, evaluation inference, or protected-source edits were performed.
- The two builds share all scientific content. The supplied JAES class and bibliography styles are unchanged. The official entry retains author metadata fields for completion. References retain the prior audit and comprise 25 peer-reviewed sources and one technical standard.

## Handoff

- Send `build/LAVA_JAES_Review_Draft.pdf` to the supervisor. The official-layout PDF is available for template comparison.
- For editable review or Overleaf, provide both entry points, the two shared TeX files, `references.bib`, `jaes.cls`, `jaes.bst`, `jaes_compat.bst`, `orcidlink.sty`, `AES-Shield.png`, and the complete `sections/`, `tables/`, and `figures/` directories. Select `main_review.tex` and pdfLaTeX for the supervisor layout. See [build instructions](README_BUILD.md).
- For an independent evidence review, include this report, `review_figure_provenance.json`, `claim_evidence_map.csv`, and `reference_audit.csv`. Figure regeneration additionally needs both figure scripts and the listed local run artifacts.
- Before journal submission, confirm author metadata and biographies, conference proceedings details and reuse rights, and the final disclosure with the authors and supervisor. Keep the substantive disclosure that OpenAI Codex assisted with drafting, editing, evidence organization and figure/table code. The preparation record does not identify the exact model version. The conference reference remains pending verified publication metadata. Follow the [AES journal author guidelines](https://aes.org/publications/journal-author-guidelines/).

Decision: `FIRST_DRAFT_READY_FOR_SUPERVISOR`.
