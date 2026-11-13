# V7 manuscript and evidence quality report

**13 November 2026. V7_READY_FOR_SUPERVISOR_REVIEW_WITH_DECLARED_LIMITATIONS.**

| Check | Result |
| --- | --- |
| Review PDF | 33 A4 pages, one column, sequential printed page numbers |
| Text length | Approximately 10,490 prose words by LaTeX-source token count, including the abstract, headings and acknowledgement and excluding floats, displayed mathematics and bibliography |
| Figures, tables and references | 14 figures, 11 tables, 26 references. The reference audit identifies 25 peer-reviewed publications and one codec standard. Cited keys and DOI strings match the prior source audit |
| Compilation | pdfLaTeX and BibTeX completed for both entry points. The official two-column compatibility build has 24 pages. Final logs have no undefined references or citations and no overfull boxes. BibTeX reports 25 inherited missing-month warnings. The bibliography bytes were preserved |
| Visual QA | Reviewed all 33 pages as renders and inspected the new table, figure and guard text at higher resolution. The final pagination and rendered changes were checked after the last source edits |
| Scientific QA | Verified all eight full run seals and selected checkpoint provenance. Replayed all four size grids of 2,576 scores on 368 development sources, matched initialization, endpoints, guard decisions, order transitions and 10,000 paired speaker draws. The replay matches the original JSON |
| Preservation | V6, both named final conference copies, conference reviewer response, bibliography, old figures and their provenance, historical source fingerprints and original four-model held-out artifacts were checked by hash. The V6 source snapshot binds to the working tree and audited V6 source hashes |
| Remaining scope | R02/R11 author confirmations and R08 external validation. See the final R01–R12 audit |

## Scientific revision

The abstract, introduction, optimization, setup, results, discussion, limitations, conclusion and reproducibility statement incorporate the one-seed size contrast. Table 7 and Figure 3 show the four selected models, and Table 1 updates the conference mapping. Table 4 identifies the original three 640-recording blocks. The original four-model held-out prediction count is 17,408. The revision used sealed scores and performed no new training or held-out scoring.

The selected larger checkpoints are ERM `u001600_e020_c000` and Channel `u001280_e016_c000`. The macro stress AUC difference-in-differences is −0.081234 with pointwise paired speaker interval [−0.142087, −0.014817]. Stressed REAL false alarms reverse from a decrease at 640 to an increase at 1,280. A clean FAKE misses and B clean REAL false alarms exceed the registered tolerances for the selected larger pair. Section 6.2 preserves the separate 9307 clean B guard failure. The V7 supplement reports paired order transitions, and the claim map records run IDs, checkpoint tags, scopes and denominators.

## Evidence and build

The revised review copy restores the UEH acknowledgement on page 31. PDF text and metadata checks and the review ZIP source scan found no drafting-tool names or assistance disclosure. The official entry point contains the submission disclosure. The scientific body matches the earlier V7 review text. Pages 31–33 were visually rechecked after the update. The original scientific verification records the original V7 build hashes. [Updated build hashes](../../outputs/jaes_v2/phase10/jaes_build_work/v7_ack/acknowledgement_verification.json)

- [V7 review PDF](../../papers/jaes_v2/jaes_manuscript/build/LAVA_JAES_Review_Draft_v7.pdf) and [Overleaf source ZIP](../../papers/jaes_v2/jaes_manuscript/build/JAES_V7_Overleaf_Source.zip). Use `main_review.tex` with pdfLaTeX and recompile from scratch. `main.tex` uses the shared content and carries the journal disclosure.
- [Verified analysis](../../outputs/jaes_v2/phase10/size1280_v1/size_sensitivity.json), [V7 supplement](TRAINING_SIZE_SENSITIVITY_V7_SUPPLEMENT.md) and [scientific verification JSON](../../outputs/jaes_v2/phase10/jaes_build_work/v7/v7_verification.json).
- [V6 source snapshot](../../outputs/jaes_v2/phase10/jaes_build_work/v7/V6_source_snapshot.zip), [initial SHA-256 manifest](../../outputs/jaes_v2/phase10/jaes_build_work/v7/before.json), [V7 review compilation log](../../outputs/jaes_v2/phase10/jaes_build_work/v7/LAVA_JAES_Review_Draft_v7.log) and [resource check](../../outputs/jaes_v2/phase10/jaes_build_work/v7/resources.json).
- [R01–R12 audit](SUPERVISOR_12_REQUIREMENTS_FINAL_AUDIT.md), [supervisor response](RESPONSE_TO_SUPERVISOR_MAJOR_REVISION.md), [R08 feasibility](EXTERNAL_VALIDATION_FINAL_FEASIBILITY.md) and [author metadata checklist](JAES_V7_SUBMISSION_METADATA_CHECKLIST.md).
