# JOURNAL FIX03 quality and submission audit

**14 November 2026. Ready for supervisor review.** Author-controlled submission fields remain open.

## Build and layout

Compiled the existing shared LaTeX sources with local pdfLaTeX and BibTeX. For each of `main_review` and `main`, ran `pdflatex -halt-on-error -interaction=nonstopmode <entry>.tex`, `bibtex <entry>` and three further pdfLaTeX passes. The review PDF has 35 A4 pages in one column. The JAES draft has 26 pages in the two-column template. Both have 14 figures, 12 tables and 26 cited references. No undefined labels, citation warnings or overfull boxes remain in the final LaTeX logs. BibTeX retains the existing bibliographic missing-month warnings. No references were added. Both entry points were rebuilt from the extracted release ZIP. The resulting page text and rendered pixels match the delivered PDFs.

Rendered and inspected the review Abstract, R08 Methods (§5.8, p. 15), R08 Results (§6.5.1, p. 24), Table 12 (p. 26), and declarations (p. 33). Checked the R08 methods and table in the JAES layout (pp. 9, 19). The new table fits in both builds. No empty pages were detected. The baseline figure, table and numbered equation cross-references retain their original numbers, including Figure 12 and Table 1. The existing history, risk guard failures, 80-to-81 direction, adverse attack findings, pointwise caveats, order scope and UEH acknowledgement were checked against the compiled text.

All 18 printed DOI strings were checked against the archived reference audit. The reused In-the-Wild reference was checked against the primary proceedings (Müller et al., Interspeech 2022, DOI `10.21437/Interspeech.2022-108`). Existing references without DOI retain their source links. The conference-extension DOI and rights still require author records. [ISCA primary record](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.html)

## Scientific checks and preservation

The independent CSV audit reproduced all six ROC-AUCs, EERs, confusion matrices and FPR/FNR, plus paired transitions and descriptive mean/sample SD. All 3,074 sealed artifact hashes matched. Historical integrity rechecked eight training seals and 1,088 within-release shards. The 405 freeze hashes and all three FIX02 hashes matched the recorded values. No training, inference, threshold tuning or change to existing approvals or sealed data took place. [R08 audit](R08_FINAL_SCIENTIFIC_VERIFICATION.md) · [Supervisor audit](SUPERVISOR_12_REQUIREMENTS_FINAL_AUDIT.md) · [Source change log](JOURNAL_FIX03_CHANGELOG.md)

Public GitHub API access returned HTTP 200 without authentication. No files under the repository output tree are tracked. The Code Availability statement therefore identifies the source separately from the local artifacts and third-party audio needed to reproduce the work. [Project repository](https://github.com/datweb07/audio-deepfake-mobilenet-lstm)

## Release files

- Review PDF SHA-256 `a2b39973559870a63515287e6f65e778d2c5890b7f61d6935cc3a2aa535566ed`.
- JAES layout draft SHA-256 `90960994c2a5cd33c5a2d3469b9cbc79a89727a1772e2454b21db4917348c5ad`.
- Overleaf source ZIP SHA-256 `13e5f4dc4aafb6ee00b31a8e50dc4c822d68b826ce6f16cd19b778bc7dcf6ea3`.

## Author checklist before submission

- Seek supervisor feedback on FIX03 and the bounded diagnostic scope. The current results support a Research Paper for review, subject to author and portal confirmation.
- Confirm author order, affiliations, corresponding author, email, ORCIDs, contributions and biographies of up to 150 words. Complete the metadata in `main.tex`.
- Verify UEH support and funding identifiers, if applicable, and complete the conflict-of-interest declaration.
- Confirm the conference publication record, DOI if assigned, reuse permissions and disclosure in the cover letter. Preserve Table 1.
- Complete the JAES AI-assistance disclosure for substantive use, including actual tool, provider, model/version, dates and role. Review and accept responsibility for the final text and evidence. The review draft has no finalized disclosure.
- Review the availability statements and the distribution rights for scores and subset metadata. Share aggregate evidence as permitted. Keep raw WAV, frozen models and private approvals local unless specifically cleared.
- Use the current JAES template for the submission PDF and editable source. Supply approved supplemental materials with matching build and hashes. The author-controlled fields and declarations should be completed before creating that package. [JAES Author Guidelines](https://aes.org/publications/journal-author-guidelines/)
