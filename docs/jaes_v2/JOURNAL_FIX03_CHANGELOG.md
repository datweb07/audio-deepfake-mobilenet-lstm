# JOURNAL FIX03 change log

Baseline Git HEAD `f458be77caad0242906cc58ad801c855d6fbb466`. The worktree was clean. The retained source snapshot rebuilt to the same extracted text on all 33 pages as the archived FIX02 LaTeX PDF. Against the approved FIX02 PDF, the baseline has 99.846% normalized word-token overlap, reflecting the earlier LaTeX synchronization and corrections to the class-error terms, per-speaker recording units and 640-recording source-visit scope. The frozen [FIX02 source ZIP](../../papers/jaes_v2/jaes_manuscript/build/JOURNAL_FIX02_Overleaf_Source.zip) is the portable snapshot.

| Class | Location | Change |
| --- | --- | --- |
| R08_METHODS_ADDITION | Abstract and §5.8 | Add the release, locked cohort, six reused 640 fits, P0 native view, FAKE score, threshold and limited evidence scope. Abstract retains the existing development, size and held-out findings. |
| R08_RESULTS_ADDITION | §6.5.1 | Add all paired AUC directions, absolute AUC and FPR, and the adverse 9207 miss result. Keep the six earlier Results subsections and their reported populations. |
| R08_TABLE_OR_FIGURE | Tables 1 and 12 | Add the journal-side scope to Table 1 and report all six AUC, EER and classwise counts/rates in Table 12. The 14 figures and their numbering stay intact. |
| R08_DISCUSSION_UPDATE | §7 | Distinguish relative paired gains from absolute external performance and state the scope of distribution-shift inference. |
| R08_LIMITATION_UPDATE | §§6.5 and 8 | Replace obsolete “untested” language with the completed native-view diagnostic and its lineage, condition, calibration and interval limits. |
| R08_CONCLUSION_UPDATE | §9 | Add measured external evidence while preserving RQ1–RQ3, the guard decisions, attack heterogeneity and within-release order scope. |
| R08_REPRODUCIBILITY_UPDATE | Shared back matter | Add selected-row hash, protocol provenance, Data Availability and Code Availability. Keep the UEH acknowledgement. |
| R08_REFERENCE_UPDATE | §5.8 and Data Availability | Reuse the existing Müller et al. citation and link to the official dataset releases. The bibliography and 26 numbered citations are unchanged. |
| SUPERVISOR_CONSISTENCY_CORRECTION | Reports and claim map | Reaudit R01–R12, preserve the historic responses, add the FIX03 section locator and tie R08 claims to the protocol, subset and seal. |
| NECESSARY_LAYOUT_CHANGE | Both builds | Allow line breaks in hashes. Pagination and float placement change. The review is 35 pages. The two-column draft is 26 pages. |

Source-level comparison confirms the Introduction, Related Work, Problem Formulation, Methodology, existing figures and bibliography are byte-identical to the editable baseline. The other changes are listed above. The review PDF was recompiled from `main_review.tex` and the secondary draft from `main.tex`. No PDF content was patched.
