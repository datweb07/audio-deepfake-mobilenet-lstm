# Artifact specification

## Deliverables

- `papers/PaperID 804 final.docx`: revised LAVA manuscript derived from the submitted DOCX while preserving its Springer-style page geometry, figures, tables, equations, citations, and measured values.
- `papers/Response_to_Reviewers_final.docx`: point-by-point response following the three-column checkpoint structure of `papers/Mẫu.docx`.

## Editorial decisions

- Treat the submitted LAVA manuscript and repository evidence as scientific source of truth.
- Apply the two supplied reviews directly: add quantitative abstract results and bullet-point contributions; document the dataset, quarantine rule, preprocessing, detector architectures, training provenance, calibration, and complete clean metrics; add a dedicated limitations section and societal implications; improve references, captions, cross-references, image resolution, editable tables, and equation preservation.
- Add specificity and PR-AUC by deriving them from the already saved full-test score files, and report the existing test-set bootstrap confidence intervals and paired statistical analysis. Do not rerun or retrain any detector.
- Address requests for stronger robustness, multi-seed training, unseen corpora, physical replay, streaming, and edge-device experiments transparently as remaining work because the repository contains no such completed evidence. Narrow deployment and robustness claims accordingly instead of fabricating results.
- Incorporate only verified recent literature from 2024--2026 and retain a compact, complete reference list whose entries are cited in the manuscript.
- Preserve all experimental numbers and scope caveats: full clean test set, fixed 100-recording diagnostic subset, simulated replay, external RawNet2/AASIST provenance, EfficientNet warm-up status, desktop-CPU timing, and exploratory Pareto interpretation.

## Layout requirements

- Preserve the manuscript's A4, single-column formatting and native Word equation objects.
- Preserve all existing figures and tables unless a reviewer-relevant adjustment is needed.
- Response document follows the checkpoint structure and provides a point-by-point reply to all six Reviewer 1 comments, Reviewer 2's general requirements, and ten specific suggestions, with exact edit locations and explicit limitations where new experiments were not performed.
- Render both final DOCX files to PDF/page images and inspect every page before delivery.
