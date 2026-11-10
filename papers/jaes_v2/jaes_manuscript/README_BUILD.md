# Build the lecturer and JAES drafts

Both entry points read `shared_metadata.tex` and `shared_body.tex`, including the same sections, figures, tables, disclosures and bibliography.

| Entry point | Format | Build filename |
| --- | --- | --- |
| `main_review.tex` | A4, single-column article, 11 pt, 25 mm margins, 1.5 body spacing for annotation | `build/LAVA_JAES_Review_Draft.pdf` |
| `main.tex` | Authentic JAES 2026.08 class, two-column publication layout | `build/LAVA_JAES_Official_Layout.pdf` |

## Overleaf

Upload the project sources with their folder paths intact. Include both entry points, both shared files, the class, bibliography styles, support files, references, `sections/`, `tables/`, and `figures/`. Choose **pdfLaTeX**, select **main_review.tex** as the main document, and use **Recompile from scratch**. Download the PDF as `LAVA_JAES_Review_Draft.pdf`. Select **main.tex** to build the JAES layout independently. The two layouts use the same title and results.

## Local build

Run from the manuscript directory with TeX Live and latexmk on PATH.

```powershell
latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=build -jobname=LAVA_JAES_Review_Draft main_review.tex
latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=build -jobname=LAVA_JAES_Official_Layout main.tex
```

The supplied `jaes.cls` and `jaes.bst` retain their original bytes. The official class forces two columns when it builds the title and has no documented one-column option suitable for the review. Both entries use `jaes_compat.bst`, a compatibility copy of the original style that removes a duplicate `and` definition and replaces an undefined `bbl.presented` call. These two defects stopped BibTeX 0.99e. The original style remains in the folder for editorial comparison. If the journal supplies a corrected official style, use that style and rebuild. [Official JAES 2026.08 template](https://www.overleaf.com/latex/templates/journal-of-the-audio-engineering-society-jaes-template/rqsctzghzcwx)

Before submission, confirm the author list, affiliations, corresponding author, biographies, publication facts and rights for the earlier work. The lecturer version marks unconfirmed journal author metadata. The held-out results remain pending.
