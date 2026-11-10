# JAES manuscript build

The source follows the Audio Engineering Society's JAES 2026.08 sample, using the `jaes` class, the JAES BibTeX style, and the sample's author, abstract, keyword, and rule macros. Template attribution is Audio Engineering Society, CC BY 4.0. [AES author guidelines](https://aes.org/publications/journal-author-guidelines/) · [Official Overleaf template](https://www.overleaf.com/latex/templates/journal-of-the-audio-engineering-society-jaes-template/rqsctzghzcwx)

## Overleaf

1. Open the official template and choose **Open as Template** while signed in.
2. Retain the official support files, including `jaes.cls` and `jaes.bst`.
3. Upload `main.tex`, `references.bib`, `sections/`, `tables/`, and `figures/` at their supplied paths. Set **Main document** to `main.tex` and the compiler to **pdfLaTeX**.
4. Recompile and download the PDF and full build log. Check citations, first-appearance numbering, floats, equation alignment, font substitution, and overfull boxes. Count PDF pages including references, and report the body and reference pages separately if they start on separate pages.

The repository already contains the author-supplied template source under `papers/jaes_v2/template`. The manuscript package includes byte-identical copies of the class and bibliography style, with the support files used by the template. The class identifies itself as JAES 2026.08. Class SHA-256 is `375a82b3bd152ca1e4125e5b76b978e98f66db849c7085052d592b452f95da20` and bibliography style SHA-256 is `6bad85655454dbef1029cb76087796f8eeaa130d36de5d1bb9daf4b757540e8e`. The duplicate class in the supplied source is byte-identical and is omitted from the manuscript package.

The four evidence figures are standalone vector PDFs. The 25–30 page target should be assessed after a complete build and after the held-out evidence is available. See the sprint handoff for the build attempt and outcome.

## Local build after obtaining the official support files

From this directory, run the template's pdfLaTeX/BibTeX workflow:

```powershell
latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex
```

If latexmk is unavailable but pdfLaTeX and BibTeX are installed, run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. Check the build log before distributing. A separate Tectonic 0.17.0 portable attempt uses XeTeX and is not a substitute for verifying the official pdfLaTeX build before submission.

## Evidence and author checks

From the repository root, run `python -X utf8 papers/jaes_v2/jaes_manuscript/generate_evidence.py` to regenerate the tables, figures, and claim map from the consolidated development JSON. The outputs shipped with the draft allow it to be built without the research repository.

Before submission, confirm the journal author list, affiliations, correspondence address, author photos and biographies, funding, conflicts, venue citation, publication and copyright status, and the precise AI tool/model/version used in substantive drafting. Confirm that the completed draft and its exhibits can be shared under the relevant publication and data terms.
