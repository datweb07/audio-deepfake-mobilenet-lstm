# Build JOURNAL FIX03

On Overleaf, choose **New Project → Upload Project**, upload this ZIP, select **pdfLaTeX**, and set the main document to `main_review.tex` for the A4 supervisor copy. `main.tex` is the two-column JAES draft with the same scientific content.

For a local build, run this sequence from the extracted directory for either entry point (change `$entry` as needed):

```powershell
$entry = 'main_review'
pdflatex -halt-on-error -interaction=nonstopmode "$entry.tex"
bibtex $entry
pdflatex -halt-on-error -interaction=nonstopmode "$entry.tex"
pdflatex -halt-on-error -interaction=nonstopmode "$entry.tex"
```

The figures and tables are included. The ZIP contains manuscript sources and template dependencies. Author metadata and required declarations for journal submission remain for the authors to complete.
