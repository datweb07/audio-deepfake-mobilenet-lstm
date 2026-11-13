# Build the lecturer and JAES drafts

Both entry points read `shared_metadata.tex` and `shared_body.tex`, including the same sections, figures, tables, acknowledgement and bibliography.

The single-column review PDF omits author metadata and includes the restored UEH acknowledgement. The JAES entry point prints the AI-assistance statement. Before submission, confirm the scope of UEH support for the journal work and complete the tool model and version in the statement where available.

| Entry point | Format | Build filename |
| --- | --- | --- |
| `main_review.tex` | A4, single-column article, 11 pt, 25 mm margins, 1.5 body spacing for annotation | `build/LAVA_JAES_Review_Draft_v7.pdf` |
| `main.tex` | Authentic JAES 2026.08 class, two-column publication layout | `build/LAVA_JAES_Official_Layout_v7.pdf` |

## Overleaf

Upload `build/JAES_V7_Overleaf_Source.zip` as a new review project to avoid stale files. Select `main_review.tex`, choose pdfLaTeX and recompile from scratch. The review ZIP contains the review entry point. Use the complete local manuscript folder and select `main.tex` for the JAES layout.

## Local build

Run from the manuscript directory with TeX Live and latexmk on PATH.

```powershell
latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=build -jobname=LAVA_JAES_Review_Draft_v7 main_review.tex
latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=build -jobname=LAVA_JAES_Official_Layout_v7 main.tex
```

The supplied `jaes.cls` and `jaes.bst` retain their original bytes. The official class forces two columns when it builds the title and has no documented one-column option suitable for the review. Both entries use `jaes_compat.bst`, a compatibility copy of the original style that removes a duplicate `and` definition and replaces an undefined `bbl.presented` call. These two defects stopped BibTeX 0.99e. The original style remains in the folder for editorial comparison. If the journal supplies a corrected official style, use that style and rebuild. [Official JAES 2026.08 template](https://www.overleaf.com/latex/templates/journal-of-the-audio-engineering-society-jaes-template/rqsctzghzcwx)

Before submission, confirm the author list, affiliations, corresponding author, biographies, publication facts and rights for the earlier work. The lecturer PDF omits author metadata. The official entry point retains fields for completion. The current sources include the authorized locked held-out results. The original supervisor and official-layout PDFs retain their baseline filenames.

The supplemental vector figures can be regenerated with `python -B generate_review_figures.py` from this directory using Matplotlib, NumPy and h5py and the original repository evidence. The prebuilt figures are sufficient for Overleaf compilation.

From the repository root, regenerate the completed held-out analysis with `python -B -X utf8 papers/jaes_v2/jaes_manuscript/analyze_locked_heldout.py`. It checks the complete score grid and replays registered metrics before producing the analysis exports and manuscript figures/tables. It requires the already completed locked run and performs analysis only.

For the V3 post-hoc diagnostics and exposure audit, run the manuscript file analyze_v3_scores.py with the existing scientific Python environment. This replays sealed scores and journals without inference.


## V4 supplement

Run from the repository root with the scientific environment. This recomputes threshold, attack and order summaries from the protected score grids without inference.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 papers/jaes_v2/jaes_manuscript/analyze_v4_scores.py
```

The supplement is under `outputs/jaes_v2/supervisor_major_revision/`. The V4 operating-point figure in the manuscript comes from that analysis. See the supervisor tracker and response for artifact roles. The development-only matched-component CLI can reproduce the 9207 evidence in a clean copy of the repo with its original output directory absent. It refuses to overwrite a completed diagnostic. It does not require official evaluation audio.

## V5 development replication

The V5 analysis reads the sealed six-model development results and original four-model held-out predictions. The registered seed summary is write-once. If the three-seed JSON already exists, retain it and run the V5 verifier/renderer from the repository root.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 papers/jaes_v2/jaes_manuscript/analyze_v5_replication.py
```

See the [replication review](../../../docs/jaes_v2/THREE_SEED_REPLICATION_REVIEW.md) and [V5 verification](../../../outputs/jaes_v2/three_seed_revision/verification_and_statistics.json). V4 threshold and standalone analyses remain separate.

## V7 training-size supplement

Use the existing sealed scores to replay the four-model size comparison with `python -B -X utf8 -m scripts.jaes_v2.phase10_size1280 compare` from the repository root. Generate the V7 figure and table with `python -B -X utf8 papers/jaes_v2/jaes_manuscript/generate_v7_size_assets.py`. The prebuilt assets are included for Overleaf. See [the supplementary analysis](../../../docs/jaes_v2/TRAINING_SIZE_SENSITIVITY_V7_SUPPLEMENT.md).
