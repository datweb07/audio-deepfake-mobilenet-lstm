# Next commands

Use the full launch package for long training and recovery. Start with the medium ERM partial baseline, then matched channel partial, before committing to full-foundation or three-seed expansion. No novel M3 training is ready. All commands below exist in the tested parsers; use a fresh output name when repeating a report/test operation.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment --help
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment preflight --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json --output outputs/jaes_v2/phase8/preflight/author_readiness_review.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment train --config configs/jaes_v2/phase8/p8_medium_erm_partial_s6107_v1.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment monitor --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment resume --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment verify --run outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_report aggregate --output outputs/jaes_v2/phase8/reproduction_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_report master --statistics outputs/jaes_v2/phase8/statistics --output outputs/jaes_v2/phase8/master_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment validate --output outputs/jaes_v2/phase8/validation_review
```

`resume` is for an interrupted unsealed run; do not execute it on a sealed completed run. `train` rejects existing IDs. The stored final master is a fixed handoff edition, so adding future full runs requires a new edition/readiness evidence rather than rewriting historical metrics. Historical Phase7 reproduction remains available unchanged through its own scripts.
