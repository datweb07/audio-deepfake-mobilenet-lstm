# Public dataset importer

`src/lava/jaes_v2/datasets.py` implements a typed adapter interface and the ASVspoof 2019 LA countermeasure protocol adapter. It parses the five official fields: speaker, recording, unused placeholder, attack condition, bona-fide/spoof key. Labels map to REAL=0 / FAKE=1. It validates official split prefixes, unsupported labels, attack/key consistency, duplicate recording IDs, local audio availability and class coverage.

`PublicSourceRecord` retains official and research splits separately. The official speaker column is preserved; source-speaker and true source-recording lineage remain UNKNOWN. A checksum-local parent key groups generated variants but does not certify original lineage. `attack_id` is populated from protocol; `generator_id` and model-relative attack familiarity stay UNKNOWN. Local file checksums detect subsequent changes, but do not by themselves prove authenticity against an unavailable official per-file checksum list.

Output directories are versioned and must not exist. Expected files: `source_inventory.csv`, `provenance_manifest.csv`, `integrity_report.json`, `partition_audit.json`, `source_metadata.json`. Missing selected audio produces an explicit inventory/status and fails; it never publishes an incomplete manifest as a successful cohort. Metadata-only execution is labelled METADATA_ONLY with an empty audio manifest. No audio is copied or downloaded.

For an actual local release extracted under `data/public/ASVspoof2019/LA`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.datasets --dataset asvspoof2019_la --dataset-root data/public/ASVspoof2019/LA --protocol data/public/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --official-split dev --selection configs/jaes_v2/asv2019_dev_ids.txt --output outputs/jaes_v2/datasets/asvspoof2019_dev_subset_v1
```

The CLI is implemented. The data path and selection file above are **required future inputs**, not files claimed present. The selection file must contain exact protocol recording IDs, one per line, chosen before inference with both classes retained. Do not run until those inputs exist. Without `--selection`, the importer attempts the full specified official split. `--metadata-only` inspects a local official protocol without claiming audio integrity.

For a public development study, supply the resulting manifest through `ControlConfig.public_manifest`, use `scope=PUBLIC_DEVELOPMENT`, set `per_class` to its exact balanced count and lock its file hash. The runner rejects train/test/calibration/eval records in this development mode. No automatic downsampling or reassignment of official eval to development is performed.

Overlap dimensions are reported independently by `overlap.py`. Candidate near-duplicate screening uses normalized correlation on an 8 kHz prefix with an explicit threshold/lag and pair-count budget. Positive shifted/gain fixtures and unrelated negatives calibrate basic correctness; flags never automatically remove samples or establish speaker independence.
