# Phase 3 acquisition and executable workflow

The first real external audio acquisition succeeded: **192 WAV files, 43,492,994 bytes**, from the official pinned MLAAD-tiny repository. Audio is stored once under `data/public/mlaad_tiny_9143e5e_dev_v1/`; the importer references those paths and does not copy audio into outputs. The shared transformed cache is separate. No model or environment upgrade was performed.

## License and attribution

Preserved raw files: `LICENSE`, `original/LICENSE`, `NOTICE_NO_COMMERCIAL_USE.txt`, `README.md`, under `outputs/jaes_v2/datasets/mlaad_tiny_metadata_v1/`. FAKE is CC-BY-NC-4.0; REAL retains the original M-AILABS copyright, conditions and disclaimer. Credit MLAAD authors / Nicolas Mueller / Fraunhofer AISEC and M-AILABS original creators. Commercial testing/training is not licensed by this workflow. Downstream modifications are explicit: a <=3 s prefix, resampling to 22,050 Hz, documented headroom and named local channel treatments. No audio was published or sent to an external service.

Every requested audio path has its repository, 40-character revision, expected size, upstream SHA-256, local SHA-256, status and bytes in `data/public/mlaad_tiny_9143e5e_dev_v1/acquisition.json`. Metadata Git blob hashes or LFS hashes are verified. Failures retain `.part` bytes and FAILED_PARTIAL status; such files cannot be imported. A retry needs inspection and a new destination; it never silently resumes or clones the full repository.

## Implemented commands

Run from the repository root with the existing `.venv`. The first three acquisition stages below are already complete; directories/files are immutable and rerunning them requires new names. The exact revision can be reused for a new acquisition.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.public_data discover --discovery outputs/jaes_v2/datasets/mlaad_tiny_metadata_v2 --revision 9143e5ea709575ebab6bec52840a1043aada7bb1
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.public_data select --discovery outputs/jaes_v2/datasets/mlaad_tiny_metadata_v2 --selection configs/jaes_v2/mlaad_tiny_dev_selection_v2.json --language en --group-count 4 --per-group 24 --seed 3804
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.public_data acquire --discovery outputs/jaes_v2/datasets/mlaad_tiny_metadata_v2 --selection configs/jaes_v2/mlaad_tiny_dev_selection_v2.json --destination data/public/mlaad_tiny_9143e5e_dev_v2 --maximum-bytes 50000000 --purpose noncommercial_research
```

Do not repeat those downloads just to read results. The implemented local integration command for the already acquired cohort, if a separate new import audit is needed, is:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.public_data import --discovery outputs/jaes_v2/datasets/mlaad_tiny_metadata_v1 --selection configs/jaes_v2/mlaad_tiny_dev_selection_v1.json --destination data/public/mlaad_tiny_9143e5e_dev_v1 --output outputs/jaes_v2/datasets/mlaad_tiny_dev_import_recheck_v1
```

`select` also supports explicit `--groups` and `--sample-ids` (exact IDs from metadata inventory); incompatible language or insufficient class/group coverage fails. `discover` is metadata-only and downloads no audio. `acquire` enforces maximum bytes and current free-disk reserve before acquisition. No supported operation downloads a full archive as a fallback.

The experiment config is `configs/jaes_v2/c1_mlaad_tiny_dev_v1.json`, with hashes of the actual imported manifest and selection, saved before inference. Its seven conditions are prepared native, common-headroom clean, PCM16, Opus 24 kb/s, fixed-absolute AWGN at reference 10 dB, noise→codec, codec→noise. Source/condition checksums and transform evidence are stored once and reused by both detector processes. Phase 2 local-SNR controls remain supported and unchanged; this prespecified matrix uses fixed absolute noise to avoid mixing policies.

Existing results can be rebuilt without inference:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase3 aggregate --run outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase3 aggregate --run outputs/jaes_v2/c1_mlaad_tiny_dev_v1_shufflenetv2_lstm
.\.venv\Scripts\python.exe -X utf8 -m src.lava.jaes_v2.phase3 verify --run outputs/jaes_v2/c1_mlaad_tiny_dev_v1_mobilenetv3_lstm --report outputs/jaes_v2/audit/phase3_reproduction_next.json
```

Aggregation verifies sealed data/config/cache inputs; it never calls `predict` or recalibrates thresholds. `verify` additionally checks the run seal and exact statistics parity; its report must use a new filename. For new experiments, `phase3 prepare --config <existing-new-config>` and `phase3 infer --config <existing-new-config> --model mobilenetv3_lstm` are implemented, but require a reviewed new config/manifest and new experiment ID. There is no confirmatory CLI flag.

Initial disk evidence: `outputs/jaes_v2/audit/phase3_resource_preflight.json`. It budgets raw audio, a single shared waveform cache, outputs, temporary files and >=1 GiB reserve; no 5–10 GB archive was downloaded. Network retries used ordinary official access after sandbox DNS failure; the renewed Edinburgh request still returned 403 and was not bypassed.
