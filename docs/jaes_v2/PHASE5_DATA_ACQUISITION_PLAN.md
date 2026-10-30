# Phase 5 acquisition decision — 28 October 2026

Decision preceded extraction: inspect the user's existing LA.zip; retain 15 GiB free plus a 512 MiB extraction allowance. No additional multi-GB download was launched. Machine-readable inventory: `outputs/jaes_v2/datasets/asv2019_archive_v1/archive_inventory.json`.

| Dataset | Role | Archive bytes | Expanded bytes | Terms/access | Decision |
|---|---|---:|---:|---|---|
| ASVspoof 2019 LA | Controlled train/dev; separate calibration | 7,646,913,333, local supplied ZIP | 7,655,650,274 full; 3,178,404,021 selected extraction | ODC-By database notices and upstream content caveat; locally available | Select train/dev and two CM protocols + README |
| RADAR 2026 dev | Possible external media-domain transfer | 5,112,845,544 from Phase 3 official inventory | UNKNOWN until archive inventory | CC-BY-4.0 release; metadata already audited, audio not acquired | Defer until role and expanded storage locked |
| In-the-Wild | Possible external speech transfer | HF repository reported 8.16 GB, not a verified local archive | UNKNOWN | Author HF Apache-2.0 repository declaration; retain content caveats | Defer; generator identities unresolved |
| MLAAD-tiny | Existing exploratory development | Already acquired | Existing Phase 3 inventory | Existing pinned provenance/licenses | Retain; not independent confirmation |

Official pointers: [ASVspoof LA, DOI 10.7488/ds/2555](https://datashare.ed.ac.uk/handle/10283/3336), [RADAR release](https://zenodo.org/records/22110815), [author In-the-Wild repository](https://huggingface.co/datasets/mueller91/In-The-Wild). RADAR numbers are the retained Phase 3 release audit, not newly downloaded audio or independently recomputed archive contents.

Free space before extraction: 29,645,201,408 bytes. After selected extraction: 26,347,540,480 bytes. Archive retained; no duplicate audio written into outputs. Full eval partition is 4,462,843,273 expanded bytes and remains in ZIP. New feature cache is projected at roughly 25 MB uncompressed feature values; each full student checkpoint roughly 5.9 MB. Existing preflight checkpoints are retained as resource evidence. Teacher weights are not downloaded; expected 380 MB would exclude loader/environment costs.

The local archive SHA-256 is `dac23f8b70f26eca009e4ba13789953d462d13fca04f8452da51485039ed8a31`. CRC and length checks passed for extracted members. An official remote checksum has **not** been independently verified; do not equate local byte integrity with authenticated provenance.
