# Phase 3 dataset decision — 27 October 2026

Primary: **MLAAD-tiny**, official author repository `mueller91/MLAAD-tiny`, immutable revision `9143e5ea709575ebab6bec52840a1043aada7bb1`. The decision was made from metadata, legal access, source coverage and storage before any detector inference. It was not selected using detector performance.

| Criterion | MLAAD-tiny | RADAR 2026 | ASVspoof 2019 LA |
|---|---|---|---|
| Authoritative source | [Fraunhofer project](https://deepfake-demo.aisec.fraunhofer.de/mlaad_tiny), [author release](https://huggingface.co/datasets/mueller91/MLAAD-tiny) | [Challenge](https://radar-challenge.github.io/), [Zenodo record 22110815](https://zenodo.org/records/22110815) | [Edinburgh release](https://datashare.ed.ac.uk/handle/10283/3336) |
| Legal conditions | FAKE: CC-BY-NC-4.0; REAL: retained M-AILABS original notice. Noncommercial research only for this acquisition | CC-BY-4.0; raw license downloaded and checksum verified | Previous ODC-By/content notice; renewed README request HTTP 403 |
| Bounded access | Individual revision-pinned files with upstream LFS SHA-256 | Archive-only audio in inspected release; individual dev metadata accessible | Official endpoint still blocked |
| Labels | Author audiofolder REAL/FAKE convention | Official bonafide/spoof protocol | Existing strict CM parser retained |
| Actual metadata coverage | 15,290 audio: 7,390 REAL / 7,900 FAKE; 69 unique author model_name values, 79 language/model metadata files | Dev protocol parsed: 10,573 REAL / 33,461 FAKE, all EN | No new local protocol/audio acquired |
| Language | Inventory: REAL en 6,070 / de 1,320; FAKE en 6,400 / de 1,500. Card descriptions are approximate and incomplete for this revision | Dev English; multilingual eval metadata not opened beyond release description | Language/source mapping needs additional evidence |
| Generator annotation | Per-record author model_name / architecture; not independently verified checkpoint identity | Per-record generator not in downloaded dev label/language files | Attack condition is not automatically exact generator |
| Source dependence | FAKE original_file and its structured source-text speaker; REAL lineage UNKNOWN | Source lineage/transform sequence not present in these protocol files | Official speaker/attack fields supported when supplied |
| Pairing | Apply deterministic local conditions to the same downloaded native waveform | Released waveforms already transformed; a locally applied treatment is not original clean/stressed pairing | Feasible after lawful local integration |
| Resource | Selected 192 audio total 43,492,994 bytes; no full clone | Dev archive 5,112,845,544 bytes, eval 8,752,649,011 bytes; neither acquired | Full archive not acquired |
| Decision | **Primary public development**, four annotated English system groups | Retain as prior art and potential independent media-domain diagnostic; no confirmation opened | Keep adapter and record blocker, do not loop on 403 |

Actual counts come from the pinned inventory plus all 79 author `meta.csv` files, not from the card's approximate 64-system wording. “69” denotes distinct annotation strings across languages, not 69 verified model checkpoints. The selected English groups are Cartesia.ai (Sonic-3), Chatterbox, DeepGram and Edge-TTS. Each has 24 FAKE, with source-text speakers balanced by seeded round-robin; 96 English REAL are selected independently of scores.

Selection: `configs/jaes_v2/mlaad_tiny_dev_selection_v1.json`. Full inventory and source evidence: `outputs/jaes_v2/datasets/mlaad_tiny_metadata_v1/`. Alternative source evidence: `outputs/jaes_v2/datasets/phase3_alternative_discovery_v1/`. No evaluation label file or evaluation audio was acquired from RADAR.
