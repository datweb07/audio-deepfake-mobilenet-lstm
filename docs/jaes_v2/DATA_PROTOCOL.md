# Data protocol V2

The executed cohort is a **development feasibility pilot**: six REAL and six FAKE from the legacy `validation` partition, selected with seed 804 before model execution. `test` is never a fallback. Source file checksums are recomputed and compared to the canonical split manifest. No legacy manifest is rewritten. The existing model was already selected/calibrated on legacy validation; thus this cohort cannot estimate independent performance.

`SourceRecord` explicitly records dataset/version, split, label, original checksum/rate/duration, known identities, source record, license and provenance status. Legacy dataset origin, license, generator and speakers remain UNKNOWN. The label `LAVA_LEGACY_INTERNAL` names the local collection, not a recovered public corpus.

The grouping key `local-sha256:<hash>` groups the variants we generate. `parent_scope=LOCAL_BYTES_ONLY` explicitly does **not** certify the original recording lineage, speaker identity or re-encoded near duplicates. Hold-out checks reject unknown identities instead of manufacturing them.

Partitions in V2: train, development, calibration, test, external_confirmation. Byte and parent groups cannot cross partitions. Optional speaker checks use speaker/source-speaker/target-speaker jointly. Generator checks apply to FAKE records. Exact decoded PCM audit preserves sample rate and channels and catches some lossless cross-container duplicates; it is not near-duplicate detection. The development shortcut diagnostic uses five simple acoustic metadata features with leave-one-local-parent-out validation; it is blocked on final/calibration data.

## External data gate

Candidates: official ASVspoof 2019/2021/5, WaveFake, In-the-Wild, CodecFake+. None was downloaded in this execution. Do not equate a reference repository directory with installed benchmark data. Official protocols must remain intact; custom strata are named separately. ASVspoof releases can share lineage: corpus/year labels alone do not establish independence. Attack ID is not automatically an exact generator identity.

CodecFake+ hosting was inspected: approximately 101 GB for the listed repository, exceeding the roughly 40 GB free local disk. A small listed CoSG archive does not by itself establish an independent balanced test cohort. Licenses differ across Codecfake and CodecFake+ releases; verify exact version and upstream voice data terms before acquisition. [Dataset listing](https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset/tree/main), [original Codecfake release](https://zenodo.org/records/11171708).

Required next inputs: accessible corpus location, version/checksums, official labels/splits, verified generator/attack mapping, speaker/parent metadata where available, license and teacher overlap disclosure. The runner intentionally rejects CONFIRMATORY scope on the legacy pilot config. A public-corpus importer and the factorial generator-shift estimator still require implementation after actual metadata inspection.

## Phase 2 update

The ASVspoof 2019 LA importer and guarded factorial estimator are now implemented. Actual public audio remains unavailable; the approved small official-file download retry returned HTTP 403. See DATASET_SELECTION_MATRIX.md and PUBLIC_DATASET_IMPORTER.md. Official-format fixtures validate code only and are not public research evidence.

A separate 64-record balanced legacy development cohort now supports the 14-condition sensitivity study. Exact decoded overlap and candidate near-duplicate screening passed their stated limited checks; true speaker/source/generator lineage remains UNKNOWN. The source audit now includes duration-only, sample-rate-only, silence-only, RMS-only and combined metadata baselines, fit exclusively within development folds. All generated variants retain local parent keys. Confirmation remains locked out.
