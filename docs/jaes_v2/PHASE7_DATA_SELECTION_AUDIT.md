# PHASE7 DATA SELECTION AUDIT

## Dataset Inventory and Provenance

| Dataset | Local status | Source | License | Counts / scope |
|---|---|---|---|---|
| LAVA_LEGACY_INTERNAL | EXECUTED | UNKNOWN | UNKNOWN | {'test': 2737, 'train': 12762, 'validation': 2733} |
| ASVspoof2019_LA | TRAIN_DEV_EXTRACTED_EVAL_UNOPENED | https://datashare.ed.ac.uk/handle/10283/3336 | ODC-By database; upstream content caveats | {'train': {'recordings': 25380, 'class_counts': {'0': 2580, '1': 22800}, 'speakers': 20}, 'development': {'recordings': 17503, 'class_counts': {'0': 2023, '1': 15480}, 'speakers': 16}, 'calibration': {'recordings': 7341, 'class_counts': {'0': 525, '1': 6816}, 'speakers': 4}} |
| MLAAD_TINY | EXPLORATORY_DEVELOPMENT_EXECUTED | ['https://huggingface.co/datasets/mueller91/MLAAD-tiny/tree/9143e5ea709575ebab6bec52840a1043aada7bb1'] | ['CC-BY-NC-4.0', 'M-AILABS original notice'] | 192 |
| RADAR2026 | METADATA_ONLY_NO_AUDIO | https://zenodo.org/records/22110815 | CC-BY-4.0 release declaration | See Phase3 release inventory; no locally executed audio |
| In-the-Wild | NOT_ACQUIRED | https://huggingface.co/datasets/mueller91/In-The-Wild | Repository Apache-2.0 declaration; upstream conditions | NOT_VERIFIED_LOCALLY |
| DeepFakeVox-HQ / ANC-Spoof | PRIOR_ART_ONLY_NOT_ACQUIRED | F-SAT / PCL-NET papers | NOT_VERIFIED_FOR_LOCAL_USE | NOT_MEASURED_LOCALLY |

RADAR and In-the-Wild are investigated options, not executed external evidence. Literature datasets DeepFakeVox-HQ and ANC-Spoof were not acquired. A cited dataset or code repository is not installed training data. No Phase6 multi-GB download or ASV re-extraction occurred. The existing user archive has local SHA/CRC checks; authenticated remote byte identity remains unverified.

| Observed dataset profile | Rates / Hz | Duration seconds min/median/max | Known speaker/lineage limits |
|---|---|---|---|
| ASVspoof2019_LA | [16000] | {'min': 0.6524375, 'median': 3.23175, 'max': 13.1879375} | ['Source/parent content and exact generator identity UNKNOWN', 'Official speaker IDs are not full source-utterance lineage', 'SHA256 and CRC establish local integrity; official remote archive checksum not verified', 'Decoded/perceptual duplicate screening not exhaustive'] |
| MLAAD_TINY | [16000] | {'min': 1.32, 'median': 6.874875, 'max': 18.349875} | ['M-AILABS:en_UK/by_book/female/elizabeth_klett', 'M-AILABS:en_US/by_book/female/judy_bieber', 'M-AILABS:en_US/by_book/female/mary_ann', 'M-AILABS:en_US/by_book/male/elliot_miller'] |

ASV archive/extraction bytes from the retained central-directory audit: {'archive_bytes': 7646913333, 'total_expanded_bytes': 7655650274, 'extracted_bytes': 3178404021, 'extracted_files': 50371}. Calibration header profiles were not reopened in Phase6.

Phase7 selected cohort:

| role | n | real | fake | speakers | attacks |
| --- | --- | --- | --- | --- | --- |
| train | 640 | 320 | 320 | 20 | {"A01": 53, "A02": 54, "A03": 54, "A04": 54, "A05": 53, "A06": 52} |
| development_A | 224 | 112 | 112 | 7 | {"A01": 19, "A02": 19, "A03": 19, "A04": 19, "A05": 18, "A06": 18} |
| development_B | 144 | 144 | 0 | 9 | {} |

Selection manifest SHA256: `029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319`. No new dataset/audio download. REAL-only speakers may have prior exploratory Phase5 exposure. MLAAD-tiny remains already-used development. RADAR audio and In-the-Wild remain unacquired.

## Dataset Split and Integrity Protocol

The canonical legacy split groups byte-identical files; it does not certify speaker, generator, original utterance or corpus independence. MLAAD-tiny has a custom exploratory development selection with its official split UNKNOWN; all earlier design use prevents relabeling it as untouched confirmation. ASV official train stays train; official development is divided by known official speakers into model-selection development and reserved calibration. Evaluation audio/protocol labels remain unopened. The new selector reads only the two train/model-selection manifests, never the reserved calibration file. It excludes development speakers lacking both classes and records the resulting target population. Byte, known-parent, known-speaker and selected decoded-PCM conflicts are checked. Re-encoding, perceptual reuse and unknown original lineage remain unresolved.

| ASV research role | Actual counts | Manifest | SHA256 |
|---|---|---|---|
| train | {'recordings': 25380, 'class_counts': {'0': 2580, '1': 22800}, 'speakers': 20} | outputs/jaes_v2/datasets/asv2019_foundation_v1/train_manifest.csv | 7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72 |
| development | {'recordings': 17503, 'class_counts': {'0': 2023, '1': 15480}, 'speakers': 16} | outputs/jaes_v2/datasets/asv2019_foundation_v1/development_manifest.csv | dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a |
| calibration | {'recordings': 7341, 'class_counts': {'0': 525, '1': 6816}, 'speakers': 4} | outputs/jaes_v2/datasets/asv2019_foundation_v1/calibration_manifest.csv | 4fb256823a83d313e2ddab938c4271aa2e72bc28f8cb6f1cb0e6e3b35f25ca74 |

Phase6 selection:

```json
{
  "status": "LOCKED_SPEAKER_BALANCED_DEVELOPMENT",
  "seed": 5806,
  "per_speaker_per_class": 4,
  "foundation_hashes": {
    "train": "7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72",
    "development": "dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a"
  },
  "counts": {
    "train": {
      "n": 160,
      "speakers": 20,
      "real": 80,
      "fake": 80,
      "attacks": [
        "A01",
        "A02",
        "A03",
        "A04"
      ]
    },
    "development": {
      "n": 56,
      "speakers": 7,
      "real": 28,
      "fake": 28,
      "attacks": [
        "A01",
        "A02",
        "A03",
        "A04"
      ]
    }
  },
  "exclusions": "Development speakers without FAKE support excluded prospectively; not representative full official dev",
  "unknown_lineage": "Original utterance/generator checkpoint UNKNOWN",
  "decoded_duplicate_groups": 0,
  "calibration_accessed": false,
  "evaluation_accessed": false,
  "manifest_sha256": "45ff123a734e11935e9e7f7a1ee9259f950cc82f7441bf06b6ca96cd56d87965"
}
```

Phase7 preserves official train/model-selection-development roles. Reserved7341calibration and official evaluation were not loaded. Byte checks, selected exact decoded PCM, known speakers and local-byte parents show no selected overlap; original utterance ancestry remains ORIGINAL_SOURCE_LINEAGE_UNKNOWN. The selector rotates the16FAKE remainder across six attacks per speaker instead of always privileging the first four. Selection and cache contracts reject stale hashes and reserved roles. `selection_rows.csv` records stratum and explicit provenance fields; manifest hash is stored in its selection metadata to avoid an impossible self-referential CSV hash.

Verified historical exposure: all9REAL-only speakers were used in Phase5development; 3Phase7B recordings also match those earlier selected IDs. This is broader-speaker sensitivity inside development, never independent confirmation. Reserved manifests may be byte-hashed for historical integrity; their records/labels are not parsed or used for training, selection, calibration or evaluation.

## All-Attack Experimental Design

Selection and protocol are locked independently of scores.

```json
{
  "calibration_accessed": false,
  "counts": {
    "development_A": {
      "attacks": {
        "A01": 19,
        "A02": 19,
        "A03": 19,
        "A04": 19,
        "A05": 18,
        "A06": 18
      },
      "fake": 112,
      "n": 224,
      "real": 112,
      "speakers": 7
    },
    "development_B": {
      "attacks": {},
      "fake": 0,
      "n": 144,
      "real": 144,
      "speakers": 9
    },
    "train": {
      "attacks": {
        "A01": 53,
        "A02": 54,
        "A03": 54,
        "A04": 54,
        "A05": 53,
        "A06": 52
      },
      "fake": 320,
      "n": 640,
      "real": 320,
      "speakers": 20
    }
  },
  "decoded_duplicates": 0,
  "evaluation_accessed": false,
  "foundation_hashes": {
    "development": "dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a",
    "train": "7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72"
  },
  "known_partition_audit": "PASS",
  "manifest_sha256": "029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319",
  "original_source_lineage": "ORIGINAL_SOURCE_LINEAGE_UNKNOWN",
  "per_cell": 16,
  "scope": "EXPLORATORY_DEVELOPMENT; A01-A06 SEEN IN TRAIN; B speakers may have prior Phase5 exposure",
  "seed": 7107,
  "selection_id": "medium_all_attack_v1",
  "selection_rows_sha256": "52b65bc3cac4cd5b415a353dd400969058cb2774223c80c0f26476d05794d9bb",
  "status": "LOCKED_ALL_ATTACK_SELECTION"
}
```

| split | speaker_id | REAL | FAKE | A01 | A02 | A03 | A04 | A05 | A06 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | LA_0069 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| development | LA_0070 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| development | LA_0071 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| development | LA_0072 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| development | LA_0075 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| development | LA_0076 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| development | LA_0077 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| development | LA_0099 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0100 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0102 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0103 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0104 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0105 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0106 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0107 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| development | LA_0108 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | LA_0079 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0080 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0081 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0082 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0083 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0084 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0085 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0086 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0087 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0088 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0089 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0090 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0091 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0092 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |
| train | LA_0093 | 16 | 16 | 2 | 2 | 3 | 3 | 3 | 3 |
| train | LA_0094 | 16 | 16 | 3 | 2 | 2 | 3 | 3 | 3 |
| train | LA_0095 | 16 | 16 | 3 | 3 | 2 | 2 | 3 | 3 |
| train | LA_0096 | 16 | 16 | 3 | 3 | 3 | 2 | 2 | 3 |
| train | LA_0097 | 16 | 16 | 3 | 3 | 3 | 3 | 2 | 2 |
| train | LA_0098 | 16 | 16 | 2 | 3 | 3 | 3 | 3 | 2 |

StratumA defines a matched-support controlled estimand. StratumB tests false-alarm sensitivity on9REAL-only speakers. No fake records are imputed. Per-attack FAKE-only ROC-AUC/EER/PR-AUC are undefined and stored null; the full112REAL reference is not silently replicated per attack.

