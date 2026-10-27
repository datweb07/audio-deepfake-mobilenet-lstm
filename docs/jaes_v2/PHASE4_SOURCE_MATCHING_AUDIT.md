# Phase 4 source matching audit — 27 October 2026

Status: **SOURCE_MATCHED_EXPERIMENT_BLOCKED for the proposed fixed-system-pair pilot**. Source matches exist; adequate repeated fixed-factor coverage does not. This is a design/precision limitation, not a claim that all incomplete-block analyses are mathematically impossible.

## Executed inventory audit

The six Phase 3 selected system pairs each share **zero** `source_recording_id` values. They share four original source-text speakers but use different source content. The existing between-system results remain observational. Historical scores and conclusions were not rewritten.

All 79 author metadata tables from pinned revision `9143e5ea709575ebab6bec52840a1043aada7bb1` were checked against their saved hashes before parsing. Across 7,900 FAKE files there are 7,601 distinct annotated original source IDs, 69 distinct author system names and nine source-text speakers across all languages. There are 290 same-language multi-system source groups: 282 shared by two systems, seven by three, one by four. Corresponding 589 candidate rows have consistent normalized author text; none was rejected for transcript disagreement in this release.

The maximum English overlap between a fixed pair is **three source IDs**. The three maximizers are MeloTTS/parler_tts_mini_v0.1, Nari Dia2/OuteTTS and e2-tts/minimax/speech-2.6-hd. The bounded design requires at least eight common sources and five source-text clusters before pilot execution; these are conservative feasibility floors, not powered confirmation criteria. The available English original-source speakers are only four. Even relaxing that speaker floor does not provide eight common sources per fixed pair.

## Matching contract

`source_matching.py` separates: level 0 unknown; level 1 speaker only; level 2 documented identical text only; level 3 identical full author source ID; level 4 independently verified original-content relationship. Identical ID with conflicting author transcript is rejected. A basename collision across full paths is AMBIGUOUS_SOURCE_MATCH, never merged. Whitespace/Unicode normalization does not turn different source IDs into level 3.

Actual verified candidates are **level 3 author annotation**, not level 4. Equal author text does not independently verify the underlying original audio. It also does not control synthesized voice, reference speaker, vocoder checkpoint, parameters or mastering. The source-matched estimator declares these remaining observational limitations.

## REAL mapping

REAL files have flattened original paths and author class labels. Their basenames can suggest a unique original-file candidate in the FAKE annotation inventory; zero ambiguous basename collisions were found in this lookup. Uniqueness is not an authoritative REAL-to-original mapping. All candidate links remain level 0 and REAL speaker/source/parent lineage remains UNKNOWN. Synthetic waveform similarity was not used to invent text/source equality.

Machine evidence: `outputs/jaes_v2/audit/phase4_source_matching/`: candidates, verified, rejected, coverage JSON, source-generator matrix, six selected-pair overlaps, all English pair overlaps, REAL candidate mapping and figures. No new audio acquisition was needed.
