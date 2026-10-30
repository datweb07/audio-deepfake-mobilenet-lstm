# Phase 5 data protocol

Foundation: `outputs/jaes_v2/datasets/asv2019_foundation_v1/`. Reuses the existing official ASVspoof adapter and PublicSourceRecord contract. Every CM recording retains dataset/version, official/research splits, recording, speaker, local parent, attack, UNKNOWN generator/source lineage, label, rate/duration, checksum and license. `research_split`/`split` are the research-partition fields; `official_recording_id` is recording ID; `source_recording_id` remains UNKNOWN. Local SHA parent identifies bytes, not certified original utterance lineage.

| Role | Recordings | REAL | FAKE | Official speakers |
|---|---:|---:|---:|---:|
| Official train → TRAIN | 25,380 | 2,580 | 22,800 | 20 |
| Official dev → MODEL_SELECTION | 17,503 | 2,023 | 15,480 | 16 |
| Official dev → CALIBRATION_RESERVED | 7,341 | 525 | 6,816 | 4 |

Calibration speakers are LA_0073, LA_0074, LA_0078, LA_0101, selected by seed 5804 before pilot outcomes. Calibration audio headers/checksums were inventoried, but no calibration features, model scores, threshold fitting or training use occurred. There are 24,986 archive dev audio files but 24,844 CM protocol trials; 142 enrollment recordings do not enter CM manifests. Evaluation audio and labels were neither extracted nor parsed. ASV protocols/scores were not imported.

All 50,224 CM records passed local audio-header/checksum inventory and known speaker/byte-parent partition checks. Full perceptual/decoded deduplication and original utterance lineage are **not** established. A01–A06 are official attack categories, not verified generator checkpoint IDs. VCTK corpus commonality is not proof of utterance independence. One dataset supplies both classes, so dataset ID cannot trivially identify class, but class-conditioned acquisition/attack artifacts remain possible.

Bounded cache selection is deterministic, class balanced, round-robin across official speakers and attacks, without looking at detector scores. Manifest and feature-cache hashes lock the actual subset. All variants of a recording stay in its partition and share transform seed/waveform across P0/P1. Clean preparation decodes to 22,050 Hz, uses at most the first 3 seconds, applies reference peak headroom .25, then pads/segments using the production contract. This is an explicit short-window design and does not establish full-utterance robustness.

Training views: clean, half gain, AWGN 10 dB and Opus 24k. Development: clean, quarter gain, AWGN 5 dB, Opus 64k and both AWGN10/Opus24 orders. Gain/severity are held out as values; the chain components themselves are familiar. Do not call this unseen channel families. Fixed absolute noise is shared across order arms; no post-channel waveform normalization is applied except the explicitly named P2 diagnostic baseline.

The original and actual prepared-window duration/rate/silence/RMS/combined numeric shortcut baselines are fitted separately inside train and development with grouped official-speaker folds. Temporary train-as-development role aliases exist only in shortcut diagnostics and do not modify published manifests. See cache grouped_shortcuts.json. Class ratio is deliberately balanced for pilot feasibility, so these rates are not deployment prevalence estimates.

Use ledger: ASV TRAIN=train; 16-speaker dev=design/selection; four-speaker dev=reserved calibration; ASV eval=unopened, confirmation not locked; MLAAD-tiny=already used exploratory development; RADAR=metadata inspected, no audio; In-the-Wild=not acquired. No corpus may silently become untouched confirmation after design use. Future cross-corpus comparisons need byte/source/teacher-overlap screening and a fixed role before inspecting scores.

Recorded-byte metadata comparison across all 50,224 ASV CM records found zero matches against 18,232 distinct canonical legacy hashes and 192 MLAAD selected hashes. Evidence: outputs/jaes_v2/audit/phase5_corpus_overlap.json. No reference audio or evaluation scores were accessed for this comparison. Different encodings, source utterance reuse, speakers and perceptual overlap remain unresolved.
