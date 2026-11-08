# Phase 9 — Training priority matrix

IDs below identify prospective designs until a frozen trainer and resolved configs exist. The measured short probe and historical audit are executable now. Runtime scenarios exclude validation and are not reservations. Per-job storage is unknown until the checkpoint policy is tested.

| Tier and experiment ID | Model / method / seed | Data / budget | Runtime and disk | Question and prerequisite | Stop / requirement / replacement |
| --- | --- | --- | --- | --- | --- |
| 1 `parity_v1` | MobileNetV3Small–LSTM / three gradient paths / 6107 fixture | 2 gradient cases, 4 validation views | 35.62 s, evidence JSON 60 KB | Same model, seed and Adam | Complete. Required arithmetic screen. |
| 1 `p9_engine_100_v1` | MobileNetV3Small–LSTM / paired channel / 9107 | 640, up to 100 updates, resume plus uninterrupted control | ~14 min update-only per path, disk TBD | Trace, numerical and resource gates, fixed runtime | Required before replication. Stop on OOM, commit or disk floor, or parity failure. Extend to 500 only after passing. |
| 2 `p9_medium_{erm,channel}_s{9107,9207,9307}_v1` | MobileNetV3Small–LSTM / ERM and channel, paired per seed | 640, 1+20 epochs max, 840 updates each | ~1.9 h each update-only, disk TBD | Reproduce stress benefit and per-group trade-offs with shared initialization, order and checkpoint schedule | Required for a three-seed claim. No substitution with historical 6107. Stop according to preregistered guards and resource policy. |
| 2b `p9_mediumplus_{erm,channel}_s9107_v1` | Same model and matched methods / 9107 | 2,560, ≤3,360 updates each | ~7.7 h each update-only, disk TBD | Test scale on same dev after matched medium evidence | Conditional. May omit if medium fails or deadline is tight. |
| 3 `p9_cue_identification_v1` | Controlled cue measurement before detector training | Paired-source pilot, budget to register | Unknown | Identifiability, rights, matched mechanism controls | Conditional. Blocked by controlled lineage. No replacement with weighting or consistency. |
| 4 `p9_foundation_{erm,channel}_s9107_v1` | Same architecture / paired baselines | 25,380, ≤33,327 updates each | ~76.6 h each update-only, disk projection separate | Only after all engine/resource/data gates and evidence justify scale | Deferred. Omit for deadline if bounded experiment answers the question. |
| 5 `p9_shufflenet_transfer_v1` | ShuffleNet / same selected protocol | Medium, budget TBD | Unknown | Architecture transfer only after effect replicated | Optional, defer. |
| 6 `p9_aasist_or_rawnet2_eval_v1` | AASIST or RawNet2, checkpoint provenance required | Matched development evaluation only if overlap resolved | Unknown | External reference with clear preprocessing and threshold semantics | Optional. If training overlap unresolved, label comparison as reference or omit. |

Ordinary equal-view, finite RawBoost and group-DRO remain potential controls. Add only the matched comparator needed for a preregistered mechanism claim. Do not launch a model grid.
