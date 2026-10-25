# Implementation log

## 24–25 October 2026

1. Read the complete attached master prompt in sequential chunks and verified the authoritative Word hash. Captured pre-change hashes for 2,376 legacy files, initial Git state and source/experiment inventory.
2. Inspected current detector/artifact/preprocessing contracts and metadata. Confirmed a local CPU pilot is feasible; no new neural training or data download required.
3. Opened primary current literature, including PCL-NET, FP-NAC, DK-CAST, CodecFake+ and RADAR; identified concrete overlap rather than assuming three novel methods.
4. Added isolated `src/lava/jaes_v2/`: typed provenance, partition guards, ordered transforms, config validation, paired-score analysis, legacy read-only runtime bridge, execution/aggregation and decoded-source audit. No legacy entry point or model identifier changed.
5. Implemented actual AWGN, environmental-noise mixing, conventional FFmpeg codec round trips, resampling, bandpass and provided-RIR convolution. Synthetic replay remains available only through legacy code; no measured RIR or physical replay was executed in V2. No neural codec is implemented here.
6. Added numerical/SNR/order/codec/partition/parent-bootstrap/integration tests. The initial decoded-audio test used two independent float-to-PCM quantizers and was corrected to an identical int16 fixture; the expected identical-PCM case now passes. Integration tests use a clearly labelled score fixture; the real pilot does not.
7. Ran real MobileNet pilot v1 and retained its evidence. Improved deterministic SVG output, execution-log sealing and encoded-stream metadata, then ran v2 under a new ID with the same preregistered pilot settings and a locked split-manifest hash. No unfavorable samples were dropped.
8. Real pilot v2: 60/60 scores successful; 0 excluded; direct bridge matches registry API within approximately 1.09e-7 absolute score. End-to-end run work approximately 37.26 s; output roughly 15.6 MB.
9. V2 suite: 19 passed, no skips. Selected baseline suite: 37 passed / 40, with three inherited UI/display contract failures recorded separately. Sandbox file-replacement failures were rechecked outside the sandbox. Preflight after tests confirmed every protected baseline hash still matches.
10. Saved a development-only metadata-shortcut audit; no decoded duplicate group found among the 12 sources. The tiny five-feature classifier is exploratory and does not validate source independence.
11. Final reproduction audit discovered libsndfile float-WAV files contained a timestamped PEAK chunk: v1/v2 decoded waveforms and scores matched, but container SHA-256 changed. Replaced only the V2 writer with deterministic IEEE float32 WAV serialization, added a test, and retained v1/v2 unchanged. Final pilot v3 completed 60/60 predictions in approximately 41.23 s with the same cohort and perturbations. The final reproduction tool checks all sealed files, rebuilds all 60 actual waveforms and compares file hashes, and rebuilds reports without importing a model framework.

No C2 training, C3 policy fitting, public-corpus download, physical recording, paid computation, manuscript results section or external submission was performed.
