# C1 Phase 2 confounder controls

The new protocol is versioned separately from the 12-recording Phase 1 pilot. No Phase 1 run was rerun or overwritten.

## Noise normalization

`NoisePlan` draws one unit-RMS noise vector per recording using a sample-derived seed. Both execution orders reuse it. `local_target_snr` computes amplitude from signal RMS at insertion. `fixed_absolute` calibrates amplitude once from the prepared clean recording and reuses the absolute noise vector/amplitude after the codec. Evidence records policy, seed, noise RMS, absolute power, insertion-point signal RMS and measured SNR.

The study compares each policy's A→B minus B→A error contrast, then the difference between those contrasts. Fixed noise only controls scaling; it does not make a nonlinear lossy codec commute with noise. Noise-before-codec is still altered by the codec. A and B arrows denote execution left to right.

## Delay and alignment

An impulse probe and per-recording waveform cross-correlation estimate net residual delay after FFmpeg/container compensation. Positive lag means the codec output lags the input. The known-delay test uses a nonperiodic signal shifted by exactly 37 samples, with independent expected alignment.

FFprobe records encoded stream rates, packet timing and available skip-sample metadata. Codec preskip must be interpreted at the encoded stream's rate, not 22,050 Hz. **Separate encoder and decoder delay cannot be identified from this black-box round trip.** Both component fields explicitly remain NOT_SEPARATELY_IDENTIFIABLE. Container preskip is not measured wall-clock encoder latency. This limitation must not be replaced with invented component estimates.

The native codec waveform remains unchanged. `codec_aligned` is a separately named sensitivity treatment; it is never silently substituted for the codec condition. Correlation can be biased by nonlinear distortion or periodic speech, and the impulse is not a speech-distribution validation. Search limit and boundary status are retained.

## Other controls

| Condition | Purpose |
|---|---|
| `clean_native` | Prepared sampling/duration path without headroom attenuation |
| `clean` | Shared peak cap 0.25, same as the original pilot preparation |
| `headroom_low` | Peak cap 0.125 sensitivity; source amplitude change stays explicit |
| `pcm16` | One documented libsndfile PCM16 quantization round trip |
| `identity` | PCM16 WAV → FFmpeg float WAV, no lossy codec or intended resampling |
| `resample` | SciPy polyphase 22,050→24,000→22,050 Hz control |
| `codec` | Conventional Opus 24 kb/s including quantization and implementation resampling |
| `codec_aligned` | Explicit net-delay sensitivity counterpart |

Resample-only uses SciPy, not FFmpeg's internal resampler, so it is a sensitivity control rather than an exact isolation of every internal codec stage. `codec−pcm16` still includes implicit sample-rate conversion and compression. There is no pure-compression causal claim. Serialization is timestamp-free float32 WAV; checksum reproduction is tested separately from inference numerical tolerances.

Primary development estimand: fixed-absolute-noise order difference in fixed-threshold classification error, averaged equally over local parent keys. Other classwise/score/ranking estimates are descriptive secondary diagnostics. Raw-score squared error is not a validated probability-calibration metric. No p-value or power claim is made from this exploratory cohort.

## Executed development observations

Run `c1_channel_controls_dev_v1`: 64 records / 896 successful predictions / no exclusions. Local insertion after the codec used absolute noise power 0.91486–1.00499 times the fixed reference (mean 0.96852). Thus the normalization confounder exists numerically; it was measured rather than assumed absent.

Both policies nevertheless produced the same fixed-threshold decisions in this cohort: order-error difference 0.125, with descriptive local-parent bootstrap 95% interval approximately [0.046875, 0.219141]. The policy difference of order-error contrasts was 0, while the raw-score-squared-error difference was about 0.001158 with interval [-0.000578, 0.003146]. A degenerate [0,0] error interval only describes these observed decisions; it is not proof of population equivalence. Unknown original grouping and development selection prevent confirmation claims.

The impulse's net residual lag was 0 at 22,050 Hz; all 64 speech correlations also chose lag 0. FFprobe showed Opus first-packet timestamp -0.0065 s and skip_samples=312 at the encoded 48 kHz clock. Container compensation explains why nominal preskip need not appear as residual waveform lag. No separate encoder/decoder component measurement is asserted.

PCM16, identity transport and resampling did not change classification error relative to prepared clean on this cohort; raw-score/ranking diagnostics are still recorded separately. Headroom changed one decision relative to the no-headroom clean path. Do not conclude those transformations have zero effect on other data, detectors or thresholds.
