# Phase 4 shortcut and preprocessing sensitivity

Scope: **diagnostic within-record interventions**, not a new preprocessing standard or proof that a detector uses silence.

Production contract was read and tested: `src/preprocessing.py` loads a <=3-second prefix, resamples to 22,050 Hz, zero-pads/crops to 66,150 samples, then forms six chronological 0.5-second Mel/RGB tensors. Signal diagnostics therefore measure the exact padded input separately from full original recordings and transformed inputs. New CSVs retain those input definitions.

Measurements include sample duration, prefix truncation fraction (duration proxy, not lost word count), amplitude RMS/peak, fixed-threshold silence fraction, leading/trailing below-threshold lengths, above-threshold sample time, 20-ms frame-RMS activity proxy, spectral-band energy fractions, 95% rolloff and out-of-unit amplitude. Activity proxies are not verified speech/VAD; RMS is not LUFS/perceptual loudness. Full-source and prepared statistics are not interchangeable.

| Policy | Controlled factor | Implementation | Information removed / introduced |
|---|---|---|---|
| global_half_v1 | Uniform gain | Exact factor .5 on prepared observed samples | Changes spectral-floor/quantization exposure; not guaranteed feature invariance |
| rms_anchor_v1 | Reference signal level | Target RMS .03 with peak cap .25 | Cap can prevent identical RMS; amplitude cues can be legitimate forensic evidence |
| edge_repack_v1 | Observed-window edge timing | Remove samples outside first/last abs(x)>1e-4; shift retained contiguous content to start and zero-pad to original length | Removes low-energy tails, moves content across six segment boundaries; semantic preservation unverified |

Each policy is applied **before** channel generation. Reference noise power is anchored once to that policy's prepared clean RMS and shared across the two execution orders. The sample-derived unit-noise realization/seed is preserved; absolute powers can differ across preprocessing policies. There is no post-noise/post-codec normalization. The experiment therefore measures combined preprocessing/channel sensitivity, not isolated causal effects of silence or pure compression.

Executed: 288 shared new waveforms, 576 successful new predictions, 448 sealed baseline scores reused. Both artifacts retain strong noisy-chain FAKE misses after controls; edge and amplitude interventions also change decisions. Complete FPR/FNR/error, balanced error, AUC/EER, paired score shifts, distribution quantiles and fixed-threshold flips are persisted. No new threshold was selected.

## Exact feature-level diagnostic

All 32 half-amplitude production waveforms match baseline×.5 exactly. Nonetheless the 0–255 Mel/RGB tensor changes by up to **19.190689**, mean absolute change **3.440610**. Production floors absolute Mel power at 1e-10 before subtracting each segment's dB maximum. Uniform attenuation therefore does not commute with this floor.

A separately implemented model-free replica exactly matches the production tensor at the original floor. On the half waveform, changing only the diagnostic floor to 2.5e-11 reduces maximum tensor residual to **0.0000610352** and mean residual to **0.0000081769**. This isolates the floor contribution to feature gain non-invariance up to floating precision. **No predictor used that counterfactual floor**, and production preprocessing was not changed. The result does not establish an improved detector or identify how much of each prediction flip is caused by this feature effect.

Evidence: signal_features.csv and condition intervention metadata in the new waveform cache; phase4_report_v1/table3_nuisance_diagnostics.csv; phase4_feature_diagnostics_v2/ with code/input hashes. Existing Phase 3 nuisance-classifier scores show available label cues; this intervention study cannot turn those into proof of detector shortcut causality. Prior art already studies silence/level shortcuts; see PHASE4_NOVELTY_DECISION.md.
