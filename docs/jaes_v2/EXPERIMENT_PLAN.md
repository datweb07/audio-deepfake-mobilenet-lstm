# Experiment plan version 1

This plan separates the executed pilot from a future locked confirmation. No journal hypothesis has been accepted on pilot performance.

## Executed feasibility experiment

Config: `configs/jaes_v2/c1_pilot_v3.json`; 12 legacy validation recordings, 6/class; selection and transform seed 804; one existing MobileNet checkpoint. A = 10 dB whole-observed-waveform AWGN; B = conventional Opus 24 kb/s FFmpeg round trip. Conditions: clean, A, B, A_then_B, B_then_A, where names mean execution left to right (A then B mathematically applies B(A(x))). Model threshold stays at legacy validation value 0.82.

Common preparation: decode/channel mean, polyphase resample to 22,050 Hz, take at most 3 s; apply a per-source attenuation `min(1, 0.25/peak)` shared by every condition, including clean. No per-condition output normalization. This headroom is part of the pilot protocol; it prevents PCM16 input clipping and makes the pilot clean path different from unmodified legacy evaluation. Short clips are padded by the unchanged detector preprocessing, not by this transform engine.

Noise realization derives from sample ID, transform key and seed, so the same base noise draw is reused across orders. Its amplitude is scaled to the signal power **at its insertion point**. Thus reversal also changes power normalization; confirmation must compare a fixed absolute-noise-power sensitivity condition to distinguish this effect. Conditions save measured SNR, actual rates/lengths, peak, PCM input policy and encoder metadata. FFmpeg container trimming handles codec timestamps; residual waveform delay is NOT_MEASURED. Confirmation needs impulse/correlation delay checks and matched quantization/resampling controls.

Risk diagnostics: fixed-threshold classification error and raw-score squared error; clipped binary log loss is a sensitivity measure (epsilon 1e-7). Scores are not calibrated probabilities. Interaction = R_AB−R_A−R_B+R_clean; order = R_AB−R_BA. Contrasts are calculated per sample and averaged within local parent, then equally across parents. Bootstrap resamples those parents, preserving all five variants. 2,000 percentile draws produce descriptive 95% intervals, not adjusted hypothesis tests. Single-class contrasts are separate. Failed/missing scores invalidate paired analysis instead of being silently deleted.

## Confirmation to register before execution

1. Acquire metadata-rich training/development/calibration and independent test cohorts. Audit both classes, overlap and license. Identify seen/unseen generator cohorts and held-out chain families before fitting.
2. Select a small physically meaningful factorial grid: seen/unseen generators × clean/held-out channels, plus reversible noise/codec and RIR/codec orders. Avoid conflating corpus with generator. Use matched parents for channel contrasts and a justified group-level design across different generators.
3. Lock primary raw-score risk and class error estimands, cohort weights, calibration/threshold criteria, interactions, correction family, sample-size rationale and failure criteria. C1 pilot results are exploratory; they must not be used to tune final test selection.
4. Controlled baselines: same data access and reasonable tuning budget; native valid preprocessing; include a strong contemporary SSL detector. At least three independent training seeds for any trained-method claim. Legacy checkpoints remain a separate artifact-transfer stratum.
5. C2 comparisons: ERM, same-views standard augmentation, standard KD, paired consistency, relevant standard group-robust objective and only then a technically distinct proposed mechanism. Teacher quality/license must pass first.
6. C3 comparisons: confidence/entropy rejection, calibrated confidence, fixed ensemble, standard cascade and random matched-cost routing. Fit on development/calibration only; compare matched coverage and total measured cost. No arbitrary-shift conformal guarantee.

Do not run the confirmation using this pilot runner by relabeling scope. It currently cannot support generator-held-out inference.
