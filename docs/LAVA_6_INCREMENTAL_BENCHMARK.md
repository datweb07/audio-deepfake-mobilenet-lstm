# LAVA six-detector incremental benchmark

`outputs/lava_6/` combines immutable LAVA-5 measurements with ShuffleNetV2-only
evaluation. No historical detector was retrained or re-evaluated.

Scope:

- clean: 2,737 canonical test samples per detector;
- efficiency: batch 1, float32, one thread, 10 warm-ups, 50 runs;
- robustness: the existing prediction-independent diagnostic subset of 100 samples;
- stress inputs: exact historical AWGN, codec, and simulated-channel files;
- unseen data and physical replay: `NOT_AVAILABLE`;
- official full-test robustness/Pareto: `NOT_RUN`.

ShuffleNet uses `models/shufflenetv2_lstm/`, threshold 0.12, REAL=0/FAKE=1,
and sigmoid `P(FAKE)`. Metadata records scratch initialization, full end-to-end
training from epoch 1, and 56/56 trainable BatchNormalization layers. The training
manifest matches the canonical manifest.

`benchmark/lava6_incremental.py` is the only runner that loads ShuffleNet.
`benchmark/lava6_report.py` never loads a model; it validates stored scores and
regenerates six-model tables, figures, statistics, and diagnostic Pareto outputs.
