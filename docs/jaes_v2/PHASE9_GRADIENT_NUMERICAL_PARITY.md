# Phase 9 — Gradient numerical parity

## Bounded measurement

The audit used TensorFlow 2.15.0 CPU, the sealed ERM warmup weights, one REAL clean view (LA_D_1097170), and one FAKE Opus-64 view (LA_D_1002910). Within each case, full-sequence autodiff, per-segment VJP, and per-segment VJP with isolated pointwise MatMul used the same input, target, seed, BCE, frozen BN and fresh Adam state. The comparison resets weights between paths and checks all 646,481 trainable scalars. The pointwise view shares variables and leaves the original convolution methods unchanged. The audited full model remains the inference/export model.

| Input and path vs full sequence | Max abs gradient | Global relative L2 | Worst tensor relative L2 | Max abs Adam delta difference |
| --- | ---: | ---: | ---: | ---: |
| REAL clean, segmentwise | 6.557e-6 | 2.0881e-6 | 2.234e-6 | 1.095e-6 |
| REAL clean, pointwise | 6.557e-6 | 2.0884e-6 | 2.234e-6 | 1.095e-6 |
| FAKE Opus, segmentwise | 6.199e-6 | 1.3164e-6 | 1.467e-6 | 8.009e-7 |
| FAKE Opus, pointwise | 6.199e-6 | 1.3169e-6 | 1.467e-6 | 8.009e-7 |

Global cosine exceeded 0.999999999999 in all four comparisons, all gradients were finite, frozen tensors were unchanged, worst loss difference was 9.537e-7, and worst relative Adam-delta difference was 5.970e-5. The audit records shapes, per-tensor cosine, norms and post-update weights. Cold/hot gradient times were 4.080/0.090 s for full sequence, 3.797/0.205 s for segmentwise and 3.390/0.226 s for pointwise. The first call includes tracing and the cases differ. These measurements provide neither throughput estimates nor statistical error bars. [Audit JSON](../../outputs/jaes_v2/phase9/parity_v1.json) · [Probe](../../scripts/jaes_v2/phase9_parity.py)

## Freeze decision

`GRADIENT_PARITY_PASS=NO` at experiment scope. Advance isolated pointwise and pure segmentwise to a fresh same-engine pilot. Predeclare tolerances for nonzero gradients, near-zero tensors, optimizer slots and score drift using an independent engineering fixture. Exercise clean and paired objectives, warmup and partial stages, saturated predictions, more than one optimizer state, every tensor, and uninterrupted versus interrupted/restarted execution. Compare update, optimizer, sampler, RNG, selected checkpoint, score and export hashes for same-engine resume. Record error and throughput separately from cross-engine floating-point tolerances. Choose a single path before creating new scientific run IDs. On OOM, preserve the last committed update and stop. Resume only under the identical fingerprint and resource contract. An engine, numerical backend, microbatch, cache or preprocessing change that affects outputs requires a fresh experiment ID. The historical Phase 8 runs remain exploratory.
