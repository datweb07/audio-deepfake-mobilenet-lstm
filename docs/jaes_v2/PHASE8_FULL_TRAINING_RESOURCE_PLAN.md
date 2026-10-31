# Full-training resource plan

Observed host:12logicalCPUs, RTX3050Laptop4GiB/driver595.71, native Windows TF2.15CPU. Current D free at evidence generation=25162997760bytes; C=3371966464bytes. The CLI redirects process temp and plotting cache to D, preserves15GiB reserve and uses the existing single-heavy-job OS lock. No dependency/environment installation or historical cleanup occurred.

## Measured tiny invocations

| experiment_id | invocation_seconds | updates_seconds | validation_seconds | peak_MiB | checkpoint_bytes |
| --- | --- | --- | --- | --- | --- |
| p8_tiny_dro_head_s6107_v2 | 17.430957799999305 | 1.7973471999939648 | 2.5677473999967333 | 594.1328125 | 49531554 |
| p8_tiny_dro_head_s6107_v3 | 16.435690300000715 | 1.7677022999996552 | 1.922864500000287 | 594.9453125 | 49531554 |
| p8_tiny_erm_head_s6107_resume_v2 | 17.642306599998847 | 1.135863999996218 | 2.24513239999942 | 593.77734375 | 49525461 |
| p8_tiny_erm_head_s6107_resume_v3 | 16.947843900001317 | 1.0145000999982585 | 2.0570679000011296 | 596.5703125 | 49525461 |
| p8_tiny_erm_head_s6107_v1 | 23.066664599999058 | 6.379174699999567 | 2.302599899998313 | 671.62109375 | 49538621 |
| p8_tiny_erm_head_s6107_v2 | 16.92577920000258 | 1.4337328000001435 | 2.0069834000023548 | 591.5234375 | 49525461 |
| p8_tiny_erm_head_s6107_v3 | 16.48634889999812 | 1.3439178000007814 | 1.7161124999984168 | 594.984375 | 49525461 |
| p8_tiny_erm_head_s6207_v2 | 18.01922530000229 | 1.4623159000002488 | 2.5548464999992575 | 592.51171875 | 49525481 |
| p8_tiny_erm_head_s6207_v3 | 17.50609700000132 | 1.3573063999974693 | 2.3128409999953874 | 593.30859375 | 49525481 |
| p8_tiny_erm_head_s6307_v2 | 17.845343399996636 | 1.2716087999979209 | 2.9973333999987517 | 592.09375 | 49525585 |
| p8_tiny_erm_head_s6307_v3 | 18.84009789999982 | 1.686547100005555 | 2.9735313999954087 | 592.359375 | 49525585 |
| p8_tiny_erm_partial_s6107_resume_v2 | 129.93436350000047 | 6.786689700002171 | 113.72618629999852 | 593.23046875 | 62844760 |
| p8_tiny_erm_partial_s6107_resume_v3 | 73.83351319999929 | 6.037354800002504 | 57.79424740000104 | 593.48046875 | 62844760 |
| p8_tiny_erm_partial_s6107_v2 | 134.23009629999797 | 8.500418499999796 | 115.08173810000153 | 592.9296875 | 62844760 |
| p8_tiny_erm_partial_s6107_v3 | 76.1902973000033 | 9.538251900001342 | 56.27032290000352 | 598.1171875 | 62844760 |
| p8_tiny_weighted_head_s6107_v2 | 16.792202099997667 | 1.477954399997543 | 1.8238217999969493 | 593.53515625 | 49531525 |
| p8_tiny_weighted_head_s6107_v3 | 17.958572800002003 | 1.4712383999940357 | 1.6652536000037799 | 593.64453125 | 49531525 |

Resume rows measure the resumed invocation only, excluding earlier process time. Source-version repeats and counterpart pairs are not independent quality replications. OS lifetime peak includes allocator/activation effects but does not isolate activations. GPU peak VRAM/time is NOT_MEASURED; nvidia-smi free memory is inventory only. Step timing includes gradient/optimizer and input preparation; validation timing includes its transforms/model/metrics. Input decode, transforms, Mel, compiled forwards, metrics, checkpoint and startup components are separately retained where instrumented. Components overlap and must not be summed twice. Figure generation is in the separate model-free report process, never inside training timing.

Original extracted audio uses3,178,404,021bytes; user LA.zip7,646,913,333bytes is retained, not copied or re-extracted. The existing Phase7 cache contains6,416embeddings /88,694,784rawfloat bytes and cost1776.57s to prepare. Head times exclude that shared historical cost. New partial training holds at most8P0 tensors plus one source waveform bank in RAM; it does not create a multi-GB image cache. Each full P0 tensor is6×224×224×3float32 (~3.45MiB);8entries about27.6MiB before framework activations. Final model exports are approximately6MB each, while Adam checkpoints are larger; measured files, not model size, define actual storage.

Validation optimization comparisons:

```json
{
  "erm_head_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 16.92577920000258,
    "v3_seconds": 16.48634889999812,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  },
  "erm_partial_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 134.23009629999797,
    "v3_seconds": 76.1902973000033,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  },
  "weighted_head_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 16.792202099997667,
    "v3_seconds": 17.958572800002003,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  },
  "dro_head_s6107": {
    "max_score_residual": 0.0,
    "v2_seconds": 17.430957799999305,
    "v3_seconds": 16.435690300000715,
    "scope": "Same12train/14dev, weights/objective fixed; implementation benchmark, not independent scientific replication"
  }
}
```

Source-first evaluation reuses one deterministic waveform bank instead of regenerating all codec chains for each requested condition. This is fixed-view caching within a traversal, not fresh online augmentation or a performance result for a new model. Prediction parity is checked by source/condition, independent of CSV row order.

## Warmed CPU benchmark and illustrative time projection

```json
{
  "scope": "RESOURCE_BENCHMARK_ONLY; no quality comparison",
  "runtime": "CPU",
  "microbatch": 1,
  "effective_batch": 8,
  "sources": [
    "ASVspoof2019_LA:LA_T_1021790",
    "ASVspoof2019_LA:LA_T_1064484",
    "ASVspoof2019_LA:LA_T_1076943",
    "ASVspoof2019_LA:LA_T_1087973",
    "ASVspoof2019_LA:LA_T_1137942",
    "ASVspoof2019_LA:LA_T_1152394",
    "ASVspoof2019_LA:LA_T_1161711",
    "ASVspoof2019_LA:LA_T_1199930"
  ],
  "preparation_seconds": 0.14199619999999413,
  "input_components": {
    "decode_seconds": 0.02979820000109612,
    "transform_seconds": 0.0004695000025094487,
    "mel_seconds": 0.11145949999627192
  },
  "warmup_update_seconds": 4.625684800001181,
  "steady_update_seconds": [
    0.9374950000019453,
    1.496697900001891,
    1.0480618000001414,
    0.7952495999998064,
    0.8152121000020998
  ],
  "steady_update_median": 0.9374950000019453,
  "forward_warmup_seconds": 2.145351200000732,
  "steady_forward_median": 0.09477520000291406,
  "checkpoint_save_seconds": 0.3031520000004093,
  "checkpoint_bytes": 10470234,
  "peak_process_mib": 556.83203125,
  "total_parameters": 1308401,
  "trainable_parameters": 646481,
  "BN_unchanged": true,
  "GPU_memory": null
}
```

```json
{
  "foundation_erm_partial_v1.json": {
    "status": "EXTRAPOLATED_NOT_FULL_RUN_MEASUREMENT",
    "training_update_seconds": 62487.79173012966,
    "uncached_train_preparation_seconds": 9460.14183449961,
    "validation_forward_seconds": 44483.40354776774,
    "development_transform_seconds": 25479.76933679566,
    "total_seconds": 141911.10644919268,
    "assumptions": "ERM partial only; microbatch1; measured8source steady update scaled to16; warmup approximated by partial backward cost; no thermal/IO/long-run guarantee"
  },
  "medium_erm_partial_v1.json": {
    "status": "EXTRAPOLATED_NOT_FULL_RUN_MEASUREMENT",
    "training_update_seconds": 1574.991600003268,
    "uncached_train_preparation_seconds": 238.55361599999014,
    "validation_forward_seconds": 689.9634560212144,
    "development_transform_seconds": 535.7113132572017,
    "total_seconds": 3039.2199852816743,
    "assumptions": "ERM partial only; microbatch1; measured8source steady update scaled to16; warmup approximated by partial backward cost; no thermal/IO/long-run guarantee"
  }
}
```

The benchmark performs6effective8-source partial-CNN updates; first tracing update is separated from5steady repetitions. Full forward timing similarly excludes its first trace. Time estimates apply only to the specified ERM21epoch schedule, separately scaling training, preparation, clean-epoch/final-view forwards and development transforms. Reused tiny inputs, cache behavior, thermal throttling and large-corpus I/O limit the extrapolation; these are not measured full epoch/run times. Channel/DRO/RawBoost require their own resource calibration and view accounting. Invocation guards remain authoritative.

## Preflight projections (not full-job measurements)

| config | train | development | updates | retained_bytes | status |
| --- | --- | --- | --- | --- | --- |
| 1777ebe9283657b532681bfd816b1f4a451bee6f3a4d7b212c868fd15fc0d5d0 | 25380 | 17503 | 33327 | 3187671040 | PASS |
| 330c06b03fd4f650249f2d3f5ba08c1a766ac61576fc2d64838fa1253643ff01 | 640 | 368 | 840 | 1551892480 | PASS |

Retained-size projection assumes24MiB per boundary snapshot plus8MiB per possible epoch export and256MiB fixed overhead; checkpoint cadence500updates plus epoch boundaries. Each job's preflight is independent; sum projections for the intended seed/method matrix and recheck remaining disk before each launch. Do not multiply this single-run PASS into a guarantee that all families fit. No existing checkpoint is automatically deleted. Full-foundation21epochs have approximately33,327updates at effective batch16, before early stopping; medium21epochs840updates. Natural final partial batches use their actual denominator. Group-balanced/oversampling changes source exposure but not the unique recording count.

For a coarse CPU planning estimate, scale the measured tiny **update-path seconds / actual source exposures** by the planned source exposures, and separately add clean-epoch validation plus full final evaluation. This includes first-trace overhead and small-cohort cache effects and is not a reliable long-run ETA. Full-foundation final evaluation has17,503×7=122,521views, whereas a tiny final has14×7=98; epoch primary validation is an additional cost. Long-run epoch time, complete multi-seed wall time, GPU speedup and sustained memory remain NOT_MEASURED. Use the first medium run and journal-based monitor to refine estimates. Jobs may need repeated24h invocations with resume; final evaluation restarts if interrupted. Three-seed storage must be budgeted explicitly and jobs run sequentially.

Do not copy data into WSL or install CUDA/WSL in this phase. See GPU_TRAINING_GUIDE for the isolated portability path. Disk/RAM/NaN/power recovery and author commands are specified in FULL_TRAINING_LAUNCH_PACKAGE.
