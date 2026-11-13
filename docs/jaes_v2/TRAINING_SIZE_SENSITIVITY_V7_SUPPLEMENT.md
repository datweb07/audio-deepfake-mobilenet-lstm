# V7 training-size supplement

The original [size report](TRAINING_SIZE_SENSITIVITY_RESULTS.md) gives endpoint counts, intervals, guard decisions and costs. Its [verified JSON](../../outputs/jaes_v2/phase10/size1280_v1/size_sensitivity.json) is the numerical source for this supplement.

## Directional decision transitions

First order is Noise10 followed by Opus24. Second order reverses the same operators on the same recording. A wrong REAL decision is a false alarm. A wrong FAKE decision is a miss.

| Group | Arm | Sources | Both correct | First only wrong | Second only wrong | Both wrong | Mean FAKE score gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_PAIRED/0 | erm640 | 112 | 8 | 63 | 0 | 41 | +0.3461 |
| A_PAIRED/0 | channel640 | 112 | 68 | 31 | 0 | 13 | +0.1362 |
| A_PAIRED/0 | erm1280 | 112 | 108 | 3 | 0 | 1 | +0.0764 |
| A_PAIRED/0 | channel1280 | 112 | 39 | 33 | 0 | 40 | +0.1528 |
| A_PAIRED/1 | erm640 | 112 | 59 | 0 | 40 | 13 | +0.2859 |
| A_PAIRED/1 | channel640 | 112 | 60 | 0 | 26 | 26 | +0.0847 |
| A_PAIRED/1 | erm1280 | 112 | 27 | 0 | 21 | 64 | +0.2008 |
| A_PAIRED/1 | channel1280 | 112 | 85 | 0 | 18 | 9 | +0.0825 |
| B_REAL_ONLY/0 | erm640 | 144 | 17 | 98 | 1 | 28 | +0.3452 |
| B_REAL_ONLY/0 | channel640 | 144 | 91 | 35 | 0 | 18 | +0.1304 |
| B_REAL_ONLY/0 | erm1280 | 144 | 137 | 6 | 0 | 1 | +0.0763 |
| B_REAL_ONLY/0 | channel1280 | 144 | 34 | 51 | 3 | 56 | +0.1506 |

## Interpretation and replay

The 640-to-1,280 shift in the A REAL signed order-error contrast is +55.36 percentage points. Its A FAKE counterpart shifts by −9.82 points. These are responses of the fitted detectors to the tested signal chains. Standalone component diagnostics remain limited to the original 640 seed blocks 9107 and 9207.

The selected 1,280 Channel model exceeds the +2-point tolerances for clean A FAKE misses and clean B REAL false alarms. The B interval contains zero. The registered guard uses the observed difference and does not change the selected CLEAN_BCE checkpoint. The 9307 B guard concerns a separate 640-recording comparison.

All contrasts use shared development and seed 9207. The fixed-epoch design doubles optimizer updates. Pointwise speaker intervals retain all views per speaker and condition on the fitted models. Broader training and external-corpus uncertainty remain unmeasured.

Replay with `python -B -X utf8 -m scripts.jaes_v2.phase10_size1280 compare` in the complete local repository. The four score grids, run contracts, seal digests and analysis code support independent score replay. The [figure and table provenance](../../papers/jaes_v2/jaes_manuscript/size_sensitivity_asset_provenance.json) binds the rendered assets to the verified JSON. Raw recordings and model checkpoints stay in the local research workspace.
