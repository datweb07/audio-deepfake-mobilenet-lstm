# Phase6 speaker/class selection audit

Selection seed5806; four recordings per speaker/class from genuinely available cells. Train has20 shared speakers; dev has7. The other9 dev speakers are excluded because no FAKE support exists. This defines a narrower target population, not fabricated balance. Official train/dev boundaries remain; calibration/evaluation paths are not loaded. Selected variants cannot cross partitions. Byte/parent/speaker and exact selected PCM conflicts passed; original utterance and exact generator lineage remain UNKNOWN.

Small cohort has160 train and56 dev, an increase in training source count relative to Phase5 but still tiny. The per-speaker round-robin attack queue selects A01–A04 at this cell budget; A05/A06 are absent. This limitation prevents all-attack conclusions. The larger plans contain all6 available attack categories. Selection never uses scores.

Foundation speaker-by-class contingency:

| Speaker | Role | REAL available | FAKE available | Attack categories |
|---|---|---|---|---|
| LA_0079 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0080 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0081 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0082 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0083 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0084 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0085 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0086 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0087 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0088 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0089 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0090 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0091 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0092 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0093 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0094 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0095 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0096 | train | 132 | 1176 | A01,A02,A03,A04,A05,A06 |
| LA_0097 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0098 | train | 127 | 1116 | A01,A02,A03,A04,A05,A06 |
| LA_0069 | development | 154 | 2484 | A01,A02,A03,A04,A05,A06 |
| LA_0070 | development | 140 | 1848 | A01,A02,A03,A04,A05,A06 |
| LA_0071 | development | 140 | 1848 | A01,A02,A03,A04,A05,A06 |
| LA_0072 | development | 154 | 2484 | A01,A02,A03,A04,A05,A06 |
| LA_0075 | development | 154 | 2484 | A01,A02,A03,A04,A05,A06 |
| LA_0076 | development | 140 | 1848 | A01,A02,A03,A04,A05,A06 |
| LA_0077 | development | 154 | 2484 | A01,A02,A03,A04,A05,A06 |
| LA_0099 | development | 77 | 0 |  |
| LA_0100 | development | 77 | 0 |  |
| LA_0102 | development | 126 | 0 |  |
| LA_0103 | development | 126 | 0 |  |
| LA_0104 | development | 126 | 0 |  |
| LA_0105 | development | 77 | 0 |  |
| LA_0106 | development | 126 | 0 |  |
| LA_0107 | development | 126 | 0 |  |
| LA_0108 | development | 126 | 0 |  |

Actual selection and size alternatives:

```json
{
  "selection": {
    "status": "LOCKED_SPEAKER_BALANCED_DEVELOPMENT",
    "seed": 5806,
    "per_speaker_per_class": 4,
    "foundation_hashes": {
      "train": "7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72",
      "development": "dcbcf4dd3583c24590246323f2e8fd943fcc4607030698069c9a0aa11b25a55a"
    },
    "counts": {
      "train": {
        "n": 160,
        "speakers": 20,
        "real": 80,
        "fake": 80,
        "attacks": [
          "A01",
          "A02",
          "A03",
          "A04"
        ]
      },
      "development": {
        "n": 56,
        "speakers": 7,
        "real": 28,
        "fake": 28,
        "attacks": [
          "A01",
          "A02",
          "A03",
          "A04"
        ]
      }
    },
    "exclusions": "Development speakers without FAKE support excluded prospectively; not representative full official dev",
    "unknown_lineage": "Original utterance/generator checkpoint UNKNOWN",
    "decoded_duplicate_groups": 0,
    "calibration_accessed": false,
    "evaluation_accessed": false,
    "manifest_sha256": "45ff123a734e11935e9e7f7a1ee9259f950cc82f7441bf06b6ca96cd56d87965"
  },
  "scaling": [
    {
      "tier": "small",
      "per_cell": 4,
      "counts": {
        "train": {
          "recordings": 160,
          "real": 80,
          "fake": 80,
          "speakers": 20,
          "attacks": 4
        },
        "development": {
          "recordings": 56,
          "real": 28,
          "fake": 28,
          "speakers": 7,
          "attacks": 4
        }
      },
      "embedding_count": 2704,
      "embedding_float32_bytes": 37380096,
      "projected_cache_seconds": 680.1762664003958,
      "training_estimate": "Measure actual optimized small-run throughput before extrapolation",
      "decision": "EXECUTE_SINGLE_SEED_BOUNDED",
      "independence": "More records do not create new original-source identity or more speakers"
    },
    {
      "tier": "medium",
      "per_cell": 16,
      "counts": {
        "train": {
          "recordings": 640,
          "real": 320,
          "fake": 320,
          "speakers": 20,
          "attacks": 6
        },
        "development": {
          "recordings": 224,
          "real": 112,
          "fake": 112,
          "speakers": 7,
          "attacks": 6
        }
      },
      "embedding_count": 10816,
      "embedding_float32_bytes": 149520384,
      "projected_cache_seconds": 2420.705065601583,
      "training_estimate": "Measure actual optimized small-run throughput before extrapolation",
      "decision": "NOT_EXECUTED_AWAITING_COMPUTE_BUDGET",
      "independence": "More records do not create new original-source identity or more speakers"
    },
    {
      "tier": "large",
      "per_cell": 64,
      "counts": {
        "train": {
          "recordings": 2560,
          "real": 1280,
          "fake": 1280,
          "speakers": 20,
          "attacks": 6
        },
        "development": {
          "recordings": 896,
          "real": 448,
          "fake": 448,
          "speakers": 7,
          "attacks": 6
        }
      },
      "embedding_count": 43264,
      "embedding_float32_bytes": 598081536,
      "projected_cache_seconds": 9382.820262406332,
      "training_estimate": "Measure actual optimized small-run throughput before extrapolation",
      "decision": "NOT_EXECUTED_AWAITING_COMPUTE_BUDGET",
      "independence": "More records do not create new original-source identity or more speakers"
    }
  ]
}
```
