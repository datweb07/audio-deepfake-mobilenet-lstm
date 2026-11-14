# R08 cross-dataset results

CROSS_DATASET_DIAGNOSTIC_WITH_LINEAGE_LIMITATIONS

Verified predictions: 3072. Speaker groups: 16.

| Model | ROC-AUC | EER | REAL false alarms | FAKE misses | FPR | FNR | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| erm_9107 | 0.3061 | 0.6523 | 234/256 | 94/256 | 0.9141 | 0.3672 | 22/234/94/162 |
| channel_9107 | 0.4997 | 0.5098 | 200/256 | 73/256 | 0.7812 | 0.2852 | 56/200/73/183 |
| erm_9207 | 0.3370 | 0.6289 | 238/256 | 66/256 | 0.9297 | 0.2578 | 18/238/66/190 |
| channel_9207 | 0.4364 | 0.5566 | 221/256 | 67/256 | 0.8633 | 0.2617 | 35/221/67/189 |
| erm_9307 | 0.3742 | 0.5996 | 230/256 | 84/256 | 0.8984 | 0.3281 | 26/230/84/172 |
| channel_9307 | 0.4635 | 0.5234 | 223/256 | 63/256 | 0.8711 | 0.2461 | 33/223/63/193 |

| Seed | Channel−ERM AUC | ΔFPR | ΔFNR |
| --- | ---: | ---: | ---: |
| 9107 | +0.1936 | -0.1328 | -0.0820 |
| 9207 | +0.0994 | -0.0664 | +0.0039 |
| 9307 | +0.0892 | -0.0273 | -0.0820 |

Paired transitions, across-seed mean and sample SD and original clean-domain references are in metrics.json.
These native-view results use the locked 0.5 threshold. Across-corpus differences describe distribution shift.
Parent lineage remains unresolved. Speaker-bootstrap publication intervals are withheld, and the three seeds are descriptive.
