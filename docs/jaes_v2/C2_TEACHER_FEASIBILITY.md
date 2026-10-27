# C2 teacher feasibility and prior-art gate

## Phase 3 update

The real public development cohort/cache now exists. Existing artifacts show noisy-chain FAKE misses, aggregate class cancellation and preprocessing sensitivity. A candidate teacher needs classwise quality on this distribution and its channel views; checkpoint availability/clean accuracy alone is insufficient.

Weight license/hash, isolated loading, measured inference feasibility and a mechanism distinct from PCL-NET/DK-CAST remain pending. No teacher weights were downloaded, environment upgraded or C2 learner trained. C3 still needs a validated student/independent calibration. The inventory below is historical evidence.

No C2 loss/training or C3 policy was implemented in Phase 2.

Candidate: the authors' [SSL_Anti-spoofing implementation](https://github.com/TakHemlata/SSL_Anti-spoofing) of the wav2vec 2.0/RawBoost anti-spoofing system. The repository identifies ASVspoof 2019 LA training, 2021 LA/DF evaluation, links XLS-R-300M frontend weights and a folder of anti-spoofing checkpoints. The **code** license is MIT. Checkpoint/frontend-weight license and redistributability must be verified separately; code licensing is not a blanket weight license.

| Gate | Status |
|---|---|
| Candidate exists with author-linked checkpoint locations | VERIFIED_LINKS; no checkpoint downloaded |
| Exact artifact version/hash/license | NOT_VERIFIED |
| Loads in an isolated compatible environment | NOT_RUN |
| Performs well on a valid development cohort | NOT_RUN; no suitable external cohort installed |
| Suitable for distillation | NOT_ESTABLISHED |

The reference setup uses an old Python/Torch/fairseq combination; the current isolated environment is Python 3.11 / Torch 2.2.2+cpu. No fairseq/CUDA upgrades were attempted. A 300M-parameter float32 frontend alone has approximately 1.2 GB of parameter storage (arithmetic estimate, not a measured checkpoint or peak-memory result). Full-gradient training plus optimizer/activations needs a bounded measurement and cannot be assumed to fit a 4 GiB device. Frozen CPU teacher inference/cached development features may be feasible but remain unmeasured.

Student candidate remains existing MobileNetV3-LSTM (1,308,401 parameters in artifact metadata), with ShuffleNet as an optional second artifact. Current single-pass Phase 2 latencies are diagnostic measurements, not a new deployment benchmark. C2's future student cost must include any added inference components; teacher training/inference preprocessing costs must be disclosed.

PCL-NET already uses original/codec pair consistency; DK-CAST already uses distortion-aware teacher–student mechanisms. The next C2 decision must distinguish a mechanism supported by the confounder-controlled C1 study from these methods. Downloadability alone does not justify a teacher or a new loss. No teacher performance or distillation benefit is asserted.

C3 still requires a validated student, independent calibration data and total-cost measurement. Planned baselines remain confidence/entropy rejection, temperature scaling, fixed ensemble, standard two-stage cascade and random matched-cost routing. Primary comparisons must match coverage and measured cost; no domain-shift guarantee is assumed.
