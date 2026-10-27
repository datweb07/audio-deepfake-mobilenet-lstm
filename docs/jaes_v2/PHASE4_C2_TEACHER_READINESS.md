# Phase 4 C2 teacher feasibility — 27 October 2026

Status: **AUDITED, NOT LOADED OR QUALITY-VALIDATED**. No teacher weights, new runtime packages, student training or C3 policy were installed or executed in this workstream. The investigation used author source/metadata plus a read-only probe of the existing `.venv-torch` environment. Machine record: `outputs/jaes_v2/audit/phase4_teacher_audit.json`.

## Primary candidate: Tak SSL anti-spoofing

The [author repository](https://github.com/TakHemlata/SSL_Anti-spoofing) links a pretrained anti-spoofing folder and a separate XLS-R frontend. Its code is MIT, but an explicit grant covering the linked anti-spoofing checkpoint was not verified. Exact checkpoint bytes, SHA-256 and size remain UNKNOWN. The documented reference environment is Python 3.7 / Torch 1.8.1 / pinned fairseq; training uses ASVspoof 2019 LA, with 2021 LA/DF evaluation. These details do not establish performance on the current development data.

The [dataset code](https://raw.githubusercontent.com/TakHemlata/SSL_Anti-spoofing/main/data_utils_SSL.py) assigns bonafide=1 and spoof=0, decodes 16 kHz, crops to 64,600 samples and repeats shorter clips. The [evaluation writer](https://raw.githubusercontent.com/TakHemlata/SSL_Anti-spoofing/main/main_SSL_LA.py) exports class 1. Thus its exported score is bona-fide-oriented, unlike LAVA's fake-oriented contract. Negation can orient a ranking score; a bounded fake score would require the full two-class output and an explicitly tested conversion. Subtracting an arbitrary raw score from one is invalid. [Model source](https://raw.githubusercontent.com/TakHemlata/SSL_Anti-spoofing/main/model.py) also loads a separate `xlsr2_300m.pt` artifact.

The [Meta frontend card](https://huggingface.co/facebook/wav2vec2-xls-r-300m) declares Apache-2.0 and 16 kHz input. A frontend's license is not a license for the fine-tuned detector, and a representation model alone is not a validated anti-spoofing teacher.

## Author alternative with explicit weight terms

[NII wav2vec-small-anti-deepfake](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake) provides a 95M-parameter alternative. The card explicitly licenses weights for noncommercial research/education under CC-BY-NC-SA-4.0. Its example takes mono 16 kHz audio, applies waveform layer normalization, permits variable length and assigns softmax index 0 to FAKE. It uses a custom fairseq encoder/classifier wrapper, so a generic Transformers pretrained-model call is not a verified replacement.

The official training table includes MLAAD, ASVspoof 2019/2021/5 and LibriTTS among other corpora. Therefore MLAAD-tiny teacher evaluation would be an exploratory, potentially training-overlapping diagnostic, not certified unseen-corpus or independent validation. Exact source/checkpoint-relative overlap remains unknown.

The [published safetensors page](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake/blob/main/model.safetensors) advertises approximately 380 MB and SHA-256 `828ee456122f86d5d631cb7895a10e5c62c78a4fcb8a8b1c42cb5838a9abcfe0`. The inspected repository history identifies `9a13264b5dcc827a8d5a4f8e01fccefa392f886b`; pinned-file retrieval was unavailable through the web tool, so commit-to-file content identity is not independently verified. No local weight hash is claimed.

NII separates [checkpoint licensing](https://raw.githubusercontent.com/nii-yamagishilab/AntiDeepfake/main/LICENSE-CHECKPOINT) from [BSD-3-Clause code licensing](https://raw.githubusercontent.com/nii-yamagishilab/AntiDeepfake/main/LICENSE-CODE). Preserve attribution, noncommercial conditions and applicable share-alike obligations for derived weights. The [author implementation](https://github.com/nii-yamagishilab/AntiDeepfake) documents an isolated Python 3.9 environment and a specific fairseq revision; its current installation recipe differs from the older Tak setup. The [paper](https://arxiv.org/abs/2506.21090) is post-training prior art, not a new LAVA method.

## Measured local compatibility and bounded next gate

The existing isolated runtime reports Python 3.11.9, Torch 2.2.2+cpu, CUDA unavailable, no fairseq and no Transformers. Free disk at the probe was 39,125,184,512 bytes. A physical GPU inventory does not supply a compatible Torch CUDA runtime. No production dependency was upgraded. Actual load success, CPU/GPU peak memory, inference latency and teacher discrimination are **NOT_MEASURED**.

The 95M float32 parameter arithmetic is approximately 380 MB; 300M is approximately 1.2 GB. Neither estimate includes activations, loader copies, allocator overhead or training state, and neither proves a 4 GiB training/inference fit. The alternative resolves a weight-terms ambiguity, but currently fails the installed-runtime and independent-quality gates. Accordingly no weight download or dummy teacher score was produced.

Before a bounded teacher run: pin the exact artifact and upstream hash; retain legal notices; reproduce the author wrapper in a separate compatible environment; validate strict state-dict loading and score orientation; cap batch and duration; then measure CPU memory/runtime on a small, sealed, two-class development subset and the same channel waveforms. Keep teacher resampling/layer normalization distinct from LAVA's Mel contract. Report classwise failure and ranking separately and disclose training overlap. Confirmation labels must remain unopened.

## Method decision

Phase 3 establishes artifact failure under noisy channels but does not justify unconditional invariance. Phase 4's source/nuisance controls must establish whether order or class-direction effects remain after source matching. Standard same-view augmentation, ordinary KD and paired consistency are mandatory established comparisons; [PCL-NET](https://arxiv.org/abs/2610.07216) is direct precedent for original/codec consistency. The prior DK-CAST overlap remains a separate unresolved distinction gate. A sum of BCE, KD and consistency is not an established novel contribution.

The defensible transition is a feasibility experiment with documented overlap and no novelty claim, followed by a mechanism decision grounded in the source-controlled results. Full C2 implementation is **NOT_READY** until useful teacher or justified teacher-free learning, controlled training data and a distinct hypothesis exist. C3 remains deferred.
