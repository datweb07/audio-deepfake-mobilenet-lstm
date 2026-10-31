# C2 method specification

## Problem and assumptions

Let y=0 REAL and y=1 FAKE, pθ(x) the sigmoid FAKE score, and ℓ(y,p)=−y log p−(1−y)log(1−p). Clip probabilities only for numerical stability. Labels are inherited from official ASV protocols; post-generation channel transformations preserve the source label, but may erase detectable evidence. Paired views share a recording, not an independent sample. Official speakers, source byte hashes and A01–A06 are verified fields. They are not verified generator checkpoints or original-utterance lineage. Development examples/conditions and repeated training seeds are dependent observations.

Define R0(c)=E[ℓ(0,pθ(c(x)))|y=0] and R1(a,c)=E[ℓ(1,pθ(c(x)))|y=1,a]. Decision metrics are P(p≥.5|REAL,c) and P(p<.5|FAKE,a,c); BCE caps do not guarantee those rates. The research direction contains clean, noise, codec, noise→codec and codec→noise. Current training controls use clean/gain/noise10/Opus24; ordered chains remain already-used development diagnostics and are not secretly inserted into training.

## M1 — standard baseline

Implemented WEIGHTED_BCE uses L1=(1/N)Σi wyᵢ ℓi, with explicit weights [1,1.5]. It does not divide by the sum of weights. Thus both relative class emphasis and overall gradient scale are declared. This choice is a baseline setting, not optimized or established optimal for the imbalanced foundation. Natural and class-balanced sampling are separate registered protocols. Weighted sampling plus class weights changes the target distribution twice and must be disclosed.

An alternative mathematical specification is minθ Eℓ subject to R0(clean)≤b0 and R1(clean)≤b1. Its standard squared-hinge relaxation adds λΣy[max(0,Ry−by)]². `clean_constraint` has independent gradient tests but is NOT wired into a training family. No claim of a trained primal-dual/constrained optimizer is made. The implemented guarded checkpoint selector is a different operation, described below.

## M2 — standard group-robust control

Groups are REAL×condition plus A01…A06 FAKE×condition; REAL never inherits an UNKNOWN generator label. For the channel bank there are 28 groups. Every effective update draws sources from all seven class/attack groups and expands every sampled source through all four training conditions. Repeated sources remain repeated exposure. Group support and counts are checked before taking a gradient.

L2(θ,q)=Σg qg (1/ng)Σi∈g ℓi. Initialize q uniformly. Train θ using the previous q, then compute pre-update losses on every group and set q′∝q exp(η R̂g); qnew=(1−ε)q′+ε/G. η=.01, ε=.05. This is an explicit smoothed group-DRO baseline, not an exact reproduction of Sagawa Algorithm1 or an unrestricted simplex-max guarantee. The floor prevents zero weights; it does not guarantee optimization or population robustness. There is no additional ERM term. Each microbatch receives coefficients qg/ng computed over the entire effective update, so accumulation cannot accidentally normalize every microbatch separately.

M2 changes sampling as well as weighting. A matched uniform-group control is required before attributing an effect solely to adversarial q. All-view exposure also costs more than a single-view ERM update. Compare supervised views, forward/backward calls and runtime; equal update counts are not equal compute.

## M3 — proposed mechanisms and rejection proofs

Let u=C(N(x)), v=N(C(x)), with the same fixed absolute noise realization and insertion reference. Noncommutativity u≠v is a signal-chain fact; it does not establish a new learning principle.

M3a proposed allocating supervision by a deterministic signal discrepancy s(u,v): L=α(s,y)ℓ(y,p(u))+(1−α(s,y))ℓ(y,p(v)). For fixed α this is importance-weighted augmentation over the expanded source/view distribution. Naming α an order coefficient does not distinguish the optimizer from weighted ERM. If α is learned from losses, its comparator becomes adaptive weighting/DRO, not a claim of automatic novelty. REJECTED for the presently specified version.

M3b proposed prioritizing the harder order with max(ℓu,ℓv). The identity max(a,b)=(a+b+|a−b|)/2 shows that the apparent order disagreement term is exactly worst-of-two robust loss. Relabeling the two conditions inside a group-DRO/CVaR objective does not add a mechanism. REJECTED.

M3c proposed matching the signed prediction commutator p(u)−p(v) to an acoustic discrepancy. No verified source-specific relation maps waveform discrepancy to surviving FAKE evidence. It could encode loudness, clipping or codec delay and suppress difficult supervision. The Phase7 ImageNet-cosine proxy is NON_SPECIFIC and cannot supply this missing target. PROPOSED, NOT_TECHNICALLY_JUSTIFIED, NOT_IMPLEMENTED. No equality of embeddings/predictions across orders is imposed.

## Optimization, selection, falsification and costs

Adam uses a common LR per stage (no discriminative parameter LR claim), linear half-decay inside each stage, and a declared warmup/fine-tune transition. Warmup masks backbone gradients; the tail variables/slots are registered in advance, frozen BN stays unchanged, then real CNN gradients are applied during fine-tuning. This retains optimizer state across stages; it differs from resetting Adam after warmup. Head-only uses the sealed frozen-feature bank. Fine-tuning streams P0 tensors through the differentiable ImageNet CNN.

Selection CLEAN_BCE uses only primary A clean BCE. CLEAN_RISK_CONSTRAINED_ROBUSTNESS first measures both clean errors at .5, rejects checkpoints above absolute caps, then minimizes the mean balanced class error across noise5 and both orders. A rejected epoch is INFEASIBLE_UNDER_REGISTERED_CONSTRAINT. No feasible checkpoint means no selected export; no fallback to an infeasible model. Guardrails .10 REAL FA and .15 FAKE miss are exploratory absolute caps, distinct from Phase7 paired 2pp margins and from training constraints. Final selected-model evaluation includes all seven conditions, REAL-only B and per-attack FAKE groups.

Potential failures: overfit codec artifacts; increased REAL FA; missed hard attacks; q focusing on noisy small groups; underfit warmup; incompatible narrow-band inputs; unknown-source leakage. Falsify a claimed improvement with fresh matched seeds, both clean guardrails, both stressed class errors, matched group sampling, uniform q, equal-view supervision, consistency and finite/online RawBoost distinctions. M3-specific ablations are not fabricated because M3 is not selected. See METHOD_COMPARISON_PLAN and INTERNATIONAL_NOVELTY_MATRIX.

ERM has one supervised view/source/update; equal-view and M2 have four on the channel bank, consistency has two forwards per sampled view. Inference architecture remains the same student. Additional training overhead is measured in actual tiny runs and is not extrapolated as a GPU latency result. No full method convergence, independent confirmation or calibrated FAR guarantee is established.
