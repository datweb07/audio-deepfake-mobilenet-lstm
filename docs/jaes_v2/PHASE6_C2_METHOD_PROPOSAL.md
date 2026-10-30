# Phase6 C2 method design review

Status: **PRIOR_ART_OVERLAP; new objective NOT_IMPLEMENTED**. This is a written rejection/deferral gate, not a claim of a technically distinct candidate.

Problem: a distorted view may lose class evidence, while unconditional consistency can penalize justified changes. Candidate input would be a training-only paired clean/distorted waveform or representation and an observable, label-free retention proxy. Required trainable modules would include a genuine trainable representation path, not only detached ImageNet embeddings.

Research-only formulation: L=.5[BCE(y,p_clean)+BCE(y,p_view)]+lambda*w(r)*MSE(p_clean,p_view), with r in[0,1], w(r)=w_min+(1-w_min)r and w_min>0. No such new weighted loss is run. All views would keep full supervised exposure so a model cannot ignore difficult examples. Lambda, w_min and estimator choice are UNLOCKED; no hyperparameter search is performed.

The available diagnostic proxy is mapped cosine similarity of frozen clean/view embeddings. It is measurable without class labels, but similarity can reflect content/scale or ImageNet texture rather than preserved spoof evidence. Development labels are used only afterward to inspect association with errors; no confirmation outcome, generator identity or test label enters r. Repeated views and speakers remain clustered. No predictive validity is inferred merely from correlation.

Nearest overlap: confidence/codec weighting in DK-CAST and ordinary paired consistency; F-SAT already selectively protects a frequency region, while PCL-NET already aligns paired representations. Changing loss location/backbone or renaming similarity as retention is insufficient technical difference. There is no defensible novelty plausibility pass yet.

Falsification: compare ERM, amplitude, channel, faithful finite/online RawBoost, equal-two-view BCE and ordinary consistency with identical pairs/updates. Reject the mechanism if an equal-view control explains improvements, if the proxy fails on cue-destroying cases, if minimum supervised exposure is violated, or if clean classwise risk exceeds the locked tolerance. Candidate-specific ablations would remove weighting, substitute a random/shuffled weight, test severity/band controls and train the same representation. Costs require extra paired passes and estimator computation; measured values for the unimplemented candidate are NOT_MEASURED.

Phase6 therefore advances standard baselines and real gradient capability, and preserves the failed novelty gate. Independent training seeds, larger all-attack coverage and untouched external evaluation are prerequisites for promotion.
