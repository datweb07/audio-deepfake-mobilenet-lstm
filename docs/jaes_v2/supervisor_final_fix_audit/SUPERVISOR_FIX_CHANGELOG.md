# FIX02 changelog

Each entry quotes the affected FIX01 text and its replacement. PDF line wraps are normalized. Original pagination is preserved.

## 1. Page 1 · Abstract

Supervisor point 4. ACADEMIC_LANGUAGE_CORRECTION. The detector is trained on LA data. ImageNet initializes the backbone. See Section 4.

**FIX01**

> pre-trained on 640 LA samples of ASVspoof 2019 reached an advantage of macro stressed

**FIX02**

> trained on 640 LA samples of ASVspoof 2019 reached an advantage of macro stressed

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 2. Page 1 · Abstract

Supervisor point 1, 2, 3. RISK_GUARD_CORRECTION. Clarifies guard failure, detector response and held-out class-specific results. See Table 7 and the sealed held-out metrics.

**FIX01**

> of stressed REAL false alarms increased from 5 to 161 of 336. Selected channel model was greater than two risk limits for clean audio. Noise and Opus transformations in reverse order caused changes of class-wise decision making, order dependent REAL class mistake variance depended on the training procedure. Within-release locked evaluation of original four models for 1,088 recordings showed better stressed classification and fewer pooled FAKE mistakes in combination with increases of clean FAKE and stressed mistakes on A17-A19.

**FIX02**

> of stressed REAL false alarms increased from 5 to 161 of 336. The selected Channel checkpoint violated both clean-risk guards. Reversing noise and Opus changed classwise decisions on the same recordings, and the REAL error contrast varied by training protocol. The locked within-release evaluation of the original four models on 1,088 recordings showed higher stressed ROC-AUC and fewer pooled stressed FAKE misses, alongside more clean FAKE misses and more stressed misses on A17–A19.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 3. Page 9 · Section 4.2

Supervisor point 4. ACADEMIC_LANGUAGE_CORRECTION. Removes the head-only implication for partial fine-tuning. See Section 4.1 and the training contract.

**FIX01**

> first epoch pre-trains the head, while the remaining 20 epochs perform partial fine-tuning on the head with batch-normalization parameters frozen. The training phases begin from learning

**FIX02**

> first epoch trains the head, while the remaining 20 epochs perform partial fine-tuning with batch-normalization parameters frozen. The training phases begin from learning

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 4. Page 16 · Section 6.1

Supervisor point 2. NUMERICAL_DIRECTION_CORRECTION. Corrects the direction for 9107 and separates REAL sources from repeated stress views. See Tables 8–9 and archived scores.

**FIX01**

> FAKE misses under stress reduced from 1,370 to 587 in 9107, and from 815 to 453 in 9207, in both cases of 2,496 correlated views, of 832 FAKE sources. The rate differences were −31.37 percentage points [−33.41, −29.36] and −14.50 points [−17.38, −11.67]. False alarms due to stress REAL alerts decreased from 80 to 81 and 322 to 136 in a total of 768 instances from 256 data sources. The corresponding confidence intervals were [−3.23, +3.76] and [−29.07, −19.03] percentage points. All endpoint and attack confidence intervals were based on 10,000 valid draws with no invalid draws.

**FIX02**

> Stressed FAKE misses fell from 1,370 to 587 in 9107 and from 815 to 453 in 9207, each out of 2,496 views from 832 FAKE sources. The rate differences were −31.37 points [−33.41, −29.36] and −14.50 points [−17.38, −11.67]. Stressed REAL false alarms increased by one in 9107, from 80/768 to 81/768 (+0.13 points), and fell from 322/768 to 136/768 in 9207. The 768 views in each comparison came from 256 REAL sources. The paired intervals were [−3.23, +3.76] and [−29.07, −19.03] percentage points. All endpoint and attack intervals used 10,000 valid draws.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 5. Page 17 · Section 6.2

Supervisor point 1. RISK_GUARD_CORRECTION. Both observed increases exceed +2 points. Clean BCE selects the evaluated checkpoints. See size_sensitivity.json.

**FIX01**

> The 1,280 Channel checkpoint passes the registered +2-point tolerance test for A clean FAKE misses (3/112 to 8/112, +4.46 points) and B clean REAL false alarms (18/144 to 22/144, +2.78 points). The pointwise speaker intervals are [1.79, 6.25] and [−4.86, 11.81] points respectively. The guard condition holds with respect to the registered criterion. In particular, the B interval contains zero and B contrast still exceeds the registered tolerance. The checkpoints pass the clean BCE selection and the guard test applies to the selected pair.

**FIX02**

> The selected 1,280 Channel checkpoint violates both registered +2-point clean-risk guards. Development A clean FAKE misses rose from 3/112 to 8/112 (+4.46 points) and B clean REAL false alarms rose from 18/144 to 22/144 (+2.78 points). Pointwise 95% speaker intervals were [1.79, 6.25] and [−4.86, 11.81] percentage points. The B interval includes zero. The guards use the observed differences for the pair selected by clean BCE, so both exceed the tolerance.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 6. Page 19 · Section 6.2

Supervisor point 2. NUMERICAL_DIRECTION_CORRECTION. Completes clean REAL comparisons by seed and clarifies Development B directions. See Tables 4 and 9.

**FIX01**

> Real false alarms reduced to 19 out of 256 and also to 16 out of 256. Clean FAKE misses increased from 125 to 129 and from 127 to 147 out of 832, while clean ROC-AUC declined from 0.9534 to 0.9436 and from 0.9507 to 0.9410. The increase in 9207 for clean FAKE misses was 2.40 percentage points [1.15, 3.86]. Both held-out pairings had less clean REAL false alarms, but the B differences for development were opposite in sign. The two cohorts differ in speaker and source support.

**FIX02**

> Clean REAL false alarms fell from 25 to 19 in 9107 and from 22 to 16 in 9207, out of 256 sources per arm. Clean FAKE misses increased from 125 to 129 and from 127 to 147 out of 832, while clean ROC-AUC declined from 0.9534 to 0.9436 and from 0.9507 to 0.9410. The 9207 clean FAKE miss increase was 2.40 percentage points [1.15, 3.86]. Development B showed opposite false-alarm directions between the two seeds. The cohorts differ in speaker and source support.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 7. Page 20 · Section 6.3

Supervisor point 3. ATTACK_INTERPRETATION_CORRECTION. Separates initialization variation and fixed-epoch training size sensitivity. See Table 10 and the 840/1,680 update budgets.

**FIX01**

> denominator represents three correlated stress views from 18 sources. The reversals compare two training protocols at one seed. The 9307 reversal below compares different initializations at 640. Table 10 and Fig. 10 show a sign reversal for both development attacks. Channel increases

**FIX02**

> denominator represents three correlated stress views from 18 sources. Both training size and update budget changed. Seed 9307 compares initializations at 640. Table 10 and Fig. 10 show both sign reversals. Channel increases

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 8. Page 21 · Section 6.3

Supervisor point 3. ATTACK_INTERPRETATION_CORRECTION. Distinguishes descriptive intervals from post-hoc grouping. Clarifies the +39 Noise-to-Opus count on A17–A19. See Figure 12, locked metrics and attack_metrics.csv.

**FIX01**

> Figure 12 shows the condition-specific changes for A07-A19. Stress misses declined on A07- A16 in both blocks. They increased on A17, A18 and A19 in both blocks, by 10, 11 and four views in 9107 and by 43, 43 and 56 views in 9207. Each attack contributes 64 sources and 192 stressed views. The 9107 intervals for these three increases include zero, while the 9207 intervals are [15.90, 29.03], [14.12, 30.81] and [20.63, 37.70] percentage points. Such attacks highlight negative subgroups against the positive overall group. The development A05/A06 observations and held-out A17-A19 observations refer to different attack sets. In the post-hoc breakdown, A07-A16 contribute net reductions of 808 and 504 stressed misses, partially offset by increases of 25 and 142 on A17-A19. In 9107 the adverse net increases are driven by Noise10 followed by Opus, which adds 13 misses to each group. In 9207 all three stress conditions add misses on these attacks. For A19, 43 of 64 distinct sources acquire at least one new stressed miss and 18 acquire new misses under multiple conditions in 9207. These groupings were identified after examining the held-out results. The intervals are pointwise and are interpreted descriptively.

**FIX02**

> Figure 12 shows the condition-specific FAKE miss contrasts. Pooled stressed misses fell on A07–A16 and rose on A17–A19 in both blocks. A17, A18 and A19 added 10, 11 and 4 misses in 9107 and 43, 43 and 56 in 9207, each out of 192 views from 64 sources per attack. The pointwise 9107 intervals include zero. The 9207 intervals exclude zero: [15.90, 29.03], [14.12, 30.81] and [20.63, 37.70] percentage points. The development A05/A06 results concern a separate attack set. In the post-hoc grouping, A07–A16 account for 808 and 504 fewer stressed misses in 9107 and 9207, partly offset by 25 and 142 additional misses on A17–A19. For A17–A19, Noise-to-Opus added 39 misses in 9107. All three stress conditions added misses in 9207. On A19, 43/64 sources acquired a miss in at least one stress condition and 18/64 in multiple conditions. These exploratory groupings were identified after reviewing held-out results. The pointwise intervals have no multiplicity adjustment.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 9. Page 22 · Section 6.4

Supervisor point 3. ORDER_EFFECT_INTERPRETATION_CORRECTION. Clarifies detector response on two views of the same source. The recorded score contrast is preserved.

**FIX01**

> +0.0764. These are pairs of scores and decisions responses under the registered transformations.

**FIX02**

> +0.0764. These paired contrasts measure detector response to the order of processing.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 10. Page 26 · Discussion

Supervisor point 1, 4. ACADEMIC_LANGUAGE_CORRECTION. Refers to Table 7 for counts and retains the risk trade-off, clean-BCE selection and budget caveat.

**FIX01**

> The nested comparison illustrates the same contrast but within one and the same initialization. In 1,280, better ranking and lower stressed misses work together with the big growth in stressed REAL false alarms and two clean guard failures. An analysis under the only AUC metric would ignore those operational risks. The smaller AUC gain in 1,280 makes it possible to limit the claim about the augmentation effect. The contrast involves the extended source pool and additional updates under the fixed-epoch protocol.

**FIX02**

> The nested comparison shows why ranking and fixed-threshold errors must be read together. At 1,280, fewer stressed FAKE misses accompany more stressed REAL false alarms and two clean-risk guard failures on the clean-BCE-selected pair. Table 7 gives the counts. The smaller ranking gain accompanies the larger cohort and additional optimizer updates under the fixed-epoch protocol.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 11. Page 30 · Limitations

Supervisor point 2, 4. NUMERICAL_DIRECTION_CORRECTION. The held-out lock identifies four fits of one architecture. ASVspoof attack families may overlap. The external-validity caveat remains.

**FIX01**

> in our experiments. In the completed official evaluation, there are four architectures with 640 recordings in the same corpus release. Certain evaluations employ attack algorithms or systems that have already been used. An external corpus with known audio labels, source lineage, and

**FIX02**

> in our experiments. The locked within-release evaluation used four fits of one architecture trained on 640 recordings each. Some attacks share algorithms with earlier partitions. An external corpus with known audio labels, source lineage, and

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 12. Page 29 · Limitations

Supervisor point 2, 4. NUMERICAL_DIRECTION_CORRECTION. The registered budgets are 840 and 1,680 updates. 1,280 is the larger recording count.

**FIX01**

> one nested expansion, which makes it twice as large with 1,280 updates. Questions about size

**FIX02**

> one nested expansion that doubles updates from 840 to 1,680. Questions about size

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## 13. Page 31 · Conclusion

Supervisor point 1, 4. RISK_GUARD_CORRECTION. Makes the clean-BCE comparison explicit and reports the increase in stressed REAL errors directly.

**FIX01**

> increment for 1,280 in seed 9207 occurred with lower number of stressed FAKE misses and significantly higher stressed REAL false alarms at 0.5 threshold. Two clean risk guards were violated in that selected case.

**FIX02**

> gain at 1,280 in seed 9207 accompanied fewer stressed FAKE misses and more stressed REAL false alarms at threshold 0.5. The clean-BCE-selected Channel checkpoint violated both clean-risk guards.

**QA** — Extracted text matches the replacement. Surrounding content and graphics pass comparison.

## Submission metadata

SUBMISSION_METADATA_CORRECTION. See the checklist for availability, authorship, UEH support, COI, conference rights and disclosure. Acknowledgement, bibliography and Table 1 remain intact.
