# Phase 4 pre-registration (written 15 Nov 2026, BEFORE any Phase 4 result was computed)

Research prototype, not a diagnostic tool. Development rows only (963). The test set stays locked.

**Status when this was written:** no Phase 4 metric exists. **The whole file was confirmed by the student on 15 Nov 2026 (16:44) as written and logged in `notes/decision_log.md` before any Phase 4 metric was computed.** `src/phase4_inner_cv.py` (inner predictions for choosing τ, see R4) was started on 15 Nov and prints timing only. Phase 3 per-class recall of the H2 model has been seen (`results/phase3/amended/per_class_chosen.csv`), and the rule-based REFER trigger figure on cleaned rows (88/98, 89.8%) is already in `results/phase2/rules_baseline.txt`. Items marked **STUDENT** were chosen by the student on 15 Nov in answer to four questions. Items marked **CLAUDE** are Claude's proposals and need the student's confirmation.

## Fixed already (proposal v5d and the decision log)

- Inputs: the amended Phase 3 out-of-fold calibrated probabilities (`results/phase3/amended/oof_ml_only.csv.gz`), F-core, chosen arms (logreg SMOTE, tree class weights, forest SMOTE, gradient boosting SMOTE). The same 25 outer folds (5 × 5, seed 42).
- H2 model: random forest + SMOTE (logged 14 Nov, before any abstention result).
- Arbitration: REJECT (validator) → REFER (trigger) → ACCEPT if the top ML class is in the rule set and its calibrated probability is at least τ → ABSTAIN otherwise. All 963 cleaned development rows passed the validator, so REJECT counts zero here.
- Hybrid without abstention (H1): the most probable class inside the rule set (top ML class if the set is empty).
- REFER choice rule: keep triggers with referral sensitivity ≥ 90%, then pick the lowest false-referral rate. If none reaches 90%, pick the most sensitive. Test set never used.
- τ: chosen inside each outer training fold by inner CV, best inner selective macro-F1 among τ values with inner coverage ≥ 70%. Coverage floors of 60% and 80% are sensitivity runs (timetable v2).
- Scoring: coverage = ACCEPT / all valid cases. Selective macro-F1 = macro-F1 over the six non-referral classes on accepted cases (a class with no accepted case scores 0, and a referral-class case that slips into ACCEPT is an error).
- Every rule and hybrid result is reported at the primary thresholds (HGB 13, MCHC 30) and again at HGB 12 and at MCHC 32 with equal prominence. The combined HGB 12 + MCHC 32 version is supplementary.

## Rules

**R1. H1, non-inferiority of the hybrid (STUDENT for the verdict).** For each of the four model types: per outer fold, macro-F1 (nine classes) of the hybrid without abstention minus macro-F1 of the same model alone. 90% CI with the corrected repeated k-fold CV test (n₂/n₁ = 1/4, df = 24). Supported for a type if the lower bound is above −0.03. The formal verdict uses the primary thresholds (logged before any ML result). The same test runs at HGB 12, MCHC 32 and both, shown in the same table. The dummy is not part of H1. F-all is descriptive. Balanced accuracy, MCC, per-class recall and filled versus other rows are reported too.

**R2. REFER trigger (CLAUDE for the measurement).** Candidates on the H2 model's outer predictions: rule (F10 fires: PLT < 150 or WBC outside 4.0 to 11.0), ML (top calibrated class is a referral class) and either. Sensitivity and false-referral rate are pooled over all 5 × 963 outer predictions (a small numerator is less noisy pooled than averaged over 25 folds). Mean ± std over folds is also reported. Tie on the false-referral rate goes to the rule trigger (fully traceable), then to ML. The trigger does not depend on HGB or MCHC, so one choice serves all threshold versions.

**R3. Arbitration details (CLAUDE).** "Top ML class" is the argmax of all nine calibrated probabilities. τ grid: 0.00, 0.01, ..., 0.99, with ACCEPT when the probability ≥ τ. Ties in inner selective macro-F1 go to the smaller τ (more coverage). If no τ reaches the coverage floor, τ = 0 is used and the fold is flagged. Abstentions are split by reason (rule-model disagreement or low confidence).

**R4. Inner CV for τ and the conformal threshold (CLAUDE).** Inside each outer training fold: StratifiedKFold(5), and in each inner fold the full Phase 3 procedure for the H2 model (grid search 3-fold on macro-F1, sigmoid calibration on probabilities, 3-fold). The inner predictions cover every outer training row once. τ is chosen on these pooled inner predictions with the same REFER trigger and rule sets, then applied once to the outer test fold. Same procedure per threshold version and per coverage floor.

**R5. H2 matched-coverage test (STUDENT for R5a and R5b).**
- R5a. All three systems share the chosen REFER step. In each outer fold the hybrid accepts n cases. Confidence-only accepts the n non-referred cases with the highest calibrated top probability (no rules) and predicts the top ML class. Random abstention accepts n non-referred cases drawn at random (1,000 draws, seed 42) and predicts the hybrid-without-abstention label.
- R5b. Selective macro-F1 per repeat (each case appears once per repeat), averaged over the 5 repeats. Paired bootstrap: 2,000 resamples of the 963 cases (seed 42). In each resample every repeat is scored on the resampled cases and the 5 scores are averaged. The random baseline uses the same 1,000 draws in every resample. 95% percentile CIs. One-sided p = (1 + number of resamples with difference ≤ 0) / 2,001. Holm over the two baselines. H2 passes if both adjusted p < 0.05.
- R5c (CLAUDE). The formal verdict uses the primary thresholds and the 70% floor, following the same principle as R1. Other versions and floors are shown in the same table.

**R6. Conformal abstention (STUDENT for α).** Score = 1 − calibrated probability of the true class, taken from the inner predictions (R4) of the outer training rows. q = the ⌈(n+1)(1−α)⌉-th smallest score. On the outer test fold a label is in the set when 1 − p ≤ q. After the shared REFER step, ACCEPT only a single-label set whose label is in the rule set. α = 0.05 primary, 0.10 secondary. Descriptive (not part of H2). Reported: coverage, selective macro-F1, empirical set coverage overall and per class, mean set size per class and empty sets. The calibration scores come from inner models, not from the outer model itself, so the coverage guarantee is approximate. The paper says so.

**R7. Risk-coverage curves (CLAUDE).** Descriptive. Sweep τ over the grid on the outer predictions (after REFER). Risk = error rate among accepted cases (main curve) and 1 − selective macro-F1 (second panel). Curves for the hybrid, confidence-only (threshold on the top probability) and random (expected value). The area under the error-rate curve (AURC, trapezoid) is reported. The curves never choose τ.

**R8. Descriptive extras (CLAUDE).** Referral and abstention rate per class. Filled versus other rows for every Phase 4 metric. Errors near the cut-offs with the Phase 3 P7 bands (hybrid without abstention against the model alone, and the abstention rate near versus away from a cut-off).

**R9. Settings frozen for Phase 6 (CLAUDE).** The final τ (each threshold version, 70% floor) and the final conformal q (α = 0.05 and 0.10) come from the same rules applied to the outer predictions of the whole development set, pooled over the 5 repeats. They are logged and scored once on the test set in Phase 6.

If any CLAUDE item changes, the affected step is redone before anything is opened.
