> Copied unaltered from the private project notes (notes/phase3_preregistration.md) on export.
> File paths it mentions under notes/ and the decision log are not part of this repository.

# Phase 3 pre-registration (written 14 Nov 2026, BEFORE any Phase 3 result was opened)

Research prototype, not a diagnostic tool. Development rows only (963). The test set stays locked.

**Status of the outputs when this was written:** `results/phase3/ml_only/` (finished 14 Nov, 18:54) and `results/phase3/h3_indices/` and `h4_probes/` (running) exist, but no metric has been computed or viewed. **All CONFIRM items were confirmed by the student on 14 Nov 2026 (19:48) as written, and logged in `notes/decision_log.md` before any Phase 3 result was opened.**

## Fixed already (proposal and earlier decisions)

- Outer CV: 5 folds × 5 repeats, stratified, seed 42 (25 outer folds). Inner 3-fold CV picks hyperparameters on macro-F1.
- Models: majority-class dummy, logistic regression, depth-limited tree, random forest and histogram gradient boosting. Feature sets: F-core (7 columns) and F-all (14). Imbalance: class weights against SMOTE.
- Metrics: macro-F1, balanced accuracy, MCC, per-class precision and recall, confusion matrices. Mean ± std over the 25 outer folds (AD11).
- Paired CV comparisons: corrected repeated k-fold cv test (Bouckaert and Frank 2004, Section 3.3, formula verified in the open full text on 14 Nov 2026), n₂/n₁ = 1/4, df = 24, Holm adjustment.
- Every main metric is also reported separately for mean-filled and other rows.
- The H2 model is the model type with the best development-CV macro-F1 here. It is logged before Phase 4 opens any abstention result.

## Proposed rules (CONFIRM)

**P1. Which predictions count (CONFIRM).** Use the **calibrated** probabilities (sigmoid, fitted on training rows only) and predict the most probable class. The agent uses calibrated probabilities, so its parts are judged the same way. Uncalibrated scores are reported as a secondary table.

**P2. Imbalance arm per model type (CONFIRM).** For each model type, keep the arm (class weights or SMOTE) with the higher mean macro-F1 on **F-core**. A tie within 0.001 goes to class weights (simpler). That arm is then used everywhere for that type (F-all, H3, H1, H2). The class weights against SMOTE comparison is reported in full (two-sided paired test per type, Holm over 4).

**P3. H2 model choice (CONFIRM).** The model type with the best mean macro-F1 on F-core with its chosen arm (calibrated). F-core because H1 is defined on F-core and it avoids the filled columns.

**P4. H3 (indices) (CONFIRM).** Primary: per model type, F-core+idx against F-core, one-sided corrected test on the per-fold mean of iron deficiency recall and other microcytic recall, Holm over the 4 model types (AD5). Secondary: the same on F-all, with its own Holm, reported as descriptive.

**P5. H4 probes.** Probe 1: fill indicator (count 0–6 + PDW flag, learned per training fold) → depth-3 tree with balanced class weights, target = diagnosis. Probe 2: F-core → depth-3 tree, target = mean-filled or not. Each against the majority-class dummy on balanced accuracy, one-sided corrected test, Holm over the 2 probes (AD6: expected to pass, the test documents the shortcut). Depth 3 is Claude's choice.

**P6. F-core against F-all.** Per model type (chosen arm), two-sided paired corrected test on macro-F1, Holm over 4. Descriptive (no hypothesis).

**P7. Errors near the cut-offs (student's request, CONFIRM the bands).** A row is "near a cut-off" if any value falls in these bands around the label-cascade cut-offs (development-data values, `results/phase2/cascade_frozen.json`): HGB 12.5–13.5, PLT 135–165, WBC 9.5–11.5 (covers both 10 and 11), MCV 78–82 and 98–102, MCHC 31.4–32.5 (covers both 31.9 and 32). Report, per model type: error rate near the cut-offs against error rate elsewhere, and the share of all errors that are near a cut-off, with the share of rows that are near. Descriptive.

**P8. Settings already used by the run (CONFIRM).** Sigmoid calibration. Grids in `src/ml_pipelines.py` (logistic C 5 values, tree depth 5 × min leaf 2, forest depth 2 × min leaf 2 with 300 trees, gradient boosting learning rate 2 × depth 2 with 200 iterations).

If any CONFIRM item changes, the affected run is redone before anything is opened.

## Post-results amendment (14 Nov 2026, AFTER the results above were seen)

Calibration is now done on predicted probabilities for every model (sigmoid, same folds, seeds and hyperparameters). Reason: scikit-learn had calibrated logistic regression and gradient boosting on their raw decision scores, which collapsed logistic regression (`results/phase3/calibration_diagnostic.txt`). Rules P1–P7 are unchanged and were applied to both versions. The original pre-registered results stay in `results/phase3/`, the amended ones are in `results/phase3/amended/`, and `results/phase3/calibration_amendment_comparison.md` puts them side by side. Phase 4 uses the amended version. This amendment is not part of the pre-registration and is reported as such.
