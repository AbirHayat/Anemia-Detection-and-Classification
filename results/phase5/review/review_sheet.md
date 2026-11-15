# Manual review of 40 explanations (Phase 5)

Research prototype, not a diagnostic tool. **To be filled in by the student only.** The AI prepared this sheet and must not rate any case.

Cases: 40 development rows drawn at random (seed 42) from the 963 clean development rows. The final model was trained on these rows, so the decisions are in-sample. Rate the explanation, not whether the diagnosis is right. The true label is shown last, for context only.

## Rubric (score each case 1, 2 or 3)

| Criterion | 3 | 2 | 1 |
|---|---|---|---|
| Correctness | every statement matches the input values and the stored rules, probability and features | one minor imprecision that does not mislead | at least one statement is wrong or misleading |
| Completeness | says the decision, the reason (rules and model), the confidence and the calibration caveat | one of these is missing or vague | a reader could not tell why the decision was made |
| Clarity | a non-specialist can follow it in one reading | needs a second reading or has unexplained jargon | confusing |

Write your scores in `review_ratings.csv` (one row per case, columns already there) and a short comment where a score is below 3. Take about 3 minutes per case.

## Case 1: `dev-689`

**Inputs (F-core):** WBC 7.9, RBC 5.1, HGB 13.3, MCV 90, MCH 29, MCHC 32, PLT 340

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.99, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 13.3, PLT 340, WBC 7.9. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Healthy:** HGB 13.3 (effect +0.283), PLT 340 (effect +0.203), WBC 7.9 (effect +0.200)

*True label (context only):* Healthy

## Case 2: `dev-450`

**Inputs (F-core):** WBC 7.2, RBC 3.56, HGB 11.3, MCV 96.6, MCH 31.7, MCHC 32.9, PLT 192

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.98, calibrated). Rule findings: anemia. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Normocytic normochromic anemia: MCV 96.6, HGB 11.3, MCHC 32.9. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.984, Other microcytic anemia 0.003, Macrocytic anemia 0.003

**SHAP top features for Normocytic normochromic anemia:** MCV 96.6 (effect +0.211), HGB 11.3 (effect +0.210), MCHC 32.9 (effect +0.173)

*True label (context only):* Normocytic normochromic anemia

## Case 3: `dev-462`

**Inputs (F-core):** WBC 7.2, RBC 3.34, HGB 9.2, MCV 85.3, MCH 27.5, MCHC 32.3, PLT 107

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.98, calibrated). Rule findings: anemia, low platelets. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Normocytic normochromic anemia: MCHC 32.3, MCV 85.3, HGB 9.2. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic, PLT < 150 x10^9/L -> low platelets, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X3, X5, X6, X9. **Labels allowed:** Normocytic normochromic anemia, Thrombocytopenia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.982, Normocytic hypochromic anemia 0.004, Other microcytic anemia 0.003

**SHAP top features for Normocytic normochromic anemia:** MCHC 32.3 (effect +0.202), MCV 85.3 (effect +0.196), HGB 9.2 (effect +0.175)

*True label (context only):* Normocytic normochromic anemia

## Case 4: `dev-1089`

**Inputs (F-core):** WBC 4.7, RBC 3.7, HGB 11.2, MCV 73.3, MCH 24.2, MCHC 33.1, PLT 89

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Other microcytic anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, low platelets. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Other microcytic anemia: MCV 73.3, MCHC 33.1, HGB 11.2. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, PLT < 150 x10^9/L -> low platelets, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X4, X5, X6, X9. **Labels allowed:** Iron deficiency anemia, Thrombocytopenia, Other microcytic anemia

**Top 3 calibrated probabilities:** Other microcytic anemia 0.986, Normocytic normochromic anemia 0.003, Healthy 0.002

**SHAP top features for Other microcytic anemia:** MCV 73.3 (effect +0.354), MCHC 33.1 (effect +0.198), HGB 11.2 (effect +0.124)

*True label (context only):* Other microcytic anemia

## Case 5: `dev-350`

**Inputs (F-core):** WBC 6.9, RBC 4.1, HGB 10.8, MCV 86.4, MCH 26.3, MCHC 30.5, PLT 153

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.98, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia. Labels the rules allow: Normocytic normochromic anemia. Second candidate: Other microcytic anemia (probability below 0.01). Values that pushed the model toward Normocytic hypochromic anemia: HGB 10.8, MCV 86.4, MCHC 30.5. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Normocytic hypochromic anemia:** HGB 10.8 (effect +0.254), MCV 86.4 (effect +0.236), MCHC 30.5 (effect +0.164)

*True label (context only):* Normocytic hypochromic anemia

## Case 6: `dev-875`

**Inputs (F-core):** WBC 7.3, RBC 4.3, HGB 11.8, MCV 85, MCH 28, MCHC 32, PLT 280

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.99, calibrated). Rule findings: anemia. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Normocytic normochromic anemia: HGB 11.8, MCV 85, MCHC 32. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.986, Other microcytic anemia 0.003, Healthy 0.002

**SHAP top features for Normocytic normochromic anemia:** HGB 11.8 (effect +0.254), MCV 85 (effect +0.167), MCHC 32 (effect +0.160)

*True label (context only):* Normocytic normochromic anemia

## Case 7: `dev-276`

**Inputs (F-core):** WBC 6.9, RBC 4.09, HGB 11.9, MCV 91, MCH 29, MCHC 31.9, PLT 131

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.98, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia, low platelets. Labels the rules allow: Normocytic normochromic anemia, Thrombocytopenia. Second candidate: Normocytic normochromic anemia (probability 0.01). Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Normocytic hypochromic anemia: HGB 11.9, MCHC 31.9, MCV 91. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic, PLT < 150 x10^9/L -> low platelets, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X3, X5, X6, X9. **Labels allowed:** Normocytic normochromic anemia, Thrombocytopenia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.981, Normocytic normochromic anemia 0.006, Other microcytic anemia 0.003

**SHAP top features for Normocytic hypochromic anemia:** HGB 11.9 (effect +0.258), MCHC 31.9 (effect +0.221), MCV 91 (effect +0.188)

*True label (context only):* Normocytic hypochromic anemia

## Case 8: `dev-647`

**Inputs (F-core):** WBC 8.2, RBC 5.2, HGB 14, MCV 91, MCH 30, MCHC 33, PLT 360

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.99, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 14, PLT 360, WBC 8.2. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Healthy:** HGB 14 (effect +0.269), PLT 360 (effect +0.181), WBC 8.2 (effect +0.181)

*True label (context only):* Healthy

## Case 9: `dev-107`

**Inputs (F-core):** WBC 3.2, RBC 4.33, HGB 11.8, MCV 82.9, MCH 27.2, MCHC 32.9, PLT 184

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.98, calibrated). Rule findings: anemia, low white count. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Normocytic normochromic anemia: HGB 11.8, MCHC 32.9, MCV 82.9. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic, WBC < 4 x10^9/L -> leukopenia, leukocytosis or leukopenia -> abnormal WBC, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X3, X5, X6, X8. **Labels allowed:** Normocytic normochromic anemia, Leukemia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.983, Normocytic hypochromic anemia 0.003, Other microcytic anemia 0.003

**SHAP top features for Normocytic normochromic anemia:** HGB 11.8 (effect +0.235), MCHC 32.9 (effect +0.188), MCV 82.9 (effect +0.164)

*True label (context only):* Normocytic normochromic anemia

## Case 10: `dev-363`

**Inputs (F-core):** WBC 6.5, RBC 4.79, HGB 12.7, MCV 83.3, MCH 26.5, MCHC 31.8, PLT 247

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.98, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia. Labels the rules allow: Normocytic normochromic anemia. Second candidate: Normocytic normochromic anemia (probability below 0.01). Values that pushed the model toward Normocytic hypochromic anemia: HGB 12.7, MCV 83.3, MCHC 31.8. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.984, Normocytic normochromic anemia 0.003, Healthy 0.003

**SHAP top features for Normocytic hypochromic anemia:** HGB 12.7 (effect +0.246), MCV 83.3 (effect +0.212), MCHC 31.8 (effect +0.164)

*True label (context only):* Normocytic hypochromic anemia

## Case 11: `dev-754`

**Inputs (F-core):** WBC 8.3, RBC 5.4, HGB 14.2, MCV 92, MCH 31, MCHC 33, PLT 360

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.99, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 14.2, WBC 8.3, PLT 360. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Healthy:** HGB 14.2 (effect +0.269), WBC 8.3 (effect +0.181), PLT 360 (effect +0.181)

*True label (context only):* Healthy

## Case 12: `dev-81`

**Inputs (F-core):** WBC 6.4, RBC 4.66, HGB 13.6, MCV 88.1, MCH 29.1, MCHC 33.1, PLT 140

**Decision:** REFER (REFER)

**Summary shown to the user:** Decision: refer to a specialist. The model's top class is Thrombocytopenia (probability 0.98, calibrated), a class that always goes to a specialist. Rule findings: low platelets. Values that pushed the model toward Thrombocytopenia: HGB 13.6, PLT 140, WBC 6.4. A value can push toward a label by counting against the other labels. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic, PLT < 150 x10^9/L -> low platelets, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X2, X3, X5, X6, X9. **Labels allowed:** Thrombocytopenia

**Top 3 calibrated probabilities:** Thrombocytopenia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Thrombocytopenia:** HGB 13.6 (effect +0.252), PLT 140 (effect +0.242), WBC 6.4 (effect +0.194)

*True label (context only):* Thrombocytopenia

## Case 13: `dev-279`

**Inputs (F-core):** WBC 10.9, RBC 4.58, HGB 7.7, MCV 62.6, MCH 16.7, MCHC 26.3, PLT 300

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, pale red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Iron deficiency anemia: MCV 62.6, MCH 16.7, HGB 7.7. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, MCHC < 30 g/dL -> hypochromic

**Exclusion rules fired:** X1, X4, X5, X7, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 62.6 (effect +0.291), MCH 16.7 (effect +0.210), HGB 7.7 (effect +0.191)

*True label (context only):* Iron deficiency anemia

## Case 14: `dev-1273`

**Inputs (F-core):** WBC 8.7, RBC 4.66, HGB 12.2, MCV 76.6, MCH 26.2, MCHC 34.1, PLT 266

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Other microcytic anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Other microcytic anemia: MCV 76.6, MCHC 34.1, MCH 26.2. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic

**Exclusion rules fired:** X1, X4, X5, X6, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Other microcytic anemia 0.986, Normocytic normochromic anemia 0.003, Healthy 0.002

**SHAP top features for Other microcytic anemia:** MCV 76.6 (effect +0.371), MCHC 34.1 (effect +0.173), MCH 26.2 (effect +0.156)

*True label (context only):* Other microcytic anemia

## Case 15: `dev-761`

**Inputs (F-core):** WBC 8.6, RBC 5.3, HGB 14.4, MCV 91, MCH 30, MCHC 33, PLT 370

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.99, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 14.4, WBC 8.6, PLT 370. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Healthy:** HGB 14.4 (effect +0.268), WBC 8.6 (effect +0.187), PLT 370 (effect +0.179)

*True label (context only):* Healthy

## Case 16: `dev-396`

**Inputs (F-core):** WBC 5.7, RBC 4.38, HGB 11.3, MCV 85, MCH 25.7, MCHC 30.3, PLT 161

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.98, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia. Labels the rules allow: Normocytic normochromic anemia. Second candidate: Other microcytic anemia (probability below 0.01). Values that pushed the model toward Normocytic hypochromic anemia: HGB 11.3, MCV 85, MCHC 30.3. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Normocytic hypochromic anemia:** HGB 11.3 (effect +0.259), MCV 85 (effect +0.231), MCHC 30.3 (effect +0.157)

*True label (context only):* Normocytic hypochromic anemia

## Case 17: `dev-36`

**Inputs (F-core):** WBC 4.9, RBC 3.84, HGB 9.5, MCV 81.4, MCH 24.7, MCHC 30.4, PLT 160

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.98, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia. Labels the rules allow: Normocytic normochromic anemia. Second candidate: Other microcytic anemia (probability below 0.01). Values that pushed the model toward Normocytic hypochromic anemia: MCV 81.4, HGB 9.5, MCHC 30.4. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Normocytic hypochromic anemia:** MCV 81.4 (effect +0.255), HGB 9.5 (effect +0.246), MCHC 30.4 (effect +0.149)

*True label (context only):* Normocytic hypochromic anemia

## Case 18: `dev-835`

**Inputs (F-core):** WBC 7.5, RBC 4.5, HGB 12.3, MCV 89, MCH 28, MCHC 32, PLT 290

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.99, calibrated). Rule findings: anemia. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Normocytic normochromic anemia: HGB 12.3, MCHC 32, MCV 89. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.986, Other microcytic anemia 0.003, Healthy 0.002

**SHAP top features for Normocytic normochromic anemia:** HGB 12.3 (effect +0.260), MCHC 32 (effect +0.162), MCV 89 (effect +0.155)

*True label (context only):* Normocytic normochromic anemia

## Case 19: `dev-1045`

**Inputs (F-core):** WBC 4.99, RBC 4.86, HGB 13, MCV 81.5, MCH 26.7, MCHC 32.8, PLT 249

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.98, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 13, PLT 249, WBC 4.99. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.983, Other microcytic anemia 0.004, Normocytic normochromic anemia 0.003

**SHAP top features for Healthy:** HGB 13 (effect +0.271), PLT 249 (effect +0.218), WBC 4.99 (effect +0.172)

*True label (context only):* Healthy

## Case 20: `dev-1110`

**Inputs (F-core):** WBC 6.49, RBC 4.11, HGB 9.5, MCV 77.1, MCH 23.1, MCHC 30, PLT 176

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Iron deficiency anemia: MCV 77.1, MCH 23.1, MCHC 30. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic

**Exclusion rules fired:** X1, X4, X5, X6, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 77.1 (effect +0.296), MCH 23.1 (effect +0.188), MCHC 30 (effect +0.178)

*True label (context only):* Iron deficiency anemia

## Case 21: `dev-911`

**Inputs (F-core):** WBC 7.7, RBC 4.7, HGB 12.5, MCV 88, MCH 29, MCHC 33, PLT 300

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.99, calibrated). Rule findings: anemia. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Normocytic normochromic anemia: HGB 12.5, MCHC 33, MCV 88. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.986, Other microcytic anemia 0.003, Healthy 0.002

**SHAP top features for Normocytic normochromic anemia:** HGB 12.5 (effect +0.263), MCHC 33 (effect +0.175), MCV 88 (effect +0.168)

*True label (context only):* Normocytic normochromic anemia

## Case 22: `dev-146`

**Inputs (F-core):** WBC 6.7, RBC 4.45, HGB 9.8, MCV 75.9, MCH 22, MCHC 29, PLT 244

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, pale red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Iron deficiency anemia: MCV 75.9, MCH 22, MCHC 29. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, MCHC < 30 g/dL -> hypochromic

**Exclusion rules fired:** X1, X4, X5, X7, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 75.9 (effect +0.297), MCH 22 (effect +0.197), MCHC 29 (effect +0.176)

*True label (context only):* Iron deficiency anemia

## Case 23: `dev-986`

**Inputs (F-core):** WBC 4.31, RBC 4.92, HGB 12.8, MCV 80.7, MCH 26, MCHC 32.2, PLT 150

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.97, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia. Labels the rules allow: Normocytic normochromic anemia. Second candidate: Normocytic normochromic anemia (probability 0.01). Values that pushed the model toward Normocytic hypochromic anemia: MCV 80.7, MCH 26, HGB 12.8. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.972, Normocytic normochromic anemia 0.012, Other microcytic anemia 0.005

**SHAP top features for Normocytic hypochromic anemia:** MCV 80.7 (effect +0.220), MCH 26 (effect +0.214), HGB 12.8 (effect +0.212)

*True label (context only):* Normocytic hypochromic anemia

## Case 24: `dev-452`

**Inputs (F-core):** WBC 3.1, RBC 5.36, HGB 12.2, MCV 75.5, MCH 22.7, MCHC 30.1, PLT 149

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, low platelets, low white count. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Iron deficiency anemia: MCV 75.5, MCH 22.7, MCHC 30.1. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, PLT < 150 x10^9/L -> low platelets, WBC < 4 x10^9/L -> leukopenia, leukocytosis or leukopenia -> abnormal WBC, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X4, X5, X6. **Labels allowed:** Iron deficiency anemia, Thrombocytopenia, Other microcytic anemia, Leukemia, Leukemia with thrombocytopenia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 75.5 (effect +0.294), MCH 22.7 (effect +0.198), MCHC 30.1 (effect +0.165)

*True label (context only):* Iron deficiency anemia

## Case 25: `dev-1201`

**Inputs (F-core):** WBC 5.7, RBC 4.14, HGB 13.6, MCV 88.5, MCH 33.4, MCHC 37.6, PLT 262

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.98, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 13.6, PLT 262, WBC 5.7. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.982, Normocytic normochromic anemia 0.005, Thrombocytopenia 0.003

**SHAP top features for Healthy:** HGB 13.6 (effect +0.274), PLT 262 (effect +0.207), WBC 5.7 (effect +0.134)

*True label (context only):* Healthy

## Case 26: `dev-28`

**Inputs (F-core):** WBC 5.7, RBC 5.23, HGB 14.4, MCV 85.2, MCH 27.5, MCHC 32.3, PLT 171

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.98, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 14.4, PLT 171, WBC 5.7. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.980, Thrombocytopenia 0.006, Other microcytic anemia 0.003

**SHAP top features for Healthy:** HGB 14.4 (effect +0.226), PLT 171 (effect +0.224), WBC 5.7 (effect +0.187)

*True label (context only):* Healthy

## Case 27: `dev-295`

**Inputs (F-core):** WBC 6.3, RBC 4.37, HGB 9.9, MCV 76.2, MCH 22.6, MCHC 29.8, PLT 205

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, pale red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Iron deficiency anemia: MCV 76.2, MCH 22.6, MCHC 29.8. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, MCHC < 30 g/dL -> hypochromic

**Exclusion rules fired:** X1, X4, X5, X7, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 76.2 (effect +0.293), MCH 22.6 (effect +0.192), MCHC 29.8 (effect +0.185)

*True label (context only):* Iron deficiency anemia

## Case 28: `dev-322`

**Inputs (F-core):** WBC 5.3, RBC 4.63, HGB 10, MCV 74.3, MCH 21.5, MCHC 29, PLT 172

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, pale red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Iron deficiency anemia: MCV 74.3, MCH 21.5, MCHC 29. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, MCHC < 30 g/dL -> hypochromic

**Exclusion rules fired:** X1, X4, X5, X7, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 74.3 (effect +0.296), MCH 21.5 (effect +0.198), MCHC 29 (effect +0.174)

*True label (context only):* Iron deficiency anemia

## Case 29: `dev-620`

**Inputs (F-core):** WBC 7.7, RBC 4.7, HGB 12.4, MCV 88, MCH 29, MCHC 33, PLT 300

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.99, calibrated). Rule findings: anemia. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Normocytic normochromic anemia: HGB 12.4, MCHC 33, MCV 88. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.986, Other microcytic anemia 0.003, Healthy 0.002

**SHAP top features for Normocytic normochromic anemia:** HGB 12.4 (effect +0.263), MCHC 33 (effect +0.175), MCV 88 (effect +0.168)

*True label (context only):* Normocytic normochromic anemia

## Case 30: `dev-85`

**Inputs (F-core):** WBC 5.9, RBC 3.15, HGB 10.8, MCV 106.2, MCH 34.2, MCHC 32.3, PLT 80

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Macrocytic anemia (probability 0.99, calibrated). Rule findings: anemia, large red cells, low platelets. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Macrocytic anemia: MCV 106.2, MCH 34.2, RBC 3.15. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV > 100 fL -> macrocytic, PLT < 150 x10^9/L -> low platelets, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X3, X4, X6, X9. **Labels allowed:** Thrombocytopenia, Macrocytic anemia

**Top 3 calibrated probabilities:** Macrocytic anemia 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Macrocytic anemia:** MCV 106.2 (effect +0.288), MCH 34.2 (effect +0.278), RBC 3.15 (effect +0.161)

*True label (context only):* Macrocytic anemia

## Case 31: `dev-713`

**Inputs (F-core):** WBC 5.9, RBC 3.9, HGB 10.8, MCV 79, MCH 26, MCHC 32, PLT 200

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Other microcytic anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Other microcytic anemia: MCV 79, HGB 10.8, MCH 26. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic

**Exclusion rules fired:** X1, X4, X5, X6, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Other microcytic anemia 0.987, Normocytic normochromic anemia 0.003, Healthy 0.002

**SHAP top features for Other microcytic anemia:** MCV 79 (effect +0.337), HGB 10.8 (effect +0.160), MCH 26 (effect +0.138)

*True label (context only):* Other microcytic anemia

## Case 32: `dev-131`

**Inputs (F-core):** WBC 14, RBC 2.56, HGB 6.2, MCV 76.5, MCH 24.2, MCHC 31.7, PLT 251

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.97, calibrated). Rule findings: anemia, small red cells, high white count. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Iron deficiency anemia: MCV 76.5, HGB 6.2, MCHC 31.7. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, WBC > 11 x10^9/L -> leukocytosis, leukocytosis or leukopenia -> abnormal WBC, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X4, X5, X6, X8. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia, Leukemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.973, Other microcytic anemia 0.014, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 76.5 (effect +0.258), HGB 6.2 (effect +0.168), MCHC 31.7 (effect +0.149)

*True label (context only):* Iron deficiency anemia

## Case 33: `dev-822`

**Inputs (F-core):** WBC 5.9, RBC 3.9, HGB 10.8, MCV 79, MCH 26, MCHC 32, PLT 200

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Other microcytic anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Other microcytic anemia: MCV 79, HGB 10.8, MCH 26. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic

**Exclusion rules fired:** X1, X4, X5, X6, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Other microcytic anemia 0.987, Normocytic normochromic anemia 0.003, Healthy 0.002

**SHAP top features for Other microcytic anemia:** MCV 79 (effect +0.337), HGB 10.8 (effect +0.160), MCH 26 (effect +0.138)

*True label (context only):* Other microcytic anemia

## Case 34: `dev-204`

**Inputs (F-core):** WBC 7.4, RBC 4.41, HGB 11.2, MCV 80.9, MCH 25.3, MCHC 31.4, PLT 110

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.98, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia, low platelets. Labels the rules allow: Normocytic normochromic anemia, Thrombocytopenia. Second candidate: Normocytic normochromic anemia (probability below 0.01). Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Normocytic hypochromic anemia: HGB 11.2, MCV 80.9, MCHC 31.4. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic, PLT < 150 x10^9/L -> low platelets, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X3, X5, X6, X9. **Labels allowed:** Normocytic normochromic anemia, Thrombocytopenia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.984, Normocytic normochromic anemia 0.003, Other microcytic anemia 0.003

**SHAP top features for Normocytic hypochromic anemia:** HGB 11.2 (effect +0.281), MCV 80.9 (effect +0.226), MCHC 31.4 (effect +0.150)

*True label (context only):* Normocytic hypochromic anemia

## Case 35: `dev-656`

**Inputs (F-core):** WBC 7, RBC 4.4, HGB 12, MCV 87, MCH 28, MCHC 32, PLT 260

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Normocytic normochromic anemia (probability 0.99, calibrated). Rule findings: anemia. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Normocytic normochromic anemia: HGB 12, MCV 87, MCHC 32. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X1, X3, X5, X6, X8, X9. **Labels allowed:** Normocytic normochromic anemia

**Top 3 calibrated probabilities:** Normocytic normochromic anemia 0.986, Other microcytic anemia 0.003, Healthy 0.002

**SHAP top features for Normocytic normochromic anemia:** HGB 12 (effect +0.256), MCV 87 (effect +0.165), MCHC 32 (effect +0.160)

*True label (context only):* Normocytic normochromic anemia

## Case 36: `dev-961`

**Inputs (F-core):** WBC 23.1, RBC 4.8, HGB 8.5, MCV 57.3, MCH 17.7, MCHC 30.8, PLT 182

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, high white count. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Iron deficiency anemia: MCV 57.3, MCH 17.7, HGB 8.5. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, WBC > 11 x10^9/L -> leukocytosis, leukocytosis or leukopenia -> abnormal WBC, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X4, X5, X6, X8. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia, Leukemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 57.3 (effect +0.291), MCH 17.7 (effect +0.203), HGB 8.5 (effect +0.187)

*True label (context only):* Iron deficiency anemia

## Case 37: `dev-1046`

**Inputs (F-core):** WBC 6.83, RBC 4.74, HGB 12.9, MCV 87.6, MCH 27.2, MCHC 31.1, PLT 135

**Decision:** ABSTAIN (DISAGREE)

**Summary shown to the user:** Decision: no answer, a person should review this case. The model's top class is Normocytic hypochromic anemia (probability 0.98, calibrated), but the rules exclude it (X6: MCHC is not below 30 g/dL, so the cells are not counted as pale). Rule findings: anemia, low platelets. Labels the rules allow: Normocytic normochromic anemia, Thrombocytopenia. Second candidate: Normocytic normochromic anemia (probability below 0.01). Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Normocytic hypochromic anemia: HGB 12.9, MCHC 31.1, MCV 87.6. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, neither microcytic nor macrocytic -> normocytic, PLT < 150 x10^9/L -> low platelets, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X3, X5, X6, X9. **Labels allowed:** Normocytic normochromic anemia, Thrombocytopenia

**Top 3 calibrated probabilities:** Normocytic hypochromic anemia 0.983, Normocytic normochromic anemia 0.004, Other microcytic anemia 0.003

**SHAP top features for Normocytic hypochromic anemia:** HGB 12.9 (effect +0.258), MCHC 31.1 (effect +0.238), MCV 87.6 (effect +0.185)

*True label (context only):* Normocytic hypochromic anemia

## Case 38: `dev-71`

**Inputs (F-core):** WBC 6, RBC 5.05, HGB 12.3, MCV 78.3, MCH 24.3, MCHC 31.1, PLT 152

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.98, calibrated). Rule findings: anemia, small red cells. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Iron deficiency anemia: MCV 78.3, MCHC 31.1, HGB 12.3. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic

**Exclusion rules fired:** X1, X4, X5, X6, X8, X9. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 78.3 (effect +0.307), MCHC 31.1 (effect +0.184), HGB 12.3 (effect +0.128)

*True label (context only):* Iron deficiency anemia

## Case 39: `dev-210`

**Inputs (F-core):** WBC 13, RBC 4.97, HGB 11.2, MCV 72.7, MCH 22.5, MCHC 31, PLT 153

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Iron deficiency anemia (probability 0.99, calibrated). Rule findings: anemia, small red cells, high white count. The rules allow this label and the probability reaches the threshold 0.92. Low platelets or an abnormal white count was found, but referral follows the model's top class, which is not a referral class, so the case was not referred. Values that pushed the model toward Iron deficiency anemia: MCV 72.7, MCH 22.5, HGB 11.2. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** HGB < 13 g/dL -> anemia, MCV < 80 fL -> microcytic, WBC > 11 x10^9/L -> leukocytosis, leukocytosis or leukopenia -> abnormal WBC, low platelets or abnormal WBC -> provisional rule-based REFER trigger

**Exclusion rules fired:** X1, X4, X5, X6, X8. **Labels allowed:** Iron deficiency anemia, Other microcytic anemia, Leukemia

**Top 3 calibrated probabilities:** Iron deficiency anemia 0.985, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Iron deficiency anemia:** MCV 72.7 (effect +0.283), MCH 22.5 (effect +0.211), HGB 11.2 (effect +0.164)

*True label (context only):* Iron deficiency anemia

## Case 40: `dev-532`

**Inputs (F-core):** WBC 8, RBC 5.1, HGB 13.5, MCV 90, MCH 29, MCHC 32, PLT 350

**Decision:** ACCEPT (ACCEPT)

**Summary shown to the user:** Decision: Healthy (probability 0.99, calibrated). Rule findings: no abnormal finding. The rules allow this label and the probability reaches the threshold 0.92. Values that pushed the model toward Healthy: HGB 13.5, PLT 350, WBC 8. The value effects are measured before the final probability adjustment (calibration), so they are approximate. Research prototype, not a diagnostic tool.

**Rule findings (true):** neither microcytic nor macrocytic -> normocytic

**Exclusion rules fired:** X2, X3, X5, X6, X8, X9. **Labels allowed:** Healthy

**Top 3 calibrated probabilities:** Healthy 0.986, Other microcytic anemia 0.003, Normocytic normochromic anemia 0.003

**SHAP top features for Healthy:** HGB 13.5 (effect +0.275), PLT 350 (effect +0.202), WBC 8 (effect +0.202)

*True label (context only):* Healthy

