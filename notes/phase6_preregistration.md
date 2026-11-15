# Phase 6 pre-registration: the single held-out test run (written 15 Nov 2026, before the freeze)

Research prototype, not a diagnostic tool. Nothing in this file has been computed on the test set. The run starts only after the student types "run the final test", happens once, and is never repeated to change a result.

## Guards (in `src/phase6_final_test.py`)

- Refuses to run unless HEAD carries the git tag `phase6-freeze` and no tracked file differs from the tagged commit.
- Refuses to run if `results/phase6/` already exists (one run only).
- Checks the SHA-256 of every frozen file against `results/phase5/freeze_manifest.json` before reading the test rows.
- Reads the test rows once with `load_test(i_understand_this_is_phase6=True)`.
- If anything crashes, nothing is rerun until the student has been told.

## What is frozen (nothing is fitted or tuned on test rows)

- Final calibrated random forest + SMOTE, F-core, fitted on all 963 development rows (`results/phase5/models/forest_smote_Fcore.joblib`).
- REFER trigger (ML), τ per threshold version and conformal q (`results/phase4/frozen_phase6.json`).
- Rule engine and thresholds (`src/rules.py`), validator (`src/validator.py`), cut-off cascade (`results/phase2/cascade_frozen.json`).
- Fill constants for the filled-row split: the development-set modes (Phase 1).

## What the run computes (all on the 247 test rows)

**T0. Validation.** Rows that fail the validator are REJECTED, counted and left out of every metric (proposal rule).

**T1. Systems compared** (valid test rows):
1. Expert system alone: rule-set coverage, mean set size, singletons, and the rule REFER trigger's routing (descriptive).
2. Best ML model alone (the random forest): macro-F1, balanced accuracy, MCC, per-class precision and recall, confusion matrix.
3. Hybrid without abstention: the same metrics.
4. The agent with abstention (frozen τ): coverage, selective macro-F1 (six non-referral classes), referral sensitivity and false-referral rate, routing per class, abstention reasons. Full JSON audit log.
5. Cut-off cascade baseline (probable labeling rule, frozen 14 Nov): the same metrics as 2.
6. Conformal abstention (frozen q, α = 0.05 primary, 0.10 secondary): coverage, selective macro-F1, set coverage per class. Descriptive.
7. Confidence-only abstention at the agent's coverage (same REFER step): selective macro-F1. Descriptive, no test (H2 is a development-CV hypothesis).

Systems 1, 3, 4 and 6 are reported at all four threshold versions (main, MCHC 32, HGB 12, both) with equal prominence. The main thresholds stay the formal ones.

**T2. McNemar tests** (Dietterich 1998, chi-square with continuity correction, two-sided, on correct versus wrong nine-class predictions). Main family, Holm over three pairs at the main thresholds: (a) forest against hybrid without abstention, (b) forest against cascade, (c) hybrid without abstention against cascade. Pairs (a) and (c) are repeated at MCHC 32, HGB 12 and both, each version with its own Holm over its pairs. The exact binomial p is reported next to each test. If both systems disagree on no row, p = 1.

**T3. Acceptance targets** (proposal Section IV-D):
1. Referral: every valid stress case built from the test rows that meets the ML trigger returns REFER (100%). Stress cases as in `src/stress.py`.
2. Validation: every injected bad value (each test row × 14 columns × 8 corruptions) returns REJECT naming that column (100%).
3. Explanations: the automatic consistency check passes on 100% of the test cases (agent at the main thresholds).
4. H1 and 5. H2: development CV, copied from Phase 4 (both failed). Not recomputed.

**T4. Descriptive extras.** Per-class results with the test support, and an explicit caveat for any class with 5 or fewer test rows. Macro-F1 on mean-filled and other rows (classes present in each group). Errors near the label cut-offs (Phase 3 P7 bands). Per-stage latency. Noise stability on the test rows.

## What the run does not do

It does not train, tune, recalibrate or choose anything. It does not touch the development rows except to read the frozen fill constants. It does not compute H1 or H2 on the test set. The manual review of 40 explanations stays on development rows.
