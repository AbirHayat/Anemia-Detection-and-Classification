# Abstaining hybrid agent for anemia types from CBC data

**Research prototype, not a diagnostic tool.** Nothing here should be used to diagnose or treat anyone.

This repository is shared privately with my course instructor for progress review (Artificial Intelligence and Expert Systems).

The project classifies a complete blood count (CBC) record into one of nine labels (five anemia types, healthy and three white-cell or platelet disorders). It combines a cited rule engine (an expert system) with machine-learning models. In the finished agent, the case goes to a person when the two disagree or confidence is low. Every decision comes with a rule trace and an explanation.

## Status

- Phase 1: data audit and cleaning (done)
- Phase 2: cited rule engine (done)
- Phase 3: ML pipeline and nested CV (done)
- Phase 4: hybrid agent, abstention and referral (done)
- Phase 5: explanations, audit log and stress tests (done)
- Phase 6: final evaluation on the held-out test set (done)

## Quick start: one notebook runs the whole project

Everything needed is in the repository, including the dataset. Python 3.12 is recommended (tested with 3.12.6).

**1. Get the code and install** (about 3 minutes, needs internet once)

Windows (PowerShell or Command Prompt):
```
git clone <this repository URL>
cd anemia-agent-public
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS or Linux:
```
git clone <this repository URL>
cd anemia-agent-public
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Run the notebook**
```
jupyter lab
```
Open `notebooks/anemia_project.ipynb` and run all cells. The setup cell checks Python, the package versions and the dataset fingerprint. Then each phase (1 to 6) shows its key tables and figures and asserts that they match the committed files in `results/`.

- `RUN_FULL = True` recomputes Phases 1-5 from the data in a temporary copy of the repository and compares every regenerated file with `results/` (about 15 minutes on a 16-thread desktop PC (the recompute itself took 864 s), longer on a laptop).
- `RUN_FULL = False` loads and checks the committed results only (a few seconds (2 s on the same PC)).
- The default is `RUN_FULL = True`.

The held-out test rows are never read by the notebook. Phase 6 only displays the committed final-test files after checking every frozen input against `results/phase5/freeze_manifest.json`. `notebooks/anemia_project.html` is a read-only copy with all outputs.

`requirements.txt` pins the direct dependencies and works on Windows, macOS and Linux. `requirements-lock.txt` pins every package exactly as tested, but it was tested on Windows only and contains Windows-only packages, so use it only on Windows.

## What the phases found

- **Phase 1, data audit.** The input validator rejected 22 development rows with impossible values (for example a negative hemoglobin), leaving 963. Many rows were mean-filled before upload, unevenly by class. A cascade of six round cut-offs on five CBC columns reproduces 98.3% of the development labels, so the labels appear to follow simple rules on the same values the models see (`results/phase1/`).
- **Phase 2, rule engine.** A forward-chaining expert system with cut-offs cited from WHO 2024, the Merck Manual, American Family Physician and MeSH. It only excludes labels that a cited finding rules out. The true label stays in its set for 81.4% of development rows. The main gap is the hypochromia cut-off (MCHC 30 cited against about 32 in the labels): with 32 the coverage is 96.5% (`results/phase2/`).
- **Phase 3, machine learning.** Nested repeated cross-validation (5 × 5 outer folds, 3-fold inner tuning) on two feature sets, with class weights or SMOTE. The random forest with SMOTE reaches macro-F1 0.9746. The errors cluster at the label cut-offs (decision tree: 1.2% error near a cut-off against 0.1% elsewhere), so high scores mostly mean the models recovered the labeling rule, not clinical skill. The discrimination indices did not help (H3 not supported). The mean-fill pattern is a real shortcut (a probe on it alone reaches balanced accuracy 0.243 against 0.111 for chance) (`results/phase3/`).

Main results, development CV, amended calibration (mean ± std over 25 folds):

| Feature set | Model (arm) | Macro-F1 all rows | Balanced acc. all rows | MCC all rows | Macro-F1 mean-filled rows | Macro-F1 other rows |
|---|---|---|---|---|---|---|
| F-core | Majority dummy | 0.047 ± 0.000 | 0.111 ± 0.000 | 0.000 ± 0.000 | 0.066 ± 0.004 | 0.023 ± 0.006 |
| F-core | Logistic regression (SMOTE) | 0.863 ± 0.050 | 0.886 ± 0.049 | 0.896 ± 0.026 | 0.852 ± 0.061 | 0.812 ± 0.093 |
| F-core | Decision tree (class weights) | 0.970 ± 0.029 | 0.974 ± 0.028 | 0.990 ± 0.008 | 0.988 ± 0.020 | 0.944 ± 0.061 |
| F-core | Random forest (SMOTE) | 0.975 ± 0.022 | 0.977 ± 0.023 | 0.988 ± 0.009 | 0.979 ± 0.039 | 0.946 ± 0.054 |
| F-core | Gradient boosting (SMOTE) | 0.945 ± 0.045 | 0.945 ± 0.046 | 0.977 ± 0.012 | 0.967 ± 0.037 | 0.894 ± 0.069 |
| F-all | Majority dummy | 0.047 ± 0.000 | 0.111 ± 0.000 | 0.000 ± 0.000 | 0.066 ± 0.004 | 0.023 ± 0.006 |
| F-all | Logistic regression (SMOTE) | 0.836 ± 0.058 | 0.854 ± 0.060 | 0.871 ± 0.034 | 0.814 ± 0.067 | 0.757 ± 0.085 |
| F-all | Decision tree (class weights) | 0.972 ± 0.028 | 0.974 ± 0.033 | 0.990 ± 0.010 | 0.983 ± 0.040 | 0.948 ± 0.064 |
| F-all | Random forest (SMOTE) | 0.954 ± 0.039 | 0.954 ± 0.045 | 0.979 ± 0.012 | 0.949 ± 0.057 | 0.917 ± 0.075 |
| F-all | Gradient boosting (SMOTE) | 0.941 ± 0.038 | 0.944 ± 0.043 | 0.976 ± 0.013 | 0.945 ± 0.054 | 0.904 ± 0.075 |

- **Phase 4, hybrid agent.** Adding the cited rules cost accuracy: the random forest scores 0.975 alone and 0.787 as a hybrid without abstention, so the pre-registered non-inferiority hypothesis H1 is not supported. The referral trigger chosen by the pre-declared rule is the model's (it catches 99.4% of referral cases). With abstention at matched coverage the hybrid scores 0.997 against 0.850 for random abstention but 1.000 for confidence-only abstention, so H2 is not supported (`results/phase4/`).
- **Phase 5, explanations and audit.** One SHAP explainer per model type, a fixed-template summary and a JSON audit record per case. The automatic consistency check passed on 963/963 development cases. Decisions are sensitive to small input noise (86.7% unchanged at 1% noise) (`results/phase5/`).
- **Phase 6, the single held-out test run.** On the 242 valid test rows: random forest macro-F1 0.934, cut-off cascade 0.975, hybrid without abstention 0.733. The agent answered 66.5% of cases with 0 of 161 errors and referred 26/26 referral cases. Acceptance targets: 1 pass, 2 pass, 3 pass, 4 fail, 5 fail (`results/phase6/`).

## How the analysis was kept honest

- **Pre-registration.** The analysis rules of Phases 3, 4 and 6 were written and confirmed before any result was opened: `notes/phase3_preregistration.md`, `notes/phase4_preregistration.md`, `notes/phase6_preregistration.md` (shared unaltered, other files they mention are not part of this repository).
- **One amendment after the results, disclosed.** scikit-learn calibrates models with raw decision scores (logistic regression, gradient boosting) on those scores, one class at a time. That collapsed logistic regression (macro-F1 0.490 against 0.854 after the fix). All models were then recalibrated on predicted probabilities. Both versions are kept: `results/phase3/` (pre-registered) and `results/phase3/amended/`, compared in `results/phase3/calibration_amendment_comparison.md`.
- **Held-out test set.** One stratified 80/20 split with seed 42 after removing duplicates: 985 development rows, 247 test rows. The test rows are in the dataset file, but only the final evaluation read them, once, from a frozen and tagged commit (`load_test()` in `src/data.py` refuses otherwise). The split file holds only row numbers, and per-row outputs of the final test are not shared.
- **No leakage across folds.** Scaling, SMOTE and calibration are fitted inside each training fold only.

## Folder layout

```
notebooks/      anemia_project.ipynb (runs and checks the whole project) and its HTML copy
src/            all project code (validator, rule engine, models, agent, explanations, statistics, reports)
tests/          unit tests (pytest)
results/        outputs per phase, committed so the notebook can check them
notes/          the three pre-registrations
data/           the dataset CSV, NOTICE.md (source and licence) and the Apache 2.0 licence text
```

## Data and licence

The dataset is *Anemia Types Classification* by Ehab Aboelnaga on Kaggle (https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification), Apache License 2.0. It is redistributed unmodified in `data/` with its licence and a notice. `python src/check_dataset.py` verifies its SHA-256 fingerprint. The dataset states no units and no provenance beyond its Kaggle page.

## Command line (optional)

The notebook calls the same scripts, which can also be run from the repository root, for example:
```
python src/check_dataset.py
python -m pytest tests -q
python src/build_notebooks.py --project      # rebuild and execute the notebook, write the HTML copy
```
