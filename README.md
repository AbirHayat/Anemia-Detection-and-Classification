# Abstaining hybrid agent for anemia types from CBC data

**Research prototype, not a diagnostic tool.** Nothing here should be used to diagnose or treat anyone.

This repository is shared privately with my course instructor for progress review.

This project classifies a complete blood count (CBC) record into one of nine labels (five anemia types, healthy and three white-cell or platelet disorders). It combines a cited rule engine (an expert system) with machine-learning classifiers. When the two disagree or confidence is low, the agent abstains or refers the case to a person. Every decision comes with a rule trace and an explanation.

## Status

- Phase 1: data audit and cleaning (done)
- Phase 2: cited rule engine (done)
- Phase 3: ML pipeline and nested CV (in progress)
- Phase 4: hybrid agent, abstention and referral (planned)
- Phase 5: explanations, audit log and stress tests (planned)
- Phase 6: final evaluation on the held-out test set (planned)

## Folder layout

```
data/          put the dataset CSV here (not included, see below)
src/           all code (data loading, validator, cleaning, rule engine, reports)
tests/         unit tests (pytest)
notebooks/     one walkthrough notebook per finished phase, with stored outputs
results/       outputs of the scripts, per phase (split = row indices only)
```

## Dataset

The data file is not in this repository. Download `diagnosed_cbc_data_v4.csv` from the Kaggle dataset *Anemia Types Classification* by E. Aboelnaga (Apache 2.0 licence):

https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification

Place it at `data/diagnosed_cbc_data_v4.csv` and check it is the exact file this project used:

```
python src/check_dataset.py
```

Expected SHA-256: `f7dee8d4fe500aec6e8e4bfbeb6df6b83bc86bef03d540721e0aeaad2759fcdb` (125,590 bytes). A different file will not reproduce the results.

## Held-out test set

Exact duplicates were removed and one stratified 80/20 split was made with seed 42 (985 development rows, 247 test rows). `results/split/dev_test_split.csv` holds only row numbers and the set name, no labels. The test rows stay locked until the final evaluation. The loader refuses to return them (`load_test()` in `src/data.py` raises an error) and no script or notebook here reads them.

## Install

Python 3.12 (tested with 3.12.6).

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (macOS / Linux)
pip install -r requirements.txt
```

`requirements.txt` pins the direct dependencies and is the file to use on macOS and Linux. `requirements-lock.txt` pins every package exactly as installed in a fresh-clone test, but it was tested on Windows only (Python 3.12.6) and contains Windows-only packages, so other systems should use `requirements.txt`.

## Run

```
python -m pytest tests -q              # unit tests (no dataset needed)
python src/check_dataset.py            # dataset present and unchanged
python src/phase1_clean.py             # Phase 1 cleaning (writes results/phase1/)
python src/phase1_data_report.py       # Phase 1 data report and figures
python src/phase2_rules_baseline.py    # Phase 2 rules-only baseline
```

Run `phase1_clean.py` before the other scripts and the notebooks. It rebuilds the per-row cleaning file that they read (`results/phase1/dev_rows.csv`, not stored in the repository).

## Notebooks

```
jupyter lab notebooks/
```

Open a walkthrough and run the cells from top to bottom. Each notebook replays one phase with the project code, has a cell for trying your own values and ends by checking that its numbers match the files in `results/`. To rebuild and re-execute them all:

```
python src/build_notebooks.py --phase 1 2
```

## Data limits to keep in mind

The dataset states no units and no provenance beyond its Kaggle page. Many rows hold filled-in average values, and the labels appear to follow simple cut-offs on the same CBC values the models see. High scores therefore mostly show that a model can recover the labeling rule. They are not evidence of clinical skill.

## AI use

This project was built with Claude Code (Anthropic) under my direction. I made the design decisions, and the AI drafted code and text that I reviewed.
