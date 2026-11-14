"""Build and execute the per-phase walkthrough notebooks in notebooks/.

Study and viva material only. Research prototype, not a diagnostic tool.

Rules for every notebook (decisions of 14 Nov 2026)
  * The notebooks RUN the project: each phase notebook first snapshots the committed results/,
    then re-runs that phase's scripts (run_step), walks through the results with the project
    code and asserts that the re-run reproduces the snapshot.
  * No new kind of result: the scripts are the same ones used on the command line.
  * The test set is never touched (no load_test), and results/phase3/ is not read before
    Phase 3 is opened (allow flag per phase below).
  * Markdown cells pass the project's writing-style check (src/style_check.py).

Usage:  python src/build_notebooks.py --phase 1 2 3 0      (0 = notebooks/00_start_here.ipynb)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
sys.path.insert(0, str(ROOT / "src"))
from style_check import style_check  # noqa: E402

PHASE3_OPEN = {0: True, 1: False, 2: False, 3: True}   # may a notebook mention or read Phase 3 outputs?

SETUP = '''import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from IPython.display import Image, display

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def run_step(script, *args, show=25):
    """Run one project script from src/ with this notebook's Python, from the repository root."""
    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-X", "faulthandler=0", str(ROOT / "src" / script), *args], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    lines = (r.stdout or "").rstrip().splitlines()
    if show:
        if len(lines) > show:
            print(f"... ({len(lines) - show} earlier lines not shown)")
        print("\\n".join(lines[-show:]))
    print(f"[{' '.join([script, *args])} finished in {time.perf_counter() - t0:.1f} s, exit code {r.returncode}]")
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"{script} failed")


print("Research prototype, not a diagnostic tool. The test set is not used in this notebook.")'''


def md(text: str):
    return ("md", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


# ----------------------------------------------------------------------------- Phase 1
PHASE1 = [
    md("""# Phase 1 walkthrough: data audit and cleaning

This notebook runs Phase 1 and then walks through it. Section 0 notes the numbers in the committed `results/phase1/` files and re-runs the Phase 1 scripts, and section 8 checks that the re-run reproduces them. The test set is never opened.

**How to use it:** run the cells from top to bottom (Shift+Enter). Read the note above each cell first, then say in your own words what the output shows. The cell marked **Try it** is for you to edit."""),
    code(SETUP + '''

from data import (CLASS_ORDER, FILL_COLS, RAW, RAW_SHA256, TARGET, load_dev, load_raw, sha256_of)
from validator import consistency_flags, screen_frame, validate_record'''),
    md("""## 0. Run Phase 1

First we note the committed reports (a snapshot). Then the three Phase 1 scripts run exactly as they would on the command line: the dataset check, the cleaning and the data report. They rewrite `results/phase1/`."""),
    code('''P1 = ROOT / "results" / "phase1"
before_clean = (P1 / "cleaning_report.txt").read_text(encoding="utf-8")
before_data = (P1 / "data_report.txt").read_text(encoding="utf-8")

run_step("check_dataset.py")
run_step("phase1_clean.py")
run_step("phase1_data_report.py")'''),
    md("""## 1. Is the raw file unchanged?

A hash is a fingerprint of a file. If one byte changes, the fingerprint changes completely. We compare today's fingerprint with the one recorded on 7 Nov."""),
    code('''raw = load_raw()
print("rows:", len(raw), " columns:", raw.shape[1], " empty cells:", int(raw.isna().sum().sum()))
print("hash matches the 7 Nov record:", sha256_of(RAW) == RAW_SHA256)'''),
    md("""## 2. The split

Exact duplicates were removed first (49 rows), then one stratified 80/20 split was made with seed 42. Stratified means each part keeps the same class mix. `src/phase1_clean.py` re-runs this and confirms it gives the same split. Only the development side is shown here. The test rows stay locked until the final evaluation."""),
    code('''split = pd.read_csv(ROOT / "results" / "split" / "dev_test_split.csv", usecols=["raw_row", "set"])
print("rows per set:", split["set"].value_counts().to_dict())
load_dev(clean=False)[TARGET].value_counts().reindex(CLASS_ORDER).rename("development rows").to_frame()'''),
    md("""## 3. The input validator

The validator is the gatekeeper. It checks each of the 14 columns against loose screening ranges (your own limits, not clinical ranges) and lists every reason a record fails. The agent uses the same code at prediction time."""),
    code('''GOOD = {"WBC": 7.4, "LYMp": 25.8, "NEUTp": 77.5, "LYMn": 1.9, "NEUTn": 5.1, "RBC": 4.6,
        "HGB": 12.2, "HCT": 46.2, "MCV": 86.6, "MCH": 27.7, "MCHC": 32.0, "PLT": 213.0,
        "PDW": 14.3, "PCT": 0.26}
print("good record ->", validate_record(GOOD) or "valid")
for col, value in [("HGB", -10), ("MCV", -79.3), ("HCT", 3715), ("NEUTp", 5317), ("MCH", 3117)]:
    print(f"{col} = {value}:", validate_record({**GOOD, col: value}))
print("NEUTn larger than WBC is only flagged:", consistency_flags({**GOOD, "NEUTn": 9.0, "WBC": 7.0}))'''),
    md("""**Try it:** change the values below and run the cell. What does the validator say about a missing column or a text value?"""),
    code('''my_record = {**GOOD, "PLT": 0, "MCHC": 50}
print(validate_record(my_record) or "valid")'''),
    md("""## 4. Cleaning the development rows

We run the validator on all 985 development rows and drop the rows it rejects (a rule fixed before cleaning). Most rejected values look like a lost decimal point, but fixing them would mean guessing, so we drop them."""),
    code('''dev_all = load_dev(clean=False)
reasons = screen_frame(dev_all)
valid = reasons == ""
n_rejected, n_kept = int((~valid).sum()), int(valid.sum())
print(f"rejected {n_rejected} of {len(dev_all)}, kept {n_kept}")
table = pd.DataFrame({"development rows": dev_all[TARGET].value_counts(),
                      "rejected": dev_all.loc[~valid, TARGET].value_counts()}).reindex(CLASS_ORDER).fillna(0).astype(int)
table'''),
    md("""## 5. The mean-fill pattern

In many rows six columns hold exactly the same value (the column average). That looks like missing values were filled before upload. The share differs a lot by class, so a model could learn the filling instead of the biology."""),
    code('''dev = load_dev(clean=True)
filled = dev["fill_count"] == 6
n_filled = int(filled.sum())
share = (dev.groupby(TARGET)["fill_count"].apply(lambda s: (s == 6).mean() * 100)).reindex(CLASS_ORDER).round(1)
print(f"mean-filled rows: {n_filled} of {len(dev)}")
display(share.rename("mean-filled share (%)").to_frame())
display(Image(filename=str(ROOT / "results" / "phase1" / "figures" / "fig_fill_by_class.png"), width=520))'''),
    md("""## 6. Do the columns mean what their names say?

Two checks on the rows that are not mean-filled. First, the absolute lymphocyte count should equal WBC times the lymphocyte percentage. Second, the textbook formula MCV = HCT × 10 / RBC should hold if HCT is in % and RBC in 10¹²/L (the formula is still unverified until cited)."""),
    code('''def within(obs, calc, tol):
    return ((obs - calc).abs() / obs.abs() <= tol).mean() * 100

o = dev[~filled]
print(f"LYMn = WBC x LYMp / 100 within 5%: {within(o['LYMn'], o['WBC'] * o['LYMp'] / 100, 0.05):.1f}% of other rows")
mcv_calc = dev["HCT"] * 10 / dev["RBC"]
print(f"MCV = HCT x 10 / RBC within 2%: other rows {within(dev.loc[~filled, 'MCV'], mcv_calc[~filled], 0.02):.1f}%, "
      f"filled rows {within(dev.loc[filled, 'MCV'], mcv_calc[filled], 0.02):.1f}%")'''),
    md("""The formula holds in real rows and fails in filled rows, because the filled HCT is a constant. That supports both the unit reading and the mean-fill reading.

## 7. The label rule (the big finding)

A simple cascade with six round cut-offs, read off the development data, reproduces almost every label. The cut-offs are not clinical and never go into the rule engine. We reuse the cascade function from `src/phase2_rules_baseline.py` and merge the two microcytic labels, as in the Phase 1 report."""),
    code('''from phase2_rules_baseline import cascade_predict

p = {"hgb": 13.0, "plt": 150.0, "wbc": 10.0, "mcv_micro": 80.0, "mcv_macro": 100.0,
     "mchc_normo": 32.0, "mchc_ida_oma": 0.0}
merge = {"Iron deficiency anemia": "Microcytic", "Other microcytic anemia": "Microcytic"}
pred = dev.apply(lambda r: cascade_predict(r, p), axis=1).replace(merge)
truth = dev[TARGET].replace(merge)
n_agree = int((pred == truth).sum())
print(f"cascade agrees with the label in {n_agree} of {len(dev)} rows ({100 * n_agree / len(dev):.1f}%)")
pd.crosstab(truth[pred != truth], pred[pred != truth])'''),
    md("""## 8. Did the re-run reproduce the committed results?

The re-run reports must be identical to the committed ones (the snapshot from section 0), and the numbers recomputed in this notebook must match them. If anything differs, this cell fails."""),
    code('''assert (P1 / "cleaning_report.txt").read_text(encoding="utf-8") == before_clean, "cleaning report changed"
assert (P1 / "data_report.txt").read_text(encoding="utf-8") == before_data, "data report changed"
clean_txt, data_txt = before_clean, before_data
saved_rejected, saved_kept = map(int, re.search(r"rows rejected: (\\d+) of \\d+; rows kept: (\\d+)", clean_txt).groups())
saved_filled = int(re.search(r"all kept rows: (\\d+) / \\d+", clean_txt).group(1))
saved_agree = int(re.search(r"label \\(IDA and other microcytic merged\\): (\\d+) /", data_txt).group(1))
assert (n_rejected, n_kept, n_filled, n_agree) == (saved_rejected, saved_kept, saved_filled, saved_agree)
print("re-run identical to the committed reports, recomputed numbers match:", (n_rejected, n_kept, n_filled, n_agree))'''),
    md("""## 9. Questions for you (answer without looking)

1. How many development rows did the validator reject, and why did we drop them instead of fixing them?
2. Give one piece of evidence that the six columns with a repeated value were mean-filled, using the CBC formulas.
3. Why must the cut-offs from the label-rule check (HGB 13, PLT 150, WBC 10 and so on) not go into the rule engine?"""),
]

# ----------------------------------------------------------------------------- Phase 2
PHASE2 = [
    md("""# Phase 2 walkthrough: the rule engine

This notebook runs Phase 2 and then walks through it. Section 0 notes the committed results and re-runs the rules-only baseline, and section 7 checks that the re-run reproduces them. The test set is never opened, and the machine-learning outputs of the next phase are not read here.

**How to use it:** run the cells in order. The **Try it** cell is for your own values."""),
    code(SETUP + '''

from data import CLASS_ORDER, TARGET, load_dev
from rules import SOURCES, VARIANTS, Thresholds, run'''),
    md("""## 0. Run Phase 2

We note the committed rules-only report (a snapshot), then re-run `src/phase2_rules_baseline.py`. It scores the rule engine on the 963 clean development rows and freezes the label-cascade baseline."""),
    code('''P2 = ROOT / "results" / "phase2"
before = (P2 / "rules_baseline.txt").read_text(encoding="utf-8")
before_cascade = (P2 / "cascade_frozen.json").read_text(encoding="utf-8")
run_step("phase2_rules_baseline.py", show=12)'''),
    md("""## 1. The thresholds the code actually uses

Each cut-off was cited and logged on 14 Nov 2026, before any machine-learning result existed. The table is built from the code itself, so it cannot drift from what the engine does."""),
    code('''t = Thresholds()
rows = [("anemia if HGB below", t.hgb_anemia, "g/dL", "WHO"), ("microcytic if MCV below", t.mcv_micro, "fL", "MERCK_ANEMIA"),
        ("macrocytic if MCV above", t.mcv_macro, "fL", "MERCK_ANEMIA"), ("hypochromic if MCHC below", t.mchc_hypo, "g/dL", "MERCK_ANEMIA"),
        ("low platelets if PLT below", t.plt_low, "10^9/L", "AAFP_PLT"), ("leukocytosis if WBC above", t.wbc_high, "10^9/L", "AAFP_WBC"),
        ("leukopenia if WBC below", t.wbc_low, "10^9/L", "MERCK_WBC")]
pd.DataFrame([(a, b, c, SOURCES[d]) for a, b, c, d in rows], columns=["rule", "cut-off", "unit", "source"])'''),
    md("""## 2. Forward chaining, one step at a time

Working memory starts with the CBC values. Each rule fires when the facts it needs exist, and its result becomes a new fact. Later rules build on earlier ones (F9 uses F7 and F8). It stops when no rule can add anything. Below is the made-up record from the unit tests."""),
    code('''example = {"WBC": 7.4, "RBC": 5.0, "HGB": 10.0, "MCV": 70.0, "MCH": 22.0, "MCHC": 29.0, "PLT": 120.0}
res = run(example)
for step in res.trace:
    print(f"{step['rule']:>4}  {step['text']}")
    print(f"      inputs {step['inputs']} -> {step['result']}")
print("\\ncompatible set:", res.compatible)
print("excluded (label: rules):", res.excluded)'''),
    md("""## 3. Exclusion only, on three contrasting records

Rules only remove labels that a cited finding rules out. They never name the answer. Look at what stays in each set and why."""),
    code('''NORMAL = {"WBC": 7.0, "RBC": 4.8, "HGB": 14.0, "MCV": 88.0, "MCH": 29.0, "MCHC": 33.0, "PLT": 250.0}
cases = {"anemia + low platelets": {**NORMAL, "HGB": 10, "MCV": 70, "PLT": 100},
         "normocytic anemia, MCHC 31": {**NORMAL, "HGB": 11, "MCHC": 31},
         "WBC 10.5, otherwise normal": {**NORMAL, "WBC": 10.5}}
for name, rec in cases.items():
    print(f"{name:28s} primary: {run(rec).compatible}")
    print(f"{'':28s} MCHC 32 check: {run(rec, VARIANTS['mchc32']).compatible}")'''),
    md("""The second case shows the MCHC gap: with the cited cut-off (30) an MCHC of 31 counts as normochromic, while with 32 it counts as hypochromic. The third case shows the WBC gap: 10.5 is normal under the cited cut-off (11), but the labels seem to treat anything above 10 as high.

**Try it:** change the values and run the cell."""),
    code('''mine = {**NORMAL, "HGB": 12.5, "MCV": 102, "PLT": 140}
r = run(mine)
print("primary:", r.compatible, "| refer trigger:", r.refer_rule_fires)
print("HGB 12 check:", run(mine, VARIANTS["hgb12"]).compatible)'''),
    md("""## 4. Rules-only baseline on the development rows

Rule-set coverage is how often the true label is still inside the set. We recompute it for all four threshold variants on the 963 clean development rows."""),
    code('''dev = load_dev(clean=True)
summary = {}
for name, th in VARIANTS.items():
    sets = [run(r.to_dict(), th).compatible for _, r in dev.iterrows()]
    covered = sum(lab in s for lab, s in zip(dev[TARGET], sets))
    summary[name] = {"covered": covered, "coverage %": round(100 * covered / len(dev), 1),
                     "mean set size": round(float(np.mean([len(s) for s in sets])), 2)}
pd.DataFrame(summary).T.astype({"covered": int})'''),
    md("""## 5. Where the cited rules and the labels disagree

Rows whose values fall between a cited cut-off and the cut-off the labels seem to use."""),
    code('''g_wbc = (dev["WBC"] > 10) & (dev["WBC"] <= 11)
normo_anemia = (dev["HGB"] < 13) & dev["MCV"].between(80, 100)
g_mchc = normo_anemia & (dev["MCHC"] >= 30) & (dev["MCHC"] < 32)
n_wbc, n_mchc = int(g_wbc.sum()), int(g_mchc.sum())
print(f"WBC in (10, 11]: {n_wbc} rows ->", dev.loc[g_wbc, TARGET].value_counts().to_dict())
print(f"normocytic anemia with MCHC in [30, 32): {n_mchc} rows ->", dev.loc[g_mchc, TARGET].value_counts().to_dict())'''),
    md("""## 6. The frozen label-cascade baseline

This is the probable labeling rule from Phase 1 with one fitted value. It was frozen on 14 Nov and will be scored once on the test set in Phase 6. It is not a clinical rule."""),
    code('''import json
print(json.dumps(json.loads((ROOT / "results" / "phase2" / "cascade_frozen.json").read_text(encoding="utf-8")), indent=2))'''),
    md("""## 7. Did the re-run reproduce the committed results?

The re-run report and frozen baseline must be identical to the committed ones, and the coverage numbers and gap counts recomputed above must match them."""),
    code('''assert (P2 / "rules_baseline.txt").read_text(encoding="utf-8") == before, "rules report changed"
assert (P2 / "cascade_frozen.json").read_text(encoding="utf-8") == before_cascade, "frozen cascade changed"
txt = before
saved = [int(x) for x in re.findall(r"rule-set coverage: (\\d+) / 963", txt)]
assert saved == [v["covered"] for v in summary.values()], (saved, summary)
assert n_wbc == int(re.search(r"WBC in \\(10, 11\\].*?: (\\d+) rows", txt).group(1))
assert n_mchc == int(re.search(r"MCHC in \\[30, 32\\).*?: (\\d+) rows", txt).group(1))
print("re-run identical to the committed report, recomputed numbers match:", saved, n_wbc, n_mchc)'''),
    md("""## 8. Questions for you (answer without looking)

1. What does forward chaining mean, and what stops it from running forever?
2. Which cited cut-off causes most of the rule-set misses, and how many rows does it cost?
3. Why do the three indices appear in the trace but never change the label set?"""),
]

# ----------------------------------------------------------------------------- Phase 3
PHASE3 = [
    md("""# Phase 3 walkthrough: the machine-learning models

This notebook runs Phase 3 and then walks through it. Section 0 notes the committed `results/phase3/` files and re-runs the Phase 3 evaluation, and section 8 checks that the re-run reproduces them. Development rows only. The test set is never opened.

**How to use it:** run the cells in order. With the default setting the notebook reuses the included cross-validation predictions and rebuilds every table and test from them. Set `RUN_FULL_CV = True` in the next cell to retrain every model from scratch first (see the README for the run time)."""),
    code(SETUP + '''

import io
import json
import warnings

from IPython.display import Markdown

from data import CLASS_ORDER, load_dev
from metrics import per_class, scores
from stats_tests import corrected_repeated_kfold_t

warnings.filterwarnings("ignore")
P3 = ROOT / "results" / "phase3"'''),
    md("""## Settings"""),
    code('''RUN_FULL_CV = False   # True: retrain all models first (three cross-validation scripts, slow)
if os.environ.get("ANEMIA_RUN_FULL_CV") == "1":
    RUN_FULL_CV = True
print("RUN_FULL_CV =", RUN_FULL_CV)'''),
    md("""## 0. Run Phase 3

First a snapshot of the committed results. Then, if `RUN_FULL_CV` is on, the three cross-validation scripts retrain every model (nested repeated CV, the index features and shortcut probes, and the probability-based calibration). In every case the evaluation runs again for both calibration versions, followed by the comparison table and the calibration diagnostic."""),
    code('''SNAP = {rel: (P3 / rel).read_text(encoding="utf-8") for rel in
        ["evaluation.txt", "amended/evaluation.txt", "summary_metrics_cal.csv", "amended/summary_metrics_cal.csv",
         "calibration_amendment_comparison.md", "calibration_diagnostic.txt"]}
if RUN_FULL_CV:
    run_step("phase3_ml_only_cv.py", show=3)
    run_step("phase3_h3_h4_cv.py", show=3)
    run_step("phase3_amended_calibration.py", show=3)
run_step("phase3_evaluate.py", show=0)
run_step("phase3_evaluate.py", "--version", "amended", show=0)
run_step("phase3_compare_calibration.py", show=0)
run_step("phase3_calibration_diagnostic.py", show=0)'''),
    md("""## 1. The rules, fixed before any result was opened

The analysis was pre-registered on 14 Nov 2026 (see `PREREGISTRATION_phase3.md` where it is shared). In short:

1. Models are judged on calibrated predictions (the class with the highest calibrated probability).
2. For each model type the imbalance arm (class weights or SMOTE) with the higher mean F-core macro-F1 is kept. A tie within 0.001 goes to class weights.
3. The H2 model is the best model type on F-core with its kept arm.
4. H3 and H4 use the corrected repeated k-fold cv test with Holm adjustment.

**Post-results amendment:** scikit-learn calibrated logistic regression and gradient boosting on their raw decision scores, which collapsed logistic regression. After seeing that, every model was recalibrated on its predicted probabilities. Both versions are kept and reported."""),
    md("""## 2. Macro-F1 per model, recomputed from the predictions

Each development row has an out-of-fold prediction in each of the 5 repeats. We recompute macro-F1 for each of the 25 outer folds, then the mean and standard deviation, for both calibration versions."""),
    code('''orig = pd.read_csv(P3 / "ml_only" / "oof_predictions.csv.gz")
amend = pd.read_csv(P3 / "amended" / "oof_ml_only.csv.gz")
for df in (orig, amend):
    df["pred"] = df[[f"p_cal_{j}" for j in range(9)]].to_numpy().argmax(axis=1)


def fold_f1(df, model, imb, fset="F-core"):
    g = df[(df.fset == fset) & (df.model == model) & (df.imb == imb)]
    return g.groupby(["rep", "fold"]).apply(lambda x: scores(x["y"], x["pred"])["macro_f1"], include_groups=False)


configs = [("dummy", "none")] + [(m, i) for m in ["logreg", "tree", "forest", "hgb"] for i in ["weights", "smote"]]
rows = []
for m, i in configs:
    a, b = fold_f1(orig, m, i), fold_f1(amend, m, i)
    rows.append({"model": m, "arm": i, "original": f"{a.mean():.3f} ± {a.std():.3f}",
                 "amended": f"{b.mean():.3f} ± {b.std():.3f}", "orig_mean": a.mean(), "amend_mean": b.mean()})
tab = pd.DataFrame(rows)
tab[["model", "arm", "original", "amended"]]'''),
    md("""## 3. The pre-registered choice rules, applied to both versions"""),
    code('''def choose(col):
    arm = {}
    for m in ["logreg", "tree", "forest", "hgb"]:
        w = tab[(tab.model == m) & (tab.arm == "weights")][col].item()
        s = tab[(tab.model == m) & (tab.arm == "smote")][col].item()
        arm[m] = "smote" if s > w + 0.001 else "weights"
    best = max(arm, key=lambda m: tab[(tab.model == m) & (tab.arm == arm[m])][col].item())
    return arm, best


arm_o, h2_o = choose("orig_mean")
arm_a, h2_a = choose("amend_mean")
print("original arms:", arm_o, "-> H2 model:", h2_o, "with", arm_o[h2_o])
print("amended arms: ", arm_a, "-> H2 model:", h2_a, "with", arm_a[h2_a])'''),
    md("""## 4. One corrected t-test, worked by hand

Random forest with class weights against SMOTE on F-core. The 25 fold scores share most of their training rows, so the variance gets the correction factor 1/25 + n2/n1 (n2/n1 = 1/4 for 5 folds) instead of 1/25 (Bouckaert and Frank 2004)."""),
    code('''d = fold_f1(orig, "forest", "weights").sort_index().to_numpy() - fold_f1(orig, "forest", "smote").sort_index().to_numpy()
k, r, ratio = 5, 5, 0.25
mean_d, var_d = d.mean(), d.var(ddof=1)
se = np.sqrt((1 / (k * r) + ratio) * var_d)
print(f"mean difference {mean_d:+.4f}, variance {var_d:.6f}, corrected standard error {se:.4f}, t = {mean_d / se:.2f}")
res = corrected_repeated_kfold_t(d, k, r, ratio)
print(f"src/stats_tests.py gives t = {res.t:.2f}, two-sided p = {res.p:.4f}")
print(f"a naive t-test would give t = {mean_d / np.sqrt(var_d / (k * r)):.2f} (too confident)")'''),
    md("""## 5. Why calibration broke logistic regression

Logistic regression and gradient boosting expose raw decision scores, and scikit-learn calibrates those when they exist, one class at a time. Raw multinomial scores only make sense relative to each other within a row, so that step destroys them. The tree, forest and dummy have no raw scores and were always calibrated on probabilities."""),
    code('''from ml_pipelines import configs as model_configs, make_pipeline

Xs, ys = np.random.default_rng(0).normal(size=(90, 7)), np.repeat(np.arange(9), 10)
print({f"{m}/{i}": hasattr(make_pipeline(m, i).fit(Xs, ys), "decision_function") for m, i in model_configs()})


def recalls(df, m, i):
    g = df[(df.fset == "F-core") & (df.model == m) & (df.imb == i)]
    return per_class(g["y"], g["pred"])["recall"].to_numpy()


display(pd.DataFrame({"class": CLASS_ORDER, "original (raw scores)": recalls(orig, "logreg", "weights"),
                      "amended (probabilities)": recalls(amend, "logreg", "weights")}).round(2))
display(Markdown((P3 / "calibration_amendment_comparison.md").read_text(encoding="utf-8")))'''),
    md("""## 6. H3 (index features) and H4 (shortcut probes)

H3 asked whether the three discrimination indices raise iron deficiency and other microcytic recall. H4 asked whether the mean-fill pattern is a shortcut. We print the H3 tests and recompute the probe scores."""),
    code('''ev = (P3 / "amended" / "evaluation.txt").read_text(encoding="utf-8")
print(re.search(r"== P4\\. H3 PRIMARY.*?(?=\\n\\n)", ev, re.S).group(0))
pr = pd.read_csv(P3 / "h4_probes" / "oof_predictions.csv.gz")


def probe_ba(probe, kind):
    g = pr[(pr.probe == probe) & (pr.kind == kind)]
    return g.groupby(["rep", "fold"]).apply(
        lambda x: scores(x["y"], x["pred"], labels=sorted(x["y"].unique()))["balanced_accuracy"], include_groups=False)


probe1 = probe_ba("probe1_fill_indicator", "probe").mean()
for probe in ["probe1_fill_indicator", "probe2_fcore_to_filled"]:
    print(f"{probe}: balanced accuracy {probe_ba(probe, 'probe').mean():.3f} against dummy {probe_ba(probe, 'dummy').mean():.3f}")'''),
    md("""## 7. Filled against other rows, and errors near the label cut-offs

Every model scores lower on the rows that are not mean-filled, so both groups are columns of the main table. The errors cluster near the cut-offs the labels appear to follow (bands fixed in the pre-registration)."""),
    code('''dev = load_dev(clean=True)
BANDS = {"HGB": [(12.5, 13.5)], "PLT": [(135, 165)], "WBC": [(9.5, 11.5)], "MCV": [(78, 82), (98, 102)],
         "MCHC": [(31.4, 32.5)]}
near = pd.Series(False, index=dev.index)
for c, bands in BANDS.items():
    for lo, hi in bands:
        near |= dev[c].between(lo, hi)
g = amend[(amend.fset == "F-core") & (amend.model == "tree") & (amend.imb == "weights")]
err, nn = (g["pred"] != g["y"]).to_numpy(), g["raw_row"].map(near).to_numpy()
near_rate, far_rate = 100 * err[nn].mean(), 100 * err[~nn].mean()
print(f"decision tree: error rate near a cut-off {near_rate:.1f}% against {far_rate:.1f}% elsewhere "
      f"({100 * near.mean():.1f}% of rows are near a cut-off)")
display(Markdown((P3 / "amended" / "main_results_table.md").read_text(encoding="utf-8")))
display(Image(filename=str(P3 / "amended" / "figures" / "fig_macro_f1_by_model.png"), width=520))'''),
    md("""## 8. Did the re-run reproduce the committed results?

Without `RUN_FULL_CV` the evaluation reads the same predictions, so every rebuilt file must be identical to the snapshot. With `RUN_FULL_CV` on another computer, tiny floating-point differences are possible, so the summary tables are compared with a small tolerance. Then the numbers recomputed in this notebook are checked against the committed reports."""),
    code('''for rel, txt in SNAP.items():
    new = (P3 / rel).read_text(encoding="utf-8")
    if not RUN_FULL_CV:
        assert new == txt, f"{rel} changed"
    elif rel.endswith(".csv"):
        a = pd.read_csv(io.StringIO(txt), header=[0, 1], index_col=[0, 1, 2])
        b = pd.read_csv(io.StringIO(new), header=[0, 1], index_col=[0, 1, 2])
        print(rel, "largest difference after retraining:", float((a - b).abs().max().max()))
        assert float((a - b).abs().max().max()) < 0.005, rel

s_o = pd.read_csv(io.StringIO(SNAP["summary_metrics_cal.csv"]), header=[0, 1], index_col=[0, 1, 2])
s_a = pd.read_csv(io.StringIO(SNAP["amended/summary_metrics_cal.csv"]), header=[0, 1], index_col=[0, 1, 2])
tol = 6e-5 if not RUN_FULL_CV else 0.005
for _, row in tab.iterrows():
    key = ("F-core", row.model, row.arm)
    assert abs(row.orig_mean - s_o.loc[key, ("macro_f1", "mean")]) < tol, key
    assert abs(row.amend_mean - s_a.loc[key, ("macro_f1", "mean")]) < tol, key
saved_h2 = re.search(r"-> H2 model: (\\w+) with (\\w+)", SNAP["evaluation.txt"]).groups()
assert (h2_o, arm_o[h2_o]) == saved_h2 == (h2_a, arm_a[h2_a])
saved_t = float(re.search(r"forest\\s+weights [\\d.]+\\s+smote [\\d.]+.*?t = (-?[\\d.]+)", SNAP["evaluation.txt"]).group(1))
assert round(res.t, 2) == saved_t
assert f"{probe1:.3f}" == re.search(r"probe1_fill_indicator: probe ([\\d.]+)", SNAP["evaluation.txt"]).group(1)
saved_near = re.search(r"F-core\\s+tree\\s+error rate near ([\\d.]+)% vs elsewhere ([\\d.]+)%", SNAP["amended/evaluation.txt"]).groups()
assert (f"{near_rate:.1f}", f"{far_rate:.1f}") == saved_near
print("all checks passed: re-run reproduced the committed results, H2 =", saved_h2, ", t =", saved_t)'''),
    md("""## 9. Questions for you (answer without looking)

1. What is nested cross-validation, and why does tuning happen only in the inner loop?
2. Why did calibration hurt logistic regression but not the random forest?
3. What does probe 2 of H4 show, and why does it matter even though F-core has no mean-filled columns?"""),
]

# ----------------------------------------------------------------------------- start here
START = [
    md("""# Start here

**An abstaining hybrid agent for anemia types from CBC data.** Research prototype, not a diagnostic tool.

The project combines a cited rule engine (an expert system) with machine-learning models on a public complete blood count dataset. It is built in six phases. Phases 1-3 are done, and each has a notebook that **runs that phase** and then explains it:

| Notebook | What it runs | Time with default settings |
|---|---|---|
| `phase1_walkthrough.ipynb` | dataset check, cleaning, data report | under a minute |
| `phase2_walkthrough.ipynb` | rule engine and rules-only baseline | under a minute |
| `phase3_walkthrough.ipynb` | evaluation of all models, both calibration versions, diagnostic | a few minutes |

Every phase notebook first notes the committed results, re-runs the phase and then checks that the re-run reproduces them. The held-out test rows stay locked until the final evaluation (no notebook or script reads them).

This notebook checks your setup and runs the unit tests. Run its cells from top to bottom."""),
    code(SETUP),
    md("""## 1. Python and package versions

The versions this project was tested with are pinned in `requirements.txt`. A different version usually still works, but exact numbers can then differ in the last digit."""),
    code('''import importlib.metadata as im

print("Python", sys.version.split()[0], "(tested with 3.12.6)")
rows = []
for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    if "==" in line and not line.startswith("#"):
        name, want = line.strip().split("==")
        try:
            have = im.version(name)
        except im.PackageNotFoundError:
            have = "missing"
        rows.append((name, want, have, "ok" if have == want else "different"))
pd.DataFrame(rows, columns=["package", "tested", "installed", "status"])'''),
    md("""## 2. Is the dataset present and unchanged?

The CSV in `data/` is the original Kaggle file (see `data/NOTICE.md`). Its SHA-256 fingerprint must match the one recorded on 7 Nov 2026."""),
    code('''run_step("check_dataset.py")'''),
    md("""## 3. Unit tests"""),
    code('''r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:faulthandler", "-q", "--color=no", "tests"], cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout.strip().splitlines()[-1])
assert r.returncode == 0, r.stdout[-3000:]'''),
    md("""## 4. Optional: run all phase notebooks from here

Set `RUN_ALL_PHASE_NOTEBOOKS = True` to execute the three phase notebooks in order and save their outputs. `RUN_FULL_CV = True` also retrains every Phase 3 model from scratch, which takes much longer (see the README). Otherwise open the phase notebooks one by one."""),
    code('''RUN_ALL_PHASE_NOTEBOOKS = False
RUN_FULL_CV = False

if RUN_ALL_PHASE_NOTEBOOKS:
    env = {**os.environ, "ANEMIA_RUN_FULL_CV": "1" if RUN_FULL_CV else "0", "PYTHONIOENCODING": "utf-8"}
    for nb in ["phase1_walkthrough.ipynb", "phase2_walkthrough.ipynb", "phase3_walkthrough.ipynb"]:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=-1", nb], cwd=ROOT / "notebooks",
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        print(f"{nb}: exit code {r.returncode}, {time.perf_counter() - t0:.0f} s")
        if r.returncode != 0:
            print(r.stderr[-3000:])
            break
else:
    print("Skipped. Open the phase notebooks one by one, or set RUN_ALL_PHASE_NOTEBOOKS = True.")'''),
]

PHASES = {0: START, 1: PHASE1, 2: PHASE2, 3: PHASE3}
NAMES = {0: "00_start_here.ipynb", 1: "phase1_walkthrough.ipynb", 2: "phase2_walkthrough.ipynb",
         3: "phase3_walkthrough.ipynb"}


def build(phase: int) -> Path:
    cells = PHASES[phase]
    problems: list[str] = []
    for i, (kind, text) in enumerate(cells):
        if kind == "md":
            problems += style_check(f"phase{phase} cell {i}", text)
        if "load_test" in text:
            problems.append(f"phase{phase} cell {i}: uses load_test (test set is locked)")
        if not PHASE3_OPEN[phase] and ("phase3" in text or "ml_only" in text):
            problems.append(f"phase{phase} cell {i}: reads Phase 3 outputs before Phase 3 is opened")
    if problems:
        raise SystemExit("notebook checks failed:\n  " + "\n  ".join(problems))
    nb = new_notebook(cells=[new_markdown_cell(t) if k == "md" else new_code_cell(t) for k, t in cells])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(NB_DIR)}}).execute()
    out = NB_DIR / NAMES[phase]
    nbformat.write(nb, out)
    return out


def main() -> None:
    if sys.platform == "win32":                  # avoids a zmq event-loop warning on Windows
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, nargs="+", required=True, choices=sorted(PHASES))
    a = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    for ph in a.phase:
        print("built and executed:", build(ph))


if __name__ == "__main__":
    main()
