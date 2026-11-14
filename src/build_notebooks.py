"""Build and execute the per-phase walkthrough notebooks in notebooks/.

Study and viva material only. Research prototype, not a diagnostic tool.

Rules for every walkthrough notebook (decision of 14 Nov 2026)
  * No new results. Every number is computed by existing src/ code or read from results/.
  * Consistency asserts: numbers recomputed in the notebook must equal the saved reports,
    otherwise the build fails.
  * The test set is never touched (no load_test), and results/phase3/ is not read before
    Phase 3 is opened (allow flag per phase below).
  * Markdown cells pass the project's writing-style check (src/style_check.py).

Usage:  python src/build_notebooks.py --phase 1 2
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

PHASE3_OPEN = {1: False, 2: False}   # set True only for notebooks built after Phase 3 opens its results

SETUP = '''import re
import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from IPython.display import Image, display

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
print("Research prototype, not a diagnostic tool. The test set is not used in this notebook.")'''


def md(text: str):
    return ("md", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


# ----------------------------------------------------------------------------- Phase 1
PHASE1 = [
    md("""# Phase 1 walkthrough: data audit and cleaning

This notebook replays Phase 1 step by step. It creates no new results. Every number is computed by the project code in `src/` or read from `results/phase1/`, and section 8 checks that both agree. The test set is never opened.

**How to use it:** run the cells from top to bottom (Shift+Enter). Read the note above each cell first, then say in your own words what the output shows. The cell marked **Try it** is for you to edit."""),
    code(SETUP + '''

from data import (CLASS_ORDER, FILL_COLS, RAW, RAW_SHA256, TARGET, load_dev, load_raw, sha256_of)
from validator import consistency_flags, screen_frame, validate_record'''),
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
    md("""## 8. Consistency check against the saved reports

The numbers above were recomputed here. They must equal the numbers in the saved Phase 1 reports. If any differs, this cell fails."""),
    code('''clean_txt = (ROOT / "results" / "phase1" / "cleaning_report.txt").read_text(encoding="utf-8")
data_txt = (ROOT / "results" / "phase1" / "data_report.txt").read_text(encoding="utf-8")
saved_rejected, saved_kept = map(int, re.search(r"rows rejected: (\\d+) of \\d+; rows kept: (\\d+)", clean_txt).groups())
saved_filled = int(re.search(r"all kept rows: (\\d+) / \\d+", clean_txt).group(1))
saved_agree = int(re.search(r"label \\(IDA and other microcytic merged\\): (\\d+) /", data_txt).group(1))
assert (n_rejected, n_kept, n_filled, n_agree) == (saved_rejected, saved_kept, saved_filled, saved_agree)
print("all recomputed numbers match results/phase1/:", (n_rejected, n_kept, n_filled, n_agree))'''),
    md("""## 9. Questions for you (answer without looking)

1. How many development rows did the validator reject, and why did we drop them instead of fixing them?
2. Give one piece of evidence that the six columns with a repeated value were mean-filled, using the CBC formulas.
3. Why must the cut-offs from the label-rule check (HGB 13, PLT 150, WBC 10 and so on) not go into the rule engine?"""),
]

# ----------------------------------------------------------------------------- Phase 2
PHASE2 = [
    md("""# Phase 2 walkthrough: the rule engine

This notebook replays Phase 2. It creates no new results. Every number is computed by `src/rules.py` or read from `results/phase2/`, and section 7 checks that both agree. The test set is never opened, and the Phase 3 machine-learning outputs are not read here.

**How to use it:** run the cells in order. The **Try it** cell is for your own values."""),
    code(SETUP + '''

from data import CLASS_ORDER, TARGET, load_dev
from rules import SOURCES, VARIANTS, Thresholds, run'''),
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
    md("""## 7. Consistency check against the saved report

The coverage numbers recomputed above must equal those in `results/phase2/rules_baseline.txt`, and so must the two gap counts."""),
    code('''txt = (ROOT / "results" / "phase2" / "rules_baseline.txt").read_text(encoding="utf-8")
saved = [int(x) for x in re.findall(r"rule-set coverage: (\\d+) / 963", txt)]
assert saved == [v["covered"] for v in summary.values()], (saved, summary)
assert n_wbc == int(re.search(r"WBC in \\(10, 11\\].*?: (\\d+) rows", txt).group(1))
assert n_mchc == int(re.search(r"MCHC in \\[30, 32\\).*?: (\\d+) rows", txt).group(1))
print("all recomputed numbers match results/phase2/rules_baseline.txt:", saved, n_wbc, n_mchc)'''),
    md("""## 8. Questions for you (answer without looking)

1. What does forward chaining mean, and what stops it from running forever?
2. Which cited cut-off causes most of the rule-set misses, and how many rows does it cost?
3. Why do the three indices appear in the trace but never change the label set?"""),
]

PHASES = {1: PHASE1, 2: PHASE2}


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
    out = NB_DIR / f"phase{phase}_walkthrough.ipynb"
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
