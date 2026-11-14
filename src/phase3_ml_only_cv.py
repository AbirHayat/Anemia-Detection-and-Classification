"""Phase 3, ML-only part: nested repeated stratified CV on the 963 clean development rows.

Research prototype, not a diagnostic tool. Test rows are never read.

Started in the background during Phase 2 under the student's condition (14 Nov 2026): every
rule-engine threshold was cited and logged BEFORE this run started.
This script prints and logs progress and timing ONLY, never a metric. Its outputs are opened
in Phase 3.

Design (project plan)
  outer: RepeatedStratifiedKFold(5 splits x 5 repeats, seed 42) -> 25 outer folds
  inner: StratifiedKFold(3) GridSearchCV on macro-F1, refit on the outer training rows
  calibration: CalibratedClassifierCV(best pipeline, method="sigmoid", cv=3, ensemble=False),
               fitted on the outer training rows only
  configs: dummy + {logreg, tree, forest, hgb} x {class weights, SMOTE}, on F-core and F-all
  saved per outer test row: true label, uncalibrated and calibrated class probabilities

Outputs: results/phase3/ml_only/oof_predictions.csv.gz, best_params.csv, run_log.txt
Run:     python src/phase3_ml_only_cv.py [--quick]   (--quick = 1 repeat, for timing only)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from data import CLASS_ORDER, F_ALL, F_CORE, ROOT, SEED, TARGET, load_dev
from ml_pipelines import GRIDS, configs, make_pipeline

OUT = ROOT / "results" / "phase3" / "ml_only"
FSETS = {"F-core": F_CORE, "F-all": F_ALL}


def run_task(X, y, tr, te, rep, fold, model, imb, fset):
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    pipe = make_pipeline(model, imb)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 100 * rep + fold)
    if GRIDS[model]:
        gs = GridSearchCV(pipe, GRIDS[model], scoring="f1_macro", cv=inner, n_jobs=1, refit=True)
        gs.fit(X[tr], y[tr])
        best, params = gs.best_estimator_, gs.best_params_
    else:
        best, params = clone(pipe).fit(X[tr], y[tr]), {}
    cal = CalibratedClassifierCV(clone(best), method="sigmoid", cv=inner, ensemble=False)
    cal.fit(X[tr], y[tr])
    p_un = best.predict_proba(X[te])
    p_ca = cal.predict_proba(X[te])
    return {"rep": rep, "fold": fold, "model": model, "imb": imb, "fset": fset, "te": te,
            "p_un": p_un, "p_ca": p_ca, "classes_un": best.classes_, "classes_ca": cal.classes_,
            "params": params, "seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / ("run_log_quick.txt" if a.quick else "run_log.txt"), "w", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    df = load_dev(clean=True)
    y = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    rows = df.index.to_numpy()
    n_rep = 1 if a.quick else 5
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
    splits = list(outer.split(np.zeros(len(y)), y))
    say(f"python {platform.python_version()}, sklearn {sklearn.__version__}, imblearn {imblearn.__version__}, "
        f"joblib {joblib.__version__}")
    say(f"rows {len(y)}, outer folds {len(splits)}, configs {len(configs())} x {len(FSETS)} feature sets")
    tasks = []
    for fset, cols in FSETS.items():
        X = df[cols].to_numpy(dtype=float)
        for model, imb in configs():
            for k, (tr, te) in enumerate(splits):
                tasks.append((X, y, tr, te, k // 5, k % 5, model, imb, fset))
    say(f"tasks: {len(tasks)}")
    t0 = time.perf_counter()
    res = Parallel(n_jobs=-1, verbose=0, return_as="generator_unordered")(delayed(run_task)(*t) for t in tasks)
    recs, params = [], []
    for n, r in enumerate(res, 1):
        for prob_key, cls_key, tag in [("p_un", "classes_un", "un"), ("p_ca", "classes_ca", "cal")]:
            P = np.zeros((len(r["te"]), len(CLASS_ORDER)))
            P[:, r[cls_key]] = r[prob_key]
            frame = pd.DataFrame(P, columns=[f"p_{tag}_{i}" for i in range(len(CLASS_ORDER))])
            if tag == "un":
                base = pd.DataFrame({"raw_row": rows[r["te"]], "y": y[r["te"]], "rep": r["rep"], "fold": r["fold"],
                                     "model": r["model"], "imb": r["imb"], "fset": r["fset"]})
                block = pd.concat([base, frame], axis=1)
            else:
                block = pd.concat([block, frame], axis=1)
        recs.append(block)
        params.append({"rep": r["rep"], "fold": r["fold"], "model": r["model"], "imb": r["imb"],
                       "fset": r["fset"], "params": json.dumps(r["params"]), "seconds": round(r["seconds"], 2)})
        if n % 25 == 0 or n == len(tasks):
            say(f"done {n}/{len(tasks)} tasks, elapsed {time.perf_counter() - t0:.0f} s")
    if a.quick:
        say("quick timing run: predictions discarded, nothing saved")
        return
    oof = pd.concat(recs, ignore_index=True).sort_values(["fset", "model", "imb", "rep", "raw_row"])
    suffix = ""
    oof.to_csv(OUT / f"oof_predictions{suffix}.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(params).to_csv(OUT / f"best_params{suffix}.csv", index=False)
    say(f"saved {len(oof)} prediction rows. Class index order: {CLASS_ORDER}")
    say("no metric computed or printed here (opened in Phase 3)")


if __name__ == "__main__":
    main()
