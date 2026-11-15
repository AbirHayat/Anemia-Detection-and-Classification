"""Phase 4, inner predictions for choosing tau and the conformal threshold without touching outer test rows.

Research prototype, not a diagnostic tool. Development rows only. Test rows are never read.

For each of the 25 outer folds of Phase 3 (RepeatedStratifiedKFold 5 x 5, seed 42), the OUTER TRAINING
rows are split again with StratifiedKFold(5). In every inner fold the full Phase 3 procedure is rerun on
the inner training rows only, for the H2 model (random forest + SMOTE, F-core, logged 14 Nov 2026):
  grid search (3-fold, macro-F1, same grid) -> sigmoid calibration on predicted probabilities
  (ProbaOnly, 3-fold, ensemble=False, the amended Phase 3 calibration)
and the inner validation rows get calibrated class probabilities. Every outer training row therefore has
one prediction from a model that never saw it, and the outer test rows are never used.

This script prints and logs progress and timing ONLY, never a metric.

Output: results/phase4/inner/inner_oof.csv.gz, inner_params.csv, run_log.txt
Run:    python src/phase4_inner_cv.py
"""
from __future__ import annotations

import json
import platform
import time
import warnings

import imblearn
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from data import CLASS_ORDER, F_CORE, ROOT, SEED, TARGET, load_dev
from ml_pipelines import GRIDS, ProbaOnly, make_pipeline

OUT = ROOT / "results" / "phase4" / "inner"
MODEL, IMB, FSET = "forest", "smote", "F-core"


def inner_task(X, y, tr_outer, rep, fold, k, itr, iva):
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    tr, va = tr_outer[itr], tr_outer[iva]
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 2000 + 100 * rep + 10 * fold + k)
    gs = GridSearchCV(make_pipeline(MODEL, IMB), GRIDS[MODEL], scoring="f1_macro", cv=cv3, n_jobs=1, refit=False)
    gs.fit(X[tr], y[tr])
    base = make_pipeline(MODEL, IMB).set_params(**gs.best_params_)
    cal = CalibratedClassifierCV(ProbaOnly(base), method="sigmoid", cv=cv3, ensemble=False).fit(X[tr], y[tr])
    P = np.zeros((len(va), len(CLASS_ORDER)))
    P[:, cal.classes_] = cal.predict_proba(X[va])
    return {"rep": rep, "fold": fold, "inner": k, "va": va, "P": P, "params": gs.best_params_,
            "seconds": time.perf_counter() - t0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "run_log.txt", "w", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    df = load_dev(clean=True)
    y = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    X = df[F_CORE].to_numpy(float)
    rows = df.index.to_numpy()
    outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(np.zeros(len(y)), y))
    tasks = []
    for j, (tr, _te) in enumerate(outer):
        rep, fold = j // 5, j % 5
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1000 + 100 * rep + fold)
        for k, (itr, iva) in enumerate(inner.split(np.zeros(len(tr)), y[tr])):
            tasks.append((X, y, tr, rep, fold, k, itr, iva))
    say(f"python {platform.python_version()}, sklearn {sklearn.__version__}, imblearn {imblearn.__version__}")
    say(f"model {MODEL} + {IMB} on {FSET}, rows {len(y)}, outer folds {len(outer)}, inner tasks {len(tasks)}")
    t0 = time.perf_counter()
    blocks, params = [], []
    res = Parallel(n_jobs=-1, return_as="generator_unordered")(delayed(inner_task)(*t) for t in tasks)
    for n, r in enumerate(res, 1):
        b = pd.DataFrame({"rep": r["rep"], "fold": r["fold"], "inner": r["inner"],
                          "raw_row": rows[r["va"]], "y": y[r["va"]]})
        blocks.append(pd.concat([b, pd.DataFrame(r["P"], columns=[f"p_cal_{j}" for j in range(9)])], axis=1))
        params.append({"rep": r["rep"], "fold": r["fold"], "inner": r["inner"],
                       "params": json.dumps(r["params"]), "seconds": round(r["seconds"], 2)})
        if n % 25 == 0 or n == len(tasks):
            say(f"done {n}/{len(tasks)} tasks, elapsed {time.perf_counter() - t0:.0f} s")
    oof = pd.concat(blocks, ignore_index=True).sort_values(["rep", "fold", "raw_row"])
    oof.to_csv(OUT / "inner_oof.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(params).sort_values(["rep", "fold", "inner"]).to_csv(OUT / "inner_params.csv", index=False)
    say(f"saved {len(oof)} inner prediction rows (each outer training row once per outer fold)")
    say("no metric computed or printed here (opened in Phase 4 after the pre-registration is confirmed)")


if __name__ == "__main__":
    main()
