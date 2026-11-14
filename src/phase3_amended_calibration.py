"""Phase 3 POST-RESULTS AMENDMENT (14 Nov 2026): sigmoid calibration on predicted probabilities for every model.

Research prototype, not a diagnostic tool. Development rows only. Test rows are never read.

Why: the calibration diagnostic showed that scikit-learn calibrates models that have a
decision_function (logistic regression, gradient boosting) on their raw scores, one class at a
time, which destroyed logistic regression's predictions. The amendment calibrates every model on
its predicted probabilities instead (ml_pipelines.ProbaOnly).

Everything else is unchanged: same outer folds and seeds, same inner calibration folds, and the
hyperparameters each outer fold chose in the original run (best_params.csv), so only the
calibration input differs. Uncalibrated probabilities are recomputed and compared with the stored
ones as a check that the refit reproduces the original models.

Outputs: results/phase3/amended/oof_ml_only.csv.gz, oof_h3.csv.gz, amendment_checks.txt
Run:     python src/phase3_amended_calibration.py
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from data import CLASS_ORDER, F_ALL, F_CORE, ROOT, SEED, TARGET, load_dev
from features import INDEX_COLS, add_indices
from ml_pipelines import ProbaOnly, configs, make_pipeline

P3 = ROOT / "results" / "phase3"
OUT = P3 / "amended"
FSETS = {"F-core": F_CORE, "F-all": F_ALL, "F-core+idx": F_CORE + INDEX_COLS, "F-all+idx": F_ALL + INDEX_COLS}


def full(proba, classes, n=9):
    P = np.zeros((proba.shape[0], n))
    P[:, classes] = proba
    return P


def task(X, y, tr, te, rep, fold, model, imb, fset, params):
    warnings.filterwarnings("ignore")
    base = make_pipeline(model, imb).set_params(**params)
    fitted = clone(base).fit(X[tr], y[tr])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 100 * rep + fold)
    cal = CalibratedClassifierCV(ProbaOnly(clone(base)), method="sigmoid", cv=inner, ensemble=False).fit(X[tr], y[tr])
    return {"rep": rep, "fold": fold, "model": model, "imb": imb, "fset": fset, "te": te,
            "p_un": full(fitted.predict_proba(X[te]), fitted.classes_),
            "p_cal": full(cal.predict_proba(X[te]), cal.classes_)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df = add_indices(load_dev(clean=True))
    y = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    rows = df.index.to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(np.zeros(len(y)), y))
    bp = pd.concat([pd.read_csv(P3 / "ml_only" / "best_params.csv"), pd.read_csv(P3 / "h3_indices" / "best_params.csv")])
    tasks = []
    for fset, cols in FSETS.items():
        X = df[cols].to_numpy(float)
        for model, imb in configs():
            for k, (tr, te) in enumerate(splits):
                rep, fold = k // 5, k % 5
                row = bp[(bp.model == model) & (bp.imb == imb) & (bp.fset == fset) & (bp.rep == rep) & (bp.fold == fold)]
                tasks.append((X, y, tr, te, rep, fold, model, imb, fset, json.loads(row["params"].iloc[0])))
    res = Parallel(n_jobs=-1)(delayed(task)(*t) for t in tasks)
    blocks = []
    for r in res:
        b = pd.DataFrame({"raw_row": rows[r["te"]], "y": y[r["te"]], "rep": r["rep"], "fold": r["fold"],
                          "model": r["model"], "imb": r["imb"], "fset": r["fset"]})
        b = pd.concat([b, pd.DataFrame(r["p_un"], columns=[f"p_un_{j}" for j in range(9)]),
                       pd.DataFrame(r["p_cal"], columns=[f"p_cal_{j}" for j in range(9)])], axis=1)
        blocks.append(b)
    new = pd.concat(blocks, ignore_index=True)
    key = ["fset", "model", "imb", "rep", "raw_row"]
    lines = [f"amended calibration run: {len(tasks)} tasks, {time.perf_counter() - t0:.0f} s"]
    for name, orig_file, fsets in [("ml_only", P3 / "ml_only" / "oof_predictions.csv.gz", ["F-core", "F-all"]),
                                   ("h3", P3 / "h3_indices" / "oof_predictions.csv.gz", ["F-core+idx", "F-all+idx"])]:
        part = new[new.fset.isin(fsets)].sort_values(key).reset_index(drop=True)
        part.to_csv(OUT / f"oof_{name}.csv.gz", index=False, float_format="%.6g")
        orig = pd.read_csv(orig_file).sort_values(key).reset_index(drop=True)
        un = [f"p_un_{j}" for j in range(9)]
        ca = [f"p_cal_{j}" for j in range(9)]
        d_un = np.abs(part[un].to_numpy() - orig[un].to_numpy()).max()
        same_rows = (part[key].to_numpy() == orig[key].to_numpy()).all()
        lines.append(f"{name}: rows aligned with the original run: {same_rows}; max |uncalibrated prob new - original| = {d_un:.2e}")
        for (m, i), g in part.groupby(["model", "imb"]):
            o = orig[(orig.model == m) & (orig.imb == i)]
            d = np.abs(g[ca].to_numpy() - o[ca].to_numpy()).max()
            same_pred = (g[ca].to_numpy().argmax(1) == o[ca].to_numpy().argmax(1)).mean()
            lines.append(f"  {m:7s} {i:8s} max |calibrated prob amended - original| = {d:.2e}, "
                         f"same predicted class in {100 * same_pred:.1f}% of rows")
    (OUT / "amendment_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
