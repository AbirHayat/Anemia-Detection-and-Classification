"""Phase 3: H3 (index features) and H4 (shortcut probes) on the same outer folds as the ML-only run.

Research prototype, not a diagnostic tool. Development rows only. Test rows are never read.
Prints progress and timing only, never a metric (results are opened in the Phase 3 evaluation).

H3: dummy + {logreg, tree, forest, hgb} x {class weights, SMOTE} on F-core+idx and F-all+idx
    (the three discrimination indices added, src/features.py), via the same task function as
    the ML-only run (src/phase3_ml_only_cv.py).
H4 probe 1: FillIndicator (learned on the training fold) -> depth-3 tree (balanced class weights),
    target = diagnosis. Baseline: majority-class dummy on the same folds.
H4 probe 2: the seven F-core columns -> depth-3 tree (balanced), target = mean-filled or not
    (Phase 1 flag). Baseline: majority-class dummy.

Outputs: results/phase3/h3_indices/oof_predictions.csv.gz, best_params.csv
         results/phase3/h4_probes/oof_predictions.csv.gz
         results/phase3/h3_h4_run_log.txt
Run:     python src/phase3_h3_h4_cv.py
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from data import CLASS_ORDER, F_ALL, F_CORE, ROOT, SEED, TARGET, load_dev
from features import INDEX_COLS, FillIndicator, add_indices
from ml_pipelines import configs
from phase3_ml_only_cv import run_task

OUT = ROOT / "results" / "phase3"
PROBE_DEPTH = 3


def probe_task(X, y, tr, te, rep, fold, probe, kind):
    warnings.filterwarnings("ignore")
    if kind == "dummy":
        est = DummyClassifier(strategy="most_frequent")
    elif probe == "probe1_fill_indicator":
        est = make_pipeline(FillIndicator(), DecisionTreeClassifier(max_depth=PROBE_DEPTH, class_weight="balanced",
                                                                    random_state=SEED))
    else:
        est = DecisionTreeClassifier(max_depth=PROBE_DEPTH, class_weight="balanced", random_state=SEED)
    est.fit(X[tr], y[tr])
    return pd.DataFrame({"row": te, "y": y[te], "pred": est.predict(X[te]), "rep": rep, "fold": fold,
                         "probe": probe, "kind": kind})


def main() -> None:
    (OUT / "h3_indices").mkdir(parents=True, exist_ok=True)
    (OUT / "h4_probes").mkdir(parents=True, exist_ok=True)
    log = open(OUT / "h3_h4_run_log.txt", "w", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    df = add_indices(load_dev(clean=True))
    y = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    rows = df.index.to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(np.zeros(len(y)), y))
    say(f"rows {len(y)}, outer folds {len(splits)} (same seed and stratification as the ML-only run)")

    # ---- H3 ----
    fsets = {"F-core+idx": F_CORE + INDEX_COLS, "F-all+idx": F_ALL + INDEX_COLS}
    tasks = [(df[cols].to_numpy(float), y, tr, te, k // 5, k % 5, m, i, fs)
             for fs, cols in fsets.items() for m, i in configs() for k, (tr, te) in enumerate(splits)]
    say(f"H3 tasks: {len(tasks)}")
    t0 = time.perf_counter()
    recs, params = [], []
    for n, r in enumerate(Parallel(n_jobs=-1, return_as="generator_unordered")(delayed(run_task)(*t) for t in tasks), 1):
        block = pd.DataFrame({"raw_row": rows[r["te"]], "y": y[r["te"]], "rep": r["rep"], "fold": r["fold"],
                              "model": r["model"], "imb": r["imb"], "fset": r["fset"]})
        for key, cls, tag in [("p_un", "classes_un", "un"), ("p_ca", "classes_ca", "cal")]:
            P = np.zeros((len(r["te"]), len(CLASS_ORDER)))
            P[:, r[cls]] = r[key]
            block = pd.concat([block, pd.DataFrame(P, columns=[f"p_{tag}_{j}" for j in range(len(CLASS_ORDER))])], axis=1)
        recs.append(block)
        params.append({"rep": r["rep"], "fold": r["fold"], "model": r["model"], "imb": r["imb"], "fset": r["fset"],
                       "params": json.dumps(r["params"]), "seconds": round(r["seconds"], 2)})
        if n % 50 == 0 or n == len(tasks):
            say(f"H3 done {n}/{len(tasks)}, elapsed {time.perf_counter() - t0:.0f} s")
    pd.concat(recs, ignore_index=True).sort_values(["fset", "model", "imb", "rep", "raw_row"]).to_csv(
        OUT / "h3_indices" / "oof_predictions.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(params).to_csv(OUT / "h3_indices" / "best_params.csv", index=False)

    # ---- H4 ----
    filled = (df["fill_count"] == 6).astype(int).to_numpy()
    X_all, X_core = df[F_ALL].to_numpy(float), df[F_CORE].to_numpy(float)
    ptasks = []
    for k, (tr, te) in enumerate(splits):
        for kind in ["probe", "dummy"]:
            ptasks.append((X_all, y, tr, te, k // 5, k % 5, "probe1_fill_indicator", kind))
            ptasks.append((X_core, filled, tr, te, k // 5, k % 5, "probe2_fcore_to_filled", kind))
    res = Parallel(n_jobs=-1)(delayed(probe_task)(*t) for t in ptasks)
    out = pd.concat(res, ignore_index=True)
    out["raw_row"] = rows[out["row"].to_numpy()]
    out.drop(columns="row").to_csv(OUT / "h4_probes" / "oof_predictions.csv.gz", index=False)
    say(f"H4 probe tasks: {len(ptasks)} done. Probe tree depth {PROBE_DEPTH}, balanced class weights")
    say("no metric computed or printed here")


if __name__ == "__main__":
    main()
