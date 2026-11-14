"""Phase 3 SECONDARY analysis: why does calibration lower logistic-regression macro-F1 so much?

Research prototype, not a diagnostic tool. Development rows only. Requested by the student on
14 Nov 2026 AFTER the Phase 3 results were seen. It does not change the pre-registered primary
(calibrated predictions, sigmoid, 3 inner folds).

Two candidate explanations
  (a) prior restoration: calibration maps the class-weighted scores back to the real, imbalanced
      class frequencies, so rare classes rarely win the argmax.
      Test: divide the calibrated probabilities by the training-fold class shares and renormalise
      ("prior-corrected" argmax). If this recovers most of the uncalibrated score, (a) explains the drop.
  (b) unstable calibrators: a sigmoid fitted for a 7-row class is noisy.
      Tests: refit with 2 and 5 inner calibration folds and with isotonic calibration, and record the
      spread of the fitted sigmoid slope per class across the 25 outer folds.
  (c) added after (a) and (b) were checked: scikit-learn calibrates logistic regression on its raw
      decision scores, one class at a time. Multinomial scores are only meaningful relative to each
      other within a row, so per-class sigmoids may throw that information away.
      Test: the same sigmoid calibration fed with the model's probabilities instead (ProbaOnly wrapper).

Hyperparameters are the ones the original run chose in each outer fold (best_params.csv), so only
the calibration step changes. Configurations: logistic regression with class weights (the chosen arm)
on F-core and F-all, and the H2 model (random forest with SMOTE, F-core) as a reference.

Output: results/phase3/calibration_diagnostic.txt
Run:    python src/phase3_calibration_diagnostic.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from data import CLASS_ORDER, F_ALL, F_CORE, ROOT, SEED, TARGET, load_dev
from metrics import per_class, scores
from ml_pipelines import ProbaOnly, make_pipeline

OUT = ROOT / "results" / "phase3"
CONFIGS = [("logreg", "weights", "F-core"), ("logreg", "weights", "F-all"), ("forest", "smote", "F-core")]
VARIANTS = ["uncalibrated", "sigmoid_cv3 (primary)", "sigmoid_cv3 + prior correction", "sigmoid_cv2",
            "sigmoid_cv5", "isotonic_cv3", "sigmoid_cv3 on probabilities"]



def full_proba(est, X, n=9):
    P = np.zeros((len(X), n))
    P[:, est.classes_] = est.predict_proba(X)
    return P


def task(X, y, tr, te, rep, fold, model, imb, fset, params):
    warnings.filterwarnings("ignore")
    base = make_pipeline(model, imb).set_params(**params)
    out, slopes = {}, {}
    fitted = clone(base).fit(X[tr], y[tr])
    out["uncalibrated"] = full_proba(fitted, X[te])
    for name, method, k in [("sigmoid_cv3 (primary)", "sigmoid", 3), ("sigmoid_cv2", "sigmoid", 2),
                            ("sigmoid_cv5", "sigmoid", 5), ("isotonic_cv3", "isotonic", 3)]:
        inner = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED + 100 * rep + fold)
        cal = CalibratedClassifierCV(clone(base), method=method, cv=inner, ensemble=False).fit(X[tr], y[tr])
        out[name] = full_proba(cal, X[te])
        if name.startswith("sigmoid_cv3"):
            cc = cal.calibrated_classifiers_[0]
            slopes = {int(c): float(cb.a_) for c, cb in zip(cal.classes_, cc.calibrators)}
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 100 * rep + fold)
    cal = CalibratedClassifierCV(ProbaOnly(clone(base)), method="sigmoid", cv=inner, ensemble=False).fit(X[tr], y[tr])
    out["sigmoid_cv3 on probabilities"] = full_proba(cal, X[te])
    prior = np.bincount(y[tr], minlength=9) / len(tr)
    pc = out["sigmoid_cv3 (primary)"] / np.where(prior > 0, prior, 1)
    out["sigmoid_cv3 + prior correction"] = pc / pc.sum(axis=1, keepdims=True)
    return {"rep": rep, "fold": fold, "cfg": (model, imb, fset), "te": te,
            "pred": {k: v.argmax(axis=1) for k, v in out.items()}, "slopes": slopes}


def main() -> None:
    df = load_dev(clean=True)
    y = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(np.zeros(len(y)), y))
    bp = pd.read_csv(OUT / "ml_only" / "best_params.csv")
    tasks = []
    for model, imb, fset in CONFIGS:
        X = df[F_CORE if fset == "F-core" else F_ALL].to_numpy(float)
        for k, (tr, te) in enumerate(splits):
            rep, fold = k // 5, k % 5
            row = bp[(bp.model == model) & (bp.imb == imb) & (bp.fset == fset) & (bp.rep == rep) & (bp.fold == fold)]
            tasks.append((X, y, tr, te, rep, fold, model, imb, fset, json.loads(row["params"].iloc[0])))
    res = Parallel(n_jobs=-1)(delayed(task)(*t) for t in tasks)

    L = ["Phase 3 SECONDARY analysis: calibration diagnostic (requested after the Phase 3 results were seen).",
         "Does not change the pre-registered primary. Development rows only. Mean ± std over 25 outer folds.", ""]
    for cfg in CONFIGS:
        rs = [r for r in res if r["cfg"] == cfg]
        L.append(f"== {cfg[0]} ({cfg[1]}), {cfg[2]} ==")
        L.append(f"  {'variant':32s} {'macro-F1':>15s} {'bal. acc':>15s}  recall: other microcytic / leuk.+thromb. / thrombocytopenia")
        for v in VARIANTS:
            f1 = [scores(y[r["te"]], r["pred"][v])["macro_f1"] for r in rs]
            ba = [scores(y[r["te"]], r["pred"][v])["balanced_accuracy"] for r in rs]
            yt = np.concatenate([y[r["te"]] for r in rs])
            yp = np.concatenate([r["pred"][v] for r in rs])
            rc = per_class(yt, yp)["recall"]
            L.append(f"  {v:32s} {np.mean(f1):.3f} ± {np.std(f1, ddof=1):.3f} {np.mean(ba):.3f} ± {np.std(ba, ddof=1):.3f}"
                     f"  {rc[5]:.2f} / {rc[8]:.2f} / {rc[4]:.2f}")
        sl = pd.DataFrame([r["slopes"] for r in rs])
        L.append("  fitted sigmoid slope a (sigmoid_cv3) across the 25 folds, per class: mean ± std "
                 "(training rows per class in brackets, typical fold)")
        counts = np.bincount(y, minlength=9) * 4 // 5
        for c in range(9):
            if c in sl:
                L.append(f"    {CLASS_ORDER[c]:32s} ({counts[c]:3d})  {sl[c].mean():8.3f} ± {sl[c].std(ddof=1):.3f}")
        L.append("")
    (OUT / "calibration_diagnostic.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
