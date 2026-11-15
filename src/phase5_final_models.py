"""Phase 5: fit the final models on ALL development rows (the proposal: "for the final model the same
inner procedure runs on the whole development set").

Research prototype, not a diagnostic tool. Development rows only. Test rows are never read.

For each model type with its chosen imbalance arm (amended rule P2: logreg SMOTE, tree class weights,
forest SMOTE, gradient boosting SMOTE), on F-core:
  grid search (StratifiedKFold(3), macro-F1, the Phase 3 grid) -> sigmoid calibration on predicted
  probabilities (ProbaOnly, same 3 folds, ensemble=False, so one underlying estimator is fitted on all rows)
The random forest is the agent's model (H2 model, logged 14 Nov). The other three exist so that the
explanation layer can show one explainer per model type, as the proposal says.

This script prints timing and the chosen settings only, never a metric.

Output: results/phase5/models/<model>_<arm>_Fcore.joblib, results/phase5/models/manifest.json
Run:    python src/phase5_final_models.py
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings

import imblearn
import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import CLASS_ORDER, F_CORE, ROOT, SEED, TARGET, load_dev
from ml_pipelines import GRIDS, ProbaOnly, make_pipeline

OUT = ROOT / "results" / "phase5" / "models"
ARMS = {"logreg": "smote", "tree": "weights", "forest": "smote", "hgb": "smote"}


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dev(clean=True)
    y = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    X = df[F_CORE].to_numpy(float)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    manifest = {"fitted_on": f"{len(y)} clean development rows", "features": F_CORE, "class_order": CLASS_ORDER,
                "seed": SEED, "sklearn": sklearn.__version__, "imblearn": imblearn.__version__,
                "numpy": np.__version__, "models": {}}
    for model, arm in ARMS.items():
        t0 = time.perf_counter()
        gs = GridSearchCV(make_pipeline(model, arm), GRIDS[model], scoring="f1_macro", cv=cv, n_jobs=-1, refit=False)
        gs.fit(X, y)
        base = make_pipeline(model, arm).set_params(**gs.best_params_)
        cal = CalibratedClassifierCV(ProbaOnly(base), method="sigmoid", cv=cv, ensemble=False).fit(X, y)
        path = OUT / f"{model}_{arm}_Fcore.joblib"
        joblib.dump(cal, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["models"][model] = {"arm": arm, "file": path.name, "sha256": sha,
                                     "best_params": {k: (None if v is None else v) for k, v in gs.best_params_.items()},
                                     "role": "agent model (H2 model)" if model == "forest" else "explainer demonstration"}
        print(f"{model:7s} {arm:8s} params {gs.best_params_}  fitted in {time.perf_counter() - t0:.1f} s  sha256 {sha[:12]}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("no metric computed here (the final models are scored once, on the test set, in Phase 6)")


if __name__ == "__main__":
    main()
