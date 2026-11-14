"""Model pipelines and hyperparameter grids for Phase 3 (ML only).

Research prototype, not a diagnostic tool.

Every pipeline is fitted on training rows only (scikit-learn / imbalanced-learn Pipeline),
so scaling and SMOTE never see validation or test rows.

Imbalance arms (project plan):
  "weights": class_weight="balanced" in the classifier, no resampling
  "smote":   SMOTE inside the pipeline, k_neighbors = min(5, smallest class count - 1)
             computed on the rows SMOTE actually sees (so it works in tiny inner folds)

Grids were chosen on 14 Nov 2026 and logged before any result. They are small on
purpose: 963 rows, and a 5 x 5 outer CV with a 3-fold inner CV.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data import SEED

MODELS = ["dummy", "logreg", "tree", "forest", "hgb"]
IMBALANCE = ["weights", "smote"]


def _adaptive_smote(X, y, seed: int = SEED, k_max: int = 5):
    """SMOTE with k_neighbors set below the smallest class count of THIS fit."""
    n_min = int(np.bincount(np.unique(y, return_inverse=True)[1]).min())
    if n_min < 2:
        return X, y
    k = max(1, min(k_max, n_min - 1))
    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)


def _classifier(model: str, balanced: bool):
    cw = "balanced" if balanced else None
    if model == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if model == "logreg":
        return LogisticRegression(max_iter=5000, class_weight=cw, random_state=SEED)
    if model == "tree":
        return DecisionTreeClassifier(class_weight=cw, random_state=SEED)
    if model == "forest":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=SEED, n_jobs=1)
    if model == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, class_weight=cw, early_stopping=False,
                                              random_state=SEED)
    raise ValueError(model)


GRIDS = {
    "dummy": {},
    "logreg": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "tree": {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [1, 5]},
    "forest": {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 3]},
    "hgb": {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None]},
}


def make_pipeline(model: str, imbalance: str) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if model != "dummy" and imbalance == "smote":
        steps.append(("smote", FunctionSampler(func=_adaptive_smote, validate=False)))
    steps.append(("clf", _classifier(model, balanced=(model != "dummy" and imbalance == "weights"))))
    return Pipeline(steps)


def configs() -> list[tuple[str, str]]:
    """(model, imbalance) pairs. The dummy has no imbalance arm."""
    out = [("dummy", "none")]
    out += [(m, i) for m in MODELS if m != "dummy" for i in IMBALANCE]
    return out


class ProbaOnly(ClassifierMixin, BaseEstimator):
    """Wraps an estimator and exposes predict_proba only, so CalibratedClassifierCV calibrates
    predicted probabilities instead of raw decision scores (post-results amendment, 14 Nov 2026)."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
