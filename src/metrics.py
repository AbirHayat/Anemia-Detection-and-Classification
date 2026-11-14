"""Evaluation metrics for imbalanced multiclass results.

Research prototype, not a diagnostic tool.

Never accuracy alone: macro-F1, balanced accuracy, Matthews correlation coefficient (MCC,
multiclass form), per-class precision and recall and the confusion matrix. Classes are the
integer indices 0-8 of data.CLASS_ORDER.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

N_CLASSES = 9


def scores(y_true, y_pred, labels=None) -> dict:
    """Headline metrics. `labels` restricts macro-F1 to those classes (default: all 9).
    A label with no true and no predicted row gets F1 = 0 only if it is listed in `labels`."""
    labels = list(range(N_CLASSES)) if labels is None else list(labels)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def per_class(y_true, y_pred) -> pd.DataFrame:
    labels = list(range(N_CLASSES))
    return pd.DataFrame({
        "precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "support": np.bincount(np.asarray(y_true), minlength=N_CLASSES),
    }, index=labels)


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))


def per_fold(df: pd.DataFrame, pred_col: str = "pred", extra=None) -> pd.DataFrame:
    """One row per (rep, fold) with the headline metrics. `extra(g)` may add columns."""
    rows = []
    for (rep, fold), g in df.groupby(["rep", "fold"], sort=True):
        r = {"rep": rep, "fold": fold, **scores(g["y"], g[pred_col])}
        if extra is not None:
            r.update(extra(g))
        rows.append(r)
    return pd.DataFrame(rows)
