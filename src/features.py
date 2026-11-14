"""Derived features and the fill indicator used in Phase 3.

Research prototype, not a diagnostic tool.

add_indices: the three discrimination indices as model inputs for H3 (same formulas as the rule
engine, Hans et al. 2024 Table 1). Computed for every row, not only microcytic ones.

FillIndicator: the H4 fill indicator, learned on TRAINING rows only (inside a pipeline):
  count = how many of the six fill columns equal their training-fold most frequent value (0-6)
  flag  = whether PDW equals its training-fold most frequent value (0 or 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from data import F_ALL, FILL_COLS, FILL_FLAG_COL

INDEX_COLS = ["mentzer", "england_fraser", "shine_lal"]


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mentzer"] = df["MCV"] / df["RBC"]
    out["england_fraser"] = df["MCV"] - df["RBC"] - 5 * df["HGB"] - 3.4
    out["shine_lal"] = df["MCV"] ** 2 * df["MCH"] * 0.01
    return out


class FillIndicator(BaseEstimator, TransformerMixin):
    """Input: an array with the F-all columns in data.F_ALL order. Output: [count, flag]."""

    def __init__(self, columns=tuple(F_ALL)):
        self.columns = columns

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        cols = list(self.columns)
        self.idx_ = [cols.index(c) for c in FILL_COLS]
        self.pdw_ = cols.index(FILL_FLAG_COL)
        self.modes_ = np.array([pd.Series(X[:, i]).mode().iloc[0] for i in self.idx_])
        self.pdw_mode_ = pd.Series(X[:, self.pdw_]).mode().iloc[0]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        count = (X[:, self.idx_] == self.modes_).sum(axis=1)
        flag = (X[:, self.pdw_] == self.pdw_mode_).astype(int)
        return np.column_stack([count, flag])
