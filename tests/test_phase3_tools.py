"""Unit tests for metrics, statistical tests and Phase 3 features (synthetic data only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import F_ALL, FILL_COLS  # noqa: E402
from features import FillIndicator, add_indices  # noqa: E402
from metrics import per_class, scores  # noqa: E402
from stats_tests import corrected_ci, corrected_repeated_kfold_t, holm  # noqa: E402


def test_scores_perfect_and_majority():
    y = np.array([0, 0, 0, 1, 2])
    s = scores(y, y, labels=[0, 1, 2])
    assert s["macro_f1"] == 1 and s["balanced_accuracy"] == 1 and s["mcc"] == 1
    s = scores(y, np.zeros(5, int), labels=[0, 1, 2])
    assert s["balanced_accuracy"] == pytest.approx(1 / 3)
    assert s["mcc"] == 0


def test_per_class_recall():
    pc = per_class(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert pc.loc[0, "recall"] == 0.5 and pc.loc[1, "recall"] == 1 and pc.loc[1, "precision"] == pytest.approx(2 / 3)


def test_corrected_t_by_hand():
    d = np.array([0.02, 0.04, 0.0, 0.06, 0.03] * 5)          # k=5, r=5
    m, s2 = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / 25 + 0.25) * s2)
    res = corrected_repeated_kfold_t(d, 5, 5, 0.25, "greater")
    assert res.t == pytest.approx(m / se) and res.df == 24
    assert res.p == pytest.approx(stats.t.sf(m / se, 24))


def test_correction_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 25)
    corrected = corrected_repeated_kfold_t(d, 5, 5, 0.25, "greater").p
    naive = stats.ttest_1samp(d, 0, alternative="greater").pvalue
    assert corrected > naive


def test_zero_differences():
    res = corrected_repeated_kfold_t(np.zeros(25), 5, 5, 0.25)
    assert res.t == 0 and res.p == 1


def test_ci_contains_mean_and_wrong_length():
    d = np.linspace(-0.01, 0.03, 25)
    lo, hi = corrected_ci(d, 5, 5, 0.25, 0.90)
    assert lo < d.mean() < hi
    with pytest.raises(ValueError):
        corrected_repeated_kfold_t(d[:20], 5, 5, 0.25)


def test_holm_known_example():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_indices_match_rule_engine_formulas():
    df = pd.DataFrame({"MCV": [70.0], "RBC": [5.0], "HGB": [10.0], "MCH": [22.0]})
    out = add_indices(df)
    assert out["mentzer"].iloc[0] == pytest.approx(14.0)
    assert out["england_fraser"].iloc[0] == pytest.approx(11.6)
    assert out["shine_lal"].iloc[0] == pytest.approx(1078.0)


def test_fill_indicator_learns_modes_on_training_rows_only():
    train = pd.DataFrame(np.arange(5 * len(F_ALL), dtype=float).reshape(5, len(F_ALL)), columns=F_ALL)
    for c in FILL_COLS + ["PDW"]:
        train.loc[:2, c] = 7.0                                # value 7 repeated: the training mode
    fi = FillIndicator().fit(train.to_numpy())
    test = train.iloc[[0, 4]].to_numpy()
    out = fi.transform(test)
    assert out[0].tolist() == [6, 1] and out[1].tolist() == [0, 0]


def test_proba_only_hides_raw_scores():
    from sklearn.linear_model import LogisticRegression
    from ml_pipelines import ProbaOnly
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(60, 3)), np.repeat([0, 1, 2], 20)
    w = ProbaOnly(LogisticRegression(max_iter=1000)).fit(X, y)
    assert not hasattr(w, "decision_function")
    assert np.allclose(w.predict_proba(X), LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X))
