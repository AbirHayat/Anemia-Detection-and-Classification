"""Paired statistical tests for comparing models on repeated k-fold CV, plus Holm adjustment.

Research prototype, not a diagnostic tool.

Corrected repeated k-fold cv test (Bouckaert and Frank 2004, Section 3.3, read in the open full
text at Waikato Research Commons, hdl 10289/1451), built on the corrected resampled t-test of
Nadeau and Bengio (2003):

    x_ij = score_A - score_B on fold i of repeat j,   m = mean of the k*r differences
    s2   = sum (x_ij - m)^2 / (k*r - 1)
    t    = m / sqrt( (1/(k*r) + n2/n1) * s2 ),   Student t with k*r - 1 degrees of freedom

n1 = training rows per fold, n2 = test rows per fold (n2/n1 = 1/4 for 5-fold CV).
The correction widens the variance because the folds share most of their training rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CorrectedT:
    mean: float
    t: float
    df: int
    p: float
    se: float


def corrected_repeated_kfold_t(diffs, k: int, r: int, n2_over_n1: float,
                               alternative: str = "two-sided") -> CorrectedT:
    """alternative: "two-sided", "greater" (A better than B) or "less"."""
    x = np.asarray(diffs, dtype=float)
    if x.size != k * r:
        raise ValueError(f"expected {k * r} differences, got {x.size}")
    m = x.mean()
    s2 = x.var(ddof=1)
    se = np.sqrt((1.0 / (k * r) + n2_over_n1) * s2)
    df = k * r - 1
    if se == 0:
        t = 0.0 if m == 0 else np.sign(m) * np.inf
    else:
        t = m / se
    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(t), df)
    elif alternative == "greater":
        p = stats.t.sf(t, df)
    elif alternative == "less":
        p = stats.t.cdf(t, df)
    else:
        raise ValueError(alternative)
    return CorrectedT(float(m), float(t), df, float(p), float(se))


def corrected_ci(diffs, k: int, r: int, n2_over_n1: float, level: float = 0.90) -> tuple[float, float]:
    """Two-sided CI for the mean difference with the corrected standard error."""
    res = corrected_repeated_kfold_t(diffs, k, r, n2_over_n1)
    q = stats.t.ppf(0.5 + level / 2, res.df)
    return res.mean - q * res.se, res.mean + q * res.se


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[i]))
        adj[i] = running
    return adj


@dataclass
class McNemar:
    b: int          # A right, B wrong
    c: int          # A wrong, B right
    chi2: float
    p: float        # continuity-corrected chi-square, two-sided (Dietterich 1998)
    p_exact: float  # exact binomial, two-sided


def mcnemar(correct_a, correct_b) -> McNemar:
    """McNemar's test on paired right/wrong outcomes of two classifiers on the same cases.

    chi2 = (|b - c| - 1)^2 / (b + c) with 1 degree of freedom (continuity correction, as recommended by
    Dietterich 1998). If b + c = 0 the two classifiers never disagree and p = 1.
    """
    a = np.asarray(correct_a, bool)
    bb = np.asarray(correct_b, bool)
    if a.shape != bb.shape:
        raise ValueError("both classifiers must be scored on the same cases")
    b, c = int((a & ~bb).sum()), int((~a & bb).sum())
    if b + c == 0:
        return McNemar(b, c, 0.0, 1.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, 1))
    p_exact = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return McNemar(b, c, float(chi2), p, p_exact)
