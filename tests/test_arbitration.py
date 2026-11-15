"""Unit tests for src/arbitration.py (Phase 4). Synthetic inputs only, no project data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arbitration as A  # noqa: E402
from data import CLASS_ORDER  # noqa: E402


def onehot_sets(*labels_per_row):
    return np.array([[i in row for i in range(9)] for row in labels_per_row])


def probs(*rows):
    P = np.zeros((len(rows), 9))
    for r, d in enumerate(rows):
        for k, v in d.items():
            P[r, k] = v
    return P


def test_referral_indices():
    assert {CLASS_ORDER[i] for i in A.REF_IDX} == {"Thrombocytopenia", "Leukemia", "Leukemia with thrombocytopenia"}
    assert len(A.NONREF_IDX) == 6


def test_set_mask_parses_phase2_strings():
    M = A.set_mask(["Healthy", "Iron deficiency anemia | Other microcytic anemia", ""])
    assert M[0, 0] and M[0].sum() == 1
    assert M[1, 3] and M[1, 5] and M[1].sum() == 2
    assert M[2].sum() == 0


def test_hybrid_picks_best_inside_set_and_falls_back_when_empty():
    P = probs({0: 0.6, 3: 0.3, 5: 0.1}, {0: 0.6, 3: 0.3, 5: 0.1})
    M = onehot_sets([3, 5], [])
    assert list(A.hybrid_no_abstention(P, M)) == [3, 0]


def test_refer_flags():
    P = probs({4: 0.9, 0: 0.1}, {0: 0.9, 6: 0.1})
    rule = np.array([False, True])
    assert list(A.refer_flags("ml", P, rule)) == [True, False]
    assert list(A.refer_flags("rule", P, rule)) == [False, True]
    assert list(A.refer_flags("either", P, rule)) == [True, True]


def test_decide_reasons():
    P = probs({0: 0.9}, {0: 0.5, 3: 0.5 - 1e-9}, {3: 0.9}, {0: 0.95})
    M = onehot_sets([0], [0], [0], [0])
    refer = np.array([False, False, False, True])
    acc, top, reason = A.decide(P, M, refer, tau=0.6)
    assert list(acc) == [True, False, False, False]
    assert list(reason) == ["ACCEPT", "LOW_CONF", "DISAGREE", "REFER"]
    assert top[2] == 3


def test_tau_equal_to_probability_is_accepted():
    P = probs({0: 0.7})
    acc, _, _ = A.decide(P, onehot_sets([0]), np.array([False]), tau=0.7)
    assert acc[0]


def test_selective_f1_counts_referral_case_as_error_and_empty_class_as_zero():
    y = np.array([0, 0, 4])           # last case is a referral class that slipped into ACCEPT
    pred = np.array([0, 0, 0])
    acc = np.array([True, True, True])
    got = A.selective_macro_f1(y, pred, acc)
    # class 0: tp 2, fp 1, fn 0 -> F1 0.8. Five other non-referral classes have nothing -> 0.
    assert got == pytest.approx(0.8 / 6)
    assert A.selective_macro_f1(y, pred, np.zeros(3, bool)) == 0.0


def test_select_tau_respects_floor_and_prefers_smaller_tau_on_ties():
    # four accepted-eligible cases, all correct: every tau up to 0.5 gives the same F1
    y = np.array([0, 0, 0, 0])
    P = probs({0: 0.5}, {0: 0.6}, {0: 0.7}, {0: 0.8})
    M = onehot_sets([0], [0], [0], [0])
    tau, met, _ = A.select_tau(y, P, M, np.zeros(4, bool), floor=0.70)
    assert met and tau == 0.0
    # floor impossible -> tau 0 and met False
    tau, met, _ = A.select_tau(y, P, M, np.ones(4, bool), floor=0.70)
    assert not met and tau == 0.0


def test_select_tau_raises_tau_when_it_removes_errors_above_floor():
    y = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 0])
    P = np.zeros((10, 9))
    P[np.arange(10), [0, 1, 1, 1, 1, 2, 2, 2, 2, 1]] = [.9, .9, .9, .9, .9, .9, .9, .9, .9, .4]  # last one wrong, low conf
    M = np.ones((10, 9), bool)
    tau, met, _ = A.select_tau(y, P, M, np.zeros(10, bool), floor=0.85)
    assert met and 0.4 < tau <= 0.9


def test_conformal_quantile_and_sets():
    scores = np.arange(1, 20) / 20.0        # 19 scores 0.05 ... 0.95
    q = A.conformal_quantile(scores, alpha=0.10)   # ceil(20 * 0.9) = 18th smallest = 0.90
    assert q == pytest.approx(0.90)
    assert A.conformal_quantile(scores[:5], alpha=0.05) == float("inf")
    S = A.conformal_sets(probs({0: 0.95, 1: 0.05}, {0: 0.5, 1: 0.5}), q=0.10)
    assert list(S[0].nonzero()[0]) == [0]
    assert S[1].sum() == 0


def test_conformal_decide_needs_singleton_inside_rule_set():
    S = onehot_sets([0], [0, 1], [3], [0])
    M = onehot_sets([0], [0, 1], [0], [0])
    acc, lab = A.conformal_decide(S, M, np.array([False, False, False, True]))
    assert list(acc) == [True, False, False, False]


def test_macro_f1_from_counts_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 200)
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 9, 200))
    labs = A.NONREF_IDX
    tp = [((y == k) & (p == k)).sum() for k in labs]
    fp = [((y != k) & (p == k)).sum() for k in labs]
    fn = [((y == k) & (p != k)).sum() for k in labs]
    assert A.macro_f1_from_counts(tp, fp, fn) == pytest.approx(
        f1_score(y, p, labels=labs, average="macro", zero_division=0))
