"""Arbitration and abstention, stage 4 of the agent (Phase 4).

Research prototype, not a diagnostic tool. It does not diagnose anyone.

Inputs per case: calibrated class probabilities P (9 columns in data.CLASS_ORDER order), the
rule-compatible set as a boolean mask M (9 columns) and a referral flag. Four outcomes:
  REJECT   the validator failed (handled before this module, never scored)
  REFER    the chosen referral trigger fired
  ACCEPT   the top ML class is inside the rule-compatible set and its probability is >= tau
  ABSTAIN  anything else (rule-model disagreement or low confidence)

Hybrid without abstention (used for H1): the most probable class inside the rule-compatible set,
or the top ML class when the set is empty.

Selective macro-F1: macro-F1 over the six non-referral classes on accepted cases only. A class with
no accepted case scores F1 = 0, and a referral-class case that slips into ACCEPT counts as an error.

Conformal sets (split conformal, score = 1 - probability of the true class): a label is in the set
when 1 - p_label <= q. q comes from calibration scores that never saw the scored rows.
"""
from __future__ import annotations

from math import ceil
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score

from data import CLASS_ORDER, REFERRAL_CLASSES

N_CLASSES = len(CLASS_ORDER)
REF_IDX = [CLASS_ORDER.index(c) for c in REFERRAL_CLASSES]
NONREF_IDX = [i for i in range(N_CLASSES) if i not in REF_IDX]
TAU_GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)          # 0.00, 0.01, ..., 0.99
TRIGGERS = ["rule", "ml", "either"]


def set_mask(sets: Iterable[str]) -> np.ndarray:
    """Rule-set strings as written by Phase 2 ("A | B | C") -> boolean mask (n, 9)."""
    rows = []
    for s in sets:
        labels = [] if not isinstance(s, str) or s == "" else s.split(" | ")
        rows.append([c in labels for c in CLASS_ORDER])
    return np.array(rows, dtype=bool)


def hybrid_no_abstention(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Most probable class inside the rule set (top ML class if the set is empty)."""
    P = np.asarray(P, float)
    pred = np.where(M, P, -np.inf).argmax(axis=1)
    empty = ~M.any(axis=1)
    pred[empty] = P[empty].argmax(axis=1)
    return pred


def refer_flags(trigger: str, P: np.ndarray, refer_rule: np.ndarray) -> np.ndarray:
    """rule: low platelets or abnormal WBC (rule F10). ml: top ML class is a referral class. either: one of both."""
    ml = np.isin(np.asarray(P).argmax(axis=1), REF_IDX)
    rule = np.asarray(refer_rule, bool)
    return {"rule": rule, "ml": ml, "either": rule | ml}[trigger]


def decide(P: np.ndarray, M: np.ndarray, refer: np.ndarray, tau: float):
    """Returns (accept, label, reason) where reason is one of REFER, ACCEPT, DISAGREE, LOW_CONF.
    label is the top ML class (the answer when accepted, the first candidate otherwise)."""
    P = np.asarray(P, float)
    top = P.argmax(axis=1)
    conf = P.max(axis=1)
    in_set = M[np.arange(len(top)), top]
    refer = np.asarray(refer, bool)
    accept = ~refer & in_set & (conf >= tau)
    reason = np.where(refer, "REFER", np.where(~in_set, "DISAGREE", np.where(conf < tau, "LOW_CONF", "ACCEPT")))
    return accept, top, reason


def selective_macro_f1(y, pred, accept) -> float:
    accept = np.asarray(accept, bool)
    if not accept.any():
        return 0.0
    return float(f1_score(np.asarray(y)[accept], np.asarray(pred)[accept], labels=NONREF_IDX,
                          average="macro", zero_division=0))


def select_tau(y, P, M, refer, floor: float, grid=TAU_GRID):
    """Best selective macro-F1 among the tau values that keep coverage >= floor. Ties go to the
    smaller tau (more coverage). If no tau reaches the floor, the smallest tau is returned and
    `met` is False. Coverage = accepted / all valid cases (referred cases count as not accepted)."""
    table = []
    for tau in grid:
        acc, top, _ = decide(P, M, refer, tau)
        table.append((float(tau), float(acc.mean()), selective_macro_f1(y, top, acc)))
    feasible = [t for t in table if t[1] >= floor - 1e-12]
    if not feasible:
        return float(grid[0]), False, table
    best = feasible[0]
    for t in feasible[1:]:
        if t[2] > best[2] + 1e-12:
            best = t
    return best[0], True, table


def conformal_quantile(scores, alpha: float) -> float:
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score (inf if that exceeds n)."""
    s = np.sort(np.asarray(scores, float))
    k = ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def conformal_sets(P: np.ndarray, q: float) -> np.ndarray:
    return (1.0 - np.asarray(P, float)) <= q + 1e-12


def conformal_decide(S: np.ndarray, M: np.ndarray, refer: np.ndarray):
    """Accept only a single-label conformal set whose label the rules also allow."""
    single = S.sum(axis=1) == 1
    label = S.argmax(axis=1)
    accept = ~np.asarray(refer, bool) & single & M[np.arange(len(label)), label]
    return accept, label


def macro_f1_from_counts(tp, fp, fn) -> np.ndarray:
    """Vectorised macro-F1 over the last axis (classes). F1 = 2tp / (2tp + fp + fn), 0 when undefined
    (same as scikit-learn with zero_division=0)."""
    tp, fp, fn = (np.asarray(a, float) for a in (tp, fp, fn))
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    return f1.mean(axis=-1)
