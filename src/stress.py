"""Stress tests for the agent (proposal Section IV-D): noise, bad input, edges and referral.

Research prototype, not a diagnostic tool.

The same functions run on development rows in Phase 5 (a dry run of the suite) and on the held-out test rows in
Phase 6, where acceptance targets 1 (referral) and 2 (validation) are measured.

  noise      multiplicative Gaussian noise on all 14 inputs (SD 1% and 3% of each value, 10 draws per row).
             Descriptive: share of decisions and labels that stay the same. The levels are Claude's choice
             (15 Nov 2026), not a clinical claim about analyser precision.
  bad input  for sampled rows, every column gets each of 8 corruptions (key missing, None, NaN, text, +inf,
             -inf, below its screening range, above it). Target 2: 100% must come out as REJECT naming that column.
  edges      every rule cut-off at cut - 0.01, cut and cut + 0.01, and every validator limit just inside and just
             outside. The rule fact must flip on the right side, the validator must accept or reject as defined,
             and every agent record must pass the consistency check.
  referral   stress cases = every row, 5 noisy copies (SD 3%) of every row, and rows with platelets set to 30 or
             WBC set to 60. Target 1: every valid case that meets the chosen REFER trigger must come out as REFER.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import arbitration as A
import rules
from agent import Agent, check_consistency
from data import F_CORE, FEATURES, SEED
from validator import CLOSED_RANGES, POSITIVE_UPPER

NOISE_LEVELS = [0.01, 0.03]
NOISE_DRAWS = 10
CORRUPTIONS = ["key_missing", "none", "nan", "text", "inf", "neg_inf", "below_range", "above_range"]


def _limits(col):
    if col in CLOSED_RANGES:
        lo, hi = CLOSED_RANGES[col]
        return lo, hi, True
    return 0.0, POSITIVE_UPPER[col], False


def noise_test(agent: Agent, df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    base = agent.decide_frame(df[FEATURES])
    rng = np.random.default_rng(seed)
    rows = []
    for level in NOISE_LEVELS:
        same_dec, same_both, rejected = [], [], []
        for _ in range(NOISE_DRAWS):
            noisy = df[FEATURES] * (1 + rng.normal(0, level, size=(len(df), len(FEATURES))))
            d = agent.decide_frame(noisy)
            same_dec.append((d.decision == base.decision).to_numpy())
            same_both.append(((d.decision == base.decision) & (d.label == base.label)).to_numpy())
            rejected.append((d.decision == "REJECT").to_numpy())
        sd, sb, rj = np.concatenate(same_dec), np.concatenate(same_both), np.concatenate(rejected)
        row = {"noise_sd": level, "cases": len(sd), "same_decision": sd.mean(), "same_decision_and_label": sb.mean(),
               "rejected_after_noise": rj.mean()}
        tiled = np.tile(base.decision.to_numpy(), NOISE_DRAWS)
        for dec in ["ACCEPT", "ABSTAIN", "REFER"]:
            m = tiled == dec
            row[f"same_decision_if_{dec}"] = sb[m].mean() if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _corrupt(rec: dict, col: str, kind: str) -> dict:
    r = dict(rec)
    lo, hi, closed = _limits(col)
    if kind == "key_missing":
        del r[col]
    else:
        r[col] = {"none": None, "nan": float("nan"), "text": "abc", "inf": float("inf"), "neg_inf": float("-inf"),
                  "below_range": (lo - 1.0) if closed else 0.0, "above_range": hi + 1.0}[kind]
    return r


def bad_input_test(agent: Agent, df: pd.DataFrame, n_base: int | None = None, seed: int = SEED):
    base = df if n_base is None or n_base >= len(df) else df.sample(n_base, random_state=seed)
    rows, fails = [], []
    for rid, r in base[FEATURES].iterrows():
        rec0 = r.to_dict()
        for col in FEATURES:
            for kind in CORRUPTIONS:
                out = agent.run(_corrupt(rec0, col, kind), f"bad-{rid}-{col}-{kind}", explain=False)
                ok = out["decision"] == "REJECT" and any(x.startswith(col) for x in out["validator"]["reasons"])
                cons = check_consistency(out)
                rows.append({"row": rid, "column": col, "corruption": kind, "caught": ok, "consistent": not cons})
                if not ok or cons:
                    fails.append({"row": rid, "column": col, "corruption": kind, "decision": out["decision"],
                                  "reasons": out["validator"]["reasons"], "consistency": cons})
    return pd.DataFrame(rows), fails


CUTS = [("HGB", "anemia", "hgb_anemia", "<"), ("MCV", "microcytic", "mcv_micro", "<"),
        ("MCV", "macrocytic", "mcv_macro", ">"), ("MCHC", "hypochromic", "mchc_hypo", "<"),
        ("PLT", "low_plt", "plt_low", "<"), ("WBC", "leukocytosis", "wbc_high", ">"), ("WBC", "leukopenia", "wbc_low", "<")]


def edge_test(agent: Agent, df: pd.DataFrame, n_base: int = 20, seed: int = SEED):
    base = df[FEATURES].sample(min(n_base, len(df)), random_state=seed)
    rows, fails = [], []
    for rid, r in base.iterrows():
        rec0 = r.to_dict()
        for col, fact, attr, op in CUTS:
            cut = getattr(agent.thresholds, attr)
            for off in (-0.01, 0.0, 0.01):
                rec = dict(rec0)
                rec[col] = round(cut + off, 4)
                expect = rec[col] < cut if op == "<" else rec[col] > cut
                got = rules.run(rec, agent.thresholds).facts[fact]
                out = agent.run(rec, f"edge-{rid}-{fact}-{off:+.2f}")
                cons = check_consistency(out)
                ok = (got == expect) and out["decision"] in ("ACCEPT", "ABSTAIN", "REFER") and not cons
                rows.append({"kind": "rule cut-off", "row": rid, "item": f"{fact} ({col} {op} {cut:g})",
                             "value": rec[col], "ok": ok})
                if not ok:
                    fails.append({"row": rid, "fact": fact, "value": rec[col], "expected": expect, "got": got,
                                  "decision": out["decision"], "consistency": cons})
        for col in FEATURES:
            lo, hi, closed = _limits(col)
            checks = [(lo, True), (hi, True), (lo - 0.01, False), (hi + 0.01, False)] if closed else \
                [(0.01, True), (hi, True), (0.0, False), (hi + 0.01, False)]
            for v, valid in checks:
                rec = dict(rec0)
                rec[col] = v
                out = agent.run(rec, f"limit-{rid}-{col}-{v:g}", explain=False)
                named = any(x.startswith(col) for x in out["validator"]["reasons"])
                ok = (not named) if valid else (out["decision"] == "REJECT" and named)
                rows.append({"kind": "validator limit", "row": rid, "item": f"{col} = {v:g} ({'valid' if valid else 'invalid'})",
                             "value": v, "ok": ok})
                if not ok:
                    fails.append({"row": rid, "column": col, "value": v, "expected_valid": valid,
                                  "decision": out["decision"], "reasons": out["validator"]["reasons"]})
    return pd.DataFrame(rows), fails


def referral_test(agent: Agent, df: pd.DataFrame, seed: int = SEED):
    rng = np.random.default_rng(seed)
    X = df[FEATURES]
    cases = [("row", rid, r.to_dict()) for rid, r in X.iterrows()]
    for k in range(5):
        noisy = X * (1 + rng.normal(0, 0.03, size=X.shape))
        cases += [(f"noise3%-{k}", rid, r.to_dict()) for rid, r in noisy.iterrows()]
    for col, val in [("PLT", 30.0), ("WBC", 60.0)]:
        for rid, r in X.sample(min(20, len(X)), random_state=seed).iterrows():
            rec = r.to_dict()
            rec[col] = val
            cases.append((f"{col}={val:g}", rid, rec))
    rows, fails = [], []
    for kind, rid, rec in cases:
        out = agent.run(rec, f"refer-{kind}-{rid}", explain=False)
        if out["decision"] == "REJECT":
            rows.append({"kind": kind, "row": rid, "valid": False, "trigger_met": False, "referred": False})
            continue
        P = agent.probabilities(np.array([[float(rec[c]) for c in F_CORE]]))
        if agent.trigger == "ml":                         # computed directly, not through the agent's own flag
            met = bool(np.isin(P.argmax(axis=1), A.REF_IDX)[0])
        else:
            met = bool(A.refer_flags(agent.trigger, P, [rules.run(rec, agent.thresholds).refer_rule_fires])[0])
        ref = out["decision"] == "REFER"
        rows.append({"kind": kind, "row": rid, "valid": True, "trigger_met": met, "referred": ref})
        if met != ref:
            fails.append({"kind": kind, "row": rid, "trigger_met": met, "decision": out["decision"]})
    return pd.DataFrame(rows), fails
