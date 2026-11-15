"""Phase 4 evaluation: hybrid agent, REFER trigger, threshold and conformal abstention, H1 and H2.

Research prototype, not a diagnostic tool. Development rows only. Test rows are never read.

Every rule below was pre-registered in notes/phase4_preregistration.md and confirmed by the student on
15 Nov 2026 (16:44), before any Phase 4 metric was computed.

Inputs
  results/phase3/amended/oof_ml_only.csv.gz   outer out-of-fold calibrated probabilities (Phase 3, amended)
  results/phase4/inner/inner_oof.csv.gz       inner predictions on each outer training fold (phase4_inner_cv.py)
  rule engine (src/rules.py), rerun here and checked against results/phase2/rule_sets_dev.csv

R1 H1 per model type: 90% corrected CI of (hybrid without abstention - model alone) macro-F1, margin -0.03
R2 REFER trigger: sensitivity >= 90%, then lowest false-referral rate (pooled over 5 x 963 predictions)
R3/R4 tau per outer fold from inner predictions, coverage floor 70% (60% and 80% sensitivity runs)
R5 H2 matched coverage vs random and confidence-only, paired bootstrap, Holm over 2
R6 conformal abstention (alpha 0.05 primary, 0.10 secondary), descriptive
R7 risk-coverage curves, R8 per-class routing, filled vs other, near cut-offs, R9 frozen settings for Phase 6

Run: python src/phase4_evaluate.py
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import arbitration as A  # noqa: E402
import rules  # noqa: E402
from data import CLASS_ORDER, ROOT, SEED, load_dev  # noqa: E402
from metrics import scores  # noqa: E402
from stats_tests import corrected_ci, corrected_repeated_kfold_t, holm  # noqa: E402

P3A = ROOT / "results" / "phase3" / "amended"
P2 = ROOT / "results" / "phase2"
OUT = ROOT / "results" / "phase4"
K, R, RATIO = 5, 5, 0.25
TYPES = ["logreg", "tree", "forest", "hgb"]
ARM = {"logreg": "smote", "tree": "weights", "forest": "smote", "hgb": "smote"}   # amended rule P2
H2M = ("forest", "smote")
NAMES = {"logreg": "logistic regression", "tree": "decision tree", "forest": "random forest", "hgb": "gradient boosting"}
VARIANTS = ["primary", "hgb12", "mchc32", "hgb12_mchc32"]
VLABEL = {"primary": "HGB 13, MCHC 30 (main)", "hgb12": "HGB 12, MCHC 30", "mchc32": "HGB 13, MCHC 32",
          "hgb12_mchc32": "HGB 12, MCHC 32 (suppl.)"}
FLOORS = [0.70, 0.60, 0.80]
ALPHAS = [0.05, 0.10]
B, NDRAW, MARGIN = 2000, 1000, -0.03
BANDS = {"HGB": [(12.5, 13.5)], "PLT": [(135, 165)], "WBC": [(9.5, 11.5)], "MCV": [(78, 82), (98, 102)],
         "MCHC": [(31.4, 32.5)]}
PCAL = [f"p_cal_{j}" for j in range(9)]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
SHORT = ["Healthy", "Norm. hypochromic", "Norm. normochromic", "Iron deficiency", "Thrombocytopenia",
         "Other microcytic", "Leukemia", "Macrocytic", "Leuk. + thrombocyt."]


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def pct(x):
    return f"{100 * x:.1f}%"


# ------------------------------------------------------------------ data
def rule_info(dev: pd.DataFrame):
    """Rule sets per threshold version and the F10 REFER flag, rerun and checked against Phase 2."""
    saved = pd.read_csv(P2 / "rule_sets_dev.csv").set_index("raw_row").loc[dev.index]
    masks, refer_rule = {}, None
    for v in VARIANTS:
        res = [rules.run(r, rules.VARIANTS[v]) for _, r in dev.iterrows()]
        strings = [" | ".join(x.compatible) for x in res]
        if strings != saved[f"set_{v}"].fillna("").tolist():
            raise RuntimeError(f"rule sets for {v} differ from results/phase2/rule_sets_dev.csv")
        masks[v] = A.set_mask(strings)
        rr = np.array([x.refer_rule_fires for x in res])
        if refer_rule is not None and not (rr == refer_rule).all():
            raise RuntimeError("REFER rule depends on the threshold version")
        refer_rule = rr
    return masks, refer_rule


def subset(oof, fset, model, imb):
    return oof[(oof.fset == fset) & (oof.model == model) & (oof.imb == imb)].sort_values(["rep", "raw_row"]).reset_index(drop=True)


def present_f1(y, p, labels):
    labs = [k for k in labels if (np.asarray(y) == k).any()]
    return scores(y, p, labels=labs)["macro_f1"] if labs else np.nan


# ------------------------------------------------------------------ bootstrap helpers
def count_matrix(y, lab) -> np.ndarray:
    """(n, 18) indicators: tp, fp, fn for each of the six non-referral classes."""
    y, lab = np.asarray(y), np.asarray(lab)
    blocks = []
    for kind in ["tp", "fp", "fn"]:
        for k in A.NONREF_IDX:
            if kind == "tp":
                blocks.append((lab == k) & (y == k))
            elif kind == "fp":
                blocks.append((lab == k) & (y != k))
            else:
                blocks.append((y == k) & (lab != k))
    return np.stack(blocks, axis=1).astype(np.float32)


def f1_from(c: np.ndarray) -> np.ndarray:
    return A.macro_f1_from_counts(c[..., :6], c[..., 6:12], c[..., 12:])


# ------------------------------------------------------------------ main
def main() -> None:
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    dev = load_dev(clean=True)
    n = len(dev)
    pos = pd.Series(np.arange(n), index=dev.index)
    filled = dev["fill_count"].eq(6).to_numpy()
    near = np.zeros(n, bool)
    for c, bands in BANDS.items():
        for lo, hi in bands:
            near |= dev[c].between(lo, hi).to_numpy()
    masks, refer_rule = rule_info(dev)
    oof = pd.read_csv(P3A / "oof_ml_only.csv.gz")
    inner = pd.read_csv(OUT / "inner" / "inner_oof.csv.gz")
    L: list[str] = []
    w = L.append
    w("Phase 4 evaluation (pre-registered in notes/phase4_preregistration.md, confirmed 15 Nov 2026 16:44, before any")
    w("Phase 4 metric was computed). Development rows only, test rows not read. Research prototype, not a diagnostic tool.")
    w(f"rows {n} (filled {filled.sum()}, other {n - filled.sum()}). REJECT: 0 (all cleaned development rows passed the validator).")
    w(f"outer folds {K} x {R} = {K * R}, corrected test n2/n1 = {RATIO}, df = {K * R - 1}. Calibrated probabilities "
      "(amended Phase 3 calibration). Rule sets rerun and identical to results/phase2/rule_sets_dev.csv.")
    w(f"rows near a label cut-off (Phase 3 P7 bands): {near.sum()} of {n} ({pct(near.mean())})")

    # ================================================================ R1 H1
    w("")
    w("== R1. H1: hybrid without abstention vs the same model alone, macro-F1 (9 classes), per outer fold ==")
    w("   difference = hybrid - model alone. Supported if the lower bound of the 90% corrected CI > -0.03.")
    w("   The formal verdict uses the main thresholds. The other versions get the same test (equal prominence).")
    fold_rows, h1_rows, pc_rows, grp_rows = [], [], [], []
    for v in VARIANTS:
        for fs in ["F-core", "F-all"]:
            for m in TYPES:
                g = subset(oof, fs, m, ARM[m])
                P, y, ix = g[PCAL].to_numpy(), g["y"].to_numpy(), pos[g["raw_row"]].to_numpy()
                pm, ph = P.argmax(1), A.hybrid_no_abstention(P, masks[v][ix])
                for (rep, fold), idx in g.groupby(["rep", "fold"]).indices.items():
                    sm, sh = scores(y[idx], pm[idx]), scores(y[idx], ph[idx])
                    r = {"variant": v, "fset": fs, "model": m, "rep": rep, "fold": fold}
                    r.update({f"ml_{k}": val for k, val in sm.items()})
                    r.update({f"hy_{k}": val for k, val in sh.items()})
                    for grp, gmask in [("filled", filled[ix[idx]]), ("other", ~filled[ix[idx]])]:
                        r[f"ml_f1_{grp}"] = present_f1(y[idx][gmask], pm[idx][gmask], range(9))
                        r[f"hy_f1_{grp}"] = present_f1(y[idx][gmask], ph[idx][gmask], range(9))
                    fold_rows.append(r)
                fm = pd.DataFrame([r for r in fold_rows if r["variant"] == v and r["fset"] == fs and r["model"] == m])
                fm = fm.sort_values(["rep", "fold"])
                d = (fm["hy_macro_f1"] - fm["ml_macro_f1"]).to_numpy()
                lo, hi = corrected_ci(d, K, R, RATIO, level=0.90)
                tt = corrected_repeated_kfold_t(d, K, R, RATIO)
                changed = pm != ph
                h1_rows.append({
                    "variant": v, "fset": fs, "model": m, "arm": ARM[m],
                    "ml_macro_f1": fm.ml_macro_f1.mean(), "ml_sd": fm.ml_macro_f1.std(),
                    "hy_macro_f1": fm.hy_macro_f1.mean(), "hy_sd": fm.hy_macro_f1.std(),
                    "ml_bal_acc": fm.ml_balanced_accuracy.mean(), "hy_bal_acc": fm.hy_balanced_accuracy.mean(),
                    "ml_mcc": fm.ml_mcc.mean(), "hy_mcc": fm.hy_mcc.mean(),
                    "diff": d.mean(), "ci90_lo": lo, "ci90_hi": hi, "p_two_sided": tt.p,
                    "supported": bool(lo > MARGIN),
                    "ml_f1_filled": fm.ml_f1_filled.mean(), "hy_f1_filled": fm.hy_f1_filled.mean(),
                    "ml_f1_other": fm.ml_f1_other.mean(), "hy_f1_other": fm.hy_f1_other.mean(),
                    "changed_per_repeat": changed.sum() / R,
                    "fixed_per_repeat": (changed & (ph == y)).sum() / R,
                    "broken_per_repeat": (changed & (pm == y)).sum() / R,
                })
                if fs == "F-core":
                    for k in range(9):
                        yk = y == k
                        pc_rows.append({"variant": v, "model": m, "class": CLASS_ORDER[k], "support_per_repeat": yk.sum() / R,
                                        "ml_recall": (pm[yk] == k).mean(), "hy_recall": (ph[yk] == k).mean()})
                    if v == "primary":
                        for name, pred in [("model alone", pm), ("hybrid", ph)]:
                            err = pred != y
                            nr = near[ix]
                            grp_rows.append(f"  {NAMES[m]:20s} {name:11s} error rate near {pct(err[nr].mean())} vs elsewhere "
                                            f"{pct(err[~nr].mean())}, share of errors near {pct(err[nr].sum() / max(err.sum(), 1))}")
    h1 = pd.DataFrame(h1_rows)
    pd.DataFrame(fold_rows).to_csv(OUT / "h1_fold_metrics.csv", index=False)
    h1.round(5).to_csv(OUT / "h1_summary.csv", index=False)
    pd.DataFrame(pc_rows).round(4).to_csv(OUT / "h1_per_class_recall.csv", index=False)
    for v in VARIANTS:
        w("")
        w(f"  -- thresholds {VLABEL[v]} --")
        w(f"  {'set':6s} {'model':20s} {'alone':>15s} {'hybrid':>15s} {'diff':>7s} {'90% CI':>17s}  verdict   changed/fixed/broken per repeat")
        for fs in ["F-core", "F-all"]:
            for _, r in h1[(h1.variant == v) & (h1.fset == fs)].iterrows():
                verdict = ("supported" if r.supported else "NOT supp.") if fs == "F-core" else "(descr.)"
                w(f"  {fs:6s} {NAMES[r.model]:20s} {fmt(r.ml_macro_f1, r.ml_sd):>15s} {fmt(r.hy_macro_f1, r.hy_sd):>15s} "
                  f"{r['diff']:+7.3f} [{r.ci90_lo:+.3f}, {r.ci90_hi:+.3f}]  {verdict:9s} "
                  f"{r.changed_per_repeat:.1f} / {r.fixed_per_repeat:.1f} / {r.broken_per_repeat:.1f}")
        core = h1[(h1.variant == v) & (h1.fset == "F-core")]
        n_ok = int(core.supported.sum())
        status = "SUPPORTED" if n_ok == 4 else ("PARTLY SUPPORTED" if n_ok > 0 else "NOT SUPPORTED")
        tag = "FORMAL VERDICT" if v == "primary" else "same test, not the formal verdict"
        w(f"  H1 at {VLABEL[v]}: {status} ({n_ok} of 4 model types pass) [{tag}]")
    w("")
    w("  balanced accuracy and MCC (F-core, mean over 25 folds), model alone -> hybrid:")
    for v in VARIANTS:
        for _, r in h1[(h1.variant == v) & (h1.fset == "F-core")].iterrows():
            w(f"  {VLABEL[v]:26s} {NAMES[r.model]:20s} bal. acc {r.ml_bal_acc:.3f} -> {r.hy_bal_acc:.3f}   "
              f"MCC {r.ml_mcc:.3f} -> {r.hy_mcc:.3f}   macro-F1 filled {r.ml_f1_filled:.3f} -> {r.hy_f1_filled:.3f}, "
              f"other {r.ml_f1_other:.3f} -> {r.hy_f1_other:.3f}")
    w("")
    w("  errors near the cut-offs (F-core, main thresholds, 5 repeats pooled):")
    L.extend(grp_rows)

    # ================================================================ R2 REFER
    g2 = subset(oof, "F-core", *H2M)
    P2o, y2, ix2 = g2[PCAL].to_numpy(), g2["y"].to_numpy(), pos[g2["raw_row"]].to_numpy()
    isref = np.isin(y2, A.REF_IDX)
    w("")
    w("== R2. REFER trigger (H2 model, outer predictions pooled over 5 x 963; mean ± std over 25 folds in brackets) ==")
    ref_rows = []
    for trig in A.TRIGGERS:
        f = A.refer_flags(trig, P2o, refer_rule[ix2])
        per_fold = g2.assign(f=f, r=isref).groupby(["rep", "fold"]).apply(
            lambda x: pd.Series({"sens": x.f[x.r].mean(), "frr": x.f[~x.r].mean()}), include_groups=False)
        row = {"trigger": trig, "caught": int(f[isref].sum()), "referral_rows": int(isref.sum()),
               "sensitivity": f[isref].mean(), "false_referrals": int(f[~isref].sum()),
               "other_rows": int((~isref).sum()), "false_referral_rate": f[~isref].mean(),
               "sens_fold_mean": per_fold.sens.mean(), "sens_fold_sd": per_fold.sens.std(),
               "frr_fold_mean": per_fold.frr.mean(), "frr_fold_sd": per_fold.frr.std(),
               "meets_floor": bool(10 * f[isref].sum() >= 9 * isref.sum())}
        for k in range(9):
            row[f"rate_{CLASS_ORDER[k]}"] = f[y2 == k].mean()
        ref_rows.append(row)
        w(f"  {trig:6s} sensitivity {row['caught']}/{row['referral_rows']} = {pct(row['sensitivity'])} "
          f"[{fmt(row['sens_fold_mean'], row['sens_fold_sd'])}]   false referrals {row['false_referrals']}/{row['other_rows']} = "
          f"{pct(row['false_referral_rate'])} [{fmt(row['frr_fold_mean'], row['frr_fold_sd'])}]   floor 90%: "
          f"{'met' if row['meets_floor'] else 'NOT met'}")
    refdf = pd.DataFrame(ref_rows)
    refdf.round(5).to_csv(OUT / "refer_candidates.csv", index=False)
    cands = [r for r in ref_rows if r["meets_floor"]]
    if cands:
        trig = cands[0]
        for r in cands[1:]:
            if r["false_referral_rate"] < trig["false_referral_rate"] - 1e-12:
                trig = r
        why = "meets the 90% floor with the lowest false-referral rate"
    else:
        trig = ref_rows[0]
        for r in ref_rows[1:]:
            if r["sensitivity"] > trig["sensitivity"] + 1e-12:
                trig = r
        why = "no trigger meets the 90% floor, so the most sensitive one is used (pre-declared fallback)"
    TRIG = trig["trigger"]
    w(f"  CHOSEN TRIGGER: {TRIG} ({why})")
    w("  referral rate per class (pooled):")
    for k in range(9):
        w(f"    {CLASS_ORDER[k]:32s} " + "   ".join(f"{r['trigger']} {pct(r['rate_' + CLASS_ORDER[k]])}" for r in ref_rows))
    refer_out = A.refer_flags(TRIG, P2o, refer_rule[ix2])

    # ================================================================ R3/R4 tau per outer fold, R5 H2
    # one bootstrap weight matrix for every comparison (paired across systems)
    rng_b = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng_b.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float32)
    in_set_by_v = {v: masks[v][ix2][np.arange(len(g2)), P2o.argmax(1)] for v in VARIANTS}
    top2, conf2 = P2o.argmax(1), P2o.max(1)
    folds = g2.groupby(["rep", "fold"]).indices
    inner_groups = inner.groupby(["rep", "fold"])
    # leak check: inner rows of a fold = outer training rows of that fold
    for (rep, fold), gi in inner_groups:
        test_rows = set(g2["raw_row"].to_numpy()[folds[(rep, fold)]])
        tr = set(gi["raw_row"])
        if tr & test_rows or len(tr | test_rows) != n:
            raise RuntimeError(f"inner rows of fold {(rep, fold)} overlap the outer test rows")
    w("")
    w("== R3/R4. tau chosen per outer fold on inner predictions (leak check passed: inner rows never include the outer test rows) ==")
    h2_rows, tau_rows, boot_store = [], [], {}
    decisions_main = None
    for v in VARIANTS:
        for floor in FLOORS:
            acc = np.zeros(len(g2), bool)
            reason = np.empty(len(g2), object)
            for (rep, fold), gi in inner_groups:
                Pi, yi, ii = gi[PCAL].to_numpy(), gi["y"].to_numpy(), pos[gi["raw_row"]].to_numpy()
                tau, met, _ = A.select_tau(yi, Pi, masks[v][ii], A.refer_flags(TRIG, Pi, refer_rule[ii]), floor)
                idx = folds[(rep, fold)]
                a, _, rs = A.decide(P2o[idx], masks[v][ix2[idx]], refer_out[idx], tau)
                acc[idx], reason[idx] = a, rs
                tau_rows.append({"variant": v, "floor": floor, "rep": rep, "fold": fold, "tau": tau, "floor_met": met,
                                 "coverage": a.mean(),
                                 "sel_f1": A.selective_macro_f1(y2[idx], top2[idx], a)})
            ph = A.hybrid_no_abstention(P2o, masks[v][ix2])
            # matched-coverage baselines, per outer fold (R5a)
            rng = np.random.default_rng(SEED)
            conf_acc = np.zeros(len(g2), bool)
            rand = {rep: np.zeros((NDRAW, n), np.float32) for rep in range(R)}
            for (rep, fold), idx in sorted(folds.items()):
                k_acc = int(acc[idx].sum())
                cand = idx[~refer_out[idx]]
                order = np.lexsort((rng.random(len(cand)), -conf2[cand]))
                conf_acc[cand[order[:k_acc]]] = True
                keys = rng.random((NDRAW, len(cand)))
                chosen = np.argsort(keys, axis=1)[:, :k_acc]
                cols = ix2[cand][chosen]
                rand[rep][np.repeat(np.arange(NDRAW), k_acc), cols.ravel()] = 1.0
            # per-repeat arrays aligned by row position
            per_rep = {}
            for rep in range(R):
                sel = np.where(g2["rep"].to_numpy() == rep)[0]
                o = np.argsort(ix2[sel])
                sel = sel[o]
                per_rep[rep] = sel
            point = {"hybrid": [], "confidence": [], "random": []}
            cov = []
            bst = {"hybrid": np.zeros((B, R)), "confidence": np.zeros((B, R)), "random": np.zeros((B, R))}
            for rep, sel in per_rep.items():
                yy = y2[sel]
                C_top = count_matrix(yy, top2[sel])
                C_ph = count_matrix(yy, ph[sel])
                ha, ca = acc[sel].astype(np.float32), conf_acc[sel].astype(np.float32)
                cov.append(acc[sel].mean())
                point["hybrid"].append(float(f1_from(ha @ C_top)))
                point["confidence"].append(float(f1_from(ca @ C_top)))
                point["random"].append(float(f1_from(rand[rep] @ C_ph).mean()))
                bst["hybrid"][:, rep] = f1_from((W * ha) @ C_top)
                bst["confidence"][:, rep] = f1_from((W * ca) @ C_top)
                for s in range(0, B, 250):
                    Wc = W[s:s + 250]                                   # (b, n)
                    Cw = (Wc.T[:, :, None] * C_ph[:, None, :]).reshape(n, -1)   # (n, b*18)
                    cnt = (rand[rep] @ Cw).reshape(NDRAW, len(Wc), 18)
                    bst["random"][s:s + 250, rep] = f1_from(cnt).mean(axis=0)
            res = {k: float(np.mean(vv)) for k, vv in point.items()}
            bmean = {k: vv.mean(axis=1) for k, vv in bst.items()}
            row = {"variant": v, "floor": floor, "coverage": float(np.mean(cov)),
                   "sel_f1_hybrid": res["hybrid"], "sel_f1_confidence": res["confidence"], "sel_f1_random": res["random"]}
            pvals = []
            for base in ["random", "confidence"]:
                dd = bmean["hybrid"] - bmean[base]
                row[f"diff_{base}"] = res["hybrid"] - res[base]
                row[f"ci95_lo_{base}"], row[f"ci95_hi_{base}"] = np.percentile(dd, [2.5, 97.5])
                row[f"p_{base}"] = (1 + (dd <= 0).sum()) / (B + 1)
                pvals.append(row[f"p_{base}"])
            adj = holm(pvals)
            row["p_holm_random"], row["p_holm_confidence"] = adj
            row["passes"] = bool((adj < 0.05).all())
            reps = pd.Series(reason)
            row["referred"] = (reps == "REFER").mean()
            row["abstain_disagree"] = (reps == "DISAGREE").mean()
            row["abstain_lowconf"] = (reps == "LOW_CONF").mean()
            h2_rows.append(row)
            if v == "primary" and floor == 0.70:
                decisions_main = pd.DataFrame({"rep": g2["rep"], "fold": g2["fold"], "raw_row": g2["raw_row"], "y": y2,
                                               "decision": np.where(reason == "ACCEPT", "ACCEPT",
                                                                    np.where(reason == "REFER", "REFER", "ABSTAIN")),
                                               "reason": reason, "label": top2, "confidence": conf2,
                                               "hybrid_no_abstention": ph, "confidence_only_accept": conf_acc})
                boot_store["main"] = {k: vv for k, vv in bmean.items()}
    taus = pd.DataFrame(tau_rows)
    taus.round(5).to_csv(OUT / "tau_per_fold.csv", index=False)
    h2 = pd.DataFrame(h2_rows)
    h2.round(5).to_csv(OUT / "h2_summary.csv", index=False)
    decisions_main.round(6).to_csv(OUT / "outer_decisions_main.csv.gz", index=False)
    for v in VARIANTS:
        for floor in FLOORS:
            t = taus[(taus.variant == v) & (taus.floor == floor)]
            w(f"  {VLABEL[v]:26s} floor {floor:.0%}: tau median {t.tau.median():.2f} (min {t.tau.min():.2f}, max {t.tau.max():.2f}), "
              f"floor reached in {int(t.floor_met.sum())}/25 folds, outer coverage {fmt(t.coverage.mean(), t.coverage.std())}, "
              f"selective macro-F1 per fold {fmt(t.sel_f1.mean(), t.sel_f1.std())}")
    w("")
    w("== R5. H2: selective macro-F1 (6 non-referral classes) at matched coverage, shared REFER step ==")
    w("   per repeat (each row once), averaged over 5 repeats. Paired bootstrap, 2,000 case resamples, 1,000 random draws.")
    w("   one-sided p = (1 + #(diff <= 0)) / 2001, Holm over the two baselines. Formal verdict: main thresholds, 70% floor.")
    for _, r in h2.iterrows():
        tag = "FORMAL VERDICT" if (r.variant == "primary" and r.floor == 0.70) else "same test"
        w("")
        w(f"  {VLABEL[r.variant]}, floor {r.floor:.0%} [{tag}]")
        w(f"    coverage {pct(r.coverage)} (referred {pct(r.referred)}, abstained: disagreement {pct(r.abstain_disagree)}, "
          f"low confidence {pct(r.abstain_lowconf)})")
        w(f"    selective macro-F1: hybrid {r.sel_f1_hybrid:.3f}, confidence-only {r.sel_f1_confidence:.3f}, random {r.sel_f1_random:.3f}")
        for base in ["random", "confidence"]:
            w(f"    hybrid - {base:10s} {r['diff_' + base]:+.3f}  95% CI [{r['ci95_lo_' + base]:+.3f}, {r['ci95_hi_' + base]:+.3f}]  "
              f"p {r['p_' + base]:.4f}  Holm p {r['p_holm_' + base]:.4f}")
        w(f"    H2: {'SUPPORTED (beats both)' if r.passes else 'NOT SUPPORTED'}")

    # ================================================================ R6 conformal
    w("")
    w("== R6. conformal abstention (score 1 - p(true class), q from inner predictions; shared REFER step; descriptive) ==")
    conf_rows, conf_pc = [], []
    for alpha in ALPHAS:
        S = np.zeros((len(g2), 9), bool)
        qs = []
        for (rep, fold), gi in inner_groups:
            Pi, yi = gi[PCAL].to_numpy(), gi["y"].to_numpy()
            q = A.conformal_quantile(1 - Pi[np.arange(len(yi)), yi], alpha)
            qs.append(q)
            idx = folds[(rep, fold)]
            S[idx] = A.conformal_sets(P2o[idx], q)
        size = S.sum(1)
        hit = S[np.arange(len(y2)), y2]
        for k in range(9):
            yk = y2 == k
            conf_pc.append({"alpha": alpha, "class": CLASS_ORDER[k], "set_coverage": hit[yk].mean(),
                            "mean_set_size": size[yk].mean(), "empty_sets_per_repeat": (size[yk] == 0).sum() / R})
        for v in VARIANTS:
            a, lab = A.conformal_decide(S, masks[v][ix2], refer_out)
            per = [(a[s].mean(), A.selective_macro_f1(y2[s], lab[s], a[s])) for s in
                   [np.where(g2["rep"].to_numpy() == rep)[0] for rep in range(R)]]
            fold_sel = [A.selective_macro_f1(y2[idx], lab[idx], a[idx]) for idx in folds.values()]
            conf_rows.append({"alpha": alpha, "variant": v, "q_median": float(np.median(qs)),
                              "set_coverage": hit.mean(), "mean_set_size": size.mean(),
                              "singleton_share": (size == 1).mean(), "empty_share": (size == 0).mean(),
                              "coverage": float(np.mean([p[0] for p in per])),
                              "sel_f1": float(np.mean([p[1] for p in per])),
                              "sel_f1_fold_mean": float(np.mean(fold_sel)), "sel_f1_fold_sd": float(np.std(fold_sel, ddof=1))})
    cdf = pd.DataFrame(conf_rows)
    cdf.round(5).to_csv(OUT / "conformal_summary.csv", index=False)
    pd.DataFrame(conf_pc).round(4).to_csv(OUT / "conformal_per_class.csv", index=False)
    for alpha in ALPHAS:
        c0 = cdf[cdf.alpha == alpha].iloc[0]
        w(f"  alpha {alpha:.2f} ({'primary' if alpha == 0.05 else 'secondary'}): median q {c0.q_median:.3f}, set coverage {pct(c0.set_coverage)} "
          f"(target >= {pct(1 - alpha)}), mean set size {c0.mean_set_size:.2f}, singletons {pct(c0.singleton_share)}, empty {pct(c0.empty_share)}")
        for _, r in cdf[cdf.alpha == alpha].iterrows():
            w(f"    {VLABEL[r.variant]:26s} coverage {pct(r.coverage)}, selective macro-F1 {r.sel_f1:.3f} "
              f"(per fold {fmt(r.sel_f1_fold_mean, r.sel_f1_fold_sd)})")
        w("    per class: set coverage / mean set size")
        for rr in [x for x in conf_pc if x["alpha"] == alpha]:
            w(f"      {rr['class']:32s} {pct(rr['set_coverage']):>6s} / {rr['mean_set_size']:.2f}")

    # ================================================================ R7 risk-coverage
    rc_rows = []
    for v in VARIANTS:
        for rep in range(R):
            s = np.where(g2["rep"].to_numpy() == rep)[0]
            ph = A.hybrid_no_abstention(P2o[s], masks[v][ix2[s]])
            nonref = ~refer_out[s]
            rand_err = (ph[nonref] != y2[s][nonref]).mean()
            for tau in A.TAU_GRID:
                for system, a in [("hybrid", nonref & in_set_by_v[v][s] & (conf2[s] >= tau)),
                                  ("confidence", nonref & (conf2[s] >= tau))]:
                    rc_rows.append({"variant": v, "rep": rep, "system": system, "tau": tau, "coverage": a.mean(),
                                    "error_rate": (top2[s][a] != y2[s][a]).mean() if a.any() else np.nan,
                                    "sel_f1": A.selective_macro_f1(y2[s], top2[s], a)})
            rc_rows.append({"variant": v, "rep": rep, "system": "random", "tau": np.nan, "coverage": nonref.mean(),
                            "error_rate": rand_err, "sel_f1": np.nan})
    rc = pd.DataFrame(rc_rows)
    rcm = rc.groupby(["variant", "system", "tau"], dropna=False)[["coverage", "error_rate", "sel_f1"]].mean().reset_index()
    rcm.round(5).to_csv(OUT / "risk_coverage.csv", index=False)
    w("")
    w("== R7. risk-coverage (tau swept on outer predictions after REFER, mean over repeats; descriptive, never picks tau) ==")
    for v in VARIANTS:
        parts = []
        for system in ["hybrid", "confidence"]:
            c = rcm[(rcm.variant == v) & (rcm.system == system)].dropna(subset=["error_rate"]).sort_values("coverage")
            aurc = np.trapezoid(c.error_rate, c.coverage)
            span = c.coverage.max() - c.coverage.min()
            parts.append(f"{system} AURC {aurc:.4f} over coverage {pct(c.coverage.min())}-{pct(c.coverage.max())} "
                         f"(mean risk {aurc / span:.4f})")
        rr = rcm[(rcm.variant == v) & (rcm.system == "random")].iloc[0]
        w(f"  {VLABEL[v]:26s} " + "; ".join(parts) + f"; random: flat error rate {rr.error_rate:.4f} up to {pct(rr.coverage)}")

    # ================================================================ R8 per class, filled, near
    d = decisions_main
    w("")
    w("== R8. routing per class (main thresholds, 70% floor, 5 repeats pooled) ==")
    w(f"  {'class':32s} {'refer':>7s} {'accept':>7s} {'abst-dis':>9s} {'abst-low':>9s} {'correct|acc':>12s}")
    routing = []
    for k in range(9):
        dk = d[d.y == k]
        accd = dk[dk.decision == "ACCEPT"]
        r = {"class": CLASS_ORDER[k], "refer": (dk.reason == "REFER").mean(), "accept": (dk.reason == "ACCEPT").mean(),
             "abstain_disagree": (dk.reason == "DISAGREE").mean(), "abstain_lowconf": (dk.reason == "LOW_CONF").mean(),
             "correct_given_accept": (accd.label == accd.y).mean() if len(accd) else np.nan}
        routing.append(r)
        w(f"  {r['class']:32s} {pct(r['refer']):>7s} {pct(r['accept']):>7s} {pct(r['abstain_disagree']):>9s} "
          f"{pct(r['abstain_lowconf']):>9s} {('-' if np.isnan(r['correct_given_accept']) else pct(r['correct_given_accept'])):>12s}")
    pd.DataFrame(routing).round(4).to_csv(OUT / "routing_per_class.csv", index=False)
    fl, nr = filled[ix2], near[ix2]
    w("")
    w("  filled vs other rows (main thresholds, 70% floor; selective macro-F1 over the non-referral classes present):")
    for name, gm in [("filled", fl), ("other", ~fl)]:
        cv_, sf = [], []
        for rep in range(R):
            s = (g2["rep"].to_numpy() == rep) & gm
            a = (d.decision.to_numpy() == "ACCEPT")[s]
            cv_.append(a.mean())
            labs = [k for k in A.NONREF_IDX if (y2[s] == k).any()]
            sf.append(scores(y2[s][a], top2[s][a], labels=labs)["macro_f1"])
        w(f"    {name:6s} coverage {pct(np.mean(cv_))}, selective macro-F1 {np.mean(sf):.3f}")
    w("")
    w("  near the cut-offs (main thresholds, 70% floor, 5 repeats pooled):")
    nonref = d.reason.to_numpy() != "REFER"
    accm = d.decision.to_numpy() == "ACCEPT"
    err = accm & (d.label.to_numpy() != y2)
    for name, gm in [("near", nr), ("elsewhere", ~nr)]:
        w(f"    {name:9s} abstention among non-referred {pct((~accm[gm & nonref]).mean())}, "
          f"error rate among accepted {pct(err[gm & accm].sum() / max((gm & accm).sum(), 1))}")

    # ================================================================ R9 frozen settings for Phase 6
    frozen = {"written": "Phase 4, 15 Nov 2026, by src/phase4_evaluate.py (development rows only)",
              "h2_model": "random forest + SMOTE, F-core, sigmoid calibration on probabilities",
              "refer_trigger": TRIG, "tau_floor": 0.70, "tau": {}, "tau_floor_met": {}, "conformal_q": {}}
    for v in VARIANTS:
        M = masks[v][ix2]
        tau, met, _ = A.select_tau(y2, P2o, M, refer_out, 0.70)
        frozen["tau"][v], frozen["tau_floor_met"][v] = tau, bool(met)
    for alpha in ALPHAS:
        frozen["conformal_q"][f"{alpha:.2f}"] = A.conformal_quantile(1 - P2o[np.arange(len(y2)), y2], alpha)
    (OUT / "frozen_phase6.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    w("")
    w("== R9. settings frozen for Phase 6 (same rules on outer predictions pooled over 5 repeats) ==")
    w(f"  REFER trigger: {TRIG}. tau (70% floor): " + ", ".join(f"{VLABEL[v]} {frozen['tau'][v]:.2f}"
                                                              + ("" if frozen["tau_floor_met"][v] else " (floor NOT met)")
                                                              for v in VARIANTS))
    w("  conformal q: " + ", ".join(f"alpha {a} -> {q:.4f}" for a, q in frozen["conformal_q"].items()))

    # ================================================================ paper table (Part B)
    T = ["| Thresholds | Model | Macro-F1 alone | Macro-F1 hybrid | Difference [90% CI] | H1 |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for _, r in h1[(h1.variant == v) & (h1.fset == "F-core")].iterrows():
            T.append(f"| {VLABEL[v]} | {NAMES[r.model]} | {fmt(r.ml_macro_f1, r.ml_sd)} | {fmt(r.hy_macro_f1, r.hy_sd)} | "
                     f"{r['diff']:+.3f} [{r.ci90_lo:+.3f}, {r.ci90_hi:+.3f}] | {'pass' if r.supported else 'fail'} |")
    T2 = ["| Thresholds | Floor | Coverage | Hybrid | Confidence-only | Random | Holm p (random, conf.) | H2 |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in h2.iterrows():
        T2.append(f"| {VLABEL[r.variant]} | {r.floor:.0%} | {pct(r.coverage)} | {r.sel_f1_hybrid:.3f} | {r.sel_f1_confidence:.3f} | "
                  f"{r.sel_f1_random:.3f} | {r.p_holm_random:.4f}, {r.p_holm_confidence:.4f} | {'pass' if r.passes else 'fail'} |")
    (OUT / "main_results_table_part_b.md").write_text(
        "Generated by src/phase4_evaluate.py (development CV, F-core, calibrated). Research prototype.\n\n"
        "H1: hybrid without abstention vs the same model alone (macro-F1, 9 classes, mean ± std over 25 folds).\n\n"
        + "\n".join(T) + "\n\nH2: selective macro-F1 (6 non-referral classes) at matched coverage, random forest + SMOTE.\n\n"
        + "\n".join(T2) + "\n", encoding="utf-8")

    # ================================================================ figures
    plot_h1(h1)
    plot_rc(rcm, h2)
    w("")
    w(f"run time {time.perf_counter() - t_start:.0f} s")
    (OUT / "evaluation.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_axisbelow(True)


def plot_h1(h1: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    vs = ["primary", "hgb12", "mchc32"]
    cols = {"primary": BLUE, "hgb12": ORANGE, "mchc32": MUTED}
    marks = {"primary": "o", "hgb12": "s", "mchc32": "^"}
    for j, v in enumerate(vs):
        sub = h1[(h1.variant == v) & (h1.fset == "F-core")].set_index("model").reindex(TYPES)
        yy = np.arange(len(TYPES)) + (j - 1) * 0.22
        ax.errorbar(sub["diff"], yy, xerr=[sub["diff"] - sub.ci90_lo, sub.ci90_hi - sub["diff"]], fmt=marks[v],
                    color=cols[v], ms=4, elinewidth=1, capsize=2, label=VLABEL[v])
    ax.axvline(MARGIN, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(0, color=GRID, lw=0.8)
    ax.text(MARGIN, -0.75, " margin −0.03", color=INK, fontsize=6, va="bottom")
    ax.set_yticks(range(len(TYPES)), [NAMES[m] for m in TYPES])
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1, hybrid minus model alone (90% CI)", color=MUTED, fontsize=7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    style(ax)
    ax.legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_h1_differences.png")
    plt.close(fig)


def plot_rc(rcm: pd.DataFrame, h2: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=300)
    v = "primary"
    for ax, metric, lab in [(axes[0], "error_rate", "Error rate among accepted cases"),
                            (axes[1], "sel_f1", "Selective macro-F1 (6 classes)")]:
        for system, col, name in [("hybrid", BLUE, "Hybrid (rules + τ)"), ("confidence", ORANGE, "Confidence only")]:
            c = rcm[(rcm.variant == v) & (rcm.system == system)].dropna(subset=[metric]).sort_values("coverage")
            ax.plot(c.coverage, c[metric], color=col, lw=2, label=name)
        rr = rcm[(rcm.variant == v) & (rcm.system == "random")].iloc[0]
        if metric == "error_rate":
            ax.hlines(rr.error_rate, 0.45, rr.coverage, color=MUTED, lw=1.2, ls=(0, (3, 2)), label="Random (expected)")
            ax.text(rr.coverage, rr.error_rate, " random", color=MUTED, fontsize=6, va="center", ha="left")
        m = h2[(h2.variant == v) & (h2.floor == 0.70)].iloc[0]
        if metric == "sel_f1":
            ax.plot([m.coverage], [m.sel_f1_hybrid], "o", color=BLUE, ms=6, mec="white", mew=1.2)
            ax.annotate("chosen τ (nested CV)", (m.coverage, m.sel_f1_hybrid), textcoords="offset points", xytext=(0, -22),
                        fontsize=6, color=INK, ha="center", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
        ax.set_xlabel("Coverage (share of valid cases accepted)", color=MUTED, fontsize=7)
        ax.set_ylabel(lab, color=MUTED, fontsize=7)
        ax.set_xlim(0.45, 0.97)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        style(ax)
    axes[0].legend(frameon=False, fontsize=6, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_risk_coverage.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
