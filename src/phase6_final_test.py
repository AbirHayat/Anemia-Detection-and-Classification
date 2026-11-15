"""Phase 6: the single held-out test run (pre-registered in notes/phase6_preregistration.md).

Research prototype, not a diagnostic tool.

Real run (ONCE, only after the student's go-ahead "run the final test" and the git tag):
    python src/phase6_final_test.py --final
  * refuses unless HEAD carries the tag `phase6-freeze` and no tracked file differs from it
  * refuses if results/phase6/ already exists (one run only, never rerun to change a result)
  * checks every frozen file against results/phase5/freeze_manifest.json before reading the test rows
  * reads the test rows once with load_test(i_understand_this_is_phase6=True)

Rehearsal (development rows, identical code path, writes to a folder you name, numbers are IN-SAMPLE and
meaningless, only checks that nothing crashes):
    python src/phase6_final_test.py --rehearsal <out_dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import arbitration as A
import rules
import stress
from agent import Agent, check_consistency, sha_file
from data import CLASS_ORDER, F_CORE, FEATURES, FILL_COLS, ROOT, SEED, TARGET, load_dev, load_test
from metrics import confusion, per_class, scores
from phase2_rules_baseline import cascade_predict
from stats_tests import holm, mcnemar
from validator import validate_record

TAG = "phase6-freeze"
OUT_FINAL = ROOT / "results" / "phase6"
MANIFEST = ROOT / "results" / "phase5" / "freeze_manifest.json"
VARIANTS = ["primary", "mchc32", "hgb12", "hgb12_mchc32"]
VN = {"primary": "HGB 13, MCHC 30 (main)", "mchc32": "HGB 13, MCHC 32", "hgb12": "HGB 12, MCHC 30",
      "hgb12_mchc32": "HGB 12, MCHC 32"}
BANDS = {"HGB": [(12.5, 13.5)], "PLT": [(135, 165)], "WBC": [(9.5, 11.5)], "MCV": [(78, 82), (98, 102)],
         "MCHC": [(31.4, 32.5)]}
SMALL = 5


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def guards() -> None:
    if OUT_FINAL.exists():
        sys.exit("results/phase6/ already exists: the final test runs once and is never rerun.")
    tags = git("tag", "--points-at", "HEAD").split()
    if TAG not in tags:
        sys.exit(f"HEAD does not carry the tag {TAG}. Tag the frozen commit first.")
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("tracked files differ from the tagged commit. Nothing was read.")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = [rel for rel, h in man["files"].items() if sha_file(ROOT / rel) != h]
    if bad:
        sys.exit(f"frozen files changed since the freeze: {bad}. Nothing was read.")


def metric_block(y, pred) -> dict:
    s = scores(y, pred)
    return {"macro_f1": s["macro_f1"], "balanced_accuracy": s["balanced_accuracy"], "mcc": s["mcc"]}


def run(df: pd.DataFrame, out: Path, label: str) -> None:
    warnings.filterwarnings("ignore")
    t_all = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stress").mkdir(exist_ok=True)
    L: list[str] = []
    w = L.append
    w(f"Phase 6 final test run ({label}). Pre-registered in notes/phase6_preregistration.md. Research prototype.")
    w(f"git HEAD {git('rev-parse', '--short', 'HEAD')}, tags at HEAD: {git('tag', '--points-at', 'HEAD') or 'none'}")
    w(f"rows read: {len(df)}")

    # ---- T0 validation
    reasons = df[FEATURES].apply(lambda r: validate_record(r.to_dict()), axis=1)
    valid = reasons.apply(len).eq(0).to_numpy()
    w(f"T0. validator: {int((~valid).sum())} row(s) REJECTED and left out of every metric, {int(valid.sum())} valid")
    for rid, rs in reasons[~valid].items():
        w(f"    row {rid}: {', '.join(rs)} (true label {df.loc[rid, TARGET]})")
    v = df[valid].copy()
    y = v[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    agents = {k: Agent(k) for k in VARIANTS}
    P = agents["primary"].probabilities(v[F_CORE].to_numpy(float))
    rr = {k: [rules.run(r, rules.VARIANTS[k]) for _, r in v.iterrows()] for k in VARIANTS}
    masks = {k: A.set_mask([" | ".join(x.compatible) for x in rr[k]]) for k in VARIANTS}
    refer_rule = np.array([x.refer_rule_fires for x in rr["primary"]])
    isref = np.isin(y, A.REF_IDX)
    dev = load_dev(clean=True)
    modes = {c: dev[c].mode().iloc[0] for c in FILL_COLS}
    filled = np.column_stack([v[c].to_numpy() == modes[c] for c in FILL_COLS]).sum(axis=1) == len(FILL_COLS)
    near = np.zeros(len(v), bool)
    for c, bands in BANDS.items():
        for lo, hi in bands:
            near |= v[c].between(lo, hi).to_numpy()
    w(f"valid rows: {len(v)} (mean-filled {int(filled.sum())}, other {int((~filled).sum())}, near a cut-off {int(near.sum())})")
    w("class support (valid test rows): " + ", ".join(f"{CLASS_ORDER[k]} {int((y == k).sum())}" for k in range(9)))

    # ---- T1 systems
    pred_ml = P.argmax(axis=1)
    cascade = json.loads((ROOT / "results" / "phase2" / "cascade_frozen.json").read_text(encoding="utf-8"))
    pred_cas = np.array([CLASS_ORDER.index(cascade_predict(r, cascade)) for _, r in v.iterrows()])
    pred_hy = {k: A.hybrid_no_abstention(P, masks[k]) for k in VARIANTS}
    sys_rows, pc_rows = [], []

    def add_system(name, variant, pred):
        m = metric_block(y, pred)
        sys_rows.append({"system": name, "thresholds": variant, **m,
                         "macro_f1_filled": scores(y[filled], pred[filled], labels=sorted(set(y[filled])))["macro_f1"] if filled.any() else np.nan,
                         "macro_f1_other": scores(y[~filled], pred[~filled], labels=sorted(set(y[~filled])))["macro_f1"] if (~filled).any() else np.nan,
                         "error_rate_near": float((pred != y)[near].mean()) if near.any() else np.nan,
                         "error_rate_elsewhere": float((pred != y)[~near].mean()) if (~near).any() else np.nan})
        pc = per_class(y, pred)
        for k in range(9):
            pc_rows.append({"system": name, "thresholds": variant, "class": CLASS_ORDER[k],
                            "support": int(pc.loc[k, "support"]), "precision": pc.loc[k, "precision"],
                            "recall": pc.loc[k, "recall"]})
        pd.DataFrame(confusion(y, pred), index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
            out / f"confusion_{name.replace(' ', '_')}_{variant}.csv")
        return m

    add_system("random forest", "-", pred_ml)
    add_system("cascade baseline", "-", pred_cas)
    for k in VARIANTS:
        add_system("hybrid no abstention", k, pred_hy[k])
    sysdf = pd.DataFrame(sys_rows)
    sysdf.round(5).to_csv(out / "systems.csv", index=False)
    pd.DataFrame(pc_rows).round(4).to_csv(out / "per_class.csv", index=False)
    w("")
    w("== T1. systems on the valid test rows (9 classes) ==")
    w(f"  {'system':22s} {'thresholds':24s} {'macro-F1':>9s} {'bal.acc':>8s} {'MCC':>7s} {'F1 filled':>10s} {'F1 other':>9s} {'err near':>9s} {'err else':>9s}")
    for _, r in sysdf.iterrows():
        w(f"  {r.system:22s} {VN.get(r.thresholds, '-'):24s} {r.macro_f1:9.3f} {r.balanced_accuracy:8.3f} {r.mcc:7.3f} "
          f"{r.macro_f1_filled:10.3f} {r.macro_f1_other:9.3f} {pct(r.error_rate_near):>9s} {pct(r.error_rate_elsewhere):>9s}")

    # expert system alone
    w("")
    w("  expert system alone (rule-set coverage = true label inside the compatible set):")
    rule_rows = []
    for k in VARIANTS:
        cov = masks[k][np.arange(len(y)), y]
        size = masks[k].sum(axis=1)
        rule_rows.append({"thresholds": k, "rule_set_coverage": cov.mean(), "mean_set_size": size.mean(),
                          "singletons": int((size == 1).sum()), "empty": int((size == 0).sum())})
        w(f"    {VN[k]:24s} coverage {int(cov.sum())}/{len(y)} ({pct(cov.mean())}), mean set size {size.mean():.2f}, "
          f"singletons {int((size == 1).sum())}, empty {int((size == 0).sum())}")
    pd.DataFrame(rule_rows).round(5).to_csv(out / "rules_only.csv", index=False)
    w(f"    rule REFER trigger (descriptive): caught {int(refer_rule[isref].sum())}/{int(isref.sum())}, "
      f"false referrals {int(refer_rule[~isref].sum())}/{int((~isref).sum())}")

    # agent with abstention
    w("")
    w("  the agent with abstention (frozen tau, ML REFER trigger), plus confidence-only at the same coverage:")
    ag_rows, routing, audit_recs = [], [], []
    conf = P.max(axis=1)
    rng = np.random.default_rng(SEED)
    for k in VARIANTS:
        ag = agents[k]
        refer = ag.trigger_met(P, refer_rule)
        acc, top, reason = A.decide(P, masks[k], refer, ag.tau)
        n_acc = int(acc.sum())
        cand = np.where(~refer)[0]
        order = cand[np.lexsort((rng.random(len(cand)), -conf[cand]))]
        cacc = np.zeros(len(y), bool)
        cacc[order[:n_acc]] = True
        row = {"thresholds": k, "tau": ag.tau, "coverage": acc.mean(), "referred": refer.mean(),
               "abstain_disagree": (reason == "DISAGREE").mean(), "abstain_lowconf": (reason == "LOW_CONF").mean(),
               "selective_macro_f1": A.selective_macro_f1(y, top, acc),
               "confidence_only_selective_macro_f1": A.selective_macro_f1(y, top, cacc),
               "errors_among_accepted": int((acc & (top != y)).sum()), "accepted": n_acc,
               "referral_sensitivity": refer[isref].mean() if isref.any() else np.nan,
               "referral_caught": int(refer[isref].sum()), "referral_rows": int(isref.sum()),
               "false_referral_rate": refer[~isref].mean(), "false_referrals": int(refer[~isref].sum())}
        ag_rows.append(row)
        for c in range(9):
            mk = y == c
            routing.append({"thresholds": k, "class": CLASS_ORDER[c], "support": int(mk.sum()),
                            "refer": (reason[mk] == "REFER").mean() if mk.any() else np.nan,
                            "accept": (reason[mk] == "ACCEPT").mean() if mk.any() else np.nan,
                            "abstain": np.isin(reason[mk], ["DISAGREE", "LOW_CONF"]).mean() if mk.any() else np.nan,
                            "correct_if_accepted": (top[mk & acc] == c).mean() if (mk & acc).any() else np.nan})
        w(f"    {VN[k]:24s} tau {ag.tau:.2f}: coverage {pct(row['coverage'])} (referred {pct(row['referred'])}, "
          f"rule disagreement {pct(row['abstain_disagree'])}, low confidence {pct(row['abstain_lowconf'])}), "
          f"selective macro-F1 {row['selective_macro_f1']:.3f} ({row['errors_among_accepted']} errors in {n_acc} accepted), "
          f"confidence-only {row['confidence_only_selective_macro_f1']:.3f}; referral {row['referral_caught']}/"
          f"{row['referral_rows']} ({pct(row['referral_sensitivity'])}), false referrals {row['false_referrals']} ({pct(row['false_referral_rate'])})")
    pd.DataFrame(ag_rows).round(5).to_csv(out / "agent.csv", index=False)
    pd.DataFrame(routing).round(4).to_csv(out / "routing_per_class.csv", index=False)

    # conformal
    frozen = json.loads((ROOT / "results" / "phase4" / "frozen_phase6.json").read_text(encoding="utf-8"))
    conf_rows = []
    w("")
    w("  conformal abstention (frozen q, descriptive):")
    for a_key, q in frozen["conformal_q"].items():
        S = A.conformal_sets(P, q)
        hit = S[np.arange(len(y)), y]
        refer = agents["primary"].trigger_met(P, refer_rule)
        for k in VARIANTS:
            acc, lab = A.conformal_decide(S, masks[k], refer)
            conf_rows.append({"alpha": a_key, "thresholds": k, "q": q, "set_coverage": hit.mean(),
                              "mean_set_size": S.sum(1).mean(), "coverage": acc.mean(),
                              "selective_macro_f1": A.selective_macro_f1(y, lab, acc)})
        w(f"    alpha {a_key}: set coverage {pct(hit.mean())}, mean set size {S.sum(1).mean():.2f}; " + "; ".join(
            f"{VN[r['thresholds']]} coverage {pct(r['coverage'])}, selective macro-F1 {r['selective_macro_f1']:.3f}"
            for r in conf_rows if r["alpha"] == a_key))
        w("      set coverage per class: " + ", ".join(
            f"{CLASS_ORDER[c]} {pct(hit[y == c].mean()) if (y == c).any() else 'n/a'}" for c in range(9)))
    pd.DataFrame(conf_rows).round(5).to_csv(out / "conformal.csv", index=False)

    # ---- T2 McNemar
    w("")
    w("== T2. McNemar tests (continuity-corrected chi-square, Dietterich 1998; exact binomial p alongside) ==")
    mc_rows = []
    ok_ml, ok_cas = pred_ml == y, pred_cas == y
    for k in VARIANTS:
        ok_hy = pred_hy[k] == y
        pairs = [("forest vs hybrid", ok_ml, ok_hy), ("hybrid vs cascade", ok_hy, ok_cas)]
        if k == "primary":
            pairs.insert(1, ("forest vs cascade", ok_ml, ok_cas))
        res = [mcnemar(a, b) for _, a, b in pairs]
        adj = holm([r.p for r in res])
        family = "MAIN FAMILY (Holm over 3)" if k == "primary" else "variant (own Holm)"
        w(f"  {VN[k]}: {family}")
        for (name, _, _), r, pa in zip(pairs, res, adj):
            mc_rows.append({"thresholds": k, "pair": name, "a_right_b_wrong": r.b, "a_wrong_b_right": r.c,
                            "chi2": r.chi2, "p": r.p, "p_exact": r.p_exact, "p_holm": pa})
            w(f"    {name:18s} A right/B wrong {r.b:3d}, A wrong/B right {r.c:3d}, chi2 {r.chi2:7.3f}, p {r.p:.4g}, "
              f"exact p {r.p_exact:.4g}, Holm p {pa:.4g}")
    pd.DataFrame(mc_rows).round(6).to_csv(out / "mcnemar.csv", index=False)

    # ---- T3 acceptance targets
    w("")
    w("== T3. acceptance targets ==")
    agent = agents["primary"]
    vrows = v[FEATURES]
    with open(out / "audit_log.jsonl", "w", encoding="utf-8") as fh:
        for rid, r in df[FEATURES].iterrows():                   # every test row, invalid ones included
            rec = agent.run(r.to_dict(), f"{label}-{rid}")
            fh.write(rec["audit_json"] + "\n")
            audit_recs.append(rec)
    cons = [check_consistency(r) for r in audit_recs]
    n_cons = sum(1 for c in cons if not c)
    ref, ref_f = stress.referral_test(agent, v)
    ref.to_csv(out / "stress" / "referral.csv", index=False)
    met = ref[ref.trigger_met]
    bad, bad_f = stress.bad_input_test(agent, v)
    bad.to_csv(out / "stress" / "bad_input.csv", index=False)
    real_rej = all(r["decision"] == "REJECT" for r, ok in zip(audit_recs, valid) if not ok)
    t1 = len(met) > 0 and bool(met.referred.all())
    t2 = bool(bad.caught.all()) and real_rej
    t3 = n_cons == len(audit_recs)
    p4 = pd.read_csv(ROOT / "results" / "phase4" / "h1_summary.csv")
    h1_pass = int(p4[(p4.variant == "primary") & (p4.fset == "F-core")].supported.sum())
    p4b = pd.read_csv(ROOT / "results" / "phase4" / "h2_summary.csv")
    h2_pass = bool(p4b[(p4b.variant == "primary") & (p4b.floor.round(2) == 0.70)].passes.iloc[0])
    targets = [
        {"target": 1, "name": "Referral", "measured_on": "test stress cases",
         "result": f"{int(met.referred.sum())}/{len(met)} trigger-met stress cases returned REFER", "pass": t1},
        {"target": 2, "name": "Validation", "measured_on": "test rows with injected values",
         "result": f"{int(bad.caught.sum())}/{len(bad)} injected values caught; real invalid rows rejected: {real_rej}", "pass": t2},
        {"target": 3, "name": "Explanations", "measured_on": "test cases",
         "result": f"consistency check passed on {n_cons}/{len(audit_recs)} cases", "pass": t3},
        {"target": 4, "name": "H1", "measured_on": "development CV (Phase 4)",
         "result": f"{h1_pass} of 4 model types pass at the main thresholds", "pass": h1_pass == 4},
        {"target": 5, "name": "H2", "measured_on": "development CV (Phase 4)",
         "result": "beats random and confidence-only" if h2_pass else "beats random but not confidence-only", "pass": h2_pass},
    ]
    pd.DataFrame(targets).to_csv(out / "acceptance_targets.csv", index=False)
    for t in targets:
        w(f"  {t['target']}. {t['name']:12s} [{t['measured_on']}] {t['result']} -> {'PASS' if t['pass'] else 'FAIL'}")
    (out / "stress" / "failures.json").write_text(json.dumps({"referral": ref_f, "bad_input": bad_f,
                                                              "consistency": [c for c in cons if c]}, indent=1, default=str),
                                                  encoding="utf-8")

    # ---- T4 extras
    w("")
    w("== T4. per-class results (valid test rows) ==")
    pcd = pd.DataFrame(pc_rows)
    for cls in CLASS_ORDER:
        sub = pcd[pcd["class"] == cls]
        sup = int(sub.support.iloc[0])
        cav = f"  CAVEAT: only {sup} test row(s), one row changes recall by {100 / max(sup, 1):.0f} points" if sup <= SMALL else ""
        parts = [f"{r.system} {'' if r.thresholds == '-' else '(' + r.thresholds + ') '}P {r.precision:.2f} R {r.recall:.2f}"
                 for _, r in sub[(sub.thresholds.isin(['-', 'primary']))].iterrows()]
        w(f"  {cls:32s} n={sup:3d}  " + " | ".join(parts) + cav)
    lat = pd.DataFrame([r["latency_ms"] for r in audit_recs])
    lat["total"] = lat.fillna(0).sum(axis=1)
    lat_s = lat.agg(["median", lambda s: s.quantile(0.95)]).T
    lat_s.columns = ["median_ms", "p95_ms"]
    lat_s.round(3).to_csv(out / "latency.csv")
    w("")
    w("  latency per case (ms): " + ", ".join(f"{s} median {r.median_ms:.2f} / p95 {r.p95_ms:.2f}" for s, r in lat_s.iterrows()))
    nz = stress.noise_test(agent, v)
    nz.round(5).to_csv(out / "stress" / "noise.csv", index=False)
    w("  noise stability: " + "; ".join(f"SD {r.noise_sd:.0%}: same decision and label {pct(r.same_decision_and_label)}"
                                        for _, r in nz.iterrows()))
    w("")
    w(f"run time {time.perf_counter() - t_all:.0f} s")
    (out / "final_test_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--final", action="store_true", help="the single real run on the test rows")
    g.add_argument("--rehearsal", metavar="OUT_DIR", help="same code on development rows (in-sample, meaningless)")
    a = ap.parse_args()
    if a.final:
        guards()
        df = load_test(i_understand_this_is_phase6=True)
        run(df, OUT_FINAL, "test")
    else:
        out = Path(a.rehearsal).resolve()
        if out == OUT_FINAL.resolve() or ROOT / "results" in out.parents or out == ROOT / "results":
            sys.exit("a rehearsal must not write into results/")
        run(load_dev(clean=True), out, "REHEARSAL-dev-in-sample")


if __name__ == "__main__":
    main()
