"""Phase 5: explanation and audit layer, run on development rows only.

Research prototype, not a diagnostic tool. Test rows are never read.

IMPORTANT: the final models were fitted on these same 963 development rows, so every decision below is
IN-SAMPLE. Phase 5 tests the machinery (explanations, audit log, consistency, latency, stress behaviour), not
accuracy. Accuracy is measured once, on the test set, in Phase 6.

Steps
  1 global explanations: one explainer per model type, mean |SHAP| per feature, additivity check,
    logistic-regression coefficients
  2 the agent on all development rows (main thresholds, frozen Phase 4 settings): JSON audit log,
    automatic consistency check, per-stage latency (median and 95th percentile)
  3 determinism: 40 cases run twice must give identical explanations
  4 stress tests (src/stress.py): noise, bad input, edges, referral (a dry run of acceptance targets 1 and 2)
  5 manual review sheet: 40 random development cases, rubric and blank rating columns (filled by the student)

Outputs: results/phase5/ (phase5_report.txt, audit_log_dev.jsonl, latency.csv, global_shap.csv,
         logreg_coefficients.csv, stress/*.csv, review/review_sheet.md, review/review_ratings.csv, figure)
Run:     python src/phase5_run.py
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import explain as E  # noqa: E402
import stress  # noqa: E402
from agent import Agent, check_consistency  # noqa: E402
from data import CLASS_ORDER, F_CORE, FEATURES, ROOT, SEED, TARGET, load_dev  # noqa: E402

OUT = ROOT / "results" / "phase5"
MODELS = OUT / "models"
ARMS = {"logreg": "smote", "tree": "weights", "forest": "smote", "hgb": "smote"}
NAMES = {"logreg": "Logistic regression", "tree": "Decision tree", "forest": "Random forest", "hgb": "Gradient boosting"}
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def pct(x):
    return f"{100 * x:.1f}%"


def main() -> None:
    warnings.filterwarnings("ignore")
    t_all = time.perf_counter()
    for d in [OUT, OUT / "stress", OUT / "review", OUT / "figures"]:
        d.mkdir(parents=True, exist_ok=True)
    dev = load_dev(clean=True)
    X = dev[F_CORE].to_numpy(float)
    y = dev[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    L: list[str] = []
    w = L.append
    w("Phase 5 report: explanation and audit layer. Development rows only, test rows not read. Research prototype.")
    w(f"rows {len(dev)}. The final models were fitted on these rows, so every decision here is IN-SAMPLE: this phase")
    w("tests the machinery, not accuracy. Frozen settings: results/phase4/frozen_phase6.json.")

    # ---------------------------------------------------------------- 1 global explanations per model type
    w("")
    w("== 1. global explanations, one explainer per model type (mean |SHAP| over rows and classes, F-core) ==")
    glob, add_rows = [], []
    for m, arm in ARMS.items():
        cal = joblib.load(MODELS / f"{m}_{arm}_Fcore.joblib")
        ex = E.make_explainer(m, cal, background_X=X)
        t0 = time.perf_counter()
        sv = E.shap_values(ex, cal, X)
        secs = time.perf_counter() - t0
        pipe = E.underlying(cal)
        Z = pipe.named_steps["scale"].transform(X)
        clf = pipe.named_steps["clf"]
        recon = E.expected_values(ex, cal)[None, :] + sv.sum(axis=1)
        if m in ("forest", "tree"):
            target = np.zeros_like(recon)
            target[:, clf.classes_] = clf.predict_proba(Z)
        elif m == "logreg":
            target = np.zeros_like(recon)
            target[:, clf.classes_] = clf.decision_function(Z)
        else:
            target = np.zeros_like(recon)
            target[:, clf.classes_] = clf._raw_predict(Z)          # boosting raw log-odds (private but stable API)
        err = float(np.abs(recon - target).max())
        add_rows.append({"model": m, "explainer": type(ex).__name__, "output_space": E.OUTPUT_SPACE[m],
                         "max_additivity_error": err, "seconds_for_963_rows": round(secs, 2)})
        imp = np.abs(sv).mean(axis=(0, 2))
        for f, v in zip(F_CORE, imp):
            glob.append({"model": m, "feature": f, "mean_abs_shap": v})
        rank = ", ".join(f"{F_CORE[i]} {imp[i]:.3g}" for i in np.argsort(-imp))
        w(f"  {NAMES[m]:20s} {type(ex).__name__:16s} ({E.OUTPUT_SPACE[m]}): {rank}")
        w(f"  {'':20s} additivity check (base + sum of SHAP = model output): max error {err:.2e}, {secs:.1f} s")
        if m == "logreg":
            coef = pd.DataFrame(clf.coef_, index=[CLASS_ORDER[c] for c in clf.classes_], columns=F_CORE)
            coef.round(4).to_csv(OUT / "logreg_coefficients.csv")
    w("  dummy: no explainer (it ignores the inputs)")
    gdf = pd.DataFrame(glob)
    gdf.round(6).to_csv(OUT / "global_shap.csv", index=False)
    pd.DataFrame(add_rows).to_csv(OUT / "shap_additivity.csv", index=False)
    w("  logistic-regression coefficients (scaled inputs, per class): results/phase5/logreg_coefficients.csv")
    plot_global(gdf)

    # ---------------------------------------------------------------- 2 the agent on every development row
    agent = Agent("primary")
    recs = []
    with open(OUT / "audit_log_dev.jsonl", "w", encoding="utf-8") as fh:
        for rid, r in dev[FEATURES].iterrows():
            rec = agent.run(r.to_dict(), f"dev-{rid}")
            fh.write(rec["audit_json"] + "\n")
            recs.append(rec)
    cons = {r["case_id"]: check_consistency(r) for r in recs}
    n_bad = sum(1 for v in cons.values() if v)
    lat = pd.DataFrame([r["latency_ms"] for r in recs])
    lat["total"] = lat[[c for c in lat.columns]].fillna(0).sum(axis=1)
    lat_s = lat.agg(["median", lambda s: s.quantile(0.95), "max"]).T
    lat_s.columns = ["median_ms", "p95_ms", "max_ms"]
    lat_s.round(3).to_csv(OUT / "latency.csv")
    dec = pd.Series([r["decision"] for r in recs])
    reasons = pd.Series([r["reason"] for r in recs])
    w("")
    w("== 2. the agent on all development rows (main thresholds, tau and REFER trigger frozen in Phase 4) ==")
    w(f"  audit log: results/phase5/audit_log_dev.jsonl ({len(recs)} JSON records, one per case)")
    w(f"  consistency check (rule trace and SHAP refer to the same case and class as the decision, summary repeats")
    w(f"  the stored label and probability, caveat and disclaimer present): {len(recs) - n_bad}/{len(recs)} pass")
    w(f"  decisions (IN-SAMPLE, not an accuracy result): " + ", ".join(f"{k} {v}" for k, v in dec.value_counts().items())
      + " | reasons: " + ", ".join(f"{k} {v}" for k, v in reasons.value_counts().items()))
    w("  latency per case in ms (desktop PC, one case at a time):")
    for st, row in lat_s.iterrows():
        w(f"    {st:12s} median {row.median_ms:8.3f}   95th percentile {row.p95_ms:8.3f}   max {row.max_ms:8.3f}")

    # ---------------------------------------------------------------- 3 determinism
    sample = dev[FEATURES].sample(40, random_state=SEED)
    same = 0
    for rid, r in sample.iterrows():
        a, b = agent.run(r.to_dict(), f"dev-{rid}"), agent.run(r.to_dict(), f"dev-{rid}")
        same += int(a["summary"] == b["summary"] and a["shap"] == b["shap"] and a["rules"] == b["rules"])
    w("")
    w(f"== 3. determinism: the same case gives the same explanation: {same}/40 identical on a second run ==")

    # ---------------------------------------------------------------- 4 stress tests
    w("")
    w("== 4. stress tests on development rows (dry run of the suite; targets 1 and 2 are measured on the test set) ==")
    t0 = time.perf_counter()
    nz = stress.noise_test(agent, dev)
    nz.round(5).to_csv(OUT / "stress" / "noise.csv", index=False)
    for _, r in nz.iterrows():
        w(f"  noise SD {r.noise_sd:.0%}: {int(r.cases)} cases, same decision {pct(r.same_decision)}, same decision and label "
          f"{pct(r.same_decision_and_label)} (ACCEPT {pct(r.same_decision_if_ACCEPT)}, ABSTAIN {pct(r.same_decision_if_ABSTAIN)}, "
          f"REFER {pct(r.same_decision_if_REFER)}), rejected after noise {pct(r.rejected_after_noise)}")
    bad, bad_f = stress.bad_input_test(agent, dev, n_base=100)
    bad.to_csv(OUT / "stress" / "bad_input.csv", index=False)
    w(f"  bad input: {int(bad.caught.sum())}/{len(bad)} injected values caught as REJECT naming the column "
      f"({pct(bad.caught.mean())}), consistency {int(bad.consistent.sum())}/{len(bad)}; failures {len(bad_f)}")
    w("    per corruption: " + ", ".join(f"{k} {int(g.caught.sum())}/{len(g)}" for k, g in bad.groupby("corruption")))
    edg, edg_f = stress.edge_test(agent, dev, n_base=20)
    edg.to_csv(OUT / "stress" / "edges.csv", index=False)
    for kind, g in edg.groupby("kind"):
        w(f"  edges, {kind}: {int(g.ok.sum())}/{len(g)} as defined")
    ref, ref_f = stress.referral_test(agent, dev)
    ref.to_csv(OUT / "stress" / "referral.csv", index=False)
    met = ref[ref.trigger_met]
    w(f"  referral: {len(ref)} stress cases ({int(ref.valid.sum())} valid), {len(met)} meet the {agent.trigger} trigger, "
      f"{int(met.referred.sum())}/{len(met)} returned REFER ({pct(met.referred.mean()) if len(met) else 'n/a'}); "
      f"referred without the trigger: {int((ref.referred & ~ref.trigger_met).sum())}")
    (OUT / "stress" / "failures.json").write_text(json.dumps({"bad_input": bad_f, "edges": edg_f, "referral": ref_f},
                                                             indent=1, default=str), encoding="utf-8")
    w(f"  stress suite time {time.perf_counter() - t0:.0f} s. Failures listed in results/phase5/stress/failures.json")

    # ---------------------------------------------------------------- 5 manual review sheet
    make_review_sheet(dev, recs)
    w("")
    w("== 5. manual review: results/phase5/review/review_sheet.md and review_ratings.csv (40 random development cases,")
    w("   seed 42). NOT YET DONE: the ratings are filled in by the student, never by the AI. ==")
    w("")
    w(f"run time {time.perf_counter() - t_all:.0f} s")
    (OUT / "phase5_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def make_review_sheet(dev: pd.DataFrame, recs: list[dict]) -> None:
    by_id = {r["case_id"]: r for r in recs}
    ids = dev.sample(40, random_state=SEED).index
    L = ["# Manual review of 40 explanations (Phase 5)", "",
         "Research prototype, not a diagnostic tool. **To be filled in by the student only.** The AI prepared this sheet "
         "and must not rate any case.", "",
         "Cases: 40 development rows drawn at random (seed 42) from the 963 clean development rows. The final model was "
         "trained on these rows, so the decisions are in-sample. Rate the explanation, not whether the diagnosis is right. "
         "The true label is shown last, for context only.", "",
         "## Rubric (score each case 1, 2 or 3)", "",
         "| Criterion | 3 | 2 | 1 |", "|---|---|---|---|",
         "| Correctness | every statement matches the input values and the stored rules, probability and features | one minor imprecision that does not mislead | at least one statement is wrong or misleading |",
         "| Completeness | says the decision, the reason (rules and model), the confidence and the calibration caveat | one of these is missing or vague | a reader could not tell why the decision was made |",
         "| Clarity | a non-specialist can follow it in one reading | needs a second reading or has unexplained jargon | confusing |",
         "", "Write your scores in `review_ratings.csv` (one row per case, columns already there) and a short comment "
         "where a score is below 3. Take about 3 minutes per case.", ""]
    rows = []
    for k, rid in enumerate(ids, 1):
        r = by_id[f"dev-{rid}"]
        vals = ", ".join(f"{c} {r['inputs'][c]:g}" for c in F_CORE)
        L += [f"## Case {k}: `{r['case_id']}`", "", f"**Inputs (F-core):** {vals}", "",
              f"**Decision:** {r['decision']} ({r['reason']})", "", f"**Summary shown to the user:** {r['summary']}", ""]
        if r["rules"]:
            fired = [s for s in r["rules"]["trace"] if s["rule"].startswith("X")]
            found = [s["text"] for s in r["rules"]["trace"] if s["rule"].startswith("F") and any(
                v is True for v in s["result"].values())]
            L += [f"**Rule findings (true):** {'; '.join(found) if found else 'none'}".replace(";", ","), "",
                  f"**Exclusion rules fired:** {', '.join(s['rule'] for s in fired) if fired else 'none'}. "
                  f"**Labels allowed:** {', '.join(r['rules']['compatible'])}", ""]
            top3 = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])[:3]
            L += ["**Top 3 calibrated probabilities:** " + ", ".join(f"{a} {b:.3f}" for a, b in top3), ""]
            if r["shap"]:
                L += ["**SHAP top features for " + r["shap"]["explained_class"] + ":** " + ", ".join(
                    f"{t['feature']} {t['value']:g} (effect {t['effect']:+.3f})" for t in r["shap"]["top_features"]), ""]
        L += [f"*True label (context only):* {dev.loc[rid, TARGET]}", ""]
        rows.append({"case": k, "case_id": r["case_id"], "decision": r["decision"], "correctness_1to3": "",
                     "completeness_1to3": "", "clarity_1to3": "", "comment": ""})
    (OUT / "review" / "review_sheet.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "review" / "review_ratings.csv", index=False)


def plot_global(gdf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), dpi=300)
    for ax, m in zip(axes, ARMS):
        s = gdf[gdf.model == m].set_index("feature").mean_abs_shap.sort_values()
        ax.barh(range(len(s)), s.to_numpy(), color=BLUE, height=0.6)
        ax.set_yticks(range(len(s)), s.index, fontsize=6)
        ax.set_title(f"{NAMES[m]}\n({E.OUTPUT_SPACE[m]})", fontsize=6.5, color=INK)
        ax.tick_params(colors=MUTED, labelsize=6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
    fig.supxlabel("Mean |SHAP| over development rows and classes (scales differ by output space)", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_global_shap.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
