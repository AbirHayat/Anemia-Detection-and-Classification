"""Phase 3 evaluation, exactly as pre-registered on 14 Nov 2026, before any result was opened.

Research prototype, not a diagnostic tool. Development rows only. Test rows are never read.

Reads the out-of-fold predictions of src/phase3_ml_only_cv.py and src/phase3_h3_h4_cv.py and writes
results/phase3/evaluation.txt plus CSV tables and one figure.

P1 calibrated predictions (argmax) are primary, uncalibrated secondary
P2 imbalance arm per model type = best mean F-core macro-F1 (tie within 0.001 -> class weights)
P3 H2 model = best F-core model type with its arm
P4 H3: F-core+idx vs F-core, one-sided corrected test on mean(IDA recall, OMA recall), Holm over 4;
   F-all secondary with its own Holm
P5 H4: probes vs majority dummy on balanced accuracy, one-sided, Holm over 2
P6 F-core vs F-all per model type, two-sided, Holm over 4 (descriptive)
P7 errors near the cut-offs (bands below), descriptive
Also: class weights vs SMOTE (two-sided, Holm over 4) and metrics split by filled vs other rows.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import CLASS_ORDER, ROOT, load_dev  # noqa: E402
from metrics import confusion, per_class, scores  # noqa: E402
from stats_tests import corrected_repeated_kfold_t, holm  # noqa: E402

P3 = ROOT / "results" / "phase3"
OUT = P3                 # set in main(): results/phase3 (original) or results/phase3/amended
VERSION = "original"
K, R, RATIO = 5, 5, 0.25
TYPES = ["logreg", "tree", "forest", "hgb"]
IDA, OMA = CLASS_ORDER.index("Iron deficiency anemia"), CLASS_ORDER.index("Other microcytic anemia")
BANDS = {"HGB": [(12.5, 13.5)], "PLT": [(135, 165)], "WBC": [(9.5, 11.5)], "MCV": [(78, 82), (98, 102)],
         "MCHC": [(31.4, 32.5)]}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def load_oof() -> pd.DataFrame:
    if VERSION == "original":
        a = pd.read_csv(P3 / "ml_only" / "oof_predictions.csv.gz")
        b = pd.read_csv(P3 / "h3_indices" / "oof_predictions.csv.gz")
    else:
        a = pd.read_csv(P3 / "amended" / "oof_ml_only.csv.gz")
        b = pd.read_csv(P3 / "amended" / "oof_h3.csv.gz")
    df = pd.concat([a, b], ignore_index=True)
    for tag in ["cal", "un"]:
        df[f"pred_{tag}"] = df[[f"p_{tag}_{j}" for j in range(9)]].to_numpy().argmax(axis=1)
    return df


def fold_metrics(df: pd.DataFrame, pred: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(["fset", "model", "imb", "rep", "fold"], sort=True):
        s = scores(g["y"], g[pred])
        pc = per_class(g["y"], g[pred])
        s["micro_pair_recall"] = (pc.loc[IDA, "recall"] + pc.loc[OMA, "recall"]) / 2
        rows.append(dict(zip(["fset", "model", "imb", "rep", "fold"], key), **s))
    return pd.DataFrame(rows)


def diffs(fm: pd.DataFrame, a: dict, b: dict, metric: str) -> np.ndarray:
    def sel(d):
        m = np.ones(len(fm), bool)
        for k, v in d.items():
            m &= fm[k].to_numpy() == v
        return fm[m].sort_values(["rep", "fold"])[metric].to_numpy()
    return sel(a) - sel(b)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def main() -> None:
    global OUT, VERSION
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["original", "amended"], default="original")
    VERSION = ap.parse_args().version
    OUT = P3 if VERSION == "original" else P3 / "amended"
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_dev(clean=True)
    oof = load_oof()
    L: list[str] = []
    w = L.append
    w("Phase 3 evaluation (pre-registered on 14 Nov 2026, before any result was opened). Development rows only, test rows not read.")
    if VERSION == "amended":
        w("VERSION: AMENDED (post-results amendment of 14 Nov 2026): every model calibrated with sigmoid on its predicted")
        w("probabilities instead of raw decision scores. Same folds, seeds, hyperparameters and pre-registered rules.")
    w(f"outer folds {K} x {R} = {K * R}, corrected test n2/n1 = {RATIO}, df = {K * R - 1}. Mean ± std over the 25 outer folds.")
    fm = {tag: fold_metrics(oof, f"pred_{tag}") for tag in ["cal", "un"]}
    for tag in fm:
        fm[tag].to_csv(OUT / f"fold_metrics_{tag}.csv", index=False)

    # ---- summary tables ----
    for tag, title in [("cal", "PRIMARY (calibrated)"), ("un", "SECONDARY (uncalibrated)")]:
        g = fm[tag].groupby(["fset", "model", "imb"])
        summ = g[["macro_f1", "balanced_accuracy", "mcc"]].agg(["mean", "std"])
        summ.round(4).to_csv(OUT / f"summary_metrics_{tag}.csv")
        w("")
        w(f"== A. all configurations, {title} ==")
        w(f"  {'feature set':11s} {'model':7s} {'arm':8s} {'macro-F1':>16s} {'balanced acc':>16s} {'MCC':>16s}")
        for (fs, m, i), r in summ.iterrows():
            w(f"  {fs:11s} {m:7s} {i:8s} {fmt(r[('macro_f1', 'mean')], r[('macro_f1', 'std')]):>16s} "
              f"{fmt(r[('balanced_accuracy', 'mean')], r[('balanced_accuracy', 'std')]):>16s} "
              f"{fmt(r[('mcc', 'mean')], r[('mcc', 'std')]):>16s}")
    f = fm["cal"]

    # ---- P2: imbalance arm ----
    w("")
    w("== P2. imbalance arm per model type (best mean F-core macro-F1, tie within 0.001 -> class weights) ==")
    arm, pvals, lines = {}, [], []
    for m in TYPES:
        mw = f[(f.fset == "F-core") & (f.model == m) & (f.imb == "weights")].macro_f1.mean()
        ms = f[(f.fset == "F-core") & (f.model == m) & (f.imb == "smote")].macro_f1.mean()
        arm[m] = "smote" if ms > mw + 0.001 else "weights"
        t = corrected_repeated_kfold_t(diffs(f, {"fset": "F-core", "model": m, "imb": "weights"},
                                             {"fset": "F-core", "model": m, "imb": "smote"}, "macro_f1"), K, R, RATIO)
        pvals.append(t.p)
        lines.append((m, mw, ms, t))
    for (m, mw, ms, t), p_adj in zip(lines, holm(pvals)):
        w(f"  {m:7s} weights {mw:.4f}  smote {ms:.4f}  -> chosen: {arm[m]:8s} | weights - smote = {t.mean:+.4f}, "
          f"t = {t.t:.2f}, p = {t.p:.4f}, Holm p = {p_adj:.4f}")

    def chosen(fs):
        return f[(f.fset == fs) & (((f.model == "dummy") & (f.imb == "none")) |
                                   f.apply(lambda r: r.model in arm and r.imb == arm[r.model], axis=1))]

    # ---- P3: H2 model ----
    core = chosen("F-core").groupby("model").macro_f1.mean().drop("dummy")
    h2 = core.idxmax()
    w("")
    w("== P3. H2 model choice (best mean F-core macro-F1 with the chosen arm, calibrated) ==")
    for m, v in core.sort_values(ascending=False).items():
        w(f"  {m:7s} ({arm[m]}) {v:.4f}")
    w(f"  -> H2 model: {h2} with {arm[h2]}")

    # ---- per-class and confusion for chosen configurations ----
    w("")
    w("== B. per-class recall and precision, chosen arms, calibrated, pooled over all 5 repeats ==")
    pc_rows = []
    for fs in ["F-core", "F-all"]:
        for m in ["dummy"] + TYPES:
            imb = "none" if m == "dummy" else arm[m]
            g = oof[(oof.fset == fs) & (oof.model == m) & (oof.imb == imb)]
            pc = per_class(g["y"], g["pred_cal"])
            pc.insert(0, "class", CLASS_ORDER)
            pc.insert(0, "model", m)
            pc.insert(0, "fset", fs)
            pc_rows.append(pc)
            np.savetxt(OUT / f"confusion_{fs}_{m}.csv", confusion(g["y"], g["pred_cal"]), fmt="%d", delimiter=",",
                       header=",".join(CLASS_ORDER), comments="")
            if m != "dummy":
                w(f"  {fs} {m}: recall " + ", ".join(f"{c.split()[0][:6]} {v:.2f}" for c, v in zip(CLASS_ORDER, pc.recall)))
    pd.concat(pc_rows).round(4).to_csv(OUT / "per_class_chosen.csv", index=False)
    w("  (confusion matrices: confusion_<feature set>_<model>.csv, rows = true class, columns = predicted, 5 repeats pooled)")

    # ---- P6: F-core vs F-all ----
    w("")
    w("== P6. F-all minus F-core, macro-F1, chosen arms (two-sided, Holm over 4, descriptive) ==")
    res = [corrected_repeated_kfold_t(diffs(f, {"fset": "F-all", "model": m, "imb": arm[m]},
                                            {"fset": "F-core", "model": m, "imb": arm[m]}, "macro_f1"), K, R, RATIO)
           for m in TYPES]
    for m, t, pa in zip(TYPES, res, holm([t.p for t in res])):
        w(f"  {m:7s} mean diff {t.mean:+.4f}, t = {t.t:.2f}, p = {t.p:.4f}, Holm p = {pa:.4f}")

    # ---- P4: H3 ----
    for fs, lab in [("F-core", "PRIMARY"), ("F-all", "SECONDARY (descriptive)")]:
        res = [corrected_repeated_kfold_t(diffs(f, {"fset": fs + "+idx", "model": m, "imb": arm[m]},
                                                {"fset": fs, "model": m, "imb": arm[m]}, "micro_pair_recall"),
                                          K, R, RATIO, "greater") for m in TYPES]
        w("")
        w(f"== P4. H3 {lab}: {fs}+idx minus {fs}, mean of iron deficiency and other microcytic recall "
          "(one-sided, Holm over 4) ==")
        for m, t, pa in zip(TYPES, res, holm([t.p for t in res])):
            base = f[(f.fset == fs) & (f.model == m) & (f.imb == arm[m])].micro_pair_recall.mean()
            w(f"  {m:7s} without {base:.4f}, diff {t.mean:+.4f}, t = {t.t:.2f}, p = {t.p:.4f}, Holm p = {pa:.4f}"
              f" -> {'supported' if pa < 0.05 else 'not supported'}")

    # ---- P5: H4 ----
    pr = pd.read_csv(P3 / "h4_probes" / "oof_predictions.csv.gz")
    w("")
    w("== P5. H4 probes vs majority-class dummy, balanced accuracy (one-sided, Holm over 2) ==")
    res, desc = [], []
    for probe in ["probe1_fill_indicator", "probe2_fcore_to_filled"]:
        ba = {}
        for kind in ["probe", "dummy"]:
            g = pr[(pr.probe == probe) & (pr.kind == kind)]
            ba[kind] = g.groupby(["rep", "fold"]).apply(lambda x: scores(x["y"], x["pred"], labels=sorted(x["y"].unique()))
                                                         ["balanced_accuracy"], include_groups=False).sort_index()
        t = corrected_repeated_kfold_t(ba["probe"].to_numpy() - ba["dummy"].to_numpy(), K, R, RATIO, "greater")
        res.append(t)
        desc.append((probe, ba["probe"].mean(), ba["probe"].std(), ba["dummy"].mean()))
    for (probe, m, s, d), t, pa in zip(desc, res, holm([t.p for t in res])):
        w(f"  {probe}: probe {fmt(m, s)}, dummy {d:.3f}, diff {t.mean:+.4f}, t = {t.t:.2f}, p = {t.p:.2e}, "
          f"Holm p = {pa:.2e} -> {'shortcut present' if pa < 0.05 else 'not shown'}")

    # ---- filled vs other ----
    group_f1 = {}
    filled = dev["fill_count"].eq(6)
    fl = oof["raw_row"].map(filled)
    w("")
    w("== C. macro-F1 within mean-filled rows and within other rows (chosen arms, calibrated, classes present "
      "in each group and fold) ==")
    for fs in ["F-core", "F-all"]:
        for m in ["dummy"] + TYPES:
            imb = "none" if m == "dummy" else arm[m]
            g = oof[(oof.fset == fs) & (oof.model == m) & (oof.imb == imb)].assign(filled=fl)
            out = {}
            for grp, gg in g.groupby("filled"):
                v = gg.groupby(["rep", "fold"]).apply(
                    lambda x: scores(x["y"], x["pred_cal"], labels=sorted(x["y"].unique()))["macro_f1"],
                    include_groups=False)
                out["filled" if grp else "other"] = (v.mean(), v.std())
            group_f1[(fs, m)] = out
            w(f"  {fs:7s} {m:7s} filled {fmt(*out['filled'])}   other {fmt(*out['other'])}")

    # ---- P7: near cut-offs ----
    near = pd.Series(False, index=dev.index)
    for c, bands in BANDS.items():
        for lo, hi in bands:
            near |= dev[c].between(lo, hi)
    nr = oof["raw_row"].map(near)
    w("")
    w(f"== P7. errors near the cut-offs (bands {BANDS}) ==")
    w(f"  rows near a cut-off: {int(near.sum())} of {len(dev)} ({100 * near.mean():.1f}%)")
    for fs in ["F-core", "F-all"]:
        for m in TYPES:
            g = oof[(oof.fset == fs) & (oof.model == m) & (oof.imb == arm[m])]
            err = g["pred_cal"] != g["y"]
            nn = nr[g.index]
            w(f"  {fs:7s} {m:7s} error rate near {100 * err[nn].mean():.1f}% vs elsewhere {100 * err[~nn].mean():.1f}%; "
              f"share of errors near {100 * err[nn].sum() / max(err.sum(), 1):.1f}% (5 repeats pooled)")

    (OUT / "evaluation.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # ---- main results table for the paper (all rows + filled + other, chosen arms, calibrated) ----
    nice = {"dummy": "Majority dummy", "logreg": "Logistic regression", "tree": "Decision tree",
            "forest": "Random forest", "hgb": "Gradient boosting"}
    T = ["| Feature set | Model (arm) | Macro-F1 all rows | Balanced acc. all rows | MCC all rows | "
         "Macro-F1 mean-filled rows | Macro-F1 other rows |", "|---|---|---|---|---|---|---|"]
    for fs in ["F-core", "F-all"]:
        cs = chosen(fs).groupby("model")[["macro_f1", "balanced_accuracy", "mcc"]].agg(["mean", "std"])
        for m in ["dummy"] + TYPES:
            r = cs.loc[m]
            a = "" if m == "dummy" else (" (class weights)" if arm[m] == "weights" else " (SMOTE)")
            gf = group_f1[(fs, m)]
            T.append(f"| {fs} | {nice[m]}{a} | {fmt(r[('macro_f1', 'mean')], r[('macro_f1', 'std')])} | "
                     f"{fmt(r[('balanced_accuracy', 'mean')], r[('balanced_accuracy', 'std')])} | "
                     f"{fmt(r[('mcc', 'mean')], r[('mcc', 'std')])} | {fmt(*gf['filled'])} | {fmt(*gf['other'])} |")
    head = ("Main results, development CV (25 outer folds, mean ± std), calibrated predictions (pre-registered "
            "primary). Generated by src/phase3_evaluate.py. The filled and other columns use the classes present "
            "in each group and fold.")
    if VERSION == "amended":
        head = head.replace("calibrated predictions (pre-registered primary)",
                            "AMENDED calibration (sigmoid on predicted probabilities, post-results amendment of "
                            "14 Nov 2026, same pre-registered rules)").replace(
            "src/phase3_evaluate.py.", "src/phase3_evaluate.py --version amended.")
    (OUT / "main_results_table.md").write_text(head + "\n\n" + "\n".join(T) + "\n", encoding="utf-8")

    # ---- figure: macro-F1 by model type, F-core vs F-all (chosen arms) ----
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    names = ["dummy"] + TYPES
    for j, (fs, col, off) in enumerate([("F-core", BLUE, -0.12), ("F-all", ORANGE, 0.12)]):
        sub = chosen(fs).groupby("model").macro_f1.agg(["mean", "std"]).reindex(names)
        y = np.arange(len(names)) + off
        ax.errorbar(sub["mean"], y, xerr=sub["std"], fmt="o", color=col, ms=4, elinewidth=1, capsize=2, label=fs)
    ax.set_yticks(range(len(names)), ["majority dummy", "logistic reg.", "decision tree", "random forest", "grad. boosting"])
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1, mean ± std over 25 folds", color=MUTED, fontsize=7)
    ax.set_xlim(0, 1)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "fig_macro_f1_by_model.png")


if __name__ == "__main__":
    main()
