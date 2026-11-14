"""Phase 1: data report on the CLEANED DEVELOPMENT rows (963 rows).

Research prototype, not a diagnostic tool. Test rows are never read.

What it checks
  A. Per-feature summary (min, 1st percentile, median, 99th percentile, max, whole-number share).
  B. Column meaning: do the absolute counts match WBC x percentage / 100? Does PCT behave like
     plateletcrit (tracks PLT, implied mean platelet volume PCT x 10000 / PLT)?
  C. Textbook CBC relations: MCV = HCT x 10 / RBC, MCH = HGB x 10 / RBC, MCHC = HGB x 100 / HCT.
     The relations are general textbook knowledge, UNVERIFIED until Phase 2 cites a source.
  D. Rounding signature in F-core (whole-number shares, MCHC = 32.0) for filled vs other rows.
  E. Per-class medians of the key rule-engine inputs (HGB, MCV, MCHC, PLT, WBC).
  F. Highly correlated feature pairs (Spearman |rho| >= 0.9) among non-filled rows.
  Figures: class counts, filled share by class, key features by class.

"Filled" = fill count 6 from results/phase1/dev_rows.csv (descriptive Phase 1 flag).
Unit statements are inferences from value ranges, marked unverified.

Outputs: results/phase1/data_report.txt, feature_summary.csv, class_medians.csv, figures/*.png
Run:  python src/phase1_clean.py  then  python src/phase1_data_report.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import CLASS_ORDER, FEATURES, ROOT, TARGET, load_dev  # noqa: E402

OUT_DIR = ROOT / "results" / "phase1"
FIG_DIR = OUT_DIR / "figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {
    "Healthy": "Healthy",
    "Normocytic hypochromic anemia": "Normo. hypochromic",
    "Normocytic normochromic anemia": "Normo. normochromic",
    "Iron deficiency anemia": "Iron deficiency",
    "Thrombocytopenia": "Thrombocytopenia",
    "Other microcytic anemia": "Other microcytic",
    "Leukemia": "Leukemia",
    "Macrocytic anemia": "Macrocytic",
    "Leukemia with thrombocytopenia": "Leuk. + thrombocyt.",
}


def rel_err(obs: pd.Series, calc: pd.Series) -> pd.Series:
    return (obs - calc).abs() / obs.abs()


def describe_relation(w, name: str, obs: pd.Series, calc: pd.Series, filled: pd.Series) -> None:
    e = rel_err(obs, calc)
    for lab, m in [("other rows", ~filled), ("filled rows", filled)]:
        x = e[m]
        w(f"  {name} [{lab}, n={int(m.sum())}]: median relative error {100 * x.median():.2f}%, "
          f"within 2%: {100 * (x <= 0.02).mean():.1f}%, within 5%: {100 * (x <= 0.05).mean():.1f}%")


def style(ax) -> None:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dev(clean=True)
    filled = df["fill_count"] == 6
    lines: list[str] = []
    w = lines.append
    w(f"cleaned development rows: {len(df)} (filled rows: {int(filled.sum())}, other rows: {int((~filled).sum())})")
    w("test rows: not read")

    # A. feature summary
    q = df[FEATURES].quantile([0.01, 0.5, 0.99]).T
    summ = pd.DataFrame({
        "min": df[FEATURES].min(), "p01": q[0.01], "median": q[0.5], "p99": q[0.99],
        "max": df[FEATURES].max(),
        "whole_number_share_other_rows": [(df.loc[~filled, c] % 1 == 0).mean() for c in FEATURES],
        "whole_number_share_filled_rows": [(df.loc[filled, c] % 1 == 0).mean() for c in FEATURES],
    })
    summ.round(4).to_csv(OUT_DIR / "feature_summary.csv")
    w("")
    w("== A. feature summary (cleaned development rows; full table in feature_summary.csv) ==")
    for c in FEATURES:
        r = summ.loc[c]
        w(f"  {c}: min {r['min']:g}, p01 {r['p01']:.3g}, median {r['median']:.4g}, p99 {r['p99']:.4g}, max {r['max']:g}")

    # B. column meaning
    o = df[~filled]
    w("")
    w("== B. column meaning (other rows only, since filled rows hold constants) ==")
    for name, obs, calc in [("LYMn vs WBC x LYMp / 100", o["LYMn"], o["WBC"] * o["LYMp"] / 100),
                            ("NEUTn vs WBC x NEUTp / 100", o["NEUTn"], o["WBC"] * o["NEUTp"] / 100)]:
        e = rel_err(obs, calc)
        w(f"  {name} [n={len(o)}]: median relative error {100 * e.median():.2f}%, "
          f"within 2%: {100 * (e <= 0.02).mean():.1f}%, within 5%: {100 * (e <= 0.05).mean():.1f}%")
    w(f"  LYMp + NEUTp: median {(o['LYMp'] + o['NEUTp']).median():.1f}, share above 100: "
      f"{100 * ((o['LYMp'] + o['NEUTp']) > 100).mean():.1f}%")
    rho = o[["PCT", "PLT"]].corr(method="spearman").iloc[0, 1]
    mpv = o["PCT"] * 10000 / o["PLT"]
    w(f"  PCT vs PLT Spearman rho: {rho:.3f}")
    w(f"  implied mean platelet volume PCT x 10000 / PLT (fL if PCT is plateletcrit in % and PLT in 10^9/L): "
      f"median {mpv.median():.2f}, p05 {mpv.quantile(0.05):.2f}, p95 {mpv.quantile(0.95):.2f}")

    # C. textbook relations
    w("")
    w("== C. textbook CBC relations (formulas UNVERIFIED until cited in Phase 2) ==")
    describe_relation(w, "MCV vs HCT x 10 / RBC", df["MCV"], df["HCT"] * 10 / df["RBC"], filled)
    describe_relation(w, "MCH vs HGB x 10 / RBC", df["MCH"], df["HGB"] * 10 / df["RBC"], filled)
    describe_relation(w, "MCHC vs HGB x 100 / HCT", df["MCHC"], df["HGB"] * 100 / df["HCT"], filled)
    describe_relation(w, "MCHC vs MCH x 100 / MCV", df["MCHC"], df["MCH"] * 100 / df["MCV"], filled)

    # D. rounding signature
    w("")
    w("== D. rounding signature in F-core columns (whole-number share, filled vs other rows) ==")
    for c in ["WBC", "RBC", "HGB", "MCV", "MCH", "MCHC", "PLT"]:
        w(f"  {c}: filled {100 * summ.loc[c, 'whole_number_share_filled_rows']:.1f}%, "
          f"other {100 * summ.loc[c, 'whole_number_share_other_rows']:.1f}%")
    m32 = df["MCHC"] == 32.0
    w(f"  MCHC == 32.0: {int(m32.sum())} rows, {int((m32 & filled).sum())} of them filled")

    # E. class medians
    keys = ["HGB", "MCV", "MCHC", "PLT", "WBC"]
    med = df.groupby(TARGET)[keys].median().reindex(CLASS_ORDER)
    med.insert(0, "n", df[TARGET].value_counts().reindex(CLASS_ORDER))
    med.round(2).to_csv(OUT_DIR / "class_medians.csv")
    w("")
    w("== E. per-class medians of key rule-engine inputs (class_medians.csv) ==")
    w("  " + med.round(2).to_string().replace("\n", "\n  "))

    # E2. per-class ranges (do labels follow sharp cut-offs on the CBC values?)
    rng = df.groupby(TARGET)[keys].agg(["min", "max"]).reindex(CLASS_ORDER)
    rng.round(2).to_csv(OUT_DIR / "class_ranges.csv")
    w("")
    w("== E2. per-class min / max of key inputs (class_ranges.csv) ==")
    for k in CLASS_ORDER:
        w(f"  {k}: " + ", ".join(f"{c} {rng.loc[k, (c, 'min')]:g}-{rng.loc[k, (c, 'max')]:g}" for c in keys))

    # G. label-rule audit: does a round-number cut-off cascade on five columns reproduce the labels?
    # The cut-offs are READ OFF THE DEVELOPMENT DATA (section E2) as round numbers. They are not
    # clinical thresholds, not cited and NOT used by the rule engine. Descriptive audit only.
    def cascade(r) -> str:
        if r["HGB"] >= 13:
            lowp, highw = r["PLT"] < 150, r["WBC"] > 10
            if lowp and highw:
                return "Leukemia with thrombocytopenia"
            if lowp:
                return "Thrombocytopenia"
            if highw:
                return "Leukemia"
            return "Healthy"
        if r["MCV"] < 80:
            return "Microcytic (IDA or other)"
        if r["MCV"] > 100:
            return "Macrocytic anemia"
        return "Normocytic normochromic anemia" if r["MCHC"] >= 32 else "Normocytic hypochromic anemia"

    pred = df.apply(cascade, axis=1)
    truth = df[TARGET].where(~df[TARGET].isin(["Iron deficiency anemia", "Other microcytic anemia"]),
                             "Microcytic (IDA or other)")
    agree = pred == truth
    w("")
    w("== G. label-rule audit (cut-offs read off development data, NOT clinical, NOT used by the rule engine) ==")
    w("  cascade: HGB >= 13 -> {PLT < 150, WBC > 10} decide Healthy / Thrombocytopenia / Leukemia / both;")
    w("           HGB < 13 -> MCV < 80 microcytic, MCV > 100 macrocytic, else MCHC >= 32 normochromic / < 32 hypochromic")
    w(f"  agreement with the label (IDA and other microcytic merged): {int(agree.sum())} / {len(df)} ({100 * agree.mean():.1f}%)")
    w(f"  agreement in filled rows: {100 * agree[filled].mean():.1f}%, other rows: {100 * agree[~filled].mean():.1f}%")
    w("  per true class (agree / rows):")
    for k in CLASS_ORDER:
        m = df[TARGET] == k
        w(f"    {k}: {int(agree[m].sum())} / {int(m.sum())}")
    micro = df[df[TARGET].isin(["Iron deficiency anemia", "Other microcytic anemia"])]
    w("  microcytic split, MCHC by label (min / median / max):")
    for k, g in micro.groupby(TARGET):
        w(f"    {k}: {g['MCHC'].min():g} / {g['MCHC'].median():g} / {g['MCHC'].max():g} (n={len(g)})")
    dis = df[~agree]
    w(f"  disagreeing rows ({len(dis)}), true label -> cascade output:")
    for (t, p), n in pd.crosstab(dis[TARGET], pred[~agree]).stack().loc[lambda s: s > 0].items():
        w(f"    {t} -> {p}: {n}")

    # F. correlated pairs
    corr = o[FEATURES].corr(method="spearman")
    w("")
    w("== F. feature pairs with Spearman |rho| >= 0.9 (other rows) ==")
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            if abs(corr.loc[a, b]) >= 0.9:
                w(f"  {a} - {b}: {corr.loc[a, b]:.3f}")

    (OUT_DIR / "data_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- figures (static PNG for the paper; single hue unless two series) ----
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    labels = [SHORT[k] for k in CLASS_ORDER][::-1]
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER).to_numpy()[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    ax.barh(labels, counts, color=BLUE, height=0.7)
    for y, v in enumerate(counts):
        ax.text(v + 3, y, str(v), va="center", fontsize=7, color=INK)
    ax.set_xlabel("Development rows after cleaning", color=MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_class_counts.png")
    plt.close(fig)

    share = (df.groupby(TARGET)["fill_count"].apply(lambda s: (s == 6).mean()).reindex(CLASS_ORDER) * 100).to_numpy()[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    ax.barh(labels, share, color=ORANGE, height=0.7, label="Mean-filled rows")
    ax.barh(labels, 100 - share, left=share + 0.6, color=BLUE, height=0.7, label="Other rows")
    for y, v in enumerate(share):
        inside = v >= 20
        ax.text(v - 1.5 if inside else v + 2, y, f"{v:.0f}%", va="center",
                ha="right" if inside else "left", fontsize=6.5, color="white")
    ax.set_xlim(0, 101)
    ax.set_xlabel("Share of class rows (%)", color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, frameon=False, fontsize=7)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fill_by_class.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(keys), figsize=(7.2, 2.8), dpi=300, sharey=True)
    for ax, k in zip(axes, keys):
        data = [df.loc[df[TARGET] == c, k].to_numpy() for c in CLASS_ORDER][::-1]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=True,
                        flierprops={"marker": "o", "markersize": 1.5, "markerfacecolor": MUTED,
                                    "markeredgecolor": "none", "alpha": 0.6},
                        medianprops={"color": INK, "linewidth": 1})
        for p in bp["boxes"]:
            p.set_facecolor(BLUE)
            p.set_alpha(0.55)
            p.set_edgecolor(BLUE)
        for part in ["whiskers", "caps"]:
            for ln in bp[part]:
                ln.set_color(MUTED)
                ln.set_linewidth(0.7)
        ax.set_title(k, fontsize=8, color=INK)
        style(ax)
    axes[0].set_yticks(range(1, len(labels) + 1), labels)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_key_features_by_class.png")
    plt.close(fig)
    print("figures written to", FIG_DIR.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
