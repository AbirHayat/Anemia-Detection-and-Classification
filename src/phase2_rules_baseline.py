"""Phase 2: rules-only baseline, cut-off gap analysis and the frozen label-cascade baseline.

Research prototype, not a diagnostic tool. DEVELOPMENT rows only (963 clean rows). Test rows are
never read. The rule engine is not fitted, so scoring it on all development rows is not leakage.

Parts
  A. Rule-set coverage (true label inside the compatible set), set size, singleton accuracy and the
     provisional rule-based REFER trigger, for four threshold variants (primary; HGB 12; MCHC 32; both).
     Every number is also split by mean-filled vs other rows.
  B. Misses: which exclusion rule removed the true label, per class.
  C. Gap analysis: development rows whose values fall between a cited cut-off and the label-cascade
     cut-off from Phase 1 (WBC 10-11, MCHC 30-32).
  D. Label-cascade baseline (decision of 14 Nov 2026, option B): structure and round cut-offs from the
     Phase 1 audit, plus ONE fitted value (the MCHC split between iron deficiency and other microcytic,
     chosen on development rows by balanced accuracy). Frozen to cascade_frozen.json and scored once on
     the test set in Phase 6. It is a probable LABELING rule, not a clinical rule. Its development score
     here is in-sample and optimistic.

Outputs: results/phase2/rules_baseline.txt, rule_sets_dev.csv, cascade_frozen.json
Run:     python src/phase2_rules_baseline.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data import CLASS_ORDER, REFERRAL_CLASSES, ROOT, TARGET, load_dev
from rules import IDA, OMA, VARIANTS, run

OUT = ROOT / "results" / "phase2"


def cascade_predict(r: pd.Series, p: dict) -> str:
    if r["HGB"] >= p["hgb"]:
        low, high = r["PLT"] < p["plt"], r["WBC"] > p["wbc"]
        if low and high:
            return "Leukemia with thrombocytopenia"
        if low:
            return "Thrombocytopenia"
        if high:
            return "Leukemia"
        return "Healthy"
    if r["MCV"] < p["mcv_micro"]:
        return IDA if r["MCHC"] < p["mchc_ida_oma"] else OMA
    if r["MCV"] > p["mcv_macro"]:
        return "Macrocytic anemia"
    return "Normocytic normochromic anemia" if r["MCHC"] >= p["mchc_normo"] else "Normocytic hypochromic anemia"


def bal_acc(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in y_true.unique()]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dev(clean=True)
    filled = df["fill_count"] == 6
    y = df[TARGET]
    lines: list[str] = []
    w = lines.append
    w(f"cleaned development rows: {len(df)} (filled {int(filled.sum())}, other {int((~filled).sum())}). Test rows not read.")
    sets_out = pd.DataFrame({"raw_row": df.index, TARGET: y.to_numpy()})

    # A + B
    for name, t in VARIANTS.items():
        res = [run(r.to_dict(), t) for _, r in df.iterrows()]
        comp = pd.Series([r.compatible for r in res], index=df.index)
        size = comp.str.len()
        cover = pd.Series([lab in c for lab, c in zip(y, comp)], index=df.index)
        single = size == 1
        refer = pd.Series([r.refer_rule_fires for r in res], index=df.index)
        is_ref = y.isin(REFERRAL_CLASSES)
        sets_out[f"set_{name}"] = [" | ".join(c) for c in comp]
        w("")
        w(f"== A. variant '{name}': HGB < {t.hgb_anemia:g}, MCHC < {t.mchc_hypo:g} (others fixed) ==")
        w(f"  rule-set coverage: {int(cover.sum())} / {len(df)} ({100 * cover.mean():.1f}%); "
          f"filled rows {100 * cover[filled].mean():.1f}%, other rows {100 * cover[~filled].mean():.1f}%")
        w(f"  mean set size: {size.mean():.2f} of 9; size distribution {size.value_counts().sort_index().to_dict()}")
        w(f"  singleton sets: {int(single.sum())} ({100 * single.mean():.1f}%); true label = the single label in "
          f"{int((single & cover).sum())} of them ({100 * (single & cover).sum() / max(single.sum(), 1):.1f}%)")
        w(f"  empty sets: {int((size == 0).sum())}")
        w(f"  provisional rule-based REFER trigger: sensitivity {int((refer & is_ref).sum())}/{int(is_ref.sum())} "
          f"({100 * (refer & is_ref).sum() / is_ref.sum():.1f}%), false referrals {int((refer & ~is_ref).sum())}/"
          f"{int((~is_ref).sum())} ({100 * (refer & ~is_ref).sum() / (~is_ref).sum():.1f}%)")
        w("  per class: coverage (in set / rows), mean set size, misses by excluding rule:")
        for k in CLASS_ORDER:
            m = y == k
            miss_rules: dict[str, int] = {}
            for i in df.index[m & ~cover]:
                rr = res[df.index.get_loc(i)]
                key = "+".join(rr.excluded.get(k, []))
                miss_rules[key] = miss_rules.get(key, 0) + 1
            w(f"    {k}: {int(cover[m].sum())}/{int(m.sum())}, size {size[m].mean():.2f}, misses {miss_rules}")

    # C. gap analysis
    w("")
    w("== C. rows between a cited cut-off and the label-cascade cut-off (Phase 1, section G) ==")
    g1 = (df["WBC"] > 10) & (df["WBC"] <= 11)
    w(f"  WBC in (10, 11] (cascade says high, cited rule says normal): {int(g1.sum())} rows, by label "
      f"{y[g1].value_counts().to_dict()}")
    anem_norm = (df["HGB"] < 13) & (df["MCV"] >= 80) & (df["MCV"] <= 100)
    g2 = anem_norm & (df["MCHC"] >= 30) & (df["MCHC"] < 32)
    w(f"  normocytic anemia with MCHC in [30, 32) (cascade says hypochromic, cited rule says normochromic): "
      f"{int(g2.sum())} rows, by label {y[g2].value_counts().to_dict()}")

    # D. label-cascade baseline
    micro = y.isin([IDA, OMA])
    cands = np.round(np.arange(28.0, 34.01, 0.1), 1)
    scores = [bal_acc(y[micro], df.loc[micro, "MCHC"].lt(c).map({True: IDA, False: OMA})) for c in cands]
    best = float(cands[int(np.argmax(scores))])          # first (lowest) value at the maximum
    params = {"hgb": 13.0, "plt": 150.0, "wbc": 10.0, "mcv_micro": 80.0, "mcv_macro": 100.0,
              "mchc_normo": 32.0, "mchc_ida_oma": best,
              "note": "Probable labeling rule read off DEVELOPMENT data (Phase 1 audit + one fitted MCHC split). "
                      "Not a clinical rule. Frozen 14 Nov 2026. Score once on the test set in Phase 6.",
              "fitted_on": "963 clean development rows", "fit_rule": "MCHC split maximising balanced accuracy "
              "between iron deficiency and other microcytic on development rows labelled with either, grid 28.0-34.0 "
              "step 0.1, lowest value at the maximum"}
    (OUT / "cascade_frozen.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    pred = df.apply(lambda r: cascade_predict(r, params), axis=1)
    w("")
    w("== D. label-cascade baseline (option B), frozen in cascade_frozen.json ==")
    w(f"  fitted MCHC split (iron deficiency below, other microcytic at or above): {best}")
    w(f"  IN-SAMPLE development agreement (optimistic, the cut-offs were read off these rows): "
      f"{int((pred == y).sum())} / {len(df)} ({100 * (pred == y).mean():.1f}%)")
    w(f"  in-sample balanced accuracy over 9 classes: {100 * bal_acc(y, pred):.1f}%")
    sets_out["cascade_pred"] = pred.to_numpy()

    sets_out.to_csv(OUT / "rule_sets_dev.csv", index=False)
    (OUT / "rules_baseline.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
