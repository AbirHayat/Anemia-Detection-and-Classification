"""Original (pre-registered) vs amended (probability-based) calibration, for every model configuration.

Research prototype, not a diagnostic tool. Reads the summary tables written by
src/phase3_evaluate.py (original) and src/phase3_evaluate.py --version amended.
Output: results/phase3/calibration_amendment_comparison.md
"""
from __future__ import annotations

import pandas as pd

from data import ROOT

P3 = ROOT / "results" / "phase3"
RAW_SCORE_MODELS = {"logreg", "hgb"}          # expose decision_function (checked 14 Nov 2026)


def load(path):
    return pd.read_csv(path, header=[0, 1], index_col=[0, 1, 2])


def cell(df, idx, metric="macro_f1"):
    r = df.loc[idx]
    return f"{r[(metric, 'mean')]:.3f} ± {r[(metric, 'std')]:.3f}"


def arm_choice(text: str) -> str:
    lines = text.split("== P2.")[1].split("== P3.")[0].strip().splitlines()[1:]
    h2 = text.split("-> H2 model: ")[1].splitlines()[0]
    return "; ".join(f"{ln.split()[0]} {ln.split('chosen: ')[1].split()[0]}" for ln in lines), h2


def main() -> None:
    orig, amend, un = (load(P3 / "summary_metrics_cal.csv"), load(P3 / "amended" / "summary_metrics_cal.csv"),
                       load(P3 / "summary_metrics_un.csv"))
    T = ["| Feature set | Model | Arm | Calibrated on (original) | Uncalibrated | Calibrated, ORIGINAL (pre-registered) | "
         "Calibrated, AMENDED (on probabilities) | Change |", "|---|---|---|---|---|---|---|---|"]
    for idx in orig.index:
        fs, m, imb = idx
        d = amend.loc[idx][("macro_f1", "mean")] - orig.loc[idx][("macro_f1", "mean")]
        T.append(f"| {fs} | {m} | {imb} | {'raw scores' if m in RAW_SCORE_MODELS else 'probabilities'} | {cell(un, idx)} | "
                 f"{cell(orig, idx)} | {cell(amend, idx)} | {d:+.3f} |")
    a_o, h_o = arm_choice((P3 / "evaluation.txt").read_text(encoding="utf-8"))
    a_a, h_a = arm_choice((P3 / "amended" / "evaluation.txt").read_text(encoding="utf-8"))
    text = ("Macro-F1, development CV (25 outer folds, mean ± std). Original = pre-registered calibration "
            "(scikit-learn uses raw decision scores when a model has them). Amended = post-results amendment of "
            "14 Nov 2026: sigmoid calibration on predicted probabilities for every model. Same folds, seeds and "
            "hyperparameters. Sources: results/phase3/summary_metrics_cal.csv, summary_metrics_un.csv and "
            "results/phase3/amended/summary_metrics_cal.csv.\n\n" + "\n".join(T) +
            f"\n\nPre-registered rules applied to each version:\n\n- Imbalance arm (P2), original: {a_o}\n"
            f"- Imbalance arm (P2), amended: {a_a}\n- H2 model (P3), original: {h_o}\n- H2 model (P3), amended: {h_a}\n")
    (P3 / "calibration_amendment_comparison.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
