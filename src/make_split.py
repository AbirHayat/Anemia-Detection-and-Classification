"""Make the one stratified development/test split used for the whole project.

Research prototype, not a diagnostic tool. Never edits the raw file.
Order fixed in the project plan: (1) remove exact duplicates (keep the first copy),
(2) one stratified 80/20 split on Diagnosis with a fixed seed.
Output: results/split/dev_test_split.csv (raw row index, Diagnosis, set) and a summary.

Run:  python src/make_split.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "diagnosed_cbc_data_v4.csv"
OUT_DIR = ROOT / "results" / "split"
SEED = 42
TEST_SIZE = 0.20


def main() -> None:
    df = pd.read_csv(RAW)
    dedup = df[~df.duplicated(keep="first")]
    dev_idx, test_idx = train_test_split(
        dedup.index, test_size=TEST_SIZE, stratify=dedup["Diagnosis"], random_state=SEED
    )
    split = pd.DataFrame({"raw_row": dedup.index, "Diagnosis": dedup["Diagnosis"].values})
    split["set"] = split["raw_row"].map(lambda i: "test" if i in set(test_idx) else "dev")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(OUT_DIR / "dev_test_split.csv", index=False)

    lines = [
        f"raw rows: {len(df)}, after exact de-duplication: {len(dedup)}",
        f"seed: {SEED}, test share: {TEST_SIZE}",
        f"development rows: {len(dev_idx)}, test rows: {len(test_idx)}",
        "rows per class (dev / test):",
    ]
    tab = pd.crosstab(split["Diagnosis"], split["set"])
    for k, r in tab.sort_values("dev", ascending=False).iterrows():
        lines.append(f"  {k}: {r['dev']} / {r['test']}")
    (OUT_DIR / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
