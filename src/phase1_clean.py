"""Phase 1: reproducible cleaning of the DEVELOPMENT rows.

Research prototype, not a diagnostic tool. The raw file is never edited and the
held-out test rows are never read here (only their row numbers, via the split file).

Steps (order fixed in the project plan before any data work)
  1. Check the raw file against the SHA-256 recorded on 7 Nov 2026.
  2. Re-run de-duplication + the seed-42 stratified 80/20 split and confirm it gives
     exactly the saved split in results/split/dev_test_split.csv.
  3. Run the input validator (src/validator.py) on every development row and drop
     the rows it rejects (rule fixed before cleaning). Count drops per class and per column.
  4. Flag (never drop) rows with NEUTn > WBC or LYMn > WBC.
  5. Describe the mean-fill pattern: fill count (0-6 of the six columns equal to their
     most frequent development value) and the PDW flag. Descriptive only. Phase 3
     recomputes the fill constants inside each training fold.

Outputs (results/phase1/)
  dev_rows.csv        one line per development row: raw_row, Diagnosis, valid,
                      reject_reason, fill_count, pdw_fill, neutn_gt_wbc, lymn_gt_wbc
  cleaning_report.txt every count quoted in the Phase 1 notes

Run:  python src/phase1_clean.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import (CLASS_ORDER, DEV_ROWS, FILL_COLS, FILL_FLAG_COL, RAW_SHA256, RAW, ROOT,
                  SEED, SPLIT, TARGET, load_raw, sha256_of)
from validator import screen_frame

OUT_DIR = ROOT / "results" / "phase1"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    w = lines.append

    # 1. hash
    h = sha256_of(RAW)
    w("== 1. raw file ==")
    w(f"sha256: {h}  matches 7 Nov record: {h == RAW_SHA256}")
    raw = load_raw()
    w(f"rows: {len(raw)}, columns: {raw.shape[1]}, empty cells: {int(raw.isna().sum().sum())}")

    # 2. de-duplication + split reproduce
    dedup = raw[~raw.duplicated(keep="first")]
    dev_idx, test_idx = train_test_split(dedup.index, test_size=0.20,
                                         stratify=dedup[TARGET], random_state=SEED)
    saved = pd.read_csv(SPLIT)
    same = (set(dev_idx) == set(saved.loc[saved["set"] == "dev", "raw_row"])
            and set(test_idx) == set(saved.loc[saved["set"] == "test", "raw_row"]))
    w("")
    w("== 2. de-duplication and split ==")
    w(f"exact duplicate rows removed (first copy kept): {len(raw) - len(dedup)}")
    w(f"rows after de-duplication: {len(dedup)}")
    w(f"re-run of seed-{SEED} stratified 80/20 split identical to results/split/dev_test_split.csv: {same}")
    w(f"development rows: {len(dev_idx)}, test rows: {len(test_idx)} (test rows not read further)")
    if not same:
        raise RuntimeError("split could not be reproduced")
    leftover_dups = dedup.loc[dev_idx].duplicated().sum()
    w(f"exact duplicates left inside development rows: {int(leftover_dups)} "
      "(no exact copy can sit on both sides, since duplicates were removed before the split)")

    # 3. validator on development rows
    dev = raw.loc[saved.loc[saved["set"] == "dev", "raw_row"]].copy()
    reasons = screen_frame(dev)
    valid = reasons == ""
    w("")
    w("== 3. input validator on development rows (own loose screening ranges, not clinical) ==")
    w(f"rows rejected: {int((~valid).sum())} of {len(dev)}; rows kept: {int(valid.sum())}")
    w("rejected rows by class (rejected / development rows of that class):")
    for k in CLASS_ORDER:
        m = dev[TARGET] == k
        w(f"  {k}: {int((~valid & m).sum())} / {int(m.sum())}")
    w("rejected rows by failing column (a row can fail several columns):")
    per_col = pd.Series([r.split("=")[0].split(":")[0] for rs in reasons[~valid] for r in rs.split(" | ")])
    for c, n in per_col.value_counts().items():
        w(f"  {c}: {n}")
    w("rows failing more than one column: "
      f"{int((reasons[~valid].str.count(r' \| ') >= 1).sum())}")
    w("every rejected row with its reasons (raw_row, class, reasons):")
    for i in dev.index[~valid]:
        w(f"  {i}, {dev.at[i, TARGET]}, {reasons[i]}")

    # 4. consistency flags (never drop)
    neut = dev["NEUTn"] > dev["WBC"]
    lym = dev["LYMn"] > dev["WBC"]
    w("")
    w("== 4. consistency flags on development rows (flagged, NOT dropped) ==")
    w(f"NEUTn > WBC: {int(neut.sum())} rows ({int((neut & valid).sum())} among kept rows)")
    w(f"LYMn > WBC: {int(lym.sum())} rows ({int((lym & valid).sum())} among kept rows)")

    # 5. fill pattern (descriptive, development-set modes)
    modes = {c: dev[c].mode().iloc[0] for c in FILL_COLS + [FILL_FLAG_COL]}
    fill_count = np.column_stack([dev[c] == modes[c] for c in FILL_COLS]).sum(axis=1)
    pdw_fill = (dev[FILL_FLAG_COL] == modes[FILL_FLAG_COL]).to_numpy()
    w("")
    w("== 5. mean-fill pattern on development rows (fill constants = development-set modes) ==")
    for c, v in modes.items():
        w(f"  {c}: most frequent value {v:.5f} in {int((dev[c] == v).sum())} rows")
    fc = pd.Series(fill_count, index=dev.index)
    w(f"fill count distribution (all development rows): {fc.value_counts().sort_index().to_dict()}")
    w(f"PDW flag = 1 in {int(pdw_fill.sum())} rows; of those, fill count 6: {int((pdw_fill & (fill_count == 6)).sum())}")
    w(f"NEUTn > WBC rows with fill count 6: {int((neut & (fc == 6)).sum())} of {int(neut.sum())}")
    w("fill count 6 by class, kept rows only (filled / kept rows of that class, share):")
    for k in CLASS_ORDER:
        m = (dev[TARGET] == k) & valid
        n6 = int(((fc == 6) & m).sum())
        w(f"  {k}: {n6} / {int(m.sum())} ({100 * n6 / m.sum():.1f}%)")
    n6 = int(((fc == 6) & valid).sum())
    w(f"  all kept rows: {n6} / {int(valid.sum())} ({100 * n6 / valid.sum():.1f}%)")

    # class counts after cleaning
    w("")
    w("== 6. development rows after cleaning, per class ==")
    kept = dev.loc[valid, TARGET].value_counts()
    for k in CLASS_ORDER:
        w(f"  {k}: {int(kept.get(k, 0))} ({100 * kept.get(k, 0) / valid.sum():.1f}%)")
    w(f"  total: {int(valid.sum())}; largest / smallest class: "
      f"{kept.max()} / {kept.min()} = {kept.max() / kept.min():.1f}")

    out = pd.DataFrame({
        "raw_row": dev.index, TARGET: dev[TARGET].to_numpy(), "valid": valid.to_numpy(),
        "reject_reason": reasons.to_numpy(), "fill_count": fill_count,
        "pdw_fill": pdw_fill.astype(int), "neutn_gt_wbc": neut.astype(int).to_numpy(),
        "lymn_gt_wbc": lym.astype(int).to_numpy(),
    })
    out.to_csv(DEV_ROWS, index=False)
    (OUT_DIR / "cleaning_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
