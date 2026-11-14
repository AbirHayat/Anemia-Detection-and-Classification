"""Shared constants and data loaders for the whole project.

Research prototype, not a diagnostic tool.

The raw CSV is never edited. Every loader rebuilds its rows from the raw file plus
an index file in results/, so there is one copy of the data on disk.

Test-set guard: `load_dev()` returns development rows only. The held-out test rows
can only be read through `load_test(i_understand_this_is_phase6=True)`, which is
meant to be called exactly once, in Phase 6, after the git tag.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "diagnosed_cbc_data_v4.csv"
SPLIT = ROOT / "results" / "split" / "dev_test_split.csv"
DEV_ROWS = ROOT / "results" / "phase1" / "dev_rows.csv"

# SHA-256 of the Kaggle file as downloaded and checked on 7 Nov 2026 (see src/check_dataset.py).
RAW_SHA256 = "f7dee8d4fe500aec6e8e4bfbeb6df6b83bc86bef03d540721e0aeaad2759fcdb"

SEED = 42
TARGET = "Diagnosis"
FEATURES = ["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT",
            "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"]
# F-core: the seven columns with no mean-fill pattern. F-all: all 14.
F_CORE = ["WBC", "RBC", "HGB", "MCV", "MCH", "MCHC", "PLT"]
F_ALL = list(FEATURES)
# Columns that hold one repeated (mean-filled) value in 781 raw rows, and PDW (417 rows).
FILL_COLS = ["LYMp", "NEUTp", "LYMn", "NEUTn", "HCT", "PCT"]
FILL_FLAG_COL = "PDW"

REFERRAL_CLASSES = ["Leukemia", "Thrombocytopenia", "Leukemia with thrombocytopenia"]
CLASS_ORDER = [
    "Healthy",
    "Normocytic hypochromic anemia",
    "Normocytic normochromic anemia",
    "Iron deficiency anemia",
    "Thrombocytopenia",
    "Other microcytic anemia",
    "Leukemia",
    "Macrocytic anemia",
    "Leukemia with thrombocytopenia",
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(check_hash: bool = True) -> pd.DataFrame:
    """The raw file, unchanged. Raises if the file differs from the recorded hash."""
    if check_hash and sha256_of(RAW) != RAW_SHA256:
        raise RuntimeError("raw CSV hash differs from the one recorded on 7 Nov 2026")
    return pd.read_csv(RAW)


def load_dev(clean: bool = True) -> pd.DataFrame:
    """Development rows (index = raw row number).

    clean=True drops the rows the Phase 1 validator rejected (results/phase1/dev_rows.csv).
    The extra columns `fill_count`, `pdw_fill` and `neutn_gt_wbc` are descriptive flags
    from Phase 1 (fill constants taken from all development rows). Phase 3 recomputes
    the fill indicator inside each training fold and must not use these columns as features.
    """
    raw = load_raw()
    if clean:
        rows = pd.read_csv(DEV_ROWS)
        rows = rows[rows["valid"]]
        df = raw.loc[rows["raw_row"]].copy()
        for c in ["fill_count", "pdw_fill", "neutn_gt_wbc"]:
            df[c] = rows[c].to_numpy()
        return df
    split = pd.read_csv(SPLIT)
    return raw.loc[split.loc[split["set"] == "dev", "raw_row"]].copy()


def load_test(i_understand_this_is_phase6: bool = False) -> pd.DataFrame:
    """Held-out test rows. Only for the single Phase 6 run after the git tag."""
    if not i_understand_this_is_phase6:
        raise PermissionError("the test set is opened once, in Phase 6, after the git tag")
    raw = load_raw()
    split = pd.read_csv(SPLIT)
    return raw.loc[split.loc[split["set"] == "test", "raw_row"]].copy()
