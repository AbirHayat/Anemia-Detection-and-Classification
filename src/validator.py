"""Input validator (stage 1 of the agent).

Research prototype, not a diagnostic tool.

The same rule is used twice so that training and use follow one rule:
  * Phase 1 drops development rows that fail it.
  * At prediction time the agent returns REJECT for any record that fails it.

The screening ranges are the student's OWN LOOSE LIMITS for catching impossible or
corrupt values. They are NOT clinical reference ranges and are not cited. Units are
assumed (the dataset states none). Unverified.

Two kinds of range:
  * closed range [lo, hi] for ten columns (fixed on 6 Nov 2026)
  * (0, hi] for WBC, LYMn, NEUTn and PLT (option (a), chosen 7 Nov 2026, units 10^9/L assumed)

Consistency checks (NEUTn > WBC, LYMn > WBC) only FLAG a record. They never reject it.
"""
from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from data import FEATURES

CLOSED_RANGES: dict[str, tuple[float, float]] = {
    "HGB": (0, 25), "MCV": (40, 150), "MCH": (10, 50), "MCHC": (20, 45),
    "HCT": (5, 75), "RBC": (0.5, 9), "NEUTp": (0, 100), "LYMp": (0, 100),
    "PDW": (5, 30), "PCT": (0, 2),
}
POSITIVE_UPPER: dict[str, float] = {"WBC": 500, "LYMn": 500, "NEUTn": 500, "PLT": 2000}

assert set(CLOSED_RANGES) | set(POSITIVE_UPPER) == set(FEATURES)


def check_value(col: str, value) -> str | None:
    """Return a reason string if the value fails, else None."""
    if value is None:
        return f"{col}: missing"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{col}: not a number ({value!r})"
    if math.isnan(v):
        return f"{col}: missing"
    if math.isinf(v):
        return f"{col}: not finite"
    if col in CLOSED_RANGES:
        lo, hi = CLOSED_RANGES[col]
        if not lo <= v <= hi:
            return f"{col}={v:g} outside [{lo:g}, {hi:g}]"
    else:
        hi = POSITIVE_UPPER[col]
        if not 0 < v <= hi:
            return f"{col}={v:g} outside (0, {hi:g}]"
    return None


def validate_record(record: Mapping) -> list[str]:
    """All reasons a single record is rejected. An empty list means the record is valid."""
    return [r for c in FEATURES if (r := check_value(c, record.get(c))) is not None]


def consistency_flags(record: Mapping) -> list[str]:
    """Flags that never reject (absolute counts larger than the total white count)."""
    out = []
    try:
        wbc = float(record["WBC"])
        if float(record["NEUTn"]) > wbc:
            out.append("NEUTn > WBC")
        if float(record["LYMn"]) > wbc:
            out.append("LYMn > WBC")
    except (KeyError, TypeError, ValueError):
        pass
    return out


def screen_frame(df: pd.DataFrame) -> pd.Series:
    """Reject reasons per row, joined with " | ". An empty string means the row is valid."""
    return df.apply(lambda r: " | ".join(validate_record(r.to_dict())), axis=1)
