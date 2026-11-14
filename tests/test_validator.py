"""Unit tests for the input validator (src/validator.py). Run: python -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import FEATURES, load_test  # noqa: E402
from validator import CLOSED_RANGES, POSITIVE_UPPER, consistency_flags, validate_record  # noqa: E402

# A plausible record built from the median development values (results/phase1/data_report.txt, section A).
GOOD = {"WBC": 7.4, "LYMp": 25.8, "NEUTp": 77.5, "LYMn": 1.9, "NEUTn": 5.1, "RBC": 4.6,
        "HGB": 12.2, "HCT": 46.2, "MCV": 86.6, "MCH": 27.7, "MCHC": 32.0, "PLT": 213.0,
        "PDW": 14.3, "PCT": 0.26}


def test_good_record_passes():
    assert validate_record(GOOD) == []


@pytest.mark.parametrize("col", FEATURES)
def test_missing_field_rejected(col):
    rec = {k: v for k, v in GOOD.items() if k != col}
    assert validate_record(rec) == [f"{col}: missing"]


@pytest.mark.parametrize("bad", [float("nan"), None, "abc", float("inf")])
def test_nan_none_text_inf_rejected(bad):
    assert len(validate_record({**GOOD, "HGB": bad})) == 1


@pytest.mark.parametrize("col", sorted(CLOSED_RANGES))
def test_closed_range_edges(col):
    lo, hi = CLOSED_RANGES[col]
    assert validate_record({**GOOD, col: lo}) == []          # edges are allowed
    assert validate_record({**GOOD, col: hi}) == []
    assert len(validate_record({**GOOD, col: lo - 0.01})) == 1
    assert len(validate_record({**GOOD, col: hi + 0.01})) == 1


@pytest.mark.parametrize("col", sorted(POSITIVE_UPPER))
def test_positive_range_edges(col):
    hi = POSITIVE_UPPER[col]
    assert len(validate_record({**GOOD, col: 0})) == 1      # zero is rejected
    assert validate_record({**GOOD, col: 0.01}) == []
    assert validate_record({**GOOD, col: hi}) == []
    assert len(validate_record({**GOOD, col: hi + 0.01})) == 1


def test_values_seen_in_the_raw_file_rejected():
    # Real corrupt values listed in results/phase1/cleaning_report.txt
    for col, v in [("HGB", -10), ("MCV", -79.3), ("HCT", 3715), ("NEUTp", 5317), ("MCH", 3117)]:
        assert validate_record({**GOOD, col: v}), (col, v)


def test_several_failures_all_reported():
    assert len(validate_record({**GOOD, "HGB": -1, "MCV": 999, "PLT": 0})) == 3


def test_numeric_strings_accepted():
    assert validate_record({k: str(v) for k, v in GOOD.items()}) == []


def test_consistency_flags_never_reject():
    rec = {**GOOD, "NEUTn": 9.0, "WBC": 7.0}
    assert consistency_flags(rec) == ["NEUTn > WBC"]
    assert validate_record(rec) == []


def test_test_set_guard():
    with pytest.raises(PermissionError):
        load_test()
