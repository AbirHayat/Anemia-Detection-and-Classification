"""Unit tests for the rule engine (src/rules.py). Run: python -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import CLASS_ORDER  # noqa: E402
from rules import (HEALTHY, IDA, LEU, LTH, MAC, NHA, NNA, OMA, THR, SOURCES, VARIANTS,  # noqa: E402
                   Thresholds, favours, run)

NORMAL = {"WBC": 7.0, "RBC": 4.8, "HGB": 14.0, "MCV": 88.0, "MCH": 29.0, "MCHC": 33.0, "PLT": 250.0}


def rec(**kw):
    return {**NORMAL, **kw}


def test_all_normal_is_healthy_only():
    r = run(NORMAL)
    assert r.compatible == [HEALTHY]
    assert not r.refer_rule_fires


# --- each cut-off: value at the edge and just past it (strict "<" or ">" as cited) ---
@pytest.mark.parametrize("hgb,anemic", [(12.99, True), (13.0, False)])
def test_hgb_edge(hgb, anemic):
    assert run(rec(HGB=hgb)).facts["anemia"] is anemic


@pytest.mark.parametrize("mcv,size", [(79.99, "microcytic"), (80.0, "normocytic"), (100.0, "normocytic"),
                                      (100.01, "macrocytic")])
def test_mcv_edges(mcv, size):
    assert run(rec(MCV=mcv)).facts[size] is True


@pytest.mark.parametrize("mchc,hypo", [(29.99, True), (30.0, False)])
def test_mchc_edge(mchc, hypo):
    assert run(rec(MCHC=mchc)).facts["hypochromic"] is hypo


@pytest.mark.parametrize("plt,low", [(149.9, True), (150.0, False)])
def test_plt_edge(plt, low):
    assert run(rec(PLT=plt)).facts["low_plt"] is low


@pytest.mark.parametrize("wbc,abn", [(3.99, True), (4.0, False), (11.0, False), (11.01, True)])
def test_wbc_edges(wbc, abn):
    assert run(rec(WBC=wbc)).facts["abnormal_wbc"] is abn


# --- label sets for typical patterns ---
def test_microcytic_anemia():
    assert run(rec(HGB=10, MCV=70, MCHC=28)).compatible == [IDA, OMA]


def test_normocytic_hypochromic_and_normochromic():
    assert run(rec(HGB=11, MCHC=29)).compatible == [NHA]
    assert run(rec(HGB=11, MCHC=33)).compatible == [NNA]


def test_macrocytic_anemia():
    assert run(rec(HGB=10, MCV=108)).compatible == [MAC]


def test_platelet_and_white_cell_labels():
    assert run(rec(PLT=100)).compatible == [THR]
    assert run(rec(WBC=15)).compatible == [LEU]
    assert run(rec(WBC=2.5)).compatible == [LEU]
    assert run(rec(PLT=100, WBC=15)).compatible == [THR, LEU, LTH]


def test_exclusion_only_keeps_both_findings():
    # anemia + low platelets: no hierarchy, both kinds of label stay
    r = run(rec(HGB=10, MCV=70, PLT=100))
    assert set(r.compatible) == {IDA, OMA, THR}
    assert r.refer_rule_fires


def test_set_never_empty_on_a_grid():
    for hgb in [8, 12.5, 13, 15]:
        for mcv in [70, 80, 90, 100, 110]:
            for mchc in [28, 30, 33]:
                for plt in [100, 150, 300]:
                    for wbc in [3, 4, 7, 11, 15]:
                        r = run(rec(HGB=hgb, MCV=mcv, MCHC=mchc, PLT=plt, WBC=wbc))
                        assert r.compatible, (hgb, mcv, mchc, plt, wbc)
                        assert set(r.compatible) | set(r.excluded) == set(CLASS_ORDER)


# --- indices ---
def test_indices_values_and_direction():
    f = run(rec(HGB=10, RBC=5.0, MCV=70, MCH=22, MCHC=28)).facts
    assert f["mentzer"] == pytest.approx(14.0)
    assert f["england_fraser"] == pytest.approx(70 - 5 - 50 - 3.4)
    assert f["shine_lal"] == pytest.approx(70 ** 2 * 22 * 0.01)
    assert f["mentzer_favours"] == "iron deficiency"
    assert f["shine_lal_favours"] == "thalassemia trait"


def test_indices_only_for_microcytic_anemia():
    assert "mentzer" not in run(NORMAL).facts
    assert "mentzer" not in run(rec(MCV=70)).facts            # microcytic, no anemia
    assert "mentzer" not in run(rec(HGB=10)).facts            # anemia, normocytic


def test_indices_never_change_the_set():
    a = run(rec(HGB=10, RBC=6.5, MCV=65, MCH=20, MCHC=29)).compatible   # indices favour thalassemia
    b = run(rec(HGB=10, RBC=3.5, MCV=75, MCH=24, MCHC=29)).compatible   # indices favour iron deficiency
    assert a == b == [IDA, OMA]


def test_favours_borderline():
    assert favours(13.0, 13.0).startswith("borderline")


# --- variants and trace ---
def test_sensitivity_variants():
    assert run(rec(HGB=12.5), VARIANTS["hgb12"]).facts["anemia"] is False
    assert run(rec(HGB=12.5), VARIANTS["primary"]).facts["anemia"] is True
    assert run(rec(HGB=11, MCHC=31), VARIANTS["mchc32"]).compatible == [NHA]
    assert run(rec(HGB=11, MCHC=31), VARIANTS["primary"]).compatible == [NNA]


def test_trace_has_source_and_inputs():
    r = run(rec(HGB=10, MCV=70))
    assert all(step["source"] in SOURCES.values() for step in r.trace)
    f1 = next(s for s in r.trace if s["rule"] == "F1")
    assert f1["inputs"] == {"HGB": 10.0} and f1["result"] == {"anemia": True}


def test_deterministic():
    a, b = run(rec(HGB=10, MCV=70, PLT=90)), run(rec(HGB=10, MCV=70, PLT=90))
    assert a.compatible == b.compatible and a.trace == b.trace


def test_missing_value_raises():
    with pytest.raises(KeyError):
        run({k: v for k, v in NORMAL.items() if k != "PLT"})


def test_default_thresholds_match_logged_values():
    t = Thresholds()
    assert (t.hgb_anemia, t.mcv_micro, t.mcv_macro, t.mchc_hypo, t.plt_low, t.wbc_high, t.wbc_low) == \
        (13.0, 80.0, 100.0, 30.0, 150.0, 11.0, 4.0)
