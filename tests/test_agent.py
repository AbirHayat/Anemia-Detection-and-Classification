"""Unit tests for the agent, the explanation layer and the consistency check (Phase 5).

Synthetic records plus a few development rows. Test rows are never read. Skipped if the final model is missing.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import MODEL, Agent, check_consistency, input_hash  # noqa: E402
from data import FEATURES, load_dev, load_test  # noqa: E402
import explain as E  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="final model not fitted yet (phase5_final_models.py)")

BASE = {"WBC": 7.0, "LYMp": 30.0, "NEUTp": 60.0, "LYMn": 2.0, "NEUTn": 4.0, "RBC": 4.8, "HGB": 14.0, "HCT": 42.0,
        "MCV": 88.0, "MCH": 29.0, "MCHC": 33.0, "PLT": 250.0, "PDW": 15.0, "PCT": 0.25}


@pytest.fixture(scope="module")
def agent():
    return Agent("primary")


def test_frozen_settings_loaded(agent):
    frozen = json.loads((Path(__file__).resolve().parents[1] / "results" / "phase4" / "frozen_phase6.json").read_text())
    assert agent.tau == frozen["tau"]["primary"] and agent.trigger == frozen["refer_trigger"]


def test_valid_record_gives_consistent_audit_record(agent):
    rec = agent.run(BASE, "t-1")
    assert rec["decision"] in {"ACCEPT", "ABSTAIN", "REFER"}
    assert check_consistency(rec) == []
    assert json.loads(rec["audit_json"])["case_id"] == "t-1"
    assert E.CAVEAT in rec["summary"] and E.DISCLAIMER in rec["summary"]
    assert set(rec["latency_ms"]) == {"validation", "rules", "ml", "arbitration", "explanation", "audit"}


def test_bad_input_is_rejected_with_reason(agent):
    bad = dict(BASE, HGB=-1)
    rec = agent.run(bad, "t-2")
    assert rec["decision"] == "REJECT" and any(r.startswith("HGB") for r in rec["validator"]["reasons"])
    assert check_consistency(rec) == []
    del bad["PLT"]
    assert agent.run(bad, "t-3")["decision"] == "REJECT"


def test_same_case_same_explanation(agent):
    a, b = agent.run(BASE, "t-4"), agent.run(BASE, "t-4")
    assert a["summary"] == b["summary"] and a["shap"] == b["shap"]


def test_trigger_met_always_refers(agent):
    dev = load_dev()
    recs = [agent.run(r.to_dict(), f"d-{i}") for i, r in dev[FEATURES].head(150).iterrows()]
    for r in recs:
        assert (r["decision"] == "REFER") == bool(r["refer_trigger_met"])


def test_consistency_check_catches_tampering(agent):
    rec = agent.run(BASE, "t-5")
    t = copy.deepcopy(rec)
    t["shap"]["explained_class"] = "Leukemia" if rec["label"] != "Leukemia" else "Healthy"
    assert any("SHAP" in e for e in check_consistency(t))
    t = copy.deepcopy(rec)
    t["summary"] = t["summary"].replace(f"{rec['probability']:.2f}", "0.12345")
    assert any("probability" in e for e in check_consistency(t))
    t = copy.deepcopy(rec)
    t["inputs"]["HGB"] = 9.9
    assert any("hash" in e for e in check_consistency(t))


def test_decide_frame_matches_run(agent):
    dev = load_dev()[FEATURES].head(60)
    frame = agent.decide_frame(dev)
    for rid, r in dev.iterrows():
        rec = agent.run(r.to_dict(), "x", explain=False)
        assert rec["decision"] == frame.loc[rid, "decision"]


def test_shap_explains_the_stated_class_and_sums_to_the_model_output(agent):
    x = np.array([[BASE[c] for c in ["WBC", "RBC", "HGB", "MCV", "MCH", "MCHC", "PLT"]]])
    sv = E.shap_values(agent.explainer, agent.model, x)[0]
    pipe = E.underlying(agent.model)
    p_un = np.zeros(9)
    p_un[pipe.classes_] = pipe.predict_proba(x)[0]
    assert np.allclose(E.expected_values(agent.explainer, agent.model) + sv.sum(axis=0), p_un, atol=1e-9)


def test_input_hash_ignores_extra_columns():
    assert input_hash(BASE) == input_hash({**BASE, "Diagnosis": "x"})


def test_test_set_lock_still_holds():
    with pytest.raises(PermissionError):
        load_test()
