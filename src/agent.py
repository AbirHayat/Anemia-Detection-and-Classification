"""The hybrid agent: six stages in a fixed order, one JSON audit record per case (Phase 5).

Research prototype, not a diagnostic tool. It must never be used for patient care.

Stages (proposal Section IV-B):
  1 validation   validator.validate_record (REJECT on any reason)
  2 rules        rules.run with the chosen threshold version (forward chaining, exclusion-only label set)
  3 ml           calibrated random forest + SMOTE on F-core, fitted on all development rows (phase5_final_models.py)
  4 arbitration  arbitration.decide with the FROZEN Phase 4 settings (results/phase4/frozen_phase6.json)
  5 explanation  TreeSHAP top features for the stated class + fixed-template summary (explain.py)
  6 audit        the whole record serialised to JSON
Per-stage latency is measured with time.perf_counter and stored in the record.

check_consistency() is the automatic explanation check of the proposal (acceptance target 3): the rule trace and
the SHAP output refer to the same case and the same class as the decision, and the summary repeats the stored
label and probability.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn

import arbitration as A
import explain as E
import rules
from data import CLASS_ORDER, F_CORE, FEATURES, ROOT, SEED
from validator import consistency_flags, validate_record

FROZEN = ROOT / "results" / "phase4" / "frozen_phase6.json"
MODEL = ROOT / "results" / "phase5" / "models" / "forest_smote_Fcore.joblib"
STAGES = ["validation", "rules", "ml", "arbitration", "explanation", "audit"]
SRC_FILES = ["validator.py", "rules.py", "arbitration.py", "explain.py", "agent.py", "ml_pipelines.py", "data.py"]


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is None or isinstance(v, (int, str, bool)):
        return v
    return repr(v)


def input_hash(inputs: Mapping) -> str:
    """Fingerprint of the 14 input values as received (missing keys count as None)."""
    norm = {c: _jsonable(inputs.get(c)) for c in FEATURES}
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:16]


class Agent:
    def __init__(self, variant: str = "primary", model_path: Path = MODEL, frozen_path: Path = FROZEN):
        frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        self.variant = variant
        self.thresholds = rules.VARIANTS[variant]
        self.trigger = frozen["refer_trigger"]
        self.tau = float(frozen["tau"][variant])
        self.model = joblib.load(model_path)
        self.explainer = E.make_explainer("forest", self.model)
        src = ROOT / "src"
        self.versions = {"rule_base_sha256": sha_file(src / "rules.py")[:16], "model_sha256": sha_file(model_path)[:16],
                         "frozen_settings_sha256": sha_file(frozen_path)[:16],
                         "code_sha256": {f: sha_file(src / f)[:12] for f in SRC_FILES},
                         "seed": SEED, "python": platform.python_version(), "sklearn": sklearn.__version__,
                         "shap": shap.__version__, "numpy": np.__version__}

    # ---- stage 3 for many rows at once (same model, same column order)
    def probabilities(self, X: np.ndarray) -> np.ndarray:
        P = np.zeros((len(X), len(CLASS_ORDER)))
        P[:, self.model.classes_] = self.model.predict_proba(np.asarray(X, float))
        return P

    def trigger_met(self, P: np.ndarray, refer_rule) -> np.ndarray:
        return A.refer_flags(self.trigger, P, np.asarray(refer_rule, bool))

    def run(self, record: Mapping, case_id: str, explain: bool = True) -> dict:
        lat = {s: None for s in STAGES}
        t = time.perf_counter()
        inputs = {c: _jsonable(record.get(c)) for c in FEATURES}
        reasons = validate_record(record)
        flags = consistency_flags(record)
        h = input_hash(record)
        rec = {"case_id": case_id, "input_hash": h, "inputs": inputs, "variant": self.variant,
               "validator": {"reasons": reasons, "flags": flags}}
        lat["validation"] = (time.perf_counter() - t) * 1000
        if reasons:
            t = time.perf_counter()
            rec.update({"decision": "REJECT", "reason": "INVALID_INPUT", "label": None, "probability": None,
                        "probabilities": None, "rules": None, "shap": None, "refer_trigger_met": None,
                        "summary": E.summary("REJECT", "", -1, 0.0, self.tau, [], {}, [], None, [], reasons)})
            lat["explanation"] = (time.perf_counter() - t) * 1000
            return self._finish(rec, lat)
        t = time.perf_counter()
        rr = rules.run(record, self.thresholds)
        lat["rules"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        x = np.array([[float(record[c]) for c in F_CORE]])
        P = self.probabilities(x)[0]
        lat["ml"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        M = A.set_mask([" | ".join(rr.compatible)])
        refer = self.trigger_met(P[None, :], [rr.refer_rule_fires])
        _, top, reason = A.decide(P[None, :], M, refer, self.tau)
        label, reason = int(top[0]), str(reason[0])
        decision = {"REFER": "REFER", "ACCEPT": "ACCEPT"}.get(reason, "ABSTAIN")
        p = round(float(P[label]), 6)
        order = np.argsort(-P, kind="stable")
        second = (int(order[1]), round(float(P[order[1]]), 6))
        lat["arbitration"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        shap_out, top3 = None, []
        if explain:
            sv = E.shap_values(self.explainer, self.model, x)[0][:, label]
            top3 = E.top_features(sv, x[0])
            shap_out = {"case_id": case_id, "input_hash": h, "explained_class": CLASS_ORDER[label],
                        "explainer": "TreeSHAP (random forest, uncalibrated)", "output_space": "probability",
                        "base_value": round(float(E.expected_values(self.explainer, self.model)[label]), 6),
                        "top_features": top3}
        text = E.summary(decision, reason, label, p, self.tau, rr.compatible, rr.facts,
                         rr.excluded.get(CLASS_ORDER[label], []), second, top3,
                         thresholds=self.thresholds)
        lat["explanation"] = (time.perf_counter() - t) * 1000
        rec.update({
            "decision": decision, "reason": reason, "label": CLASS_ORDER[label], "probability": p,
            "probabilities": {CLASS_ORDER[k]: round(float(P[k]), 6) for k in range(len(CLASS_ORDER))},
            "second_candidate": {"label": CLASS_ORDER[second[0]], "probability": second[1]},
            "tau": self.tau, "refer_trigger": self.trigger, "refer_trigger_met": bool(refer[0]),
            "rules": {"case_id": case_id, "input_hash": h, "thresholds": self.variant, "compatible": rr.compatible,
                      "excluded": rr.excluded, "refer_rule_fires": rr.refer_rule_fires,
                      "trace": json.loads(json.dumps(rr.trace, default=_jsonable))},
            "shap": shap_out, "summary": text})
        return self._finish(rec, lat)

    def _finish(self, rec: dict, lat: dict) -> dict:
        rec["versions"] = self.versions
        t = time.perf_counter()
        rec["audit_json"] = json.dumps({k: v for k, v in rec.items()}, sort_keys=True, default=_jsonable)
        lat["audit"] = (time.perf_counter() - t) * 1000
        rec["latency_ms"] = lat
        return rec

    # ---- the same stages for a whole table, decisions only (used by the noise stress test)
    def decide_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        valid = df.apply(lambda r: not validate_record(r.to_dict()), axis=1).to_numpy()
        out = pd.DataFrame({"decision": "REJECT", "label": -1, "reason": "INVALID_INPUT"}, index=df.index)
        if valid.any():
            sub = df[valid]
            rr = [rules.run(r, self.thresholds) for _, r in sub.iterrows()]
            M = A.set_mask([" | ".join(x.compatible) for x in rr])
            P = self.probabilities(sub[F_CORE].to_numpy(float))
            refer = self.trigger_met(P, [x.refer_rule_fires for x in rr])
            _, top, reason = A.decide(P, M, refer, self.tau)
            out.loc[sub.index, "label"] = top
            out.loc[sub.index, "reason"] = reason
            out.loc[sub.index, "decision"] = [{"REFER": "REFER", "ACCEPT": "ACCEPT"}.get(r, "ABSTAIN") for r in reason]
        return out


def check_consistency(rec: dict) -> list[str]:
    """Automatic explanation check (acceptance target 3). An empty list means the record passed."""
    errs = []
    h = input_hash(rec["inputs"])
    if rec["input_hash"] != h:
        errs.append("input hash does not match the stored inputs")
    s = rec["summary"]
    if E.DISCLAIMER not in s:
        errs.append("summary lacks the disclaimer")
    if rec["decision"] == "REJECT":
        errs += [f"reject reason missing from summary: {r}" for r in rec["validator"]["reasons"] if r not in s]
        if not rec["validator"]["reasons"]:
            errs.append("REJECT without a reason")
        return errs
    R, S = rec["rules"], rec["shap"]
    if R["case_id"] != rec["case_id"] or R["input_hash"] != h:
        errs.append("rule trace refers to another case")
    for step in R["trace"]:
        for k, v in step["inputs"].items():
            if float(rec["inputs"][k]) != float(v):
                errs.append(f"rule {step['rule']} used {k}={v}, input was {rec['inputs'][k]}")
    if S is None:
        errs.append("no SHAP output")
    else:
        if S["case_id"] != rec["case_id"] or S["input_hash"] != h:
            errs.append("SHAP output refers to another case")
        if S["explained_class"] != rec["label"]:
            errs.append("SHAP explains a different class than the decision")
    if rec["label"] not in s:
        errs.append("summary does not repeat the stored label")
    if f"{rec['probability']:.2f}" not in s:
        errs.append("summary does not repeat the stored probability")
    if E.CAVEAT not in s:
        errs.append("summary lacks the calibration caveat")
    if rec["decision"] == "ACCEPT" and (rec["label"] not in R["compatible"] or rec["probability"] < rec["tau"]):
        errs.append("ACCEPT although the rules exclude the label or the probability is below tau")
    if rec["decision"] == "REFER" and not rec["refer_trigger_met"]:
        errs.append("REFER without the trigger")
    if rec["refer_trigger_met"] and rec["decision"] != "REFER":
        errs.append("trigger met but not referred")
    return errs
