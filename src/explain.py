"""Explanation layer, stage 5 of the agent (Phase 5): SHAP explainers and the fixed-template summary.

Research prototype, not a diagnostic tool.

One explainer per model type (proposal Section IV-C):
  decision tree, random forest, gradient boosting -> TreeSHAP (shap.TreeExplainer)
  logistic regression                             -> linear SHAP (shap.LinearExplainer, background = training rows)
  majority dummy                                  -> none
SHAP explains the UNDERLYING estimator of the calibrated model (CalibratedClassifierCV with ensemble=False keeps
one estimator fitted on all rows). The calibration step rescales each class score afterwards, so the attributions
only approximate why the displayed probability came out as it did. Every summary says so.

Wording revised on 15 Nov 2026 after the 40-case review (results/phase5/review/review_summary_ai.md): exclusion
rules in plain words, why a rule finding did not lead to referral, second candidate on ABSTAIN only, plainer
caveat, grouped feature list. No decision logic changed.

What each explainer explains (its "output space"): forest and tree -> class probabilities, gradient boosting ->
raw log-odds per class, logistic regression -> raw class scores (log-odds before the softmax). Inputs are the
scaled F-core values the classifier sees. For trees the scaling does not change the attribution.

The summary is filled only from stored values, so it cannot state a value the system did not compute.
"""
from __future__ import annotations

import numpy as np
import shap

from data import CLASS_ORDER, F_CORE

CAVEAT = ("The value effects are measured before the final probability adjustment (calibration), "
          "so they are approximate.")
DISCLAIMER = "Research prototype, not a diagnostic tool."
OUTPUT_SPACE = {"forest": "probability", "tree": "probability", "hgb": "log-odds", "logreg": "log-odds"}


def underlying(cal):
    """The single fitted pipeline inside a CalibratedClassifierCV(ProbaOnly(pipeline), ensemble=False)."""
    return cal.calibrated_classifiers_[0].estimator.estimator_


def make_explainer(model_type: str, cal, background_X=None):
    pipe = underlying(cal)
    clf = pipe.named_steps["clf"]
    if model_type in ("tree", "forest", "hgb"):
        return shap.TreeExplainer(clf)
    if model_type == "logreg":
        if background_X is None:
            raise ValueError("linear SHAP needs background rows")
        return shap.LinearExplainer(clf, pipe.named_steps["scale"].transform(background_X))
    return None                                     # dummy: nothing to explain


def shap_values(explainer, cal, X: np.ndarray) -> np.ndarray:
    """SHAP values with shape (rows, features, 9 classes) for raw F-core rows X."""
    pipe = underlying(cal)
    Z = pipe.named_steps["scale"].transform(np.asarray(X, float))
    sv = explainer.shap_values(Z)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == len(pipe.classes_) and sv.shape[1] == len(Z):   # (classes, rows, features)
        sv = np.transpose(sv, (1, 2, 0))
    out = np.zeros((len(Z), len(F_CORE), len(CLASS_ORDER)))
    out[:, :, pipe.classes_] = sv
    return out


def expected_values(explainer, cal) -> np.ndarray:
    ev = np.atleast_1d(np.asarray(explainer.expected_value, float))
    out = np.zeros(len(CLASS_ORDER))
    out[underlying(cal).classes_] = ev
    return out


def top_features(sv_row_class: np.ndarray, x_raw: np.ndarray, k: int = 3) -> list[dict]:
    """The k features with the largest absolute effect on one class, with their raw values."""
    order = np.argsort(-np.abs(sv_row_class), kind="stable")[:k]
    return [{"feature": F_CORE[i], "value": round(float(x_raw[i]), 4), "effect": round(float(sv_row_class[i]), 6),
             "direction": "raises" if sv_row_class[i] > 0 else "lowers"} for i in order]


ANEMIA = {"Normocytic hypochromic anemia", "Normocytic normochromic anemia", "Iron deficiency anemia",
          "Other microcytic anemia", "Macrocytic anemia"}
REFERRAL_LABELS = {"Thrombocytopenia", "Leukemia", "Leukemia with thrombocytopenia"}


def _prob(x: float) -> str:
    return "below 0.01" if x < 0.005 else f"{x:.2f}"


def _feats(top: list[dict], label: str) -> str:
    """Values grouped by direction (wording fix 4 of the 15 Nov review)."""
    up = [f"{t['feature']} {t['value']:g}" for t in top if t["direction"] == "raises"]
    down = [f"{t['feature']} {t['value']:g}" for t in top if t["direction"] != "raises"]
    out = []
    if up:
        out.append(f"Values that pushed the model toward {label}: {', '.join(up)}.")
    if down:
        out.append(f"Values that pushed it away from {label}: {', '.join(down)}.")
    if label in REFERRAL_LABELS and any(t["feature"] == "HGB" and t["direction"] == "raises" for t in top):
        out.append("A value can push toward a label by counting against the other labels.")   # fix 5
    return " ".join(out)


def _findings(facts: dict) -> str:
    names = [("anemia", "anemia"), ("microcytic", "small red cells"), ("macrocytic", "large red cells"),
             ("hypochromic", "pale red cells"), ("low_plt", "low platelets"), ("leukocytosis", "high white count"),
             ("leukopenia", "low white count")]
    found = [txt for key, txt in names if facts.get(key)]
    return ", ".join(found) if found else "no abnormal finding"


def exclusion_text(rid: str, t) -> str:
    """Plain words for an exclusion rule of src/rules.py (wording fix 1). t = rules.Thresholds."""
    return {
        "X1": "anemia, low platelets or an abnormal white count was found, so not Healthy",
        "X2": f"HGB is not below {t.hgb_anemia:g} g/dL, so no anemia label",
        "X3": f"MCV is not below {t.mcv_micro:g} fL, so not a small-cell anemia",
        "X4": "the red cells are not normal-sized, so no normocytic label",
        "X5": f"MCV is not above {t.mcv_macro:g} fL, so not macrocytic anemia",
        "X6": f"MCHC is not below {t.mchc_hypo:g} g/dL, so the cells are not counted as pale",
        "X7": f"MCHC is below {t.mchc_hypo:g} g/dL (pale cells), so not normocytic normochromic",
        "X8": f"platelets are not below {t.plt_low:g}, so no thrombocytopenia label",
        "X9": f"WBC is between {t.wbc_low:g} and {t.wbc_high:g}, so no leukemia label",
    }.get(rid, rid)


def summary(decision: str, reason: str, label: int, p: float, tau: float, compatible: list[str], facts: dict,
            excluded_by: list[str], second: tuple[int, float] | None, top: list[dict], reject_reasons=None,
            thresholds=None) -> str:
    """Fixed template. Uses only the values passed in. Wording revised after the 40-case review of 15 Nov 2026."""
    if decision == "REJECT":
        return f"Decision: input rejected, no medical answer. Reasons: {', '.join(reject_reasons)}. {DISCLAIMER}"
    name = CLASS_ORDER[label]
    allowed = ", ".join(compatible) if compatible else "none"
    sec = f" Second candidate: {CLASS_ORDER[second[0]]} (probability {_prob(second[1])})." if second else ""
    feats = f" {_feats(top, name)}" if top else ""
    rules_txt = f"Rule findings: {_findings(facts)}."
    not_referred = ""
    if facts.get("refer_rule") and decision != "REFER":                                      # fix 2
        not_referred = (" Low platelets or an abnormal white count was found, but referral follows the model's top "
                        "class, which is not a referral class, so the case was not referred.")
    if decision == "REFER":
        head = (f"Decision: refer to a specialist. The model's top class is {name} (probability {p:.2f}, calibrated), "
                f"a class that always goes to a specialist. {rules_txt}")
    elif decision == "ACCEPT":
        head = (f"Decision: {name} (probability {p:.2f}, calibrated). {rules_txt} The rules allow this label and the "
                f"probability reaches the threshold {tau:.2f}.{not_referred}")
    elif reason == "DISAGREE":
        why = " and ".join(f"{rid}: {exclusion_text(rid, thresholds)}" for rid in excluded_by) if (
            excluded_by and thresholds is not None) else ", ".join(excluded_by) or "an exclusion rule"
        head = (f"Decision: no answer, a person should review this case. The model's top class is {name} "
                f"(probability {p:.2f}, calibrated), but the rules exclude it ({why}). {rules_txt} "
                f"Labels the rules allow: {allowed}.{sec}{not_referred}")
    else:
        head = (f"Decision: no answer, a person should review this case. The model's top class is {name} and the rules "
                f"allow it, but its probability {p:.2f} (calibrated) is below the threshold {tau:.2f}. {rules_txt}"
                f"{sec}{not_referred}")
    return f"{head}{feats} {CAVEAT} {DISCLAIMER}"
