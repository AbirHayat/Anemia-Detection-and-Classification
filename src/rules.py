"""Rule engine (expert system), stage 2 of the agent.

Research prototype, not a diagnostic tool. It does not diagnose anyone.

What it does
  A small forward-chaining engine. Working memory starts with the 14 CBC values. Each rule
  has conditions on facts already in memory and adds one new fact. The engine keeps firing
  every rule whose conditions hold until no rule can add anything (a fixed point). Every
  firing is written to a trace with its inputs, its result and its cited source.

Two layers of rules
  1. Fact rules turn values into findings (anemia, cell size, chromicity, platelets, white
     cells) and, for microcytic anemia, compute three discrimination indices (trace only).
  2. Exclusion rules (decision of 14 Nov 2026, option A): a label stays in the compatible set
     unless a cited finding contradicts it. Rules never add a label, they only remove one.
     Tying the two leukemia labels to an abnormal WBC and treating iron deficiency as
     microcytic come from the project design of 7 Nov 2026, written before the label audit.

Thresholds (chosen and logged with their sources on 14 Nov 2026, before any ML result existed)
  Units of the data are inferred, not stated by the dataset (Phase 1).
  The label-cascade cut-offs found in Phase 1 are NOT used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from data import CLASS_ORDER

HEALTHY = "Healthy"
NHA = "Normocytic hypochromic anemia"
NNA = "Normocytic normochromic anemia"
IDA = "Iron deficiency anemia"
THR = "Thrombocytopenia"
OMA = "Other microcytic anemia"
LEU = "Leukemia"
MAC = "Macrocytic anemia"
LTH = "Leukemia with thrombocytopenia"
ANEMIA_LABELS = [NHA, NNA, IDA, OMA, MAC]

SOURCES = {
    "WHO": "WHO 2024, Guideline on haemoglobin cutoffs to define anaemia (ISBN 978-92-4-008854-2), Table 2, "
           "printed p. 9 (adult men 15-65: < 130 g/L)",
    "MERCK_ANEMIA": "Gerber, Evaluation of Anemia, Merck Manual Professional (Oct 2026): MCV < 80 fL microcytic, "
                    "> 100 fL macrocytic, MCHC < 30% hypochromic (read as g/dL)",
    "AAFP_PLT": "Gauer and Whitaker, Am Fam Physician 2022;106(3):288-298: platelets < 150 x 10^3/uL",
    "AAFP_WBC": "Riley and Rupert, Am Fam Physician 2015;92(11):1004-1011: WBC > 11.0 x 10^9/L",
    "MERCK_WBC": "Dale, Overview of Leukopenias, Merck Manual Professional (Feb 2025): WBC < 4 x 10^9/L; MeSH D007970",
    "INDICES": "Hans, Atreja and Batra 2024, Cureus (doi:10.7759/cureus.64048), Table 1. The 1970s originals "
               "(Mentzer 1973, England and Fraser 1973, Shine and Lal 1977) were not read. Cross-checked against "
               "Jahangiri et al. 2019, Sci Rep (doi:10.1038/s41598-019-54575-3), Table 2 (Shine-Lal printed without the square there)",
    "DESIGN": "Project design of 7 Nov 2026 (written before the label audit)",
    "DERIVED": "logical consequence of earlier facts",
}


@dataclass(frozen=True)
class Thresholds:
    hgb_anemia: float = 13.0      # g/dL, anemia if HGB < this (primary). Sensitivity check: 12.0
    mcv_micro: float = 80.0       # fL, microcytic if MCV < this
    mcv_macro: float = 100.0      # fL, macrocytic if MCV > this
    mchc_hypo: float = 30.0       # g/dL, hypochromic if MCHC < this (primary). Sensitivity check: 32.0
    plt_low: float = 150.0        # 10^9/L, low if PLT < this
    wbc_high: float = 11.0        # 10^9/L, leukocytosis if WBC > this
    wbc_low: float = 4.0          # 10^9/L, leukopenia if WBC < this
    mentzer_cut: float = 13.0
    england_fraser_cut: float = 0.0
    shine_lal_cut: float = 1530.0


VARIANTS = {
    "primary": Thresholds(),
    "hgb12": Thresholds(hgb_anemia=12.0),
    "mchc32": Thresholds(mchc_hypo=32.0),
    "hgb12_mchc32": Thresholds(hgb_anemia=12.0, mchc_hypo=32.0),
}


@dataclass
class Rule:
    rid: str
    text: str
    needs: tuple[str, ...]                       # facts that must already be in memory
    when: Callable[[dict], bool]                 # condition on working memory
    adds: Callable[[dict], dict]                 # new facts
    source: str
    inputs: tuple[str, ...] = ()                 # values shown in the trace


@dataclass
class RuleResult:
    compatible: list[str]
    excluded: dict[str, list[str]]               # label -> rule ids that excluded it
    facts: dict
    trace: list[dict] = field(default_factory=list)
    refer_rule_fires: bool = False


def build_rules(t: Thresholds) -> list[Rule]:
    R: list[Rule] = []
    # ---- layer 1: findings ----
    R.append(Rule("F1", f"HGB < {t.hgb_anemia:g} g/dL -> anemia", ("HGB",),
                  lambda m: True, lambda m: {"anemia": m["HGB"] < t.hgb_anemia}, "WHO", ("HGB",)))
    R.append(Rule("F2", f"MCV < {t.mcv_micro:g} fL -> microcytic", ("MCV",),
                  lambda m: True, lambda m: {"microcytic": m["MCV"] < t.mcv_micro}, "MERCK_ANEMIA", ("MCV",)))
    R.append(Rule("F3", f"MCV > {t.mcv_macro:g} fL -> macrocytic", ("MCV",),
                  lambda m: True, lambda m: {"macrocytic": m["MCV"] > t.mcv_macro}, "MERCK_ANEMIA", ("MCV",)))
    R.append(Rule("F4", "neither microcytic nor macrocytic -> normocytic", ("microcytic", "macrocytic"),
                  lambda m: True, lambda m: {"normocytic": not m["microcytic"] and not m["macrocytic"]},
                  "DERIVED", ()))
    R.append(Rule("F5", f"MCHC < {t.mchc_hypo:g} g/dL -> hypochromic", ("MCHC",),
                  lambda m: True, lambda m: {"hypochromic": m["MCHC"] < t.mchc_hypo}, "MERCK_ANEMIA", ("MCHC",)))
    R.append(Rule("F6", f"PLT < {t.plt_low:g} x10^9/L -> low platelets", ("PLT",),
                  lambda m: True, lambda m: {"low_plt": m["PLT"] < t.plt_low}, "AAFP_PLT", ("PLT",)))
    R.append(Rule("F7", f"WBC > {t.wbc_high:g} x10^9/L -> leukocytosis", ("WBC",),
                  lambda m: True, lambda m: {"leukocytosis": m["WBC"] > t.wbc_high}, "AAFP_WBC", ("WBC",)))
    R.append(Rule("F8", f"WBC < {t.wbc_low:g} x10^9/L -> leukopenia", ("WBC",),
                  lambda m: True, lambda m: {"leukopenia": m["WBC"] < t.wbc_low}, "MERCK_WBC", ("WBC",)))
    R.append(Rule("F9", "leukocytosis or leukopenia -> abnormal WBC", ("leukocytosis", "leukopenia"),
                  lambda m: True, lambda m: {"abnormal_wbc": m["leukocytosis"] or m["leukopenia"]}, "DERIVED", ()))
    R.append(Rule("F10", "low platelets or abnormal WBC -> provisional rule-based REFER trigger",
                  ("low_plt", "abnormal_wbc"), lambda m: True,
                  lambda m: {"refer_rule": m["low_plt"] or m["abnormal_wbc"]}, "DESIGN", ()))
    # indices: only for microcytic anemia, trace only (they never exclude a label)
    R.append(Rule("I1", f"microcytic anemia -> Mentzer MCV/RBC (< {t.mentzer_cut:g} favours thalassemia trait)",
                  ("anemia", "microcytic"), lambda m: m["anemia"] and m["microcytic"],
                  lambda m: {"mentzer": (v := m["MCV"] / m["RBC"]), "mentzer_favours": favours(v, t.mentzer_cut)}, "INDICES", ("MCV", "RBC")))
    R.append(Rule("I2", f"microcytic anemia -> England-Fraser MCV - RBC - 5*HGB - 3.4 "
                        f"(< {t.england_fraser_cut:g} favours thalassemia trait)",
                  ("anemia", "microcytic"), lambda m: m["anemia"] and m["microcytic"],
                  lambda m: {"england_fraser": (v := m["MCV"] - m["RBC"] - 5 * m["HGB"] - 3.4),
                             "england_fraser_favours": favours(v, t.england_fraser_cut)}, "INDICES",
                  ("MCV", "RBC", "HGB")))
    R.append(Rule("I3", f"microcytic anemia -> Shine-Lal MCV^2 * MCH * 0.01 (< {t.shine_lal_cut:g} favours "
                        "thalassemia trait)",
                  ("anemia", "microcytic"), lambda m: m["anemia"] and m["microcytic"],
                  lambda m: {"shine_lal": (v := m["MCV"] ** 2 * m["MCH"] * 0.01),
                             "shine_lal_favours": favours(v, t.shine_lal_cut)}, "INDICES", ("MCV", "MCH")))
    # ---- layer 2: exclusions (only remove labels) ----
    ex = [
        ("X1", "anemia, low platelets or abnormal WBC -> not Healthy", ("anemia", "low_plt", "abnormal_wbc"),
         lambda m: m["anemia"] or m["low_plt"] or m["abnormal_wbc"], [HEALTHY], "DERIVED"),
        ("X2", "no anemia -> none of the five anemia labels", ("anemia",),
         lambda m: not m["anemia"], ANEMIA_LABELS, "WHO"),
        ("X3", "not microcytic -> not iron deficiency, not other microcytic", ("microcytic",),
         lambda m: not m["microcytic"], [IDA, OMA], "DESIGN"),
        ("X4", "not normocytic -> no normocytic label", ("normocytic",),
         lambda m: not m["normocytic"], [NHA, NNA], "MERCK_ANEMIA"),
        ("X5", "not macrocytic -> not macrocytic anemia", ("macrocytic",),
         lambda m: not m["macrocytic"], [MAC], "MERCK_ANEMIA"),
        ("X6", "not hypochromic -> not normocytic hypochromic", ("hypochromic",),
         lambda m: not m["hypochromic"], [NHA], "MERCK_ANEMIA"),
        ("X7", "hypochromic -> not normocytic normochromic", ("hypochromic",),
         lambda m: m["hypochromic"], [NNA], "MERCK_ANEMIA"),
        ("X8", "platelets not low -> no thrombocytopenia label", ("low_plt",),
         lambda m: not m["low_plt"], [THR, LTH], "AAFP_PLT"),
        ("X9", "WBC not abnormal -> no leukemia label", ("abnormal_wbc",),
         lambda m: not m["abnormal_wbc"], [LEU, LTH], "DESIGN"),
    ]
    for rid, text, needs, cond, labels, src in ex:
        R.append(Rule(rid, text, needs, cond,
                      (lambda labs, r: (lambda m: {f"excl:{r}": list(labs)}))(labels, rid), src))
    return R


REQUIRED = ("WBC", "RBC", "HGB", "MCV", "MCH", "MCHC", "PLT")


def favours(value: float, cut: float) -> str:
    """Hans et al. 2024, Table 1: below the cut favours thalassemia trait, above favours iron deficiency."""
    if value < cut:
        return "thalassemia trait"
    if value > cut:
        return "iron deficiency"
    return "borderline (equal to the cut-off)"


def run(record: Mapping, t: Thresholds = Thresholds()) -> RuleResult:
    """Forward chaining until no rule can add a fact. Expects a record that passed the validator."""
    missing = [c for c in REQUIRED if c not in record]
    if missing:
        raise KeyError(f"rule engine needs {missing}")
    mem: dict = {c: float(record[c]) for c in REQUIRED}
    rules = build_rules(t)
    fired: set[str] = set()
    trace: list[dict] = []
    changed = True
    while changed:                               # each pass fires every rule that is ready
        changed = False
        for r in rules:
            if r.rid in fired or not all(n in mem for n in r.needs):
                continue
            fired.add(r.rid)                     # a rule is evaluated once, when its inputs exist
            if not r.when(mem):
                continue
            new = r.adds(mem)
            mem.update(new)
            changed = True
            trace.append({"rule": r.rid, "text": r.text,
                          "inputs": {k: mem[k] for k in r.inputs},
                          "result": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in new.items()},
                          "source": SOURCES[r.source]})
    excluded: dict[str, list[str]] = {}
    for k, labs in mem.items():
        if isinstance(k, str) and k.startswith("excl:"):
            for lab in labs:
                excluded.setdefault(lab, []).append(k[5:])
    compatible = [c for c in CLASS_ORDER if c not in excluded]
    facts = {k: v for k, v in mem.items() if not (isinstance(k, str) and k.startswith("excl:")) and k not in REQUIRED}
    return RuleResult(compatible, excluded, facts, trace, bool(mem.get("refer_rule", False)))
