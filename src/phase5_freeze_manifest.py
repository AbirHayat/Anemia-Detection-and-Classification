"""Freeze manifest for Phase 6: SHA-256 of every file the single test run depends on.

Research prototype, not a diagnostic tool. Written once at the end of Phase 5, committed and tagged
`phase6-freeze`. src/phase6_final_test.py refuses to read the test rows if any of these files changed.

Output: results/phase5/freeze_manifest.json
Run:    python src/phase5_freeze_manifest.py
"""
from __future__ import annotations

import json
from datetime import date

from agent import sha_file
from data import ROOT

FILES = [
    "data/diagnosed_cbc_data_v4.csv",
    "results/split/dev_test_split.csv",
    "results/phase1/dev_rows.csv",
    "results/phase2/cascade_frozen.json",
    "results/phase4/frozen_phase6.json",
    "results/phase4/h1_summary.csv",
    "results/phase4/h2_summary.csv",
    "results/phase5/models/forest_smote_Fcore.joblib",
    "results/phase5/models/manifest.json",
    "notes/phase6_preregistration.md",
] + [f"src/{f}" for f in ["data.py", "validator.py", "rules.py", "ml_pipelines.py", "arbitration.py", "explain.py",
                          "agent.py", "stress.py", "metrics.py", "stats_tests.py", "phase2_rules_baseline.py",
                          "phase6_final_test.py"]]


def main() -> None:
    man = {"written": str(date.today()), "git_tag": "phase6-freeze",
           "note": "Every file the Phase 6 test run depends on. The run refuses to start if one of them changed.",
           "files": {f: sha_file(ROOT / f) for f in FILES}}
    out = ROOT / "results" / "phase5" / "freeze_manifest.json"
    out.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    for f, h in man["files"].items():
        print(f"{h[:16]}  {f}")


if __name__ == "__main__":
    main()
