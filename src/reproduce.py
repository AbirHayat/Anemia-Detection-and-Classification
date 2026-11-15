"""Helpers for notebooks/anemia_project.ipynb: setup checks, a full recompute in a scratch copy, comparison with
the committed results/, read-only checks of the committed files, and display helpers.

Research prototype, not a diagnostic tool.

No project logic lives here. A full recompute runs the existing scripts in src/ unchanged, inside a temporary
copy of the repository (src/, data/ and the frozen split file), so the committed results/ are never overwritten.
Then every regenerated file is compared with its committed version.

Test-set safety: PIPELINE never contains the Phase 6 script, run_script() refuses any script outside PIPELINE,
and no function here calls load_test. The Phase 6 helpers only read the committed results/phase6/ files.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata as im
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from data import RAW, RAW_SHA256, ROOT, sha256_of

KAGGLE_URL = "https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification"
PYTHON_TESTED = "3.12"
PIPELINE: dict[int, list[tuple[str, ...]]] = {
    1: [("phase1_clean.py",), ("phase1_data_report.py",)],
    2: [("phase2_rules_baseline.py",)],
    3: [("phase3_ml_only_cv.py",), ("phase3_h3_h4_cv.py",), ("phase3_amended_calibration.py",),
        ("phase3_evaluate.py",), ("phase3_evaluate.py", "--version", "amended"), ("phase3_compare_calibration.py",),
        ("phase3_calibration_diagnostic.py",)],
    4: [("phase4_inner_cv.py",), ("phase4_evaluate.py",), ("phase4_variants_table.py",)],
    5: [("phase5_final_models.py",), ("phase5_run.py",)],
}
# files written by the scripts that are logs or timings only (never compared)
NOT_COMPARED = re.compile(r"(run_log|stdout|pytest_output|latency\.csv$)")
# timing text inside otherwise compared files (masked before comparing)
MASKS = [re.compile(p, re.M) for p in [
    r"^run time .*$", r"stress suite time \d+ s", r"(max error [\d.e+-]+), [\d.]+ s$",
    r"^\s+(validation|rules|ml|arbitration|explanation|audit|total)\s+median .*$", r"amended calibration run: (\d+) tasks, \d+ s"]]


class SetupError(RuntimeError):
    """Raised when the environment or the dataset is not as required."""


# ------------------------------------------------------------------ setup
def check_python() -> str:
    v = ".".join(map(str, sys.version_info[:3]))
    ok = v.startswith(PYTHON_TESTED)
    return f"Python {v} ({'tested version' if ok else 'NOT the tested ' + PYTHON_TESTED + '.x, numbers may differ'})"


def check_packages() -> pd.DataFrame:
    rows = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, want = line.strip().split("==")
            try:
                have = im.version(name)
            except im.PackageNotFoundError:
                have = "missing"
            rows.append({"package": name, "required": want, "installed": have,
                         "status": "ok" if have == want else ("MISSING" if have == "missing" else "different")})
    df = pd.DataFrame(rows)
    if (df.status == "MISSING").any():
        raise SetupError("missing packages: " + ", ".join(df[df.status == "MISSING"].package)
                         + ". Run: pip install -r requirements.txt")
    return df


def check_dataset() -> str:
    """Runs src/check_dataset.py. Stops with a clear message if the file is missing or different."""
    r = subprocess.run([sys.executable, str(ROOT / "src" / "check_dataset.py")], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SetupError("dataset check failed:\n" + r.stdout.strip() + "\n\nDownload diagnosed_cbc_data_v4.csv from "
                         + KAGGLE_URL + " and place it at data/diagnosed_cbc_data_v4.csv (the notebook does not "
                         "download it).")
    return r.stdout.strip()


# ------------------------------------------------------------------ full recompute in a scratch copy
class Workspace:
    """A temporary copy of src/, data/ and the frozen split file. Scripts write their results there."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="anemia_recompute_"))
        shutil.copytree(ROOT / "src", self.root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data", self.root / "data")
        (self.root / "results" / "split").mkdir(parents=True)
        shutil.copy2(ROOT / "results" / "split" / "dev_test_split.csv", self.root / "results" / "split")
        self.seconds: dict[str, float] = {}

    def run_script(self, *args: str) -> None:
        if args not in [s for steps in PIPELINE.values() for s in steps]:
            raise PermissionError(f"{args[0]} is not a development-side pipeline step")
        t0 = time.perf_counter()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([sys.executable, str(self.root / "src" / args[0]), *args[1:]], cwd=self.root,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        self.seconds[" ".join(args)] = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(args)} failed:\n{r.stderr[-3000:]}")

    def run_phase(self, phase: int) -> pd.DataFrame:
        before = sum(self.seconds.values())
        for step in PIPELINE[phase]:
            self.run_script(*step)
        print(f"Phase {phase} recomputed in {sum(self.seconds.values()) - before:.0f} s")
        return compare_phase(self.root, phase)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _norm_text(rel: str, raw: bytes) -> str:
    if rel.endswith(".gz"):
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    for m in MASKS:
        text = m.sub(lambda x: x.group(1) if x.groups() and x.group(1) else "<timing>", text)
    return text


def _same_csv_without_timing(a: bytes, b: bytes) -> bool:
    """Same rows apart from timing columns (parallel jobs may write rows in a different order)."""
    fa, fb = pd.read_csv(io.BytesIO(a)), pd.read_csv(io.BytesIO(b))
    keep = [c for c in fa.columns if "second" not in c.lower()]
    if list(fa.columns) != list(fb.columns):
        return False
    sa = fa[keep].sort_values(keep).reset_index(drop=True)
    sb = fb[keep].sort_values(keep).reset_index(drop=True)
    return sa.equals(sb)


def _short_hashes(path: Path) -> set[str]:
    return {h[:12] for h in _sha_variants(path)} | {h[:16] for h in _sha_variants(path)}


def _compare_audit_log(a: bytes, b: bytes, work_root: Path) -> str:
    """Audit records must be identical except the code fingerprints in 'versions', which depend on the line
    endings git used at checkout. Those fingerprints are then checked against the source files, LF or CRLF."""
    ra = [json.loads(x) for x in a.decode("utf-8").splitlines() if x.strip()]
    rb = [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()]
    strip = lambda r: {k: v for k, v in r.items() if k != "versions"}  # noqa: E731
    if len(ra) != len(rb) or any(strip(x) != strip(y) for x, y in zip(ra, rb)):
        return "DIFFERS: audit records"
    if a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"):
        return "identical after line endings and timing are ignored"
    for f, h in ra[0]["versions"]["code_sha256"].items():
        if h not in _short_hashes(work_root / "src" / f):
            return f"DIFFERS: stored fingerprint of src/{f}"
    return "identical records, code fingerprints differ only by checkout line endings"


def _same_model(a: Path, b: Path) -> bool:
    import joblib
    import numpy as np
    from data import F_CORE, load_dev
    X = load_dev(clean=True)[F_CORE].to_numpy(float)
    return bool(np.array_equal(joblib.load(a).predict_proba(X), joblib.load(b).predict_proba(X)))


def compare_phase(work_root: Path, phase: int) -> pd.DataFrame:
    """Compare every file the recompute wrote for this phase with the committed one."""
    new_dir, old_dir = work_root / "results" / f"phase{phase}", ROOT / "results" / f"phase{phase}"
    rows, model_same = [], {}
    for p in sorted(x for x in new_dir.rglob("*") if x.is_file()):
        rel = p.relative_to(work_root / "results").as_posix()
        old = ROOT / "results" / rel
        if NOT_COMPARED.search(rel):
            rows.append((rel, "log or timing, not compared"))
            continue
        if not old.exists():
            rows.append((rel, "DIFFERS: not in committed results"))
            continue
        a, b = old.read_bytes(), p.read_bytes()
        if a == b:
            status = "identical"
        elif rel.endswith(".joblib"):
            same = _same_model(old, p)
            model_same[p.name] = same
            status = "same predictions (pickle bytes differ)" if same else "DIFFERS: predictions"
        elif rel.endswith(".png"):
            status = "figure bytes differ"
        elif rel.endswith(".jsonl"):
            status = _compare_audit_log(a, b, work_root)
        elif rel.endswith((".csv",)) and "second" in a.decode("utf-8", "replace").split("\n", 1)[0].lower():
            status = "identical except timing column" if _same_csv_without_timing(a, b) else "DIFFERS"
        else:
            ta, tb = _norm_text(rel, a), _norm_text(rel, b)
            if rel.endswith("models/manifest.json") and ta != tb:
                ja, jb = json.loads(ta), json.loads(tb)
                for d in (ja, jb):
                    for m in d["models"].values():
                        m.pop("sha256", None)
                ta, tb = json.dumps(ja, sort_keys=True), json.dumps(jb, sort_keys=True)
            if ta == tb:
                status = "identical after line endings and timing are ignored"
            else:
                status = "DIFFERS"
        rows.append((rel, status))
    for p in sorted(x for x in old_dir.rglob("*") if x.is_file()):
        rel = p.relative_to(ROOT / "results").as_posix()
        if not (work_root / "results" / rel).exists():
            rows.append((rel, "committed only (not written by the phase scripts)"))
    return pd.DataFrame(rows, columns=["file", "status"])


def assert_reproduced(report: pd.DataFrame, allow_figures: bool = True) -> None:
    bad = report[report.status.str.startswith("DIFFERS")]
    if not allow_figures:
        bad = pd.concat([bad, report[report.status == "figure bytes differ"]])
    if len(bad):
        raise AssertionError("recompute differs from the committed results:\n" + bad.to_string(index=False))
    print(f"{len(report)} files: " + ", ".join(f"{k} {v}" for k, v in report.status.value_counts().items()))


# ------------------------------------------------------------------ read-only checks of the committed files
def _txt(rel: str) -> str:
    return (ROOT / "results" / rel).read_text(encoding="utf-8")


def _num(pat: str, rel: str) -> str:
    m = re.search(pat, _txt(rel), re.S)
    if not m:
        raise AssertionError(f"pattern not found in results/{rel}: {pat}")
    return m.group(1)


def check_phase1() -> pd.DataFrame:
    from data import CLASS_ORDER, load_dev
    dev = load_dev(clean=True)
    rows = pd.read_csv(ROOT / "results" / "phase1" / "dev_rows.csv")
    kept, rejected = int(_num(r"rows kept: (\d+)", "phase1/cleaning_report.txt")), int(
        _num(r"rows rejected: (\d+) of", "phase1/cleaning_report.txt"))
    assert kept == len(dev) == int(rows["valid"].sum()), "kept rows"
    assert rejected == int((~rows["valid"]).sum()), "rejected rows"
    counts = dev["Diagnosis"].value_counts().reindex(CLASS_ORDER)
    for c, n in counts.items():
        assert int(_num(r"== 6\..*?\n  " + re.escape(c) + r": (\d+)", "phase1/cleaning_report.txt")) == n, c
    cascade = _num(r"merged.: \d+ / \d+ \(([\d.]+)%\)", "phase1/data_report.txt")
    print(f"development rows {len(rows)}, rejected {rejected}, kept {kept}, label cascade reproduces "
          f"{cascade}% of development labels")
    return counts.rename("clean development rows").to_frame()


def check_phase2() -> pd.DataFrame:
    import arbitration as A
    from data import CLASS_ORDER
    sets = pd.read_csv(ROOT / "results" / "phase2" / "rule_sets_dev.csv")
    out = []
    for v in ["primary", "hgb12", "mchc32", "hgb12_mchc32"]:
        M = A.set_mask(sets[f"set_{v}"].fillna(""))
        y = sets["Diagnosis"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
        hit = int(M[range(len(y)), y].sum())
        assert hit == int(_num(rf"variant '{v}'.*?rule-set coverage: (\d+) /", "phase2/rules_baseline.txt")), v
        out.append({"thresholds": v, "rule-set coverage": f"{hit}/{len(y)} ({100 * hit / len(y):.1f}%)",
                    "mean set size": round(float(M.sum(1).mean()), 2)})
    return pd.DataFrame(out)


def check_phase3() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "results" / "phase3" / "amended" / "summary_metrics_cal.csv", header=[0, 1], index_col=[0, 1, 2])
    f1 = s.loc[("F-core", "forest", "smote"), ("macro_f1", "mean")]
    saved = float(_num(r"== P3\..*?forest\s+\(smote\) ([\d.]+)", "phase3/amended/evaluation.txt"))
    assert f"{f1:.4f}" == f"{saved:.4f}", "summary table and evaluation report disagree"
    print(f"H2 model (pre-declared rule): random forest + SMOTE, F-core macro-F1 {f1:.4f}")
    return s.xs("F-core", level=0)[[("macro_f1", "mean"), ("macro_f1", "std")]].round(3)


def check_phase4() -> dict:
    fz = json.loads(_txt("phase4/frozen_phase6.json"))
    h1 = pd.read_csv(ROOT / "results" / "phase4" / "h1_summary.csv")
    h2 = pd.read_csv(ROOT / "results" / "phase4" / "h2_summary.csv")
    r = h1[(h1.variant == "primary") & (h1.fset == "F-core") & (h1.model == "forest")].iloc[0]
    assert f"{r['diff']:+.3f}" == _num(r"thresholds HGB 13, MCHC 30 \(main\) --.*?F-core random forest\s+\S+ ± \S+\s+\S+ ± \S+\s+([+-][\d.]+)",
                                       "phase4/evaluation.txt")
    m = h2[(h2.variant == "primary") & (h2.floor.round(2) == 0.70)].iloc[0]
    assert f"{m.sel_f1_hybrid:.3f}" == _num(r"FORMAL VERDICT\].*?hybrid ([\d.]+), confidence", "phase4/evaluation.txt")
    assert fz["refer_trigger"] == _num(r"CHOSEN TRIGGER: (\w+)", "phase4/evaluation.txt")
    assert f"{fz['tau']['primary']:.2f}" == _num(r"tau \(70% floor\): HGB 13, MCHC 30 \(main\) ([\d.]+)", "phase4/evaluation.txt")
    return {"H1 supported (main thresholds)": int(h1[(h1.variant == "primary") & (h1.fset == "F-core")].supported.sum()),
            "H2 passes": bool(m.passes), "REFER trigger": fz["refer_trigger"], "tau (main)": fz["tau"]["primary"]}


def check_phase5() -> pd.DataFrame:
    import warnings
    with warnings.catch_warnings():                # shap's optional progress-bar warning names local paths
        warnings.simplefilter("ignore")
        from agent import check_consistency
    recs = [json.loads(ln) for ln in _txt("phase5/audit_log_dev.jsonl").splitlines() if ln.strip()]
    passed = sum(1 for r in recs if not check_consistency(r))
    assert f"{passed}/{len(recs)} pass" in _txt("phase5/phase5_report.txt"), "consistency count"
    bad = pd.read_csv(ROOT / "results" / "phase5" / "stress" / "bad_input.csv")
    ref = pd.read_csv(ROOT / "results" / "phase5" / "stress" / "referral.csv")
    met = ref[ref.trigger_met]
    rep = _txt("phase5/phase5_report.txt")
    assert f"{int(bad.caught.sum())}/{len(bad)} injected values caught" in rep
    assert f"{int(met.referred.sum())}/{len(met)} returned REFER" in rep
    return pd.DataFrame([{"check": "consistency of saved audit records (recomputed)", "result": f"{passed}/{len(recs)}"},
                         {"check": "bad input caught", "result": f"{int(bad.caught.sum())}/{len(bad)}"},
                         {"check": "referral stress cases referred", "result": f"{int(met.referred.sum())}/{len(met)}"}])


# ------------------------------------------------------------------ Phase 6: read-only
def _sha_variants(path: Path) -> set[str]:
    raw = path.read_bytes()
    out = {hashlib.sha256(raw).hexdigest()}
    if b"\0" not in raw[:8000]:                    # text: accept LF or CRLF checkouts of the same content
        lf = raw.replace(b"\r\n", b"\n")
        out |= {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}
    return out


def check_freeze_manifest() -> pd.DataFrame:
    """Every file the final test depended on, against results/phase5/freeze_manifest.json."""
    man = json.loads(_txt("phase5/freeze_manifest.json"))
    rows = []
    for rel, h in man["files"].items():
        p = ROOT / rel
        if not p.exists():
            status = "MISSING"
        elif h in _sha_variants(p):
            status = "matches"
        elif rel == "results/split/dev_test_split.csv" and list(pd.read_csv(p, nrows=1).columns) == ["raw_row", "set"]:
            split, dev = pd.read_csv(p), pd.read_csv(ROOT / "results" / "phase1" / "dev_rows.csv")
            ok = (split.set == "dev").sum() == len(dev) and set(dev.raw_row) == set(split.raw_row[split.set == "dev"])
            status = "public reduced copy (row number + set), consistent with dev_rows.csv" if ok else "DIFFERS"
        else:
            status = "DIFFERS"
        rows.append({"file": rel, "status": status})
    df = pd.DataFrame(rows)
    bad = df[df.status.isin(["MISSING", "DIFFERS"])]
    if len(bad):
        raise AssertionError("frozen files changed since the final test:\n" + bad.to_string(index=False))
    return df


def phase6_available() -> bool:
    return (ROOT / "results" / "phase6" / "final_test_report.txt").exists()


def phase6_summary() -> dict:
    rep = _txt("phase6/final_test_report.txt")
    assert "phase6-freeze" in rep, "report does not name the freeze tag"
    t = pd.read_csv(ROOT / "results" / "phase6" / "acceptance_targets.csv")
    s = pd.read_csv(ROOT / "results" / "phase6" / "systems.csv")
    forest = s[s.system == "random forest"].iloc[0]
    assert f"{forest.macro_f1:9.3f}" in rep, "systems.csv and the report disagree"
    return {"report": rep, "targets": t, "systems": s}


def show_md(rel: str):
    from IPython.display import Markdown
    text = _txt(rel)
    first = text.split("\n\n", 1)
    return Markdown(first[1] if len(first) == 2 and not first[0].startswith(("|", "#")) else text)


def show_text(rel: str, start: str | None = None, stop: str | None = None) -> None:
    text = _txt(rel)
    if start:
        text = text[text.index(start):]
    if stop and stop in text[1:]:
        text = text[: text.index(stop, 1)]
    print(text.rstrip())


def figure(rel: str, width: int = 700):
    from IPython.display import Image
    return Image(filename=str(ROOT / "results" / rel), width=width)
