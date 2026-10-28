"""Anti-tampering checks for the retrieval ladder (spec section 3).

The executor implements against these tests; it may not change them.
LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # python-ingestion/
LOCK = ROOT / "tests" / "LOCK.sha256"

IMPLEMENTATION_MODULES = tuple(f"evals/{m}.py" for m in (
    "live_guard", "freeze_graph", "run_bm25", "fusion", "rerank", "topup",
    "ladder_stats", "latency_bench", "run_ladder", "score_ladder"))
# Test-control machinery has no business inside implementation code.
FORBIDDEN_IN_IMPLEMENTATION = ("pytest", "xfail", "importorskip", "sys.modules[", "unittest.mock")
SKIP_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__", ".pytest_cache"}


def locked_entries():
    entries = []
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, rel = line.split(None, 1)
            entries.append((digest, rel.strip()))
    return entries


def test_lock_file_exists_and_is_not_empty():
    assert LOCK.exists() and locked_entries()


@pytest.mark.parametrize("digest,rel", locked_entries() if LOCK.exists() else [])
def test_locked_file_is_unchanged(digest, rel):
    path = ROOT / rel
    assert path.exists(), f"locked file {rel} was deleted"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == digest, f"locked file {rel} was modified"


def test_every_ladder_test_file_is_locked():
    locked = {rel for _, rel in locked_entries()}
    ladder_tests = {f"tests/{p.name}" for p in (ROOT / "tests").glob("test_ladder_*.py")}
    assert ladder_tests <= locked, f"unlocked ladder tests: {sorted(ladder_tests - locked)}"
    assert "tests/conftest.py" in locked


def test_no_extra_conftest_anywhere():
    found = []
    for path in ROOT.rglob("conftest.py"):
        if not SKIP_DIRS & set(path.relative_to(ROOT).parts):
            found.append(str(path.relative_to(ROOT)))
    assert found == ["tests/conftest.py"], f"unexpected conftest files: {found}"


def test_no_pytest_config_overrides():
    for name in ("pytest.ini", "tox.ini", "setup.cfg"):
        assert not (ROOT / name).exists() and not (ROOT.parent / name).exists(), name
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[tool.pytest.ini_options]", 1)[-1].split("\n[", 1)[0]
    assert "addopts" not in section, "pytest addopts may not be set"
    assert not re.search(r"filterwarnings|xfail_strict|-p\s+no:", section)


@pytest.mark.parametrize("module", IMPLEMENTATION_MODULES)
def test_implementation_contains_no_test_control(module):
    path = ROOT / module
    if not path.exists():
        return  # not written yet; its own unit's tests are red
    src = path.read_text(encoding="utf-8")
    hits = [t for t in FORBIDDEN_IN_IMPLEMENTATION if t in src]
    assert hits == [], f"{module} contains {hits}"


def test_no_sitecustomize_shadowing():
    for d in [ROOT, ROOT / "evals", ROOT / "tests"]:
        for name in ("sitecustomize.py", "usercustomize.py"):
            assert not (d / name).exists(), f"{d / name} must not exist"


PRE_REGISTERED = (
    "ladder_s0_cosine->ladder_s1_cos_bm25",
    "ladder_s1_cos_bm25->ladder_s2_cos_bm25_rr",
    "ladder_s1_cos_bm25->ladder_s3_cos_bm25_graph",
    "ladder_s3_cos_bm25_graph->ladder_s4_cos_bm25_graph_rr",
    "ladder_s2_cos_bm25_rr->ladder_s4_cos_bm25_graph_rr",
    "ladder_s0_cosine->ladder_s4_cos_bm25_graph_rr",
)


def test_predictions_are_registered_for_every_comparison():
    import json
    preds = json.loads((ROOT / "evals" / "ladder_predictions.json").read_text(encoding="utf-8"))
    assert preds["registered_at"] == "2026-10-18"
    assert set(preds["comparisons"]) == set(PRE_REGISTERED)
    for key, p in preds["comparisons"].items():
        assert p["expected_verdict"] in ("improved", "regressed", "no detectable difference"), key
        assert p["reason"].strip(), key
