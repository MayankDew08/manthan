import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import GradeResult
import incremental_import as ii


CHAT = """14/08/2026, 10:30:00 AM Alice: Check this https://example.com/a
14/08/2026, 10:30:10 AM Bob: ok
14/08/2026, 10:30:20 AM Charlie: 👍
14/08/2026, 10:30:30 AM Dave: just a short note
"""


def _fake_grade(messages):
    out = []
    for d in messages:
        quality = 5 if "http" in d.text else 2
        out.append(GradeResult(
            quality=quality, confidence=1.0, category="resource",
            reason="test", topics=[], original_text=d.text,
        ))
    return out


@pytest.fixture
def chat_file(tmp_path):
    p = tmp_path / "chat.txt"
    p.write_text(CHAT, encoding="utf-8")
    return str(p)


@pytest.fixture
def db_file(tmp_path):
    return str(tmp_path / "import_state.sqlite")


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    monkeypatch.setattr(ii, "grade_messages", _fake_grade)
    monkeypatch.setattr(ii, "grade_pass2", lambda msgs: msgs)
    monkeypatch.setattr(ii, "process_candidates",
                        lambda candidates: ([], [], [], [], {}, set()))
    monkeypatch.setattr(ii, "push_to_stores", lambda *a, **k: {})


def test_first_run_processes_all_and_reports_outcomes(chat_file, db_file):
    summary = ii.run_incremental_import(
        "local:test-chat", chat_file, revision=1, db_path=db_file)

    assert summary["total_messages"] == 4
    assert summary["already_processed"] == 0
    assert summary["new_messages"] == 4
    assert summary["heuristically_dropped"] == 2   # "ok", "👍"
    assert summary["low_quality"] == 1             # "just a short note"
    assert summary["stored"] == 1                  # the link message
    assert (summary["heuristically_dropped"]
            + summary["low_quality"] + summary["stored"] == 4)


def test_second_run_is_idempotent(chat_file, db_file):
    ii.run_incremental_import("local:test-chat", chat_file, revision=1, db_path=db_file)
    second = ii.run_incremental_import(
        "local:test-chat", chat_file, revision=1, db_path=db_file)

    assert second["new_messages"] == 0
    assert second["already_processed"] == 4
    assert second["heuristically_dropped"] == 0
    assert second["low_quality"] == 0
    assert second["stored"] == 0


def test_processed_rows_recorded(chat_file, db_file):
    ii.run_incremental_import("local:test-chat", chat_file, revision=1, db_path=db_file)

    from import_state import ImportStateStore
    store = ImportStateStore(db_file)
    rows = store.connection.execute(
        "SELECT outcome, COUNT(*) AS c FROM processed_messages GROUP BY outcome"
    ).fetchall()
    tally = {r["outcome"]: r["c"] for r in rows}
    store.close()

    assert tally == {"heuristic_drop": 2, "low_quality": 1, "stored": 1}


def test_no_unseen_short_circuits_before_grading(chat_file, db_file, monkeypatch):
    ii.run_incremental_import("local:test-chat", chat_file, revision=1, db_path=db_file)
    calls = {"n": 0}
    original = _fake_grade

    def counting(messages):
        calls["n"] += 1
        return original(messages)

    monkeypatch.setattr(ii, "grade_messages", counting)
    ii.run_incremental_import("local:test-chat", chat_file, revision=1, db_path=db_file)
    assert calls["n"] == 0
