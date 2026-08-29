import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import GradeResult
from parser import parse_chat
from heuristic_filter import heuristic_filter
import incremental_import as ii


CHAT = """14/08/2026, 10:30:00 AM Alice: Check this https://example.com/a
14/08/2026, 10:30:10 AM Bob: ok
14/08/2026, 10:30:20 AM Charlie: 👍
14/08/2026, 10:30:30 AM Dave: just a short note
"""

DUPLICATE_CHAT = """14/08/2026, 10:30:00 AM Alice: ok
14/08/2026, 10:30:10 AM Alice: ok
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
    # run_pipeline is the existing graph entry; we avoid the real LLM/scrape by
    # re-parsing the temp file, re-running the real heuristic filter, and faking
    # grades. This keeps the returned `final` aligned with the code's kept_unseen.
    def fake_run_pipeline(chat_path):
        msgs = parse_chat(chat_path)
        kept, _ = heuristic_filter(msgs)
        return {
            "kept": kept,
            "final": _fake_grade(kept),
            "enriched": [],
            "scraped": [],
            "blocked": [],
            "ask_user": [],
            "stats": {},
            "chat_path": chat_path,
        }

    monkeypatch.setattr(ii, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(ii, "build_records_node",
                        lambda state: {"enriched": state.get("enriched") or []})
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


def test_no_unseen_short_circuits_before_graph(chat_file, db_file, monkeypatch):
    ii.run_incremental_import("local:test-chat", chat_file, revision=1, db_path=db_file)
    calls = {"n": 0}

    def counting(chat_path):
        calls["n"] += 1
        return {
            "kept": [], "final": [], "enriched": [], "scraped": [],
            "blocked": [], "ask_user": [], "stats": {}, "chat_path": chat_path,
        }

    monkeypatch.setattr(ii, "run_pipeline", counting)
    ii.run_incremental_import("local:test-chat", chat_file, revision=1, db_path=db_file)
    assert calls["n"] == 0


def test_duplicate_messages_get_distinct_ids(tmp_path, db_file):
    p = tmp_path / "dup.txt"
    p.write_text(DUPLICATE_CHAT, encoding="utf-8")
    summary = ii.run_incremental_import(
        "local:dup", str(p), revision=1, db_path=db_file)

    assert summary["total_messages"] == 2
    assert summary["new_messages"] == 2
    assert summary["heuristically_dropped"] == 2

    from import_state import ImportStateStore, identify_messages
    datas = parse_chat(str(p))
    ids = [m.message_id for m in identify_messages("local:dup", datas)]
    assert len(ids) == 2 and ids[0] != ids[1]

    store = ImportStateStore(db_file)
    rows = store.connection.execute(
        "SELECT message_id, outcome FROM processed_messages WHERE source_id = ?",
        ("local:dup",),
    ).fetchall()
    store.close()
    recorded_ids = {r["message_id"] for r in rows}
    assert recorded_ids == set(ids)
    assert all(r["outcome"] == "heuristic_drop" for r in rows)


def test_revision_not_advanced_when_persistence_fails(tmp_path, db_file, monkeypatch):
    p = tmp_path / "chat.txt"
    p.write_text(CHAT, encoding="utf-8")

    captured = {}
    real_store_cls = ii.ImportStateStore

    def capture_store(*a, **k):
        inst = real_store_cls(*a, **k)
        captured["store"] = inst
        return inst

    monkeypatch.setattr(ii, "ImportStateStore", capture_store)

    def boom(*a, **k):
        raise RuntimeError("store down")

    monkeypatch.setattr(ii, "push_to_stores", boom)

    with pytest.raises(RuntimeError):
        ii.run_incremental_import("local:fail", str(p), revision=1, db_path=db_file)

    from import_state import ImportStateStore
    store = ImportStateStore(db_file)
    row = store.get_source("local:fail")
    processed = store.connection.execute(
        "SELECT COUNT(*) AS c FROM processed_messages WHERE source_id = ?",
        ("local:fail",),
    ).fetchone()["c"]
    store.close()

    assert row is not None
    assert row["revision"] is None
    assert processed == 0

    # B1: the connection created inside run_incremental_import must be released
    # even though persistence failed.
    with pytest.raises(sqlite3.ProgrammingError):
        captured["store"].connection.execute("SELECT 1")


def test_zero_new_advances_revision(chat_file, db_file, monkeypatch):
    calls = {"n": 0}

    def counting(chat_path):
        calls["n"] += 1
        msgs = parse_chat(chat_path)
        kept, _ = heuristic_filter(msgs)
        return {
            "kept": kept,
            "final": _fake_grade(kept),
            "enriched": [],
            "scraped": [],
            "blocked": [],
            "ask_user": [],
            "stats": {},
            "chat_path": chat_path,
        }

    monkeypatch.setattr(ii, "run_pipeline", counting)

    ii.run_incremental_import("local:zero", chat_file, revision=1, db_path=db_file)
    assert calls["n"] == 1

    # New revision, but no unseen messages -> run_pipeline must NOT run again,
    # and the revision must still be recorded (B2).
    summary = ii.run_incremental_import("local:zero", chat_file, revision=2, db_path=db_file)
    assert calls["n"] == 1
    assert summary["new_messages"] == 0

    from import_state import ImportStateStore
    store = ImportStateStore(db_file)
    row = store.get_source("local:zero")
    assert row is not None
    store.close()
    assert row["revision"] == "2"
