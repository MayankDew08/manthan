"""A re-import must never reuse (or silently accept) a stale graph run.

Regression for 2026-10-09: the graph thread was derived from source_id+revision,
so after revisions were reset a re-import resumed an unfinished morning run whose
texts were empty, recording 148 bogus "low quality" outcomes in 12 seconds. The
count-only guard passed because 148 == 148.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import GradeResult
from heuristic_filter import heuristic_filter
from parser import parse_chat
import incremental_import as ii

CHAT_A = "14/08/2026, 10:30:00 AM Alice: Check this https://example.com/a\n"
CHAT_B = "14/08/2026, 10:30:00 AM Alice: A completely different message https://example.com/b\n"


def _grade(kept, text_override=None):
    return [GradeResult(quality=5, confidence=1.0, category="resource", reason="t",
                        topics=[], original_text=d.text if text_override is None
                        else text_override) for d in kept]


@pytest.fixture
def seen_paths(monkeypatch):
    paths = []

    def fake_run_pipeline(chat_path):
        paths.append(chat_path)
        kept, _ = heuristic_filter(parse_chat(chat_path))
        return {"kept": kept, "final": _grade(kept), "enriched": [], "scraped": [],
                "blocked": [], "ask_user": [], "stats": {}, "chat_path": chat_path}

    monkeypatch.setattr(ii, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(ii, "build_records_node",
                        lambda state: {"enriched": state.get("enriched") or []})
    monkeypatch.setattr(ii, "push_to_stores", lambda *a, **k: {})
    return paths


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _wipe_processed(db):
    """What the sender migration did: forget outcomes, keep the source row."""
    con = sqlite3.connect(db)
    with con:
        con.execute("DELETE FROM processed_messages")
        con.execute("UPDATE ingestion_sources SET revision = NULL")
    con.close()


def test_different_content_same_source_and_revision_gets_a_new_graph_thread(tmp_path, seen_paths):
    db = str(tmp_path / "s.sqlite")
    ii.run_incremental_import("gdrive:x", _write(tmp_path, "a.txt", CHAT_A), revision="r1", db_path=db)
    _wipe_processed(db)
    ii.run_incremental_import("gdrive:x", _write(tmp_path, "b.txt", CHAT_B), revision="r1", db_path=db)
    assert len(seen_paths) == 2 and seen_paths[0] != seen_paths[1]


def test_identical_content_keeps_the_same_thread_so_crash_resume_still_works(tmp_path, seen_paths):
    db = str(tmp_path / "s.sqlite")
    ii.run_incremental_import("gdrive:x", _write(tmp_path, "a.txt", CHAT_A), revision="r1", db_path=db)
    _wipe_processed(db)
    ii.run_incremental_import("gdrive:x", _write(tmp_path, "a2.txt", CHAT_A), revision="r1", db_path=db)
    assert len(seen_paths) == 2 and seen_paths[0] == seen_paths[1]


def test_grades_for_other_texts_are_rejected_even_when_counts_match(tmp_path, monkeypatch):
    def stale_run_pipeline(chat_path):
        kept, _ = heuristic_filter(parse_chat(chat_path))
        return {"kept": kept, "final": _grade(kept, text_override=""), "enriched": [],
                "scraped": [], "blocked": [], "ask_user": [], "stats": {},
                "chat_path": chat_path}

    monkeypatch.setattr(ii, "run_pipeline", stale_run_pipeline)
    monkeypatch.setattr(ii, "push_to_stores", lambda *a, **k: {})
    db = str(tmp_path / "s.sqlite")

    with pytest.raises(RuntimeError, match="do not match"):
        ii.run_incremental_import("gdrive:x", _write(tmp_path, "a.txt", CHAT_A),
                                  revision="r1", db_path=db)

    con = sqlite3.connect(db)
    processed = con.execute("SELECT COUNT(*) FROM processed_messages").fetchone()[0]
    revision = con.execute("SELECT revision FROM ingestion_sources").fetchone()[0]
    con.close()
    assert processed == 0 and revision is None
