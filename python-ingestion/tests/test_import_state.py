import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser import Data
from import_state import ImportStateStore, identify_messages


def test_same_message_same_id():
    m = [Data(datetime_iso="2026-08-28T10:30:00", sender="Alice",
             text="Hello", is_media=False)]
    first = identify_messages("local:test-chat", m)
    second = identify_messages("local:test-chat", m)
    assert first[0].message_id == second[0].message_id


def test_different_messages_different_ids():
    msgs = [
        Data("2026-08-28T10:30:00", "Alice", "Hello", False),
        Data("2026-08-28T10:31:00", "Alice", "Goodbye", False),
    ]
    identified = identify_messages("local:test-chat", msgs)
    assert identified[0].message_id != identified[1].message_id


def test_duplicate_messages_separate_ids_and_occurrence():
    dup = [
        Data("2026-08-28T10:30:00", "Alice", "Yes", False),
        Data("2026-08-28T10:30:00", "Alice", "Yes", False),
    ]
    dup_id = identify_messages("local:test-chat", dup)
    assert dup_id[0].message_id != dup_id[1].message_id
    assert dup_id[0].occurrence == 1
    assert dup_id[1].occurrence == 2


def test_different_source_separate_ids():
    g1 = identify_messages("gdrive:file-one",
                           [Data("2026-08-28T10:30:00", "Alice", "Hello", False)])
    g2 = identify_messages("gdrive:file-two",
                           [Data("2026-08-28T10:30:00", "Alice", "Hello", False)])
    assert g1[0].message_id != g2[0].message_id


def test_append_keeps_previous_ids():
    original = [
        Data("2026-08-28T10:30:00", "Alice", "A", False),
        Data("2026-08-28T10:31:00", "Alice", "B", False),
    ]
    updated = original + [Data("2026-08-28T10:32:00", "Alice", "C", False)]
    first = identify_messages("local:test-chat", original)
    second = identify_messages("local:test-chat", updated)
    assert first[0].message_id == second[0].message_id
    assert first[1].message_id == second[1].message_id


def test_find_unseen_messages_filters_processed():
    store = ImportStateStore(db_path=":memory:")
    store.upsert_source("local:test-chat", "local", "chat.txt")

    all_ids = [
        "id-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "id-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "id-cccccccccccccccccccccccccccccccc",
    ]
    # First run: none processed yet -> all unseen
    assert store.find_unseen_messages("local:test-chat", all_ids) == all_ids

    # Mark two as processed, then only the third should remain unseen
    store.mark_processed("local:test-chat", [
        (all_ids[0], "stored"),
        (all_ids[1], "low_quality"),
    ])
    assert store.find_unseen_messages("local:test-chat", all_ids) == [all_ids[2]]


def test_find_unseen_messages_chunks_large_input():
    # Exceeds SQLite's default 999 host-parameter limit to exercise chunking.
    store = ImportStateStore(db_path=":memory:")
    store.upsert_source("local:big", "local", "big.txt")

    ids = [f"id-{i:04d}" for i in range(1500)]
    # None processed yet -> all 1500 unseen
    assert len(store.find_unseen_messages("local:big", ids)) == 1500

    # Mark the first 500 as processed; the remaining 1000 must come back unseen.
    store.mark_processed("local:big", [(mid, "stored") for mid in ids[:500]])
    unseen = store.find_unseen_messages("local:big", ids)
    assert len(unseen) == 1000
    assert set(unseen) == set(ids[500:])


def test_mark_processed_records_outcome_and_is_idempotent():
    store = ImportStateStore(db_path=":memory:")
    store.upsert_source("local:test-chat", "local", "chat.txt")

    store.mark_processed("local:test-chat", [
        ("id-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "stored"),
        ("id-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "low_quality"),
        ("id-cccccccccccccccccccccccccccccccc", "heuristic_drop"),
    ])

    rows = {
        r["message_id"]: r["outcome"]
        for r in store.connection.execute(
            "SELECT message_id, outcome FROM processed_messages"
        ).fetchall()
    }
    assert rows == {
        "id-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": "stored",
        "id-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": "low_quality",
        "id-cccccccccccccccccccccccccccccccc": "heuristic_drop",
    }

    # Re-marking with a different outcome updates the row (idempotent upsert)
    store.mark_processed("local:test-chat",
                        [("id-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "stored")])
    updated = store.connection.execute(
        "SELECT outcome FROM processed_messages WHERE message_id = ?",
        ("id-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",),
    ).fetchone()
    assert updated["outcome"] == "stored"


def test_get_source_returns_metadata_and_none_when_missing():
    store = ImportStateStore(db_path=":memory:")
    assert store.get_source("local:missing") is None

    store.upsert_source("local:chat", "local", "chat.txt")
    row = store.get_source("local:chat")
    assert row is not None
    assert row["source_id"] == "local:chat"
    assert row["source_type"] == "local"
    assert row["file_name"] == "chat.txt"
    # Revision is owned solely by update_source_revision; upsert sets no
    # revision, so a freshly registered source has none yet.
    assert row["revision"] is None


def test_update_source_revision_sets_revision_and_import_time():
    store = ImportStateStore(db_path=":memory:")
    store.upsert_source("local:chat", "local", "chat.txt")

    store.update_source_revision("local:chat", "rev-2")
    row = store.get_source("local:chat")
    assert row is not None
    assert row["revision"] == "rev-2"
    assert row["imported_at"] is not None


def test_close_releases_connection():
    store = ImportStateStore(db_path=":memory:")
    store.close()
    # Connection is closed; a fresh query must fail.
    try:
        store.connection.execute("SELECT 1")
        assert False, "expected closed connection error"
    except sqlite3.ProgrammingError:
        pass
