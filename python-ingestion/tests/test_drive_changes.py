import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drive_changes import (
    initialize_change_tracking,
    fetch_changes_since_saved_token,
    is_relevant_drive_change,
)
from import_state import ImportStateStore

ACCOUNT = "me@example.com"
FOLDER = "folder-123"


def make_service(start_token, pages, fail_on_token=None):
    svc = Mock()
    svc.about.return_value.get.return_value.execute.return_value = {
        "user": {"emailAddress": ACCOUNT},
    }
    svc.changes.return_value.getStartPageToken.return_value.execute.return_value = {
        "startPageToken": start_token,
    }

    def _execute():
        tok = svc.changes.return_value.list.call_args.kwargs["pageToken"]
        if fail_on_token is not None and tok == fail_on_token:
            raise RuntimeError("API down")
        return pages[tok]

    svc.changes.return_value.list.return_value.execute.side_effect = _execute
    return svc


def zip_change(file_id, parents=None):
    return {
        "type": "file",
        "fileId": file_id,
        "file": {
            "name": "chat.zip",
            "mimeType": "application/zip",
            "parents": parents or [FOLDER],
            "modifiedTime": "2026-09-01T08:00:00Z",
            "version": 3,
        },
    }


def pdf_change(file_id="pdf-1", parents=("other-folder",)):
    return {
        "type": "file",
        "fileId": file_id,
        "file": {
            "name": "doc.pdf",
            "mimeType": "application/pdf",
            "parents": list(parents),
        },
    }


def stored_token(db_path, account=ACCOUNT, folder=FOLDER):
    store = ImportStateStore(db_path=str(db_path))
    row = store.get_drive_sync_state(account, folder)
    store.close()
    return row


def test_initialization_saves_start_token(tmp_path):
    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {})

    assert initialize_change_tracking(svc, FOLDER, db_path=str(db)) == "tok-A"

    row = stored_token(db)
    assert row is not None
    assert row["page_token"] == "tok-A"
    assert row["last_sync_at"] is None

    # No pipeline execution: no changes().list calls happened at init time.
    svc.changes().list.assert_not_called()


def test_restart_reuses_existing_token(tmp_path):
    db = tmp_path / "state.sqlite"
    svc1 = make_service("tok-A", {})
    assert initialize_change_tracking(svc1, FOLDER, db_path=str(db)) == "tok-A"

    # Restart with a different baseline on offer; existing token must win.
    svc2 = make_service("tok-NEW-BASELINE", {})
    assert initialize_change_tracking(svc2, FOLDER, db_path=str(db)) == "tok-A"

    row = stored_token(db)
    if row is None:
        row={}
    assert row["page_token"] == "tok-A"


def test_relevant_file_update_returned_and_token_advanced(tmp_path):
    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {"tok-A": {"changes": [zip_change("file-1")]}})
    initialize_change_tracking(svc, FOLDER, db_path=str(db))

    relevant, new_tok = fetch_changes_since_saved_token(svc, FOLDER, db_path=str(db))

    assert [c["fileId"] for c in relevant] == ["file-1"]
    assert new_tok == "tok-A"

    row = stored_token(db)
    if row is None:
            row={}
    assert row["page_token"] == "tok-A"
    assert row["last_error"] is None


def test_unrelated_changes_ignored_but_token_advances(tmp_path):
    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {
        "tok-A": {
            "changes": [
                zip_change("file-1"),
                pdf_change("pdf-1"),
                zip_change("file-2"),
            ],
            "nextPageToken": "tok-B",
        },
        "tok-B": {"changes": []},
    })
    initialize_change_tracking(svc, FOLDER, db_path=str(db))

    relevant, new_tok = fetch_changes_since_saved_token(svc, FOLDER, db_path=str(db))

    assert {c["fileId"] for c in relevant} == {"file-1", "file-2"}
    assert not is_relevant_drive_change(pdf_change("pdf-1"), FOLDER)
    assert new_tok == "tok-B"

    row = stored_token(db)
    if row is None:
            row={}
    assert row["page_token"] == "tok-B"

    # Next poll continues from tok-B: the ignored PDF is not replayed.
    svc3 = make_service("tok-A", {"tok-B": {"changes": [pdf_change("pdf-1")]}})
    relevant2, _ = fetch_changes_since_saved_token(svc3, FOLDER, db_path=str(db))
    assert relevant2 == []


def test_api_failure_retains_token_and_records_error(tmp_path):
    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {"tok-A": {"changes": [zip_change("file-1")]}})
    initialize_change_tracking(svc, FOLDER, db_path=str(db))
    fetch_changes_since_saved_token(svc, FOLDER, db_path=str(db))

    broken = make_service("tok-A", {"tok-A": {"changes": []}}, fail_on_token="tok-A")
    try:
        fetch_changes_since_saved_token(broken, FOLDER, db_path=str(db))
        assert False, "expected the API failure to propagate"
    except RuntimeError:
        pass

    row = stored_token(db)
    if row is None:
            row={}
    assert row["page_token"] == "tok-A"
    assert row["last_error"] == "API down"

    # Healed service: retry pulls the change again from the retained token.
    healed = make_service("tok-A", {"tok-A": {"changes": [zip_change("file-1")]}})
    relevant, _ = fetch_changes_since_saved_token(healed, FOLDER, db_path=str(db))
    assert [c["fileId"] for c in relevant] == ["file-1"]


def test_multiple_pages_all_collected_final_token_saved(tmp_path):
    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {
        "tok-A": {"changes": [zip_change("file-a")], "nextPageToken": "tok-B"},
        "tok-B": {"changes": [zip_change("file-b")]},
    })
    initialize_change_tracking(svc, FOLDER, db_path=str(db))

    relevant, new_tok = fetch_changes_since_saved_token(svc, FOLDER, db_path=str(db))

    assert {c["fileId"] for c in relevant} == {"file-a", "file-b"}
    assert new_tok == "tok-B"

    row = stored_token(db)
    if row is None:
            row={}
    assert row["page_token"] == "tok-B"


def test_final_page_prefers_new_start_page_token(tmp_path):
    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {
        "tok-A": {
            "changes": [zip_change("file-1")],
            "newStartPageToken": "tok-Z",
        },
    })
    initialize_change_tracking(svc, FOLDER, db_path=str(db))

    relevant, new_tok = fetch_changes_since_saved_token(svc, FOLDER, db_path=str(db))

    assert [c["fileId"] for c in relevant] == ["file-1"]
    assert new_tok == "tok-Z"

    row = stored_token(db)
    if row is None:
        row = {}
    assert row["page_token"] == "tok-Z"


def test_offline_recovery_finds_changes_made_during_downtime(tmp_path):
    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {})
    initialize_change_tracking(svc, FOLDER, db_path=str(db))

    # Manthan stopped. The export was updated on Drive. Restart:
    restarted = make_service("tok-NEW", {"tok-A": {"changes": [zip_change("offline-1")]}})
    relevant, _ = fetch_changes_since_saved_token(restarted, FOLDER, db_path=str(db))

    assert [c["fileId"] for c in relevant] == ["offline-1"]

    # The request used the persisted position, not a fresh baseline.
    used = restarted.changes.return_value.list.call_args.kwargs["pageToken"]
    assert used == "tok-A"


def test_concurrent_sync_file_lock_serializes(tmp_path):
    import threading
    import time
    from unittest.mock import patch

    from drive_changes import sync_import_relevant_changes

    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {})
    initialize_change_tracking(svc, FOLDER, db_path=str(db))

    pages = {
        "tok-A": {
            "changes": [zip_change("file-1")],
            "newStartPageToken": "tok-B",
        },
        "tok-B": {"changes": []},
    }

    def _exec():
        tok = svc.changes.return_value.list.call_args.kwargs["pageToken"]
        return pages.get(tok, {"changes": []})

    svc.changes.return_value.list.return_value.execute.side_effect = _exec

    call_count = 0

    def slow_import(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.6)
        return {"ok": call_count}

    with patch("drive_changes.get_file_metadata", return_value={"modifiedTime": "2026-09-01T00:00:00Z"}), \
         patch("drive_changes.download_file", return_value="/tmp/dummy.zip"), \
         patch("drive_changes._extract_chat_txt", return_value=True), \
         patch("drive_changes.run_incremental_import", side_effect=slow_import):

        def run():
            try:
                sync_import_relevant_changes(svc, FOLDER, db_path=str(db))
            except Exception:
                pass

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        time.sleep(0.15)
        t2.start()
        t1.join()
        t2.join()

    # Only one import should have run — second saw new token tok-B and had nothing to do
    assert call_count == 1
    row = stored_token(db)
    assert row is not None
    assert row["page_token"] == "tok-B"


def test_notification_queue_serializes_burst(tmp_path):
    import time
    from unittest.mock import patch

    import drive_changes
    from drive_changes import notify_drive_change

    db = tmp_path / "state.sqlite"
    svc = make_service("tok-A", {})
    initialize_change_tracking(svc, FOLDER, db_path=str(db))

    pages = {
        "tok-A": {
            "changes": [zip_change("file-1")],
            "newStartPageToken": "tok-B",
        },
        "tok-B": {"changes": []},
    }

    def _exec():
        tok = svc.changes.return_value.list.call_args.kwargs["pageToken"]
        return pages.get(tok, {"changes": []})

    svc.changes.return_value.list.return_value.execute.side_effect = _exec

    call_count = 0

    def slow_import(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.4)
        return {"ok": call_count}

    # Drain any leftover queue state from previous tests
    while not drive_changes._sync_queue.empty():
        try:
            drive_changes._sync_queue.get_nowait()
            drive_changes._sync_queue.task_done()
        except Exception:
            break

    with patch("drive_changes.authenticate", return_value=svc), \
         patch("drive_changes.get_file_metadata", return_value={"modifiedTime": "2026-09-01T00:00:00Z"}), \
         patch("drive_changes.download_file", return_value="/tmp/dummy.zip"), \
         patch("drive_changes._extract_chat_txt", return_value=True), \
         patch("drive_changes.run_incremental_import", side_effect=slow_import):

        notify_drive_change(FOLDER, db_path=str(db))
        notify_drive_change(FOLDER, db_path=str(db))
        drive_changes._sync_queue.join()
        time.sleep(0.1)

    assert call_count == 1
    row = stored_token(db)
    assert row is not None
    assert row["page_token"] == "tok-B"

    # Cleanup queue for next tests
    while not drive_changes._sync_queue.empty():
        try:
            drive_changes._sync_queue.get_nowait()
            drive_changes._sync_queue.task_done()
        except Exception:
            break