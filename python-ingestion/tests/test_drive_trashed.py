"""Trashed Drive files must never be imported (trashing is itself a change event)."""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drive_changes import fetch_changes_since_saved_token, initialize_change_tracking, \
    is_relevant_drive_change
from drive_client import list_files

FOLDER = "folder-123"


def zip_change(file_id, trashed=None):
    file_meta = {"name": "chat.zip", "mimeType": "application/zip", "parents": [FOLDER],
                 "modifiedTime": "2026-09-01T08:00:00Z"}
    if trashed is not None:
        file_meta["trashed"] = trashed
    return {"type": "file", "fileId": file_id, "file": file_meta}


def test_trashed_zip_in_folder_is_not_relevant():
    assert is_relevant_drive_change(zip_change("f1", trashed=True), FOLDER) is False


def test_untrashed_zip_in_folder_is_still_relevant():
    assert is_relevant_drive_change(zip_change("f1", trashed=False), FOLDER) is True
    assert is_relevant_drive_change(zip_change("f1"), FOLDER) is True


def test_change_feed_requests_the_trashed_field(tmp_path):
    svc = Mock()
    svc.about.return_value.get.return_value.execute.return_value = {
        "user": {"emailAddress": "me@example.com"}}
    svc.changes.return_value.getStartPageToken.return_value.execute.return_value = {
        "startPageToken": "tok-A"}
    svc.changes.return_value.list.return_value.execute.return_value = {
        "changes": [zip_change("live"), zip_change("binned", trashed=True)],
        "newStartPageToken": "tok-B"}
    db = str(tmp_path / "state.sqlite")
    initialize_change_tracking(svc, FOLDER, db_path=db)

    relevant, _ = fetch_changes_since_saved_token(svc, FOLDER, db_path=db)

    fields = svc.changes.return_value.list.call_args.kwargs["fields"]
    assert "trashed" in fields
    assert [c["fileId"] for c in relevant] == ["live"]


def test_folder_listing_excludes_trashed_files():
    svc = Mock()
    svc.files.return_value.list.return_value.execute.return_value = {"files": []}

    list_files(svc, FOLDER)

    q = svc.files.return_value.list.call_args.kwargs["q"]
    assert f"'{FOLDER}' in parents" in q
    assert "trashed = false" in q
