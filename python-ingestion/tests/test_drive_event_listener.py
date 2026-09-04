import copy
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drive_event_listener import handle_drive_event


def _valid_event():
    return {
        "installation_id": "manthan-mayank-01",
        "channel_id": "channel-1",
        "resource_id": "resource-1",
        "resource_state": "change",
        "message_number": "1",
        "received_at": "2026-09-03T15:00:00Z",
    }


def test_valid_event_calls_notify(monkeypatch):
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-123")
    monkeypatch.setenv("MANTHAN_IMPORT_DB", "import_state.sqlite")
    notify = Mock()
    event = _valid_event()
    handle_drive_event(event, "manthan-mayank-01", notify)
    notify.assert_called_once_with("folder-123", "import_state.sqlite")


def test_wrong_installation_not_called(monkeypatch):
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-123")
    monkeypatch.setenv("MANTHAN_IMPORT_DB", "import_state.sqlite")
    notify = Mock()
    event = _valid_event()
    event["installation_id"] = "another-laptop"
    try:
        handle_drive_event(event, "manthan-mayank-01", notify)
        assert False, "should have raised"
    except ValueError as e:
        assert "Wrong installation_id" in str(e) or "installation_id" in str(e).lower()
    notify.assert_not_called()


def test_missing_field_not_called(monkeypatch):
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-123")
    notify = Mock()
    event = _valid_event()
    del event["resource_id"]
    try:
        handle_drive_event(event, "manthan-mayank-01", notify)
        assert False, "should have raised"
    except ValueError as e:
        assert "resource_id" in str(e)
    notify.assert_not_called()


def test_invalid_resource_state(monkeypatch):
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-123")
    notify = Mock()
    event = _valid_event()
    event["resource_state"] = "deleted"
    try:
        handle_drive_event(event, "manthan-mayank-01", notify)
        assert False, "should have raised"
    except ValueError as e:
        assert "resource_state" in str(e).lower() or "Invalid" in str(e)
    notify.assert_not_called()


def test_callback_failure_propagates(monkeypatch):
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-123")
    monkeypatch.setenv("MANTHAN_IMPORT_DB", "import_state.sqlite")
    event = _valid_event()

    def failing_callback(folder_id, db_path):
        raise RuntimeError("boom")

    try:
        handle_drive_event(event, "manthan-mayank-01", failing_callback)
        assert False, "should have propagated"
    except RuntimeError as e:
        assert "boom" in str(e)


def test_event_not_mutated(monkeypatch):
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-123")
    monkeypatch.setenv("MANTHAN_IMPORT_DB", "import_state.sqlite")
    notify = Mock()
    event = _valid_event()
    orig = copy.deepcopy(event)
    handle_drive_event(event, "manthan-mayank-01", notify)
    assert event == orig
