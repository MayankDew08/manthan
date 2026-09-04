import datetime
import sys
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drive_watch
from drive_watch import (
    WEBHOOK_URL,
    create_watch,
    ensure_watch,
    get_active_watch,
    renew_watch,
    stop_watch,
    watch_needs_renewal,
)
from import_state import ImportStateStore

ACCOUNT = "me@example.com"
FOLDER = "folder-123"
TOKEN = "fake-token"


def _future_ms(hours):
    return str(
        int(
            (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(hours=hours)
            ).timestamp()
            * 1000
        )
    )


def make_service(start_token="tok0", watch_resp=None):
    svc = Mock()
    svc.about.return_value.get.return_value.execute.return_value = {
        "user": {"emailAddress": ACCOUNT},
    }
    svc.changes.return_value.getStartPageToken.return_value.execute.return_value = {
        "startPageToken": start_token,
    }
    if watch_resp is None:
        watch_resp = {
            "resourceId": "res-123",
            "resourceUri": "https://example.com",
            "expiration": _future_ms(24),
        }
    svc.changes.return_value.watch.return_value.execute.return_value = watch_resp
    return svc


def seed_sync_token(db_path, token="tok0"):
    store = ImportStateStore(db_path=str(db_path))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    store.upsert_drive_sync_state(
        account_id=ACCOUNT,
        folder_id=FOLDER,
        page_token=token,
        updated_at=now,
        last_sync_at=now,
        last_error=None,
    )
    store.close()


def seed_active_watch(db_path, channel_id="chan-old", resource_id="res-old", exp_hours=48):
    store = ImportStateStore(db_path=str(db_path))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    store.connection.execute(
        """CREATE TABLE IF NOT EXISTS drive_watch_state (
            channel_id TEXT PRIMARY KEY,
            installation_id TEXT NOT NULL,
            resource_id TEXT,
            webhook_url TEXT,
            expiration TEXT,
            created_at TEXT,
            status TEXT,
            page_token TEXT
        )"""
    )
    store.connection.execute(
        "INSERT INTO drive_watch_state (channel_id, installation_id, resource_id, webhook_url, expiration, created_at, status, page_token)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (channel_id, "manthan-mayank-01", resource_id, WEBHOOK_URL, _future_ms(exp_hours), now, "active", "tok0"),
    )
    store.connection.commit()
    store.close()


def _env(monkeypatch):
    monkeypatch.setenv("DRIVE_CHANNEL_TOKEN", TOKEN)
    monkeypatch.setenv("INSTALLATION_ID", "manthan-mayank-01")


def test_no_page_token_initializes_tracking_first(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    svc = make_service(start_token="tok0")
    # fresh DB: no drive_sync_state row -> initialize_change_tracking must run
    with patch.object(drive_watch, "initialize_change_tracking", wraps=drive_watch.initialize_change_tracking) as spy:
        create_watch(svc, FOLDER, db_path=str(db))
        assert spy.call_count == 1
    store = ImportStateStore(db_path=str(db))
    row = store.get_drive_sync_state(ACCOUNT, FOLDER)
    store.close()
    assert row is not None
    assert row["page_token"] == "tok0"


def test_existing_page_token_passed_to_watch(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_sync_token(db, token="tok-saved")
    svc = make_service()
    create_watch(svc, FOLDER, db_path=str(db))
    kwargs = svc.changes.return_value.watch.call_args.kwargs
    assert kwargs["pageToken"] == "tok-saved"
    # must not re-initialize tracking
    svc.changes.return_value.getStartPageToken.assert_not_called()


def test_create_watch_sends_correct_body(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_sync_token(db)
    svc = make_service()
    create_watch(svc, FOLDER, db_path=str(db))
    kwargs = svc.changes.return_value.watch.call_args.kwargs
    body = kwargs["body"]
    assert body["address"] == WEBHOOK_URL
    assert body["type"] == "web_hook"
    uuid.UUID(body["id"])  # valid UUID
    assert body["token"] == TOKEN
    exp_ms = int(body["expiration"])
    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    assert 0 < exp_ms - now_ms <= 25 * 3600 * 1000


def test_successful_response_stored(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_sync_token(db)
    svc = make_service(
        watch_resp={"resourceId": "res-abc", "resourceUri": "uri-abc", "expiration": _future_ms(20)}
    )
    out = create_watch(svc, FOLDER, db_path=str(db))
    assert out["resource_id"] == "res-abc"
    assert out["status"] == "active"
    active = get_active_watch(db_path=str(db))
    assert active is not None
    assert active["channel_id"] == out["channel_id"]
    assert active["resource_id"] == "res-abc"


def test_registration_failure_saves_no_active_channel(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_sync_token(db)
    svc = make_service()
    svc.changes.return_value.watch.side_effect = RuntimeError("watch failed")
    try:
        create_watch(svc, FOLDER, db_path=str(db))
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert get_active_watch(db_path=str(db)) is None


def test_healthy_channel_ensure_does_nothing(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_active_watch(db, exp_hours=48)
    svc = make_service()
    out = ensure_watch(svc, FOLDER, db_path=str(db), window_hours=24)
    assert out["action"] == "healthy"
    svc.changes.return_value.watch.assert_not_called()
    svc.channels.assert_not_called()


def test_nearly_expired_channel_replaced(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_active_watch(db, channel_id="chan-old", exp_hours=1)
    seed_sync_token(db)
    svc = make_service(
        watch_resp={"resourceId": "res-new", "resourceUri": "uri", "expiration": _future_ms(48)}
    )
    assert watch_needs_renewal(db_path=str(db), window_hours=24) is True
    out = ensure_watch(svc, FOLDER, db_path=str(db), window_hours=24)
    assert out["action"] == "renewed"
    assert out["channel_id"] != "chan-old"


def test_renewal_creates_before_stopping_old(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_active_watch(db, channel_id="chan-old", resource_id="res-old", exp_hours=1)
    seed_sync_token(db)
    svc = make_service(
        watch_resp={"resourceId": "res-new", "resourceUri": "uri", "expiration": _future_ms(48)}
    )
    order = []
    svc.changes.return_value.watch.return_value.execute.side_effect = (
        lambda: (order.append("watch"), {"resourceId": "res-new", "resourceUri": "uri", "expiration": _future_ms(48)})[1]
    )
    svc.channels.return_value.stop.return_value.execute.side_effect = lambda: order.append("stop")

    out = renew_watch(svc, FOLDER, db_path=str(db))
    assert order == ["watch", "stop"]
    assert out["old_channel_id"] == "chan-old"
    assert out["channel_id"] != "chan-old"


def test_replacement_failure_keeps_old_active(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_active_watch(db, channel_id="chan-old", resource_id="res-old", exp_hours=1)
    seed_sync_token(db)
    svc = make_service()
    svc.changes.return_value.watch.side_effect = RuntimeError("watch failed")
    try:
        renew_watch(svc, FOLDER, db_path=str(db))
        assert False, "should have raised"
    except RuntimeError:
        pass
    active = get_active_watch(db_path=str(db))
    assert active is not None
    assert active["channel_id"] == "chan-old"
    svc.channels.assert_not_called()


def test_stop_watch_sends_both_ids(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_active_watch(db, channel_id="chan-old", resource_id="res-old", exp_hours=48)
    svc = make_service()
    out = stop_watch(svc, "chan-old", "res-old", db_path=str(db))
    body = svc.channels.return_value.stop.call_args.kwargs["body"]
    assert body == {"id": "chan-old", "resourceId": "res-old"}
    assert out["status"] == "stopped"
    assert get_active_watch(db_path=str(db)) is None


def test_channel_token_never_stored_or_returned(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = tmp_path / "watch.sqlite"
    seed_sync_token(db)
    svc = make_service(
        watch_resp={"resourceId": "res-x", "resourceUri": "uri-x", "expiration": _future_ms(20)}
    )
    out = create_watch(svc, FOLDER, db_path=str(db))
    assert "token" not in out
    assert TOKEN not in str(out)
    store = ImportStateStore(db_path=str(db))
    rows = store.connection.execute("SELECT * FROM drive_watch_state").fetchall()
    dump = " ".join(str(dict(r)) for r in rows)
    store.close()
    assert TOKEN not in dump
