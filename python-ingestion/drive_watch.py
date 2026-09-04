import datetime
import os
import uuid

from dotenv import load_dotenv

from drive_changes import initialize_change_tracking
from import_state import ImportStateStore

load_dotenv()

WEBHOOK_URL = "https://manthan-drive-relay-md-20260903.azurewebsites.net/api/drive-webhook"


def _ensure_table(db_path="import_state.sqlite", store=None):
    if store is None:
        store = ImportStateStore(db_path)
        store.connection.executescript("""
            CREATE TABLE IF NOT EXISTS drive_watch_state (
                channel_id TEXT PRIMARY KEY,
                installation_id TEXT NOT NULL,
                resource_id TEXT,
                webhook_url TEXT,
                expiration TEXT,
                created_at TEXT,
                status TEXT,
                page_token TEXT
            );
        """)
        store.close()
    else:
        store.connection.executescript("""
            CREATE TABLE IF NOT EXISTS drive_watch_state (
                channel_id TEXT PRIMARY KEY,
                installation_id TEXT NOT NULL,
                resource_id TEXT,
                webhook_url TEXT,
                expiration TEXT,
                created_at TEXT,
                status TEXT,
                page_token TEXT
            );
        """)


def create_watch(service, folder_id: str, db_path="import_state.sqlite") -> dict:
    token = os.getenv("DRIVE_CHANNEL_TOKEN", "")
    if not token:
        raise RuntimeError("DRIVE_CHANNEL_TOKEN not set")
    installation_id = os.getenv("INSTALLATION_ID")
    if not installation_id:
        raise RuntimeError("INSTALLATION_ID not set")

    store = ImportStateStore(db_path)
    try:
        _ensure_table(db_path, store=store)
        # ensure page token exists
        about = service.about().get(fields="user").execute()
        account_id = about["user"]["emailAddress"]
        state = store.get_drive_sync_state(account_id, folder_id)
        if not state or not state.get("page_token"):
            page_token = initialize_change_tracking(service, folder_id, db_path)
        else:
            page_token = state["page_token"]

        channel_id = str(uuid.uuid4())
        expiration = str(int((datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)).timestamp() * 1000))
        body = {
            "id": channel_id,
            "type": "web_hook",
            "address": WEBHOOK_URL,
            "token": token,
            "expiration": expiration,
        }
        resp = service.changes().watch(pageToken=page_token, body=body).execute()

        now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        store.connection.execute(
            "UPDATE drive_watch_state SET status='replaced' WHERE installation_id=? AND status='active'",
            (installation_id,),
        )
        store.connection.execute(
            "INSERT INTO drive_watch_state (channel_id, installation_id, resource_id, webhook_url, expiration, created_at, status, page_token) VALUES (?,?,?,?,?,?,?,?)",
            (channel_id, installation_id, resp.get("resourceId"), WEBHOOK_URL, resp.get("expiration", expiration), now, "active", page_token),
        )
        store.connection.commit()
    finally:
        store.close()

    return {
        "channel_id": channel_id,
        "resource_id": resp.get("resourceId"),
        "resourceUri": resp.get("resourceUri"),
        "expiration": resp.get("expiration", expiration),
        "webhook_url": WEBHOOK_URL,
        "created_at": now,
        "status": "active",
        "installation_id": installation_id,
    }


def get_active_watch(db_path="import_state.sqlite") -> dict | None:
    _ensure_table(db_path)
    store = ImportStateStore(db_path)
    row = store.connection.execute(
        "SELECT * FROM drive_watch_state WHERE status='active' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    store.close()
    return dict(row) if row else None

def watch_needs_renewal(db_path="import_state.sqlite", window_hours=24) -> bool:
    w = get_active_watch(db_path=db_path)
    if w is None:
        return True
    exp = w.get("expiration")
    if exp is None or not str(exp).strip():
        return True
    try:
        exp_ms = int(str(exp).strip())
    except (ValueError, TypeError, AttributeError):
        return True
    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    if exp_ms <= now_ms:
        return True
    if exp_ms - now_ms <= window_hours * 3600 * 1000:
        return True
    return False


def renew_watch(service, folder_id: str, db_path: str = "import_state.sqlite") -> dict:
    old = get_active_watch(db_path=db_path)
    if old is None:
        return create_watch(service=service, folder_id=folder_id, db_path=db_path)
    old_channel_id = old.get("channel_id")
    old_resource_id = old.get("resource_id")
    # Safe order: create replacement first (already saves new and marks old replaced)
    new = create_watch(service=service, folder_id=folder_id, db_path=db_path)
    # Then stop previous channel — best-effort, overlap is safe (duplicates handled by page-token)
    try:
        if old_channel_id and old_resource_id:
            service.channels().stop(body={"id": old_channel_id, "resourceId": old_resource_id}).execute()
        elif old_channel_id:
            service.channels().stop(body={"id": old_channel_id}).execute()
    except Exception as e:
        print(f"Warning: failed to stop old channel {old_channel_id}: {e}")
    new["old_channel_id"] = old_channel_id
    return new


def stop_watch(service, channel_id: str, resource_id: str | None = None, db_path="import_state.sqlite") -> dict:
    store = ImportStateStore(db_path)
    try:
        _ensure_table(db_path, store=store)
        row = store.connection.execute(
            "SELECT * FROM drive_watch_state WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        installation_id = row["installation_id"] if row else None
        if resource_id is None and row:
            resource_id = row["resource_id"]
        try:
            body = {"id": channel_id}
            if resource_id:
                body["resourceId"] = resource_id
            service.channels().stop(body=body).execute()
        except Exception as e:
            print(f"Warning: failed to stop channel {channel_id}: {e}")
        store.connection.execute(
            "UPDATE drive_watch_state SET status='stopped' WHERE channel_id = ?",
            (channel_id,),
        )
        store.connection.commit()
    finally:
        store.close()
    return {
        "channel_id": channel_id,
        "resource_id": resource_id,
        "status": "stopped",
        "installation_id": installation_id,
    }


def ensure_watch(service, folder_id: str, db_path: str = "import_state.sqlite", window_hours: int = 24) -> dict:
    w = get_active_watch(db_path=db_path)
    if w is None:
        new = create_watch(service=service, folder_id=folder_id, db_path=db_path)
        new["action"] = "created"
        return new
    if watch_needs_renewal(db_path=db_path, window_hours=window_hours):
        new = renew_watch(service=service, folder_id=folder_id, db_path=db_path)
        new["action"] = "renewed"
        return new
    result = dict(w)
    result["action"] = "healthy"
    return result


def _format_expiration(exp_ms: str | None) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(int(str(exp_ms).strip()) / 1000, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError, AttributeError):
        return str(exp_ms)


def main() -> None:
    import sys

    from drive_client import authenticate

    cmd = sys.argv[1] if len(sys.argv) > 1 else "create"
    folder_id = os.getenv("DRIVE_FOLDER_ID", "")
    if not folder_id:
        raise SystemExit("No folder configured: set DRIVE_FOLDER_ID in .env")
    service = authenticate()

    if cmd == "create":
        out = create_watch(service, folder_id)
        print("Drive watch created")
        print(f"Channel ID: {out['channel_id']}")
        print(f"Resource ID: {out['resource_id']}")
        print(f"Expires: {_format_expiration(out['expiration'])}")
    elif cmd == "status":
        w = get_active_watch()
        if w is None:
            print("No active watch")
            return
        print(f"Channel ID: {w['channel_id']}")
        print(f"Resource ID: {w['resource_id']}")
        print(f"Expires: {_format_expiration(w.get('expiration'))}")
        print(f"Status: {w['status']}")
    else:
        raise SystemExit(f"Unknown command {cmd!r}: use 'create' or 'status'")


if __name__ == "__main__":
    main()