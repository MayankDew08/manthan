
import datetime
import os
import re
import tempfile
import threading

from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from threading import Thread

from dotenv import load_dotenv
from filelock import FileLock

from drive_client import authenticate, get_file_metadata, download_file
from drive_sync import _extract_chat_txt
from import_state import ImportStateStore
from incremental_import import run_incremental_import

load_dotenv()

_ZIP_MIMES = {"application/zip", "application/x-zip-compressed"}

# --- Concurrency: file lock + notification queue (single host, no Redis) ---
_DEFAULT_LOCK_PATH = Path(__file__).resolve().parent / "import_state.sqlite.lock"
_sync_queue: Queue = Queue()
_worker_thread: Thread | None = None
_worker_lock = threading.Lock()


@contextmanager
def drive_sync_lock(db_path: str = "import_state.sqlite", timeout: float = -1):
    """File lock so only one sync runs at a time. timeout=-1 waits forever (A tells B)."""
    if db_path == "import_state.sqlite":
        lock_path = _DEFAULT_LOCK_PATH
    else:
        lock_path = Path(db_path + ".lock")
    lock = FileLock(str(lock_path), timeout=timeout)
    try:
        lock.acquire()
        yield
    finally:
        try:
            lock.release()
        except Exception:
            print("Error releasing the lock")
            pass


def notify_drive_change(folder_id: str, db_path: str = "import_state.sqlite"):
    """Enqueue a folder for sync — worker will run it after current sync finishes (A tells B)."""
    _ensure_worker()
    _sync_queue.put((folder_id, db_path))


def _ensure_worker():
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = Thread(target=_worker_loop, daemon=True)
            _worker_thread.start()


def _worker_loop():
    while True:
        folder_id, db_path = _sync_queue.get()
        try:
            # Worker creates its own service — notifications don't need to pass service
            service = authenticate()
            sync_import_relevant_changes(service, folder_id, db_path=db_path)
        except Exception as e:
            # Error already recorded via record_sync_error inside sync; just log
            print(f"[worker] sync failed for {folder_id}: {e}")
        finally:
            _sync_queue.task_done()






def initialize_change_tracking(
    service,
    folder_id: str,
    db_path: str = "import_state.sqlite",
) -> str:
    """Initialize Drive change tracking for the authenticated account and folder.

    Returns the start page token to use for subsequent calls to
    service.changes().list(pageToken=...).
    """
    store = ImportStateStore(db_path=db_path)

    about = service.about().get(fields="user").execute()
    account_id = about["user"]["emailAddress"]

    start = service.changes().getStartPageToken().execute()
    new_token = start["startPageToken"]

    existing = store.get_drive_sync_state(account_id, folder_id)
    if existing and existing.get("page_token"):
        return existing["page_token"]

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    store.upsert_drive_sync_state(
        account_id=account_id,
        folder_id=folder_id,
        page_token=new_token,
        updated_at=now,
        last_sync_at=None,
        last_error=None,
    )
    return new_token


def fetch_changes_since_saved_token(
    service,
    folder_id: str,
    db_path: str = "import_state.sqlite",
    save:bool = True
) -> tuple[list[dict], str]:
    """Fetch pending Drive changes since the last saved token.

    Returns (changes, new_page_token) where:
      - changes: list of dicts with keys {file_id, file_name, mime_type, modified_time, type, removed}
      - new_page_token: the token to save for the next poll
    """
    store = ImportStateStore(db_path)
    about = service.about().get(fields="user").execute()
    account_id = about["user"]["emailAddress"]

    state = store.get_drive_sync_state(account_id, folder_id)
    if not state or not state.get("page_token"):
        raise RuntimeError("No page token found — call initialize_change_tracking first")

    page_token = state["page_token"]

    changes = []
    try:
        while True:
            resp = (
                service.changes()
                .list(
                    pageToken=page_token,
                    spaces="drive",
                    fields="nextPageToken, changes(fileId, file(name,mimeType,parents,modifiedTime,version), type, removed)",
                    includeRemoved=True,
                    pageSize=1000,
                )
                .execute()
            )
            changes.extend(resp.get("changes", []))
            if resp.get("newStartPageToken"):
                saved_token = resp["newStartPageToken"]
            else:
                saved_token = page_token
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        record_sync_error(service, folder_id, exc, db_path)
        store.close()
        raise

    relevant = [ch for ch in changes if is_relevant_drive_change(ch, folder_id)]

    if save:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        store.upsert_drive_sync_state(
            account_id=account_id,
            folder_id=folder_id,
            page_token=saved_token,
            updated_at=now,
            last_sync_at=now,
            last_error=None,
        )
    store.close()

    return relevant, saved_token

def is_relevant_drive_change(change: dict, folder_id: str) -> bool:
    """Return True if a Drive change is a file we care about."""
    if change.get("removed"):
        return False
    if change.get("type") != "file":
        return False
    file_meta = change.get("file") or {}
    if file_meta.get("mimeType", "").lower() not in _ZIP_MIMES:
        return False
    parents = file_meta.get("parents") or []
    return folder_id in parents

def save_page_token(
    service,
    folder_id: str,
    page_token: str,
    db_path: str = "import_state.sqlite",
) -> None:
    """Persist the latest page token after a successful sync.

    Updates the successful-sync timestamp and clears any previous error.
    """
    store = ImportStateStore(db_path=db_path)
    about = service.about().get(fields="user").execute()
    account_id = about["user"]["emailAddress"]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    store.upsert_drive_sync_state(
        account_id=account_id,
        folder_id=folder_id,
        page_token=page_token,
        updated_at=now,
        last_sync_at=now,
        last_error=None,
    )
    store.close()


def record_sync_error(
    service,
    folder_id: str,
    error: Exception | str,
    db_path: str = "import_state.sqlite",
) -> None:
    """Record a sync failure while retaining the current page token."""
    store = ImportStateStore(db_path=db_path)
    about = service.about().get(fields="user").execute()
    account_id = about["user"]["emailAddress"]
    state = store.get_drive_sync_state(account_id, folder_id) or {}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    store.upsert_drive_sync_state(
        account_id=account_id,
        folder_id=folder_id,
        page_token=state.get("page_token"),
        updated_at=state.get("updated_at") or now,
        last_sync_at=state.get("last_sync_at"),
        last_error=str(error),
    )
    store.close()
    
def sync_import_relevant_changes(service, folder_id: str, db_path: str = "import_state.sqlite") -> None:
    # Only one sync at a time — file lock (A tells B via queue, B blocks on lock)
    with drive_sync_lock(db_path=db_path, timeout=-1):
        # Fetch without saving — bookmark only moves after the whole batch succeeds
        relevant, new_token = fetch_changes_since_saved_token(
            service=service, folder_id=folder_id, db_path=db_path, save=False
        )

        # Nothing relevant → still advance bookmark (vacuously successful)
        if not relevant:
            save_page_token(service=service, folder_id=folder_id, page_token=new_token, db_path=db_path)
            print(f"No relevant changes — advanced to token {new_token}")
            return

        store = ImportStateStore(db_path=db_path)
        try:
            for ch in relevant:
                file_meta = ch.get("file") or {}
                file_id = ch["fileId"]
                name = file_meta.get("name", file_id)
                source_id = f"gdrive:{file_id}"

                meta = get_file_metadata(service=service, file_id=file_id)
                revision = meta.get("modifiedTime")
                if not revision:
                    raise RuntimeError(f"could not fetch metadata for {name}")

                row = store.get_source(source_id)
                if row and row["revision"] == revision:
                    print(f"Already imported, skipping: {name}")
                    continue

                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
                zip_path = os.path.join(tempfile.gettempdir(), f"manthan-drive-{file_id}.zip")
                txt_path = os.path.join(tempfile.gettempdir(), f"manthan-drive-{file_id}-{safe}.txt")

                try:
                    if download_file(service, file_id, zip_path) is None:
                        raise RuntimeError(f"download failed for {name}")
                    if not _extract_chat_txt(zip_path, txt_path):
                        raise RuntimeError(f"no .txt inside archive {name}")
                    summary = run_incremental_import(source_id, txt_path, revision=revision, db_path=db_path)
                    print(f"Imported {name}: {summary}")
                finally:
                    for p in (txt_path, zip_path):
                        if os.path.exists(p):
                            os.remove(p)

            # Every relevant file succeeded → now save new bookmark
            save_page_token(service=service, folder_id=folder_id, page_token=new_token, db_path=db_path)
            print(f"Saved new page token {new_token}")

        except Exception as e:
            # One file failed → retain old bookmark, record error, retry will skip duplicates
            record_sync_error(service=service, folder_id=folder_id, error=e, db_path=db_path)
            print(f"Failed: {e} — retained old token, will retry")
            raise
        finally:
            store.close()



