import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file secrets/token.json.
# drive.readonly (vs the old drive.metadata.readonly) also allows downloading
# file content, not just metadata.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_BASE_DIR = Path(__file__).resolve().parent


def authenticate(
    credentials_file: str = str(_BASE_DIR / "secrets" / "credentials.json"),
    token_file: str = str(_BASE_DIR / "secrets" / "token.json"),
):
    """Build and authorize a Drive v3 service, (re)authorizing when needed.

    Loads the saved OAuth token, refreshes it if expired, or runs the
    local-server consent flow on first use, then persists the token.
    """
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_file, "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def list_files(
    service,
    folder_id: str | None = None,
    page_size: int = 100,
) -> list[dict]:
    """List files inside a configured folder (defaults to DRIVE_FOLDER_ID)."""
    folder_id = folder_id or os.environ.get("DRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError(
            "No folder configured: pass folder_id or set DRIVE_FOLDER_ID"
        )

    files: list[dict] = []
    page_token = None
    while True:
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents",
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    return files


def get_file_metadata(service, file_id: str) -> dict:
    try:
        return (
            service.files()
            .get(fileId=file_id,
                 fields="name,id,mimeType,size,createdTime,modifiedTime")
            .execute()
        )
    except HttpError:
        return {}


def download_file(service, file_id: str, destination_path: str) -> str | None:
    """Stream a file's content to destination_path and return the path.

    Uses get_media (plain files); export_media is only for Google Workspace
    docs and needs a valid export mimeType.
    """
    request = service.files().get_media(fileId=file_id)
    try:
        with open(destination_path, "wb") as out:
            downloader = MediaIoBaseDownload(out, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download {int(status.progress() * 100)}%.")
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None
    return destination_path


def main():
    service = authenticate()

    try:
        items = list_files(service)
    except HttpError as error:
        print(f"An error occurred: {error}")
        return

    if not items:
        print("No files found.")
        return

    os.makedirs("downloads", exist_ok=True)
    for item in items:
        print(f"{item['name']} ({item['id']})")

        meta = get_file_metadata(service, item["id"])
        print(f"  metadata: {meta or 'unavailable'}")

        dest = os.path.join("downloads", item["name"])
        path = download_file(service, item["id"], dest)
        if path:
            print(f"  downloaded to {path}")


if __name__ == "__main__":
    main()