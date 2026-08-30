import os
import tempfile
import zipfile

from dotenv import load_dotenv

from drive_client import authenticate, list_files, get_file_metadata, download_file
from import_state import ImportStateStore
from incremental_import import run_incremental_import

load_dotenv()


def _extract_chat_txt(zip_path: str, out_path: str) -> bool:
    """Extract the single .txt member of a WhatsApp export zip to out_path."""
    with zipfile.ZipFile(zip_path) as archive:
        txt_member = next(
            (name for name in archive.namelist() if name.endswith(".txt")),
            None,
        )
        if txt_member is None:
            return False
        data = archive.read(txt_member)
    with open(out_path, "wb") as f:
        f.write(data)
    return True


def main():
    service = authenticate()
    store = ImportStateStore()

    files = list_files(service, os.getenv("DRIVE_FOLDER_ID"))
    zip_mimes = {"application/zip", "application/x-zip-compressed"}
    zips = [
        item for item in files
        if item.get("mimeType", "").lower() in zip_mimes
    ]

    summaries = []
    for item in zips:
        file_id = item["id"]
        source_id = f"gdrive:{file_id}"
        marker = get_file_metadata(service, file_id).get("modifiedTime")

        if marker is None:
            print(f"Skipping {item['name']}: could not fetch metadata")
            continue

        row = store.get_source(source_id)
        stored = row["revision"] if row else None

        if stored == marker:
            print(f"Unchanged, skipping: {item['name']}")
            continue

        print(f"Changed, ingesting: {item['name']}")
        zip_path = os.path.join(tempfile.gettempdir(), f"manthan-drive-{file_id}.zip")
        txt_path = zip_path + "-chat.txt"
        try:
            download_file(service, file_id, zip_path)
            if not _extract_chat_txt(zip_path, txt_path):
                print(f"Skipping {item['name']}: no .txt inside archive")
                continue
            summary = run_incremental_import(source_id, txt_path, revision=marker)
        finally:
            for path in (txt_path, zip_path):
                if os.path.exists(path):
                    os.remove(path)
        summaries.append((item["name"], summary))

    store.close()

    for name, summary in summaries:
        print(f"\n{name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()