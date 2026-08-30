from dataclasses import dataclass
import datetime
import hashlib
import json

from parser import Data


@dataclass
class IdentifiedMessage:
    message_id: str
    occurrence: int
    data: Data


def normalize_text(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def identify_messages(
    source_id: str,
    messages: list[Data],
) -> list[IdentifiedMessage]:

    occurrences: dict[tuple[str, str, str], int] = {}
    results: list[IdentifiedMessage] = []

    for message in messages:
        sender = message.sender.strip()
        text = normalize_text(message.text)

        message_signature = (
            message.datetime_iso,
            sender,
            text,
        )

        occurrence = occurrences.get(message_signature, 0) + 1
        occurrences[message_signature] = occurrence

        identity_data = [
            source_id,
            message.datetime_iso,
            sender,
            text,
            occurrence,
        ]

        identity_string = json.dumps(
            identity_data,
            ensure_ascii=False,
        )

        message_id = hashlib.sha256(
            identity_string.encode("utf-8")
        ).hexdigest()

        results.append(
            IdentifiedMessage(
                message_id=message_id,
                occurrence=occurrence,
                data=message,
            )
        )

    return results


# Storing the ids in db and checking to avoid re-process everything
import sqlite3


class ImportStateStore:
    def __init__(self, db_path: str = "import_state.sqlite"):
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row

        self.create_tables()

    def create_tables(self) -> None:
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS ingestion_sources (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    revision TEXT,
                    imported_at TEXT
                );

                CREATE TABLE IF NOT EXISTS processed_messages (
                    source_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    outcome TEXT NOT NULL,

                    PRIMARY KEY (source_id, message_id)
                );
            """)
            # Migrate DBs created before imported_at existed.
            try:
                self.connection.execute(
                    "ALTER TABLE ingestion_sources ADD COLUMN imported_at TEXT"
                )
            except sqlite3.OperationalError:
                pass

    def upsert_source(
        self,
        source_id: str,
        source_type: str,
        file_name: str,
    ) -> None:
        # Metadata only. The `revision` column is owned solely by
        # update_source_revision and must advance only after a successful
        # persistence run, never here at registration time.
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO ingestion_sources
                    (source_id, source_type, file_name)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    file_name   = excluded.file_name
                """,
                (source_id, source_type, file_name),
            )

    def find_unseen_messages(
        self,
        source_id: str,
        message_ids: list[str],
    ) -> list[str]:
        if not message_ids:
            return []

        # Chunk to stay under SQLite's host-parameter limit (default 999) for chats
        # with more than ~1000 messages.
        seen: set[str] = set()
        batch_size = 500
        for start in range(0, len(message_ids), batch_size):
            batch = message_ids[start:start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"""
                SELECT message_id FROM processed_messages
                WHERE source_id = ? AND message_id IN ({placeholders})
                """,
                [source_id, *batch],
            ).fetchall()
            seen.update(row["message_id"] for row in rows)

        return [mid for mid in message_ids if mid not in seen]

    def mark_processed(
        self,
        source_id: str,
        results: list[tuple[str, str]],
    ) -> None:
        if not results:
            return

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rows = [(source_id, mid, now, outcome) for mid, outcome in results]

        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO processed_messages
                    (source_id, message_id, processed_at, outcome)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, message_id) DO UPDATE SET
                    processed_at = excluded.processed_at,
                    outcome = excluded.outcome
                """,
                rows,
            )

    def get_source(self, source_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM ingestion_sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()

    def update_source_revision(self, source_id: str, revision: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE ingestion_sources
                SET revision = ?, imported_at = ?
                WHERE source_id = ?
                """,
                (revision,
                 datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 source_id),
            )

    def close(self) -> None:
        self.connection.close()