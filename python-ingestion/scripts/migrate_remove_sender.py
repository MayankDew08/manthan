"""One-way migration to sender-free identity (remove-sender branch).

What it does (only with --apply):
  SQLite : backs up to <db>.bak-<timestamp>, then DELETES processed_messages
           and resets ingestion_sources revisions (message IDs change without
           sender, so old rows are unusable). drive_sync_state and
           drive_watch_state are left untouched.
  Neo4j  : DETACH DELETEs all :Person nodes (+ :SENT edges) and REMOVEs the
           sender property from all :Message nodes. No backup is taken here —
           rollback is: restore the SQLite .bak file and re-run a full import
           (deterministic: same export reproduces the same graph).
  Qdrant : DELETES the message/link collection (point IDs were sender-based).
           It is recreated lazily with content-based IDs on the next import.

Default is --dry-run: reads everything, changes nothing, prints the audit.
Run with writers stopped (make stop): API, workers, listener.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AUDIT_KEYS = ("persons", "sent_edges", "messages", "messages_with_sender",
              "links", "processed_messages", "sources", "qdrant_points")


def audit_sqlite(db_path):
    con = sqlite3.connect(db_path)
    try:
        out = {}
        out["processed_messages"] = con.execute(
            "SELECT COUNT(*) FROM processed_messages").fetchone()[0]
        out["sources"] = con.execute(
            "SELECT COUNT(*) FROM ingestion_sources").fetchone()[0]
        out["sync_rows"] = con.execute(
            "SELECT COUNT(*) FROM drive_sync_state").fetchone()[0]
        try:
            out["watch_rows"] = con.execute(
                "SELECT COUNT(*) FROM drive_watch_state").fetchone()[0]
        except sqlite3.OperationalError:
            out["watch_rows"] = 0
        return out
    finally:
        con.close()


def audit_neo4j(driver):
    out = {}
    with driver.session() as s:
        out["persons"] = s.run("MATCH (p:Person) RETURN count(*) AS c").single()["c"]
        out["sent_edges"] = s.run("MATCH ()-[r:SENT]->() RETURN count(*) AS c").single()["c"]
        out["messages"] = s.run("MATCH (m:Message) RETURN count(*) AS c").single()["c"]
        out["messages_with_sender"] = s.run(
            "MATCH (m:Message) WHERE m.sender IS NOT NULL RETURN count(*) AS c").single()["c"]
        out["links"] = s.run("MATCH (l:Link) RETURN count(*) AS c").single()["c"]
    return out


def audit_qdrant(collection):
    from qdrant_client import QdrantClient
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port)
    try:
        out = {}
        # Both the legacy and "-fresh" collections may hold sender-keyed points.
        for name in dict.fromkeys([collection, "manthan", "manthan-fresh"]):
            try:
                info = client.get_collection(name)
                out[name] = info.points_count or 0
            except Exception:
                out[name] = 0
        out["qdrant_points"] = out.get(collection, 0)
        return out
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser(description="Migrate stores to sender-free identity.")
    ap.add_argument("--db", default="import_state.sqlite")
    ap.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "manthan"))
    ap.add_argument("--apply", action="store_true",
                    help="Actually mutate. Without it, dry-run only.")
    args = ap.parse_args()

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j driver not installed; cannot audit Neo4j.")
        return 2

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        print("=== DRY-RUN AUDIT (no changes) ===" if not args.apply else "=== APPLY ===")
        sq = audit_sqlite(args.db)
        print("sqlite:", json.dumps(sq, indent=2))
        nq = audit_neo4j(driver)
        print("neo4j:", json.dumps(nq, indent=2))
        qq = audit_qdrant(args.collection)
        print("qdrant:", json.dumps(qq, indent=2))

        if not args.apply:
            print("\nWould: backup sqlite; wipe processed_messages; reset revisions; "
                  "delete Person/SENT/sender props; delete qdrant collection.")
            print("drive_sync_state + drive_watch_state rows preserved:",
                  sq["sync_rows"], "+", sq["watch_rows"])
            return 0

        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = f"{args.db}.bak-{stamp}"
        shutil.copy2(args.db, backup)
        print(f"sqlite backup -> {backup}")

        con = sqlite3.connect(args.db)
        try:
            with con:
                con.execute("DELETE FROM processed_messages")
                con.execute("UPDATE ingestion_sources SET revision = NULL, imported_at = NULL")
        finally:
            con.close()
        print("sqlite: processed_messages wiped, revisions reset")

        with driver.session() as s:
            s.run("MATCH (p:Person) DETACH DELETE p").consume()
            s.run("MATCH (m:Message) WHERE m.sender IS NOT NULL REMOVE m.sender").consume()
        print("neo4j: Person/SENT deleted, sender props removed")

        from qdrant_client import QdrantClient
        qc = QdrantClient(host=os.environ.get("QDRANT_HOST", "localhost"),
                          port=int(os.environ.get("QDRANT_PORT", "6333")))
        try:
            for name in dict.fromkeys([args.collection, "manthan", "manthan-fresh"]):
                try:
                    qc.delete_collection(name)
                    print(f"qdrant: collection {name} deleted (recreated on next import)")
                except Exception as e:
                    print(f"qdrant: delete {name} skipped ({e})")
        finally:
            qc.close()

        print("\nDone. Next: re-run a full import to rebuild IDs/vectors, then verify:")
        print("  128 Messages, 0 Person nodes, 0 sender props, search clean.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
