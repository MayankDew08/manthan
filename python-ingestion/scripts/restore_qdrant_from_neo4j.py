"""Restore Qdrant entries that Neo4j still holds but the live Drive exports cannot supply.

Why: after the sender-removal rebuild, Qdrant was refilled only from the exports
that are live in Drive, while Neo4j kept graded messages and links from older
(now trashed) exports. The stores then hold different data.

Rule (the live export's own grading wins):
  messages  Neo4j quality >= 4, not already in Qdrant, and NOT present in any
            live export (a message that is live and absent from Qdrant was
            graded below 4 by the current run, so Neo4j's old grade is stale).
  links     status 'scraped' (or unset) with summary/what_it_is/problem_solved,
            not already in Qdrant.

Point IDs are content-based, so re-running is idempotent. Nothing is deleted.
No LLM and no re-grading: it re-embeds stored text only.

Default is --dry-run. Run with the Drive listener stopped.
  uv run python scripts/restore_qdrant_from_neo4j.py [--apply]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_QUALITY = 4


def _key(sent_at, text) -> tuple:
    return (sent_at or "", (text or "").strip())


def _has_content(link: dict) -> bool:
    return any((link.get(k) or "").strip()
               for k in ("summary", "what_it_is", "problem_solved"))


def select_restorable(neo_messages: list[dict], neo_links: list[dict],
                      qdrant_msg_keys: set, qdrant_link_urls: set,
                      live_keys: set) -> tuple[list[dict], list[dict]]:
    """Pure selection of what to re-embed. Keys are (sent_at, stripped text)."""
    messages = [m for m in neo_messages
                if (m.get("quality") or 0) >= MIN_QUALITY
                and _key(m["sent_at"], m["text"]) not in qdrant_msg_keys
                and _key(m["sent_at"], m["text"]) not in live_keys]
    links = [l for l in neo_links
             if (l.get("status") or "scraped") == "scraped"
             and _has_content(l) and l["url"] not in qdrant_link_urls]
    return messages, links


def _load_neo4j():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "")))
    try:
        with driver.session() as s:
            msgs = s.run("MATCH (m:Message) RETURN m.sent_at AS sent_at, m.text AS text, "
                         "m.quality AS quality, m.topics AS topics, m.entities AS entities, "
                         "m.trusted AS trusted").data()
            links = s.run("MATCH (l:Link) RETURN l.url AS url, l.title AS title, "
                          "l.summary AS summary, l.what_it_is AS what_it_is, "
                          "l.problem_solved AS problem_solved, l.topics AS topics, "
                          "l.entities AS entities, l.status AS status").data()
    finally:
        driver.close()
    return msgs, links


def _qdrant_state(client, collection: str):
    msg_keys, urls, offset = set(), set(), None
    while True:
        points, offset = client.scroll(collection_name=collection, limit=256,
                                       with_payload=True, with_vectors=False, offset=offset)
        for p in points:
            pl = p.payload or {}
            if pl.get("type") == "link":
                urls.add(pl.get("url"))
            else:
                msg_keys.add(_key(pl.get("sent_at"), pl.get("text")))
        if offset is None:
            return msg_keys, urls


def _live_keys() -> set:
    """(sent_at, text) of every message in the exports currently live in Drive."""
    from drive_client import authenticate, download_file, list_files
    from drive_sync import _extract_chat_txt
    from parser import parse_chat
    service = authenticate()
    keys = set()
    zips = [f for f in list_files(service, os.environ["DRIVE_FOLDER_ID"])
            if f.get("mimeType", "").lower() in ("application/zip", "application/x-zip-compressed")]
    for f in zips:
        with tempfile.TemporaryDirectory() as tmp:
            zpath, tpath = os.path.join(tmp, "x.zip"), os.path.join(tmp, "x.txt")
            download_file(service, f["id"], zpath)
            if _extract_chat_txt(zpath, tpath):
                keys |= {_key(m.datetime_iso, m.text) for m in parse_chat(tpath)}
    print(f"live exports read: {len(zips)} file(s), {len(keys)} distinct messages")
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write to Qdrant (default: dry run)")
    ap.add_argument("--collection", default=None)
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from qdrant_client import QdrantClient
    from vector_store import VectorStore

    collection = args.collection or os.environ.get("QDRANT_COLLECTION", "manthan")
    host, port = os.environ.get("QDRANT_HOST", "localhost"), int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port,
                          api_key=os.environ.get("QDRANT_API_KEY") or None)
    try:
        before = client.get_collection(collection).points_count or 0
        q_msgs, q_urls = _qdrant_state(client, collection)
    finally:
        client.close()

    neo_msgs, neo_links = _load_neo4j()
    live = _live_keys()
    msgs, links = select_restorable(neo_msgs, neo_links, q_msgs, q_urls, live)

    print(f"qdrant {collection}: {before} points ({len(q_msgs)} messages, {len(q_urls)} links)")
    print(f"neo4j: {len(neo_msgs)} messages, {len(neo_links)} links")
    print(f"to restore: {len(msgs)} messages, {len(links)} links "
          f"(expected qdrant after: {before + len(msgs) + len(links)})")
    if not args.apply:
        print("dry run: nothing written")
        return 0

    client = QdrantClient(host=host, port=port,
                          api_key=os.environ.get("QDRANT_API_KEY") or None)
    try:
        snap = client.create_snapshot(collection_name=collection)
        print(f"snapshot taken: {snap.name}")
    finally:
        client.close()

    vs = VectorStore(host=host, port=port, collection=collection)
    try:
        vs.ensure_collection()
        for m in msgs:
            vs.upsert_message({"original_text": m["text"], "sent_at": m["sent_at"],
                               "quality": m["quality"], "trusted": m.get("trusted"),
                               "topics": m.get("topics") or [], "entities": m.get("entities") or []})
        for l in links:
            vs.upsert_link({k: l.get(k) for k in ("url", "title", "summary", "what_it_is",
                                                  "problem_solved", "topics", "entities")})
        after = vs.client.get_collection(collection).points_count or 0
    finally:
        vs.close()
    print(f"restored: qdrant {before} -> {after} points")
    return 0 if after == before + len(msgs) + len(links) else 1


if __name__ == "__main__":
    raise SystemExit(main())
