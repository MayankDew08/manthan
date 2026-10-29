"""Backfill Message.category onto the live Qdrant/Neo4j stores (spec section
4). Owner-run: --apply writes to live stores. Never upserts or re-embeds --
only Qdrant ``set_payload`` and Neo4j ``SET``, so vectors and ids never
change (``evals.parity_check`` still passes after --apply).

What it does:
  Neo4j  : SET m.category on existing :Message nodes (never MERGE/CREATE a
           node). Also SET l.category = "resource" on every :Link missing it.
  Qdrant : set_payload({"category": ...}) on existing points, grouped by
           category value to minimize calls. Creates a keyword payload
           index on "category".

Source: graded_messages.json (quality/category/original_text -- no
sent_at). Each grade is paired with the live :Message that shares its
text, matching the i-th grade of a repeated text to the i-th
(chronologically earliest-first) live message with that text -- the
same kind of positional alignment enrich._load_existing_grades uses when
a grade has no timestamp of its own. If a text's grade count and live
message count don't agree exactly, every grade for that text is reported
unmatched rather than guessed (never fuzzy-matched).

Default is --dry-run: reads everything, writes nothing, prints the counts.

Usage (from python-ingestion/):
  uv run python scripts/backfill_category.py --dry-run
  uv run python scripts/backfill_category.py --apply
"""

import argparse
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from models import load_list  # noqa: E402

GRADED_FILE = "graded_messages.json"
LINK_CATEGORY = "resource"


def message_id(sent_at: str, text: str) -> str:
    """M8: the id vector_store.upsert_message gives this message in Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"msg-{sent_at or ''}-{text or ''}"))


def link_id(url: str) -> str:
    """M8: the id vector_store.upsert_link gives this link in Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"link-{url or ''}"))


def live_messages(driver) -> list:
    """Every Message's (sent_at, text, category), oldest first within a
    repeated text (the order grades for that text are paired against)."""
    with driver.session() as session:
        rows = session.run(
            "MATCH (m:Message) RETURN m.sent_at AS sent_at, m.text AS text, "
            "m.category AS category"
        ).data()
    rows.sort(key=lambda r: r["sent_at"] or "")
    return rows


def live_links(driver) -> list:
    with driver.session() as session:
        return session.run(
            "MATCH (l:Link) RETURN l.url AS url, l.category AS category"
        ).data()


def align_grades_to_messages(grades: list, messages: list) -> tuple:
    """Pair each grade with exactly one live message by original_text.

    A repeated text's i-th grade (in graded_messages.json's own order)
    pairs with the i-th (chronologically earliest-first) live message
    with that text. Never fuzzy-matched: if a text's grade count and live
    message count disagree, every grade for that text is unmatched --
    picking any specific pairing then would be a guess.

    Returns ``(matched, unmatched)``; a matched row is
    ``{"sent_at", "text", "old_category", "new_category"}``.
    """
    messages_by_text: dict = defaultdict(list)
    for m in messages:
        messages_by_text[m["text"] or ""].append(m)
    for group in messages_by_text.values():
        group.sort(key=lambda m: m["sent_at"] or "")

    grades_by_text: dict = defaultdict(list)
    for g in grades:
        grades_by_text[g["original_text"] or ""].append(g)

    matched, unmatched = [], []
    for text, text_grades in grades_by_text.items():
        text_messages = messages_by_text.get(text, [])
        if text_messages and len(text_grades) == len(text_messages):
            for grade, message in zip(text_grades, text_messages):
                matched.append({
                    "sent_at": message["sent_at"], "text": text,
                    "old_category": message.get("category"),
                    "new_category": grade["category"],
                })
        else:
            unmatched.extend(text_grades)
    return matched, unmatched


def plan(grades: list, driver) -> dict:
    """Read-only: everything --dry-run and --apply both need to decide what
    to change, without writing to Qdrant or Neo4j."""
    matched, unmatched = align_grades_to_messages(grades, live_messages(driver))
    to_update = [row for row in matched if row["old_category"] != row["new_category"]]
    already_set = [row for row in matched if row["old_category"] == row["new_category"]]

    links = live_links(driver)
    links_to_update = [lk for lk in links if lk.get("category") != LINK_CATEGORY]
    links_already_set = [lk for lk in links if lk.get("category") == LINK_CATEGORY]

    return {
        "matched": matched, "unmatched": unmatched,
        "to_update": to_update, "already_set": already_set,
        "links_to_update": links_to_update, "links_already_set": links_already_set,
    }


def apply_messages(driver, to_update: list) -> None:
    """``SET m.category`` on an existing node -- never ``MERGE``/creates one."""
    with driver.session() as session:
        for row in to_update:
            session.run(
                "MATCH (m:Message {sent_at: $sent_at, text: $text}) "
                "SET m.category = $category",
                sent_at=row["sent_at"], text=row["text"], category=row["new_category"])


def apply_links_neo4j(driver, links_to_update: list) -> None:
    with driver.session() as session:
        for link in links_to_update:
            session.run(
                "MATCH (l:Link {url: $url}) SET l.category = $category",
                url=link["url"], category=LINK_CATEGORY)


def apply_qdrant(client, collection: str, to_update: list, links_to_update: list) -> None:
    """``set_payload`` only -- never ``upsert``, no re-embedding, no id
    change. Grouped by category value to minimize calls."""
    ids_by_category: dict = defaultdict(list)
    for row in to_update:
        ids_by_category[row["new_category"]].append(message_id(row["sent_at"], row["text"]))
    for link in links_to_update:
        ids_by_category[LINK_CATEGORY].append(link_id(link["url"]))
    for category, ids in ids_by_category.items():
        if ids:
            client.set_payload(collection_name=collection, payload={"category": category},
                               points=ids)


def ensure_category_index(client, collection: str) -> None:
    from qdrant_client.models import PayloadSchemaType
    client.create_payload_index(collection_name=collection, field_name="category",
                                field_schema=PayloadSchemaType.KEYWORD)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="read-only: print counts, write nothing (the default)")
    mode.add_argument("--apply", action="store_true",
                      help="write to Qdrant/Neo4j")
    ap.add_argument("--graded-file", default=GRADED_FILE)
    ap.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "manthan"))
    args = ap.parse_args(argv)

    from qdrant_client import QdrantClient
    from store import KnowledgeStore

    store = KnowledgeStore()
    client = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        https=False,
    )
    try:
        grades = load_list(args.graded_file)
        result = plan(grades, store.driver)

        print(f"matched={len(result['matched'])} unmatched={len(result['unmatched'])} "
              f"already_set={len(result['already_set'])} "
              f"to_update={len(result['to_update'])}")
        print(f"links: already_set={len(result['links_already_set'])} "
              f"to_update={len(result['links_to_update'])}")
        if result["unmatched"]:
            sample = [row["original_text"][:60] for row in result["unmatched"][:5]]
            print(f"unmatched sample (up to 5 of {len(result['unmatched'])}): {sample}")

        if args.apply:
            apply_messages(store.driver, result["to_update"])
            apply_links_neo4j(store.driver, result["links_to_update"])
            apply_qdrant(client, args.collection, result["to_update"],
                        result["links_to_update"])
            ensure_category_index(client, args.collection)
            print(f"applied: {len(result['to_update'])} message(s), "
                  f"{len(result['links_to_update'])} link(s)")
            print("restart the API or wait for the next request")
        else:
            print("--dry-run: no writes made")
    finally:
        store.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
