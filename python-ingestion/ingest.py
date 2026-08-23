"""Centralized checkpointed ingestion graph for single-message ingestion.

Composes the LangGraph workflow from ``main.py`` behind one entrypoint that
accepts a single Telegram message: grading (skipped for trusted senders),
conditional link scraping, LLM summarization, record building for linkless
messages, artifact persistence, and a final sync into Neo4j and Qdrant. The
graph is built once per process and every invocation is checkpointed, so a
crashed or retried message resumes from its last completed node instead of
repeating LLM work.
"""

import datetime as dt
import hashlib
import os
import sqlite3
import threading
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

from grader import GradeResult, grade_messages
from link_extract import extract_links
from models import load_list
from parser import Data
from main import (ChatState, grade_pass2_node, merge_node,
                  persist_node, scrape_candidates_node, scrape_promoted_node)
import enhancer
import enrich

from pipeline import ASK_USER_FILE, ENRICHED_FILE, MIN_QUALITY, SCRAPED_FILE
from pipeline import push_to_stores

CHECKPOINT_DB = os.environ.get("MANTHAN_CHECKPOINT_DB", "state.sqlite")


class IngestState(ChatState, total=False):
    """Graph state extended with the raw bot records being ingested."""

    records: list
    trusted_flags: list
    stores: dict


def load_record_node(state: IngestState) -> dict:
    """Wrap bot message records as parsed candidates without heuristic filtering."""
    records = state.get("records") or []
    kept = [
        Data(
            datetime_iso=r.get("sent_at") or "",
            sender=r.get("sender") or "",
            text=(r.get("text") or "").strip(),
            is_media=False,
        )
        for r in records
    ]
    return {
        "kept": kept,
        "discarded": [],
        "trusted_flags": [bool(r.get("trusted")) for r in records],
    }


def grade_records_node(state: IngestState) -> dict:
    """Grade untrusted messages in one batch; trusted ones skip the LLM entirely."""
    kept = state["kept"]
    flags = state.get("trusted_flags") or [False] * len(kept)
    pending = [d for d, trusted in zip(kept, flags) if not trusted]
    results = iter(grade_messages(pending)) if pending else iter([])
    graded = []
    for d, trusted in zip(kept, flags):
        if trusted:
            graded.append(GradeResult(
                quality=5,
                confidence=1.0,
                category="resource",
                reason="trusted ingestion",
                topics=[],
                original_text=d.text,
            ))
        else:
            graded.append(next(results))
    print(f"\n--- graded: pass 1 ({sum(flags)} trusted skipped) ---")
    for msg in graded:
        print(msg)
    return {"graded": graded}


def route_links(state: IngestState) -> str:
    """Scrape only when at least one passing candidate carries a link."""
    for d, g in zip(state["kept"], state["graded"]):
        if g.quality >= enrich.DEFAULT_MIN_QUALITY and extract_links(g.original_text):
            return "scrape"
    return "skip"


def route_promoted(state: IngestState) -> str:
    """Re-scrape only when pass two promoted candidates above the threshold."""
    return "scrape" if state.get("promoted") else "records"


def build_records_node(state: IngestState) -> dict:
    """Synthesize Message records for passing messages that carry no links."""
    enriched = list(state.get("enriched") or [])
    by_key = {(r.get("sent_at"), r.get("sender"), r.get("original_text")): r
              for r in enriched}
    kept = state["kept"]
    flags = state.get("trusted_flags") or [False] * len(kept)
    added = 0
    for d, g, trusted in zip(kept, state["final"], flags):
        if g.quality < enrich.DEFAULT_MIN_QUALITY:
            continue
        key = (d.datetime_iso, d.sender, g.original_text)
        row = by_key.get(key)
        if row is not None:
            row["trusted"] = trusted
            continue
        en = enhancer.enrich_message(g.original_text)
        record = {
            "sent_at": d.datetime_iso,
            "sender": d.sender,
            "quality": g.quality,
            "original_text": g.original_text,
            "links": [],
            "link_intent": en["link_intent"],
            "entities": en["entities"],
            "topics": en["topics"],
            "trusted": trusted,
        }
        enriched.append(record)
        by_key[key] = record
        added += 1
    if added:
        print(f"\n[ingest] built {added} linkless message records "
              "(metadata via enhancer)")
    return {"enriched": enriched}


def sync_stores_node(state: IngestState, config) -> dict:
    """Push graded messages, links, and pending links into Neo4j and Qdrant."""
    cfg = config.get("configurable") or {}
    summary = push_to_stores(
        state["enriched"], state["scraped"], state.get("ask_user", []),
        MIN_QUALITY, blocked=state.get("blocked", []),
        store=cfg.get("store"), vs=cfg.get("vs"),
    )
    print(f"\n[ingest] synced stores: {summary}")
    return {"stores": summary}


def _skip_scrape_node(state: IngestState) -> dict:
    """Emit empty link artifacts so persist sees the shape the scrape path produces."""
    no_links = sum(1 for g in state["graded"]
                   if g.quality >= enrich.DEFAULT_MIN_QUALITY)
    return {
        "enriched": [],
        "scraped": [],
        "blocked": [],
        "ask_user": [],
        "seen": set(),
        "stats": {"scraped": 0, "blocked": 0, "ask_user": 0,
                  "skipped_existing": 0, "no_links": no_links},
    }


def build_ingestion_graph(checkpointer=None):
    """Build the single-message DAG with conditional link-scraping branches."""
    graph = StateGraph(IngestState)
    graph.add_node("load_record", load_record_node)
    graph.add_node("grade_pass1", grade_records_node)
    graph.add_node("grade_pass2", grade_pass2_node)
    graph.add_node("scrape_candidates", scrape_candidates_node)
    graph.add_node("skip_scrape", _skip_scrape_node)
    graph.add_node("merge", merge_node)
    graph.add_node("scrape_promoted", scrape_promoted_node)
    graph.add_node("build_records", build_records_node)
    graph.add_node("persist", persist_node)
    graph.add_node("sync_stores", sync_stores_node)
    graph.add_edge(START, "load_record")
    graph.add_edge("load_record", "grade_pass1")
    graph.add_conditional_edges(
        "grade_pass1", route_links,
        {"scrape": "scrape_candidates", "skip": "skip_scrape"},
    )
    graph.add_edge("grade_pass1", "grade_pass2")
    graph.add_edge("scrape_candidates", "merge")
    graph.add_edge("skip_scrape", "merge")
    graph.add_edge("grade_pass2", "merge")
    graph.add_conditional_edges(
        "merge", route_promoted,
        {"scrape": "scrape_promoted", "records": "build_records"},
    )
    graph.add_edge("scrape_promoted", "build_records")
    graph.add_edge("build_records", "persist")
    graph.add_edge("persist", "sync_stores")
    graph.add_edge("sync_stores", END)
    return graph.compile(checkpointer=checkpointer)


_GRAPH = None
_SAVER_CONN = None
_GRAPH_LOCK = threading.Lock()


def _get_graph():
    """Compile the graph once with a long-lived checkpoint connection."""
    global _GRAPH, _SAVER_CONN
    with _GRAPH_LOCK:
        if _GRAPH is None:
            _SAVER_CONN = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
            _SAVER_CONN.execute("PRAGMA journal_mode=WAL")
            _GRAPH = build_ingestion_graph(checkpointer=SqliteSaver(_SAVER_CONN))
        return _GRAPH


def _reset_graph() -> None:
    """Drop the compiled graph and its checkpoint connection (used by tests)."""
    global _GRAPH, _SAVER_CONN
    with _GRAPH_LOCK:
        if _SAVER_CONN is not None:
            _SAVER_CONN.close()
        _GRAPH = None
        _SAVER_CONN = None


def _invoke_graph(input_state: dict, thread_id: str,
                  store=None, vs=None) -> dict:
    """Run, resume, or reuse a checkpointed invocation for ``thread_id``.

    Store clients ride in ``configurable`` so they are available to nodes on
    every path without ever being written into the checkpoint.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id,
                               "store": store, "vs": vs}}
    snap = graph.get_state(config)
    if snap is not None and getattr(snap, "next", None):
        print(f"[ingest] resuming thread {thread_id} "
              f"(pending nodes: {list(snap.next)})")
        return graph.invoke(None, config=config)
    if snap is not None and snap.values:
        print(f"[ingest] thread {thread_id} already completed; reusing checkpoint")
        return dict(snap.values)
    return graph.invoke(input_state, config=config)


def _message_thread_id(sender: str, sent_at: str, text: str) -> str:
    digest = hashlib.sha1(f"{sender}|{sent_at}|{text}".encode("utf-8")).hexdigest()
    return f"msg-{digest[:16]}"


def _link_outcomes(row: dict) -> list:
    """Condense stored link previews into the API response shape."""
    outcomes = []
    for preview in row.get("link_previews") or []:
        item = {"url": preview.get("url"), "status": preview.get("status")}
        for key in ("block_reason", "title", "summary"):
            if preview.get(key):
                item[key] = preview[key]
        outcomes.append(item)
    return outcomes


def ingest_message(text: str, sender: str = "", sent_at: Optional[str] = None,
                   trusted: bool = False,
                   store=None, vs=None) -> dict:
    """Run the full pipeline for one bot message and persist the results."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is required")
    sent_at = sent_at or dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        "text": text,
        "sender": sender or "",
        "sent_at": sent_at,
        "trusted": bool(trusted),
    }
    thread_id = _message_thread_id(record["sender"], sent_at, text)
    state = _invoke_graph({"records": [record]}, thread_id,
                          store=store, vs=vs)

    row = next(
        (r for r in state.get("enriched", [])
         if (r.get("sent_at"), r.get("sender"), r.get("original_text"))
         == (sent_at, record["sender"], text)),
        None,
    ) or {}
    final = state.get("final") or []
    quality = final[0].quality if final else 0

    return {
        "ok": True,
        "message": {
            "sender": record["sender"],
            "sent_at": sent_at,
            "quality": quality,
            "trusted": bool(trusted),
            "topics": row.get("topics") or [],
            "entities": row.get("entities") or [],
            "links": row.get("links") or [],
        },
        "vectored": bool(row) and quality >= MIN_QUALITY,
        "link_outcomes": _link_outcomes(row),
        "stats": state.get("stats", {}),
        "stores": state.get("stores", {}),
    }


def main():
    """Legacy artifact-push entrypoint retained for manual store backfills."""
    load_dotenv()
    enriched = load_list(ENRICHED_FILE)
    scraped = load_list(SCRAPED_FILE)
    ask_user = load_list(ASK_USER_FILE)
    blocked = load_list(enrich.BLOCKED_FILE)
    summary = push_to_stores(enriched, scraped, ask_user, MIN_QUALITY,
                             blocked=blocked)
    print(f"Ingested {summary['messages']} messages -> Neo4j "
          f"({summary['vectored_messages']} quality >= {MIN_QUALITY} -> Qdrant)")
    print(f"Ingested {summary['scraped_links']} scraped links -> Neo4j "
          f"({summary['vectored_links']} with content -> Qdrant)")
    print(f"Marked {summary['blocked_links']} blocked links and "
          f"{summary['pending_links']} ask-user links in Neo4j")


if __name__ == "__main__":
    main()