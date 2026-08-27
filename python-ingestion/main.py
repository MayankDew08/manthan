"""Run the checkpointed LangGraph ingestion pipeline for a chat export.

The graph overlaps pass-two grading with link scraping, then persists the
resulting JSON artifacts. ``enrich.py`` also exposes the same core processing
for callers that do not need LangGraph checkpointing.
"""

import hashlib
import os
import sys
from dataclasses import asdict
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from parser import parse_chat
from heuristic_filter import heuristic_filter
from grader import grade_messages, grade_pass2
import enrich
import llm
import metrics

from store import KnowledgeStore
from vector_store import VectorStore

CHAT_PATH = os.environ.get("MANTHAN_CHAT_PATH", "tests/test_chat.txt")
CHECKPOINT_DB = os.environ.get("MANTHAN_CHECKPOINT_DB", "state.sqlite")

store = KnowledgeStore()
vs = VectorStore()



class ChatState(TypedDict, total=False):
    """Values accumulated as a chat moves through the ingestion graph."""

    chat_path: str
    kept: list
    discarded: list
    graded: list
    verified: list
    final: list
    promoted: list
    enriched: list
    scraped: list
    blocked: list
    ask_user: list
    seen: set
    stats: dict


def parse_node(state):
    """Parse the export and remove records rejected by cheap heuristics."""
    messages = parse_chat(state["chat_path"])
    kept, discarded = heuristic_filter(messages)
    reasons = {}
    for d in discarded:
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    print(f"Total: {len(kept) + len(discarded)}")
    print(f"Kept: {len(kept)}")
    print(f"Discarded: {len(discarded)}")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")
    return {"kept": kept, "discarded": discarded}


def grade_pass1_node(state):
    """Assign initial quality scores and metadata to retained messages."""
    graded = grade_messages(state["kept"])
    print("\n--- graded: pass 1 ---")
    for msg in graded:
        print(msg)
    return {"graded": graded}


def grade_pass2_node(state):
    """Recheck borderline pass-one messages using neighboring context."""
    verified = grade_pass2(state["graded"])
    print("\n--- graded: pass 2 (re-graded quality=3 only) ---")
    for msg in verified:
        print(msg)
    return {"verified": verified}


def scrape_candidates_node(state):
    """Scrape high-quality pass-one candidates while pass two runs in parallel."""
    kept, graded = state["kept"], state["graded"]
    candidates = [(d, g) for d, g in zip(kept, graded)
                  if g.quality >= enrich.DEFAULT_MIN_QUALITY]
    print(f"\n[main] scraping {len(candidates)} pass-1 candidates "
          f"(parallel with grade pass 2)")
    enriched, scraped, blocked, ask_user, stats, seen = enrich.process_candidates(candidates)
    return {"enriched": enriched, "scraped": scraped, "blocked": blocked,
            "ask_user": ask_user, "seen": seen, "stats": stats}


def merge_node(state):
    """Merge verified grades and identify newly promoted scrape candidates."""
    final = enrich._merge_pass2(state["graded"], state["verified"])
    promoted = [(d, g) for d, g, p1 in zip(state["kept"], final, state["graded"])
                if p1.quality == 3 and g.quality >= enrich.DEFAULT_MIN_QUALITY]
    print("\n--- final: merged (quality overwritten by pass 2) ---")
    for msg in final:
        print(msg)
    if promoted:
        print(f"\n[main] {len(promoted)} pass-2 promoted candidates -> scrape phase B")
    return {"final": final, "promoted": promoted}


def scrape_promoted_node(state):
    """Scrape messages promoted above the threshold by pass two."""
    if not state.get("promoted"):
        return {}
    enriched, scraped, blocked, ask_user, stats, seen = enrich.process_candidates(
        state["promoted"], seen=state["seen"], scraped=state["scraped"],
        blocked=state["blocked"], enriched=state["enriched"], ask_user=state["ask_user"])
    merged = dict(state.get("stats", {}))
    for k, v in stats.items():
        merged[k] = merged.get(k, 0) + v
    return {"enriched": enriched, "scraped": scraped, "blocked": blocked,
            "ask_user": ask_user, "seen": seen, "stats": merged}


def persist_node(state):
    """Write final grades and link artifacts after both graph branches converge."""
    preview_backfill_keys = {
        (record.get("sent_at"), record.get("sender"), record.get("original_text"))
        for record in state["enriched"]
        if record.get("links") and "link_previews" not in record
    }
    enriched = (
        enrich._attach_link_previews(
            state["enriched"],
            state["scraped"],
            state["blocked"],
            state["ask_user"],
            preview_backfill_keys,
        )
        if preview_backfill_keys
        else state["enriched"]
    )
    enrich.save_list(enrich.GRADED_FILE, [asdict(m) for m in state["final"]])
    enrich.save_list(enrich.SCRAPED_FILE, state["scraped"])
    enrich.save_list(enrich.BLOCKED_FILE, state["blocked"])
    enrich.save_list(enrich.ASK_USER_FILE, state["ask_user"])
    enrich.save_list(enrich.ENRICHED_FILE, enriched)
    stats = state.get("stats", {})
    print(f"\nSaved {len(state['final'])} messages to {enrich.GRADED_FILE}")
    print(f"[main] done: {stats.get('scraped', 0)} scraped, {stats.get('blocked', 0)} blocked, "
          f"{stats.get('ask_user', 0)} ask-user, {stats.get('skipped_existing', 0)} existing, "
          f"{stats.get('no_links', 0)} candidates without links")
    print(f"[main] tokens used: {llm.usage_stats.total_tokens} "
          f"({llm.usage_stats.llm_calls} LLM calls)")
    metrics.write_last_run(source="langgraph", chat_path=state.get("chat_path"),
                           stats=stats)
    return {"enriched": enriched}


def build_graph(checkpointer=None):
    """Build the ingestion DAG, including its parallel grading/scraping branch."""
    graph = StateGraph(ChatState)
    graph.add_node("parse", parse_node)
    graph.add_node("grade_pass1", grade_pass1_node)
    graph.add_node("grade_pass2", grade_pass2_node)
    graph.add_node("scrape_candidates", scrape_candidates_node)
    graph.add_node("merge", merge_node)
    graph.add_node("scrape_promoted", scrape_promoted_node)
    graph.add_node("persist", persist_node)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "grade_pass1")
    graph.add_edge("grade_pass1", "scrape_candidates")
    graph.add_edge("grade_pass1", "grade_pass2")
    graph.add_edge("scrape_candidates", "merge")
    graph.add_edge("grade_pass2", "merge")
    graph.add_edge("merge", "scrape_promoted")
    graph.add_edge("scrape_promoted", "persist")
    graph.add_edge("persist", END)
    return graph.compile(checkpointer=checkpointer)


def run_pipeline(chat_path: str) -> dict:
    from langchain_core.runnables.config import RunnableConfig
    """Run or resume a chat-specific graph using a stable checkpoint thread ID."""
    thread = "manthan-" + hashlib.sha1(chat_path.encode("utf-8")).hexdigest()[:16]
    config = RunnableConfig({"configurable": {"thread_id": thread}})
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        app = build_graph(checkpointer=saver)
        snap = app.get_state(config)
        pending = bool(snap is not None and getattr(snap, "next", None))
        if pending:
            print(f"[main] resuming run from last checkpoint "
                  f"(pending nodes: {list(snap.next)})")
        return app.invoke({"chat_path": chat_path}, config=config) \
            if not pending else app.invoke(None, config=config)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CHAT_PATH
    run_pipeline(path)
