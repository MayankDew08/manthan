"""Freeze the production graph retrieval to a run file (spec section 6)."""

import hashlib
import inspect
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from search import _graph_candidates, _terms

from evals import common, live_guard

RUN_ID = "graph_lexical_v0"


def source_sha256() -> str:
    """SHA256 of the production _terms and _graph_candidates source code."""
    src = inspect.getsource(_terms) + inspect.getsource(_graph_candidates)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def candidate_id_for(result: dict) -> str:
    """Deterministic candidate ID matching vector_store.py formula."""
    if result.get("type") == "message":
        sent_at = result.get("sent_at", "")
        text = result.get("text", "")
        return "message:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"msg-{sent_at}-{text}"))
    elif result.get("type") == "link":
        url = result.get("url", "")
        return "link:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"link-{url}"))
    raise ValueError(f"Unknown result type: {result.get('type')}")


def matched_terms(result: dict, terms: list[str]) -> list[str]:
    """Distinct terms from `terms` that occur as case-insensitive substrings in result fields."""
    text = result.get("text", "") or ""
    title = result.get("title", "") or ""
    entities = " ".join(result.get("entities", []) or [])
    topics = " ".join(result.get("topics", []) or "")
    haystack = f"{text} {title} {entities} {topics}".lower()
    matched = []
    for term in terms:
        if term.lower() in haystack:
            matched.append(term)
    return matched


def graph_rows_for_query(
    query_id: str, terms: list[str], raw_results: list[dict], corpus_ids: set[str]
) -> tuple[list[dict], dict[str, Any]]:
    """Convert raw graph results to ranked rows with stats."""
    seen: dict[str, dict] = {}
    for res in raw_results:
        cid = candidate_id_for(res)
        if cid in seen:
            continue
        seen[cid] = res

    stats = {
        "query_id": query_id,
        "terms": terms,
        "raw_results": len(raw_results),
        "distinct": len(seen),
        "in_corpus": 0,
        "out_of_corpus": 0,
    }

    in_corpus_results = []
    for cid, res in seen.items():
        if cid in corpus_ids:
            in_corpus_results.append((cid, res))
        else:
            stats["out_of_corpus"] += 1

    stats["in_corpus"] = len(in_corpus_results)

    ranked = []
    for cid, res in in_corpus_results:
        matched = matched_terms(res, terms)
        ranked.append((cid, res, matched))

    ranked.sort(key=lambda x: (-len(x[2]), x[0]))

    rows = []
    for rank, (cid, res, matched) in enumerate(ranked, start=1):
        rows.append({
            "run_id": RUN_ID,
            "query_id": query_id,
            "candidate_id": cid,
            "rank": rank,
            "score": float(len(matched)),
            "matched_terms": matched,
        })

    return rows, stats


def cosine_mismatches(
    live_rows: list[dict], frozen_rows: list[dict], depth: int = 30, tol: float = 1e-4
) -> list[str]:
    """Detect real ranking changes between live and frozen cosine runs."""
    by_query_live: dict[str, list[dict]] = {}
    for r in live_rows:
        by_query_live.setdefault(r["query_id"], []).append(r)
    for qid in by_query_live:
        by_query_live[qid].sort(key=lambda x: x["rank"])

    by_query_frozen: dict[str, list[dict]] = {}
    for r in frozen_rows:
        by_query_frozen.setdefault(r["query_id"], []).append(r)
    for qid in by_query_frozen:
        by_query_frozen[qid].sort(key=lambda x: x["rank"])

    problems = []

    for qid, frozen_q in by_query_frozen.items():
        if qid not in by_query_live:
            problems.append(f"query {qid}: missing from live rows")
            continue

        live_q = by_query_live[qid]

        live_by_rank = {r["rank"]: r for r in live_q if r["rank"] <= depth}
        frozen_by_rank = {r["rank"]: r for r in frozen_q if r["rank"] <= depth}

        live_by_cid = {r["candidate_id"]: r for r in live_q if r["rank"] <= depth}

        for rank in range(1, depth + 1):
            if rank not in frozen_by_rank:
                break
            frow = frozen_by_rank[rank]
            cid = frow["candidate_id"]
            fscore = frow["score"]

            if cid not in live_by_cid:
                problems.append(f"query {qid}: frozen candidate {cid} at rank {rank} missing in live")
                continue

            lrow = live_by_cid[cid]
            lscore = lrow["score"]

            if abs(fscore - lscore) > tol:
                problems.append(
                    f"query {qid}: candidate {cid} frozen score {fscore} vs live {lscore} (diff > {tol})"
                )

            if rank in live_by_rank:
                lrow_at_rank = live_by_rank[rank]
                if abs(fscore - lrow_at_rank["score"]) > tol:
                    problems.append(
                        f"query {qid}: rank {rank} frozen score {fscore} vs live score {lrow_at_rank['score']} (diff > {tol})"
                    )

    return problems


def _run_cli() -> None:
    """CLI entry point: freeze graph retrieval live."""
    live_guard.assert_listener_stopped()

    qdrant_client = live_guard.qdrant_read_only_client()
    collection = os.environ.get("QDRANT_COLLECTION", "manthan")
    store = live_guard.read_only_store()
    neo4j_driver = store.driver  # ReadOnlyDriver, enforces READ_ACCESS

    fp_before = live_guard.fingerprint(qdrant_client, collection, neo4j_driver)

    queries = common.load_queries()

    all_rows: list[dict] = []
    per_query_stats: dict[str, dict] = {}
    latency_ms: dict[str, int] = {}

    try:
        for q in queries:
            qid = q["query_id"]
            text = q["query_text"]
            terms = _terms(text)

            start = time.perf_counter()
            raw = _graph_candidates(store, terms, None, None, 1.0)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            latency_ms[qid] = elapsed_ms

            corpus_docs = common.load_documents(common.DEFAULT_CORPUS_VERSION)
            corpus_ids = {d["candidate_id"] for d in corpus_docs}

            rows, stats = graph_rows_for_query(qid, terms, raw, corpus_ids)
            all_rows.extend(rows)
            per_query_stats[qid] = stats

        # Fingerprint again BEFORE closing the store
        fp_after = live_guard.fingerprint(qdrant_client, collection, neo4j_driver)
        live_guard.assert_unchanged(fp_before, fp_after)
    finally:
        store.close()

    runs_dir = common.data_dir(common.DEFAULT_CORPUS_VERSION) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    common.write_jsonl(runs_dir / f"{RUN_ID}.jsonl", all_rows)

    meta = {
        "run_id": RUN_ID,
        "source_sha256": source_sha256(),
        "per_query": per_query_stats,
        "latency_ms": latency_ms,
        "fingerprint_before": fp_before,
        "fingerprint_after": fp_after,
        "created_at": common.utc_now(),
        **common.git_state(),
    }

    (runs_dir / f"{RUN_ID}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    _run_cli()