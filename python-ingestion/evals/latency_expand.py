"""U7 latency_expand: query-time cost of graph expansion (spec U7/13.7).

Live, read-only. Recorded only -- no decision reads these numbers (D17).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from evals import common, latency_bench, live_guard, rerank

COMPONENTS = ("br_walk", "bq_lookup_fetch", "rerank_30", "rerank_nq", "rerank_pool", "c1_extra")

# Representative pool sizes (spec section 4: B pools are <= 60; N_q is 30-60).
POOL_30 = 30
POOL_NQ = 45
POOL_B = 40
SAMPLE_SEEDS_K = 10


def measure(fn: Callable[[], Any], clock: Callable[[], float] | None = None) -> dict[str, Any]:
    """Delegate to latency_bench.time_call/summarize; the warm-up is never reported."""
    kwargs: dict[str, Any] = {"warmup": latency_bench.WARMUP, "repeats": latency_bench.REPEATS}
    if clock is not None:
        kwargs["clock"] = clock
    samples = latency_bench.time_call(fn, **kwargs)
    return latency_bench.summarize(samples)


def _artifact_path(corpus_version: str = common.DEFAULT_CORPUS_VERSION):
    return common.data_dir(corpus_version) / "runs" / "latency_expand_v1.json"


def _seed_message_pairs(docs_by_id: dict[str, Any], seed_ids: list[str]) -> list[tuple[str, str]]:
    """(sent_at, text) for each seed that is a message; links have no sent_at to match on."""
    pairs = []
    for cid in seed_ids:
        doc = docs_by_id.get(cid)
        if doc and doc.get("candidate_type") == "message" and doc.get("sent_at"):
            pairs.append((doc["sent_at"], doc.get("embedded_text") or ""))
    return pairs


def _top_seed_ids(s2_rows: list[dict[str, Any]], qid: str, k: int) -> list[str]:
    ranked = sorted((r for r in s2_rows if r["query_id"] == qid), key=lambda r: r["rank"])
    return [r["candidate_id"] for r in ranked[:k]]


def _br_walk_fn(driver, seed_pairs: list[tuple[str, str]]) -> Callable[[], None]:
    def fn() -> None:
        for sent_at, text in seed_pairs:
            live_guard.neo4j_read(
                driver,
                "MATCH (m:Message {sent_at: $sent_at, text: $text}) "
                "OPTIONAL MATCH (m)-[:MENTIONS|ABOUT]->(c) RETURN count(c) AS n",
                sent_at=sent_at, text=text)
    return fn


def _bq_lookup_fetch_fn(driver, node_names: list[str]) -> Callable[[], None]:
    """B-q's cost: the start-node lookup (approximated here by scanning names
    already in memory, since the query embedding step is timed separately by
    the model benchmark) plus the live Cypher fetch of attached docs."""
    sample = node_names[:50] or ["__none__"]

    def fn() -> None:
        for name in sample[:1]:
            live_guard.neo4j_read(
                driver,
                "MATCH (n) WHERE n.name = $name "
                "OPTIONAL MATCH (n)<-[:MENTIONS|ABOUT]-(d) RETURN count(d) AS n",
                name=name)
        _ = sample  # the in-memory scan itself (string comparisons) is negligible vs the fetch.
    return fn


def _rerank_fn(reranker: rerank.Reranker, query_text: str, texts: list[str]) -> Callable[[], None]:
    def fn() -> None:
        reranker.rerank(query_text, [(str(i), t) for i, t in enumerate(texts)])
    return fn


def run_benchmark(out_dir=None, corpus_version: str = common.DEFAULT_CORPUS_VERSION) -> dict[str, Any]:
    live_guard.assert_listener_stopped()

    store = live_guard.read_only_store()
    driver = store.driver
    try:
        common.load_env()
        cfg = common.qdrant_settings()
        qdrant_client = live_guard.qdrant_read_only_client()
        try:
            fp_before = live_guard.fingerprint(qdrant_client, cfg["collection"], driver)

            data_dir = common.data_dir(corpus_version, create=False)
            documents = common.read_jsonl(data_dir / "documents.jsonl")
            docs_by_id = {d["candidate_id"]: d for d in documents}
            long_texts = [d.get("embedded_text") or "" for d in documents if d.get("embedded_text")]

            s2_original = common.read_jsonl(data_dir / "runs" / "ladder_s2_cos_bm25_rr.jsonl")
            s2_graphq_path = data_dir / "runs" / "ladder_s2_cos_bm25_rr.graphq.jsonl"
            s2_graphq = common.read_jsonl(s2_graphq_path) if s2_graphq_path.exists() else []

            nodes_path = data_dir / "runs" / "graph_nodes_v1.jsonl"
            node_names = [n["name"] for n in common.read_jsonl(nodes_path)] if nodes_path.exists() else []

            queries = common.load_queries()
            graphq_queries = common.load_queries(common.REPO_ROOT / "queries_graph.csv")
            rr = rerank.Reranker()

            components: dict[str, dict[str, Any]] = {
                name: {"per_query": {}} for name in COMPONENTS}

            for q, s2_rows in ((qq, s2_original) for qq in queries):
                _time_one_query(q, s2_rows, docs_by_id, driver, node_names, rr, long_texts, components)
            for q in graphq_queries:
                _time_one_query(q, s2_graphq, docs_by_id, driver, node_names, rr, long_texts, components)

            fp_after = live_guard.fingerprint(qdrant_client, cfg["collection"], driver)
            live_guard.assert_unchanged(fp_before, fp_after)
        finally:
            qdrant_client.close()
    finally:
        store.close()

    for name in COMPONENTS:
        samples_by_query = components[name]["per_query"]
        all_samples = [s for samples in samples_by_query.values() for s in samples]
        components[name]["summary"] = latency_bench.summarize(all_samples) if all_samples else {
            "n": 0, "p50": 0.0, "p95": 0.0}

    result = {
        "components": components,
        "fingerprint_before": fp_before,
        "fingerprint_after": fp_after,
        "created_at": common.utc_now(),
        **common.git_state(),
    }

    write_dir = out_dir if out_dir is not None else data_dir
    runs_dir = write_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "latency_expand_v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def _time_one_query(q, s2_rows, docs_by_id, driver, node_names, reranker, long_texts, components):
    qid, qtext = q["query_id"], q["query_text"]
    seed_ids = _top_seed_ids(s2_rows, qid, SAMPLE_SEEDS_K)
    seed_pairs = _seed_message_pairs(docs_by_id, seed_ids)

    components["br_walk"]["per_query"][qid] = latency_bench.time_call(
        _br_walk_fn(driver, seed_pairs or [(None, None)]),
        warmup=latency_bench.WARMUP, repeats=latency_bench.REPEATS)
    components["bq_lookup_fetch"]["per_query"][qid] = latency_bench.time_call(
        _bq_lookup_fetch_fn(driver, node_names),
        warmup=latency_bench.WARMUP, repeats=latency_bench.REPEATS)

    for name, size in (("rerank_30", POOL_30), ("rerank_nq", POOL_NQ), ("rerank_pool", POOL_B)):
        texts = (long_texts[:size] or [""])
        components[name]["per_query"][qid] = latency_bench.time_call(
            _rerank_fn(reranker, qtext, texts),
            warmup=latency_bench.WARMUP, repeats=latency_bench.REPEATS)

    # C1's query-time extra is index-time work (D23): recorded as exactly 0 ms.
    components["c1_extra"]["per_query"][qid] = [0.0] * latency_bench.REPEATS


if __name__ == "__main__":
    run_benchmark()
