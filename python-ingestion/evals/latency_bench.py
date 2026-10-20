"""U8 — Live latency benchmark for the retrieval ladder (spec section 12)."""

import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Callable

# Constants per spec section 12
WARMUP = 1
REPEATS = 5
COMPONENTS = ("cosine_qdrant", "graph_neo4j", "bm25", "rrf", "rerank_cpu", "rerank_gpu")


def time_call(
    fn: Callable[[], Any],
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Call ``fn`` ``warmup + repeats`` times, discard warm-up, return elapsed ms.

    Each measured interval is ``clock()`` after the call minus ``clock()`` before it,
    converted to milliseconds. ``clock`` is injectable for testing.
    """
    # Warm-up calls
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(repeats):
        start = clock()
        fn()
        end = clock()
        samples.append((end - start) * 1000.0)
    return samples


def summarize(samples: list[float]) -> dict[str, Any]:
    """Nearest-rank percentiles: s[ceil(q*n) - 1] on sorted samples.

    Returns ``{"n": n, "p50": ..., "p95": ...}``. Empty input raises ValueError.
    """
    if not samples:
        raise ValueError("summarize requires at least one sample")
    s = sorted(samples)
    n = len(s)

    def pctl(q: float) -> float:
        idx = math.ceil(q * n) - 1
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        return s[idx]

    return {"n": n, "p50": pctl(0.5), "p95": pctl(0.95)}


def component_summary(per_query: dict[str, list[float]]) -> dict[str, Any]:
    """Summarize a component over per-query medians.

    For each query, take the median of its REPEATS samples, then run
    ``summarize`` over those medians. Returns ``{"n": n_queries, "p50", "p95"}``.
    """
    medians = []
    for qid, samples in per_query.items():
        if not samples:
            continue
        s = sorted(samples)
        m = len(s)
        median = s[m // 2] if m % 2 == 1 else (s[m // 2 - 1] + s[m // 2]) / 2.0
        medians.append(median)
    return summarize(medians)


def machine_info() -> dict[str, Any]:
    """Return platform information for the latency report."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": platform.os.cpu_count() if hasattr(platform.os, "cpu_count") else 1,
        "gpu": None,
    }
    # Try to detect CUDA GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _load_corpus() -> tuple[list[dict], list[dict]]:
    """Load queries and documents for in-process components."""
    from evals import common

    queries = common.load_queries()
    docs = common.load_documents(common.DEFAULT_CORPUS_VERSION)
    return queries, docs


def _load_frozen_cosine() -> list[dict]:
    """Load the frozen vector_cosine_v1 run for RRF."""
    from evals import common

    return common.load_run(common.DEFAULT_CORPUS_VERSION, "vector_cosine_v1")


def _build_rrf_rankings(
    queries: list[dict],
    cosine_rows: list[dict],
    bm25_scorer,
    doc_ids: list[str],
) -> dict[str, list[tuple[str, float]]]:
    """Build RRF-fused rankings per query from cosine + BM25.

    Returns a mapping query_id -> list of (candidate_id, score) pairs.
    """
    from evals.fusion import ranking, rrf, RRF_K, DEPTH

    # Group cosine rows by query
    cosine_by_q: dict[str, list[str]] = {}
    for row in cosine_rows:
        qid = row["query_id"]
        cid = row["candidate_id"]
        cosine_by_q.setdefault(qid, []).append(cid)

    # Build BM25 rankings per query
    bm25_by_q: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        scored = bm25_scorer.top(q["query_text"], len(doc_ids))
        bm25_by_q[qid] = [doc_ids[i] for i, _ in scored]

    # Fuse per query
    fused: dict[str, list[tuple[str, float]]] = {}
    for q in queries:
        qid = q["query_id"]
        cos = cosine_by_q.get(qid, [])
        bm = bm25_by_q.get(qid, [])
        fused[qid] = rrf([cos, bm], k=RRF_K, depth=DEPTH)
    return fused


def main() -> None:
    """Run the latency benchmark and write runs/latency_v1.json."""
    from evals import common
    from evals.live_guard import (
        assert_listener_stopped,
        assert_unchanged,
        fingerprint,
        read_only_store,
        read_only_vector_store,
    )
    from evals.bm25 import BM25
    from evals.fusion import rrf, RRF_K, DEPTH
    from evals.rerank import Reranker
    from search import _graph_candidates, _terms

    # Load environment from main checkout (gitignored .env)
    # This is done by the caller via `set -a; . /path/.env; set +a`

    # Listener check and initial fingerprint
    assert_listener_stopped()
    vs = read_only_vector_store()
    ns = read_only_store()
    fp_before = fingerprint(vs.client, vs.collection, ns.driver)

    queries, docs = _load_corpus()
    qids = [q["query_id"] for q in queries]

    # Build document ID list for BM25 (embedded_text)
    doc_ids = [d["candidate_id"] for d in docs]
    doc_texts = [common.embedded_text(d) for d in docs]
    bm25_scorer = BM25(doc_texts)

    # Load frozen cosine run for RRF
    cosine_rows = _load_frozen_cosine()

    # Build RRF rankings (cosine + BM25) for timing the rrf component
    rrf_rankings = _build_rrf_rankings(queries, cosine_rows, bm25_scorer, doc_ids)

    # Prepare rerank inputs: 30 (query, text) pairs per query from RRF top-30
    rerank_inputs: dict[str, list[tuple[str, str]]] = {}
    doc_by_id = {d["candidate_id"]: d for d in docs}
    for q in queries:
        qid = q["query_id"]
        fused = rrf_rankings.get(qid, [])
        top30 = fused[:DEPTH]
        pairs = []
        for cid, _score in top30:
            # Find document by id
            doc = doc_by_id.get(cid)
            if doc:
                pairs.append((q["query_text"], common.embedded_text(doc)))
        rerank_inputs[qid] = pairs

    # Components to time
    components_per_query: dict[str, dict[str, list[float]]] = {
        name: {qid: [] for qid in qids} for name in COMPONENTS
    }

    # --- cosine_qdrant ---
    for q in queries:
        qid = q["query_id"]
        text = q["query_text"]

        def cosine_fn(t=text):
            vs.search(t, 30)

        samples = time_call(cosine_fn)
        components_per_query["cosine_qdrant"][qid] = samples

    # --- graph_neo4j ---
    for q in queries:
        qid = q["query_id"]
        text = q["query_text"]
        terms = _terms(text)

        def graph_fn(t=terms):
            _graph_candidates(ns, t, None, None, 1.0)

        samples = time_call(graph_fn)
        components_per_query["graph_neo4j"][qid] = samples

    # --- bm25 ---
    for q in queries:
        qid = q["query_id"]
        text = q["query_text"]

        def bm25_fn(t=text):
            bm25_scorer.scores(t)

        samples = time_call(bm25_fn)
        components_per_query["bm25"][qid] = samples

    # --- rrf ---
    # RRF takes pre-built rankings (cosine + BM25) and fuses them
    for q in queries:
        qid = q["query_id"]
        cos = [row["candidate_id"] for row in cosine_rows if row["query_id"] == qid]
        bm_scored = bm25_scorer.top(q["query_text"], len(doc_ids))
        bm = [doc_ids[i] for i, _ in bm_scored]

        def rrf_fn(c=cos, b=bm):
            rrf([c, b], k=RRF_K, depth=DEPTH)

        samples = time_call(rrf_fn)
        components_per_query["rrf"][qid] = samples

    # --- rerank_cpu ---
    reranker = Reranker()  # defaults to CPU
    for q in queries:
        qid = q["query_id"]
        pairs = rerank_inputs.get(qid, [])

        def rr_fn(p=pairs):
            if p:
                reranker.score_pairs(p)

        samples = time_call(rr_fn)
        components_per_query["rerank_cpu"][qid] = samples

    # --- rerank_gpu (only if CUDA available) ---
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass

    if has_cuda:
        # Create a GPU reranker
        from sentence_transformers import CrossEncoder
        from evals.rerank import MODEL_NAME, MODEL_REVISION, MAX_LENGTH

        gpu_model = CrossEncoder(
            MODEL_NAME,
            revision=MODEL_REVISION,
            device="cuda",
            max_length=MAX_LENGTH,
            local_files_only=True,
        )
        gpu_model.model.eval()
        gpu_reranker = Reranker(score_pairs=gpu_model.predict)

        for q in queries:
            qid = q["query_id"]
            pairs = rerank_inputs.get(qid, [])

            def gpu_fn(p=pairs):
                if p:
                    gpu_reranker.score_pairs(p)

            samples = time_call(gpu_fn)
            components_per_query["rerank_gpu"][qid] = samples
    else:
        # Fill with empty lists (won't be checked by artifact test for rerank_gpu)
        for qid in qids:
            components_per_query["rerank_gpu"][qid] = []

    # Final fingerprint
    fp_after = fingerprint(vs.client, vs.collection, ns.driver)
    assert_unchanged(fp_before, fp_after)

    # Build output
    components_out = {}
    for name in COMPONENTS:
        per_query = components_per_query[name]
        summary = component_summary(per_query)
        components_out[name] = {"per_query": per_query, "summary": summary}

    output = {
        "machine": machine_info(),
        "components": components_out,
        "fingerprint_before": fp_before,
        "fingerprint_after": fp_after,
    }

    # Write output
    runs_dir = common.data_dir(common.DEFAULT_CORPUS_VERSION, create=True) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "latency_v1.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # Clean up
    vs.close()
    ns.close()


if __name__ == "__main__":
    main()