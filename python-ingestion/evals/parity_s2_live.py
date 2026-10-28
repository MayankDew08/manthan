"""Phase 1 parity gate for S2 (docs/mcp-server-spec.md section 2, Q9).

1. Run evals.parity_check.
2. Call retrieval.search_s2 in-process for every non-graph query in queries.csv.
3. Compare against the frozen run evals/data/pilot-001/runs/ladder_s2_cos_bm25_rr.jsonl:
   - parity_check PASSED -> exact branch: top-5 ids must be identical, in the
     same order, for every query. A live id is mapped to the corpus's
     candidate_id (directly, or through the D25 alias map) before comparing.
   - parity_check FAILED (corpus drifted) -> nDCG@5 branch: mean nDCG@5 on
     qrels_v3.csv, over answerable queries only, must be within +/-0.02 of
     the frozen benchmark 0.7517.
4. Exits non-zero on failure. Always prints which branch ran.

Usage (from python-ingestion/):
  uv run python -m evals.parity_s2_live
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import retrieval  # noqa: E402
from evals import common, ladder_stats, metrics as M, parity_check  # noqa: E402
from evals.freeze_neighbors import build_alias_map  # noqa: E402
from evals.fusion import DEPTH  # noqa: E402
from evals.rerank import Reranker  # noqa: E402
from vector_store import VectorStore  # noqa: E402

TOLERANCE = 0.02
BENCHMARK_NDCG = 0.7517
FROZEN_RUN = "ladder_s2_cos_bm25_rr"


def _non_graph_queries(corpus_version: str) -> list[dict]:
    return [q for q in common.load_queries() if not q["query_id"].startswith("g")]


def _frozen_top5(corpus_version: str, run_id: str) -> dict[str, list[str]]:
    rows = common.load_run(corpus_version, run_id)
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    out = {}
    for qid, qrows in by_query.items():
        qrows.sort(key=lambda r: r["rank"])
        out[qid] = [r["candidate_id"] for r in qrows[:5]]
    return out


def run_exact_branch(queries: list[dict], vs, reranker, corpus_version: str) -> tuple[bool, dict]:
    """Top-5 live ids, mapped to corpus candidate_id, must equal the frozen run."""
    corpus_docs = common.load_documents(corpus_version)
    by_qdrant_id = {d["qdrant_point_id"]: d["candidate_id"] for d in corpus_docs}
    alias_map = build_alias_map(corpus_docs)
    frozen = _frozen_top5(corpus_version, FROZEN_RUN)

    live_corpus = retrieval.Corpus()
    mismatches = []
    for q in queries:
        out = retrieval.search_s2(q["query_text"], top_k=5, vs=vs, reranker=reranker,
                                  corpus=live_corpus)
        live_ids = [by_qdrant_id.get(r["id"]) or alias_map.get(r["id"])
                    or f"{r['type']}:{r['id']}" for r in out["results"]]
        expected = frozen.get(q["query_id"], [])
        if live_ids != expected:
            mismatches.append({"query_id": q["query_id"], "live": live_ids, "frozen": expected})
    return (not mismatches), {"queries_checked": len(queries), "mismatches": mismatches}


def run_ndcg_branch(queries: list[dict], vs, reranker, corpus_version: str) -> tuple[bool, dict]:
    """Mean nDCG@5 on qrels_v3, over answerable queries, within tolerance of the benchmark."""
    data_dir = common.data_dir(corpus_version, create=False)
    corpus_docs = common.load_documents(corpus_version)
    valid_ids = {d["candidate_id"] for d in corpus_docs}
    qrels = common.load_qrels(data_dir / "qrels_v3.csv", valid_ids=valid_ids)

    query_ids = [q["query_id"] for q in queries]
    answerable_ids = ladder_stats.answerable(qrels, query_ids)
    by_qid = {q["query_id"]: q for q in queries}

    live_corpus = retrieval.Corpus()
    per_query_ndcg: dict[str, float] = {}
    for qid in answerable_ids:
        out = retrieval.search_s2(by_qid[qid]["query_text"], top_k=DEPTH, vs=vs,
                                  reranker=reranker, corpus=live_corpus)
        grades = []
        for r in out["results"]:
            entry = qrels.get(qid, {}).get(f"{r['type']}:{r['id']}")
            grades.append(int(entry["relevance"]) if entry else 0)
        judged = [int(e["relevance"]) for e in qrels.get(qid, {}).values()]
        ndcg = M.ndcg_at_k(grades, judged, M.TOP_K)
        per_query_ndcg[qid] = 0.0 if ndcg is None else float(ndcg)

    mean_ndcg = sum(per_query_ndcg.values()) / len(per_query_ndcg) if per_query_ndcg else 0.0
    ok = abs(mean_ndcg - BENCHMARK_NDCG) <= TOLERANCE
    return ok, {"answerable_queries": len(answerable_ids), "mean_ndcg": mean_ndcg,
                "benchmark": BENCHMARK_NDCG, "tolerance": TOLERANCE,
                "per_query_ndcg": per_query_ndcg}


def main(argv: list[str] | None = None) -> int:
    common.load_env()
    corpus_version = common.DEFAULT_CORPUS_VERSION

    parity_result = parity_check.run_parity(corpus_version)
    print(f"parity_check: {'PASS' if parity_result['ok'] else 'FAIL'} "
          f"(min_cosine={parity_result['min_cosine']}, "
          f"missing_in_qdrant={len(parity_result['missing_in_qdrant'])}, "
          f"duplicates={len(parity_result['duplicate_documents'])})")

    queries = _non_graph_queries(corpus_version)
    print(f"queries loaded from {common.DEFAULT_QUERIES}: {len(queries)} (g* excluded)")

    vs = VectorStore()
    reranker = Reranker()
    try:
        if parity_result["ok"]:
            branch = "exact"
            ok, detail = run_exact_branch(queries, vs, reranker, corpus_version)
        else:
            branch = "ndcg_tolerance"
            ok, detail = run_ndcg_branch(queries, vs, reranker, corpus_version)
    finally:
        vs.close()

    print(f"branch run: {branch}")
    print(json.dumps(detail, indent=2, default=str))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
