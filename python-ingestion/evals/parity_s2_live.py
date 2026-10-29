"""Phase 1 parity gate for S2 (docs/mcp-server-spec.md section 2, Q9).

1. Run evals.parity_check.
2. Call retrieval.search_s2 in-process for every non-graph query in queries.csv.
3. Compare against the frozen run evals/data/pilot-001/runs/ladder_s2_cos_bm25_rr.jsonl:
   - parity_check PASSED, or its only failure is docs whose own recorded
     qdrant_point_id drifted but whose D25 alias id is live (the same drift
     freeze_neighbors.build_payload_entities already tolerates) -> exact
     branch: top-5 ids must be identical, in the same order, for every
     query. A live id is mapped to the corpus's candidate_id (directly, or
     through the D25 alias map) before comparing.
   - otherwise -> nDCG@5 branch: mean nDCG@5 on qrels_v3.csv, over answerable
     queries only, must be within +/-0.02 of the frozen benchmark 0.7517.
     Ids are mapped through the same by-qdrant-id/alias lookup before
     scoring against qrels.
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
from store import KnowledgeStore  # noqa: E402
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


def _resolve_candidate_id(result: dict, by_qdrant_id: dict[str, str],
                           alias_map: dict[str, str]) -> str:
    """A live result's ``id`` (a Qdrant point id), mapped to the corpus's
    stable candidate_id -- directly, or through the D25 alias map."""
    return (by_qdrant_id.get(result["id"]) or alias_map.get(result["id"])
            or f"{result['type']}:{result['id']}")


def alias_resolves_drift(parity_result: dict, corpus_docs: list[dict], vs) -> bool:
    """True when parity_check's only failure is docs findable by their D25
    alias id: the same point, live under a different id because its own
    recorded qdrant_point_id drifted (re-upsert, hash repair, etc.) -- not a
    real corpus/index divergence. ``build_payload_entities`` already accepts
    this drift when freezing; the live parity gate should too, rather than
    dropping to the nDCG tolerance branch for a non-issue.
    """
    missing = parity_result["missing_in_qdrant"]
    if not missing:
        return False  # nothing to resolve; parity_result would already be ok
    if parity_result["duplicate_documents"] or parity_result["min_cosine"] is None:
        return False
    if parity_result["min_cosine"] < common.MIN_PARITY_COSINE:
        return False

    alias_by_candidate = {cid: pid for pid, cid in build_alias_map(corpus_docs).items()}
    alias_ids = [alias_by_candidate[cid] for cid in missing if cid in alias_by_candidate]
    if len(alias_ids) != len(missing):
        return False  # some missing doc has no alias at all (not just a drifted id)

    live = {str(rec.id) for rec in
            vs.client.retrieve(collection_name=vs.collection, ids=alias_ids, with_vectors=False)}
    return all(alias_id in live for alias_id in alias_ids)


def run_exact_branch(queries: list[dict], vs, reranker, store, by_qdrant_id: dict,
                      alias_map: dict, corpus_version: str) -> tuple[bool, dict]:
    """Top-5 live ids, mapped to corpus candidate_id, must equal the frozen run."""
    frozen = _frozen_top5(corpus_version, FROZEN_RUN)

    live_corpus = retrieval.Corpus()
    mismatches = []
    for q in queries:
        out = retrieval.search_s2(q["query_text"], top_k=5, vs=vs, reranker=reranker,
                                  corpus=live_corpus, store=store)
        live_ids = [_resolve_candidate_id(r, by_qdrant_id, alias_map) for r in out["results"]]
        expected = frozen.get(q["query_id"], [])
        if live_ids != expected:
            mismatches.append({"query_id": q["query_id"], "live": live_ids, "frozen": expected})
    return (not mismatches), {"queries_checked": len(queries), "mismatches": mismatches}


def run_ndcg_branch(queries: list[dict], vs, reranker, store, by_qdrant_id: dict,
                     alias_map: dict, corpus_version: str) -> tuple[bool, dict]:
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
                                  reranker=reranker, corpus=live_corpus, store=store)
        grades = []
        for r in out["results"]:
            cid = _resolve_candidate_id(r, by_qdrant_id, alias_map)
            entry = qrels.get(qid, {}).get(cid)
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
    corpus_docs = common.load_documents(corpus_version)
    by_qdrant_id = {d["qdrant_point_id"]: d["candidate_id"] for d in corpus_docs}
    alias_map = build_alias_map(corpus_docs)

    queries = _non_graph_queries(corpus_version)
    print(f"queries loaded from {common.DEFAULT_QUERIES}: {len(queries)} (g* excluded)")

    vs = VectorStore()
    reranker = Reranker()
    store = KnowledgeStore()
    try:
        alias_resolved = (not parity_result["ok"]
                          and alias_resolves_drift(parity_result, corpus_docs, vs))
        print(f"parity_check: {'PASS' if parity_result['ok'] else 'FAIL'} "
              f"(min_cosine={parity_result['min_cosine']}, "
              f"missing_in_qdrant={len(parity_result['missing_in_qdrant'])}, "
              f"duplicates={len(parity_result['duplicate_documents'])})")
        if alias_resolved:
            print("missing_in_qdrant is alias-resolvable (D25 drift, not a real "
                  "divergence): running the exact branch anyway")

        if parity_result["ok"] or alias_resolved:
            branch = "exact"
            ok, detail = run_exact_branch(queries, vs, reranker, store, by_qdrant_id,
                                          alias_map, corpus_version)
        else:
            branch = "ndcg_tolerance"
            ok, detail = run_ndcg_branch(queries, vs, reranker, store, by_qdrant_id,
                                         alias_map, corpus_version)
    finally:
        vs.close()
        store.close()

    print(f"branch run: {branch}")
    print(json.dumps(detail, indent=2, default=str))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
