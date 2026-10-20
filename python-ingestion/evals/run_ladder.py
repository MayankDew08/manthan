"""U9: assemble the five ladder steps from frozen inputs (spec section 13).

Offline: loads the three prerequisite runs (vector_cosine_v1, bm25_v1, graph_lexical_v0)
and the real cross-encoder re-ranker, then writes the five ladder runs and their meta
sidecars. Deterministic — no network, no randomness, no wall-clock time in rankings.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evals import common, fusion, rerank

# Module constants exactly as in spec section 4
STEPS = (
    "ladder_s0_cosine",
    "ladder_s1_cos_bm25",
    "ladder_s2_cos_bm25_rr",
    "ladder_s3_cos_bm25_graph",
    "ladder_s4_cos_bm25_graph_rr",
)

SCORE_SCALE = {
    "ladder_s0_cosine": "cosine",
    "ladder_s1_cos_bm25": "rrf",
    "ladder_s2_cos_bm25_rr": "cross_encoder",
    "ladder_s3_cos_bm25_graph": "rrf",
    "ladder_s4_cos_bm25_graph_rr": "cross_encoder",
}

# Pre-registered constants from spec section 8
RRF_K = 60
DEPTH = 30


def _cosine_ranking_per_query(cosine_rows: list[dict]) -> dict[str, list[str]]:
    """Extract ranking (candidate_id list in rank order) per query from cosine rows."""
    ranking: dict[str, list[str]] = {}
    for row in cosine_rows:
        qid = row["query_id"]
        cid = row["candidate_id"]
        rank = row["rank"]
        if qid not in ranking:
            ranking[qid] = []
        # Ensure list is long enough
        while len(ranking[qid]) < rank - 1:
            ranking[qid].append(None)  # placeholder, shouldn't happen with contiguous ranks
        if len(ranking[qid]) == rank - 1:
            ranking[qid].append(cid)
        else:
            # Should not happen with well-formed input
            ranking[qid][rank - 1] = cid
    # Filter out any None placeholders (shouldn't exist)
    return {qid: [c for c in cids if c is not None] for qid, cids in ranking.items()}


def _bm25_ranking_per_query(bm25_rows: list[dict]) -> dict[str, list[str]]:
    """Extract ranking per query from BM25 rows."""
    return _cosine_ranking_per_query(bm25_rows)  # same format


def _graph_ranking_per_query(graph_rows: list[dict]) -> dict[str, list[str]]:
    """Extract ranking per query from graph rows."""
    return _cosine_ranking_per_query(graph_rows)  # same format


def _rrf_fuse_for_query(
    cosine_ranking: list[str],
    bm25_ranking: list[str],
    graph_ranking: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Fuse rankings for a single query using RRF."""
    rankings = [cosine_ranking, bm25_ranking]
    if graph_ranking is not None:
        rankings.append(graph_ranking)
    return fusion.rrf(rankings, k=RRF_K, depth=DEPTH)


def _rerank_candidates(
    reranker: rerank.Reranker,
    query_text: str,
    candidate_ids: list[str],
    docs_by_id: dict[str, dict],
) -> list[tuple[str, float]]:
    """Re-rank candidates using the cross-encoder, preserving membership."""
    if not candidate_ids:
        return []
    # Read embedded_text in fused order
    candidates = [(cid, docs_by_id[cid]["embedded_text"]) for cid in candidate_ids]
    return reranker.rerank(query_text, candidates)


def build_ladder(
    queries: list[dict],
    docs_by_id: dict[str, dict],
    cosine_rows: list[dict],
    bm25_rows: list[dict],
    graph_rows: list[dict],
    reranker: rerank.Reranker,
) -> dict[str, list[dict]]:
    """Build all five ladder steps (spec section 4, 13).

    Returns a dict mapping step run_id to its rows.
    """
    # Build per-query rankings from input runs
    cosine_rankings = _cosine_ranking_per_query(cosine_rows)
    bm25_rankings = _bm25_ranking_per_query(bm25_rows)
    graph_rankings = _graph_ranking_per_query(graph_rows)

    query_text_by_id = {q["query_id"]: q["query_text"] for q in queries}

    # Step 0: cosine rows with run_id replaced
    s0_rows = []
    for row in cosine_rows:
        new_row = dict(row)
        new_row["run_id"] = STEPS[0]
        s0_rows.append(new_row)

    # Step 1: RRF(cosine, BM25) top 30
    s1_rows = []
    for query in queries:
        qid = query["query_id"]
        cosine_rank = cosine_rankings.get(qid, [])
        bm25_rank = bm25_rankings.get(qid, [])
        fused = _rrf_fuse_for_query(cosine_rank, bm25_rank)
        for rank, (candidate_id, score) in enumerate(fused, start=1):
            s1_rows.append({
                "run_id": STEPS[1],
                "query_id": qid,
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
            })

    # Step 2: s1's 30 candidates → cross-encoder re-rank
    s2_rows = []
    for query in queries:
        qid = query["query_id"]
        query_text = query_text_by_id[qid]
        s1_candidates = [r["candidate_id"] for r in s1_rows if r["query_id"] == qid]
        reranked = _rerank_candidates(reranker, query_text, s1_candidates, docs_by_id)
        for rank, (candidate_id, score) in enumerate(reranked, start=1):
            s2_rows.append({
                "run_id": STEPS[2],
                "query_id": qid,
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
            })

    # Step 3: RRF(cosine, BM25, graph) top 30
    s3_rows = []
    for query in queries:
        qid = query["query_id"]
        cosine_rank = cosine_rankings.get(qid, [])
        bm25_rank = bm25_rankings.get(qid, [])
        graph_rank = graph_rankings.get(qid, [])
        fused = _rrf_fuse_for_query(cosine_rank, bm25_rank, graph_rank)
        for rank, (candidate_id, score) in enumerate(fused, start=1):
            s3_rows.append({
                "run_id": STEPS[3],
                "query_id": qid,
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
            })

    # Step 4: s3's 30 candidates → cross-encoder re-rank
    s4_rows = []
    for query in queries:
        qid = query["query_id"]
        query_text = query_text_by_id[qid]
        s3_candidates = [r["candidate_id"] for r in s3_rows if r["query_id"] == qid]
        reranked = _rerank_candidates(reranker, query_text, s3_candidates, docs_by_id)
        for rank, (candidate_id, score) in enumerate(reranked, start=1):
            s4_rows.append({
                "run_id": STEPS[4],
                "query_id": qid,
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
            })

    return {
        STEPS[0]: s0_rows,
        STEPS[1]: s1_rows,
        STEPS[2]: s2_rows,
        STEPS[3]: s3_rows,
        STEPS[4]: s4_rows,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """Path relative to python-ingestion/, without resolving symlinks."""
    try:
        return str(path.absolute().relative_to(common.ROOT))
    except ValueError:
        return str(path.absolute())


def run_meta(step: str, inputs: dict[str, Path]) -> dict:
    """Sidecar run configuration (spec section 5)."""
    is_rerank_step = step in (STEPS[2], STEPS[4])
    is_fused_step = step in (STEPS[1], STEPS[2], STEPS[3], STEPS[4])
    return {
        "run_id": step,
        "score_scale": SCORE_SCALE[step],
        "rrf_k": RRF_K if is_fused_step else None,
        "depth": DEPTH if is_fused_step else None,
        "rerank_model": rerank.MODEL_NAME if is_rerank_step else None,
        "rerank_revision": rerank.MODEL_REVISION if is_rerank_step else None,
        "inputs": {
            name: {"path": _rel(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "created_at": common.utc_now(),
        **common.git_state(),
    }


def main() -> int:
    corpus_version = common.DEFAULT_CORPUS_VERSION
    data_dir = common.data_dir(corpus_version, create=False)
    runs_dir = data_dir / "runs"

    # Load input runs
    cosine_rows = common.read_jsonl(runs_dir / "vector_cosine_v1.jsonl")
    bm25_rows = common.read_jsonl(runs_dir / "bm25_v1.jsonl")
    graph_rows = common.read_jsonl(runs_dir / "graph_lexical_v0.jsonl")

    # Load queries and documents
    queries = common.load_queries()
    docs = common.read_jsonl(data_dir / "documents.jsonl")
    docs_by_id = {d["candidate_id"]: d for d in docs}

    # Create real Reranker (loads model lazily from local cache)
    rr = rerank.Reranker()

    # Build ladder
    ladder = build_ladder(queries, docs_by_id, cosine_rows, bm25_rows, graph_rows, rr)

    # Write output runs and meta files
    input_paths = {
        "cosine": runs_dir / "vector_cosine_v1.jsonl",
        "bm25": runs_dir / "bm25_v1.jsonl",
        "graph": runs_dir / "graph_lexical_v0.jsonl",
        "documents": data_dir / "documents.jsonl",
        "queries": common.DEFAULT_QUERIES,
    }

    for step in STEPS:
        rows = ladder[step]
        out = runs_dir / f"{step}.jsonl"
        common.write_jsonl(out, rows)

        meta = run_meta(step, input_paths)
        (runs_dir / f"{step}.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        per_query: dict[str, int] = {}
        for row in rows:
            per_query[row["query_id"]] = per_query.get(row["query_id"], 0) + 1
        print(f"Wrote {len(rows)} rows for {len(queries)} queries to {out}")
        if per_query:
            print(f"  Ranks per query: {min(per_query.values())}-{max(per_query.values())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())