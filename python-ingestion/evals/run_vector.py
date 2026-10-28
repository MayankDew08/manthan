"""Stage 3: Run A, the vector-only cosine baseline.

Calls VectorStore.search() directly and ranks the *whole* corpus per query.
It never calls the hybrid search() (no Neo4j, no dedupe, no 0.25 threshold),
so scores are raw cosine similarities.

Usage (from python-ingestion/):
  uv run python -m evals.run_vector [--corpus-version pilot-001] [--run-id vector_cosine_v1]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common  # noqa: E402


def rank_all(queries: list[dict], vs, depth: int, run_id: str) -> list[dict]:
    """One row per (query, candidate), ranks 1..depth, raw cosine scores."""
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        hits = vs.search(q["query_text"], limit=depth)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            rows.append({
                "run_id": run_id,
                "query_id": q["query_id"],
                "candidate_id": f"{payload.get('type')}:{hit.id}",
                "rank": rank,
                "score": float(hit.score),
                "latency_ms": latency_ms,
            })
    return rows


def ranking_signature(rows: list[dict]) -> list[tuple]:
    """What determinism compares: (query_id, candidate_id, rank) in order."""
    return [(r["query_id"], r["candidate_id"], r["rank"]) for r in rows]


def run(corpus_version: str, run_id: str, queries_path: Path, collection: str | None = None):
    common.load_env()
    cfg = common.qdrant_settings()
    collection = collection or cfg["collection"]
    docs = common.load_documents(corpus_version)
    doc_ids = {d["candidate_id"] for d in docs}
    queries = common.load_queries(queries_path)

    from vector_store import VectorStore
    vs = VectorStore(host=cfg["host"], port=cfg["port"], collection=collection)
    try:
        rows = rank_all(queries, vs, depth=len(docs), run_id=run_id)
    finally:
        vs.close()

    per_query = {}
    for r in rows:
        per_query.setdefault(r["query_id"], []).append(r["candidate_id"])
    for qid, ids in per_query.items():
        unknown = set(ids) - doc_ids
        if unknown:
            raise RuntimeError(f"{qid}: {len(unknown)} ranked ids are not in the manifest")
        if len(ids) != len(doc_ids):
            raise RuntimeError(f"{qid}: ranked {len(ids)} of {len(doc_ids)} documents")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    ap.add_argument("--run-id", default=common.DEFAULT_RUN_ID)
    ap.add_argument("--queries", type=Path, default=common.DEFAULT_QUERIES)
    ap.add_argument("--collection", default=None)
    args = ap.parse_args(argv)

    rows = run(args.corpus_version, args.run_id, args.queries, args.collection)
    out = common.data_dir(args.corpus_version) / "runs" / f"{args.run_id}.jsonl"
    common.write_jsonl(out, rows)
    n_q = len({r["query_id"] for r in rows})
    print(f"Wrote {len(rows)} rows ({n_q} queries) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
