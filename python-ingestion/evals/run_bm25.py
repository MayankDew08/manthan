"""U3: BM25 run over the frozen corpus (spec section 7).

Offline: no Qdrant, no model, no network. The in-repo ``evals.bm25.BM25`` scores the
whole corpus for every query and every document with a positive score is ranked,
so the run is a standalone lexical baseline rather than a component of a fusion.

Usage (from python-ingestion/):
  uv run python -m evals.run_bm25
"""

import hashlib
import json
from pathlib import Path

from evals import common
from evals.bm25 import BM25

RUN_ID = "bm25_v1"
SCORE_SCALE = "bm25"


def bm25_rows(queries: list[dict], docs: list[dict]) -> list[dict]:
    """Rank every document with a positive BM25 score for every query.

    One index is built over the corpus ``embedded_text``. Only documents whose
    score is greater than zero are ranked; ties break on ``candidate_id`` so the
    output never depends on corpus order. Rows follow the input query order.
    """
    index = BM25([d["embedded_text"] for d in docs])
    rows: list[dict] = []
    for query in queries:
        scored = [(d["candidate_id"], score) for d, score in
                  zip(docs, index.scores(query["query_text"])) if score > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        for rank, (candidate_id, score) in enumerate(scored, start=1):
            rows.append({
                "run_id": RUN_ID,
                "query_id": query["query_id"],
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
            })
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """Path relative to python-ingestion/, without resolving symlinks."""
    try:
        return str(path.absolute().relative_to(common.ROOT))
    except ValueError:
        return str(path.absolute())


def run_meta(inputs: dict[str, Path]) -> dict:
    """Sidecar run configuration (spec section 5); a header line would break readers."""
    return {
        "run_id": RUN_ID,
        "score_scale": SCORE_SCALE,
        "rrf_k": None,
        "depth": None,
        "rerank_model": None,
        "rerank_revision": None,
        "inputs": {name: {"path": _rel(path), "sha256": _sha256(path)}
                   for name, path in inputs.items()},
        "created_at": common.utc_now(),
        **common.git_state(),
    }


def main() -> int:
    corpus_version = common.DEFAULT_CORPUS_VERSION
    docs_path = common.data_dir(corpus_version, create=False) / "documents.jsonl"
    docs = common.read_jsonl(docs_path)
    queries = common.load_queries()

    rows = bm25_rows(queries, docs)

    runs_dir = common.data_dir(corpus_version) / "runs"
    out = runs_dir / f"{RUN_ID}.jsonl"
    common.write_jsonl(out, rows)
    meta = run_meta({"documents": docs_path, "queries": common.DEFAULT_QUERIES})
    (runs_dir / f"{RUN_ID}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    per_query: dict[str, int] = {}
    for row in rows:
        per_query[row["query_id"]] = per_query.get(row["query_id"], 0) + 1
    print(f"Wrote {len(rows)} rows for {len(queries)} queries to {out}")
    print(f"Ranked documents per query: {min(per_query.values(), default=0)}-"
          f"{max(per_query.values(), default=0)} of {len(docs)}; "
          f"corpus {corpus_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())