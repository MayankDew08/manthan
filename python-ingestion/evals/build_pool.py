"""Stage 4: build the blind annotation sheet.

Pool per query = cosine top-30 ∪ BM25 top-20 ∪ whole-token keyword hits from the
reviewed evals/pool_keywords.json. Writes, into evals/data/<version>/:
  annotation.csv         7 columns, shuffled (seed 42), no score/rank/source
  pool_provenance.jsonl  which source contributed each candidate (never shown)

Usage (from python-ingestion/):
  uv run python -m evals.build_pool [--corpus-version pilot-001] [--run-id vector_cosine_v1]
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common, pooling  # noqa: E402
from evals.bm25 import BM25  # noqa: E402


def cosine_order(run_rows: list[dict]) -> dict[str, list[str]]:
    by_query: dict[str, list[dict]] = {}
    for r in run_rows:
        by_query.setdefault(r["query_id"], []).append(r)
    return {qid: [r["candidate_id"] for r in sorted(rows, key=lambda r: r["rank"])]
            for qid, rows in by_query.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    ap.add_argument("--run-id", default=common.DEFAULT_RUN_ID)
    ap.add_argument("--queries", type=Path, default=common.DEFAULT_QUERIES)
    ap.add_argument("--keywords", type=Path, default=common.KEYWORDS_FILE)
    args = ap.parse_args(argv)

    queries = common.load_queries(args.queries)
    docs = common.load_documents(args.corpus_version)
    docs_by_id = {d["candidate_id"]: d for d in docs}
    ranked = cosine_order(common.load_run(args.corpus_version, args.run_id))
    kw = common.load_keywords(args.keywords)
    bm25 = BM25([d["embedded_text"] for d in docs])

    pools: dict[str, dict[str, dict]] = {}
    annotation: list[dict] = []
    provenance: list[dict] = []
    rng = random.Random(pooling.SHUFFLE_SEED)
    for q in queries:
        qid = q["query_id"]
        if qid not in ranked:
            print(f"FAIL: {qid} is missing from run {args.run_id}")
            return 1
        pool = pooling.select_pool(q["query_text"], docs, ranked[qid], bm25,
                                   kw["keywords"].get(qid, []))
        pools[qid] = pool
        ordered = pooling.shuffle_pool(list(pool), ranked[qid], rng)
        annotation.extend(pooling.annotation_rows(q, ordered, docs_by_id))
        for cid in sorted(pool):
            provenance.append({"query_id": qid, "candidate_id": cid, **pool[cid]})
        by_source = {"cosine": sum(1 for p in pool.values() if p["cosine_rank"]),
                     "bm25": sum(1 for p in pool.values() if p["bm25_rank"]),
                     "keyword": sum(1 for p in pool.values()
                                    if any(s.startswith("keyword:") for s in p["sources"]))}
        print(f"{qid}: pool={len(pool)} (cosine={by_source['cosine']} "
              f"bm25={by_source['bm25']} keyword={by_source['keyword']})")

    problems = pooling.verify_must_pool(kw["must_pool"], docs, pools)
    if problems:
        print("FAIL: pool coverage")
        for p in problems:
            print("  ", p)
        return 1

    out = common.data_dir(args.corpus_version)
    pooling.write_annotation_csv(out / "annotation.csv", annotation)
    common.write_jsonl(out / "pool_provenance.jsonl", provenance)
    blind_problems = pooling.verify_annotation(out / "annotation.csv", ranked)
    if blind_problems:
        print("FAIL: blindness", blind_problems)
        return 1
    print(f"Wrote {len(annotation)} rows to {out / 'annotation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
