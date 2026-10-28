"""U1 headroom: count relevant docs outside a run's top N (spec section 13.4).

Offline, pure. Used on the original set (must reproduce 19/104, section 2.3)
and, after U6 grades the graph set, on queries_graph.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evals import common

RELEVANT_GRADE = 2
DEFAULT_DEPTH = 30


def headroom(
    s2_rows: list[dict[str, Any]],
    qrels: dict[str, dict[str, dict[str, Any]]],
    depth: int = DEFAULT_DEPTH,
) -> dict[str, dict[str, Any]]:
    """Per query: relevant (grade >= 2) docs, how many sit outside the top ``depth``.

    ``s2_rows``: rows with ``query_id``, ``candidate_id``, ``rank``.
    ``qrels``: ``{query_id: {candidate_id: {"relevance": grade, ...}}}``.

    Returns ``{query_id: {"relevant", "outside", "outside_ids", "answerable"}}``
    for every query present in ``qrels``. A query is ``answerable`` when it has
    at least one grade >= 2 document.
    """
    ranks_by_query: dict[str, dict[str, int]] = {}
    for row in s2_rows:
        ranks_by_query.setdefault(row["query_id"], {})[row["candidate_id"]] = row["rank"]

    result: dict[str, dict[str, Any]] = {}
    for qid, docs in qrels.items():
        relevant_ids = sorted(
            cid for cid, info in docs.items() if info.get("relevance", 0) >= RELEVANT_GRADE
        )
        ranks = ranks_by_query.get(qid, {})
        outside_ids = [cid for cid in relevant_ids if ranks.get(cid, depth + 1) > depth]
        result[qid] = {
            "relevant": len(relevant_ids),
            "outside": len(outside_ids),
            "outside_ids": outside_ids,
            "answerable": len(relevant_ids) > 0,
        }
    return result


def _print_table(result: dict[str, dict[str, Any]]) -> None:
    total_relevant = sum(v["relevant"] for v in result.values())
    total_outside = sum(v["outside"] for v in result.values())
    for qid in sorted(result):
        entry = result[qid]
        if entry["outside"] > 0:
            print(f"{qid}\trelevant={entry['relevant']}\toutside={entry['outside']}")
    print(f"TOTAL\trelevant={total_relevant}\toutside={total_outside}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="U1 headroom")
    parser.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--run-id", default="ladder_s2_cos_bm25_rr")
    parser.add_argument("--qrels", default=None)
    args = parser.parse_args(argv)

    corpus = common.load_documents(args.corpus_version)
    s2_rows = common.load_run(args.corpus_version, args.run_id)
    qrels_path = Path(args.qrels) if args.qrels else (
        common.data_dir(args.corpus_version) / "qrels_v2.csv"
    )
    qrels = common.load_qrels(
        qrels_path, valid_ids={d["candidate_id"] for d in corpus}
    )

    result = headroom(s2_rows, qrels, depth=args.depth)
    _print_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
