"""U6 topup_b: blind top-up sheets and qrels_v3 (spec U6/13.6).

Reuses ``topup.py`` (``unjudged_pairs``, ``_grade``, ``_key``); never edits
it. Message-to-link relations come from the frozen ``graph_neighbors_v1.jsonl``
CONTAINS adjacency (D21), never a ``doc["links"]`` field.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from evals import common, denorm_control, pooling, topup

TOPUP_ESCALATE_ORIGINAL = 300
TOPUP_ESCALATE_GRAPH = 800

GRAPH_SYSTEMS = (
    "ladder_s2_cos_bm25_rr",
    "ctrl_c0_deeper_rr",
    "graph_b1r_rr",
    "graph_b2r_ctx_rr",
    "graph_b1q_rr",
    "graph_b2q_ctx_rr",
    "ctrl_c1_denorm_rr",
    "ctrl_c2_payload_rr",
)

# Section 13.6: the original-set sheet is unjudged top-5 of the *new* systems
# only -- S2 already contributed its grades via qrels_v2.
ORIGINAL_SET_SYSTEMS = tuple(s for s in GRAPH_SYSTEMS if s != "ladder_s2_cos_bm25_rr")


def graph_content(doc: dict[str, Any], neighbors: dict[str, list[dict[str, Any]]],
                   docs_by_id: dict[str, Any]) -> str:
    """D24: display_content plus, for a message, each in-corpus link's title."""
    base = pooling.display_content(doc)
    if doc.get("candidate_type") != "message":
        return base
    link_ids = denorm_control._link_ids_for(doc["candidate_id"], docs_by_id, neighbors)
    suffix = "".join(
        f' [shared link: "{docs_by_id[lid].get("title") or lid}"]' for lid in link_ids)
    return base + suffix


def build_sheet(
    pairs: list[tuple[str, str]],
    queries_by_id: dict[str, Any],
    docs_by_id: dict[str, Any],
    kind: str,
    neighbors: dict[str, list[dict[str, Any]]] | None = None,
    seed: int = topup.SHUFFLE_SEED,
) -> list[dict[str, Any]]:
    """The blind sheet: common.ANNOTATION_COLUMNS, blank grades, shuffled.

    ``kind="original"`` uses ``pooling.display_content`` (as Run A);
    ``kind="graph"`` uses ``graph_content`` (D24). ``content`` is always the
    doc's own text -- never C1's glued text, never a B2 note.
    """
    order = sorted(set(pairs))
    random.Random(seed).shuffle(order)

    rows = []
    for qid, cid in order:
        doc = docs_by_id[cid]
        if qid not in queries_by_id:
            raise ValueError(f"top-up pair {qid}/{cid}: query {qid} is not in the query set")
        content = (graph_content(doc, neighbors or {}, docs_by_id) if kind == "graph"
                   else pooling.display_content(doc))
        rows.append({
            "query_id": qid,
            "query_text": queries_by_id[qid]["query_text"],
            "candidate_id": cid,
            "candidate_type": doc["candidate_type"],
            "content": content,
            "relevance": "",
            "reason": "",
        })
    return rows


def original_pairs(
    runs: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, dict[str, Any]]],
    k: int = topup.TOPUP_K,
) -> list[tuple[str, str]]:
    """Unjudged top-k pairs in the 7 new systems only (not S2); reuses topup."""
    new_systems = {rid: rows for rid, rows in runs.items() if rid in ORIGINAL_SET_SYSTEMS}
    return topup.unjudged_pairs(new_systems, qrels, k=k)


def _top_k_pairs(rows: list[dict[str, Any]], k: int) -> set[tuple[str, str]]:
    """The k rows with the smallest rank per query (positional, not rank <= k)."""
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    pairs = set()
    for qid, qrows in by_query.items():
        qrows = sorted(qrows, key=lambda r: r["rank"])
        for row in qrows[:k]:
            pairs.add((qid, row["candidate_id"]))
    return pairs


def graph_pairs(
    runs: dict[str, list[dict[str, Any]]],
    cosine_rows: list[dict[str, Any]],
    k: int = 30,
) -> list[tuple[str, str]]:
    """Union of the top k of all 8 systems plus the graph-set cosine run (D4)."""
    pairs: set[tuple[str, str]] = set()
    for rid in GRAPH_SYSTEMS:
        pairs |= _top_k_pairs(runs.get(rid, []), k)
    pairs |= _top_k_pairs(cosine_rows, k)
    return sorted(pairs)


def check_sheet_size(kind: str, count: int) -> None:
    """Escalate (section 10) rather than silently grade an oversized sheet."""
    limit = TOPUP_ESCALATE_ORIGINAL if kind == "original" else TOPUP_ESCALATE_GRAPH
    if count > limit:
        raise ValueError(
            f"{kind} top-up sheet has {count} rows (limit {limit}); escalate per spec section 10")


def _merge_sheet(
    v2_keys: set[tuple[str, str]], rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    seen: set[tuple[str, str]] = set()
    out = []
    for row in rows:
        key = topup._key(row)
        if key in v2_keys or key in seen:
            qid, cid = key
            raise ValueError(f"{qid}/{cid} is already graded; rebuild the sheet against the current qrels")
        grade = topup._grade(row)  # raises ValueError naming the row on a blank/invalid grade
        seen.add(key)
        out.append({
            "query_id": key[0], "candidate_id": key[1], "relevance": grade,
            "answer_bearing": (row.get("answer_bearing") or "0").strip() or "0",
            "reason": (row.get("reason") or "").strip(),
        })
    return out, seen


def _merge_v3(
    v2_rows: list[dict[str, Any]],
    original_sheet_rows: list[dict[str, Any]],
    graph_sheet_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """qrels_v2 u both graded top-up sheets -> qrels_v3 (U6-B7)."""
    v2_keys = {topup._key(row) for row in v2_rows}
    out = list(v2_rows)
    for rows in (original_sheet_rows, graph_sheet_rows):
        graded, seen = _merge_sheet(v2_keys, rows)
        out.extend(graded)
        v2_keys |= seen
    return out


merge_qrels_v3 = _merge_v3


def read_csv(path: Path) -> list[dict[str, Any]]:
    return topup.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    topup.write_csv(path, rows, columns)


# ---- CLI (review #4 K5) -----------------------------------------------------

def _adjacency_from_frozen(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["candidate_id"], []).append({
            "neighbor": row["neighbor_id"], "path": row["path"],
            "connector": row["connector"], "degree": row["connector_degree"],
            "in_corpus": row["in_corpus"]})
    return out


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return common.read_jsonl(path) if path.exists() else []


def cmd_build(data_dir: Path, queries_path: Path, graphq_queries_path: Path) -> int:
    """Both sheets (with check_sheet_size), printing only the two row counts."""
    documents = common.read_jsonl(data_dir / "documents.jsonl")
    docs_by_id = {d["candidate_id"]: d for d in documents}
    queries_by_id = {
        q["query_id"]: q
        for q in common.load_queries(queries_path) + common.load_queries(graphq_queries_path)
    }

    runs_dir = data_dir / "runs"
    original_runs = {rid: _read_jsonl_if_exists(runs_dir / f"{rid}.jsonl") for rid in GRAPH_SYSTEMS}
    qrels_v2_path = data_dir / "qrels_v2.csv"
    qrels_v2 = (common.load_qrels(qrels_v2_path, valid_ids=set(docs_by_id))
                if qrels_v2_path.exists() else {})

    pairs_original = original_pairs(original_runs, qrels_v2)
    check_sheet_size("original", len(pairs_original))
    original_sheet = build_sheet(pairs_original, queries_by_id, docs_by_id, kind="original")
    write_csv(data_dir / "annotation_topup_b.csv", original_sheet, common.ANNOTATION_COLUMNS)

    graphq_runs = {rid: _read_jsonl_if_exists(runs_dir / f"{rid}.graphq.jsonl")
                   for rid in GRAPH_SYSTEMS}
    graphq_cosine = _read_jsonl_if_exists(runs_dir / "vector_cosine_graphq_v1.jsonl")
    neighbors = _adjacency_from_frozen(_read_jsonl_if_exists(runs_dir / "graph_neighbors_v1.jsonl"))

    pairs_graph = graph_pairs(graphq_runs, graphq_cosine)
    check_sheet_size("graph", len(pairs_graph))
    graph_sheet = build_sheet(pairs_graph, queries_by_id, docs_by_id, kind="graph",
                              neighbors=neighbors)
    write_csv(data_dir / "annotation_graphq.csv", graph_sheet, common.ANNOTATION_COLUMNS)

    # Counts only -- never a candidate_id or any doc content (the owner grades blind).
    print(f"original_sheet_rows={len(original_sheet)}")
    print(f"graph_sheet_rows={len(graph_sheet)}")
    return 0


def cmd_merge(data_dir: Path) -> int:
    """qrels_v2 + both graded sheets -> qrels_v3.csv."""
    v2_rows = topup.read_csv(data_dir / "qrels_v2.csv")
    original_graded = topup.read_csv(data_dir / "annotation_topup_b.csv")
    graph_graded = topup.read_csv(data_dir / "annotation_graphq.csv")

    v3_rows = merge_qrels_v3(v2_rows, original_graded, graph_graded)
    write_csv(data_dir / "qrels_v3.csv", v3_rows, topup.QRELS_COLUMNS)

    print(f"qrels_v3_rows={len(v3_rows)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="U6 topup_b")
    parser.add_argument("command", choices=["build", "merge"])
    parser.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--queries", type=Path, default=common.DEFAULT_QUERIES)
    parser.add_argument("--queries-graph", type=Path, default=common.REPO_ROOT / "queries_graph.csv")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else common.data_dir(
        args.corpus_version, create=False)

    if args.command == "build":
        return cmd_build(data_dir, args.queries, args.queries_graph)
    return cmd_merge(data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
