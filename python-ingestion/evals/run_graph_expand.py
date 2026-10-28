"""U5 run_graph_expand: build the eight Run B systems (spec U5/13.5, section 4).

Offline: reuses fusion.rrf, run_bm25.bm25_rows, run_ladder._rerank_candidates,
rerank.Reranker, graph_expand and denorm_control. No live store access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import embedder
from evals import common, denorm_control, fusion, graph_expand, rerank, run_bm25, run_ladder

RUN_IDS = (
    "ladder_s2_cos_bm25_rr",
    "ctrl_c0_deeper_rr",
    "graph_b1r_rr",
    "graph_b2r_ctx_rr",
    "graph_b1q_rr",
    "graph_b2q_ctx_rr",
    "ctrl_c1_denorm_rr",
    "ctrl_c2_payload_rr",
)

SEEDS_K = graph_expand.SEEDS_K
QUERY_NODES_K = graph_expand.QUERY_NODES_K
NEIGHBOR_CAP = graph_expand.NEIGHBOR_CAP
HUB_MAX_DEGREE = graph_expand.HUB_MAX_DEGREE
RRF_K = fusion.RRF_K
DEPTH = fusion.DEPTH
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
WIN_TIE_THRESHOLD = 0.01

# Pinned independently of rerank.MODEL_REVISION so a drift there is caught (F2).
PINNED_RERANK_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"


def _assert_reranker_pin() -> None:
    if rerank.MODEL_REVISION != PINNED_RERANK_REVISION:
        raise ValueError(
            f"re-ranker revision drifted from the pinned value: "
            f"expected {PINNED_RERANK_REVISION}, got {rerank.MODEL_REVISION!r}")


def _assert_file_sha(path: Path, expected_sha256: str, name: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"{name}: sha256 mismatch at {path} (expected {expected_sha256}, got {actual})")


def _empty_counter() -> dict[str, int]:
    return {"dead_ends": 0, "bq_nodes_used": 0, "links_trimmed": 0, "links_dropped_no_room": 0}


def _row(run_id: str, qid: str, cid: str, rank: int, score: float,
         via: dict[str, Any] | None, note: str | None) -> dict[str, Any]:
    return {"run_id": run_id, "query_id": qid, "candidate_id": cid, "rank": rank,
            "score": score, "via": via, "note": note}


def _with_note(text: str, note: str | None) -> str:
    return text if note is None else f"[connected: {note}] {text}"


def _inject_dead_end_edges(
    adjacency: dict[str, list[dict[str, Any]]], nodes: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """R7: one {neighbor: None, ...} edge per seed, for every degree-1 node
    (from graph_nodes_v1.jsonl) whose corpus_doc_ids contains that seed."""
    augmented = {cid: list(edges) for cid, edges in adjacency.items()}
    for node in nodes:
        if node["degree"] != 1:
            continue
        for seed_id in node["corpus_doc_ids"]:
            augmented.setdefault(seed_id, []).append({
                "neighbor": None, "path": node["node_type"], "connector": node["name"],
                "degree": 1, "in_corpus": True,
            })
    return augmented


def _full_glued_texts(
    docs: dict[str, Any], neighbors: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """C1's full (untrimmed) glued text: message + every in-corpus link's full
    text, read-only (D23: only the embedding input is trimmed, R5)."""
    texts = {}
    for cid, doc in docs.items():
        if doc.get("candidate_type") != "message":
            texts[cid] = doc.get("embedded_text") or ""
            continue
        parts = [doc.get("embedded_text") or ""]
        for link_id in denorm_control._link_ids_for(cid, docs, neighbors):
            parts.append(docs[link_id].get("embedded_text") or "")
        texts[cid] = " ".join(parts)
    return texts


def _rank_by_cosine(query_vec: Any, doc_vecs: dict[str, Any]) -> list[str]:
    qv = np.asarray(query_vec, dtype=np.float64)
    q_norm = np.linalg.norm(qv) or 1.0
    scored = []
    for cid, vec in doc_vecs.items():
        v = np.asarray(vec, dtype=np.float64)
        v_norm = np.linalg.norm(v) or 1.0
        cos = float(np.dot(qv, v) / (q_norm * v_norm))
        scored.append((cos, cid))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [cid for _cos, cid in scored]


def build_s2(
    queries: list[dict[str, Any]],
    docs_by_id: dict[str, Any],
    cosine_rows: list[dict[str, Any]],
    bm25_rows_in: list[dict[str, Any]],
    reranker: Any,
    run_id: str = "ladder_s2_cos_bm25_rr",
) -> list[dict[str, Any]]:
    """rrf(cosine, BM25) top DEPTH -> cross-encoder re-rank (section 4, S2)."""
    rows = []
    for q in queries:
        qid = q["query_id"]
        cosine_rank = fusion.ranking(cosine_rows, qid)
        bm25_rank = fusion.ranking(bm25_rows_in, qid)
        fused = fusion.rrf([cosine_rank, bm25_rank], k=RRF_K, depth=DEPTH)
        candidate_ids = [cid for cid, _score in fused]
        reranked = run_ladder._rerank_candidates(reranker, q["query_text"], candidate_ids, docs_by_id)
        for rank, (cid, score) in enumerate(reranked, start=1):
            rows.append(_row(run_id, qid, cid, rank, score, None, None))
    return rows


def build_systems(
    queries: list[dict[str, Any]],
    docs: dict[str, Any],
    cosine_rows: list[dict[str, Any]],
    bm25_rows_in: list[dict[str, Any]],
    adjacency: dict[str, list[dict[str, Any]]],
    nodes: list[dict[str, Any]],
    node_vecs: Any,
    payload: dict[str, Any],
    reranker: Any,
    *,
    stored_vectors: dict[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, int]]]]:
    """Build the 8 systems in spec section 4 for one query set.

    Returns ``(runs, counters)``: ``runs[run_id]`` is that system's rows;
    ``counters[run_id][query_id]`` is ``{dead_ends, bq_nodes_used,
    links_trimmed, links_dropped_no_room}`` (review #2 I4; not all four apply
    to every system, the rest default to 0).

    When ``stored_vectors`` (U2's frozen, read-only Qdrant vectors) is given,
    the C1 parity check (U4, escalate on failure) runs before anything is
    ranked (review #4 K4): the "unchanged" docs -- links, and messages with
    no in-corpus link, whose C1 text equals their own raw text -- are
    re-embedded and compared against what Qdrant actually stores.
    """
    corpus = set(docs)
    augmented_adjacency = _inject_dead_end_edges(adjacency, nodes)

    # C1 inputs: query-independent, computed once.
    full_glued_texts = _full_glued_texts(docs, adjacency)
    trimmed_texts, denorm_meta = denorm_control.denorm_texts(docs, adjacency)
    sorted_ids = sorted(docs)

    if stored_vectors:
        unchanged_texts = {
            cid: trimmed_texts[cid] for cid in docs
            if trimmed_texts[cid] == (docs[cid].get("embedded_text") or "")
            and cid in stored_vectors
        }
        if unchanged_texts:
            denorm_control.assert_parity(unchanged_texts, stored_vectors, embedder.embed_batch)

    glued_vectors = embedder.embed_batch([trimmed_texts[cid] for cid in sorted_ids])
    glued_vecs_by_id = dict(zip(sorted_ids, glued_vectors))

    runs: dict[str, list[dict[str, Any]]] = {rid: [] for rid in RUN_IDS}
    counters: dict[str, dict[str, dict[str, int]]] = {rid: {} for rid in RUN_IDS}

    for q in queries:
        qid, qtext = q["query_id"], q["query_text"]
        cosine_rank = fusion.ranking(cosine_rows, qid)
        bm25_rank = fusion.ranking(bm25_rows_in, qid)

        # --- S2: rrf(cosine, bm25) top DEPTH -> re-rank -----------------------
        s2_rows = build_s2([q], docs, cosine_rows, bm25_rows_in, reranker)
        runs["ladder_s2_cos_bm25_rr"].extend(s2_rows)
        counters["ladder_s2_cos_bm25_rr"][qid] = _empty_counter()

        s2_ids_ranked = [r["candidate_id"] for r in sorted(s2_rows, key=lambda r: r["rank"])]
        s2_pool = set(s2_ids_ranked)
        seed_ids = s2_ids_ranked[:SEEDS_K]
        seeds = [{"candidate_id": cid, "rank": i + 1} for i, cid in enumerate(seed_ids)]

        # --- B1-r / B2-r --------------------------------------------------------
        expanded_r, stats_r = graph_expand.expand_from_results(
            seeds, augmented_adjacency, s2_pool, corpus)
        via_r = {r["candidate_id"]: r["via"] for r in expanded_r}
        br_pool_ids = s2_ids_ranked + [r["candidate_id"] for r in expanded_r]

        b1r_reranked = reranker.rerank(
            qtext, [(cid, docs[cid]["embedded_text"]) for cid in br_pool_ids])
        for rank, (cid, score) in enumerate(b1r_reranked, start=1):
            runs["graph_b1r_rr"].append(_row("graph_b1r_rr", qid, cid, rank, score,
                                              via_r.get(cid), None))
        counters["graph_b1r_rr"][qid] = {**_empty_counter(), "dead_ends": stats_r["dead_ends"]}

        notes_r = {cid: graph_expand.note_for(cid, seed_ids, augmented_adjacency, docs)
                   for cid in br_pool_ids}
        b2r_reranked = reranker.rerank(
            qtext, [(cid, _with_note(docs[cid]["embedded_text"], notes_r[cid]))
                    for cid in br_pool_ids])
        for rank, (cid, score) in enumerate(b2r_reranked, start=1):
            runs["graph_b2r_ctx_rr"].append(_row("graph_b2r_ctx_rr", qid, cid, rank, score,
                                                  via_r.get(cid), notes_r[cid]))
        counters["graph_b2r_ctx_rr"][qid] = counters["graph_b1r_rr"][qid]

        # --- B1-q / B2-q --------------------------------------------------------
        query_vec = embedder.embed_text(qtext)
        expanded_q, stats_q = graph_expand.expand_from_query(
            query_vec, node_vecs, nodes, s2_pool, corpus)
        via_q = {r["candidate_id"]: r["via"] for r in expanded_q}
        bq_pool_ids = s2_ids_ranked + [r["candidate_id"] for r in expanded_q]

        b1q_reranked = reranker.rerank(
            qtext, [(cid, docs[cid]["embedded_text"]) for cid in bq_pool_ids])
        for rank, (cid, score) in enumerate(b1q_reranked, start=1):
            runs["graph_b1q_rr"].append(_row("graph_b1q_rr", qid, cid, rank, score,
                                              via_q.get(cid), None))
        counters["graph_b1q_rr"][qid] = {**_empty_counter(), "bq_nodes_used": stats_q["bq_nodes_used"]}

        start_nodes = [
            {"path": "query_entity" if node_type == "entity" else "query_topic",
             "connector": name, "score": cos}
            for cos, node_type, name, _node in
            graph_expand.select_start_nodes(query_vec, node_vecs, nodes)
        ]
        notes_q = {cid: graph_expand.note_for_query(cid, start_nodes, nodes)
                   for cid in bq_pool_ids}
        b2q_reranked = reranker.rerank(
            qtext, [(cid, _with_note(docs[cid]["embedded_text"], notes_q[cid]))
                    for cid in bq_pool_ids])
        for rank, (cid, score) in enumerate(b2q_reranked, start=1):
            runs["graph_b2q_ctx_rr"].append(_row("graph_b2q_ctx_rr", qid, cid, rank, score,
                                                  via_q.get(cid), notes_q[cid]))
        counters["graph_b2q_ctx_rr"][qid] = counters["graph_b1q_rr"][qid]

        # --- C0: rrf(cosine, bm25) top N_q -> re-rank ---------------------------
        n_q = max(len(br_pool_ids), len(bq_pool_ids))
        fused_c0 = fusion.rrf([cosine_rank, bm25_rank], k=RRF_K, depth=n_q)
        c0_ids = [cid for cid, _score in fused_c0]
        c0_reranked = reranker.rerank(
            qtext, [(cid, docs[cid]["embedded_text"]) for cid in c0_ids])
        for rank, (cid, score) in enumerate(c0_reranked, start=1):
            runs["ctrl_c0_deeper_rr"].append(_row("ctrl_c0_deeper_rr", qid, cid, rank, score,
                                                   None, None))
        counters["ctrl_c0_deeper_rr"][qid] = _empty_counter()

        # --- C1: rrf(cosine over glued, bm25 over glued) top DEPTH -> re-rank --
        glued_cosine_rank = _rank_by_cosine(query_vec, glued_vecs_by_id)
        glued_docs_list = [{"candidate_id": cid, "embedded_text": full_glued_texts[cid]}
                           for cid in sorted_ids]
        glued_bm25_rows = run_bm25.bm25_rows([q], glued_docs_list)
        glued_bm25_rank = fusion.ranking(glued_bm25_rows, qid)
        fused_c1 = fusion.rrf([glued_cosine_rank, glued_bm25_rank], k=RRF_K, depth=DEPTH)
        c1_ids = [cid for cid, _score in fused_c1]
        c1_reranked = reranker.rerank(
            qtext, [(cid, full_glued_texts[cid]) for cid in c1_ids])
        for rank, (cid, score) in enumerate(c1_reranked, start=1):
            runs["ctrl_c1_denorm_rr"].append(_row("ctrl_c1_denorm_rr", qid, cid, rank, score,
                                                   None, None))
        counters["ctrl_c1_denorm_rr"][qid] = {**_empty_counter(), **denorm_meta}

        # --- C2: S2's 30 U payload entity-overlap neighbors -> re-rank ---------
        payload_rows = denorm_control.payload_expand(seeds, payload, s2_pool, corpus)
        via_c2 = {r["candidate_id"]: r["via"] for r in payload_rows}
        c2_pool_ids = s2_ids_ranked + [r["candidate_id"] for r in payload_rows]
        c2_reranked = reranker.rerank(
            qtext, [(cid, docs[cid]["embedded_text"]) for cid in c2_pool_ids])
        for rank, (cid, score) in enumerate(c2_reranked, start=1):
            runs["ctrl_c2_payload_rr"].append(_row("ctrl_c2_payload_rr", qid, cid, rank, score,
                                                    via_c2.get(cid), None))
        counters["ctrl_c2_payload_rr"][qid] = _empty_counter()

    return runs, counters


# ---- Real orchestration (offline; depends on U2's frozen files) ------------

def _load_stored_vectors(data_dir: Path) -> dict[str, Any]:
    """U2's frozen Qdrant vectors for corpus docs (K4). Required for the C1
    parity check; a missing file raises instead of silently skipping it
    (review #5 P6)."""
    ids_path = data_dir / "runs" / "document_vectors_v1.ids.json"
    vecs_path = data_dir / "runs" / "document_vectors_v1.npy"
    if not ids_path.exists() or not vecs_path.exists():
        raise ValueError(
            f"missing document_vectors_v1.npy/.ids.json under {data_dir / 'runs'}; "
            "re-run the U2 freeze (the C1 parity check needs the stored vectors)")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vecs = np.load(vecs_path)
    return dict(zip(ids, vecs))


def build_from_frozen(
    data_dir: Path,
    expected_sha256: dict[str, str],
    *,
    queries: list[dict[str, Any]] | None = None,
    cosine: list[dict[str, Any]] | None = None,
    suffix: str = "",
    node_vecs: Any = None,
    node_vectors_sha256: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Load the frozen files under ``data_dir``, checked by sha256 before any
    write (F1/F3), build all 8 systems, and write ``runs/<run_id><suffix>.jsonl``
    plus ``.meta.json`` under ``out_dir`` (default: ``data_dir``).

    One code path for both query sets (review #4 K3): ``suffix=""`` for the
    original set, ``suffix=".graphq"`` for the graph set, with ``queries``/
    ``cosine`` the only things that differ between the two calls.

    ``node_vecs``/``node_vectors_sha256`` let a caller (``main``) embed the
    node names once and pass the same vectors to both calls. Left out, the
    node names are embedded fresh here -- never from a cached file on disk,
    which could silently be stale (review #5 P5).
    """
    data_dir = Path(data_dir)
    for name, sha in sorted(expected_sha256.items()):
        _assert_file_sha(data_dir / name, sha, name)

    _assert_reranker_pin()

    documents = common.read_jsonl(data_dir / "documents.jsonl")
    docs = {d["candidate_id"]: d for d in documents}

    neighbor_rows = common.read_jsonl(data_dir / "runs" / "graph_neighbors_v1.jsonl")
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for row in neighbor_rows:
        adjacency.setdefault(row["candidate_id"], []).append({
            "neighbor": row["neighbor_id"], "path": row["path"],
            "connector": row["connector"], "degree": row["connector_degree"],
            "in_corpus": row["in_corpus"]})

    nodes = common.read_jsonl(data_dir / "runs" / "graph_nodes_v1.jsonl")
    payload = json.loads((data_dir / "runs" / "payload_entities_v1.json").read_text(encoding="utf-8"))
    stored_vectors = _load_stored_vectors(data_dir)

    write_dir = out_dir if out_dir is not None else data_dir
    runs_dir = write_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if node_vecs is None:
        node_names = [n["name"] for n in nodes]
        node_vecs = np.asarray(embedder.embed_batch(node_names), dtype=np.float64)
        node_vectors_sha256 = graph_expand.save_node_vectors(
            node_vecs, node_names, runs_dir / "graph_node_vectors_v1.npy")
    else:
        node_vecs = np.asarray(node_vecs, dtype=np.float64)
        if node_vectors_sha256 is None:
            node_vectors_sha256 = graph_expand.save_node_vectors(
                node_vecs, [n["name"] for n in nodes], runs_dir / "graph_node_vectors_v1.npy")

    queries = queries if queries is not None else common.load_queries()
    cosine = cosine if cosine is not None else common.load_run(data_dir.name, "vector_cosine_v1")
    bm25 = run_bm25.bm25_rows(queries, documents)

    runs, counters = build_systems(
        queries, docs, cosine, bm25, adjacency, nodes, node_vecs, payload,
        rerank.Reranker(), stored_vectors=stored_vectors)

    for run_id, rows in runs.items():
        common.write_jsonl(runs_dir / f"{run_id}{suffix}.jsonl", rows)
        meta = {
            "SEEDS_K": SEEDS_K, "QUERY_NODES_K": QUERY_NODES_K, "NEIGHBOR_CAP": NEIGHBOR_CAP,
            "NAME_NORM": "lowercase, strip, [\\s\\-_]+ -> ' ' (D19)",
            "HUB_RULE": "skip connectors with merged degree > HUB_MAX_DEGREE (D20)",
            "HUB_MAX_DEGREE": HUB_MAX_DEGREE, "RRF_K": RRF_K, "DEPTH": DEPTH,
            "rerank_model": rerank.MODEL_NAME, "rerank_revision": rerank.MODEL_REVISION,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "win_tie_threshold": WIN_TIE_THRESHOLD,
            "node_vectors_sha256": node_vectors_sha256,
            "per_query": counters[run_id],
            "created_at": common.utc_now(),
            **common.git_state(),
        }
        (runs_dir / f"{run_id}{suffix}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"runs": runs, "counters": counters}


def main() -> int:
    corpus_version = common.DEFAULT_CORPUS_VERSION
    data_dir = common.data_dir(corpus_version, create=False)

    u2_meta_path = data_dir / "runs" / "graph_neighbors_v1.meta.json"
    u2_meta = json.loads(u2_meta_path.read_text(encoding="utf-8"))
    file_sha256 = u2_meta.get("file_sha256")
    if not file_sha256:
        raise ValueError(
            f"{u2_meta_path.name} has no file_sha256; re-run the U2 freeze "
            "(review #4 K3)")

    # Every frozen input U5 reads (K3b), all from U2's own meta -- including
    # documents.jsonl's hash, taken at freeze time, not re-hashed from the
    # same file a moment before reading it (that can never fail; review #5 P5).
    expected = {
        "documents.jsonl": file_sha256["documents.jsonl"],
        **{f"runs/{name}": sha for name, sha in file_sha256.items() if name != "documents.jsonl"},
    }

    # Embedded once, passed to both calls -- never read back from disk (P5).
    nodes = common.read_jsonl(data_dir / "runs" / "graph_nodes_v1.jsonl")
    node_names = [n["name"] for n in nodes]
    node_vecs = np.asarray(embedder.embed_batch(node_names), dtype=np.float64)
    node_vectors_sha256 = graph_expand.save_node_vectors(
        node_vecs, node_names, data_dir / "runs" / "graph_node_vectors_v1.npy")

    original_queries = common.load_queries()
    original_cosine = common.load_run(corpus_version, "vector_cosine_v1")
    build_from_frozen(data_dir, expected, queries=original_queries, cosine=original_cosine,
                      suffix="", node_vecs=node_vecs, node_vectors_sha256=node_vectors_sha256)

    graphq_queries = common.load_queries(common.REPO_ROOT / "queries_graph.csv")
    graphq_cosine = common.read_jsonl(data_dir / "runs" / "vector_cosine_graphq_v1.jsonl")
    build_from_frozen(data_dir, expected, queries=graphq_queries, cosine=graphq_cosine,
                      suffix=".graphq", node_vecs=node_vecs, node_vectors_sha256=node_vectors_sha256)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
