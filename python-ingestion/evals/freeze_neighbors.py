"""U2 freeze_neighbors: freeze the live graph for offline expansion (spec U2/13.3).

Live, read-only. Reads Qdrant and Neo4j only through ``evals.live_guard``
(never a bare client/driver, never the write-capable API key). Checks the
listener is stopped before the first read, fingerprints before and after,
and raises if anything changed. Writes:

- ``runs/graph_neighbors_v1.jsonl``: one row per (corpus doc, neighbor, path).
- ``runs/graph_nodes_v1.jsonl``: one row per merged Entity/Topic node.
- ``runs/payload_entities_v1.json``: ``{candidate_id: {entities, topics}}``.
- ``runs/vector_cosine_graphq_v1.jsonl``: cosine run for queries_graph.csv.
- ``runs/graph_neighbors_v1.meta.json``: counts, fingerprints, drift check.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common, live_guard  # noqa: E402
from evals.freeze_graph import candidate_id_for  # noqa: E402
from evals.graph_expand import NAME_NORM  # noqa: E402

QUERIES_GRAPH = common.REPO_ROOT / "queries_graph.csv"

# D22 (pinned, planning-time graph; spec section 2.3 / 5 U2 point 5).
D22_DOCS_TOTAL = 228
D22_MESSAGES = 156
D22_LINKS = 72
D22_CONTAINS_EDGES = 79
D22_MERGED_ENTITIES = 365
D22_MERGED_TOPICS = 373
D22_CONNECTORS_GT_16 = 4

HUB_MAX_DEGREE = 16  # only for the meta histogram; freezing itself never filters hubs.


def assert_all_mapped(required: set[str], mapped: set[str]) -> None:
    """Every corpus id must be among the live graph's ids; no fuzzy matching."""
    missing = sorted(required - mapped)
    if missing:
        raise ValueError(
            f"corpus docs not mapped in the live graph (no fuzzy matching): {missing}")


def run_freeze(counts: dict[str, int], write_fn: Callable[[], None]) -> None:
    """Gate: ``counts`` must equal D22 exactly, checked before ``write_fn`` runs."""
    expected = {
        "docs_total": D22_DOCS_TOTAL, "messages": D22_MESSAGES, "links": D22_LINKS,
        "contains_edges": D22_CONTAINS_EDGES, "merged_entities": D22_MERGED_ENTITIES,
        "merged_topics": D22_MERGED_TOPICS, "connectors_gt_16": D22_CONNECTORS_GT_16,
    }
    for key, want in expected.items():
        got = counts.get(key)
        if got != want:
            raise ValueError(f"graph_counts drift: {key} = {got!r}, expected {want!r}")
    write_fn()


# ---- Live graph read (Neo4j) -----------------------------------------------

def _message_cid(row: dict[str, Any]) -> str:
    return candidate_id_for({"type": "message", "sent_at": row.get("sent_at") or "",
                             "text": row.get("text") or ""})


def _link_cid(row: dict[str, Any]) -> str:
    return candidate_id_for({"type": "link", "url": row.get("url") or ""})


def read_graph(driver) -> dict[str, Any]:
    """Read every doc, CONTAINS edge and MENTIONS/ABOUT edge from the live graph.

    Returns a dict with ``messages`` (set of cid), ``links`` (set of cid),
    ``contains`` (set of (message_cid, link_cid)), ``entity_raw`` and
    ``topic_raw`` (``{raw_name: set(cid)}``).
    """
    messages = {_message_cid(r) for r in live_guard.neo4j_read(
        driver, "MATCH (m:Message) RETURN m.sent_at AS sent_at, m.text AS text")}
    links = {_link_cid(r) for r in live_guard.neo4j_read(
        driver, "MATCH (l:Link) RETURN l.url AS url")}

    contains = {
        (_message_cid(r), _link_cid(r))
        for r in live_guard.neo4j_read(
            driver,
            "MATCH (m:Message)-[:CONTAINS]->(l:Link) "
            "RETURN m.sent_at AS sent_at, m.text AS text, l.url AS url")
    }

    entity_raw: dict[str, set[str]] = {}
    for r in live_guard.neo4j_read(
        driver,
        "MATCH (m:Message)-[:MENTIONS]->(e:Entity) "
        "RETURN m.sent_at AS sent_at, m.text AS text, e.name AS name",
    ):
        entity_raw.setdefault(r["name"], set()).add(_message_cid(r))
    for r in live_guard.neo4j_read(
        driver, "MATCH (l:Link)-[:MENTIONS]->(e:Entity) RETURN l.url AS url, e.name AS name",
    ):
        entity_raw.setdefault(r["name"], set()).add(_link_cid(r))

    topic_raw: dict[str, set[str]] = {}
    for r in live_guard.neo4j_read(
        driver,
        "MATCH (m:Message)-[:ABOUT]->(t:Topic) "
        "RETURN m.sent_at AS sent_at, m.text AS text, t.name AS name",
    ):
        topic_raw.setdefault(r["name"], set()).add(_message_cid(r))
    for r in live_guard.neo4j_read(
        driver, "MATCH (l:Link)-[:ABOUT]->(t:Topic) RETURN l.url AS url, t.name AS name",
    ):
        topic_raw.setdefault(r["name"], set()).add(_link_cid(r))

    return {
        "messages": messages, "links": links, "contains": contains,
        "entity_raw": entity_raw, "topic_raw": topic_raw,
    }


def merge_by_name_norm(raw_groups: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    """Merge raw connector names by NAME_NORM (D19).

    Returns ``{normalized_name: {"raw_names": [...], "docs": set(cid)}}``.
    """
    merged: dict[str, dict[str, Any]] = {}
    for raw_name, docs in raw_groups.items():
        norm = NAME_NORM(raw_name)
        entry = merged.setdefault(norm, {"raw_names": set(), "docs": set()})
        entry["raw_names"].add(raw_name)
        entry["docs"] |= docs
    return merged


def graph_counts_from(graph: dict[str, Any], merged_entities: dict, merged_topics: dict) -> dict[str, int]:
    return {
        "docs_total": len(graph["messages"]) + len(graph["links"]),
        "messages": len(graph["messages"]),
        "links": len(graph["links"]),
        "contains_edges": len(graph["contains"]),
        "merged_entities": len(merged_entities),
        "merged_topics": len(merged_topics),
        "connectors_gt_16": sum(
            1 for group in (*merged_entities.values(), *merged_topics.values())
            if len(group["docs"]) > HUB_MAX_DEGREE),
    }


def build_neighbor_rows(
    corpus_ids: set[str],
    graph: dict[str, Any],
    merged_entities: dict[str, dict[str, Any]],
    merged_topics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per (corpus doc, neighbor, path); CONTAINS both directions (D21)."""
    rows: list[dict[str, Any]] = []

    for msg_cid, link_cid in graph["contains"]:
        if msg_cid in corpus_ids:
            rows.append({
                "candidate_id": msg_cid, "neighbor_id": link_cid, "path": "contains",
                "connector": None, "connector_degree": None,
                "in_corpus": link_cid in corpus_ids,
            })
        if link_cid in corpus_ids:
            rows.append({
                "candidate_id": link_cid, "neighbor_id": msg_cid, "path": "contains",
                "connector": None, "connector_degree": None,
                "in_corpus": msg_cid in corpus_ids,
            })

    for path, merged in (("entity", merged_entities), ("topic", merged_topics)):
        for name, entry in merged.items():
            docs = entry["docs"]
            degree = len(docs)
            for doc in docs:
                if doc not in corpus_ids:
                    continue
                for neighbor in docs:
                    if neighbor == doc:
                        continue
                    rows.append({
                        "candidate_id": doc, "neighbor_id": neighbor, "path": path,
                        "connector": name, "connector_degree": degree,
                        "in_corpus": neighbor in corpus_ids,
                    })

    return rows


def build_node_rows(
    corpus_ids: set[str],
    merged_entities: dict[str, dict[str, Any]],
    merged_topics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per merged node; degree is the full-graph count, corpus_doc_ids is the subset."""
    rows = []
    for node_type, merged in (("entity", merged_entities), ("topic", merged_topics)):
        for name, entry in merged.items():
            rows.append({
                "node_type": node_type,
                "name": name,
                "raw_names": sorted(entry["raw_names"]),
                "degree": len(entry["docs"]),
                "corpus_doc_ids": sorted(d for d in entry["docs"] if d in corpus_ids),
            })
    return rows


# ---- Live Qdrant read -------------------------------------------------------

def _alias_point_id(doc: dict[str, Any]) -> str:
    """D25: the Qdrant point id this doc's own (sent_at, embedded_text) hashes
    to right now -- the id its current text actually lives under, if its own
    recorded ``qdrant_point_id`` has drifted."""
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"msg-{doc.get('sent_at') or ''}-{doc.get('embedded_text') or ''}"))


def build_alias_map(corpus_docs: list[dict[str, Any]]) -> dict[str, str]:
    """``{alias_point_id: corpus_candidate_id}`` for every message doc (D25).

    Only messages: the formula is ``candidate_id_for``'s message branch; a
    link's id is url-based and unrelated to this drift.
    """
    return {
        _alias_point_id(d): d["candidate_id"]
        for d in corpus_docs if d.get("candidate_type") == "message"
    }


def build_payload_entities(
    client, collection: str, corpus_docs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, str]]:
    """Scroll the collection once; keep only corpus points, keyed by candidate_id.

    Returns ``(payload_entities, stored_vectors, qdrant_id_aliases)``:
    ``stored_vectors`` (review #4 K4) lets U5's C1 parity check compare a
    re-embedded unchanged doc against what Qdrant actually stores, offline,
    without a second live scroll; ``qdrant_id_aliases`` is
    ``{corpus_id: live_point_id}`` for every doc found only by its D25 alias.

    A corpus doc's candidate_id never changes (D25). If its own
    ``qdrant_point_id`` isn't live, it may still be found by the EXACT
    uuid5 of its own (sent_at, embedded_text) -- never any other fuzzy
    match -- and is then keyed by its corpus candidate_id, same as always.
    A doc matching neither still raises: the manifest and the live store
    have drifted in some other way; escalate, don't silently write a short
    file (spec section 10).
    """
    by_point_id = {d["qdrant_point_id"]: d["candidate_id"] for d in corpus_docs}
    alias_by_point_id = build_alias_map(corpus_docs)

    payload_out: dict[str, Any] = {}
    vectors_out: dict[str, list[float]] = {}
    aliases_used: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset, with_payload=True,
            with_vectors=True)
        for point in points:
            point_id = str(point.id)
            cid = by_point_id.get(point_id)
            if cid is None:
                alias_cid = alias_by_point_id.get(point_id)
                if alias_cid is not None and alias_cid not in payload_out:
                    cid = alias_cid
                    aliases_used[cid] = point_id
            if cid is None:
                continue
            payload = point.payload or {}
            payload_out[cid] = {
                "entities": payload.get("entities") or [],
                "topics": payload.get("topics") or [],
            }
            if point.vector is not None:
                vectors_out[cid] = list(point.vector)
        if offset is None:
            break

    missing = sorted(set(by_point_id.values()) - set(payload_out))
    if missing:
        raise ValueError(
            f"{len(missing)} corpus doc(s) not found in the live Qdrant collection "
            f"by their own qdrant_point_id or an exact D25 alias (manifest/live-store "
            f"drift): {missing}")
    return payload_out, vectors_out, aliases_used


def build_graphq_cosine_rows(
    queries: list[dict[str, Any]], vs, corpus_ids: set[str], run_id: str,
    alias_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """All 188 corpus ranks per graph query (closed corpus: outside docs never ranked).

    ``alias_map`` (``{live_point_id: corpus_candidate_id}``, from
    ``build_alias_map``) translates a hit on a D25-aliased live id back to
    its stable corpus candidate_id.
    """
    alias_map = alias_map or {}
    rows = []
    for q in queries:
        hits = vs.search(q["query_text"], limit=len(corpus_ids) * 2 + 16)
        resolved = []
        for hit in hits:
            live_cid = f"{(hit.payload or {}).get('type')}:{hit.id}"
            if live_cid in corpus_ids:
                resolved.append((live_cid, hit))
                continue
            aliased_cid = alias_map.get(str(hit.id))
            if aliased_cid in corpus_ids:
                resolved.append((aliased_cid, hit))
        if len(resolved) < len(corpus_ids):
            raise ValueError(
                f"{q['query_id']}: only {len(resolved)} of {len(corpus_ids)} "
                "corpus docs ranked; widen the search limit")
        for rank, (cid, hit) in enumerate(resolved[:len(corpus_ids)], start=1):
            rows.append({
                "run_id": run_id, "query_id": q["query_id"],
                "candidate_id": cid,
                "rank": rank, "score": float(hit.score),
            })
    return rows


# ---- Orchestration -----------------------------------------------------------

def _run_cli(out_dir: Path | None = None, corpus_version: str = common.DEFAULT_CORPUS_VERSION) -> dict:
    live_guard.assert_listener_stopped()

    common.load_env()
    cfg = common.qdrant_settings()
    qdrant_client = live_guard.qdrant_read_only_client()
    store = live_guard.read_only_store()
    driver = store.driver

    try:
        fp_before = live_guard.fingerprint(qdrant_client, cfg["collection"], driver)

        corpus_docs = common.load_documents(corpus_version)
        corpus_ids = {d["candidate_id"] for d in corpus_docs}

        graph = read_graph(driver)
        assert_all_mapped(corpus_ids, graph["messages"] | graph["links"])

        merged_entities = merge_by_name_norm(graph["entity_raw"])
        merged_topics = merge_by_name_norm(graph["topic_raw"])
        counts = graph_counts_from(graph, merged_entities, merged_topics)

        neighbor_rows = build_neighbor_rows(corpus_ids, graph, merged_entities, merged_topics)
        node_rows = build_node_rows(corpus_ids, merged_entities, merged_topics)

        vs = live_guard.read_only_vector_store()
        try:
            payload_entities, stored_vectors, qdrant_id_aliases = build_payload_entities(
                vs.client, cfg["collection"], corpus_docs)
            alias_map = build_alias_map(corpus_docs)
            queries_graph = common.load_queries(QUERIES_GRAPH)
            graphq_rows = build_graphq_cosine_rows(
                queries_graph, vs, corpus_ids, "vector_cosine_graphq_v1", alias_map=alias_map)
        finally:
            vs.close()

        fp_after = live_guard.fingerprint(qdrant_client, cfg["collection"], driver)
        live_guard.assert_unchanged(fp_before, fp_after)

        outside_ids = {r["neighbor_id"] for r in neighbor_rows if not r["in_corpus"]}
        meta = {
            "fingerprint_before": fp_before,
            "fingerprint_after": fp_after,
            "unmapped_nodes": 0,
            "out_of_corpus_neighbors": len(outside_ids),
            "degree_histogram": _degree_histogram(merged_entities, merged_topics),
            "graph_counts": counts,
            "qdrant_id_aliases": qdrant_id_aliases,
            "created_at": common.utc_now(),
            **common.git_state(),
        }

        def write() -> None:
            runs_dir = (out_dir / "runs") if out_dir is not None else (
                common.data_dir(corpus_version) / "runs")
            runs_dir.mkdir(parents=True, exist_ok=True)
            common.write_jsonl(runs_dir / "graph_neighbors_v1.jsonl", neighbor_rows)
            common.write_jsonl(runs_dir / "graph_nodes_v1.jsonl", node_rows)
            (runs_dir / "payload_entities_v1.json").write_text(
                json.dumps(payload_entities, ensure_ascii=False, indent=2), encoding="utf-8")
            common.write_jsonl(runs_dir / "vector_cosine_graphq_v1.jsonl", graphq_rows)
            # K4: the stored Qdrant vectors for corpus docs, so U5's C1 parity
            # check can run offline against what Qdrant actually holds.
            ordered_ids = sorted(stored_vectors)
            (runs_dir / "document_vectors_v1.ids.json").write_text(
                json.dumps(ordered_ids, ensure_ascii=False, indent=2), encoding="utf-8")
            np.save(runs_dir / "document_vectors_v1.npy",
                    np.asarray([stored_vectors[cid] for cid in ordered_ids], dtype=np.float64))

            # K3a: the sha256 of every file this freeze writes, for U5 to check
            # before it builds anything (K3b), for either query set (K3c).
            # documents.jsonl isn't written by this freeze, but U5 checks its
            # hash too (review #5 P5): recording it here means that check is
            # against a hash taken at freeze time, not against the same file
            # a moment before reading it (which could never fail).
            file_sha256 = {}
            for name in ("graph_neighbors_v1.jsonl", "graph_nodes_v1.jsonl",
                         "payload_entities_v1.json", "vector_cosine_graphq_v1.jsonl",
                         "document_vectors_v1.npy", "document_vectors_v1.ids.json"):
                file_sha256[name] = hashlib.sha256((runs_dir / name).read_bytes()).hexdigest()
            documents_path = (out_dir if out_dir is not None else
                              common.data_dir(corpus_version)) / "documents.jsonl"
            file_sha256["documents.jsonl"] = hashlib.sha256(documents_path.read_bytes()).hexdigest()
            meta["file_sha256"] = file_sha256

            (runs_dir / "graph_neighbors_v1.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        run_freeze(counts, write)
        return meta
    finally:
        store.close()


def _degree_histogram(merged_entities: dict, merged_topics: dict) -> dict[str, int]:
    hist: dict[str, int] = {}
    for group in (*merged_entities.values(), *merged_topics.values()):
        key = str(len(group["docs"]))
        hist[key] = hist.get(key, 0) + 1
    return hist


if __name__ == "__main__":
    _run_cli()
