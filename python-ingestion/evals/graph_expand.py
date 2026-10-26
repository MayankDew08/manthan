"""U3 graph_expand: offline graph expansion (spec section 5/13.1).

Pure, deterministic functions: no live store access, no random/time/datetime,
no environment reads. All scores are plain Python ``float`` (never numpy
scalars), rounded to 9 decimals once, computed with ``math.fsum`` over a
sorted list of contributions so the result does not depend on edge order
(review #2 I1, I11).

A dead end is an adjacency edge with ``neighbor: None`` (spec R7, review #3
T2): U5 synthesizes one such edge per seed for every degree-1 node in
``graph_nodes_v1.jsonl`` whose ``corpus_doc_ids`` contains that seed. This
module only has to recognise and count them.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

# ---- Module constants (spec section 4) ----
SEEDS_K = 10
QUERY_NODES_K = 5
NEIGHBOR_CAP = 30
HUB_MAX_DEGREE = 16

# NAME_NORM (D19): lowercase, strip, collapse runs of whitespace/'-'/'_' into one space.
_NAME_NORM_RE = re.compile(r"[\s\-_]+")


def NAME_NORM(name: str) -> str:
    return _NAME_NORM_RE.sub(" ", name.strip().lower())


def _round9(x: float) -> float:
    return round(x, 9)


def _seed_weight(rank: int) -> float:
    """w(seed) = 1 / rank."""
    return 1.0 / rank


def _via_key(path: str) -> int:
    """Tie-break order: contains < entity/query_entity < topic/query_topic."""
    return {
        "contains": 0,
        "entity": 1, "query_entity": 1,
        "topic": 2, "query_topic": 2,
    }.get(path, 99)


def _path_weight(path: str, degree: int | None, hub_rule: int = HUB_MAX_DEGREE) -> float | None:
    """Path weight. ``None`` means "hub, skip entirely" (not a dead end).

    ``contains`` = 1.0. ``entity``/``topic`` = ``1/ln(1+degree)``, after the
    hub check (degree > hub_rule -> skip) and the degree-0 check (raise, F2).
    A real degree-1 edge (should not occur under R7; see the module docstring)
    is defensively treated as a dead end by the caller, not scored here.
    """
    if path == "contains":
        return 1.0
    if path in ("entity", "topic"):
        if degree is None or degree <= 0:
            raise ValueError(f"{path} connector degree must be positive, got {degree!r}")
        if degree > hub_rule:
            return None
        return 1.0 / math.log(1 + degree)
    raise ValueError(f"unknown path type: {path!r}")


def _best_path(
    candidates: list[tuple[float, str, str | None, str | None]],
) -> tuple[float, str, str | None, str | None] | None:
    """Pick the highest-contribution candidate; ties: via_key, then connector name.

    Each candidate is ``(contribution, path, connector, seed_id)``. Used for
    both ``via`` (expand_from_*) and the notes (note_for*), so the two always
    agree (review #2 I5).
    """
    best = None
    best_key = None
    for candidate in candidates:
        contrib, path, connector, _seed = candidate
        key = (-contrib, _via_key(path), connector or "")
        if best is None or key < best_key:
            best, best_key = candidate, key
    return best


def expand_from_results(
    seeds: list[dict[str, Any]],
    adjacency: dict[str, list[dict[str, Any]]],
    exclude: set[str],
    corpus: set[str],
    k: int = SEEDS_K,
    cap: int = NEIGHBOR_CAP,
    hub_rule: int = HUB_MAX_DEGREE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Result-start expansion (B-r).

    Returns ``(rows, stats)``: ``rows`` is
    ``[{"candidate_id", "score", "via"}]`` sorted by ``(-score, candidate_id)``;
    ``stats`` is ``{"dead_ends": n}`` (review #2 I4).
    """
    seen_seeds: set[str] = set()
    for seed in seeds:
        cid = seed["candidate_id"]
        if cid in seen_seeds:
            raise ValueError(f"duplicate seed id: {cid}")
        seen_seeds.add(cid)

    used_seeds = seeds[:k]

    contributions: dict[str, list[tuple[float, str, str | None, str]]] = {}
    dead_ends = 0

    for seed in used_seeds:
        seed_id = seed["candidate_id"]
        w_seed = _seed_weight(seed["rank"])

        for edge in adjacency.get(seed_id, []):
            neighbor = edge["neighbor"]
            path = edge["path"]
            connector = edge["connector"]
            degree = edge["degree"]

            if neighbor is None:
                # R7: a synthesized dead-end edge. Counted, nothing to score.
                dead_ends += 1
                continue

            if neighbor in seen_seeds or neighbor in exclude or neighbor not in corpus:
                continue

            w_path = _path_weight(path, degree, hub_rule)
            if w_path is None:
                continue  # hub: skip entirely, not a dead end
            if w_path == 0.0:
                # Defensive: a real degree-1 neighbor (shouldn't occur under R7).
                dead_ends += 1
                continue

            contrib = w_seed * w_path
            contributions.setdefault(neighbor, []).append((contrib, path, connector, seed_id))

    rows = []
    for cid, contribs in contributions.items():
        score = _round9(math.fsum(sorted(c[0] for c in contribs)))
        _contrib, path, connector, best_seed = _best_path(contribs)
        rows.append({
            "candidate_id": cid,
            "score": float(score),
            "via": {"path": path, "connector": connector, "seed": best_seed},
        })

    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return rows[:cap], {"dead_ends": dead_ends}


def select_start_nodes(
    query_vec: list[float] | np.ndarray,
    node_vecs: list[list[float]] | np.ndarray,
    nodes: list[dict[str, Any]],
    k: int = QUERY_NODES_K,
    hub_rule: int = HUB_MAX_DEGREE,
) -> list[tuple[float, str, str, dict[str, Any]]]:
    """B-q's start nodes: cosine desc, ties by (node_type, name), hubs removed first.

    Returns up to ``k`` ``(cosine, node_type, name, node)`` tuples. Exposed so a
    caller building a connection note (``note_for_query``) can use exactly the
    nodes and scores ``expand_from_query`` actually used, instead of
    recomputing the selection and risking it diverging.
    """
    query_vec = np.asarray(query_vec, dtype=np.float64)
    node_vecs = np.asarray(node_vecs, dtype=np.float64)

    if query_vec.ndim != 1 or node_vecs.ndim != 2 or query_vec.shape[0] != node_vecs.shape[1]:
        q_dim = query_vec.shape[-1] if query_vec.ndim else 0
        n_dim = node_vecs.shape[-1] if node_vecs.ndim == 2 else 0
        raise ValueError(f"query vector dimension {q_dim} != node vector dimension {n_dim}")

    cosines = node_vecs @ query_vec  # float64; vectors are pre-normalized by the embedder.

    usable: list[tuple[float, str, str, dict[str, Any]]] = []
    for i, node in enumerate(nodes):
        degree = node["degree"]
        if degree is None or degree <= 0:
            raise ValueError(f"node degree must be positive, got {degree!r}")
        if degree > hub_rule:
            continue
        usable.append((float(cosines[i]), node["node_type"], node["name"], node))

    usable.sort(key=lambda row: (-row[0], row[1], row[2]))
    return usable[:k]


def expand_from_query(
    query_vec: list[float] | np.ndarray,
    node_vecs: list[list[float]] | np.ndarray,
    nodes: list[dict[str, Any]],
    exclude: set[str],
    corpus: set[str],
    k: int = QUERY_NODES_K,
    cap: int = NEIGHBOR_CAP,
    hub_rule: int = HUB_MAX_DEGREE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Query-start expansion (B-q).

    Returns ``(rows, stats)``: ``rows`` as above; ``stats`` is
    ``{"bq_nodes_used": n}`` -- the number of start nodes actually used,
    after hub removal (review #2 I4; D13).
    """
    start_nodes = select_start_nodes(query_vec, node_vecs, nodes, k, hub_rule)
    nodes_used = len(start_nodes)

    contributions: dict[str, list[tuple[float, str, str | None, None]]] = {}
    for cos_sim, node_type, name, node in start_nodes:
        degree = node["degree"]
        if degree == 1:
            continue  # a degree-1 start node reaches nothing new (defensive; see R7)
        w_path = 1.0 / math.log(1 + degree)
        contrib = cos_sim * w_path
        path = "query_entity" if node_type == "entity" else "query_topic"

        for doc_id in node["corpus_doc_ids"]:
            if doc_id in exclude or doc_id not in corpus:
                continue
            contributions.setdefault(doc_id, []).append((contrib, path, name, None))

    rows = []
    for cid, contribs in contributions.items():
        score = _round9(math.fsum(sorted(c[0] for c in contribs)))
        _contrib, path, connector, seed = _best_path(contribs)
        rows.append({
            "candidate_id": cid,
            "score": float(score),
            "via": {"path": path, "connector": connector, "seed": seed},
        })

    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return rows[:cap], {"bq_nodes_used": nodes_used}


def _seed_label(doc: dict[str, Any]) -> str:
    """Link -> its title; message -> the first 80 chars of embedded_text."""
    if doc.get("candidate_type") == "link":
        return doc.get("title") or ""
    return (doc.get("embedded_text") or "")[:80]


def _render_note(path: str, connector: str | None, seed_label: str | None) -> str | None:
    if path == "contains":
        return f'shared in the same message as "{seed_label}"'
    if path == "entity":
        return f'both mention "{connector}", like "{seed_label}"'
    if path == "topic":
        return f'both about "{connector}", like "{seed_label}"'
    if path == "query_entity":
        return f'mentions "{connector}"'
    if path == "query_topic":
        return f'about "{connector}"'
    return None


def note_for(
    doc_id: str,
    starts: list[str],
    adjacency: dict[str, list[dict[str, Any]]],
    docs_by_id: dict[str, dict[str, Any]],
    hub_rule: int = HUB_MAX_DEGREE,
) -> str | None:
    """A connection note for ``doc_id``, for any path from ``starts`` (B2-r/B2-q).

    ``starts`` is rank-ordered (position 0 = rank 1, as ``expand_from_results``
    would receive it), so the same seed weight and the same highest-contribution
    tie-break as ``via`` apply (review #2 I5). A hub connector never yields a
    note (I6); a seed with no doc in ``docs_by_id`` falls back to its id as the
    label.
    """
    contribs: list[tuple[float, str, str | None, str]] = []
    for rank, seed_id in enumerate(starts, start=1):
        w_seed = _seed_weight(rank)
        for edge in adjacency.get(seed_id, []):
            if edge["neighbor"] != doc_id:
                continue
            w_path = _path_weight(edge["path"], edge["degree"], hub_rule)
            if w_path is None or w_path == 0.0:
                continue
            contribs.append((w_seed * w_path, edge["path"], edge["connector"], seed_id))

    best = _best_path(contribs)
    if best is None:
        return None
    _contrib, path, connector, seed_id = best
    seed_doc = docs_by_id.get(seed_id)
    seed_label = _seed_label(seed_doc) if seed_doc is not None else seed_id
    return _render_note(path, connector, seed_label)


def note_for_query(
    doc_id: str,
    starts: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    hub_rule: int = HUB_MAX_DEGREE,
) -> str | None:
    """A connection note for ``doc_id`` against B-q's start nodes.

    ``starts`` is ``[{"path": "query_entity"|"query_topic", "connector": name,
    "score": cosine}]``. Same hub rule and tie-break as ``note_for``/``via``.
    """
    by_key = {(n["node_type"], n["name"]): n for n in nodes}

    contribs: list[tuple[float, str, str | None, None]] = []
    for start in starts:
        path = start["path"]
        connector = start["connector"]
        node_type = "entity" if path == "query_entity" else "topic"
        node = by_key.get((node_type, connector))
        if node is None or doc_id not in node.get("corpus_doc_ids", ()):
            continue
        w_path = _path_weight(node_type, node["degree"], hub_rule)
        if w_path is None or w_path == 0.0:
            continue
        contribs.append((start.get("score", 0.0) * w_path, path, connector, None))

    best = _best_path(contribs)
    if best is None:
        return None
    _contrib, path, connector, _seed = best
    return _render_note(path, connector, None)


def save_node_vectors(node_vecs: np.ndarray, node_names: list[str], path: str | Path) -> str:
    """Write ``node_vecs`` to ``path`` (.npy) and return the sha256 of the file bytes.

    The sha is of the written file, not of the names (review #2 I3): a
    tampered or stale file is caught on load, not on an in-memory recompute.
    """
    if len(node_names) != len(node_vecs):
        raise ValueError(
            f"node_names has {len(node_names)} entries, node_vecs has {len(node_vecs)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(node_vecs, dtype=np.float64))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_node_vectors(path: str | Path, expected_sha256: str) -> np.ndarray:
    """Load node vectors from ``path``, raising ValueError on a sha256 mismatch."""
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"node vector file sha256 mismatch at {path}: expected {expected_sha256}, got {actual}")
    return np.load(path)
