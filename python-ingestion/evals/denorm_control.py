"""U4 denorm_control: C1 glued texts and C2 payload expansion (spec section 13.2).

C1 (``denorm_texts``) is the shortcut you would ship: cosine, BM25 and the
re-ranker read the full glued message+link text; only the embedding input is
trimmed to the embedder's token budget (D23, R5, R8). Offline: runs under
``HF_HUB_OFFLINE=1``, no Qdrant writes, no network.

C2 (``payload_expand``) is the same walk as U3's ``expand_from_results``, but
over payload entity/topic strings instead of a graph adjacency.

Message-to-link relations are read from the frozen ``graph_neighbors_v1.jsonl``
CONTAINS adjacency (both directions, D21); this module never reads a
``doc["links"]`` field.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from evals import common
from evals.graph_expand import (  # noqa: F401 (re-exported for C2-A1)
    HUB_MAX_DEGREE,
    NAME_NORM,
    NEIGHBOR_CAP,
    QUERY_NODES_K,
    SEEDS_K,
    _path_weight,
    _round9,
    _seed_weight,
)

MAX_SEQ_LENGTH = 256  # all-MiniLM-L6-v2's max_seq_length (D23 note).

_tokenizer = None


def _get_tokenizer():
    """The real embedder's tokenizer, loaded lazily from the local HF cache."""
    global _tokenizer
    if _tokenizer is None:
        import embedder

        model = embedder._get_model()
        _tokenizer = model[0].tokenizer
    return _tokenizer


def _trim_to_tokens(text: str, budget: int, tokenizer: Any) -> str:
    """The longest whole-word prefix of ``text`` whose encoded length fits ``budget``."""
    words = text.split()
    kept: list[str] = []
    for word in words:
        candidate = " ".join(kept + [word])
        if len(tokenizer.encode(candidate)) > budget:
            break
        kept.append(word)
    return " ".join(kept)


def _link_ids_for(cid: str, docs: dict[str, Any], neighbors: dict[str, list[dict[str, Any]]]) -> list[str]:
    """In-corpus links a message CONTAINS, sorted by candidate_id (C1-B2).

    Reads only edges *from* ``cid`` whose target is itself a link, so a
    reverse (link -> message) row never glues the message back onto the link.
    """
    ids = {
        edge["neighbor"]
        for edge in neighbors.get(cid, [])
        if edge["path"] == "contains"
        and edge.get("in_corpus", True)
        and edge["neighbor"] in docs
        and docs[edge["neighbor"]].get("candidate_type") == "link"
    }
    return sorted(ids)


def denorm_texts(
    docs: dict[str, Any],
    neighbors: dict[str, list[dict[str, Any]]],
    tokenizer: Any = None,
    max_tokens: int | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """C1: glue each message to its in-corpus links; links are unchanged.

    Returns ``(texts, meta)`` where ``meta`` is
    ``{"links_trimmed", "links_dropped_no_room"}`` (R8, counted per link: a
    link that keeps some but not all of its words is trimmed; one that keeps
    none is dropped for lack of room; one that fits whole is neither).
    """
    tok = tokenizer if tokenizer is not None else _get_tokenizer()
    budget = max_tokens if max_tokens is not None else MAX_SEQ_LENGTH

    texts: dict[str, str] = {}
    links_trimmed = 0
    links_dropped_no_room = 0

    for cid, doc in docs.items():
        if doc.get("candidate_type") != "message":
            texts[cid] = doc.get("embedded_text") or ""
            continue

        message_text = doc.get("embedded_text") or ""
        parts = [message_text]
        tokens_used = len(tok.encode(message_text))

        for link_id in _link_ids_for(cid, docs, neighbors):
            link_text = docs[link_id].get("embedded_text") or ""
            remaining = budget - tokens_used - 1  # the separator costs 1 token (R8)

            if remaining <= 0:
                links_dropped_no_room += 1
                continue

            full_len = len(tok.encode(link_text))
            if full_len <= remaining:
                parts.append(link_text)
                tokens_used += 1 + full_len
                continue

            trimmed = _trim_to_tokens(link_text, remaining, tok)
            if not trimmed:
                links_dropped_no_room += 1
                continue
            parts.append(trimmed)
            links_trimmed += 1
            tokens_used += 1 + len(tok.encode(trimmed))

        texts[cid] = " ".join(parts)

    return texts, {"links_trimmed": links_trimmed, "links_dropped_no_room": links_dropped_no_room}


def assert_parity(
    texts: dict[str, str],
    stored_vectors: dict[str, list[float]],
    embed_fn,
) -> None:
    """Re-embed ``texts`` and compare to ``stored_vectors``; raise below MIN_PARITY_COSINE."""
    from evals import parity_check

    ids = sorted(set(texts) & set(stored_vectors))
    if not ids:
        return

    fresh = np.asarray(embed_fn([texts[cid] for cid in ids]), dtype=np.float64)
    stored = np.asarray([stored_vectors[cid] for cid in ids], dtype=np.float64)
    cos = parity_check.cosine_rows(fresh, stored)
    min_cos = float(cos.min())

    if min_cos < common.MIN_PARITY_COSINE:
        worst = ids[int(np.argmin(cos))]
        raise ValueError(
            f"C1 parity check failed: min cosine {min_cos:.6f} < "
            f"{common.MIN_PARITY_COSINE} (worst: {worst})")


# ---- C2: payload_expand ----

_PAYLOAD_VIA_KEY = {"payload_entity": 1, "payload_topic": 2}


def _best_payload_path(
    candidates: list[tuple[float, str, str, str]],
) -> tuple[float, str, str, str] | None:
    """Same selection rule as graph_expand._best_path, for payload_* paths."""
    best, best_key = None, None
    for candidate in candidates:
        contrib, path, connector, _seed = candidate
        key = (-contrib, _PAYLOAD_VIA_KEY.get(path, 99), connector or "")
        if best is None or key < best_key:
            best, best_key = candidate, key
    return best


def _normalized_groups(
    payload_entities: dict[str, dict[str, list[str]]], corpus: set[str], field: str,
) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for cid in corpus:
        for raw in payload_entities[cid].get(field, []):
            groups.setdefault(NAME_NORM(raw), set()).add(cid)
    return groups


def payload_expand(
    seeds: list[dict[str, Any]],
    payload_entities: dict[str, dict[str, list[str]]],
    exclude: set[str],
    corpus: set[str],
    k: int = SEEDS_K,
    cap: int = NEIGHBOR_CAP,
    hub_rule: int = HUB_MAX_DEGREE,
) -> list[dict[str, Any]]:
    """C2: the same walk as expand_from_results, over payload entity/topic strings.

    ``degree`` for a normalized string is the number of corpus docs carrying
    it (seed included). Raises ValueError naming any corpus doc missing from
    ``payload_entities``.
    """
    for cid in corpus:
        if cid not in payload_entities:
            raise ValueError(f"{cid} missing from payload_entities")

    seen_seeds: set[str] = set()
    for seed in seeds:
        cid = seed["candidate_id"]
        if cid in seen_seeds:
            raise ValueError(f"duplicate seed id: {cid}")
        seen_seeds.add(cid)

    entity_groups = _normalized_groups(payload_entities, corpus, "entities")
    topic_groups = _normalized_groups(payload_entities, corpus, "topics")

    used_seeds = seeds[:k]
    contributions: dict[str, list[tuple[float, str, str, str]]] = {}

    for seed in used_seeds:
        seed_id = seed["candidate_id"]
        w_seed = _seed_weight(seed["rank"])
        seed_payload = payload_entities.get(seed_id, {})

        for field, path, groups in (
            ("entities", "payload_entity", entity_groups),
            ("topics", "payload_topic", topic_groups),
        ):
            names = {NAME_NORM(n) for n in seed_payload.get(field, [])}
            for name in names:
                group = groups[name]
                degree = len(group)
                w_path = _path_weight("entity" if path == "payload_entity" else "topic",
                                       degree, hub_rule)
                if w_path is None or w_path == 0.0:
                    continue
                contrib = w_seed * w_path
                for neighbor in group:
                    if (neighbor == seed_id or neighbor in seen_seeds
                            or neighbor in exclude or neighbor not in corpus):
                        continue
                    contributions.setdefault(neighbor, []).append(
                        (contrib, path, name, seed_id))

    rows = []
    for cid, contribs in contributions.items():
        score = _round9(math.fsum(sorted(c[0] for c in contribs)))
        _contrib, path, connector, best_seed = _best_payload_path(contribs)
        rows.append({
            "candidate_id": cid,
            "score": float(score),
            "via": {"path": path, "connector": connector, "seed": best_seed},
        })

    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return rows[:cap]
