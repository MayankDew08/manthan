"""U4 — Reciprocal Rank Fusion (retrieval-ladder spec section 8).

Pure functions over plain data, standard library only. The module fuses several
ranked candidate lists for the same query into one ranking, and it extracts a
single query's ranking out of a flat run file's rows.

Determinism: the result depends only on the inputs — no clock, no randomness,
no environment value, no I/O.
"""

from collections.abc import Iterable, Mapping, Sequence

# Pre-registered constants (spec section 8). Do not tune after seeing results.
RRF_K = 60
DEPTH = 30


def ranking(rows: Iterable[Mapping[str, object]], query_id: str) -> list[str]:
    """Return the candidate ids of one query, ordered by ascending rank.

    ``rows`` is a flat, arbitrarily ordered sequence of run-row mappings with at
    least ``query_id``, ``candidate_id`` and ``rank`` keys (extra keys ignored).
    Rows belonging to other queries are skipped; an unknown query id yields an
    empty list. Ranks are contiguous per query by the shared data format, so the
    candidate id is only a deterministic tie-break.
    """
    selected = [row for row in rows if row["query_id"] == query_id]
    selected.sort(key=lambda row: (row["rank"], row["candidate_id"]))
    return [row["candidate_id"] for row in selected]


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
    depth: int = DEPTH,
) -> list[tuple[str, float]]:
    """Fuse ranked candidate lists with reciprocal rank fusion.

    Each input list is already in rank order (rank 1 first) and is first cut to
    ``depth`` entries. A candidate repeated inside one list counts once, at its
    best (lowest) rank position. Each surviving entry contributes
    ``1 / (k + rank)`` with a 1-based rank, and the scores of a candidate are
    summed across lists; empty lists contribute nothing.

    Returns ``(candidate_id, score)`` pairs sorted by ``(-score, candidate_id)``
    and cut to ``depth``, so equal scores are ordered by candidate id.
    """
    scores: dict[str, float] = {}
    for candidates in rankings:
        best_rank: dict[str, int] = {}
        for rank, candidate_id in enumerate(candidates[:depth], start=1):
            if candidate_id not in best_rank:
                best_rank[candidate_id] = rank
        for candidate_id, rank in best_rank.items():
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:depth]
