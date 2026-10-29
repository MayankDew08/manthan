"""S2 retrieval: cosine + BM25 fused with RRF, then cross-encoder re-rank.

docs/mcp-server-spec.md section 2. S2 lives in one place (M2): the fusion
constants, the BM25 scorer and the re-ranker come from ``evals`` by import,
so this module and the offline eval can never score the ranking differently.
"""

from dataclasses import dataclass
from typing import Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from evals.bm25 import BM25
from evals.common import embedded_text
from evals.fusion import DEPTH, RRF_K, rrf

# Fit on the 25 graded queries (M5). Not re-tuned here.
CONFIDENCE_STRONG_MIN = 2.0

# Not applied in this module (/search returns full text). Reserved for
# Phase 2's MCP layer, which truncates for the model's context budget.
SNIPPET_CHARS = 300


@dataclass(frozen=True)
class _Snapshot:
    """One immutable view of the corpus. ``Corpus.refresh`` builds a new one
    in locals and swaps it in with a single assignment, so a reader never
    sees ``ids``/``payloads``/``bm25`` from two different scrolls mixed
    together -- it sees the old snapshot whole, or the new one whole."""

    size: int
    ids: list[str]
    payloads: dict[str, dict]
    text_by_id: dict[str, str]
    bm25: BM25


_EMPTY = _Snapshot(size=-1, ids=[], payloads={}, text_by_id={}, bm25=BM25([]))


class Corpus:
    """``{point_id: (embedded_text, payload)}``, refreshed on size change.

    One instance lives for the app's lifetime (``app.state.corpus``), next to
    the long-lived ``VectorStore`` and ``Reranker``. ``refresh`` re-scrolls the
    whole collection only when ``client.count()`` differs from the cached
    size -- one cheap call per request against a ~200-doc corpus.
    """

    def __init__(self):
        self._snapshot: _Snapshot = _EMPTY

    @property
    def ids(self) -> list[str]:
        return self._snapshot.ids

    @property
    def payloads(self) -> dict[str, dict]:
        return self._snapshot.payloads

    @property
    def text_by_id(self) -> dict[str, str]:
        return self._snapshot.text_by_id

    @property
    def bm25(self) -> BM25:
        return self._snapshot.bm25

    @bm25.setter
    def bm25(self, value: BM25) -> None:
        """Tests substitute a deterministic fake ranker. Still one swap: the
        rest of the snapshot (ids/payloads/text_by_id/size) is unchanged."""
        s = self._snapshot
        self._snapshot = _Snapshot(s.size, s.ids, s.payloads, s.text_by_id, value)

    def refresh(self, vs) -> None:
        live_size = vs.client.count(collection_name=vs.collection, exact=True).count
        if live_size == self._snapshot.size:
            return
        ids: list[str] = []
        payloads: dict[str, dict] = {}
        text_by_id: dict[str, str] = {}
        offset = None
        while True:
            points, offset = vs.client.scroll(
                collection_name=vs.collection, limit=256, offset=offset,
                with_payload=True, with_vectors=False)
            for point in points:
                point_id = str(point.id)
                payload = point.payload or {}
                ids.append(point_id)
                payloads[point_id] = payload
                text_by_id[point_id] = embedded_text(payload)
            if offset is None:
                break
        bm25 = BM25([text_by_id[i] for i in ids])
        self._snapshot = _Snapshot(live_size, ids, payloads, text_by_id, bm25)


def _passes(payload: dict, topics: Optional[list], min_quality: Optional[int],
            kind: str) -> bool:
    """The M6 pre-pool filter: a candidate failing this never reaches RRF.

    Mirrors ``search.py``'s current filter semantics: ``min_quality`` only
    constrains messages (links carry no quality score), ``topics`` requires
    overlap with the candidate's topics, and ``kind="introduction"`` (Phase 3)
    requires the matching payload category.
    """
    if min_quality is not None and payload.get("type") == "message":
        if (payload.get("quality") or 0) < min_quality:
            return False
    if topics and not set(payload.get("topics") or []).intersection(topics):
        return False
    if kind == "introduction" and payload.get("category") != "introduction":
        return False
    return True


def _qdrant_filter(topics: Optional[list], min_quality: Optional[int],
                    kind: str) -> Optional[Filter]:
    """The same predicate as a Qdrant filter, for the cosine query itself."""
    must = []
    if topics:
        must.append(FieldCondition(key="topics", match=MatchAny(any=list(topics))))
    if min_quality is not None:
        must.append(Filter(should=[
            FieldCondition(key="type", match=MatchValue(value="link")),
            FieldCondition(key="quality", range=Range(gte=min_quality)),
        ]))
    if kind == "introduction":
        must.append(FieldCondition(key="category", match=MatchValue(value="introduction")))
    return Filter(must=must) if must else None


def _cosine_ranking(vs, query: str, qfilter: Optional[Filter]) -> list[str]:
    hits = vs.search(query, limit=DEPTH, query_filter=qfilter)
    return [str(hit.id) for hit in hits]


def _bm25_ranking(corpus: Corpus, query: str, topics: Optional[list],
                   min_quality: Optional[int], kind: str) -> list[str]:
    if topics is None and min_quality is None and kind == "any":
        bm25, ids = corpus.bm25, corpus.ids
    else:
        keep = [i for i in corpus.ids if _passes(corpus.payloads[i], topics, min_quality, kind)]
        bm25 = BM25([corpus.text_by_id[i] for i in keep])
        ids = keep
    return [ids[i] for i, _score in bm25.top(query, DEPTH)]


def _result(point_id: str, payload: dict, score: float) -> dict:
    """One result. Keeps every key ``search.py``'s results carry today (so
    ``query_worker.format_results`` keeps working), and adds ``id`` (M4/M8).
    ``related_links``/``sent_at`` start empty; ``_graph_enrich`` fills them
    in for the final top-k only.
    """
    if payload.get("type") == "link":
        return {
            "id": point_id,
            "type": "link",
            "sent_at": "",
            "text": payload.get("summary") or "",
            "title": payload.get("title", ""),
            "url": payload.get("url", ""),
            "topics": payload.get("topics") or [],
            "entities": payload.get("entities") or [],
            "score": score,
            "source": "s2",
        }
    return {
        "id": point_id,
        "type": "message",
        "sent_at": payload.get("sent_at", ""),
        "text": payload.get("text") or "",
        "topics": payload.get("topics") or [],
        "entities": payload.get("entities") or [],
        "score": score,
        "source": "s2",
        "related_links": [],
    }


# One read-only round trip for the whole top-k batch: a message's related
# links (its own CONTAINS edges) and a link's attribution (the earliest
# message that CONTAINS it) -- the batched equivalent of search.py's
# per-result `_related_links` / `_link_attribution`.
_GRAPH_ENRICH_QUERY = """
CALL {
    UNWIND $messages AS msg
    MATCH (m:Message {sent_at: msg.sent_at, text: msg.text})
    OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)
    WITH msg, l
    WHERE l IS NOT NULL
    RETURN msg.sent_at AS sent_at, msg.text AS text, l.url AS url, l.title AS title
}
WITH collect({sent_at: sent_at, text: text, url: url, title: title}) AS message_links
CALL {
    UNWIND $urls AS url
    MATCH (m2:Message)-[:CONTAINS]->(l2:Link {url: url})
    RETURN url AS link_url, min(m2.sent_at) AS earliest_sent_at
}
RETURN message_links, collect({url: link_url, sent_at: earliest_sent_at}) AS link_attribution
"""


def _graph_enrich(store, results: list[dict]) -> None:
    """Fill ``related_links`` (messages) and ``sent_at`` (links) on the final
    results in place, with one Neo4j query regardless of how many of each."""
    messages = [{"sent_at": r["sent_at"], "text": r["text"]}
                for r in results if r["type"] == "message"]
    urls = [r["url"] for r in results if r["type"] == "link" and r.get("url")]
    if not messages and not urls:
        return

    with store.driver.session() as session:
        record = session.run(_GRAPH_ENRICH_QUERY, messages=messages, urls=urls).single()

    links_by_message: dict[tuple, list] = {}
    for row in (record["message_links"] if record else []):
        key = (row["sent_at"], row["text"])
        links_by_message.setdefault(key, []).append(
            {"url": row["url"], "title": row.get("title") or ""})
    earliest_by_url = {row["url"]: row["sent_at"]
                       for row in (record["link_attribution"] if record else [])
                       if row.get("sent_at")}

    for r in results:
        if r["type"] == "message":
            r["related_links"] = links_by_message.get((r["sent_at"], r["text"]), [])
        elif r["type"] == "link":
            r["sent_at"] = earliest_by_url.get(r["url"], "")


def search_s2(query: str, *, top_k: int = 5, kind: str = "any",
              topics: Optional[list] = None, min_quality: Optional[int] = None,
              vs, reranker, store, corpus: Optional[Corpus] = None) -> dict:
    """Cosine + BM25, fused with RRF, re-ranked by the cross-encoder (spec section 2).

    ``corpus`` is the long-lived cache; callers that omit it (tests) get a
    private, one-call cache. ``vs`` is a ``VectorStore``, ``reranker`` an
    ``evals.rerank.Reranker``, ``store`` a ``store.KnowledgeStore`` used only
    to enrich the final top-k (never the candidate pool).
    """
    corpus = corpus if corpus is not None else Corpus()
    corpus.refresh(vs)

    qfilter = _qdrant_filter(topics, min_quality, kind)
    cosine_ids = _cosine_ranking(vs, query, qfilter)
    bm25_ids = _bm25_ranking(corpus, query, topics, min_quality, kind)

    fused = rrf([cosine_ids, bm25_ids], k=RRF_K, depth=DEPTH)
    candidates = [(cid, corpus.text_by_id.get(cid, "")) for cid, _score in fused]
    reranked = reranker.rerank(query, candidates)

    top = reranked[:top_k]
    if not top:
        return {"results": [], "confidence": "none", "top_score": None}

    top_score = top[0][1]
    confidence = "strong" if top_score >= CONFIDENCE_STRONG_MIN else "weak"
    results = [_result(cid, corpus.payloads.get(cid, {}), score) for cid, score in top]
    _graph_enrich(store, results)
    return {"results": results, "confidence": confidence, "top_score": top_score}
