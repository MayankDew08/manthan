"""S2 retrieval: cosine + BM25 fused with RRF, then cross-encoder re-rank.

docs/mcp-server-spec.md section 2. S2 lives in one place (M2): the fusion
constants, the BM25 scorer and the re-ranker come from ``evals`` by import,
so this module and the offline eval can never score the ranking differently.
"""

from typing import Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from evals.bm25 import BM25
from evals.common import embedded_text
from evals.fusion import DEPTH, RRF_K, rrf

# Fit on the 25 graded queries (M5). Not re-tuned here.
CONFIDENCE_STRONG_MIN = 2.0

SNIPPET_CHARS = 300


class Corpus:
    """``{point_id: (embedded_text, payload)}``, refreshed on size change.

    One instance lives for the app's lifetime (``app.state.corpus``), next to
    the long-lived ``VectorStore`` and ``Reranker``. ``refresh`` re-scrolls the
    whole collection only when ``client.count()`` differs from the cached
    size -- one cheap call per request against a ~200-doc corpus.
    """

    def __init__(self):
        self._size: Optional[int] = None
        self.ids: list[str] = []
        self.payloads: dict[str, dict] = {}
        self.text_by_id: dict[str, str] = {}
        self.bm25: Optional[BM25] = None

    def refresh(self, vs) -> None:
        live_size = vs.client.count(collection_name=vs.collection, exact=True).count
        if live_size == self._size:
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
        self.ids = ids
        self.payloads = payloads
        self.text_by_id = text_by_id
        self.bm25 = BM25([text_by_id[i] for i in ids])
        self._size = live_size


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
    if qfilter is None:
        hits = vs.search(query, limit=DEPTH)
    else:
        hits = vs.client.search(collection_name=vs.collection, query_vector=vs._embed(query),
                                 limit=DEPTH, query_filter=qfilter)
    return [str(hit.id) for hit in hits]


def _bm25_ranking(corpus: Corpus, query: str, topics: Optional[list],
                   min_quality: Optional[int], kind: str) -> list[str]:
    if topics is None and min_quality is None and kind == "any":
        assert corpus.bm25 is not None  # set by refresh(), always called first
        bm25, ids = corpus.bm25, corpus.ids
    else:
        keep = [i for i in corpus.ids if _passes(corpus.payloads[i], topics, min_quality, kind)]
        bm25 = BM25([corpus.text_by_id[i] for i in keep])
        ids = keep
    return [ids[i] for i, _score in bm25.top(query, DEPTH)]


def _result(point_id: str, payload: dict, score: float) -> dict:
    """One result. Keeps every key ``search.py``'s results carry today (so
    ``query_worker.format_results`` keeps working), and adds ``id`` (M4/M8).
    """
    if payload.get("type") == "link":
        return {
            "id": point_id,
            "type": "link",
            "sent_at": "",
            "text": (payload.get("summary") or "")[:SNIPPET_CHARS],
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
        "text": (payload.get("text") or "")[:SNIPPET_CHARS],
        "topics": payload.get("topics") or [],
        "entities": payload.get("entities") or [],
        "score": score,
        "source": "s2",
        "related_links": [],
    }


def search_s2(query: str, *, top_k: int = 5, kind: str = "any",
              topics: Optional[list] = None, min_quality: Optional[int] = None,
              vs, reranker, corpus: Optional[Corpus] = None) -> dict:
    """Cosine + BM25, fused with RRF, re-ranked by the cross-encoder (spec section 2).

    ``corpus`` is the long-lived cache; callers that omit it (tests) get a
    private, one-call cache. ``vs`` is a ``VectorStore`` and ``reranker`` an
    ``evals.rerank.Reranker``.
    """
    corpus = corpus if corpus is not None else Corpus()
    corpus.refresh(vs)
    assert corpus.bm25 is not None  # set by refresh(), which always runs above

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
    return {"results": results, "confidence": confidence, "top_score": top_score}
