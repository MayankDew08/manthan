import re
from typing import List, Optional

from dotenv import load_dotenv

from store import KnowledgeStore
from vector_store import VectorStore

load_dotenv()

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "for", "and", "or", "of",
    "to", "in", "on", "at", "what", "how", "who", "which", "that", "this",
    "it", "with", "you", "do", "does", "did", "can", "could", "would", "should",
}

_MESSAGE_PROPS = ("sender", "sent_at", "text", "quality", "link_intent",
                  "entities", "topics")

MIN_SCORE = 0.25


def _terms(query: str) -> List[str]:
    raw = re.split(r"[^\w]+", query.lower())
    return [t for t in raw if len(t) > 2 and t not in _STOP]


def _qdrant_pass(payload: dict, topics: Optional[List[str]],
                 min_quality: Optional[int]) -> bool:
    if min_quality is not None and payload.get("type") == "message":
        if (payload.get("quality") or 0) < min_quality:
            return False
    if topics:
        pt = set(payload.get("topics") or [])
        if not pt.intersection(topics):
            return False
    return True


def _message_key(sender, sent_at, text):
    return ("m", sender, sent_at, text)


def _link_key(url):
    return ("l", url)


def _message_result(props: dict, score: float, source: str,
                    related_links: Optional[list] = None) -> dict:
    return {
        "type": "message",
        "sender": props.get("sender", ""),
        "sent_at": props.get("sent_at", ""),
        "text": props.get("text", ""),
        "topics": props.get("topics") or [],
        "entities": props.get("entities") or [],
        "score": round(score, 4),
        "source": source,
        "related_links": related_links or [],
    }


def _link_result(payload: dict, score: float, source: str,
                 attribution: Optional[dict] = None) -> dict:
    attribution = attribution or {}
    return {
        "type": "link",
        "sender": attribution.get("sender", ""),
        "sent_at": attribution.get("sent_at", ""),
        "text": payload.get("summary", ""),
        "title": payload.get("title", ""),
        "url": payload.get("url", ""),
        "topics": payload.get("topics") or [],
        "entities": payload.get("entities") or [],
        "score": round(score, 4),
        "source": source,
    }


def search(query: str, *, topics: Optional[List[str]] = None,
           min_quality: Optional[int] = None, top_k: int = 10,
           vs: Optional[VectorStore] = None,
           store: Optional[KnowledgeStore] = None) -> List[dict]:
    own_vs = vs is None
    own_store = store is None
    if own_vs:
        vs = VectorStore()
    if own_store:
        store = KnowledgeStore()

    try:
        terms = _terms(query)
        seen = set()
        results = []

        hits = vs.search(query, limit=max(top_k * 2, 20))
        for hit in hits:
            p = hit.payload or {}
            if not _qdrant_pass(p, topics, min_quality):
                continue
            if p.get("type") == "message":
                key = _message_key(p.get("sender"), p.get("sent_at"), p.get("text"))
                seen.add(key)
                results.append(_message_result(p, hit.score, "semantic",
                                               related_links=_related_links(store, p)))
            elif p.get("type") == "link":
                key = _link_key(p.get("url"))
                seen.add(key)
                results.append(_link_result(p, hit.score, "semantic",
                                            attribution=_link_attribution(store, p)))
            if len(results) >= top_k:
                break

        best_sem = max((r["score"] for r in results), default=1.0)
        graph_scale = 0.5 * best_sem

        if terms:
            graph_results = _graph_candidates(store, terms, topics, min_quality,
                                              graph_scale)
            for res in graph_results:
                if res["type"] == "message":
                    key = _message_key(res["sender"], res["sent_at"], res["text"])
                else:
                    key = _link_key(res["url"])
                if key not in seen:
                    seen.add(key)
                    results.append(res)

        results.sort(key=lambda r: r["score"], reverse=True)

        best_by_url = {}
        for i, r in enumerate(results):
            urls = ([r["url"]] if r["type"] == "link" and r["url"]
                    else [lk["url"] for lk in r.get("related_links") or []])
            for url in urls:
                cur = best_by_url.get(url)
                if cur is None or r["score"] > cur[0]:
                    best_by_url[url] = (r["score"], i)
        best_idx = {i for _, i in best_by_url.values()}
        drop_idx = set()
        for i, r in enumerate(results):
            if i in best_idx:
                continue
            urls = ([r["url"]] if r["type"] == "link" and r["url"]
                    else [lk["url"] for lk in r.get("related_links") or []])
            if any(url in best_by_url for url in urls):
                drop_idx.add(i)
        results = [r for i, r in enumerate(results)
                   if i not in drop_idx and r["score"] >= MIN_SCORE]
        return results[:top_k]
    finally:
        if own_vs:
            vs.close()
        if own_store:
            store.close()


def _related_links(store: KnowledgeStore, props: dict) -> list:
    with store.driver.session() as session:
        rows = session.run(
            """
            MATCH (m:Message {sender: $sender, sent_at: $sent_at, text: $text})
            OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)
            RETURN l.url AS url, l.title AS title
            """,
            sender=props.get("sender", ""),
            sent_at=props.get("sent_at", ""),
            text=props.get("text", ""),
        ).data()
    return [{"url": r["url"], "title": r["title"] or ""}
            for r in rows if r["url"]]


def _link_attribution(store: KnowledgeStore, payload: dict) -> dict:
    with store.driver.session() as session:
        row = session.run(
            """
            MATCH (m:Message)-[:CONTAINS]->(l:Link {url: $url})
            RETURN m.sender AS sender, m.sent_at AS sent_at
            ORDER BY m.sent_at
            LIMIT 1
            """,
            url=payload.get("url", ""),
        ).single()
    return {"sender": row["sender"], "sent_at": row["sent_at"]} if row else {}


def _graph_candidates(store: KnowledgeStore, terms: List[str],
                      topics: Optional[List[str]], min_quality: Optional[int],
                      scale: float) -> List[dict]:
    results = []
    with store.driver.session() as session:
        for row in session.run(
            """
            MATCH (m:Message)
            WHERE any(t IN $terms WHERE
                toLower(m.text) CONTAINS t
                OR toLower(coalesce(m.link_intent, '')) CONTAINS t)
              AND ($min_quality IS NULL OR m.quality >= $min_quality)
            RETURN DISTINCT m
            """,
            terms=terms, min_quality=min_quality,
        ).data():
            results.append(_message_result(dict(row["m"]), scale, "graph"))
        for row in session.run(
            """
            MATCH (m:Message)-[:MENTIONS]->(e:Entity)
            WHERE any(t IN $terms WHERE toLower(e.name) CONTAINS t)
              AND ($min_quality IS NULL OR m.quality >= $min_quality)
            RETURN DISTINCT m
            """,
            terms=terms, min_quality=min_quality,
        ).data():
            results.append(_message_result(dict(row["m"]), scale, "graph"))
        for row in session.run(
            """
            MATCH (m:Message)-[:ABOUT]->(t:Topic)
            WHERE any(term IN $terms WHERE toLower(t.name) CONTAINS term)
              AND ($min_quality IS NULL OR m.quality >= $min_quality)
            RETURN DISTINCT m
            """,
            terms=terms, min_quality=min_quality,
        ).data():
            results.append(_message_result(dict(row["m"]), scale, "graph"))
        for row in session.run(
            """
            MATCH (l:Link)
            WHERE coalesce(l.status, 'scraped') = 'scraped'
              AND (coalesce(l.summary, '') <> ''
                   OR coalesce(l.what_it_is, '') <> ''
                   OR coalesce(l.problem_solved, '') <> '')
              AND any(t IN $terms WHERE
                  toLower(coalesce(l.title, '')) CONTAINS t
                  OR toLower(coalesce(l.summary, '')) CONTAINS t
                  OR toLower(coalesce(l.what_it_is, '')) CONTAINS t)
            RETURN l
            """,
            terms=terms,
        ).data():
            props = dict(row["l"])
            payload = {
                "summary": props.get("summary", ""),
                "title": props.get("title", ""),
                "url": props.get("url", ""),
                "topics": props.get("topics") or [],
                "entities": props.get("entities") or [],
            }
            if _qdrant_pass(payload, topics, min_quality):
                results.append(_link_result(payload, scale, "graph",
                                            attribution=_link_attribution(store, payload)))
    return results