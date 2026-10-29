"""FastAPI surface for search, ingestion, and Telegram worker callbacks."""

import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

import embedder
import enhancer
import ingest
import pipeline
import retrieval
from evals.rerank import Reranker
from link_extract import parse_github_url
from link_ingest import mark_resolved
from store import KnowledgeStore, _url_to_title
from vector_store import VectorStore

load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_REQUESTS = Counter("manthan_search_requests_total", "search requests", ["engine"])
SEARCH_LATENCY = Histogram("manthan_search_seconds", "search latency", ["engine"])
INGEST_JOBS = Counter("manthan_ingest_jobs_total", "ingest background jobs", ["outcome"])
PASTE_REQUESTS = Counter("manthan_paste_requests_total", "paste-link requests", ["outcome"])
SKIP_REQUESTS = Counter("manthan_skip_requests_total", "skip-link requests", ["outcome"])
MESSAGES_QUERIES = Counter("manthan_messages_queries_total", "messages list queries")
PENDING_QUERIES = Counter("manthan_pending_queries_total", "pending-links queries")

JOBS: dict = {}
JOBS_LOCK = threading.Lock()


def _new_job(chat_path: str) -> str:
    """Register an in-memory background ingestion job and return its ID."""
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "chat_path": chat_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "message": None,
            "summary": None,
            "error": None,
        }
    return job_id


def _set_job(job_id: str, **kw) -> None:
    """Apply partial status updates under the shared job lock."""
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    topics: Optional[List[str]] = None
    min_quality: Optional[int] = Field(default=None, ge=0, le=10)


class IngestRequest(BaseModel):
    chat_path: str
    force: bool = False
    no_scrape: bool = False
    min_quality: int = Field(default=4, ge=0, le=10)


class IngestMessageRequest(BaseModel):
    job_id: Optional[str] = None
    telegram_update_id: Optional[int] = None
    type: str = "ingest"
    chat_id: Optional[int] = None
    text: str
    telegram_message_id: Optional[int] = None
    skip_grading: bool = False
    received_at: Optional[str] = None


class PasteRequest(BaseModel):
    url: str = ""
    content: str
    link_id: Optional[str] = None


class SkipRequest(BaseModel):
    link_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open long-lived Neo4j/Qdrant clients for the API process."""
    app.state.store = KnowledgeStore()
    app.state.vs = VectorStore()
    app.state.vs.ensure_collection()
    app.state.reranker = Reranker()
    app.state.corpus = retrieval.Corpus()
    yield
    app.state.vs.close()
    app.state.store.close()
    embedder.unload()


app = FastAPI(title="Manthan API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def do_search(req: SearchRequest):
    """Run S2 (cosine + BM25 + RRF + re-rank) with optional topic/quality filters."""
    t0 = time.monotonic()
    try:
        out = retrieval.search_s2(req.query, top_k=req.top_k, topics=req.topics,
                                  min_quality=req.min_quality, vs=app.state.vs,
                                  reranker=app.state.reranker, corpus=app.state.corpus,
                                  store=app.state.store)
    except Exception:  # pragma: no cover
        SEARCH_REQUESTS.labels(engine="s2").inc()
        logger.exception("Search request failed")
        raise HTTPException(status_code=500, detail="search failed")
    SEARCH_REQUESTS.labels(engine="s2").inc()
    SEARCH_LATENCY.labels(engine="s2").observe(time.monotonic() - t0)
    results = out["results"]
    return {"query": req.query, "count": len(results), "results": results,
            "confidence": out["confidence"], "top_score": out["top_score"]}


def _run_ingest_job(job_id: str, chat_path: str, force: bool,
                    no_scrape: bool, min_quality: int) -> None:
    """Execute a full file ingestion outside the request thread."""
    try:
        summary = pipeline.run_full_ingest(chat_path, force=force,
                                           no_scrape=no_scrape,
                                           min_quality=min_quality)
        _set_job(job_id, status="done", finished_at=datetime.now(timezone.utc).isoformat(),
                 summary=summary,
                 message=f"ingested {summary['messages']} messages, "
                         f"{summary['scraped_links']} links")
        INGEST_JOBS.labels(outcome="success").inc()
    except Exception:
        logger.exception("Ingestion job %s failed", job_id)
        _set_job(job_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                 error="ingestion failed")
        INGEST_JOBS.labels(outcome="error").inc()


@app.post("/ingest", status_code=202)
def start_ingest(req: IngestRequest, bg: BackgroundTasks):
    """Queue a chat export for background ingestion."""
    if not os.path.exists(req.chat_path):
        raise HTTPException(status_code=404,
                            detail=f"chat file not found: {req.chat_path}")
    job_id = _new_job(req.chat_path)
    bg.add_task(_run_ingest_job, job_id, req.chat_path, req.force,
                req.no_scrape, req.min_quality)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Return the latest status for an in-memory ingestion job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")
    return {"job_id": job_id, **job}


@app.post("/ingest-message")
def ingest_message(req: IngestMessageRequest):
    """Run the checkpointed pipeline for one message and persist the results."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    sent_at = req.received_at or datetime.now(timezone.utc).isoformat()
    try:
        result = ingest.ingest_message(
            text,
            sent_at=sent_at,
            trusted=bool(req.skip_grading),
            store=app.state.store,
            vs=app.state.vs,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail="text is required")
    except Exception:
        logger.exception("Message ingestion failed (checkpoint retained for resume)")
        raise HTTPException(status_code=502, detail="ingestion failed")
    INGEST_JOBS.labels(outcome="success").inc()
    return result


@app.get("/links/pending")
def pending_links():
    """List links awaiting manual paste plus links blocked during scraping."""
    PENDING_QUERIES.inc()
    rows = app.state.store.pending_links()
    return {"count": len(rows), "links": rows}


def _existing_title(url: str) -> str:
    """Reuse an existing title when a pending link is completed later."""
    with app.state.store.driver.session() as session:
        row = session.run(
            "MATCH (l:Link {url: $url}) RETURN l.title AS title",
            url=url,
        ).single()
    return (row["title"] or "") if row else ""


@app.post("/links/paste")
def paste_link(req: PasteRequest):
    """Summarize pasted content for a blocked link and persist the result."""
    url = req.url.strip()
    if not url and req.link_id:
        found = app.state.store.link_by_id(req.link_id.strip())
        if found is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown link id: {req.link_id}")
        url = found["url"]
    content = req.content.strip()
    if not url or not content:
        raise HTTPException(status_code=422, detail="url and content are required")
    PASTE_REQUESTS.labels(outcome="attempt").inc()
    existing = _existing_title(url)
    try:
        ref = parse_github_url(url)
        if ref is not None and ref.kind == "repo":
            summary = enhancer.summarize_repo(url, existing, content)
        else:
            summary = enhancer.summarize_content(url, existing, content)
    except Exception:
        PASTE_REQUESTS.labels(outcome="error").inc()
        logger.exception("Pasted-link summarization failed")
        raise HTTPException(status_code=502, detail="summarization failed")

    readable = (summary.get("what_it_is") or summary.get("summary")
                or _url_to_title(url))
    title = (existing or readable)[:120]
    link = {
        "url": url,
        "final_url": url,
        "title": title,
        "source": "paste",
        "raw_text": content[:20000],
        "summary": summary.get("summary", ""),
        "what_it_is": summary.get("what_it_is", ""),
        "problem_solved": summary.get("problem_solved", ""),
        "how_useful": summary.get("how_useful", ""),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "entities": summary.get("entities", []),
        "topics": summary.get("topics", []),
    }
    app.state.store.add_link(link, status="scraped")
    app.state.vs.upsert_link(link)
    try:
        resolved = mark_resolved(url)
    except Exception:
        resolved = 0
        logger.exception("Failed to sync ask_user_links.json for %s", url)
    PASTE_REQUESTS.labels(outcome="success").inc()
    return {
        "ok": True,
        "url": url,
        "title": title,
        "summary": summary.get("summary", ""),
        "topics": summary.get("topics", []),
        "entities": summary.get("entities", []),
        "resolved_records": resolved,
    }


@app.post("/links/skip")
def skip_link(req: SkipRequest):
    """Dismiss a pending or blocked link without ingesting it."""
    SKIP_REQUESTS.labels(outcome="attempt").inc()
    found = app.state.store.skip_link(req.link_id.strip())
    if found is None:
        SKIP_REQUESTS.labels(outcome="error").inc()
        existing = app.state.store.link_by_id(req.link_id.strip())
        if existing is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown link id: {req.link_id}")
        raise HTTPException(status_code=409,
                            detail=f"link already {existing['status']}: {req.link_id}")
    SKIP_REQUESTS.labels(outcome="success").inc()
    return {"ok": True, "url": found["url"], "status": "skipped"}


def _query_messages(*, topic: Optional[str],
                    entity: Optional[str], since: Optional[str],
                    min_quality: Optional[int], limit: int) -> list:
    """Read message records from Neo4j with lightweight graph filters."""
    where = []
    params = {}
    if since:
        where.append("m.sent_at >= $since")
        params["since"] = since
    if min_quality is not None:
        where.append("m.quality >= $min_quality")
        params["min_quality"] = min_quality
    if topic:
        where.append("EXISTS((m)-[:ABOUT]->(:Topic {name: $topic}))")
        params["topic"] = topic
    if entity:
        where.append("EXISTS((m)-[:MENTIONS]->(:Entity {name: $entity}))")
        params["entity"] = entity
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params["limit"] = limit
    query = f"""
    MATCH (m:Message)
    {clause}
    OPTIONAL MATCH (m)-[:ABOUT]->(t:Topic)
    OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)
    RETURN m,
           collect(DISTINCT t.name) AS topics,
           collect(DISTINCT e.name) AS entities,
           collect(DISTINCT {{url: l.url, title: l.title}}) AS links
    ORDER BY m.sent_at DESC
    LIMIT $limit
    """
    with app.state.store.driver.session() as session:
        data = session.run(query, **params).data()
    out = []
    for row in data:
        m = dict(row["m"])
        out.append({
            "type": "message",
            "sent_at": m.get("sent_at"),
            "text": m.get("text"),
            "quality": m.get("quality"),
            "link_intent": m.get("link_intent"),
            "topics": [x for x in row["topics"] if x],
            "entities": [x for x in row["entities"] if x],
            "links": [x for x in row["links"] if x.get("url")],
        })
    return out


@app.get("/messages")
def list_messages(topic: Optional[str] = None,
                  entity: Optional[str] = None, since: Optional[str] = None,
                  min_quality: Optional[int] = Query(default=None, ge=0, le=10),
                  limit: int = Query(default=50, ge=1, le=500)):
    """Expose stored messages for debugging and local inspection."""
    MESSAGES_QUERIES.inc()
    rows = _query_messages(topic=topic, entity=entity,
                           since=since, min_quality=min_quality, limit=limit)
    return {"count": len(rows), "messages": rows}


def _message_id(sent_at: Optional[str], text: Optional[str]) -> str:
    """M8: the id vector_store.upsert_message gives this message in Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"msg-{sent_at or ''}-{text or ''}"))


def _link_id(url: Optional[str]) -> str:
    """M8: the id vector_store.upsert_link gives this link in Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"link-{url or ''}"))


_TOPIC_KEY_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def topic_key(name: str) -> str:
    """Fold topic spelling variants to one key.

    Topics are free-text from the grader, so one idea gets several spellings
    ("agentic-workflows", "Agentic Workflow"). Case, punctuation/whitespace,
    and a simple trailing plural don't matter; different words still do --
    "Agentic AI" stays its own key, distinct from "agentic-workflows".
    """
    collapsed = _TOPIC_KEY_NON_ALNUM.sub(" ", (name or "").lower()).strip()
    words = []
    for word in collapsed.split(" "):
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


SNIPPET_CHARS = 300


def _snippet(text: str) -> dict:
    """Cut to SNIPPET_CHARS, flagging the cut; a cut snippet ends with "…"."""
    text = text or ""
    if len(text) <= SNIPPET_CHARS:
        return {"text": text, "truncated": False}
    return {"text": text[:SNIPPET_CHARS - 1].rstrip() + "…", "truncated": True}


@app.get("/items/{item_id}")
def get_item(item_id: str):
    """One message or link's full record. The id IS the Qdrant point id
    (M8): retrieve it directly instead of scanning Neo4j, then one Neo4j
    read for its topics and (message: related_links; link: earliest
    attributing sent_at).
    """
    points = app.state.vs.client.retrieve(
        collection_name=app.state.vs.collection, ids=[item_id], with_payload=True)
    if not points:
        raise HTTPException(status_code=404, detail=f"unknown item id: {item_id}")
    payload = points[0].payload or {}

    with app.state.store.driver.session() as session:
        if payload.get("type") == "link":
            row = session.run(
                """
                MATCH (l:Link {url: $url})
                OPTIONAL MATCH (l)-[:ABOUT]->(t:Topic)
                OPTIONAL MATCH (m:Message)-[:CONTAINS]->(l)
                RETURN collect(DISTINCT t.name) AS topics,
                       min(m.sent_at) AS earliest_sent_at
                """,
                url=payload.get("url", ""),
            ).single()
            return {
                "id": item_id,
                "type": "link",
                "sent_at": (row["earliest_sent_at"] if row else None) or "",
                "text": payload.get("summary") or "",
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "topics": [t for t in (row["topics"] if row else []) if t],
            }

        row = session.run(
            """
            MATCH (m:Message {sent_at: $sent_at, text: $text})
            OPTIONAL MATCH (m)-[:ABOUT]->(t:Topic)
            OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)
            RETURN collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT {url: l.url, title: l.title}) AS links
            """,
            sent_at=payload.get("sent_at", ""), text=payload.get("text", ""),
        ).single()
        return {
            "id": item_id,
            "type": "message",
            "sent_at": payload.get("sent_at", ""),
            "text": payload.get("text") or "",
            "topics": [t for t in (row["topics"] if row else []) if t],
            "related_links": [lk for lk in (row["links"] if row else []) if lk.get("url")],
        }


def _topic_name_items(session) -> dict:
    """{topic_name: {item_id, ...}}, Messages and Links ABOUT it.

    Two plain queries, not one combining them: an empty side dropping the
    other's rows is exactly the retrieval.py graph-enrich bug (Phase 1).
    One full scan of the topic edges -- every endpoint below (listing,
    browsing, neighbors, suggestions) folds spelling variants in Python
    from this same map, with no further Neo4j calls.
    """
    message_rows = session.run(
        "MATCH (m:Message)-[:ABOUT]->(t:Topic) "
        "RETURN t.name AS name, m.sent_at AS sent_at, m.text AS text"
    ).data()
    link_rows = session.run(
        "MATCH (l:Link)-[:ABOUT]->(t:Topic) RETURN t.name AS name, l.url AS url"
    ).data()
    by_name: dict = {}
    for row in message_rows:
        by_name.setdefault(row["name"], set()).add(_message_id(row["sent_at"], row["text"]))
    for row in link_rows:
        by_name.setdefault(row["name"], set()).add(_link_id(row["url"]))
    return by_name


def _topic_groups(by_name: dict) -> dict:
    """``{topic_key: {"names": {original_name: {item_id,...}}, "items": {item_id,...}}}``."""
    groups: dict = {}
    for name, item_ids in by_name.items():
        g = groups.setdefault(topic_key(name), {"names": {}, "items": set()})
        g["names"][name] = item_ids
        g["items"] |= item_ids
    return groups


def _canonical_name(names: dict) -> str:
    """The spelling tagged on the most items; ties broken alphabetically."""
    return min(names.keys(), key=lambda n: (-len(names[n]), n))


@app.get("/topics")
def list_topics():
    """[{name, count, variants}], one row per spelling-variant group (M1 §3
    review fix), sorted by count (Messages + Links ABOUT any variant,
    counted once per item) descending."""
    with app.state.store.driver.session() as session:
        groups = _topic_groups(_topic_name_items(session))
    topics = [{"name": _canonical_name(g["names"]), "count": len(g["items"]),
              "variants": sorted(g["names"])} for g in groups.values()]
    topics.sort(key=lambda t: (-t["count"], t["name"]))
    return {"topics": topics}


def _items_for_topic_names(session, names: set) -> list:
    """Every message/link ABOUT any of ``names``, deduped by id, newest first."""
    names_list = list(names)
    message_rows = session.run(
        "MATCH (m:Message)-[:ABOUT]->(t:Topic) WHERE t.name IN $names "
        "RETURN DISTINCT m.sent_at AS sent_at, m.text AS text",
        names=names_list,
    ).data()
    link_rows = session.run(
        "MATCH (l:Link)-[:ABOUT]->(t:Topic) WHERE t.name IN $names "
        "RETURN DISTINCT l.url AS url, l.title AS title, l.summary AS summary, l.sent_at AS sent_at",
        names=names_list,
    ).data()
    by_id: dict = {}
    for row in message_rows:
        item_id = _message_id(row["sent_at"], row["text"])
        by_id[item_id] = {"id": item_id, "type": "message",
                          "sent_at": row["sent_at"] or "", **_snippet(row["text"])}
    for row in link_rows:
        item_id = _link_id(row["url"])
        by_id[item_id] = {"id": item_id, "type": "link", "sent_at": row.get("sent_at") or "",
                          "url": row.get("url") or "", "title": row.get("title") or "",
                          **_snippet(row.get("summary") or "")}
    items = list(by_id.values())
    items.sort(key=lambda it: it["sent_at"], reverse=True)
    return items


def _topic_neighbors(groups: dict, key: str) -> list:
    """Top 5 topic keys co-occurring on the same items as ``key``, with counts."""
    this_items = groups[key]["items"]
    scored = [(len(g["items"] & this_items), _canonical_name(g["names"]))
             for other_key, g in groups.items()
             if other_key != key and g["items"] & this_items]
    scored.sort(key=lambda sc: (-sc[0], sc[1]))
    return [{"name": name, "count": count} for count, name in scored[:5]]


def _topic_suggestions(groups: dict, name: str) -> list:
    """5 closest topic keys by substring match, for a 404 -- one canonical
    name per matching group, compared on keys so plural/case/punctuation
    variants of a near-miss still surface."""
    needle = topic_key(name)
    matches = [(key, _canonical_name(g["names"])) for key, g in groups.items()
              if needle in key or key in needle]
    matches.sort(key=lambda kc: (abs(len(kc[0]) - len(needle)), kc[1]))
    return [canonical for _key, canonical in matches[:5]]


@app.get("/topics/{name:path}")
def browse_topic(name: str, limit: int = Query(default=50, ge=1, le=500)):
    """Every message/link ABOUT any spelling variant of ``name``, newest
    first, plus total and neighbors. ``:path`` so a slash in the topic
    (e.g. "AI/ML") round-trips through the MCP's percent-encoding."""
    key = topic_key(name)
    with app.state.store.driver.session() as session:
        groups = _topic_groups(_topic_name_items(session))
        group = groups.get(key)
        if group is None:
            suggestions = _topic_suggestions(groups, name)
            detail = f"unknown topic: {name}"
            if suggestions:
                detail += f". Did you mean: {', '.join(suggestions)}?"
            raise HTTPException(status_code=404, detail=detail)
        items = _items_for_topic_names(session, set(group["names"]))
    neighbors = _topic_neighbors(groups, key)
    return {"topic": name, "total": len(items), "items": items[:limit],
            "neighbors": neighbors}


@app.get("/metrics")
def prometheus_metrics():
    """Expose Prometheus metrics for the local deployment."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
