"""FastAPI surface for search, ingestion, and Telegram worker callbacks."""

import logging
import os
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


@app.get("/items/{item_id}")
def get_item(item_id: str):
    """One message or link's full record, found among all items by its M8 id."""
    with app.state.store.driver.session() as session:
        messages = session.run(
            """
            MATCH (m:Message)
            OPTIONAL MATCH (m)-[:ABOUT]->(t:Topic)
            OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)
            RETURN m.sent_at AS sent_at, m.text AS text,
                   collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT {url: l.url, title: l.title}) AS links
            """
        ).data()
        for row in messages:
            if _message_id(row["sent_at"], row["text"]) == item_id:
                return {
                    "id": item_id,
                    "type": "message",
                    "sent_at": row["sent_at"] or "",
                    "text": row["text"] or "",
                    "topics": [t for t in row["topics"] if t],
                    "related_links": [lk for lk in row["links"] if lk.get("url")],
                }

        links = session.run(
            """
            MATCH (l:Link)
            OPTIONAL MATCH (l)-[:ABOUT]->(t:Topic)
            RETURN l.url AS url, l.title AS title, l.summary AS summary,
                   l.sent_at AS sent_at, collect(DISTINCT t.name) AS topics
            """
        ).data()
        for row in links:
            if _link_id(row["url"]) == item_id:
                return {
                    "id": item_id,
                    "type": "link",
                    "sent_at": row.get("sent_at") or "",
                    "text": row.get("summary") or "",
                    "url": row.get("url") or "",
                    "title": row.get("title") or "",
                    "topics": [t for t in row["topics"] if t],
                }
    raise HTTPException(status_code=404, detail=f"unknown item id: {item_id}")


def _topic_counts(session) -> dict:
    """Topic name -> Messages + Links ABOUT it.

    Two plain queries, not one combining them: an empty side dropping the
    other's rows is exactly the retrieval.py graph-enrich bug (Phase 1).
    """
    message_rows = session.run(
        "MATCH (m:Message)-[:ABOUT]->(t:Topic) RETURN t.name AS name, count(m) AS n"
    ).data()
    link_rows = session.run(
        "MATCH (l:Link)-[:ABOUT]->(t:Topic) RETURN t.name AS name, count(l) AS n"
    ).data()
    counts: dict = {}
    for row in message_rows + link_rows:
        counts[row["name"]] = counts.get(row["name"], 0) + row["n"]
    return counts


@app.get("/topics")
def list_topics():
    """[{name, count}], count = Messages + Links ABOUT the topic, sorted desc."""
    with app.state.store.driver.session() as session:
        counts = _topic_counts(session)
    topics = sorted(({"name": name, "count": n} for name, n in counts.items()),
                    key=lambda t: (-t["count"], t["name"]))
    return {"topics": topics}


def _topic_items(session, name: str) -> list:
    """Every message/link ABOUT ``name``, newest first, 300-char snippet."""
    message_rows = session.run(
        "MATCH (m:Message)-[:ABOUT]->(:Topic {name: $name}) "
        "RETURN m.sent_at AS sent_at, m.text AS text",
        name=name,
    ).data()
    link_rows = session.run(
        "MATCH (l:Link)-[:ABOUT]->(:Topic {name: $name}) "
        "RETURN l.url AS url, l.title AS title, l.summary AS summary, l.sent_at AS sent_at",
        name=name,
    ).data()
    items = [
        {"id": _message_id(row["sent_at"], row["text"]), "type": "message",
         "sent_at": row["sent_at"] or "", "text": (row["text"] or "")[:300]}
        for row in message_rows
    ]
    items += [
        {"id": _link_id(row["url"]), "type": "link", "sent_at": row.get("sent_at") or "",
         "text": (row.get("summary") or "")[:300], "url": row.get("url") or "",
         "title": row.get("title") or ""}
        for row in link_rows
    ]
    items.sort(key=lambda it: it["sent_at"], reverse=True)
    return items


def _topic_neighbors(session, name: str) -> list:
    """Top 5 topics co-occurring on the same items as ``name``, with counts."""
    message_rows = session.run(
        "MATCH (:Topic {name: $name})<-[:ABOUT]-(m:Message)-[:ABOUT]->(other:Topic) "
        "WHERE other.name <> $name "
        "RETURN other.name AS name, count(DISTINCT m) AS n",
        name=name,
    ).data()
    link_rows = session.run(
        "MATCH (:Topic {name: $name})<-[:ABOUT]-(l:Link)-[:ABOUT]->(other:Topic) "
        "WHERE other.name <> $name "
        "RETURN other.name AS name, count(DISTINCT l) AS n",
        name=name,
    ).data()
    counts: dict = {}
    for row in message_rows + link_rows:
        counts[row["name"]] = counts.get(row["name"], 0) + row["n"]
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"name": n, "count": c} for n, c in ranked[:5]]


def _topic_suggestions(session, name: str) -> list:
    """5 closest topic names by case-insensitive substring match, for a 404."""
    rows = session.run("MATCH (t:Topic) RETURN t.name AS name").data()
    needle = name.lower()
    matches = [row["name"] for row in rows
              if needle in row["name"].lower() or row["name"].lower() in needle]
    matches.sort(key=lambda n: (abs(len(n) - len(name)), n))
    return matches[:5]


@app.get("/topics/{name}")
def browse_topic(name: str, limit: int = Query(default=50, ge=1, le=500)):
    """Every message/link ABOUT ``name``, newest first, plus total and neighbors."""
    with app.state.store.driver.session() as session:
        items = _topic_items(session, name)
        if not items:
            suggestions = _topic_suggestions(session, name)
            detail = f"unknown topic: {name}"
            if suggestions:
                detail += f". Did you mean: {', '.join(suggestions)}?"
            raise HTTPException(status_code=404, detail=detail)
        neighbors = _topic_neighbors(session, name)
    return {"topic": name, "total": len(items), "items": items[:limit],
            "neighbors": neighbors}


@app.get("/metrics")
def prometheus_metrics():
    """Expose Prometheus metrics for the local deployment."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
