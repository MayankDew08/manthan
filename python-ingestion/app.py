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
import pipeline
from grader import grade_messages
from link_extract import extract_links, parse_github_url
from search import search
from store import KnowledgeStore, _url_to_title
from vector_store import VectorStore

load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_REQUESTS = Counter("manthan_search_requests_total", "search requests", ["engine"])
SEARCH_LATENCY = Histogram("manthan_search_seconds", "search latency", ["engine"])
INGEST_JOBS = Counter("manthan_ingest_jobs_total", "ingest background jobs", ["outcome"])
PASTE_REQUESTS = Counter("manthan_paste_requests_total", "paste-link requests", ["outcome"])
MESSAGES_QUERIES = Counter("manthan_messages_queries_total", "messages list queries")
PENDING_QUERIES = Counter("manthan_pending_queries_total", "pending-links queries")

JOBS: dict = {}
JOBS_LOCK = threading.Lock()


def _new_job(chat_path: str) -> str:
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
    sender_name: str = ""
    text: str
    telegram_message_id: Optional[int] = None
    skip_grading: bool = False
    received_at: Optional[str] = None


class PasteRequest(BaseModel):
    url: str
    content: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = KnowledgeStore()
    app.state.vs = VectorStore()
    app.state.vs.ensure_collection()
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
    t0 = time.monotonic()
    try:
        results = search(req.query, topics=req.topics, min_quality=req.min_quality,
                         top_k=req.top_k, vs=app.state.vs, store=app.state.store)
    except Exception:  # pragma: no cover
        SEARCH_REQUESTS.labels(engine="hybrid").inc()
        logger.exception("Search request failed")
        raise HTTPException(status_code=500, detail="search failed")
    SEARCH_REQUESTS.labels(engine="hybrid").inc()
    SEARCH_LATENCY.labels(engine="hybrid").observe(time.monotonic() - t0)
    return {"query": req.query, "count": len(results), "results": results}


def _run_ingest_job(job_id: str, chat_path: str, force: bool,
                    no_scrape: bool, min_quality: int) -> None:
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
    if not os.path.exists(req.chat_path):
        raise HTTPException(status_code=404,
                            detail=f"chat file not found: {req.chat_path}")
    job_id = _new_job(req.chat_path)
    bg.add_task(_run_ingest_job, job_id, req.chat_path, req.force,
                req.no_scrape, req.min_quality)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")
    return {"job_id": job_id, **job}


@app.post("/ingest-message")
def ingest_message(req: IngestMessageRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    sent_at = req.received_at or datetime.now(timezone.utc).isoformat()
    sender = req.sender_name or ""

    if req.skip_grading:
        quality, trusted = 5, True
    else:
        try:
            graded = grade_messages([text])
        except Exception:
            logger.exception("Message grading failed")
            raise HTTPException(status_code=502, detail="grading failed")
        quality, trusted = graded[0].quality, False

    links = extract_links(text)
    try:
        en = enhancer.enrich_message(text, links)
    except Exception:
        logger.exception("Message enrichment failed; using empty metadata")
        en = {"link_intent": None, "entities": [], "topics": []}

    record = {
        "sent_at": sent_at,
        "sender": sender,
        "quality": quality,
        "original_text": text,
        "links": links,
        "link_intent": en.get("link_intent"),
        "entities": en.get("entities") or [],
        "topics": en.get("topics") or [],
        "trusted": trusted,
    }

    app.state.store.add_message(record)
    vectored = False
    if quality >= pipeline.MIN_QUALITY:
        app.state.vs.upsert_message(record)
        vectored = True

    INGEST_JOBS.labels(outcome="success").inc()
    return {
        "ok": True,
        "message": {
            "sender": sender,
            "sent_at": sent_at,
            "quality": quality,
            "trusted": trusted,
            "topics": record["topics"],
            "entities": record["entities"],
            "links": links,
        },
        "vectored": vectored,
    }


@app.get("/links/pending")
def pending_links():
    PENDING_QUERIES.inc()
    rows = app.state.store.pending_links()
    return {"count": len(rows), "links": rows}


def _existing_title(url: str) -> str:
    with app.state.store.driver.session() as session:
        row = session.run(
            "MATCH (l:Link {url: $url}) RETURN l.title AS title",
            url=url,
        ).single()
    return (row["title"] or "") if row else ""


@app.post("/links/paste")
def paste_link(req: PasteRequest):
    url = req.url.strip()
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
    PASTE_REQUESTS.labels(outcome="success").inc()
    return {
        "ok": True,
        "url": url,
        "title": title,
        "summary": summary.get("summary", ""),
        "topics": summary.get("topics", []),
        "entities": summary.get("entities", []),
    }


def _query_messages(*, sender: Optional[str], topic: Optional[str],
                    entity: Optional[str], since: Optional[str],
                    min_quality: Optional[int], limit: int) -> list:
    where = []
    params = {}
    if sender:
        where.append("m.sender = $sender")
        params["sender"] = sender
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
            "sender": m.get("sender"),
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
def list_messages(sender: Optional[str] = None, topic: Optional[str] = None,
                  entity: Optional[str] = None, since: Optional[str] = None,
                  min_quality: Optional[int] = Query(default=None, ge=0, le=10),
                  limit: int = Query(default=50, ge=1, le=500)):
    MESSAGES_QUERIES.inc()
    rows = _query_messages(sender=sender, topic=topic, entity=entity,
                           since=since, min_quality=min_quality, limit=limit)
    return {"count": len(rows), "messages": rows}


@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
