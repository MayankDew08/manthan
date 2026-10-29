"""Manthan MCP server: a thin stdio adapter over the FastAPI app (M1).

Holds no DB clients and loads no models -- only HTTP calls to
``MANTHAN_API_URL``. stdout is the MCP protocol channel: never ``print()``.
Logging goes to stderr; one JSON line per tool call is appended to
``logs/mcp_calls.jsonl``.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("manthan_mcp")

API_URL = os.environ.get("MANTHAN_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 10.0
SNIPPET_CHARS = 300  # matches retrieval.SNIPPET_CHARS; kept local -- no DB/model
                     # imports here (M1), so not imported from retrieval.py.

LOG_PATH = Path(__file__).resolve().parent / "logs" / "mcp_calls.jsonl"

# A module-level client so tests can swap in an httpx.MockTransport.
_client = httpx.Client(timeout=TIMEOUT)

INSTRUCTIONS = (
    "Manthan holds the user's saved knowledge from their tech WhatsApp groups. When you\n"
    "use a result, cite it by its date and link (or a short quote). Never present\n"
    "something as coming from the user's groups unless a tool returned it."
)

# Verbatim from docs/mcp-server-spec.md section 3 (Q6/Q7), minus the
# kind="introduction" sentence: kind isn't exposed until Phase 3, and that
# sentence is added together with the parameter then.
SEARCH_DESCRIPTION = (
    "Searches the user's saved knowledge from their tech WhatsApp groups: messages\n"
    "and shared links (articles, repos, videos) where members discussed tools,\n"
    "libraries, errors, and trends, with their experiences and opinions.\n"
    "Call it when the user is choosing a tool or approach, hits an error or crash,\n"
    "or asks what their groups said or shared about something; skip it for general\n"
    "knowledge or routine coding steps that don't need the community's view.\n"
    "Pass a natural-language question (e.g. \"problems people had with Qdrant\n"
    "filters\"), not single keywords.\n"
    "Returns up to 5 results, each with an id, date, short snippet, link title/URL\n"
    "if any, plus an overall confidence: strong, weak, or none. On weak, check the\n"
    "snippets really answer the question before using them.\n"
    "Call get_item(id) for the full text; if confidence is weak or none, rephrase\n"
    "once or tell the user their groups haven't discussed it. Never invent results."
)

GET_ITEM_DESCRIPTION = (
    "Returns the full text of one search or browse result by its id, including "
    "the shared link's title and summary. Use it only for results you need in full."
)

LIST_TOPICS_DESCRIPTION = (
    "Lists the topics in the user's saved group knowledge with how many items "
    "each has. Use it to see what the groups talk about, or to find the exact "
    "topic name for browse_topic."
)

BROWSE_TOPIC_DESCRIPTION = (
    "Returns every saved item tagged with one exact topic, plus related topics. "
    "Use it when the user wants a complete list or overview of a topic (e.g. "
    "'everything shared about RAG'); use search for a specific question. "
    "Summarize the count and offer the list; don't dump all items unprompted."
)

mcp = FastMCP("manthan", instructions=INSTRUCTIONS)


def _log_call(tool: str, args: dict, result_count: Optional[int],
              confidence: Optional[str], error: Optional[str]) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "tool": tool, "args": args,
           "result_count": result_count, "confidence": confidence, "error": error}
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("failed to append to %s", LOG_PATH)


def _unreachable_message() -> str:
    return (f"Manthan API is not reachable at {API_URL}. Ask the user to start it "
            "(docker compose up + the FastAPI app).")


def _handle_response(resp: httpx.Response, path: str) -> tuple[Optional[dict], Optional[str]]:
    if resp.status_code == 404:
        try:
            detail = resp.json().get("detail", f"not found: {path}")
        except ValueError:
            detail = f"not found: {path}"
        return None, detail
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception("request to %s failed: %s", path, resp.status_code)
        return None, f"Manthan API error ({resp.status_code}) for {path}."
    return resp.json(), None


def _get(path: str, **params) -> tuple[Optional[dict], Optional[str]]:
    """GET ``path``; returns ``(json, None)`` on success, ``(None, message)`` otherwise."""
    try:
        resp = _client.get(f"{API_URL}{path}", params=params)
    except httpx.RequestError:
        logger.exception("GET %s unreachable", path)
        return None, _unreachable_message()
    return _handle_response(resp, path)


def _post(path: str, json_body: dict) -> tuple[Optional[dict], Optional[str]]:
    try:
        resp = _client.post(f"{API_URL}{path}", json=json_body)
    except httpx.RequestError:
        logger.exception("POST %s unreachable", path)
        return None, _unreachable_message()
    return _handle_response(resp, path)


@mcp.tool(description=SEARCH_DESCRIPTION)
def search(query: str) -> str:
    data, error = _post("/search", {"query": query, "top_k": 5})
    if error:
        _log_call("search", {"query": query}, None, None, error)
        return error
    assert data is not None  # error is None, so _post returned a body
    results = (data.get("results") or [])[:5]
    confidence = data.get("confidence", "none")
    items = [{
        "id": r.get("id"),
        "sent_at": r.get("sent_at", ""),
        "text": (r.get("text") or "")[:SNIPPET_CHARS],
        "url": r.get("url", ""),
        "title": r.get("title", ""),
        "related_links": r.get("related_links") or [],
    } for r in results]
    _log_call("search", {"query": query}, len(items), confidence, None)
    return json.dumps({"results": items, "confidence": confidence}, ensure_ascii=False)


@mcp.tool(description=GET_ITEM_DESCRIPTION)
def get_item(id: str) -> str:
    data, error = _get(f"/items/{id}")
    if error:
        _log_call("get_item", {"id": id}, None, None, error)
        return error
    _log_call("get_item", {"id": id}, 1, None, None)
    return json.dumps(data, ensure_ascii=False)


@mcp.tool(description=LIST_TOPICS_DESCRIPTION)
def list_topics() -> str:
    data, error = _get("/topics")
    if error:
        _log_call("list_topics", {}, None, None, error)
        return error
    assert data is not None
    topics = data.get("topics") or []
    _log_call("list_topics", {}, len(topics), None, None)
    return json.dumps({"topics": topics}, ensure_ascii=False)


@mcp.tool(description=BROWSE_TOPIC_DESCRIPTION)
def browse_topic(topic: str) -> str:
    data, error = _get(f"/topics/{topic}")
    if error:
        _log_call("browse_topic", {"topic": topic}, None, None, error)
        return error
    assert data is not None
    items = data.get("items") or []
    _log_call("browse_topic", {"topic": topic}, len(items), None, None)
    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
