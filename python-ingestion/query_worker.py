"""Consume Redis query jobs, call search, and return results to Telegram."""

import json
from pathlib import Path

import redis
import requests
from dotenv import load_dotenv
from worker_config import load_worker_config
from typing import cast, List, Tuple, Dict

load_dotenv(Path(__file__).resolve().parent / ".env")

STREAM = "manthan:queries"
GROUP = "query-workers"
CONSUMER = "query-worker-1"

CONFIG = load_worker_config()
BOT_TOKEN = CONFIG.bot_token
URL = f"{CONFIG.api_base_url}/search"
PENDING_URL = f"{CONFIG.api_base_url}/links/pending"
SKIP_URL = f"{CONFIG.api_base_url}/links/skip"

WELCOME = (
    "👋 Welcome to Manthan — your personal knowledge base.\n"
    "\n"
    "/ingest <note>        store a note (--trusted skips grading)\n"
    "/via Name: <note>     attribute a note to someone else\n"
    "<any link>            auto-scrape & store it\n"
    "/blocked              links awaiting manual content\n"
    "/paste LNK-xxx : ...  complete a blocked link\n"
    "/skip LNK-xxx         dismiss a link you don't want\n"
    "/ask <question>       search everything you've saved"
)


r = redis.Redis(
    host=CONFIG.redis_host,
    port=CONFIG.redis_port,
    decode_responses=True,
)

StreamReply = List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]



def ensure_group():
    """Create the query consumer group once and tolerate repeat starts."""
    try:
        r.xgroup_create(
            STREAM,
            GROUP,
            id="0",
            mkstream=True,
        )
    except redis.ResponseError as exc: 
        if "BUSYGROUP" not in str(exc):
            raise


def send_telegram_message(chat_id: int, text: str):
    """Send search results back to the originating Telegram chat."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=10,
    )

    response.raise_for_status()


def format_results(data: dict) -> str:
    """Render search hits into a message that fits Telegram reasonably well."""
    results = data.get("results", [])
    if not results:
        return "No results found."

    lines = [f"Found {data.get('count', len(results))} results:"]
    for i, result in enumerate(results, 1):
        header = result.get("sender") or result.get("title") or "?"
        lines.append(f"{i}. {header} (score {result.get('score', 0)})")
        lines.append(f"   {result.get('text', '')}")
        if result.get("url"):
            lines.append(f"   {result['url']}")
    return "\n".join(lines)


def format_blocked_reply(data: dict) -> str:
    """Render blocked links as JSON blocks ready for the /paste command."""
    links = data.get("links", [])
    if not links:
        return "No blocked links 🎉"

    blocks = []
    for link in links:
        blocks.append(json.dumps({
            "action": "awaiting_link_content",
            "link_id": link.get("link_id") or "",
            "url": link.get("url", ""),
        }, indent=2))
    header = f"Blocked links ({data.get('count', len(links))}):"
    return header + "\n\n" + "\n\n".join(blocks)


def call_skip_api(link_id: str) -> dict:
    """Dismiss a link that is not worth completing."""
    response = requests.post(SKIP_URL, json={"link_id": link_id}, timeout=30)
    if response.status_code == 404:
        return {"ok": False, "error": f"unknown link id: {link_id}"}
    if response.status_code == 409:
        detail = response.json().get("detail", "cannot skip this link")
        return {"ok": False, "error": detail}
    response.raise_for_status()
    return response.json()


def format_skip_reply(result: dict) -> str:
    """Confirm a skipped link or explain why the skip was rejected."""
    if not result.get("ok"):
        return f"Skip failed — {result.get('error', 'unknown error')}"
    return f"Skipped ✓\n{result.get('url', '')}"


def main():
    """Block on the query stream and acknowledge only after a reply is sent."""
    ensure_group()

    print("Query worker started")

    while True:
        raw = r.xreadgroup(
                    groupname=GROUP,
                    consumername=CONSUMER,
                    streams={
                        STREAM: ">"
                    },
                    count=1,
                    block=5000,
                )
        messages = cast(StreamReply, raw)

        if not messages:
            continue

        for _,entries in messages:
            
            for message_id, fields in entries:
                payload = json.loads(fields["data"])

                print("Redis message ID:", message_id)
                print("Job ID:", payload.get("job_id"))

                if payload.get("type") == "start":
                    reply = WELCOME
                elif payload.get("type") == "skip":
                    reply = format_skip_reply(
                        call_skip_api(payload.get("text", "").strip()))
                elif payload.get("type") == "blocked":
                    resp = requests.get(PENDING_URL, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    blocked = [link for link in data.get("links", [])
                               if link.get("status") == "blocked"]
                    reply = format_blocked_reply(
                        {"count": len(blocked), "links": blocked})
                else:
                    resp = requests.post(URL, json={"query": payload.get("text", ""), "top_k": 5})
                    resp.raise_for_status()

                    results = resp.json()
                    print("Search count:", results.get("count"))
                    reply = format_results(results)

                send_telegram_message(
                    payload["chat_id"],
                    reply,
                )

                r.xack(STREAM, GROUP, message_id)


if __name__ == "__main__":
    main()
