"""Consume ingest jobs from Redis, call the API, and reply on Telegram."""

import json
from pathlib import Path

import redis
import requests
from dotenv import load_dotenv
from worker_config import load_worker_config
from typing import cast, List, Tuple, Dict

load_dotenv(Path(__file__).resolve().parent / ".env")

STREAM = "manthan:ingest"
GROUP = "ingest-workers"
CONSUMER = "ingest-worker-1"

CONFIG = load_worker_config()
BOT_TOKEN = CONFIG.bot_token
URL = f"{CONFIG.api_base_url}/ingest-message"
PASTE_URL = f"{CONFIG.api_base_url}/links/paste"

StreamReply = List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]

r = redis.Redis(
    host=CONFIG.redis_host,
    port=CONFIG.redis_port,
    decode_responses=True,
)


def ensure_group():
    """Create the Redis consumer group once and tolerate repeat starts."""
    try:
        r.xgroup_create(
            STREAM,
            GROUP,
            id="$",
            mkstream=True,
        )
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def send_telegram_message(chat_id: int, text: str):
    """Send the worker result back to the originating Telegram chat."""
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


def call_ingest_api(payload: dict) -> dict:
    """Forward one queued ingest job to the FastAPI endpoint."""
    response = requests.post(
        URL,
        json={
            "job_id": payload.get("job_id"),
            "telegram_update_id": payload.get("telegram_update_id"),
            "chat_id": payload.get("chat_id"),
            "sender_name": payload.get("sender_name", ""),
            "text": payload["text"],
            "telegram_message_id": payload.get("telegram_message_id"),
            "skip_grading": payload.get("skip_grading", False),
            "received_at": payload.get("received_at"),
        },
        timeout=(10, CONFIG.ingest_timeout),
    )

    response.raise_for_status()
    return response.json()


def call_paste_api(payload: dict) -> dict:
    """Complete a blocked link with manually pasted content."""
    response = requests.post(
        PASTE_URL,
        json={
            "link_id": payload.get("link_id", ""),
            "content": payload.get("text", ""),
        },
        timeout=(10, CONFIG.ingest_timeout),
    )

    if response.status_code == 404:
        return {"ok": False, "error": "unknown link id",
                "link_id": payload.get("link_id", "")}
    if response.status_code == 422:
        return {"ok": False, "error": "link id and content are required"}

    response.raise_for_status()
    return response.json()


def format_paste_reply(result: dict) -> str:
    """Confirm a completed paste with the stored title and topics."""
    if not result.get("ok"):
        detail = result.get("error") or result.get("detail") or "paste failed"
        return f"Paste failed — {detail}"
    parts = [
        "Pasted successfully",
        result.get("url", ""),
        f"Title: {result.get('title', '') or '-'}",
        f"Topics: {', '.join(result.get('topics') or []) or '-'}",
    ]
    if result.get("summary"):
        parts.append(f"Summary: {result['summary'][:300]}")
    return "\n".join(parts)


def _format_link_outcomes(result: dict) -> str:
    """Render per-link outcomes so senders see what happened to each URL."""
    outcomes = result.get("link_outcomes") or []
    if not outcomes:
        return ""
    lines = []
    for out in outcomes:
        line = f"• {out.get('url', '')} — {out.get('status', 'unknown')}"
        if out.get("block_reason"):
            line += f" ({out['block_reason']})"
        if out.get("title"):
            line += f"\n  {out['title']}"
        lines.append(line)
    return "Links:\n" + "\n".join(lines)


def format_ingest_reply(result: dict) -> str:
    """Keep Telegram replies compact while surfacing metadata and link outcomes."""
    msg = result.get("message", {})
    parts = [
        "Stored successfully" if result.get("ok") else "Ingestion failed",
        f"Quality: {msg.get('quality', 'skipped')}",
        f"Topics: {', '.join(msg.get('topics') or []) or '-'}",
        f"Entities: {', '.join(msg.get('entities') or []) or '-'}",
    ]
    links = _format_link_outcomes(result)
    if links:
        parts.append(links)
    parts.append(f"Vectored: {'yes' if result.get('vectored') else 'no'}")
    return "\n".join(parts)


def main():
    """Block on the ingest stream and acknowledge jobs only after replying."""
    ensure_group()

    print("Ingest worker started")

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

        for _, entries in messages:
            for message_id, fields in entries:
                payload = json.loads(fields["data"])

                print("Redis message ID:", message_id)
                print("Job ID:", payload.get("job_id"))

                if payload.get("type") == "paste":
                    result = call_paste_api(payload)
                    reply = format_paste_reply(result)
                else:
                    result = call_ingest_api(payload)
                    reply = format_ingest_reply(result)

                send_telegram_message(
                    payload["chat_id"],
                    reply,
                )

                r.xack(STREAM, GROUP, message_id)


if __name__ == "__main__":
    main()
