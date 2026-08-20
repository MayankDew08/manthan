import json
from pathlib import Path

import redis
import requests
from dotenv import load_dotenv
from worker_config import load_worker_config

load_dotenv(Path(__file__).resolve().parent / ".env")

STREAM = "manthan:ingest"
GROUP = "ingest-workers"
CONSUMER = "ingest-worker-1"

CONFIG = load_worker_config()
BOT_TOKEN = CONFIG.bot_token
URL = f"{CONFIG.api_base_url}/ingest-message"

r = redis.Redis(
    host=CONFIG.redis_host,
    port=CONFIG.redis_port,
    decode_responses=True,
)


def ensure_group():
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
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def format_ingest_reply(result: dict) -> str:
    msg = result.get("message", {})
    return (
        "Stored successfully\n"
        f"Quality: {msg.get('quality', 'skipped')}\n"
        f"Topics: {', '.join(msg.get('topics') or [])}\n"
        f"Entities: {', '.join(msg.get('entities') or [])}"
    )


def main():
    ensure_group()

    print("Ingest worker started")

    while True:
        messages = r.xreadgroup(
            groupname=GROUP,
            consumername=CONSUMER,
            streams={
                STREAM: ">"
            },
            count=1,
            block=5000,
        )

        if not messages:
            continue

        for _, entries in messages:
            for message_id, fields in entries:
                payload = json.loads(fields["data"])

                print("Redis message ID:", message_id)
                print("Job ID:", payload.get("job_id"))

                result = call_ingest_api(payload)
                reply = format_ingest_reply(result)

                send_telegram_message(
                    payload["chat_id"],
                    reply,
                )

                r.xack(STREAM, GROUP, message_id)


if __name__ == "__main__":
    main()
