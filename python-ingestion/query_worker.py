import json
from pathlib import Path

import redis
import requests
from dotenv import load_dotenv
from worker_config import load_worker_config

load_dotenv(Path(__file__).resolve().parent / ".env")

STREAM = "manthan:queries"
GROUP = "query-workers"
CONSUMER = "query-worker-1"

CONFIG = load_worker_config()
BOT_TOKEN = CONFIG.bot_token
URL = f"{CONFIG.api_base_url}/search"


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
            id="0",
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


def format_results(data: dict) -> str:
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


def main():
    ensure_group()

    print("Query worker started")

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

                resp = requests.post(URL, json={"query": payload.get("text", ""), "top_k": 5})
                resp.raise_for_status()

                results = resp.json()
                print("Search count:", results.get("count"))
                send_telegram_message(
                    payload["chat_id"],
                    format_results(results),
                )

                r.xack(STREAM, GROUP, message_id)


if __name__ == "__main__":
    main()
