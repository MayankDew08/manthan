import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
os.environ["REDIS_ADDR"] = "redis.internal:6380"
os.environ["MANTHAN_API_URL"] = "http://api.internal:9000/"

import ingest_worker
import query_worker


FAILS = []


def check(name, condition, extra=""):
    if not condition:
        FAILS.append(name)
    print(f"[{'ok' if condition else 'FAIL'}] {name} {extra}")


for name, worker, endpoint in (
    ("ingest", ingest_worker, "/ingest-message"),
    ("query", query_worker, "/search"),
):
    connection = worker.r.connection_pool.connection_kwargs
    check(f"{name} worker reads Redis host", connection["host"] == "redis.internal",
          connection["host"])
    check(f"{name} worker reads Redis port", connection["port"] == 6380,
          connection["port"])
    check(f"{name} worker reads API URL",
          worker.URL == f"http://api.internal:9000{endpoint}", worker.URL)
    check(f"{name} worker reads bot token", worker.BOT_TOKEN == "test-token")

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
