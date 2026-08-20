import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import link_scraper as ls
from link_scraper import LinkScraper

# fixtures are served from 127.0.0.1 (is_x() would return False); force the
# X path so truncation/paywall handling for X posts is exercised
ls.is_x = lambda url: "x_post" in url

FAILS = []
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(FIXTURES), **kw)

    def log_message(self, *a):
        pass


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{server.server_address[1]}"

with LinkScraper(headless=True, delay=0, timeout_ms=15000) as scraper:
    r = scraper.scrape(f"{base}/x_post.html")
    check("x not-logged-in page scraped", r.status == "scraped", f"{r.status} {r.block_reason}")
    check("x show more expanded full post",
          "256k context window" in r.raw_text and "Apache 2.0 license" in r.raw_text
          and "Docker image" in r.raw_text, f"{len(r.raw_text)} chars")
    check("x teaser present", "teaser text visible" in r.raw_text)
    check("x first post only (reply excluded)", "must be excluded" not in r.raw_text,
          r.raw_text[:200])
    check("x title", r.title == "Alibaba Qwen on X", r.title)

    r = scraper.scrape(f"{base}/x_post_stuck.html")
    check("x stuck show-more -> truncated",
          r.status == "blocked" and r.block_reason == "truncated",
          f"{r.status} {r.block_reason}")
    check("x truncated keeps partial text",
          "visible teaser only" in r.raw_text and "hidden full article" not in r.raw_text,
          f"{len(r.raw_text)} chars")

server.shutdown()

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)