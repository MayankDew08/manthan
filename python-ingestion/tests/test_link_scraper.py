import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from link_scraper import LinkScraper

FAILS = []
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(FIXTURES), **kw)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/forbidden"):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"forbidden")
            return
        if self.path.startswith("/notfound"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"nope")
            return
        super().do_GET()


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{server.server_address[1]}"

with LinkScraper(headless=True, delay=0, timeout_ms=15000) as scraper:
    r = scraper.scrape(f"{base}/article_readmore.html")
    check("readmore -> scraped", r.status == "scraped", f"{r.status} {r.block_reason}")
    check("readmore expanded hidden section", "LangGraph" in r.raw_text and "CUDA 12.1" in r.raw_text,
          f"{len(r.raw_text)} chars")
    check("readmore title", r.title == "Read More Fixture", r.title)

    r = scraper.scrape(f"{base}/article_plain.html")
    check("plain article scraped", r.status == "scraped" and "27b dense model" in r.raw_text,
          f"{r.status} {len(r.raw_text)} chars")

    r = scraper.scrape(f"{base}/login_wall.html")
    check("login wall blocked", r.status == "blocked" and r.block_reason == "auth_required",
          f"{r.status} {r.block_reason}")

    r = scraper.scrape(f"{base}/empty.html")
    check("empty blocked", r.status == "blocked" and r.block_reason == "empty_content",
          f"{r.status} {r.block_reason}")

    r = scraper.scrape(f"{base}/paywall.html")
    check("paywall blocked", r.status == "blocked" and r.block_reason == "paywall",
          f"{r.status} {r.block_reason}")

    r = scraper.scrape(f"{base}/forbidden")
    check("403 blocked", r.status == "blocked" and r.block_reason == "auth_required",
          f"{r.status} {r.block_reason}")

    r = scraper.scrape(f"{base}/notfound")
    check("404 blocked", r.status == "blocked" and r.block_reason == "not_found",
          f"{r.status} {r.block_reason}")

server.shutdown()

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
