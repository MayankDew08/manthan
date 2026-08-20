import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import link_scraper
from link_scraper import LinkScraper, ScrapeResult

FAILS = []


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


class FakeResp:
    def __init__(self, status_code, text="", json=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._json = json
        self.headers = headers or {}

    def json(self):
        return self._json


META = {
    "full_name": "qdrant/qdrant",
    "description": "High-performance, massive-scale vector database.",
    "homepage": "https://qdrant.tech",
    "topics": ["vector-database", "rust", "search"],
    "language": "Rust",
    "stargazers_count": 19000,
    "license": {"spdx_id": "Apache-2.0"},
    "default_branch": "master",
}
README = ("# Qdrant\nQdrant is a vector similarity search engine that provides a "
          "production-ready service for vector search. It solves the problem of "
          "efficient nearest-neighbor search at scale.")
RAW_TEXT = README


def api_repo(url, headers, **kw):
    return FakeResp(200, json=dict(META))


def api_readme(url, headers, **kw):
    return FakeResp(200, text=README)


calls = []
def fake_get(url, **kw):
    calls.append(url)
    if "api.github.com" in url:
        if url.endswith("/readme"):
            return api_readme(url, kw.get("headers"))
        return api_repo(url, kw.get("headers"))
    return FakeResp(404)


def make_scraper():
    s = LinkScraper(delay=0)
    LinkScraper.start = lambda self: None
    return s


link_scraper.requests.get = fake_get

s = make_scraper()
r = s.scrape("https://github.com/qdrant/qdrant")
check("repo scraped", r.status == "scraped", r)
check("repo title full_name", r.title == "qdrant/qdrant", r.title)
check("repo raw_text has metadata + README",
      "vector similarity search" in r.raw_text and "Stars: 19000" in r.raw_text
      and "Apache-2.0" in r.raw_text, r.raw_text[:120])
check("repo note via api", "github readme via api" in r.notes)
check("repo no raw.githubusercontent fetch", not any("raw.githubusercontent" in u for u in calls))


def blocked_web(url):
    return ScrapeResult(url=url, status="blocked", block_reason="auth_required")


# API 404 (private/deleted, no token) -> web fallback
fake404 = lambda url, **kw: FakeResp(404)
link_scraper.requests.get = fake404
s = make_scraper()
s._scrape_web = blocked_web
r = s.scrape("https://github.com/private/secret-repo")
check("404 falls back to web", r.status == "blocked" and r.block_reason == "auth_required", r)
check("404 note added", any("falling back to web" in n for n in r.notes), r.notes)

# rate-limit 403 -> web fallback
link_scraper.requests.get = lambda url, **kw: FakeResp(
    403, headers={"X-RateLimit-Remaining": "0"})
s = make_scraper()
s._scrape_web = blocked_web
r = s.scrape("https://github.com/qdrant/qdrant")
check("rate-limit falls back to web", r.status == "blocked", r)


# blob link -> repo context, no raw-code fetch
calls.clear()
link_scraper.requests.get = fake_get
s = make_scraper()
r = s.scrape("https://github.com/qdrant/qdrant/blob/master/src/main.rs")
check("blob resolves to repo context", r.status == "scraped" and "vector similarity search" in r.raw_text)
check("blob notes file path", any("file: master/src/main.rs" in n for n in r.notes), r.notes)
check("blob did not fetch raw code", not any("raw.githubusercontent" in u for u in calls))

# issues -> web scrape
seen = {}
def issues_web(url):
    seen["url"] = url
    return ScrapeResult(url=url, status="scraped")
s = make_scraper()
s._scrape_web = issues_web
r = s.scrape("https://github.com/qdrant/qdrant/issues/123")
check("issues routes to web", seen.get("url", "").endswith("/issues/123"), seen)

# gist -> web scrape
s = make_scraper()
s._scrape_web = issues_web
r = s.scrape("https://gist.github.com/u/abc123")
check("gist routes to web", seen.get("url", "").startswith("https://gist.github.com"), seen)

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)