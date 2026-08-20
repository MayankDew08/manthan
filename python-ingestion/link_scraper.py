"""Retrieve generic web, GitHub, X, and YouTube link content for enrichment."""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
import trafilatura
from playwright.sync_api import sync_playwright

from link_extract import is_github, is_x, is_youtube, parse_github_url, youtube_video_id

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

_CONSENT_RE = re.compile(r"(accept( all)?|agree|allow( all)?|got it|i agree|ok)$", re.I)
_EXPAND_RE = re.compile(
    r"(read more|see more|show more|load more|continue reading|view full"
    r"( article| story| post)?|expand|open full)", re.I)
_LOGIN_PATH_RE = re.compile(r"(/login|/signin|/sign-in|/auth|/checkpoint|/captcha|/sessions)", re.I)
_CHALLENGE_TITLE_RE = re.compile(r"(just a moment|attention required|checking your browser|cf-chl)", re.I)
_PAYWALL_RE = re.compile(
    r"(become a member|to continue reading|subscriber[- ]only|subscribe to read|"
    r"sign in to read|unlock the (full|rest|article)|paid subscribers|premium article|"
    r"this article is (behind|only for|available to)|membership required|"
    r"read the full article with)", re.I)

MAX_EXPAND_CLICKS = 5
EXPAND_GROWTH_THRESHOLD = 200
CONSENT_WAIT_MS = 500
EXPAND_WAIT_MS = 1200

GITHUB_RAW_TEXT_LIMIT = 2_000_000


@dataclass
class ScrapeResult:
    """Captured link content or a reason automated retrieval could not complete."""

    url: str
    status: str = "scraped"
    block_reason: Optional[str] = None
    final_url: str = ""
    title: str = ""
    raw_text: str = ""
    notes: List[str] = field(default_factory=list)


class LinkScraper:
    """Dispatch links to source-specific scrapers with browser fallback."""

    def __init__(self, headless: bool = True, delay: float = 1.5,
                 max_expand_clicks: int = MAX_EXPAND_CLICKS,
                 timeout_ms: int = 25000, max_html_bytes: int = 5_000_000):
        self.headless = headless
        self.delay = delay
        self.max_expand_clicks = max_expand_clicks
        self.timeout_ms = timeout_ms
        self.max_html_bytes = max_html_bytes
        self._pw = None
        self._browser = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        if self._browser is not None:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )

    def close(self):
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._pw is not None:
            self._pw.stop()
            self._pw = None

    def scrape(self, url: str) -> ScrapeResult:
        """Select a source-specific strategy and return a normalized outcome."""
        gh = parse_github_url(url)
        if gh is not None and gh.kind in ("repo", "tree", "blob", "raw"):
            return self._scrape_github(url)
        self.start()
        try:
            if is_youtube(url):
                return self._scrape_youtube(url)
            if gh is not None:
                return self._scrape_github(url)
            if is_x(url):
                return self._scrape_x(url)
            return self._scrape_web(url)
        finally:
            if self.delay:
                time.sleep(self.delay)

    # ------------------------------------------------------------ web

    def _scrape_web(self, url: str) -> ScrapeResult:
        """Render a page, expand content, extract text, and classify access blocks."""
        self.start()
        context = self._browser.new_context(user_agent=UA, locale="en-US")
        page = context.new_page()
        result = ScrapeResult(url=url)
        status = None
        try:
            document_responses = []
            page.on("response", lambda r: document_responses.append(r)
                    if r.request.resource_type == "document" else None)
            resp = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            status = resp.status if resp is not None else None
            result.final_url = page.url
            self._dismiss_cookie_banner(page)
            self._expand_see_more(page)
            html = page.content()
            if len(html.encode("utf-8", "ignore")) > self.max_html_bytes:
                result.status = "blocked"
                result.block_reason = "too_heavy"
                return result
            result.title = self._get_title(page)
            result.raw_text = self._extract_text(page, html)
            reason = self._classify_web(status, result.final_url, result.title, html,
                                        result.raw_text)
            if reason:
                result.status = "blocked"
                result.block_reason = reason
            else:
                result.status = "scraped"
            return result
        except Exception as e:
            result.status = "blocked"
            result.block_reason = self._classify_exception(e)
            result.notes.append(f"{type(e).__name__}: {e}")
            return result
        finally:
            context.close()

    def _dismiss_cookie_banner(self, page):
        for sel in ("button", "a", "[role=button]", "input[type=submit]"):
            try:
                loc = page.locator(sel)
                for i in range(loc.count()):
                    el = loc.nth(i)
                    if not el.is_visible():
                        continue
                    txt = el.inner_text().strip()[:50]
                    if txt and _CONSENT_RE.match(txt):
                        try:
                            el.click(timeout=2000)
                            page.wait_for_timeout(CONSENT_WAIT_MS)
                            return
                        except Exception:
                            continue
            except Exception:
                continue

    def _expand_see_more(self, page):
        for _ in range(self.max_expand_clicks):
            before = self._body_len(page)
            if not self._click_expand(page):
                break
            page.wait_for_timeout(EXPAND_WAIT_MS)
            after = self._body_len(page)
            if after - before < EXPAND_GROWTH_THRESHOLD:
                break
        self._scroll_bottom(page)

    def _click_expand(self, page) -> bool:
        for sel in ("button", "a", "[role=button]", "summary", "span"):
            try:
                loc = page.locator(sel)
                for i in range(loc.count()):
                    el = loc.nth(i)
                    if not el.is_visible():
                        continue
                    txt = el.inner_text().strip()[:80]
                    if txt and _EXPAND_RE.search(txt):
                        try:
                            el.click(timeout=2000)
                            return True
                        except Exception:
                            continue
            except Exception:
                continue
        return False

    def _scroll_bottom(self, page):
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(600)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(600)
        except Exception:
            pass

    @staticmethod
    def _body_len(page) -> int:
        try:
            return len(page.evaluate("document.body.innerText || ''"))
        except Exception:
            return 0

    def _get_title(self, page) -> str:
        try:
            t = page.title().strip()
            if t:
                return t
        except Exception:
            pass
        try:
            meta = page.locator('meta[property="og:title"]').first
            if meta.count():
                return (meta.get_attribute("content") or "").strip()
        except Exception:
            pass
        return ""

    def _extract_text(self, page, html: str) -> str:
        try:
            text = trafilatura.extract(html, output_format="markdown",
                                       include_tables=True, include_links=True,
                                       favor_recall=True)
        except Exception:
            text = None
        text = (text or "").strip()
        if not text:
            try:
                text = page.inner_text("body").strip()
            except Exception:
                text = ""
        return text

    def _classify_web(self, status, final_url, title, html, text) -> Optional[str]:
        if status is not None and status in (401, 403, 407):
            return "auth_required"
        if status is not None and status == 404:
            return "not_found"
        if status is not None and status >= 500:
            return "failed"
        if _PAYWALL_RE.search(f"{title or ''} {(text or '')[:6000]}"):
            return "paywall"
        if _LOGIN_PATH_RE.search(final_url or ""):
            return "auth_required"
        if _CHALLENGE_TITLE_RE.search(title or ""):
            return "challenge_blocked"
        if len(text) < 50:
            if len(html) > 50_000:
                return "auth_required"
            return "empty_content"
        return None

    def _classify_exception(self, e) -> str:
        name = type(e).__name__
        msg = str(e).lower()
        if "timeout" in msg or name == "TimeoutError":
            return "failed"
        if "certificate" in msg or "tls" in msg:
            return "failed"
        if "net::err" in msg:
            if "name_not_resolved" in msg or "connection_refused" in msg:
                return "failed"
        return "failed"

    # ------------------------------------------------------------ x / twitter

    def _scrape_x(self, url: str) -> ScrapeResult:
        context = self._browser.new_context(user_agent=UA, locale="en-US")
        page = context.new_page()
        result = ScrapeResult(url=url)
        status = None
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            status = resp.status if resp is not None else None
            result.final_url = page.url
            self._dismiss_cookie_banner(page)
            self._expand_see_more(page)
            result.title = self._get_title(page)
            result.raw_text = self._x_post_text(page)
            if re.search(r"show more", result.raw_text[-120:], re.I):
                result.status = "blocked"
                result.block_reason = "truncated"
                result.notes.append("post still collapsed after show-more; ask user to paste")
                return result
            reason = self._classify_web(status, result.final_url, result.title,
                                        page.content(), result.raw_text)
            if reason:
                result.status = "blocked"
                result.block_reason = reason
            else:
                result.status = "scraped"
            return result
        except Exception as e:
            result.status = "blocked"
            result.block_reason = self._classify_exception(e)
            result.notes.append(f"{type(e).__name__}: {e}")
            return result
        finally:
            context.close()

    @staticmethod
    def _x_post_text(page) -> str:
        try:
            article = page.locator("article").first
            if article.count():
                return article.inner_text().strip()
        except Exception:
            pass
        try:
            tweet = page.locator('[data-testid="tweetText"]').first
            if tweet.count():
                return tweet.inner_text().strip()
        except Exception:
            pass
        try:
            return page.inner_text("body").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------ youtube

    def _scrape_youtube(self, url: str) -> ScrapeResult:
        vid = youtube_video_id(url)
        if not vid:
            return ScrapeResult(url=url, status="blocked", block_reason="invalid_youtube_url")
        text, src = self._fetch_youtube_transcript(vid)
        if not text:
            return ScrapeResult(url=url, status="blocked", block_reason="transcript_unavailable",
                                notes=[f"no transcript for {vid}"])
        result = ScrapeResult(url=url, status="scraped", final_url=url,
                              title=self._youtube_title(url) or f"youtube:{vid}",
                              raw_text=text)
        result.notes.append(f"transcript via {src}")
        return result

    @staticmethod
    def _youtube_title(url: str) -> str:
        try:
            r = requests.get("https://www.youtube.com/oembed",
                             params={"url": url, "format": "json"}, timeout=10)
            r.raise_for_status()
            return (r.json().get("title") or "").strip()
        except Exception:
            return ""

    def _fetch_youtube_transcript(self, vid: str):
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            ts = api.fetch(vid)
            text = self._transcript_to_text(ts)
            if text:
                return text, "youtube-transcript-api"
        except Exception as e:
            print(f"[scraper] youtube-transcript-api failed for {vid}: {type(e).__name__}: {e}")
        text = self._supadata_transcript(vid)
        if text:
            return text, "supadata"
        return None, None

    @staticmethod
    def _transcript_to_text(ts) -> str:
        raw = None
        if hasattr(ts, "snippets"):
            raw = ts.snippets
        elif hasattr(ts, "to_raw_data"):
            raw = ts.to_raw_data()
        elif hasattr(ts, "__iter__"):
            try:
                raw = list(ts)
            except TypeError:
                raw = None
        if not raw:
            return ""
        parts = []
        for s in raw:
            if isinstance(s, dict):
                parts.append(str(s.get("text") or ""))
            else:
                parts.append(str(getattr(s, "text", "") or ""))
        return " ".join(p for p in parts if p)

    @staticmethod
    def _supadata_transcript(vid: str) -> Optional[str]:
        key = os.environ.get("SUPADATA_API_KEY")
        if not key:
            print("[scraper] SUPADATA_API_KEY not set; cannot fall back to supadata")
            return None
        try:
            from supadata import Supadata
            sd = Supadata(api_key=key)
            tr = sd.youtube.transcript(video_id=vid, text=True)
            return (tr.content or "").strip() or None
        except Exception as e:
            print(f"[scraper] supadata fallback failed for {vid}: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------ github

    def _scrape_github(self, url: str) -> ScrapeResult:
        ref = parse_github_url(url)
        if ref is None or ref.kind == "other":
            return self._scrape_web(url)
        if ref.kind in ("issues", "pulls", "discussions", "gist"):
            return self._scrape_web(url)
        file_note = ""
        if ref.kind in ("blob", "raw"):
            file_note = f"file: {'/'.join(x for x in (ref.ref, ref.path) if x)}"
            ref.kind = "repo"
        result = self._scrape_github_repo(url, ref)
        if file_note:
            result.notes.append(file_note)
        return result

    def _scrape_github_repo(self, url: str, ref) -> ScrapeResult:
        result = ScrapeResult(url=url)
        try:
            meta = self._github_repo_meta(ref.owner, ref.repo)
            if not meta:
                web = self._scrape_web(url)
                web.notes.append("github api unavailable (404/rate-limit); falling back to web")
                return web
            readme = self._github_readme(ref.owner, ref.repo)
            if readme is None:
                web = self._scrape_web(url)
                web.notes.append("no README via api; falling back to web")
                return web
            result.status = "scraped"
            result.final_url = url
            result.title = meta.get("full_name") or f"{ref.owner}/{ref.repo}"
            result.raw_text = self._github_context_block(meta, readme)
            result.notes.append("github readme via api")
            return result
        except Exception as e:
            result.status = "blocked"
            result.block_reason = self._classify_exception(e)
            result.notes.append(f"{type(e).__name__}: {e}")
            return result

    def _github_repo_meta(self, owner: str, repo: str) -> Optional[dict]:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
                         headers=self._github_headers_json(), timeout=15)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 403 and int(r.headers.get("X-RateLimit-Remaining", "1")) == 0:
            print("[scraper] github api rate-limited")
        return None

    def _github_readme(self, owner: str, repo: str) -> Optional[str]:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme",
                         headers=self._github_headers_raw(), timeout=15)
        if r.status_code == 200:
            return r.text
        return None

    @staticmethod
    def _github_headers(accept: str) -> dict:
        headers = {"Accept": accept}
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _github_headers_json() -> dict:
        return LinkScraper._github_headers("application/vnd.github+json")

    @staticmethod
    def _github_headers_raw() -> dict:
        return LinkScraper._github_headers("application/vnd.github.raw")

    @staticmethod
    def _github_context_block(meta: dict, readme: str) -> str:
        lines = [f"# {meta.get('full_name') or ''}".strip()]
        desc = (meta.get("description") or "").strip()
        if desc:
            lines.append(desc)
        bits = []
        if meta.get("homepage"):
            bits.append(f"Homepage: {meta['homepage']}")
        topics = [t for t in (meta.get("topics") or []) if t]
        if topics:
            bits.append("Topics: " + ", ".join(topics))
        if meta.get("language"):
            bits.append(f"Language: {meta['language']}")
        if meta.get("stargazers_count") is not None:
            bits.append(f"Stars: {meta['stargazers_count']}")
        lic = meta.get("license")
        if isinstance(lic, dict) and lic.get("spdx_id"):
            bits.append(f"License: {lic['spdx_id']}")
        if bits:
            lines.append(" | ".join(bits))
        lines.append("---")
        text = "\n".join(lines).strip() + "\n\n" + readme
        return text[:GITHUB_RAW_TEXT_LIMIT]
