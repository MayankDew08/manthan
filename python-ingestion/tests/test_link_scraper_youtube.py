import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import link_scraper
from link_scraper import LinkScraper

FAILS = []


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


class FakeSentence:
    text = "Hello from the official captions."


class FakeTranscript:
    snippets = [FakeSentence()]


def make_fake_yt(raise_exc):
    mod = types.ModuleType("youtube_transcript_api")
    api_cls = type("YouTubeTranscriptApi", (), {})
    if raise_exc:
        def fetch(self, vid):
            raise Exception("TranscriptsDisabled")
        api_cls.fetch = fetch
    else:
        def fetch(self, vid):
            return FakeTranscript()
        api_cls.fetch = fetch
    mod.YouTubeTranscriptApi = api_cls
    return mod


def run(vid, yt_ok, supadata_text):
    scraper = LinkScraper(delay=0)
    sys.modules["youtube_transcript_api"] = make_fake_yt(not yt_ok)
    def fake_supadata(self, vid):
        return supadata_text
    scraper._supadata_transcript = types.MethodType(fake_supadata, scraper)
    try:
        return scraper._scrape_youtube(f"https://www.youtube.com/watch?v={vid}")
    finally:
        scraper.close()


def run_all():
    FAILS.clear()
    r = run("abc123", yt_ok=True, supadata_text=None)
    check("primary transcript wins", r.status == "scraped" and "official captions" in r.raw_text
          and any("youtube-transcript-api" in n for n in r.notes), r)

    r = run("abc123", yt_ok=False, supadata_text="Supadata generated transcript text.")
    check("supadata fallback fires", r.status == "scraped" and "Supadata generated" in r.raw_text
          and any("supadata" in n for n in r.notes), r)

    r = run("abc123", yt_ok=False, supadata_text=None)
    check("both fail -> blocked", r.status == "blocked" and r.block_reason == "transcript_unavailable",
          f"{r.status} {r.block_reason}")

    r = LinkScraper(delay=0)._scrape_youtube("https://youtube.com/watch?v=")
    check("bad url -> invalid", r.status == "blocked" and r.block_reason == "invalid_youtube_url",
          f"{r.status} {r.block_reason}")

    print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
    return FAILS


def test_all():
    fails = run_all()
    assert not fails, f"failed checks: {fails}"


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)