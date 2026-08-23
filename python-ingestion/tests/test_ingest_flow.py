"""Tests for the unified single-message ingestion graph and API contract."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import enrich
import grader
import ingest
import link_ingest
import llm
import metrics

FAILS = []

CRASH = {"summarizer": False}


class FakeScraper:
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def close(self):
        pass

    def scrape(self, url):
        from link_scraper import ScrapeResult
        if "paywall" in url:
            return ScrapeResult(url=url, status="blocked",
                                block_reason="paywall", raw_text="teaser only")
        if "heavy" in url:
            return ScrapeResult(url=url, status="blocked",
                                block_reason="too_heavy", raw_text="")
        return ScrapeResult(url=url, status="scraped", final_url=url,
                            title="Example Article",
                            raw_text="Ground truth text about vector databases.")


class FakeStore:
    def __init__(self):
        self.messages, self.links, self.pending = [], [], []

    def add_message(self, m):
        self.messages.append(m)

    def add_link(self, l, status="scraped"):
        rec = dict(l)
        rec["status"] = status
        self.links.append(rec)

    def add_pending_link(self, r):
        self.pending.append(r)

    def close(self):
        pass


class FakeVS:
    def __init__(self):
        self.msgs, self.links = [], []

    def upsert_message(self, m):
        self.msgs.append(m)

    def upsert_link(self, l):
        self.links.append(l)

    def close(self):
        pass


def fake_llm(system_prompt, user_content, **kw):
    tag = kw.get("tag")
    if tag == "summarizer" and CRASH["summarizer"]:
        raise RuntimeError("simulated crash")
    if tag == "grader":
        return [{"quality": 5, "confidence": 0.9, "category": "resource",
                 "reason": "fake grade", "topics": ["fake-topic"]}]
    if tag == "enhancer":
        return [{"link_intent": "shares a resource",
                 "entities": ["qdrant"], "topics": ["vector-databases"]}]
    if tag == "summarizer":
        return [{"summary": "A useful summary about vector databases.",
                 "what_it_is": "An article.", "problem_solved": "A problem.",
                 "how_useful": "Very useful.",
                 "entities": ["qdrant"], "topics": ["vector-databases"]}]
    return [{}]


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


def run_all():
    FAILS.clear()
    tmp = tempfile.mkdtemp(prefix="manthan_test_ingest_")
    tmp = Path(tmp)
    originals = {
        "scraped": enrich.SCRAPED_FILE,
        "blocked": enrich.BLOCKED_FILE,
        "ask_user": enrich.ASK_USER_FILE,
        "enriched": enrich.ENRICHED_FILE,
        "graded": enrich.GRADED_FILE,
        "checkpoint": grader.GRADING_CHECKPOINT,
        "checkpoint_db": ingest.CHECKPOINT_DB,
        "metrics_dir": metrics.METRICS_DIR,
        "last_run": metrics.LAST_RUN_FILE,
        "llm": llm.call_completion,
        "scraper": enrich.LinkScraper,
        "link_ask_user": link_ingest.ASK_USER_FILE,
    }
    enrich.SCRAPED_FILE = str(tmp / "scraped.json")
    enrich.BLOCKED_FILE = str(tmp / "blocked.json")
    enrich.ASK_USER_FILE = str(tmp / "ask_user.json")
    enrich.ENRICHED_FILE = str(tmp / "enriched.json")
    enrich.GRADED_FILE = str(tmp / "graded.json")
    grader.GRADING_CHECKPOINT = str(tmp / "grading_checkpoint.json")
    ingest.CHECKPOINT_DB = str(tmp / "state.sqlite")
    metrics.METRICS_DIR = str(tmp / "metrics")
    metrics.LAST_RUN_FILE = str(tmp / "metrics" / "last_run.json")
    llm.call_completion = fake_llm
    enrich.LinkScraper = FakeScraper
    link_ingest.ASK_USER_FILE = enrich.ASK_USER_FILE
    ingest._reset_graph()
    store, vs = FakeStore(), FakeVS()
    try:
        calls = []
        real_fake = fake_llm

        def counting_llm(*a, **kw):
            calls.append(kw.get("tag"))
            return real_fake(*a, **kw)

        llm.call_completion = counting_llm

        print("\n--- untrusted message with link runs the full pipeline ---")
        before = len(calls)
        r1 = ingest.ingest_message("Check this https://example.com/article",
                                   sender="Alice", sent_at="2026-01-01T10:00:00",
                                   store=store, vs=vs)
        new = calls[before:]
        check("graded once via llm", new.count("grader") == 1, new)
        check("message enriched", new.count("enhancer") >= 1)
        check("link scraped + summarized", new.count("summarizer") == 1)
        check("quality 5 and vectored",
              r1["message"]["quality"] == 5 and r1["vectored"])
        check("link outcome scraped with title",
              r1["link_outcomes"][0]["status"] == "scraped"
              and r1["link_outcomes"][0]["title"] == "Example Article")
        check("stored in neo4j-shaped store",
              len(store.messages) == 1 and len(store.links) == 1)
        check("vectorized in qdrant-shaped store",
              len(vs.msgs) == 1 and len(vs.links) == 1)
        check("stats report one scrape", r1["stats"].get("scraped") == 1)
        check("response reports in-graph sync",
              r1["stores"].get("messages") == 1
              and r1["stores"].get("vectored_messages") == 1
              and r1["stores"].get("scraped_links") == 1, r1["stores"])

        print("\n--- message without links bypasses scraping ---")
        before = len(calls)
        r2 = ingest.ingest_message("Pure text insight about retrieval",
                                   sender="Bob", sent_at="2026-01-01T10:05:00",
                                   store=store, vs=vs)
        new = calls[before:]
        check("graded but never summarized",
              new.count("grader") == 1 and "summarizer" not in new, new)
        check("enhancer metadata fetched for linkless",
              new.count("enhancer") == 1, new)
        check("no link outcomes reported", r2["link_outcomes"] == [])
        check("no_links stat counts candidate",
              r2["stats"].get("no_links") == 1, r2["stats"])
        check("linkless message stored and vectorized",
              len(store.messages) == 2 and len(vs.msgs) == 2
              and r2["vectored"] is True
              and r2["message"]["topics"] == ["vector-databases"],
              f"stored={len(store.messages)} vec={len(vs.msgs)} "
              f"vectored={r2['vectored']}")

        print("\n--- trusted messages skip grading but still scrape ---")
        graders_before = sum(1 for t in calls if t == "grader")
        r3 = ingest.ingest_message("Trusted note https://example.com/trusted",
                                   sender="Cara", sent_at="2026-01-01T10:10:00",
                                   trusted=True, store=store, vs=vs)
        graders_after = sum(1 for t in calls if t == "grader")
        check("zero grader calls when trusted",
              graders_after == graders_before)
        check("trusted quality 5 flagged",
              r3["message"]["quality"] == 5 and r3["message"]["trusted"])
        check("trusted link still scraped",
              r3["link_outcomes"][0]["status"] == "scraped")
        check("trusted flag stamped on scraped-path row",
              store.messages[-1].get("trusted") is True
              and vs.msgs[-1].get("trusted") is True)

        print("\n--- paywall links land in ask-user pending paste ---")
        r4 = ingest.ingest_message("Read https://example.com/paywall-article",
                                   sender="Dan", sent_at="2026-01-01T10:15:00",
                                   store=store, vs=vs)
        out4 = r4["link_outcomes"][0]
        check("ask_user outcome with reason",
              out4["status"] == "ask_user" and out4["block_reason"] == "paywall",
              out4)
        check("pending link pushed to store",
              len(store.pending) == 1
              and store.pending[0]["url"] == "https://example.com/paywall-article")

        print("\n--- hard-blocked links sync with their reason ---")
        r5 = ingest.ingest_message("Big page https://example.com/heavy-page",
                                   sender="Hank", sent_at="2026-01-01T10:18:00",
                                   store=store, vs=vs)
        out5 = r5["link_outcomes"][0]
        check("blocked outcome with reason",
              out5["status"] == "blocked"
              and out5["block_reason"] == "too_heavy", out5)
        check("summary counts one blocked link",
              r5["stores"].get("blocked_links") == 1, r5["stores"])
        check("blocked record stored with status and reason",
              any(l.get("status") == "blocked"
                  and l.get("block_reason") == "too_heavy"
                  for l in store.links), store.links[-1])

        print("\n--- blocked links expose stable paste-ready ids ---")
        from store import link_id
        heavy_url = "https://example.com/heavy-page"
        check("derived id is deterministic",
              link_id(heavy_url) == link_id(heavy_url)
              and link_id(heavy_url).startswith("LNK-"), link_id(heavy_url))
        try:
            from query_worker import format_blocked_reply
            reply = format_blocked_reply({
                "count": 1,
                "links": [{"url": heavy_url, "status": "blocked",
                           "block_reason": "too_heavy",
                           "link_id": link_id(heavy_url)}],
            })
            check("reply renders awaiting_link_content blocks",
                  '"action": "awaiting_link_content"' in reply
                  and f'"link_id": "{link_id(heavy_url)}"' in reply
                  and heavy_url in reply, reply.splitlines()[0:4])
            check("empty list still renders friendly note",
                  format_blocked_reply({"count": 0, "links": []})
                  == "No blocked links 🎉")
        except Exception as exc:
            check("formatter import skipped without env", False, str(exc))
        try:
            from ingest_worker import format_paste_reply
            ok = format_paste_reply({
                "ok": True, "url": heavy_url, "title": "Example Article",
                "topics": ["vector-databases"], "summary": "Short summary.",
            })
            check("paste reply confirms storage and topics",
                  ok.startswith("Pasted successfully") and heavy_url in ok
                  and "Topics: vector-databases" in ok
                  and "Summary: Short summary." in ok, ok)
            bad = format_paste_reply({"ok": False, "error": "unknown link id"})
            check("paste reply surfaces failures",
                  bad.startswith("Paste failed — unknown link id"), bad)
        except Exception as exc:
            check("paste formatter import skipped without env", False, str(exc))
        try:
            from query_worker import format_skip_reply, WELCOME
            check("welcome lists every v1 command",
                  all(cmd in WELCOME for cmd in
                      ("/ingest", "/via", "/blocked", "/paste", "/skip", "/ask")),
                  WELCOME.splitlines()[0])
            good = format_skip_reply(
                {"ok": True, "url": heavy_url, "status": "skipped"})
            check("skip reply confirms with url",
                  good.startswith("Skipped ✓") and heavy_url in good, good)
            bad = format_skip_reply(
                {"ok": False, "error": f"unknown link id: LNK-000000"})
            check("skip reply surfaces failures",
                  bad.startswith("Skip failed — unknown link id"), bad)
        except Exception as exc:
            check("skip formatter import skipped without env", False, str(exc))

        print("\n--- identical retry reuses the completed checkpoint ---")
        total_before = len(calls)
        r1b = ingest.ingest_message("Check this https://example.com/article",
                                    sender="Alice", sent_at="2026-01-01T10:00:00",
                                    store=store, vs=vs)
        check("no llm calls on completed-thread retry",
              len(calls) == total_before, f"{total_before} -> {len(calls)}")
        check("replayed result matches original",
              r1b["message"]["quality"] == r1["message"]["quality"]
              and r1b["link_outcomes"] == r1["link_outcomes"])

        print("\n--- mid-graph crash resumes without re-grading ---")
        s7 = tmp / "scenario7"
        s7.mkdir()
        enrich.SCRAPED_FILE = str(s7 / "scraped.json")
        enrich.BLOCKED_FILE = str(s7 / "blocked.json")
        enrich.ASK_USER_FILE = str(s7 / "ask_user.json")
        enrich.ENRICHED_FILE = str(s7 / "enriched.json")
        enrich.GRADED_FILE = str(s7 / "graded.json")
        link_ingest.ASK_USER_FILE = enrich.ASK_USER_FILE
        store7, vs7 = FakeStore(), FakeVS()
        args7 = dict(sender="Fay", sent_at="2026-01-01T10:25:00",
                     text="Crash test https://example.com/crash")
        graders_before = sum(1 for t in calls if t == "grader")
        CRASH["summarizer"] = True
        raised = False
        try:
            ingest.ingest_message(args7["text"], store=store7, vs=vs7,
                                  sender=args7["sender"],
                                  sent_at=args7["sent_at"])
        except RuntimeError as exc:
            raised = "simulated crash" in str(exc)
        CRASH["summarizer"] = False
        check("crash propagates out of ingest_message", raised)
        check("nothing stored when graph crashed",
              len(store7.messages) == 0 and len(vs7.msgs) == 0)
        graders_mid = sum(1 for t in calls if t == "grader")
        check("grade_pass1 checkpointed before crash",
              graders_mid == graders_before + 1,
              f"{graders_before} -> {graders_mid}")
        r7 = ingest.ingest_message(args7["text"], store=store7, vs=vs7,
                                   sender=args7["sender"],
                                   sent_at=args7["sent_at"])
        graders_final = sum(1 for t in calls if t == "grader")
        check("resume does not re-grade (invoke(None))",
              graders_final == graders_mid,
              f"{graders_mid} -> {graders_final}")
        check("completes after resume",
              r7["link_outcomes"][0]["status"] == "scraped"
              and r7["message"]["quality"] == 5)
        check("stores exactly once after resume",
              len(store7.messages) == 1 and len(vs7.msgs) == 1,
              f"msgs={len(store7.messages)} vec={len(vs7.msgs)}")

        print("\n--- crash inside sync_stores resumes without re-running pipeline ---")
        s8 = tmp / "scenario8"
        s8.mkdir()
        enrich.SCRAPED_FILE = str(s8 / "scraped.json")
        enrich.BLOCKED_FILE = str(s8 / "blocked.json")
        enrich.ASK_USER_FILE = str(s8 / "ask_user.json")
        enrich.ENRICHED_FILE = str(s8 / "enriched.json")
        enrich.GRADED_FILE = str(s8 / "graded.json")
        link_ingest.ASK_USER_FILE = enrich.ASK_USER_FILE

        class FlakyVS(FakeVS):
            def __init__(self):
                super().__init__()
                self.fail_first = True

            def upsert_message(self, m):
                if self.fail_first:
                    self.fail_first = False
                    raise RuntimeError("simulated sync crash")
                super().upsert_message(m)

        store8, vs8 = FakeStore(), FlakyVS()
        args8 = dict(sender="Gus", sent_at="2026-01-01T10:30:00",
                     text="Sync crash test https://example.com/sync")
        graders_before = sum(1 for t in calls if t == "grader")
        summarizers_before = sum(1 for t in calls if t == "summarizer")
        raised = False
        try:
            ingest.ingest_message(args8["text"], sender=args8["sender"],
                                  sent_at=args8["sent_at"],
                                  store=store8, vs=vs8)
        except RuntimeError as exc:
            raised = "simulated sync crash" in str(exc)
        check("sync crash propagates out of ingest_message", raised)
        check("neo4j write landed before qdrant crashed",
              len(store8.messages) == 1 and len(vs8.msgs) == 0,
              f"msgs={len(store8.messages)} vec={len(vs8.msgs)}")
        r8 = ingest.ingest_message(args8["text"], sender=args8["sender"],
                                   sent_at=args8["sent_at"],
                                   store=store8, vs=vs8)
        graders_after = sum(1 for t in calls if t == "grader")
        summarizers_after = sum(1 for t in calls if t == "summarizer")
        check("resume does not re-grade or re-summarize",
              graders_after == graders_before + 1
              and summarizers_after == summarizers_before + 1,
              f"graders {graders_before}->{graders_after} "
              f"summarizers {summarizers_before}->{summarizers_after}")
        check("sync completed on resume",
              len(vs8.msgs) == 1
              and r8["stores"].get("vectored_messages") == 1, r8["stores"])

        print("\n--- mark_resolved syncs ask-user records ---")
        enrich.SCRAPED_FILE = str(tmp / "scraped.json")
        enrich.BLOCKED_FILE = str(tmp / "blocked.json")
        enrich.ASK_USER_FILE = str(tmp / "ask_user.json")
        enrich.ENRICHED_FILE = str(tmp / "enriched.json")
        enrich.GRADED_FILE = str(tmp / "graded.json")
        link_ingest.ASK_USER_FILE = enrich.ASK_USER_FILE
        hits = link_ingest.mark_resolved("https://example.com/paywall-article")
        records = [r for r in enrich.load_list(enrich.ASK_USER_FILE)
                   if r.get("url") == "https://example.com/paywall-article"]
        check("record flipped to resolved",
              hits == 1 and records and records[0]["resolved"] is True
              and records[0].get("ingested_at"))
        check("second mark is a no-op",
              link_ingest.mark_resolved("https://example.com/paywall-article") == 0)

        print(f"\nRESULT: {'FAILED' if FAILS else 'ALL PASS'} "
              f"({len(FAILS)} failures)")
        return FAILS
    finally:
        enrich.SCRAPED_FILE = originals["scraped"]
        enrich.BLOCKED_FILE = originals["blocked"]
        enrich.ASK_USER_FILE = originals["ask_user"]
        enrich.ENRICHED_FILE = originals["enriched"]
        enrich.GRADED_FILE = originals["graded"]
        grader.GRADING_CHECKPOINT = originals["checkpoint"]
        ingest.CHECKPOINT_DB = originals["checkpoint_db"]
        metrics.METRICS_DIR = originals["metrics_dir"]
        metrics.LAST_RUN_FILE = originals["last_run"]
        llm.call_completion = originals["llm"]
        enrich.LinkScraper = originals["scraper"]
        ingest._reset_graph()


def test_all():
    fails = run_all()
    assert not fails, f"failed checks: {fails}"


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)