import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm
import enrich
import grader
from link_scraper import ScrapeResult

FAILS = []
LLM_TAGS = []


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


def fake_llm(system_prompt, user_content, **kw):
    tag = kw.get("tag")
    LLM_TAGS.append(tag)
    if tag == "grader":
        numbered = re.findall(r"^(\d+)\)", user_content, re.M)
        blocks = re.findall(r"^MESSAGE \d+:", user_content, re.M)
        n = len(numbered) or len(blocks) or 1
        grades = []
        for line in user_content.splitlines():
            m = re.match(r"^(\d+)\) (.*)$", line)
            if m:
                text = m.group(2)
                q = 5 if "http" in text else 3
                grades.append({"quality": q, "confidence": 0.9,
                               "category": "resource" if q == 5 else "discussion",
                               "reason": "fake", "topics": ["fake"]})
        if len(grades) != n:
            grades = [{"quality": 3, "confidence": 0.9, "category": "discussion",
                       "reason": "fake", "topics": ["fake"]}] * n
        return grades
    if tag == "enhancer":
        return [{"link_intent": "Shared a technical resource for the group.",
                 "entities": ["qdrant"], "topics": ["vector-databases"]}]
    if tag == "repo":
        return [{"summary": "Repo overview: a vector database.",
                 "what_it_is": "A Rust vector database.",
                 "problem_solved": "Efficient similarity search at scale.",
                 "how_useful": "Use as a RAG backend.",
                 "entities": ["qdrant"], "topics": ["vector-databases"]}]
    if tag == "summarizer":
        return [{"summary": "A useful technical summary about vector databases.",
                 "what_it_is": "A technical article about vector databases.",
                 "problem_solved": "It explains a vector-search implementation problem.",
                 "how_useful": "Use it to evaluate a vector database approach.",
                 "entities": ["qdrant"], "topics": ["vector-databases"]}]
    return [{}]


class FakeScraper:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def close(self):
        pass

    def scrape(self, url):
        if "example.com" in url:
            return ScrapeResult(url=url, status="scraped", final_url=url,
                                title="Example Article",
                                raw_text="Ground truth text about vector databases.")
        if "github.com" in url:
            return ScrapeResult(url=url, status="scraped", final_url=url,
                                title="owner/repo",
                                raw_text="# Repo README about a vector database.")
        if "producthunt" in url:
            return ScrapeResult(url=url, status="blocked", block_reason="not_found")
        return ScrapeResult(url=url, status="blocked", block_reason="auth_required")


llm.call_completion = fake_llm
enrich.LinkScraper = FakeScraper

tmp = Path(tempfile.mkdtemp(prefix="manthan_e2e_"))
grader.GRADING_CHECKPOINT = str(tmp / "grading_checkpoint.json")
enrich.GRADED_FILE = str(tmp / "graded.json")
enrich.SCRAPED_FILE = str(tmp / "scraped.json")
enrich.BLOCKED_FILE = str(tmp / "blocked.json")
enrich.ASK_USER_FILE = str(tmp / "ask_user.json")
enrich.ENRICHED_FILE = str(tmp / "enriched.json")

enrich.enrich_chat("tests/test_chat.txt", min_quality=4)

scraped = enrich.load_list(enrich.SCRAPED_FILE)
blocked = enrich.load_list(enrich.BLOCKED_FILE)
ask_user = enrich.load_list(enrich.ASK_USER_FILE)
enriched = enrich.load_list(enrich.ENRICHED_FILE)
graded = enrich.load_list(enrich.GRADED_FILE)

check("one article scraped (example.com)", len(scraped) == 4
      and any(s["url"] == "https://example.com/article" for s in scraped),
      [s["url"] for s in scraped])
check("summary + raw_text stored", scraped and scraped[0]["summary"]
      and scraped[0]["raw_text"])
check("github links scraped with repo overview",
      sum(1 for s in scraped if "github.com" in s["url"]) == 3)
gh = [s for s in scraped if "github.com" in s["url"]]
check("github entries have problem_solved/how_useful",
      all(s["what_it_is"] and s["problem_solved"] and s["how_useful"] for s in gh),
      gh)
check("blocked = 1 (not_found)", len(blocked) == 1
      and blocked[0]["block_reason"] == "not_found", [b["url"] for b in blocked])
check("ask_user = 4 auth_required links", len(ask_user) == 4
      and all(a["block_reason"] == "auth_required" for a in ask_user),
      [a["url"] for a in ask_user])
check("ask_user carries message context", ask_user and ask_user[0]["message_context"]["link_intent"])
check("enriched has 8 records", len(enriched) == 8 and all(r["quality"] == 5 for r in enriched))
check("enriched has link_intent", all(r["link_intent"] for r in enriched))
check("enriched exposes one preview per link",
      all(len(r.get("link_previews", [])) == len(r["links"]) for r in enriched))
article_message = next(
    r for r in enriched if "https://example.com/article" in r["links"]
)
check("scraped preview includes instant summary and parsed data",
      article_message.get("link_previews") == [{
          "url": "https://example.com/article",
          "status": "scraped",
          "final_url": "https://example.com/article",
          "title": "Example Article",
          "summary": "A useful technical summary about vector databases.",
          "what_it_is": "A technical article about vector databases.",
          "problem_solved": "It explains a vector-search implementation problem.",
          "how_useful": "Use it to evaluate a vector database approach.",
          "source": "auto",
          "topics": ["vector-databases"],
          "entities": ["qdrant"],
      }], article_message.get("link_previews"))
blocked_message = next(
    r for r in enriched
    if "https://www.producthunt.com/posts/agent-orchestrator" in r["links"]
)
check("blocked preview includes actionable status",
      blocked_message.get("link_previews") == [{
          "url": "https://www.producthunt.com/posts/agent-orchestrator",
          "status": "blocked",
          "block_reason": "not_found",
      }], blocked_message.get("link_previews"))
check("graded persisted", isinstance(graded, list) and len(graded) > 0,
      f"{len(graded)} graded records")

print("\n--- second run (incremental) ---")
enrich.enrich_chat("tests/test_chat.txt", min_quality=4)
scraped2 = enrich.load_list(enrich.SCRAPED_FILE)
blocked2 = enrich.load_list(enrich.BLOCKED_FILE)
ask_user2 = enrich.load_list(enrich.ASK_USER_FILE)
check("idempotent: no duplicates", len(scraped2) == 4 and len(blocked2) == 1
      and len(ask_user2) == 4,
      f"scraped={len(scraped2)} blocked={len(blocked2)} ask_user={len(ask_user2)}")

print("\n--- third run (scrape-only force rebuild) ---")
# Force-refreshing this chat must not delete cumulative records owned by another
# chat/run.
historical_scraped = {"url": "https://history.example/ok", "summary": "keep me"}
historical_blocked = {"url": "https://history.example/blocked", "block_reason": "failed"}
historical_ask_user = {"url": "https://history.example/auth", "block_reason": "auth_required",
                       "resolved": True}
historical_enriched = {
    "sent_at": "2020-01-01T00:00:00",
    "sender": "Historical Sender",
    # Identical text is valid across different chats. Sender + timestamp own the
    # row, so this must survive a force refresh of the current chat.
    "original_text": enriched[0]["original_text"],
    "links": ["https://github.com/qdrant/qdrant"],
    "link_previews": [{
        "url": "https://github.com/qdrant/qdrant",
        "status": "blocked",
        "block_reason": "historical_failure",
    }],
}
enrich.save_list(enrich.SCRAPED_FILE, scraped2 + [historical_scraped])
enrich.save_list(enrich.BLOCKED_FILE, blocked2 + [historical_blocked])
enrich.save_list(enrich.ASK_USER_FILE, ask_user2 + [historical_ask_user])
enrich.save_list(enrich.ENRICHED_FILE, enriched + [historical_enriched])
grader_calls_before = LLM_TAGS.count("grader")
enrich.enrich_chat("tests/test_chat.txt", min_quality=4,
                   force=True, scrape_only=True)
scraped3 = enrich.load_list(enrich.SCRAPED_FILE)
blocked3 = enrich.load_list(enrich.BLOCKED_FILE)
ask_user3 = enrich.load_list(enrich.ASK_USER_FILE)
enriched3 = enrich.load_list(enrich.ENRICHED_FILE)
check("scrape-only makes no grader calls",
      LLM_TAGS.count("grader") == grader_calls_before,
      f"before={grader_calls_before} after={LLM_TAGS.count('grader')}")
check("force rebuilds rather than skipping existing URLs",
      len(scraped3) == 5 and len(blocked3) == 2 and len(ask_user3) == 5,
      f"scraped={len(scraped3)} blocked={len(blocked3)} ask_user={len(ask_user3)}")
check("force rebuild keeps summaries",
      all(x.get("summary") for x in scraped3))
historical_enriched_preserved = any(
    r.get("sent_at") == historical_enriched["sent_at"]
    and r.get("sender") == historical_enriched["sender"]
    and r.get("original_text") == historical_enriched["original_text"]
    and r.get("link_previews") == historical_enriched["link_previews"]
    for r in enriched3
)
check("force preserves unrelated cumulative records",
      historical_scraped in scraped3 and historical_blocked in blocked3
      and historical_ask_user in ask_user3 and historical_enriched_preserved)
check("force rebuild keeps enriched messages", len(enriched3) == 9)

print("\n--- fourth run (force + no-scrape is non-destructive) ---")
outcomes_before = (scraped3, blocked3, ask_user3)
enrich.enrich_chat("tests/test_chat.txt", min_quality=4, force=True,
                   scrape_only=True, no_scrape=True)
outcomes_after = (
    enrich.load_list(enrich.SCRAPED_FILE),
    enrich.load_list(enrich.BLOCKED_FILE),
    enrich.load_list(enrich.ASK_USER_FILE),
)
check("force + no-scrape preserves all scrape outcomes",
      outcomes_after == outcomes_before)

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
