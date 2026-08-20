import re
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm
import enrich
import grader
import main
from link_scraper import ScrapeResult

FAILS = []


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


def fake_llm(system_prompt, user_content, **kw):
    tag = kw.get("tag")
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
        if "meet.google.invalid" in url:
            return ScrapeResult(url=url, status="blocked", block_reason="truncated",
                                raw_text="teaser only, still collapsed")
        return ScrapeResult(url=url, status="blocked", block_reason="auth_required")


llm.call_completion = fake_llm
enrich.LinkScraper = FakeScraper

tmp = Path(tempfile.mkdtemp(prefix="manthan_pipe_"))
main.CHECKPOINT_DB = str(tmp / "state.sqlite")
grader.GRADING_CHECKPOINT = str(tmp / "grading_checkpoint.json")
enrich.GRADED_FILE = str(tmp / "graded.json")
enrich.SCRAPED_FILE = str(tmp / "scraped.json")
enrich.BLOCKED_FILE = str(tmp / "blocked.json")
enrich.ASK_USER_FILE = str(tmp / "ask_user.json")
enrich.ENRICHED_FILE = str(tmp / "enriched.json")

result = main.run_pipeline("tests/test_chat.txt")
persisted_enriched = enrich.load_list(enrich.ENRICHED_FILE)

check("graph scraped 4", result["stats"]["scraped"] == 4, result["stats"])
check("graph blocked 1 (not_found)", result["stats"]["blocked"] == 1, result["stats"])
check("graph ask_user 4 (3 auth_required + 1 truncated)",
      result["stats"]["ask_user"] == 4, result["stats"])
check("graph enriched 8", len(result["enriched"]) == 8, len(result["enriched"]))
check("graph enriched includes one preview per link",
      all(len(r.get("link_previews", [])) == len(r["links"]) for r in result["enriched"]),
      result["enriched"])
check("persisted enriched includes one preview per link",
      all(len(r.get("link_previews", [])) == len(r["links"]) for r in persisted_enriched),
      persisted_enriched)
graph_article = next(
    r for r in persisted_enriched if "https://example.com/article" in r["links"]
)
check("persisted preview includes scraped summary",
      graph_article.get("link_previews") == [{
          "url": "https://example.com/article",
          "status": "scraped",
          "final_url": "https://example.com/article",
          "title": "Example Article",
          "summary": "A useful technical summary about vector databases.",
          "what_it_is": "",
          "problem_solved": "",
          "how_useful": "",
          "source": "auto",
          "topics": ["vector-databases"],
          "entities": ["qdrant"],
      }],
      graph_article.get("link_previews"))
check("graph final merged persisted",
      len(enrich.load_list(enrich.GRADED_FILE)) == len(result["final"]),
      f"{len(result['final'])} final records")
check("ask_user json written", len(enrich.load_list(enrich.ASK_USER_FILE)) == 4)
check("scraped json written", len(enrich.load_list(enrich.SCRAPED_FILE)) == 4)
check("blocked json written", len(enrich.load_list(enrich.BLOCKED_FILE)) == 1)

stale_enriched = [
    {k: v for k, v in row.items() if k != "link_previews"}
    for row in result["enriched"]
]
main.persist_node({
    "final": result["final"],
    "scraped": result["scraped"],
    "blocked": result["blocked"],
    "ask_user": result["ask_user"],
    "enriched": stale_enriched,
    "stats": result["stats"],
})
repaired_enriched = enrich.load_list(enrich.ENRICHED_FILE)
check("persist_node backfills stale previewless rows",
      all(len(r.get("link_previews", [])) == len(r["links"]) for r in repaired_enriched),
      repaired_enriched)

# parallel execution: both scrape_candidates and grade_pass2 ran (pass2 only
# re-grades borderline quality=3 messages, a strict subset of pass 1)
check("pass2 ran in graph (subset of graded)",
      0 < len(result["verified"]) < len(result["graded"]),
      f"{len(result['verified'])}/{len(result['graded'])}")

# ---- promotion path unit test ----
d = SimpleNamespace(datetime_iso="2026-08-14T22:00:00", sender="Alice")
g3 = SimpleNamespace(quality=3, confidence=0.8, category="discussion",
                     reason="", topics=[], original_text="check this https://promo.example/x")
g5 = SimpleNamespace(quality=5, confidence=0.9, category="resource",
                     reason="upgraded", topics=[], original_text="check this https://promo.example/x")
state = {"kept": [d], "graded": [g3], "verified": [g5]}
out = main.merge_node(state)
check("merge promotes 3->5", len(out["promoted"]) == 1 and out["promoted"][0][1].quality == 5,
      out["promoted"])
state2 = {"kept": [d], "graded": [g5], "verified": [g5]}
out2 = main.merge_node(state2)
check("merge does not promote pass-1 >=4", len(out2["promoted"]) == 0, out2["promoted"])

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
