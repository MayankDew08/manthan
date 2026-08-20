import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm
import enhancer

FAILS = []


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


calls = []


def fake_llm(system_prompt, user_content, **kw):
    calls.append((kw.get("tag"), system_prompt, user_content))
    if kw.get("tag") == "enhancer":
        return [{"link_intent": "Explains why vector DB choice matters at scale.",
                 "entities": ["qdrant", "pinecone"], "topics": ["vector-databases"]}]
    if kw.get("tag") == "repo":
        return [{"summary": "Qdrant is a vector similarity search engine.",
                 "what_it_is": "A Rust vector database.",
                 "problem_solved": "Efficient nearest-neighbor search at scale.",
                 "how_useful": "Use for RAG and similarity search backends.",
                 "entities": ["qdrant"], "topics": ["vector-databases"]}]
    return [{"summary": "A concise technical summary about local LLM quantization.",
             "what_it_is": "A technical article about local inference.",
             "problem_solved": "Choosing effective model quantization settings.",
             "how_useful": "Use it to configure a local inference stack.",
             "entities": ["llama.cpp", "q8_0"], "topics": ["local-llm", "quantization"]}]


llm.call_completion = fake_llm

r = enhancer.enrich_message("Check this out https://example.com/vector-db", ["https://example.com/vector-db"])
check("enrich_message link_intent", r["link_intent"] and "vector" in r["link_intent"], r)
check("enrich_message entities", r["entities"] == ["qdrant", "pinecone"], r["entities"])
check("enrich_message topics", r["topics"] == ["vector-databases"], r["topics"])
check("enrich tag used", calls[-1][0] == "enhancer")

r = enhancer.summarize_content("https://example.com/a", "Title", "Some article text")
check("summarize_content summary", "quantization" in r["summary"], r)
check("summarize_content has no empty required fields",
      all(r[field] for field in ("summary", "what_it_is", "problem_solved", "how_useful")), r)
check("summarize_content entities", r["entities"] == ["llama.cpp", "q8_0"], r["entities"])

big = ("alpha beta gamma delta. " * 4000) + "\n" + ("epsilon zeta eta theta. " * 4000)
calls.clear()
r = enhancer.summarize_content("https://example.com/big", "Big", big)
n_sum = sum(1 for c in calls if c[0] == "summarizer")
check("map-reduce used for huge content", n_sum > 2, f"{n_sum} summarizer calls")
check("map-reduce produced summary", bool(r["summary"]), f"{len(r['summary'])} chars")

calls.clear()
r = enhancer.summarize_repo("https://github.com/qdrant/qdrant", "qdrant/qdrant",
                            "# Qdrant\nA vector database.")
check("repo overview summary", "vector similarity search" in r["summary"], r)
check("repo what_it_is", r["what_it_is"] == "A Rust vector database.", r["what_it_is"])
check("repo problem_solved", "nearest-neighbor" in r["problem_solved"])
check("repo how_useful", r["how_useful"] == "Use for RAG and similarity search backends.")
check("repo entities/topics", r["entities"] == ["qdrant"] and r["topics"] == ["vector-databases"])
check("repo tag used", calls[-1][0] == "repo")

calls.clear()
r = enhancer.summarize_repo("https://github.com/qdrant/qdrant", "qdrant/qdrant", big)
n_repo = sum(1 for c in calls if c[0] == "repo")
check("repo map-reduce for huge README", n_repo > 2, f"{n_repo} repo calls")
check("repo map-reduce keeps fields", bool(r["what_it_is"]) and bool(r["problem_solved"]))

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
