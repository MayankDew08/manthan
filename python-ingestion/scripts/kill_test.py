import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import MIN_SCORE, search

QUESTIONS = [
    "what is qdrant?",
    "tool to manage coding agents",
    "state of AI report",
]

TOP_K = 5

NOTES = (
    "Q1 diagnosis A (diluted link embedding): the qdrant/qdrant link scores 0.283 "
    "vs the referencing message 0.432 for 'what is qdrant?'. The link embedding is "
    "a 174-word blob (summary 861 + what_it_is 191 + problem_solved 280 chars) while "
    "the message is 110 chars / 15 words, so the short query matches the topic-dense "
    "message better. Dedupe suppresses the standalone link when the carrying message "
    "outranks it."
)


def main():
    artifact = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": "hybrid",
        "min_score": MIN_SCORE,
        "top_k": TOP_K,
        "notes": NOTES,
        "questions": [],
    }
    for q in QUESTIONS:
        results = search(q, top_k=TOP_K)
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        if not results:
            print("  (no results)")
        for r in results:
            kind = f"[{r['type']}]".ljust(8)
            title = r.get("title") or r.get("text", "")[:60]
            print(f"  {kind} {r['score']:.3f} {r['source']:<9} "
                  f"{r.get('sender', '')} | {r.get('sent_at', '')[:16]} | {title[:70]}")
            if r.get("related_links"):
                for lk in r["related_links"][:2]:
                    print(f"        -> link: {lk['title'] or lk['url']}")
        artifact["questions"].append({"question": q, "results": results})

    evals_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals")
    os.makedirs(evals_dir, exist_ok=True)
    path = os.path.join(evals_dir, "kill_test_baseline.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved eval artifact -> {path}")


if __name__ == "__main__":
    main()