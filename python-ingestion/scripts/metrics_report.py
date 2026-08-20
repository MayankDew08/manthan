"""Generate a privacy-safe Markdown report from local run artifacts."""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from parser import parse_chat
from heuristic_filter import heuristic_filter
import metrics

PUBLIC_SOURCE_LABEL = "Private chat export (redacted)"


def _est_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _load(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _source_summary(message_count: int,
                    source_label: str = PUBLIC_SOURCE_LABEL) -> str:
    """Format a public label without exposing a path or breaking Markdown."""
    safe_label = " ".join(source_label.split()).replace("`", "'")
    return f"- **Source:** {safe_label} — **{message_count}** messages parsed"


def _storage() -> dict:
    """Collect best-effort live storage counts without aborting the report."""
    out = {"neo4j": {}, "qdrant": {}}
    try:
        from store import KnowledgeStore
        store = KnowledgeStore()
        with store.driver.session() as session:
            for row in session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c"
            ).data():
                out["neo4j"][row["label"]] = row["c"]
            out["neo4j"]["rels"] = session.run(
                "MATCH ()-[r]->() RETURN count(*) AS c"
            ).single()["c"]
        store.close()
    except Exception as e:
        out["neo4j"]["error"] = f"{type(e).__name__}"
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        vs.ensure_collection()
        info = vs.client.get_collection(vs.collection)
        out["qdrant"]["points"] = info.points_count
        out["qdrant"]["vectors"] = info.config.params.vectors.size
        vs.close()
    except Exception as e:
        out["qdrant"]["error"] = f"{type(e).__name__}"
    return out


def build_report(chat_path: str,
                 chat_label: str = PUBLIC_SOURCE_LABEL) -> str:
    """Render aggregate pipeline, LLM, link, and storage metrics as Markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Manthan metrics report", f"", f"_generated {now}_", ""]

    msgs = parse_chat(chat_path)
    kept, discarded = heuristic_filter(msgs)
    drop = len(discarded) / max(1, len(msgs)) * 100
    lines += [
        "## 1. Pipeline funnel",
        "",
        _source_summary(len(msgs), chat_label),
        f"- **Kept → grading:** {len(kept)}",
        f"- **Dropped (heuristic):** {len(discarded)} (**drop {drop:.1f}%**)",
        "",
    ]
    if discarded:
        lines += ["| drop reason | count | share |", "|---|---|---|"]
        reasons = {}
        for d in discarded:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {r} | {c} | {c / len(discarded) * 100:.0f}% |")
        lines.append("")

    graded = _load("graded_messages.json")
    lines += ["## 2. Grading", ""]
    if graded:
        ql = {}
        for g in graded:
            ql[g["quality"]] = ql.get(g["quality"], 0) + 1
        lines += ["**Quality distribution (1–5):**"]
        for q in range(1, 6):
            n = ql.get(q, 0)
            lines.append(f"- q{q}: **{n}** `{'#' * n}`")
        conf = [g["confidence"] for g in graded]
        lines += [
            f"- confidence: mean **{statistics.mean(conf):.3f}** "
            f"(min {min(conf):.2f}, max {max(conf):.2f})",
            f"- avg topics/message: **{sum(len(g['topics']) for g in graded) / len(graded):.1f}**",
            "",
            "**Categories:**",
        ]
        cats = {}
        for g in graded:
            cats[g["category"]] = cats.get(g["category"], 0) + 1
        for c, n in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"- {c}: {n}")
    else:
        lines.append("_no graded_messages.json_\n")
    lines.append("")

    enriched = _load("enriched_messages.json")
    lines += ["## 3. Enrichment", ""]
    if enriched:
        n_links = sum(len(e.get("links", [])) for e in enriched)
        n_ent = sum(len(e.get("entities", [])) for e in enriched)
        n_top = sum(len(e.get("topics", [])) for e in enriched)
        n_int = sum(1 for e in enriched if e.get("link_intent"))
        previews = {}
        for e in enriched:
            for p in e.get("link_previews", []):
                previews[p.get("status")] = previews.get(p.get("status"), 0) + 1
        lines += [
            f"- records: **{len(enriched)}** ({sum(1 for e in enriched if e.get('links'))} with links)",
            f"- total links: **{n_links}** (avg {n_links / len(enriched):.2f}/record)",
            f"- avg entities/record: **{n_ent / len(enriched):.1f}**",
            f"- avg topics/record: **{n_top / len(enriched):.1f}**",
            f"- records with link_intent: **{n_int}/{len(enriched)}**",
            f"- link previews: {previews or 'n/a'}",
        ]
    else:
        lines.append("_no enriched_messages.json_\n")
    lines.append("")

    scraped = _load("scraped_links.json")
    lines += ["## 4. Scraped links", ""]
    if scraped:
        raw = [len(s.get("raw_text", "")) for s in scraped]
        lines += [
            f"- links: **{len(scraped)}**",
            f"- raw_text chars: mean **{statistics.mean(raw):.0f}** "
            f"(min {min(raw)}, max {max(raw)})",
        ]
        for f in ("summary", "what_it_is", "problem_solved", "how_useful"):
            n = sum(1 for s in scraped if (s.get(f) or "").strip())
            lines.append(f"- `{f}` present in **{n}/{len(scraped)}**")
        lines.append(
            f"- avg topics={sum(len(s.get('topics', [])) for s in scraped) / len(scraped):.1f}, "
            f"avg entities={sum(len(s.get('entities', [])) for s in scraped) / len(scraped):.1f}"
        )
    else:
        lines.append("_no scraped_links.json_\n")
    lines.append("")

    blocked = _load("blocked_links.json")
    ask_user = _load("ask_user_links.json")
    lines += ["## 5. Blocked / ask-user", ""]
    if blocked:
        breasons = {}
        for b in blocked:
            breasons[b.get("block_reason")] = breasons.get(b.get("block_reason"), 0) + 1
        lines.append(f"- blocked: **{len(blocked)}** {breasons}")
    else:
        lines.append("- blocked: none")
    if ask_user:
        areasons = {}
        for a in ask_user:
            areasons[a.get("block_reason")] = areasons.get(a.get("block_reason"), 0) + 1
        lines.append(f"- ask-user: **{len(ask_user)}** {areasons}")
    else:
        lines.append("- ask-user: none")
    lines.append("")

    sys_prompts = {
        "system_prompt_v2.txt": "grader pass1",
        "system_prompt_pass2_v1.txt": "grader pass2",
        "system_prompt_message_enhancer_v1.txt": "message enhancer",
        "system_prompt_content_summarizer_v1.txt": "content summarizer",
    }
    sys_tokens = sum(_est_tokens(open(p, encoding="utf-8").read()) for p in sys_prompts)

    lines += ["## 6. Tokens & latency", ""]
    run = metrics.load_last_run()
    if run and run.get("llm"):
        ll = run["llm"]
        lines += [
            "**Measured (from this run, via llama.cpp `usage` + client timing):**",
            f"- LLM calls: **{ll['calls']}**",
            f"- prompt tokens: **{ll['prompt_tokens']}**",
            f"- completion tokens: **{ll['completion_tokens']}**",
            f"- total tokens: **{ll['total_tokens']}**",
            f"- LLM wall time: **{ll['llm_time_seconds']}s** "
            f"(avg {ll['avg_call_seconds']}s/call, max {ll['max_call_seconds']}s)",
            "",
        ]
        if run.get("stats"):
            lines.append("**Pipeline stats:**")
            for k, v in run["stats"].items():
                lines.append(f"- {k}: {v}")
            lines.append("")
    else:
        lines += [
            "**Measured:** none stored yet — run the pipeline on actual data to populate "
            "`metrics/last_run.json`.",
            "",
        ]
    lines += [
        "**Estimated (char-based, `(len+3)//4`, from stored files):**",
        f"- system prompts: **{sys_tokens}**",
    ]
    if graded:
        lines.append(f"- grading input: **{sum(_est_tokens(g['original_text']) for g in graded)}**")
    if enriched:
        lines.append(f"- enrichment input: **{sum(_est_tokens(e['original_text']) for e in enriched)}**")
    if scraped:
        lines.append(f"- summarization input: **{sum(_est_tokens(s.get('raw_text', '')) for s in scraped)}**")
    lines.append("")

    lines += ["## 7. Storage", ""]
    st = _storage()
    if st["neo4j"]:
        lines.append("**Neo4j nodes:**")
        nums = {k: v for k, v in st["neo4j"].items()
                if isinstance(v, (int, float))}
        for label, count in sorted(nums.items(), key=lambda x: -x[1]):
            if label != "rels":
                lines.append(f"- {label}: {count}")
        if "rels" in nums:
            lines.append(f"- relationships: {nums['rels']}")
        if "error" in st["neo4j"]:
            lines.append(f"- (storage query error: {st['neo4j']['error']})")
    else:
        lines.append("- Neo4j: n/a")
    if st["qdrant"]:
        lines.append("**Qdrant:**")
        for k, v in st["qdrant"].items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- Qdrant: n/a")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Generate the Manthan metrics report")
    ap.add_argument("--chat", default="tests/test_chat.txt",
                    help="path to the chat export used for the funnel stats")
    ap.add_argument("--chat-label", default=PUBLIC_SOURCE_LABEL,
                    help="privacy-safe label shown in the public report")
    ap.add_argument("--out", default="metrics/manthan_report.md",
                    help="output markdown path")
    args = ap.parse_args()
    load_dotenv()
    report = build_report(args.chat, chat_label=args.chat_label)
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[metrics] report written to {args.out}")


if __name__ == "__main__":
    main()
