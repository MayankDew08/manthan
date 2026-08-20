"""Apply LLM-based metadata extraction and summaries to messages and links."""

from typing import List, Optional

import llm

MESSAGE_PROMPT = "system_prompt_message_enhancer_v1.txt"
SUMMARY_PROMPT = "system_prompt_content_summarizer_v1.txt"
REPO_PROMPT = "system_prompt_repo_overview_v1.txt"

STRICT_RETRY_NOTE = (
    "\n\nREMINDER: Your previous reply could not be parsed as JSON. "
    "Reply with EXACTLY ONE JSON object and nothing else: no markdown, "
    "no code fences, no commentary before or after."
)


def _first(data, *, system_prompt, user, tag, max_tokens, fallback=None) -> dict:
    """Return the first JSON object, retrying once with stricter instructions."""
    if data:
        return data[0]
    print(f"[{tag}] LLM returned no JSON objects; retrying with strict instruction")
    data = llm.call_completion(system_prompt, user + STRICT_RETRY_NOTE,
                               tag=tag, max_tokens=max_tokens)
    if data:
        return data[0]
    print(f"[{tag}] still no JSON; using empty fallback")
    if fallback is not None:
        return fallback
    raise ValueError("LLM returned no JSON objects")


def _extract_str_list(item: dict, key: str) -> List[str]:
    raw = item.get(key, [])
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw:
        return [str(raw).strip()]
    return []


def _dedupe(items: List[str]) -> List[str]:
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _normalize_summary(item: dict) -> dict:
    """Normalize a web summary into the persisted link-summary schema."""
    return {
        "summary": (item.get("summary") or "").strip(),
        "what_it_is": (item.get("what_it_is") or "").strip(),
        "problem_solved": (item.get("problem_solved") or "").strip(),
        "how_useful": (item.get("how_useful") or "").strip(),
        "entities": _extract_str_list(item, "entities"),
        "topics": _extract_str_list(item, "topics"),
    }


def enrich_message(original_text: str, link_urls: List[str] = None) -> dict:
    """Extract link intent, entities, and topics from one chat message."""
    system_prompt = llm.load_system_prompt(MESSAGE_PROMPT)
    parts = [f"Original message:\n{original_text}"]
    if link_urls:
        links = "\n".join(f"{i + 1}) {u}" for i, u in enumerate(link_urls))
        parts.append(f"Links detected:\n{links}")
    parts.append("Respond with STRICT JSON only.")
    user = "\n\n".join(parts)
    item = _first(llm.call_completion(system_prompt, user, tag="enhancer",
                                      max_tokens=512),
                  system_prompt=system_prompt, user=user, tag="enhancer",
                  max_tokens=512,
                  fallback={"link_intent": None, "entities": [], "topics": []})
    return {
        "link_intent": (item.get("link_intent") or "").strip() or None,
        "entities": _extract_str_list(item, "entities"),
        "topics": _extract_str_list(item, "topics"),
    }


def summarize_content(url: str, title: str, raw_text: str) -> dict:
    """Summarize web content directly or with map-reduce when it is oversized."""
    system_prompt = llm.load_system_prompt(SUMMARY_PROMPT)
    budget = llm.budget_for(system_prompt)
    if llm.estimate_tokens(raw_text) <= budget:
        user = _summary_user(url, title, raw_text)
        data = llm.call_completion(system_prompt, user, tag="summarizer",
                                   max_tokens=1024)
        item = _first(data, system_prompt=system_prompt, user=user,
                      tag="summarizer", max_tokens=1024, fallback={})
        return _normalize_summary(item)
    return _map_reduce_summary(system_prompt, url, title, raw_text, budget)


def _summary_user(url: str, title: str, raw_text: str) -> str:
    return (
        f"URL: {url}\nTitle: {title or '(none)'}\n\n{raw_text}\n\n"
        "Produce a faithful summary, what the resource is, the problem it addresses, "
        "how it is useful, plus entities and topics. Respond with STRICT JSON only."
    )


def _map_reduce_summary(system_prompt: str, url: str, title: str,
                        raw_text: str, budget: int) -> dict:
    """Summarize chunks independently and merge their structured key points."""
    chunk_tokens = min(llm.HARD_CHUNK_TOKENS, budget)
    parts = llm.split_text(raw_text, chunk_tokens)
    print(f"[summarizer] content too large, splitting into {len(parts)} parts")
    key_points = []
    entities = []
    topics = []
    for k, part in enumerate(parts, 1):
        user = (
            f"URL: {url}\nPart {k}/{len(parts)} of the resource (raw excerpt):\n{part}\n\n"
            "Extract the key points of this part only. Respond with STRICT JSON only."
        )
        item = _first(llm.call_completion(system_prompt, user, tag="summarizer",
                                          max_tokens=1024),
                      system_prompt=system_prompt, user=user, tag="summarizer",
                      max_tokens=1024, fallback={})
        key_points.append(item.get("summary") or "")
        entities += _extract_str_list(item, "entities")
        topics += _extract_str_list(item, "topics")
    merge_user = (
        f'Combine these part summaries of the resource "{title or url}" into ONE coherent, '
        "faithful summary (150-250 words), what_it_is, problem_solved, how_useful, "
        "entities, and topics.\n\nPart summaries:\n"
        + "\n".join(f"{k}) {s}" for k, s in enumerate(key_points, 1))
        + "\n\nRespond with STRICT JSON only."
    )
    merged = _first(llm.call_completion(system_prompt, merge_user, tag="summarizer",
                                        max_tokens=1024),
                    system_prompt=system_prompt, user=merge_user, tag="summarizer",
                    max_tokens=1024, fallback={})
    result = _normalize_summary(merged)
    if len(result["entities"]) < 3:
        result["entities"] = _dedupe(result["entities"] + entities)[:5]
    if len(result["topics"]) < 2:
        result["topics"] = _dedupe(result["topics"] + topics)[:3]
    return result


def summarize_repo(url: str, title: str, raw_text: str) -> dict:
    """Build a practitioner-oriented repository overview from captured context."""
    system_prompt = llm.load_system_prompt(REPO_PROMPT)
    budget = llm.budget_for(system_prompt)
    if llm.estimate_tokens(raw_text) <= budget:
        user = _repo_user(url, title, raw_text)
        data = llm.call_completion(system_prompt, user, tag="repo",
                                   max_tokens=1024)
        item = _first(data, system_prompt=system_prompt, user=user,
                      tag="repo", max_tokens=1024, fallback={})
        return _normalize_repo(item)
    return _map_reduce_repo(system_prompt, url, title, raw_text, budget)


def _repo_user(url: str, title: str, raw_text: str) -> str:
    return (
        f"Repository: {title or '(no title)'}\nURL: {url}\n\n{raw_text}\n\n"
        "Describe what this repo is, the problem it solves, and how it is useful. "
        "Respond with STRICT JSON only."
    )


def _normalize_repo(item: dict) -> dict:
    """Normalize a repository overview into the persisted summary schema."""
    return {
        "summary": (item.get("summary") or "").strip(),
        "what_it_is": (item.get("what_it_is") or "").strip(),
        "problem_solved": (item.get("problem_solved") or "").strip(),
        "how_useful": (item.get("how_useful") or "").strip(),
        "entities": _extract_str_list(item, "entities"),
        "topics": _extract_str_list(item, "topics"),
    }


def _map_reduce_repo(system_prompt: str, url: str, title: str,
                     raw_text: str, budget: int) -> dict:
    """Summarize README chunks and merge them into one repository overview."""
    chunk_tokens = min(llm.HARD_CHUNK_TOKENS, budget)
    parts = llm.split_text(raw_text, chunk_tokens)
    print(f"[repo] content too large, splitting into {len(parts)} parts")
    key_points = []
    entities = []
    topics = []
    for k, part in enumerate(parts, 1):
        user = (
            f"Repository: {title or url}\nPart {k}/{len(parts)} of the README:\n{part}\n\n"
            "Extract the key points of this part only (what it is, problem solved, usefulness). "
            "Respond with STRICT JSON only."
        )
        item = _first(llm.call_completion(system_prompt, user, tag="repo",
                                          max_tokens=1024),
                      system_prompt=system_prompt, user=user, tag="repo",
                      max_tokens=1024, fallback={})
        key_points.append(item.get("summary") or "")
        entities += _extract_str_list(item, "entities")
        topics += _extract_str_list(item, "topics")
    merge_user = (
        f'Combine these part overviews of the repo "{title or url}" into ONE coherent '
        "practitioner-focused overview: what it is, the problem it solves, and how it is useful "
        "(summary 150-250 words, plus what_it_is, problem_solved, how_useful, entities, topics).\n\n"
        "Part overviews:\n"
        + "\n".join(f"{k}) {s}" for k, s in enumerate(key_points, 1))
        + "\n\nRespond with STRICT JSON only."
    )
    merged = _first(llm.call_completion(system_prompt, merge_user, tag="repo",
                                        max_tokens=1024),
                    system_prompt=system_prompt, user=merge_user, tag="repo",
                    max_tokens=1024, fallback={})
    result = _normalize_repo(merged)
    if len(result["entities"]) < 3:
        result["entities"] = _dedupe(result["entities"] + entities)[:5]
    if len(result["topics"]) < 2:
        result["topics"] = _dedupe(result["topics"] + topics)[:3]
    return result
