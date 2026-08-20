import hashlib
import json
import os
import tempfile
import llm
from dataclasses import asdict, dataclass
from typing import List, Callable

from llm import (
    usage_stats, load_system_prompt, estimate_tokens, budget_for,
    split_text, COMPLETION_RESERVE, HARD_CHUNK_TOKENS, reset_usage,
)

MAX_BATCH = 5
PASS2_MAX_BATCH = max(1, int(os.environ.get("GRADER_PASS2_MAX_BATCH", 1)))
PASS2_COMPLETION_TOKENS = max(
    COMPLETION_RESERVE,
    int(os.environ.get("GRADER_PASS2_COMPLETION_TOKENS", 4096)),
)
PASS2_TARGET_CHARS = max(200, int(os.environ.get("GRADER_PASS2_TARGET_CHARS", 1200)))
PASS2_CONTEXT_CHARS = max(100, int(os.environ.get("GRADER_PASS2_CONTEXT_CHARS", 500)))
GRADER_DISABLE_THINKING = os.environ.get(
    "GRADER_DISABLE_THINKING", "1"
).strip().lower() not in {"0", "false", "no", "off"}
GRADING_CHECKPOINT = os.environ.get("GRADING_CHECKPOINT", "grading_checkpoint.json")
CHECKPOINT_VERSION = 2
GRADER_PROMPTS = ("system_prompt_v2.txt", "system_prompt_pass2_v1.txt")


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _grader_sha() -> str:
    digest = hashlib.sha1()
    digest.update(f"checkpoint-v{CHECKPOINT_VERSION}\0{llm.MODEL}".encode())
    for path in GRADER_PROMPTS:
        digest.update(b"\0")
        digest.update(load_system_prompt(path).encode("utf-8"))
    return digest.hexdigest()


def _new_checkpoint(chat_sha: str) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "chat_sha1": chat_sha,
        "grader_sha1": _grader_sha(),
        "pass1": {},
        "pass2": {},
    }


def _checkpoint_matches(cp: dict, chat_sha: str) -> bool:
    return (
        isinstance(cp, dict)
        and cp.get("version") == CHECKPOINT_VERSION
        and cp.get("chat_sha1") == chat_sha
        and cp.get("grader_sha1") == _grader_sha()
        and isinstance(cp.get("pass1"), dict)
        and isinstance(cp.get("pass2"), dict)
    )


def _load_checkpoint() -> dict:
    try:
        with open(GRADING_CHECKPOINT, encoding="utf-8") as f:
            cp = json.load(f)
    except (OSError, ValueError):
        cp = {}
    return cp if isinstance(cp, dict) else {}


def _save_checkpoint(cp: dict) -> None:
    directory = os.path.dirname(os.path.abspath(GRADING_CHECKPOINT))
    fd, tmp_path = tempfile.mkstemp(prefix=".grading-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cp, f, indent=2)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, GRADING_CHECKPOINT)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

@dataclass
class GradeResult:
    quality: int
    confidence: float
    category: str
    reason: str
    topics: List[str]
    original_text: str


def _parse(item: dict, category_fallback: str = None) -> dict:
    quality = item.get("quality")
    confidence = item.get("confidence")
    if quality is None or confidence is None:
        raise ValueError(f"missing quality/confidence: {item}")
    raw_topics = item.get("topics", [])
    if isinstance(raw_topics, list):
        topics = [str(t) for t in raw_topics]
    elif raw_topics:
        topics = [str(raw_topics)]
    else:
        topics = []
    return {
        "quality": max(1, min(5, int(quality))),
        "confidence": float(confidence),
        "category": str(item.get("category") or category_fallback or "noise").strip(),
        "reason": str(item.get("reason") or "").strip(),
        "topics": topics,
    }


def _msg_text(m) -> str:
    if isinstance(m, str):
        return m
    return getattr(m, "text", None) or getattr(m, "original_text", None) or ""


def _messages_sha(messages) -> str:
    payload = json.dumps([_msg_text(message) for message in messages],
                         ensure_ascii=False, separators=(",", ":"))
    return _sha(payload)


def _restore_grades(stored, messages) -> List["GradeResult"] | None:
    if not isinstance(stored, list) or len(stored) != len(messages):
        return None
    restored = []
    try:
        for item, message in zip(stored, messages):
            if not isinstance(item, dict) or item.get("original_text") != _msg_text(message):
                return None
            parsed = _parse(item)
            restored.append(GradeResult(original_text=_msg_text(message), **parsed))
    except (TypeError, ValueError):
        return None
    return restored


def _with_cached(cp: dict, pass_name: str, key: str,
                 results: List["GradeResult"]) -> dict:
    updated_pass = {**cp[pass_name], key: [asdict(result) for result in results]}
    return {**cp, pass_name: updated_pass}


def _pack(items, budget: int, size_fn: Callable,
          max_batch: int = MAX_BATCH) -> List[list]:
    batches = []
    cur = []
    cur_tokens = 0
    for item in items:
        tokens = size_fn(item)
        if cur and (cur_tokens + tokens > budget or len(cur) >= max_batch):
            batches.append(cur)
            cur = []
            cur_tokens = 0
        cur.append(item)
        cur_tokens += tokens
    if cur:
        batches.append(cur)
    return batches


def _merge_chunk_grades(results: List["GradeResult"], original_text: str) -> "GradeResult":
    best = max(results, key=lambda r: r.quality)
    topics = []
    for r in results:
        for t in r.topics:
            if t not in topics and len(topics) < 6:
                topics.append(t)
    reasons = [r.reason for r in results if r.quality == best.quality]
    reason = f"merged from {len(results)} chunks; " + "; ".join(reasons)
    return GradeResult(
        quality=best.quality,
        confidence=sum(r.confidence for r in results) / len(results),
        category=best.category,
        reason=reason[:500],
        topics=topics,
        original_text=original_text,
    )


def _make_user_content(batch) -> str:
    lines = "\n".join(f"{i + 1}) {_msg_text(m)}" for i, m in enumerate(batch))
    return (
        f"Grade each of the {len(batch)} messages below.\n"
        "Respond with STRICT JSON only: a single JSON array with exactly "
        f"{len(batch)} objects, in the same order as the messages. No markdown, no code fences.\n\n"
        f"Messages:\n{lines}"
    )


def grade_batch(batch: list, system_prompt: str, user_content: str,
                category_fallback: List[str] = None,
                max_tokens: int = COMPLETION_RESERVE,
                disable_thinking: bool = GRADER_DISABLE_THINKING) -> List[GradeResult]:
    data = llm.call_completion(system_prompt, user_content, max_tokens=max_tokens,
                               tag="grader", disable_thinking=disable_thinking)
    if len(data) != len(batch):
        raise ValueError(f"expected {len(batch)} grades, got {len(data)}")
    results = [
        GradeResult(
            quality=p["quality"], confidence=p["confidence"],
            category=p["category"], reason=p["reason"],
            topics=p["topics"], original_text=_msg_text(batch[i]),
        )
        for i, item in enumerate(data)
        for p in [_parse(item, category_fallback[i] if category_fallback else None)]
    ]
    print(f"[grader] got {len(results)} grades")
    return results


def grade_messages(messages: list) -> List[GradeResult]:
    system_prompt = load_system_prompt("system_prompt_v2.txt")
    budget = budget_for(system_prompt)
    batches = _pack(messages, budget, size_fn=lambda m: estimate_tokens(_msg_text(m)))
    print(f"[grader] grading {len(messages)} messages in {len(batches)} batches "
          f"(max {MAX_BATCH}/batch, {budget} token budget)")

    cp = _load_checkpoint()
    chat_sha = _messages_sha(messages)
    if not _checkpoint_matches(cp, chat_sha):
        cp = _new_checkpoint(chat_sha)
        _save_checkpoint(cp)

    results = []
    for batch in batches:
        batch_tokens = sum(estimate_tokens(_msg_text(m)) for m in batch)
        print(f"[grader] batch: {len(batch)} messages, {batch_tokens} tokens")
        if len(batch) == 1 and batch_tokens > budget:
            msg = batch[0]
            chunk_tokens = min(HARD_CHUNK_TOKENS, budget)
            pieces = split_text(_msg_text(msg), chunk_tokens)
            print(f"[grader] message too large for budget, splitting into {len(pieces)} chunks")
            chunk_results = []
            for piece in pieces:
                key = _messages_sha([piece])
                restored = _restore_grades(cp["pass1"].get(key), [piece])
                if restored is not None:
                    chunk_results.extend(restored)
                    print(f"[grader] resumed {len(restored)} chunk grades from checkpoint")
                else:
                    fres = grade_batch([piece], system_prompt,
                                       _make_user_content([piece]))
                    cp = _with_cached(cp, "pass1", key, fres)
                    _save_checkpoint(cp)
                    chunk_results.extend(fres)
            results.append(_merge_chunk_grades(chunk_results, _msg_text(msg)))
        else:
            key = _messages_sha(batch)
            restored = _restore_grades(cp["pass1"].get(key), batch)
            if restored is not None:
                results.extend(restored)
                print(f"[grader] resumed {len(restored)} grades from checkpoint")
            else:
                fres = grade_batch(batch, system_prompt, _make_user_content(batch))
                cp = _with_cached(cp, "pass1", key, fres)
                _save_checkpoint(cp)
                results.extend(fres)
    print(f"[grader] done: {usage_stats.llm_calls} LLM calls, {usage_stats.prompt_tokens} in, "
          f"{usage_stats.completion_tokens} out, {usage_stats.total_tokens} total")
    return results


def grade_pass2(messages: list) -> List[GradeResult]:
    target = [i for i, m in enumerate(messages) if m.quality == 3]
    print(f"[grader] pass2: re-grading {len(target)} borderline (quality=3) messages")
    if not target:
        return []
    system_prompt = load_system_prompt("system_prompt_pass2_v1.txt")
    budget = budget_for(system_prompt, PASS2_COMPLETION_TOKENS)

    def _truncate(text, limit):
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " [...]"

    def _block_for(item, k):
        idx, before, after = item
        target_text = _truncate(_msg_text(messages[idx]), PASS2_TARGET_CHARS)
        block = [f"MESSAGE {k}:", "TARGET MESSAGE:", target_text]
        if before:
            block.append("CONTEXT MESSAGES BEFORE:")
            block += [f"- {_truncate(_msg_text(m), PASS2_CONTEXT_CHARS)}" for m in before]
        if after:
            block.append("CONTEXT MESSAGES AFTER:")
            block += [f"- {_truncate(_msg_text(m), PASS2_CONTEXT_CHARS)}" for m in after]
        return "\n".join(block)

    def _size_fn(item):
        return estimate_tokens(_block_for(item, 1))

    targets = [(idx, messages[max(0, idx - 3):idx], messages[idx + 1:idx + 4])
               for idx in target]
    batches = _pack(targets, budget, _size_fn, max_batch=PASS2_MAX_BATCH)

    cp = _load_checkpoint()
    chat_sha = _messages_sha(messages)
    if not _checkpoint_matches(cp, chat_sha):
        cp = _new_checkpoint(chat_sha)
        _save_checkpoint(cp)

    results = []
    for batch in batches:
        blocks = [_block_for(item, k)
                  for k, item in enumerate(batch, 1)]
        user_content = (
            f"Re-grade the following {len(batch)} borderline messages using their surrounding context.\n"
            "Respond with STRICT JSON only: a single JSON array with exactly "
            f"{len(batch)} objects, one per message, in the same order. No markdown, no code fences.\n\n"
            + "\n\n".join(blocks)
        )
        print(f"[grader] pass2 batch: {len(batch)} message(s), "
              f"{estimate_tokens(user_content)} prompt tokens, "
              f"{PASS2_COMPLETION_TOKENS} completion tokens")
        msgs = [messages[idx] for idx, _, _ in batch]
        fallback_categories = [m.category for m in msgs]
        cache_input = json.dumps(
            {"user_content": user_content, "category_fallback": fallback_categories},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key = _sha(cache_input)
        restored = _restore_grades(cp["pass2"].get(key), msgs)
        if restored is not None:
            results.extend(restored)
            print(f"[grader] pass2 resumed {len(restored)} grades from checkpoint")
        else:
            fres = grade_batch(msgs, system_prompt, user_content,
                               category_fallback=fallback_categories,
                               max_tokens=PASS2_COMPLETION_TOKENS)
            cp = _with_cached(cp, "pass2", key, fres)
            _save_checkpoint(cp)
            results.extend(fres)
    print(f"[grader] pass2 done: {usage_stats.llm_calls} LLM calls, {usage_stats.prompt_tokens} in, "
          f"{usage_stats.completion_tokens} out, {usage_stats.total_tokens} total")
    return results
