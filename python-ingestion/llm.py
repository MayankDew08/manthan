"""Call a local OpenAI-compatible LLM and track approximate token usage."""

import os
import json
import time
from dataclasses import dataclass
from typing import List

import requests


def _default_base_url() -> str:
    url = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1/chat/completions")
    if url.endswith("/chat/completions") or url.endswith("/v1/chat/completions"):
        return url
    return url.rstrip("/") + "/v1/chat/completions"


BASE_URL = _default_base_url()
MODEL = os.environ.get("LLAMA_MODEL", "gemma-4-e4b")

COMPLETION_RESERVE = int(os.environ.get("GRADER_COMPLETION_RESERVE", 2048))
CONTEXT_WINDOW = int(os.environ.get("GRADER_CONTEXT_WINDOW", 32768))
SAFETY_MARGIN = int(os.environ.get("GRADER_SAFETY_MARGIN", 2000))
HARD_CHUNK_TOKENS = int(os.environ.get("GRADER_CHUNK_TOKENS", 12000))
DISABLE_THINKING = os.environ.get("LLAMA_DISABLE_THINKING", "0").strip().lower() \
    not in {"0", "false", "no", "off"}

MAX_RETRIES = 2
REQUEST_TIMEOUT = int(os.environ.get("LLAMA_REQUEST_TIMEOUT", 600))


@dataclass
class UsageStats:
    """Process-local counters collected across LLM requests."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_time: float = 0.0
    max_call_time: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_call_time(self) -> float:
        return self.llm_time / self.llm_calls if self.llm_calls else 0.0


usage_stats = UsageStats()


def reset_usage() -> None:
    usage_stats.llm_calls = 0
    usage_stats.prompt_tokens = 0
    usage_stats.completion_tokens = 0
    usage_stats.llm_time = 0.0
    usage_stats.max_call_time = 0.0


def load_system_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def budget_for(system_prompt: str, completion_reserve: int = COMPLETION_RESERVE) -> int:
    """Estimate the input budget used for batching and chunking prompts."""
    return max(1, CONTEXT_WINDOW - estimate_tokens(system_prompt)
               - completion_reserve - SAFETY_MARGIN)


def clean_json(raw: str) -> List[dict]:
    """Normalize a fenced object or nested array response into JSON objects."""
    if not isinstance(raw, str):
        return []
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            lines = lines[1:-1]
            text = "\n".join(lines).strip()
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []

    values = decoded if isinstance(decoded, list) else [decoded]
    out = []
    for value in values:
        if isinstance(value, dict):
            out.append(value)
        elif isinstance(value, list):
            out.extend(item for item in value if isinstance(item, dict))
    return out


def split_text(text: str, chunk_tokens: int) -> List[str]:
    """Split text near natural separators using the repository's token estimate."""
    step_chars = max(1, chunk_tokens * 4)
    if len(text) <= step_chars:
        return [text]
    pieces = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + step_chars)
        cut = end
        if end < n:
            lo = start + max(1, step_chars // 3)
            best = -1
            for sep in ("\n", ". ", ", ", "; ", "? ", "! ", " "):
                pos = text.rfind(sep, lo, end)
                if pos != -1:
                    best = max(best, pos + len(sep))
            if best != -1:
                cut = best
        piece = text[start:cut]
        if piece.strip():
            pieces.append(piece)
        if cut <= start:
            cut = start + 1
        start = cut
    return pieces


def call_completion(system_prompt: str, user_content: str, *,
                    model: str = MODEL, max_tokens: int = COMPLETION_RESERVE,
                    temperature: float = 0.1, tag: str = "llm",
                    parse_json: bool = True,
                    disable_thinking: bool = DISABLE_THINKING):
    """Call the configured model, retry failures, and optionally parse JSON output."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking and parse_json:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    available_completion = max(
        1,
        CONTEXT_WINDOW - estimate_tokens(system_prompt) - estimate_tokens(user_content)
        - SAFETY_MARGIN,
    )
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            print(f"[{tag}] calling LLM (attempt {attempt}/{MAX_RETRIES + 1})...")
            usage_stats.llm_calls += 1
            call_start = time.perf_counter()
            resp = requests.post(BASE_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
            elapsed = time.perf_counter() - call_start
            usage_stats.llm_time += elapsed
            usage_stats.max_call_time = max(usage_stats.max_call_time, elapsed)
            usage = body.get("usage") or {}
            usage_stats.prompt_tokens += usage.get("prompt_tokens", 0)
            usage_stats.completion_tokens += usage.get("completion_tokens", 0)
            choice = body["choices"][0]
            content = choice["message"]["content"]
            if not parse_json:
                print(f"[{tag}] LLM call succeeded")
                return content
            parsed = clean_json(content)
            if parsed:
                print(f"[{tag}] LLM call succeeded ({len(parsed)} JSON object(s))")
            else:
                finish_reason = choice.get("finish_reason", "unknown")
                content_len = len(content) if isinstance(content, str) else 0
                print(f"[{tag}] LLM call completed but returned no valid JSON "
                      f"(finish_reason={finish_reason}, content_chars={content_len})")
                if attempt <= MAX_RETRIES:
                    if finish_reason == "length":
                        payload["max_tokens"] = min(
                            available_completion,
                            max(payload["max_tokens"] + 1024, payload["max_tokens"] * 2),
                        )
                    print(f"[{tag}] retrying invalid JSON response "
                          f"({attempt}/{MAX_RETRIES}, max_tokens={payload['max_tokens']})")
                    time.sleep(1)
                    continue
                print(f"[{tag}] no valid JSON after {MAX_RETRIES} retries")
            return parsed
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            if attempt <= MAX_RETRIES:
                print(f"[{tag}] LLM call FAILED ({reason}) - retry {attempt}/{MAX_RETRIES}")
                time.sleep(1)
            else:
                print(f"[{tag}] LLM call FAILED ({reason}) - giving up after {MAX_RETRIES} retries")
                raise
