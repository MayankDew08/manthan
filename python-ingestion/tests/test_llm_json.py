import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm


FAILS = []


def check(name, condition, extra=""):
    if not condition:
        FAILS.append(name)
    print(f"[{'ok' if condition else 'FAIL'}] {name} {extra}")


object_text = (
    '{"summary":"ok","entities":["qdrant"],'
    '"topics":["vector-databases"]}'
)
expected_object = [{
    "summary": "ok",
    "entities": ["qdrant"],
    "topics": ["vector-databases"],
}]

check(
    "top-level object with nested arrays parses",
    llm.clean_json(object_text) == expected_object,
    llm.clean_json(object_text),
)
check(
    "top-level grading array still parses",
    llm.clean_json('[{"quality":4,"topics":["qdrant"]}]')
    == [{"quality": 4, "topics": ["qdrant"]}],
)
check(
    "fenced object parses",
    llm.clean_json(f"```json\n{object_text}\n```") == expected_object,
)
check(
    "parser rejects surrounding non-JSON text",
    llm.clean_json(f"Reasoning [not JSON]. Final: {object_text}") == [],
)
check(
    "parser rejects double-encoded JSON strings",
    llm.clean_json(json.dumps(object_text)) == [],
)


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": object_text},
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 11},
        }


original_post = llm.requests.post
try:
    llm.requests.post = lambda *args, **kwargs: FakeResponse()
    result = llm.call_completion("system", "user", tag="test")
finally:
    llm.requests.post = original_post

check("completion parses successful object response", result == expected_object, result)


captured_payload = {}


class ThinkingSensitiveResponse(FakeResponse):
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        thinking_disabled = (
            self.payload.get("chat_template_kwargs", {}).get("enable_thinking") is False
        )
        content = object_text if thinking_disabled else ""
        return {
            "choices": [{
                "finish_reason": "stop" if thinking_disabled else "length",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 32},
        }


def thinking_sensitive_post(*args, **kwargs):
    captured_payload.update(kwargs["json"])
    return ThinkingSensitiveResponse(kwargs["json"])


try:
    llm.requests.post = thinking_sensitive_post
    no_think_result = llm.call_completion(
        "system", "user", tag="test-no-think", disable_thinking=True
    )
finally:
    llm.requests.post = original_post

check(
    "completion disables thinking for structured extraction",
    captured_payload.get("chat_template_kwargs") == {"enable_thinking": False},
    captured_payload.get("chat_template_kwargs"),
)
check("disabled thinking produces visible JSON", no_think_result == expected_object,
      no_think_result)


semantic_attempts = []


class SemanticRetryResponse(FakeResponse):
    def __init__(self, content, finish_reason):
        self.content = content
        self.finish_reason = finish_reason

    def json(self):
        return {
            "choices": [{
                "finish_reason": self.finish_reason,
                "message": {"role": "assistant", "content": self.content},
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 32},
        }


def semantic_retry_post(*args, **kwargs):
    semantic_attempts.append(dict(kwargs["json"]))
    if kwargs["json"]["max_tokens"] <= 2048:
        return SemanticRetryResponse("", "length")
    return SemanticRetryResponse(object_text, "stop")


original_sleep = llm.time.sleep
try:
    llm.requests.post = semantic_retry_post
    llm.time.sleep = lambda _seconds: None
    semantic_retry_result = llm.call_completion("system", "user", tag="test-retry")
finally:
    llm.requests.post = original_post
    llm.time.sleep = original_sleep

check("empty length completion is retried", len(semantic_attempts) == 2,
      len(semantic_attempts))
check("length retry increases the completion budget",
      [attempt["max_tokens"] for attempt in semantic_attempts] == [2048, 4096],
      [attempt["max_tokens"] for attempt in semantic_attempts])
check("semantic retry returns valid JSON", semantic_retry_result == expected_object,
      semantic_retry_result)

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
