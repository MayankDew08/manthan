import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grader
import llm


FAILS = []


def check(name, condition, extra=""):
    if not condition:
        FAILS.append(name)
    print(f"[{'ok' if condition else 'FAIL'}] {name} {extra}")


message = SimpleNamespace(text="Use Qdrant for this workload.")
chat_sha = hashlib.sha1(message.text.encode()).hexdigest()
batch_sha = chat_sha
tmp = Path(tempfile.mkdtemp(prefix="manthan_grader_"))
checkpoint = tmp / "grading.json"
checkpoint.write_text(json.dumps({
    "chat_sha1": chat_sha,
    "pass1": {
        batch_sha: [{
            "quality": 3,
            "confidence": 0.9,
            "category": "discussion",
            "reason": "fake",
            "topics": ["fake"],
            "original_text": message.text,
        }],
    },
    "pass2": {},
}), encoding="utf-8")

calls = []


def fake_completion(system_prompt, user_content, **kwargs):
    calls.append(user_content)
    return [{
        "quality": 4,
        "confidence": 0.95,
        "category": "insight",
        "reason": "Specific recommendation naming a vector database.",
        "topics": ["qdrant", "vector-databases"],
    }]


original_checkpoint = grader.GRADING_CHECKPOINT
original_completion = llm.call_completion
try:
    grader.GRADING_CHECKPOINT = str(checkpoint)
    llm.call_completion = fake_completion
    result = grader.grade_messages([message])
    first_call_count = len(calls)
    calls.clear()
    resumed = grader.grade_messages([message])
finally:
    grader.GRADING_CHECKPOINT = original_checkpoint
    llm.call_completion = original_completion

check("legacy unversioned checkpoint is ignored", first_call_count == 1, first_call_count)
check("legacy fake value is replaced", result[0].reason != "fake", result[0])
check("current checkpoint is reusable", not calls, len(calls))
check("reused grade matches fresh grade", resumed == result, resumed)

saved = json.loads(checkpoint.read_text(encoding="utf-8"))
check("checkpoint records a schema version", "version" in saved, saved.keys())
check("checkpoint records a grader fingerprint", bool(saved.get("grader_sha1")), saved)


pass2_calls = []


def fake_pass2_completion(system_prompt, user_content, **kwargs):
    pass2_calls.append(user_content)
    return [{
        "quality": 3,
        "confidence": 0.8,
        "reason": "Borderline message retains its pass-one category.",
        "topics": ["qdrant"],
    }]


discussion = grader.GradeResult(
    quality=3,
    confidence=0.8,
    category="discussion",
    reason="borderline",
    topics=["qdrant"],
    original_text=message.text,
)
resource = grader.GradeResult(
    quality=3,
    confidence=0.8,
    category="resource",
    reason="borderline",
    topics=["qdrant"],
    original_text=message.text,
)

try:
    grader.GRADING_CHECKPOINT = str(checkpoint)
    llm.call_completion = fake_pass2_completion
    first_pass2 = grader.grade_pass2([discussion])
    second_pass2 = grader.grade_pass2([resource])
finally:
    grader.GRADING_CHECKPOINT = original_checkpoint
    llm.call_completion = original_completion

check("pass2 category participates in cache identity", len(pass2_calls) == 2,
      len(pass2_calls))
check("pass2 fallback category is not stale",
      first_pass2[0].category == "discussion" and second_pass2[0].category == "resource",
      (first_pass2, second_pass2))


small_model_checkpoint = tmp / "small-model-grading.json"
small_model_calls = []


def small_model_completion(system_prompt, user_content, **kwargs):
    message_count = user_content.count("MESSAGE ")
    small_model_calls.append({
        "message_count": message_count,
        "max_tokens": kwargs.get("max_tokens"),
        "content_chars": len(user_content),
        "disable_thinking": kwargs.get("disable_thinking"),
    })
    if message_count != 1:
        return []
    return [{
        "quality": 2,
        "confidence": 0.9,
        "reason": "Context confirms this is social or logistical glue.",
        "topics": ["social"],
    }]


long_context = "Detailed neighboring technical context. " * 500
small_model_messages = [
    grader.GradeResult(5, 0.9, "insight", "context", ["ai"], long_context),
    grader.GradeResult(3, 0.8, "validation", "borderline", ["social"], "Sounds good"),
    grader.GradeResult(3, 0.8, "social", "borderline", ["meeting"], "What time?"),
    grader.GradeResult(3, 0.8, "validation", "borderline", ["social"], "Will check"),
    grader.GradeResult(3, 0.8, "social", "borderline", ["social"], "See you"),
]

small_model_result = None
try:
    grader.GRADING_CHECKPOINT = str(small_model_checkpoint)
    llm.call_completion = small_model_completion
    small_model_result = grader.grade_pass2(small_model_messages)
except ValueError:
    pass
finally:
    grader.GRADING_CHECKPOINT = original_checkpoint
    llm.call_completion = original_completion

check("pass2 grades each target in a small-model-sized request",
      small_model_result is not None and len(small_model_result) == 4,
      small_model_result)
check("pass2 uses one target per LLM request",
      len(small_model_calls) == 4
      and all(call["message_count"] == 1 for call in small_model_calls),
      small_model_calls)
check("pass2 reserves a larger completion budget",
      small_model_calls and all(call["max_tokens"] >= 4096 for call in small_model_calls),
      small_model_calls)
check("grader disables model thinking",
      small_model_calls and all(call["disable_thinking"] is True
                                for call in small_model_calls),
      small_model_calls)
check("pass2 always bounds surrounding context",
      small_model_calls and max(call["content_chars"] for call in small_model_calls) < 6000,
      small_model_calls)

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
