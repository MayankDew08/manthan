import datetime as dt
import json
import os
from pathlib import Path

import llm

METRICS_DIR = "metrics"
LAST_RUN_FILE = os.path.join(METRICS_DIR, "last_run.json")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_last_run(**extra) -> dict:
    data = {
        "generated_at": _now(),
        "llm": {
            "calls": llm.usage_stats.llm_calls,
            "prompt_tokens": llm.usage_stats.prompt_tokens,
            "completion_tokens": llm.usage_stats.completion_tokens,
            "total_tokens": llm.usage_stats.total_tokens,
            "llm_time_seconds": round(llm.usage_stats.llm_time, 3),
            "avg_call_seconds": round(llm.usage_stats.avg_call_time, 3),
            "max_call_seconds": round(llm.usage_stats.max_call_time, 3),
        },
        **extra,
    }
    Path(METRICS_DIR).mkdir(exist_ok=True)
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def load_last_run() -> dict:
    try:
        with open(LAST_RUN_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}