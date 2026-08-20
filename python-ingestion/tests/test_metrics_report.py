import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import metrics_report


FAILS = []


def check(name, condition, extra=""):
    if not condition:
        FAILS.append(name)
    print(f"[{'ok' if condition else 'FAIL'}] {name} {extra}")


default_line = metrics_report._source_summary(146)
check("default source is redacted", "Private chat export (redacted)" in default_line,
      default_line)
check("default source contains no filesystem path", "data/" not in default_line,
      default_line)

custom_line = metrics_report._source_summary(
    44,
    "Synthetic fixture\n`tests/test_chat.txt`",
)
check("custom source is kept on one Markdown-safe line", "\n" not in custom_line,
      custom_line)
check("custom source cannot inject code formatting", "`" not in custom_line,
      custom_line)

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
