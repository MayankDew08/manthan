import subprocess
import sys
from pathlib import Path

TESTS = [
    "tests/test_app_errors.py",
    "tests/test_metrics_report.py",
    "tests/test_worker_config.py",
    "tests/test_llm_json.py",
    "tests/test_grader_checkpoint.py",
    "tests/test_link_extract.py",
    "tests/test_link_scraper.py",
    "tests/test_link_scraper_youtube.py",
    "tests/test_link_scraper_github.py",
    "tests/test_link_scraper_x.py",
    "tests/test_enhancer.py",
    "tests/test_summary_only.py",
    "tests/test_link_ingest.py",
    "tests/test_e2e_enrich.py",
    "tests/test_pipeline_langgraph.py",
]

root = Path(__file__).resolve().parents[1]
failed = []
for t in TESTS:
    print(f"\n{'=' * 60}\nRUN {t}\n{'=' * 60}")
    r = subprocess.run([sys.executable, t], cwd=root, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:])
        failed.append(t)

print("\n" + ("ALL TESTS PASS" if not failed else f"FAILED: {failed}"))
sys.exit(1 if failed else 0)
