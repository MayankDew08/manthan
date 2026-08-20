import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException

import app


FAILS = []
app.logger.disabled = True


def check(name, condition, extra=""):
    if not condition:
        FAILS.append(name)
    print(f"[{'ok' if condition else 'FAIL'}] {name} {extra}")


def run_all():
    FAILS.clear()
    private_detail = "private database path and credential"
    original_search = app.search
    app.search = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(private_detail))
    app.app.state.vs = object()
    app.app.state.store = object()
    try:
        app.do_search(app.SearchRequest(query="test"))
    except HTTPException as exc:
        check("search errors are user-safe", exc.detail == "search failed", exc.detail)
        check("search errors do not expose internal detail", private_detail not in exc.detail,
              exc.detail)
    finally:
        app.search = original_search

    original_ingest = app.pipeline.run_full_ingest
    app.pipeline.run_full_ingest = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError(private_detail))
    job_id = app._new_job("private-chat.txt")
    try:
        app._run_ingest_job(job_id, "private-chat.txt", False, False, 4)
        check("job errors are user-safe", app.JOBS[job_id]["error"] == "ingestion failed",
              app.JOBS[job_id]["error"])
        check("job errors do not expose internal detail",
              private_detail not in app.JOBS[job_id]["error"], app.JOBS[job_id]["error"])
    finally:
        app.pipeline.run_full_ingest = original_ingest

    print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
    return FAILS


def test_all():
    fails = run_all()
    assert not fails, f"failed checks: {fails}"


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)