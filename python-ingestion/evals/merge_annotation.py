"""Merge human grades from an existing annotation sheet into a freshly built one.

Used whenever queries.csv grows (spec scaling: 5 -> 20 -> 50): build_pool.py
regenerates annotation.csv for every query currently in queries.csv, which
would otherwise overwrite already-graded rows with blank ones. This merges
the old grades back in by (query_id, candidate_id) and reports anything that
could not be carried over, so no labelling work is silently lost.

Usage (from python-ingestion/):
  uv run python -m evals.merge_annotation --corpus-version pilot-001 --old path/to/graded.csv
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common  # noqa: E402

GRADE_COLUMNS = ("relevance", "reason")


def _key(row: dict) -> tuple:
    return (row["query_id"], row["candidate_id"])


def merge_preserving_grades(old_rows: list[dict], new_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Copy relevance/reason from ``old_rows`` onto matching ``new_rows``.

    Returns (merged_rows, orphaned_rows) in ``new_rows``' order. An orphaned
    row was graded before but its (query_id, candidate_id) is no longer in
    the freshly built pool (the query's pool composition changed) -- its
    grade cannot be carried over and is reported, never silently dropped.
    """
    old_graded = {_key(r): r for r in old_rows if (r.get("relevance") or "").strip() != ""}
    new_keys = {_key(r) for r in new_rows}

    merged = []
    for row in new_rows:
        old = old_graded.get(_key(row))
        merged.append({**row, "relevance": old["relevance"], "reason": old.get("reason", "")}
                      if old else dict(row))

    orphaned = [r for k, r in old_graded.items() if k not in new_keys]
    return merged, orphaned


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    ap.add_argument("--old", required=True, type=Path,
                    help="the previously graded annotation.csv to pull grades from")
    args = ap.parse_args(argv)

    out_dir = common.data_dir(args.corpus_version, create=False)
    new_path = out_dir / "annotation.csv"
    with open(args.old, encoding="utf-8", newline="") as f:
        old_rows = list(csv.DictReader(f))
    with open(new_path, encoding="utf-8", newline="") as f:
        new_rows = list(csv.DictReader(f))

    merged, orphaned = merge_preserving_grades(old_rows, new_rows)
    carried = sum(1 for r in merged if (r.get("relevance") or "").strip() != "")
    still_blank = len(merged) - carried

    backup = new_path.with_suffix(".csv.pre-merge-bak")
    shutil.copy2(new_path, backup)
    with open(new_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=common.ANNOTATION_COLUMNS)
        w.writeheader()
        w.writerows(merged)

    print(f"carried over {carried} grade(s); {still_blank} row(s) still need grading")
    if orphaned:
        print(f"WARNING: {len(orphaned)} previously graded row(s) have no match in the new "
              f"pool and were NOT carried over (their query's pool composition changed):")
        for r in orphaned:
            print(f"   {r['query_id']} / {r['candidate_id']}: relevance={r.get('relevance')!r}")
    print(f"pre-merge sheet backed up to {backup}")
    return 1 if orphaned else 0


if __name__ == "__main__":
    raise SystemExit(main())
