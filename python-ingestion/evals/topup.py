"""U6: top-up grading sheet, hard-fail judging gate and qrels_v2 (spec section 10).

The ladder is only measurable when every (query, candidate) pair a ladder step
can show in its top 5 carries a human grade. ``annotation.csv`` -- the owner's
complete graded sheet -- covers the cosine/BM25/keyword pool, but fusion, the
graph run and the re-ranker surface documents that pool never contained. This
module finds those gaps, writes a blind top-up sheet for the owner to grade,
and merges the graded sheet into ``qrels_v2.csv``.

Usage (from python-ingestion/):
  uv run python -m evals.topup build    # writes annotation_topup.csv
  uv run python -m evals.topup merge    # writes qrels_v2.csv

``build`` reads ``annotation.csv`` as the judged set -- never ``qrels.csv``,
which is the historic 5-query file. ``merge`` only ever writes
``qrels_v2.csv``; ``qrels.csv`` and ``annotation.csv`` are read-only inputs and
are never rewritten, deleted or renamed.
"""

import argparse
import csv
import random
from pathlib import Path

from evals import common, pooling

TOPUP_K = 5
QRELS_V2 = "qrels_v2.csv"
ANNOTATION = "annotation.csv"
ANNOTATION_TOPUP = "annotation_topup.csv"

QRELS_COLUMNS = ["query_id", "candidate_id", "relevance", "answer_bearing", "reason"]

# The five ladder steps (spec section 4), spelled out here rather than imported
# from the wave-2 runner so this module stays light and import-clean.
LADDER_RUN_IDS = (
    "ladder_s0_cosine",
    "ladder_s1_cos_bm25",
    "ladder_s2_cos_bm25_rr",
    "ladder_s3_cos_bm25_graph",
    "ladder_s4_cos_bm25_graph_rr",
)

# Spec section 17: a top-up sheet this large means the pool was wrong, not that
# the owner should be asked to grade it.
TOPUP_ESCALATE_ROWS = 300

SHUFFLE_SEED = 42


class UnjudgedError(RuntimeError):
    """Raised by :func:`assert_fully_judged` when top-k pairs have no grade.

    A hard failure, never a skip: a ladder number computed over unjudged
    documents is not a measurement.
    """

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = list(pairs)
        shown = ", ".join(f"{q}/{c}" for q, c in self.pairs[:10])
        if len(self.pairs) > 10:
            shown += f", ... (+{len(self.pairs) - 10} more)"
        super().__init__(
            f"{len(self.pairs)} unjudged (query, candidate) pair(s) inside the top "
            f"{TOPUP_K}; grade them ('uv run python -m evals.topup build' then "
            f"'merge') before scoring: {shown}"
        )


def unjudged_pairs(runs: dict[str, list[dict]], qrels: dict[str, dict[str, dict]],
                   k: int = TOPUP_K) -> list[tuple[str, str]]:
    """(query_id, candidate_id) pairs ranked <= k in any run with no qrels entry.

    ``runs`` is ``{run_id: rows}`` with ``query_id``/``candidate_id``/``rank``
    per row; ``qrels`` is ``common.load_qrels`` output. Rows ranked below the
    cut are ignored. The result is sorted and deduplicated, so a caller cannot
    infer which run surfaced a pair.
    """
    pairs: set[tuple[str, str]] = set()
    for rows in runs.values():
        for row in rows:
            if int(row["rank"]) > k:
                continue
            qid, cid = row["query_id"], row["candidate_id"]
            if cid not in (qrels.get(qid) or {}):
                pairs.add((qid, cid))
    return sorted(pairs)


def assert_fully_judged(runs: dict[str, list[dict]], qrels: dict[str, dict[str, dict]],
                        k: int = TOPUP_K) -> None:
    """Raise :class:`UnjudgedError` unless every top-k pair has been graded."""
    pairs = unjudged_pairs(runs, qrels, k=k)
    if pairs:
        raise UnjudgedError(pairs)


def topup_rows(pairs: list[tuple[str, str]], queries_by_id: dict[str, dict],
               docs_by_id: dict[str, dict], seed: int = SHUFFLE_SEED) -> list[dict]:
    """Build the blind top-up sheet: annotation columns, blank grades.

    The pairs are sorted, deduplicated and then shuffled with
    ``random.Random(seed)``, so the grading order is reproducible but is not
    the alphabetical order the annotator would otherwise see. No run id, rank,
    score or provenance column is written: the owner must not be able to tell
    which step surfaced a document.
    """
    order = sorted(set(pairs))
    random.Random(seed).shuffle(order)

    rows = []
    for qid, cid in order:
        doc = docs_by_id[cid]
        if qid not in queries_by_id:
            raise ValueError(f"top-up pair {qid}/{cid}: query {qid} is not in queries.csv")
        rows.append({
            "query_id": qid,
            "query_text": queries_by_id[qid]["query_text"],
            "candidate_id": cid,
            "candidate_type": doc["candidate_type"],
            "content": pooling.display_content(doc),
            "relevance": "",
            "reason": "",
        })
    return rows


def _grade(row: dict) -> str:
    """The canonical integer grade of a sheet row; ValueError names the row."""
    qid, cid = (row.get("query_id") or "").strip(), (row.get("candidate_id") or "").strip()
    raw = (row.get("relevance") or "").strip()
    try:
        grade = int(raw)
    except ValueError:
        raise ValueError(
            f"row {qid}/{cid} has relevance {raw!r}: every row of {ANNOTATION} and "
            f"{ANNOTATION_TOPUP} must carry an integer grade 0-3 before {QRELS_V2} "
            f"can be built") from None
    if grade not in (0, 1, 2, 3):
        raise ValueError(f"row {qid}/{cid} has relevance {grade}: the grade must be 0-3")
    return str(grade)


def _key(row: dict) -> tuple[str, str]:
    return ((row.get("query_id") or "").strip(), (row.get("candidate_id") or "").strip())


def merge_qrels(annotation_rows: list[dict], topup_rows: list[dict]) -> list[dict]:
    """Qrels rows for the annotation sheet first, then the graded top-up sheet.

    Every input row must carry an integer grade in 0-3, and a top-up pair the
    annotation sheet already grades is an error (a stale sheet), not something
    to quietly de-duplicate.
    """
    annotation_keys = {_key(row) for row in annotation_rows}

    out: list[dict] = []
    for sheet, rows in ((ANNOTATION, annotation_rows), (ANNOTATION_TOPUP, topup_rows)):
        for row in rows:
            qid, cid = _key(row)
            if sheet == ANNOTATION_TOPUP and (qid, cid) in annotation_keys:
                raise ValueError(
                    f"top-up row {qid}/{cid} is already graded in {ANNOTATION}; rebuild "
                    f"the top-up sheet against the current annotation.csv")
            out.append({
                "query_id": qid,
                "candidate_id": cid,
                "relevance": _grade(row),
                "answer_bearing": (row.get("answer_bearing") or "0").strip() or "0",
                "reason": (row.get("reason") or "").strip(),
            })

    return out


# ---- CLI --------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _has_grades(path: Path) -> bool:
    return any((row.get("relevance") or "").strip() for row in read_csv(path))


def cmd_build(corpus_version: str, k: int) -> int:
    data = common.data_dir(corpus_version, create=False)
    annotation_path = data / ANNOTATION
    if not annotation_path.exists():
        raise SystemExit(f"{annotation_path} not found: it is the owner's graded sheet "
                         f"and is the judged set for the top-up")
    out_path = data / ANNOTATION_TOPUP
    if out_path.exists() and _has_grades(out_path):
        raise SystemExit(f"{out_path} already holds grades; refusing to overwrite "
                         f"grading work. Move it aside by hand if it is stale.")

    docs = common.load_documents(corpus_version)
    docs_by_id = {d["candidate_id"]: d for d in docs}
    queries_by_id = {q["query_id"]: q for q in common.load_queries()}
    qrels = common.load_qrels(annotation_path, valid_ids=set(docs_by_id))

    runs = {run_id: common.load_run(corpus_version, run_id) for run_id in LADDER_RUN_IDS}
    pairs = unjudged_pairs(runs, qrels, k=k)
    rows = topup_rows(pairs, queries_by_id, docs_by_id)

    write_csv(out_path, rows, common.ANNOTATION_COLUMNS)
    print(f"{common.utc_now()}  {len(rows)} unjudged top-{k} pair(s) across "
          f"{len(runs)} ladder run(s) -> {out_path}")
    if len(rows) > TOPUP_ESCALATE_ROWS:
        print(f"ESCALATE: {len(rows)} rows is above the {TOPUP_ESCALATE_ROWS}-row "
              f"planning bound (spec section 17) -- stop and report before grading.")
        return 3
    return 0


def cmd_merge(corpus_version: str) -> int:
    data = common.data_dir(corpus_version, create=False)
    annotation_path = data / ANNOTATION
    topup_path = data / ANNOTATION_TOPUP
    for path in (annotation_path, topup_path):
        if not path.exists():
            raise SystemExit(f"{path} not found: run 'uv run python -m evals.topup build' "
                             f"and grade the sheet before merging")

    rows = merge_qrels(read_csv(annotation_path), read_csv(topup_path))
    out_path = data / QRELS_V2
    write_csv(out_path, rows, QRELS_COLUMNS)
    print(f"{common.utc_now()}  wrote {len(rows)} judged row(s) -> {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help=f"write {ANNOTATION_TOPUP} from the ladder runs")
    build.add_argument("--k", type=int, default=TOPUP_K)
    sub.add_parser("merge", help=f"write {QRELS_V2} from {ANNOTATION} + {ANNOTATION_TOPUP}")
    args = ap.parse_args(argv)

    if args.command == "build":
        return cmd_build(args.corpus_version, args.k)
    return cmd_merge(args.corpus_version)


if __name__ == "__main__":
    raise SystemExit(main())