"""Stage 6: score a run against human judgments.

Reads evals/data/<version>/{documents.jsonl,runs/<run>.jsonl,qrels.csv} and writes
metrics.json and report.md next to them. Grades 2-3 are relevant; unjudged = 0.

Usage (from python-ingestion/):
  uv run python -m evals.evaluate [--corpus-version pilot-001] [--run-id vector_cosine_v1]
  uv run python -m evals.evaluate --qrels-from-annotation   # build qrels.csv first
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common, metrics as M  # noqa: E402

THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(9)]  # 0.20 .. 0.60
PREDICTIONS_FILE = common.EVALS_DIR / "predictions.json"


def annotation_to_qrels(annotation_csv: Path, qrels_csv: Path) -> int:
    """Copy judged rows of the annotation sheet into qrels.csv. Returns row count."""
    rows = []
    with open(annotation_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("relevance") or "").strip() != "":
                rows.append({"query_id": r["query_id"], "candidate_id": r["candidate_id"],
                             "relevance": r["relevance"].strip(), "answer_bearing": "",
                             "reason": (r.get("reason") or "").strip()})
    with open(qrels_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "candidate_id", "relevance",
                                          "answer_bearing", "reason"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def ranked_rows(run_rows: list[dict], docs_by_id: dict, judged: dict) -> dict[str, list[dict]]:
    """Per query: rows in rank order with grade (unjudged = 0) and doc facts."""
    by_query: dict[str, list[dict]] = {}
    for r in sorted(run_rows, key=lambda r: (r["query_id"], r["rank"])):
        doc = docs_by_id[r["candidate_id"]]
        grade = judged.get(r["query_id"], {}).get(r["candidate_id"], {}).get("relevance", 0)
        by_query.setdefault(r["query_id"], []).append({
            "candidate_id": r["candidate_id"], "rank": r["rank"], "score": r["score"],
            "grade": grade, "truncated": doc["truncated"],
            "intro": common.looks_like_intro(doc["embedded_text"]) if
            doc["candidate_type"] == "message" else False,
        })
    return by_query


def evaluate_query(rows: list[dict], judged_grades: list[int]) -> dict:
    grades = [r["grade"] for r in rows]
    first_rel = M.first_rank(grades)
    best3 = M.first_rank(grades, min_grade=3)
    total_rel = sum(1 for g in judged_grades if M.is_relevant(g))
    above = rows[:first_rel - 1] if first_rel else []
    return {
        "best_grade3_rank": best3,
        "first_relevant_rank": first_rel,
        "ndcg_at_5": M.ndcg_at_k(grades, judged_grades, M.TOP_K),
        "success_at_5": M.success_at_k(grades, M.TOP_K),
        "candidate_success_at_30": M.success_at_k(grades, M.CANDIDATE_DEPTH),
        "mrr_at_5": M.mrr_at_k(grades, M.TOP_K),
        "recall_at_5": M.recall_at_k(grades, total_rel, M.TOP_K),
        "failure_class": M.classify_failure(rows),
        "distractors_above_first_relevant": (first_rel - 1) if first_rel else None,
        "intro_distractors": sum(1 for r in above if r["intro"]) if first_rel else None,
        "top1_score": rows[0]["score"] if rows else None,
        "roc_auc": M.roc_auc([r["score"] for r in rows], [M.is_relevant(g) for g in grades]),
        "n_judged": len(judged_grades),
        "n_relevant": total_rel,
        "relevant_docs": [{"candidate_id": r["candidate_id"], "grade": r["grade"],
                           "rank": r["rank"], "score": round(r["score"], 4),
                           "truncated": r["truncated"]}
                          for r in rows if M.is_relevant(r["grade"])],
    }


def evaluate(queries, docs_by_id, run_rows, qrels, predictions=None) -> dict:
    ranked = ranked_rows(run_rows, docs_by_id, qrels)
    per_query, unlabelled = {}, []
    for q in queries:
        qid = q["query_id"]
        if qid not in qrels or not qrels[qid]:
            unlabelled.append(qid)
            continue
        per_query[qid] = evaluate_query(ranked[qid],
                                        [v["relevance"] for v in qrels[qid].values()])
        per_query[qid]["intent"] = q.get("intent")
    vals = per_query.values()
    n = len(per_query)
    agg = {
        "queries_evaluated": n,
        "queries_unlabelled": unlabelled,
        "candidate_success_at_30": M.mean([float(v["candidate_success_at_30"]) for v in vals]),
        "success_at_5": M.mean([float(v["success_at_5"]) for v in vals]),
        "mrr_at_5": M.mean([v["mrr_at_5"] for v in vals]),
        "ndcg_at_5": M.mean([v["ndcg_at_5"] for v in vals]),
        "recall_at_5": M.mean([v["recall_at_5"] for v in vals]),
        "pooled_roc_auc": M.roc_auc(
            [r["score"] for qid in per_query for r in ranked[qid]],
            [M.is_relevant(r["grade"]) for qid in per_query for r in ranked[qid]]),
        "failure_breakdown": dict(Counter(v["failure_class"] for v in vals)),
        "threshold_sweep": M.threshold_sweep([ranked[qid] for qid in per_query], THRESHOLDS),
    }
    result = {"aggregate": agg, "per_query": per_query}
    if predictions:
        result["predictions"] = {
            qid: {"expected": predictions[qid]["expected"],
                  "actual": "success" if per_query[qid]["success_at_5"] else "failure",
                  "actual_class": per_query[qid]["failure_class"],
                  "match": predictions[qid]["expected"] == (
                      "success" if per_query[qid]["success_at_5"] else "failure"),
                  "reason": predictions[qid].get("reason", "")}
            for qid in per_query if qid in predictions}
    return result


def _f(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def render_report(result: dict, meta: dict, run_id: str, corpus_version: str) -> str:
    agg, pq = result["aggregate"], result["per_query"]
    L = [f"# Retrieval report: {run_id} on {corpus_version}", "",
         f"Corpus: {meta.get('point_count')} documents "
         f"({meta.get('by_type')}), {meta.get('truncated_documents')} truncated at "
         f"{meta.get('max_seq_length')} tokens. Queries evaluated: "
         f"{agg['queries_evaluated']}"
         + (f" (unlabelled, skipped: {', '.join(agg['queries_unlabelled'])})"
            if agg["queries_unlabelled"] else "") + ".", "",
         "> Pilot size: averages over this few queries are noise. Read the per-query "
         "table, not the means. Predictions below are hypotheses, not results.", "",
         "## Aggregates", "", "| Metric | Value |", "| --- | ---: |"]
    for key, label in (("candidate_success_at_30", "Candidate Success@30"),
                       ("success_at_5", "Success@5"), ("mrr_at_5", "MRR@5"),
                       ("ndcg_at_5", "nDCG@5 (graded, primary)"),
                       ("recall_at_5", "Recall@5"), ("pooled_roc_auc", "Pooled ROC-AUC")):
        L.append(f"| {label} | {_f(agg[key])} |")
    L += ["", "## Per query", "",
          "| query | intent | best grade-3 rank | first grade≥2 rank | nDCG@5 | Success@5 | "
          "MRR@5 | Recall@5 | failure class | distractors (intro) | top-1 score | AUC |",
          "| --- | --- | ---: | ---: | ---: | :---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for qid, v in pq.items():
        d = "n/a" if v["distractors_above_first_relevant"] is None else \
            f"{v['distractors_above_first_relevant']} ({v['intro_distractors']})"
        L.append(f"| {qid} | {v.get('intent') or ''} | {_f(v['best_grade3_rank'])} | "
                 f"{_f(v['first_relevant_rank'])} | {_f(v['ndcg_at_5'])} | "
                 f"{_f(v['success_at_5'])} | {_f(v['mrr_at_5'])} | {_f(v['recall_at_5'])} | "
                 f"{v['failure_class']} | {d} | {_f(v['top1_score'])} | {_f(v['roc_auc'])} |")
    L += ["", "## Failure breakdown", "", "| Class | Queries |", "| --- | ---: |"]
    for cls in M.FAILURE_CLASSES:
        note = " (not assessable from a Qdrant-sourced corpus)" if cls == "not_vectorized" else ""
        L.append(f"| {cls}{note} | {agg['failure_breakdown'].get(cls, 0)} |")
    L += ["", "## Relevant documents per query", ""]
    for qid, v in pq.items():
        L.append(f"- **{qid}**: " + (", ".join(
            f"g{d['grade']}@{d['rank']}{' (truncated)' if d['truncated'] else ''}"
            for d in v["relevant_docs"]) or "none judged relevant"))
    L += ["", "## Threshold sweep (pooled; relevant = grade ≥ 2)", "",
          f"The product currently applies {M.PRODUCT_THRESHOLD}.", "",
          "| threshold | returned | precision | recall | Success@5 |",
          "| ---: | ---: | ---: | ---: | ---: |"]
    for t in agg["threshold_sweep"]:
        mark = " ← current" if abs(t["threshold"] - M.PRODUCT_THRESHOLD) < 1e-9 else ""
        L.append(f"| {t['threshold']:.2f}{mark} | {t['returned']} | {_f(t['precision'])} | "
                 f"{_f(t['recall'])} | {_f(t['success_at_k'])} |")
    if "predictions" in result:
        L += ["", "## Predictions vs outcomes", "",
              "| query | predicted | actual | actual class | match |",
              "| --- | --- | --- | --- | :---: |"]
        for qid, p in result["predictions"].items():
            L.append(f"| {qid} | {p['expected']} | {p['actual']} | {p['actual_class']} | "
                     f"{_f(p['match'])} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    ap.add_argument("--run-id", default=common.DEFAULT_RUN_ID)
    ap.add_argument("--queries", type=Path, default=common.DEFAULT_QUERIES)
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS_FILE)
    ap.add_argument("--qrels-from-annotation", action="store_true",
                    help="build qrels.csv from the filled annotation.csv first")
    args = ap.parse_args(argv)

    out = common.data_dir(args.corpus_version)
    if args.qrels_from_annotation:
        n = annotation_to_qrels(out / "annotation.csv", out / "qrels.csv")
        print(f"Wrote {n} judged rows to {out / 'qrels.csv'}")
    if not (out / "qrels.csv").exists():
        print("FAIL: qrels.csv not found. Label annotation.csv, then rerun with "
              "--qrels-from-annotation.")
        return 1

    docs = common.load_documents(args.corpus_version)
    docs_by_id = {d["candidate_id"]: d for d in docs}
    qrels = common.load_qrels(out / "qrels.csv", set(docs_by_id))
    run_rows = common.load_run(args.corpus_version, args.run_id)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8")) \
        if args.predictions.exists() else None
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))

    result = evaluate(common.load_queries(args.queries), docs_by_id, run_rows, qrels,
                      predictions)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(result, meta, args.run_id,
                                                 args.corpus_version), encoding="utf-8")
    print(f"Wrote {out / 'metrics.json'} and {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
