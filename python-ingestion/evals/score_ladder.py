"""U10: score the retrieval ladder and render Appendix C (spec section 14).

This module computes per-step metrics, pairwise comparisons with verdicts,
and renders the appendix for the final report.
"""

import json
from pathlib import Path
from typing import Any

from evals import common, ladder_stats as S, metrics as M, run_ladder, topup

# Pre-registered comparisons in exact order from spec section 4
COMPARISONS = (
    (run_ladder.STEPS[0], run_ladder.STEPS[1]),  # s0 -> s1
    (run_ladder.STEPS[1], run_ladder.STEPS[2]),  # s1 -> s2
    (run_ladder.STEPS[1], run_ladder.STEPS[3]),  # s1 -> s3
    (run_ladder.STEPS[3], run_ladder.STEPS[4]),  # s3 -> s4
    (run_ladder.STEPS[2], run_ladder.STEPS[4]),  # s2 -> s4
    (run_ladder.STEPS[0], run_ladder.STEPS[4]),  # s0 -> s4
)


def score(
    runs: dict[str, list[dict]],
    qrels: dict[str, dict[str, dict]],
    queries: list[dict],
    predictions: dict[str, Any],
) -> dict[str, Any]:
    """Score the five ladder steps and their comparisons.

    Args:
        runs: {run_id: rows} for all five ladder steps
        qrels: graded relevance judgments (qrels_v2)
        queries: list of query dicts with query_id, query_text, intent
        predictions: ladder_predictions.json content

    Returns:
        JSON-serialisable dict with steps, comparisons, abstention, and metadata.
    """
    # Gate: all top-5 pairs must be judged
    topup.assert_fully_judged(runs, qrels)

    # Determine answerable queries (any grade >= 2)
    query_ids = [q["query_id"] for q in queries]
    answerable_ids = S.answerable(qrels, query_ids)
    no_answer_ids = [qid for qid in query_ids if qid not in answerable_ids]

    # Per-step metrics over answerable queries only
    steps_result = {}
    for step in run_ladder.STEPS:
        per_q = S.per_query(runs[step], qrels, answerable_ids)
        mean_ndcg = sum(v["ndcg"] for v in per_q.values()) / len(per_q) if per_q else 0.0
        mean_success = sum(v["success"] for v in per_q.values()) / len(per_q) if per_q else 0.0
        mean_mrr = sum(v["mrr"] for v in per_q.values()) / len(per_q) if per_q else 0.0
        mean_candidate_success = sum(v["candidate_success"] for v in per_q.values()) / len(per_q) if per_q else 0.0

        unjudged_30 = S.unjudged_at(runs[step], qrels, answerable_ids, M.CANDIDATE_DEPTH)
        candidate_success_is_lower_bound = unjudged_30 > 0

        steps_result[step] = {
            "mean": {
                "ndcg": mean_ndcg,
                "success": mean_success,
                "mrr": mean_mrr,
                "candidate_success": mean_candidate_success,
            },
            "unjudged_at_30": unjudged_30,
            "candidate_success_is_lower_bound": candidate_success_is_lower_bound,
        }

    # Per-comparison results
    comparisons_result = []
    pred_comparisons = predictions.get("comparisons", {})

    for base, new in COMPARISONS:
        # Get per-query nDCG for base and new over answerable queries
        base_per_q = S.per_query(runs[base], qrels, answerable_ids)
        new_per_q = S.per_query(runs[new], qrels, answerable_ids)

        base_ndcg = {qid: base_per_q[qid]["ndcg"] for qid in answerable_ids}
        new_ndcg = {qid: new_per_q[qid]["ndcg"] for qid in answerable_ids}

        wtl = S.win_tie_loss(base_ndcg, new_ndcg)
        bootstrap = S.paired_bootstrap(base_ndcg, new_ndcg)
        verdict_str = S.verdict(bootstrap["ci_low"], bootstrap["ci_high"], bootstrap["n"])

        # Prediction lookup
        pred_key = f"{base}->{new}"
        pred = pred_comparisons.get(pred_key)
        prediction_held = None
        if pred:
            expected = pred.get("expected_verdict")
            prediction_held = verdict_str.startswith(expected)

        comparisons_result.append({
            "base": base,
            "new": new,
            "win_tie_loss": wtl,
            "bootstrap": bootstrap,
            "verdict": verdict_str,
            "prediction": pred,
            "prediction_held": prediction_held,
        })

    # Abstention only for s0, s2, s4
    abstention_result = {}
    for step in (run_ladder.STEPS[0], run_ladder.STEPS[2], run_ladder.STEPS[4]):
        abstention_result[step] = S.abstention(runs[step], no_answer_ids, answerable_ids)

    return {
        "answerable": answerable_ids,
        "no_answer": no_answer_ids,
        "n_answerable": len(answerable_ids),
        "steps": steps_result,
        "comparisons": comparisons_result,
        "abstention": abstention_result,
    }


def render_appendix(result: dict[str, Any], latency: dict[str, Any]) -> str:
    """Render Appendix C — Retrieval ladder as Markdown."""
    lines = ["## Appendix C — Retrieval ladder", ""]
    lines.append(f"n={result['n_answerable']} answerable queries.")
    lines.append("")

    # Per-step summary table
    lines.append("### Per-step metrics (answerable queries)")
    lines.append("")
    lines.append("| Step | nDCG@5 | Success@5 | MRR@5 | Candidate Success@30 (lower bound) |")
    lines.append("|------|--------|-----------|-------|-----------------------------------|")
    for step in run_ladder.STEPS:
        m = result["steps"][step]["mean"]
        lb_marker = " (lower bound)" if result["steps"][step]["candidate_success_is_lower_bound"] else ""
        lines.append(
            f"| {step} | {m['ndcg']:.4f} | {m['success']:.4f} | {m['mrr']:.4f} | "
            f"{m['candidate_success']:.4f}{lb_marker} |"
        )
    lines.append("")

    # Comparisons with verdicts verbatim
    lines.append("### Pairwise comparisons")
    lines.append("")
    for comp in result["comparisons"]:
        lines.append(f"**{comp['base']} → {comp['new']}**")
        lines.append(f"- Verdict: {comp['verdict']}")
        wtl = comp["win_tie_loss"]
        lines.append(
            f"- Win/tie/loss: {wtl['wins']} / {wtl['ties']} / {wtl['losses']} "
            f"(improved: {wtl['improved']}; regressed: {wtl['regressed']})"
        )
        bs = comp["bootstrap"]
        lines.append(
            f"- Bootstrap (n={bs['n']}): mean Δ={bs['mean_delta']:.4f}, "
            f"CI=[{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]"
        )
        if comp["prediction"]:
            pred = comp["prediction"]
            held = "✓" if comp["prediction_held"] else "✗"
            lines.append(
                f"- Prediction: expected \"{pred['expected_verdict']}\" ({pred['reason']}) — {held}"
            )
        else:
            lines.append("- Prediction: not pre-registered")
        lines.append("")

    # Abstention
    lines.append("### Abstention (no-answer queries)")
    lines.append("")
    for step in (run_ladder.STEPS[0], run_ladder.STEPS[2], run_ladder.STEPS[4]):
        ab = result["abstention"][step]
        lines.append(f"**{step}**")
        lines.append(f"- No-answer top-1 scores: {ab['no_answer_top1']}")
        lines.append(f"- Answerable top-1 scores: {ab['answerable_top1']}")
        lines.append(f"- Median no-answer: {ab['median_no_answer']:.4f}")
        lines.append(f"- Median answerable: {ab['median_answerable']:.4f}")
        lines.append("")

    # Latency
    lines.append("### End-to-end latency (estimate across machines)")
    lines.append("")
    if "machine" in latency:
        m = latency["machine"]
        lines.append(f"- Machine: {m.get('processor', 'unknown')}, {m.get('cpu_count', '?')} CPUs, GPU: {m.get('gpu', 'none')}")
    if "components" in latency:
        for name, comp in latency["components"].items():
            if "summary" in comp:
                s = comp["summary"]
                lines.append(f"- {name}: n={s['n']}, p50={s['p50']:.1f}ms, p95={s['p95']:.1f}ms")
    lines.append("")

    return "\n".join(lines)


def append_appendix(report_path: Path, text: str) -> None:
    """Append appendix to report, keeping existing bytes unchanged.

    Raises:
        ValueError: if "## Appendix C" is already present in the report.
    """
    existing = report_path.read_bytes()
    if b"## Appendix C" in existing:
        raise ValueError("## Appendix C already present; refusing second append")
    with open(report_path, "ab") as f:
        f.write(b"\n\n")
        f.write(text.encode("utf-8"))


# ---- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--append-report", action="store_true", help="Append to manthan_report.md")
    args = ap.parse_args(argv)

    corpus_version = common.DEFAULT_CORPUS_VERSION
    data_dir = common.data_dir(corpus_version, create=False)
    runs_dir = data_dir / "runs"

    # Load the five ladder runs
    runs = {}
    for step in run_ladder.STEPS:
        runs[step] = common.load_run(corpus_version, step)

    # Load qrels_v2.csv
    qrels_path = data_dir / "qrels_v2.csv"
    qrels = common.load_qrels(qrels_path, valid_ids=None)

    # Load queries
    queries = common.load_queries()

    # Load predictions
    pred_path = common.EVALS_DIR / "ladder_predictions.json"
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Load latency
    latency_path = runs_dir / "latency_v1.json"
    with open(latency_path, "r", encoding="utf-8") as f:
        latency = json.load(f)

    # Score
    result = score(runs, qrels, queries, predictions)

    # Write ladder_metrics.json
    out_metrics = data_dir / "ladder_metrics.json"
    result_with_qrels = dict(result)
    result_with_qrels["qrels_version"] = "qrels_v2.csv"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(result_with_qrels, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_metrics}")

    # Write ladder_report.md
    appendix = render_appendix(result, latency)
    out_report = data_dir / "ladder_report.md"
    out_report.write_text(appendix, encoding="utf-8")
    print(f"Wrote {out_report}")

    # Optionally append to manthan_report.md
    if args.append_report:
        manthan_report = common.ROOT / "metrics" / "manthan_report.md"
        append_appendix(manthan_report, appendix)
        print(f"Appended to {manthan_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())