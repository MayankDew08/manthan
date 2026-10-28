"""U8 score_graph_expand: score sets, pick branches, write Appendix D (spec U8/13.8).

The two query sets are always reported separately (never pooled) and the
branch rules are pre-registered (D16). Decision functions take no latency
input (D17) -- latency is reported next to a branch, never fed into one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals import common, ladder_stats, topup

STARTS = (("r", "graph_b2r_ctx_rr"), ("q", "graph_b2q_ctx_rr"))

# Predictions-file short names -> run ids (spec section 11).
SHORT_NAMES = {
    "ladder_s2_cos_bm25_rr": "S2",
    "ctrl_c0_deeper_rr": "C0",
    "ctrl_c1_denorm_rr": "C1",
    "ctrl_c2_payload_rr": "C2",
    "graph_b1r_rr": "B1-r",
    "graph_b2r_ctx_rr": "B2-r",
    "graph_b1q_rr": "B1-q",
    "graph_b2q_ctx_rr": "B2-q",
}

SYSTEMS = (
    "ladder_s2_cos_bm25_rr",
    "ctrl_c0_deeper_rr",
    "ctrl_c1_denorm_rr",
    "ctrl_c2_payload_rr",
    "graph_b1r_rr",
    "graph_b2r_ctx_rr",
    "graph_b1q_rr",
    "graph_b2q_ctx_rr",
)

# Spec section 10, in order (U8-B3).
COMPARISONS = (
    ("ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr"),
    ("ladder_s2_cos_bm25_rr", "ctrl_c1_denorm_rr"),
    ("ladder_s2_cos_bm25_rr", "ctrl_c2_payload_rr"),
    ("ctrl_c0_deeper_rr", "graph_b2r_ctx_rr"),
    ("ctrl_c1_denorm_rr", "graph_b2r_ctx_rr"),
    ("ctrl_c2_payload_rr", "graph_b2r_ctx_rr"),
    ("ctrl_c0_deeper_rr", "graph_b2q_ctx_rr"),
    ("ctrl_c1_denorm_rr", "graph_b2q_ctx_rr"),
    ("ctrl_c2_payload_rr", "graph_b2q_ctx_rr"),
    ("graph_b1r_rr", "graph_b2r_ctx_rr"),
    ("graph_b1q_rr", "graph_b2q_ctx_rr"),
)

ORIGINAL_BEATS_MIN_DELTA = 0.02
# D16: n answerable graph queries -> wins needed for 0 losses to "beat". n<=4: too few.
SCOREBOARD = {8: 7, 7: 7, 6: 6, 5: 5}

APPENDIX_HEADER = "## Appendix D — Graph as candidate generator"

D18_DISCLOSURE = (
    "Disclosure (D18): the planner drafted the 8 questions (2026-10-23) from "
    "documents.jsonl text only, and the owner adjusted them. The planner had "
    "already seen connector degrees (section 2.3) before drafting."
)

D24_GRADER_NOTE = (
    "grader view (D24): the original-set top-up sheet shows doc text only, "
    "as Run A. The graph-set sheet shows message text plus the titles of "
    "its in-corpus shared links, in candidate_id order."
)


def score_sets(
    runs: dict[str, dict[str, list[dict[str, Any]]]],
    qrels: dict[str, dict[str, dict[str, dict[str, Any]]]],
    queries: dict[str, list[str]],
    systems: tuple[str, ...] = SYSTEMS,
) -> dict[str, dict[str, Any]]:
    """Score every requested system, per set, never pooling the two sets.

    Calls ``topup.assert_fully_judged`` for every (set, system) first, across
    all sets, before any metric is computed for any set (U8-B1).
    """
    for set_name, set_runs in runs.items():
        scoped = {rid: set_runs.get(rid, []) for rid in systems if rid in set_runs}
        topup.assert_fully_judged(scoped, qrels.get(set_name, {}), k=topup.TOPUP_K)

    result: dict[str, dict[str, Any]] = {}
    for set_name, set_runs in runs.items():
        set_qrels = qrels.get(set_name, {})
        set_query_ids = ladder_stats.answerable(set_qrels, queries.get(set_name, []))

        per_system = {
            rid: ladder_stats.per_query(set_runs.get(rid, []), set_qrels, set_query_ids)
            for rid in systems
        }

        comparisons = []
        for base, new in COMPARISONS:
            if base not in per_system or new not in per_system:
                continue
            base_ndcg = {qid: v["ndcg"] for qid, v in per_system[base].items()}
            new_ndcg = {qid: v["ndcg"] for qid, v in per_system[new].items()}
            boot = ladder_stats.paired_bootstrap(base_ndcg, new_ndcg)
            comparisons.append({
                "base": base, "new": new,
                "verdict": ladder_stats.verdict(boot["ci_low"], boot["ci_high"], boot["n"]),
                **boot,
            })

        result[set_name] = {"per_system": per_system, "comparisons": comparisons}
    return result


def beats_original(base: dict[str, float], new: dict[str, float], verdict: str) -> bool:
    """Original set: ladder_stats.verdict == improved and mean delta >= 0.02."""
    if verdict != "improved":
        return False
    deltas = [new[qid] - base[qid] for qid in base]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return mean_delta >= ORIGINAL_BEATS_MIN_DELTA


def graph_beats(n: int, wins: int, losses: int) -> bool | None:
    """D16 scoreboard: 0 losses and the threshold win count for n; None if n <= 4."""
    if n <= 4:
        return None
    required_wins = SCOREBOARD.get(n)
    if required_wins is None:
        raise ValueError(f"graph_beats: no scoreboard threshold registered for n={n}")
    return losses == 0 and wins >= required_wins


def decide_branch(
    step1_deeper_pool: bool, step2_graph_over_c0: bool, step3_graph_over_shortcuts: bool,
) -> dict[str, Any]:
    """Spec section 6's decision flow. Step 1 is recorded, not a gate."""
    if not step2_graph_over_c0:
        branch = "GRAPH OUT"
    elif not step3_graph_over_shortcuts:
        branch = "SHIP THE SHORTCUT(S)"
    else:
        branch = "GRAPH STAYS"
    return {
        "step1_deeper_pool": step1_deeper_pool,
        "step2_graph_over_c0": step2_graph_over_c0,
        "step3_graph_over_shortcuts": step3_graph_over_shortcuts,
        "branch": branch,
    }


def path_split(rows: list[dict[str, Any]], qrels: dict[str, dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Share of top-k grade>=2 graph-sourced hits by via.path (U8-B7).

    ``rows`` spans every query a run answered, so the top-k must be taken
    per query_id and the hits summed across queries -- a single global
    top-k over the pooled rows would let one query's rows crowd out
    another's (review follow-up)."""
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)

    path_counts: dict[str, int] = {}
    hits = 0
    for qid, qrows in by_query.items():
        ranked = sorted(qrows, key=lambda r: r["rank"])[:k]
        for row in ranked:
            grade = qrels.get(qid, {}).get(row["candidate_id"], {}).get("relevance", 0)
            via = row.get("via")
            if grade >= 2 and via:
                hits += 1
                path_counts[via["path"]] = path_counts.get(via["path"], 0) + 1
    if hits == 0:
        return {"hits": 0, "shares": {}}
    return {"hits": hits, "shares": {p: c / hits for p, c in path_counts.items()}}


def recall_pool(rows: list[dict[str, Any]], qrels: dict[str, dict[str, Any]]) -> float:
    """Grade>=2 docs anywhere in ``rows`` / all grade>=2 docs for that query."""
    if not rows:
        return 0.0
    qid = rows[0]["query_id"]
    all_relevant = {cid for cid, info in qrels.get(qid, {}).items() if info.get("relevance", 0) >= 2}
    if not all_relevant:
        return 0.0
    ranked_ids = {r["candidate_id"] for r in rows}
    return len(all_relevant & ranked_ids) / len(all_relevant)


def judged_at_5(rows: list[dict[str, Any]], qrels: dict[str, dict[str, Any]], k: int = 5) -> float:
    """Judged top-k rows / k."""
    if not rows or k <= 0:
        return 0.0
    qid = rows[0]["query_id"]
    doc_qrels = qrels.get(qid, {})
    top_k = sorted(rows, key=lambda r: r["rank"])[:k]
    return sum(1 for r in top_k if r["candidate_id"] in doc_qrels) / k


METRIC_ROWS = ("ndcg_at_5", "recall_pool", "judged_at_5", "n")


def _fmt_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _short(run_id: str) -> str:
    return SHORT_NAMES.get(run_id, run_id)


def _metrics_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    header = "| metric | " + " | ".join(_short(rid) for rid in SYSTEMS) + " |"
    sep = "|---" * (len(SYSTEMS) + 1) + "|"
    lines = [header, sep]
    for metric in METRIC_ROWS:
        cells = [_fmt_cell(metrics.get(rid, {}).get(metric, "")) for rid in SYSTEMS]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return lines


def _verdicts_table(comparisons: list[dict[str, Any]]) -> list[str]:
    header = "| base | new | n | mean_delta | ci_low | ci_high | verdict |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for c in comparisons:
        lines.append(
            f"| {_short(c['base'])} | {_short(c['new'])} | {c.get('n', '')} | "
            f"{_fmt_cell(c.get('mean_delta', ''))} | {_fmt_cell(c.get('ci_low', ''))} | "
            f"{_fmt_cell(c.get('ci_high', ''))} | {c['verdict']} |"
        )
    return lines


def _predictions_table(comparisons: list[dict[str, Any]]) -> list[str]:
    header = ("| base | new | owner_prediction | owner_prediction_held | "
              "planner_prediction | planner_prediction_held |")
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for c in comparisons:
        lines.append(
            f"| {_short(c['base'])} | {_short(c['new'])} | "
            f"{c.get('owner_prediction')} | {c.get('owner_prediction_held')} | "
            f"{c.get('planner_prediction')} | {c.get('planner_prediction_held')} |"
        )
    return lines


def _decisions_table(decisions: dict[str, dict[str, Any]]) -> list[str]:
    header = "| start | branch | step1_deeper_pool | step2_graph_over_c0 | step3_graph_over_shortcuts |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for start, d in decisions.items():
        lines.append(
            f"| {start} | {d.get('branch')} | {d.get('step1_deeper_pool')} | "
            f"{d.get('step2_graph_over_c0')} | {d.get('step3_graph_over_shortcuts')} |"
        )
    return lines


def _path_split_table(path_splits: dict[str, dict[str, Any]]) -> list[str]:
    header = "| start | hits | shares |"
    sep = "|---|---|---|"
    lines = [header, sep]
    for start, p in path_splits.items():
        shares = ", ".join(f"{path}={share:.2f}" for path, share in p.get("shares", {}).items())
        lines.append(f"| {start} | {p.get('hits', 0)} | {shares} |")
    return lines


def render_appendix(result: dict[str, Any]) -> bytes:
    """## Appendix D, the D18 disclosure, the D24 grader-view note, then
    per-set markdown tables (metrics, verdicts, predictions held, decisions,
    path split) -- not a raw JSON dump of ``result``."""
    lines = [
        APPENDIX_HEADER, "",
        D18_DISCLOSURE, "",
        D24_GRADER_NOTE, "",
    ]
    for set_name, set_result in result.get("sets", {}).items():
        lines.append(f"### {set_name.capitalize()} set")
        lines.append("")
        lines.append("**Metrics**")
        lines.append("")
        lines.extend(_metrics_table(set_result.get("metrics", {})))
        lines.append("")
        lines.append("**Verdicts**")
        lines.append("")
        lines.extend(_verdicts_table(set_result.get("comparisons", [])))
        lines.append("")
        lines.append("**Predictions held**")
        lines.append("")
        lines.extend(_predictions_table(set_result.get("comparisons", [])))
        lines.append("")
        lines.append("**Decisions**")
        lines.append("")
        lines.extend(_decisions_table(set_result.get("decisions", {})))
        lines.append("")
        lines.append("**Path split**")
        lines.append("")
        lines.extend(_path_split_table(set_result.get("path_split", {})))
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def append_appendix(report: bytes, appendix: bytes) -> bytes:
    """Append ``appendix`` to ``report``'s existing bytes; raise if already present."""
    if APPENDIX_HEADER.encode("utf-8") in report:
        raise ValueError(f"{APPENDIX_HEADER} is already present in the report")
    sep = b"" if report.endswith(b"\n\n") else (b"\n" if report.endswith(b"\n") else b"\n\n")
    return report + sep + appendix


def _comparison_index(comparisons: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(c["base"], c["new"]): c for c in comparisons}


def _original_delta_beats(verdict: str, mean_delta: float) -> bool:
    return verdict == "improved" and mean_delta >= ORIGINAL_BEATS_MIN_DELTA


def _graph_win_loss(per_system: dict[str, Any], base: str, new: str) -> tuple[int, int]:
    wtl = ladder_stats.win_tie_loss(
        {qid: v["ndcg"] for qid, v in per_system[base].items()},
        {qid: v["ndcg"] for qid, v in per_system[new].items()})
    return wtl["wins"], wtl["losses"]


def decide_for_set(set_name: str, set_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Step 1/2/3 per start (r, q), from one set's score_sets() output.

    Original set (D11/D16's companion rule, section 6): step N holds when
    ``beats_original``'s rule does (bootstrap verdict + mean delta >= 0.02),
    read straight from the comparison's own ``verdict``/``mean_delta`` --
    not re-derived from raw per-query scores.
    Graph set: step N holds when ``graph_beats`` (D16 scoreboard) does, fed
    by ``ladder_stats.win_tie_loss`` over the raw per-query ndcg in
    ``per_system`` (a bootstrap verdict alone can't give win/loss counts).
    """
    comparisons = _comparison_index(set_result["comparisons"])
    per_system = set_result["per_system"]

    def pair(base: str, new: str) -> dict[str, Any]:
        c = comparisons.get((base, new))
        if c is None:
            raise ValueError(f"missing comparison {base}->{new}")
        return c

    s2_c0 = pair("ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr")

    out: dict[str, dict[str, Any]] = {}
    for start, b2 in STARTS:
        if set_name == "original":
            step1 = _original_delta_beats(s2_c0["verdict"], s2_c0["mean_delta"])
            c0 = pair("ctrl_c0_deeper_rr", b2)
            c1 = pair("ctrl_c1_denorm_rr", b2)
            c2 = pair("ctrl_c2_payload_rr", b2)
            step2 = _original_delta_beats(c0["verdict"], c0["mean_delta"])
            step3 = (_original_delta_beats(c1["verdict"], c1["mean_delta"])
                     and _original_delta_beats(c2["verdict"], c2["mean_delta"]))
        else:
            n = len(per_system["ctrl_c0_deeper_rr"])
            if n <= 4:
                # D16: "too few to decide" is its own outcome -- never coerce
                # graph_beats' None into a False branch decision (review #5 P2).
                out[start] = {"branch": "TOO FEW TO DECIDE", "n": n}
                continue
            s2_w, s2_l = _graph_win_loss(per_system, "ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr")
            step1 = bool(graph_beats(n, s2_w, s2_l))
            c0_w, c0_l = _graph_win_loss(per_system, "ctrl_c0_deeper_rr", b2)
            c1_w, c1_l = _graph_win_loss(per_system, "ctrl_c1_denorm_rr", b2)
            c2_w, c2_l = _graph_win_loss(per_system, "ctrl_c2_payload_rr", b2)
            step2 = bool(graph_beats(n, c0_w, c0_l))
            step3 = bool(graph_beats(n, c1_w, c1_l)) and bool(graph_beats(n, c2_w, c2_l))

        out[start] = decide_branch(step1, step2, step3)
    return out


def _prediction_holds(expected: str, verdict: str) -> bool:
    """A prediction of "X" holds when the verdict is exactly "X", or carries
    ladder_stats.verdict's sample-size suffix ("X at n=<k>"). An exact-string
    match alone missed every prediction against a suffixed verdict."""
    return verdict == expected or verdict.startswith(f"{expected} at n=")


def attach_predictions(
    comparisons: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Each comparison gets ``prediction`` and ``prediction_held`` from a flat
    ``{"base->new": {"expected_verdict": ...}}`` map. For the real predictions
    file's shape (nested, owner/planner, short names), use
    ``attach_set_predictions`` instead."""
    out = []
    for c in comparisons:
        pred = predictions.get(f"{c['base']}->{c['new']}")
        if pred is None:
            out.append({**c, "prediction": None, "prediction_held": None})
        else:
            expected = pred["expected_verdict"]
            out.append({**c, "prediction": expected,
                        "prediction_held": _prediction_holds(expected, c["verdict"])})
    return out


def load_predictions(path: Path) -> dict[str, Any]:
    """The locked predictions file; raises if it's missing (review #5 P1c)."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"predictions file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_prediction_entry(predictions_for_set: Any, short_key: str) -> dict[str, Any] | None:
    """Look up ``short_key`` in one role's set block; "abstain" strings and
    combined keys (``"B1-q->B2-q and B1-r->B2-r"``) are handled."""
    if not isinstance(predictions_for_set, dict):
        return None  # e.g. "abstain (D18)"
    if short_key in predictions_for_set:
        return predictions_for_set[short_key]
    for key, entry in predictions_for_set.items():
        if key == "expected_branch" or not isinstance(entry, dict):
            continue
        if short_key in (part.strip() for part in key.split(" and ")):
            return entry
    return None


def attach_set_predictions(
    comparisons: list[dict[str, Any]], predictions: dict[str, Any], set_name: str,
) -> list[dict[str, Any]]:
    """Each comparison gets owner_prediction/owner_prediction_held and the
    same for planner, from the real predictions file (owner/planner nested
    under original_set/graph_set, short run names; review #5 P1b)."""
    set_key = f"{set_name}_set"
    out = []
    for c in comparisons:
        short_key = f"{SHORT_NAMES.get(c['base'], c['base'])}->{SHORT_NAMES.get(c['new'], c['new'])}"
        enriched = dict(c)
        for role in ("owner", "planner"):
            entry = _find_prediction_entry(predictions.get(role, {}).get(set_key), short_key)
            if entry is None:
                enriched[f"{role}_prediction"] = None
                enriched[f"{role}_prediction_held"] = None
            else:
                expected = entry["expected_verdict"]
                enriched[f"{role}_prediction"] = expected
                enriched[f"{role}_prediction_held"] = _prediction_holds(expected, c["verdict"])
        out.append(enriched)
    return out


# ---- CLI (review #4 K6) -----------------------------------------------------

def _rows_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["query_id"], []).append(row)
    return out


def _avg_over_queries(fn, rows: list[dict[str, Any]], qrels: dict[str, Any],
                       query_ids: list[str]) -> float:
    by_query = _rows_by_query(rows)
    vals = [fn(by_query.get(qid, []), qrels) for qid in query_ids]
    return sum(vals) / len(vals) if vals else 0.0


def system_metrics(
    rows: list[dict[str, Any]],
    qrels: dict[str, Any],
    per_system_entry: dict[str, Any],
    all_query_ids: list[str],
) -> dict[str, float]:
    """nDCG@5, recall_pool, judged@5, averaged over ladder_stats.answerable
    query ids only -- not every query id handed in (review #5 P3)."""
    answerable_ids = ladder_stats.answerable(qrels, all_query_ids)
    if not answerable_ids:
        return {"ndcg_at_5": 0.0, "recall_pool": 0.0, "judged_at_5": 0.0}
    ndcg_vals = [per_system_entry[qid]["ndcg"] for qid in answerable_ids if qid in per_system_entry]
    return {
        "ndcg_at_5": sum(ndcg_vals) / len(answerable_ids),
        "recall_pool": _avg_over_queries(recall_pool, rows, qrels, answerable_ids),
        "judged_at_5": _avg_over_queries(judged_at_5, rows, qrels, answerable_ids),
        "n": len(answerable_ids),
    }


def main(argv: list[str] | None = None) -> int:
    """Per set, per start: step 1 (S2->C0), step 2 (C0->B2-x), step 3
    (C1->B2-x and C2->B2-x). Then attach_predictions, render_appendix with
    the 4 metrics x 8 systems x 2 sets table and the path split."""
    parser = argparse.ArgumentParser(description="U8 score_graph_expand")
    parser.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    parser.add_argument("--append-report", action="store_true")
    args = parser.parse_args(argv)

    data_dir = common.data_dir(args.corpus_version, create=False)
    corpus = common.load_documents(args.corpus_version)
    valid_ids = {d["candidate_id"] for d in corpus}

    qrels_v3 = common.load_qrels(data_dir / "qrels_v3.csv", valid_ids=valid_ids)
    original_query_ids = [q["query_id"] for q in common.load_queries()]
    graphq_query_ids = [q["query_id"] for q in
                        common.load_queries(common.REPO_ROOT / "queries_graph.csv")]

    runs_dir = data_dir / "runs"
    runs = {
        "original": {rid: common.read_jsonl(runs_dir / f"{rid}.jsonl") for rid in SYSTEMS},
        "graph": {rid: common.read_jsonl(runs_dir / f"{rid}.graphq.jsonl") for rid in SYSTEMS},
    }
    qrels = {"original": qrels_v3, "graph": qrels_v3}
    queries = {"original": original_query_ids, "graph": graphq_query_ids}

    scored = score_sets(runs, qrels, queries)

    # A missing predictions file raises (P1c), not a silent {}.
    predictions = load_predictions(common.EVALS_DIR / "graph_expand_predictions.json")

    result: dict[str, Any] = {"sets": {}}
    for set_name, set_result in scored.items():
        qids = queries[set_name]
        decisions = decide_for_set(set_name, set_result)
        result["sets"][set_name] = {
            "comparisons": attach_set_predictions(set_result["comparisons"], predictions, set_name),
            "decisions": decisions,
            "path_split": {
                start: path_split(runs[set_name].get(b2, []), qrels[set_name])
                for start, b2 in STARTS
            },
            "metrics": {
                rid: system_metrics(runs[set_name].get(rid, []), qrels[set_name],
                                    set_result["per_system"][rid], qids)
                for rid in SYSTEMS
            },
        }
        for start, decision in decisions.items():
            print(f"{set_name} {start}: {decision['branch']}")

    (data_dir / "graph_expand_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    appendix = render_appendix(result)
    (data_dir / "graph_expand_report.md").write_bytes(appendix)

    if args.append_report:
        manthan_report_path = common.ROOT / "metrics" / "manthan_report.md"
        updated = append_appendix(manthan_report_path.read_bytes(), appendix)
        manthan_report_path.write_bytes(updated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
