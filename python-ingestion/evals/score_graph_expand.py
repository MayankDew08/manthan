"""U8 score_graph_expand: score sets, pick branches, write Appendix D (spec U8/13.8).

The two query sets are always reported separately (never pooled) and the
branch rules are pre-registered (D16). Decision functions take no latency
input (D17) -- latency is reported next to a branch, never fed into one.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from evals import common, ladder_stats, topup

STARTS = (("r", "graph_b2r_ctx_rr"), ("q", "graph_b2q_ctx_rr"))

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
    """Share of top-k grade>=2 graph-sourced hits by via.path (U8-B7)."""
    ranked = sorted(rows, key=lambda r: r["rank"])[:k]
    path_counts: dict[str, int] = {}
    hits = 0
    for row in ranked:
        grade = qrels.get(row["query_id"], {}).get(row["candidate_id"], {}).get("relevance", 0)
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


def render_appendix(result: dict[str, Any]) -> bytes:
    """## Appendix D, the D18 disclosure, the D24 grader-view note, and the result."""
    lines = [
        APPENDIX_HEADER, "",
        D18_DISCLOSURE, "",
        D24_GRADER_NOTE, "",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), "",
    ]
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
            s2_w, s2_l = _graph_win_loss(per_system, "ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr")
            step1 = bool(graph_beats(n, s2_w, s2_l))
            c0_w, c0_l = _graph_win_loss(per_system, "ctrl_c0_deeper_rr", b2)
            c1_w, c1_l = _graph_win_loss(per_system, "ctrl_c1_denorm_rr", b2)
            c2_w, c2_l = _graph_win_loss(per_system, "ctrl_c2_payload_rr", b2)
            step2 = bool(graph_beats(n, c0_w, c0_l))
            step3 = bool(graph_beats(n, c1_w, c1_l)) and bool(graph_beats(n, c2_w, c2_l))

        out[start] = decide_branch(step1, step2, step3)
    return out


def attach_predictions(
    comparisons: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Each comparison gets ``prediction`` and ``prediction_held`` from the locked file."""
    out = []
    for c in comparisons:
        pred = predictions.get(f"{c['base']}->{c['new']}")
        if pred is None:
            out.append({**c, "prediction": None, "prediction_held": None})
        else:
            expected = pred["expected_verdict"]
            out.append({**c, "prediction": expected, "prediction_held": expected == c["verdict"]})
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

    predictions_path = common.EVALS_DIR / "graph_expand_predictions.json"
    predictions = (json.loads(predictions_path.read_text(encoding="utf-8"))
                   if predictions_path.exists() else {})

    result: dict[str, Any] = {"sets": {}}
    for set_name, set_result in scored.items():
        qids = queries[set_name]
        decisions = decide_for_set(set_name, set_result)
        result["sets"][set_name] = {
            "comparisons": attach_predictions(set_result["comparisons"], predictions),
            "decisions": decisions,
            "path_split": {
                start: path_split(runs[set_name].get(b2, []), qrels[set_name])
                for start, b2 in STARTS
            },
            "metrics": {
                rid: {
                    "ndcg_at_5": sum(v["ndcg"] for v in set_result["per_system"][rid].values())
                                / len(qids) if qids else 0.0,
                    "recall_pool": _avg_over_queries(recall_pool, runs[set_name].get(rid, []),
                                                     qrels[set_name], qids),
                    "judged_at_5": _avg_over_queries(judged_at_5, runs[set_name].get(rid, []),
                                                     qrels[set_name], qids),
                }
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
