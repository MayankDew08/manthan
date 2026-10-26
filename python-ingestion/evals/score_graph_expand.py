"""U8 score_graph_expand: score sets, pick branches, write Appendix D (spec U8/13.8).

The two query sets are always reported separately (never pooled) and the
branch rules are pre-registered (D16). Decision functions take no latency
input (D17) -- latency is reported next to a branch, never fed into one.
"""

from __future__ import annotations

import json
from typing import Any

from evals import ladder_stats, topup

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
