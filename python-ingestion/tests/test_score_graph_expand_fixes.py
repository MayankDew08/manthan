"""Fixes for two score_graph_expand reporting bugs (U8 follow-up):

1. Prediction match: a prediction "X" holds when the verdict is exactly "X"
   or starts with "X at n=" (verdicts carry a sample-size suffix that an
   exact string match was missing).
2. path_split: must group rows by query_id, take each query's own top-k by
   rank, then sum hits across queries -- not take one global top-k across
   every query's rows pooled together.

New test file only; does not touch tests/test_score_graph_expand.py.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _score_graph_expand():
    from evals import score_graph_expand
    return score_graph_expand


def _row(run_id, qid, cid, rank, via=None):
    return {"run_id": run_id, "query_id": qid, "candidate_id": cid,
            "rank": rank, "score": 1.0, "via": via, "note": None}


def _qrels(pairs):
    out = {}
    for qid, cid, grade in pairs:
        out.setdefault(qid, {})[cid] = {"relevance": grade, "answer_bearing": 0}
    return out


class TestPredictionMatchWithSampleSizeSuffix:
    def test_prediction_holds_when_verdict_has_an_n_suffix(self):
        """"no detectable difference" must hold against a verdict that
        carries the "at n=<k>" suffix ladder_stats.verdict always appends."""
        sge = _score_graph_expand()
        comparisons = [
            {"base": "a", "new": "b", "verdict": "no detectable difference at n=22"},
        ]
        predictions = {"a->b": {"expected_verdict": "no detectable difference"}}

        out = sge.attach_predictions(comparisons, predictions)

        assert out[0]["prediction_held"] is True

    def test_prediction_does_not_hold_for_a_different_verdict(self):
        sge = _score_graph_expand()
        comparisons = [
            {"base": "a", "new": "b", "verdict": "no detectable difference at n=22"},
        ]
        predictions = {"a->b": {"expected_verdict": "improved"}}

        out = sge.attach_predictions(comparisons, predictions)

        assert out[0]["prediction_held"] is False

    def test_exact_match_without_a_suffix_still_holds(self):
        sge = _score_graph_expand()
        comparisons = [{"base": "a", "new": "b", "verdict": "regressed"}]
        predictions = {"a->b": {"expected_verdict": "regressed"}}

        out = sge.attach_predictions(comparisons, predictions)

        assert out[0]["prediction_held"] is True

    def test_does_not_false_match_a_different_verdict_sharing_a_prefix(self):
        """"improved" must never match "improved-ish" or similar near-miss
        text -- only itself or itself + " at n=..."."""
        sge = _score_graph_expand()
        comparisons = [{"base": "a", "new": "b", "verdict": "improved-ish"}]
        predictions = {"a->b": {"expected_verdict": "improved"}}

        out = sge.attach_predictions(comparisons, predictions)

        assert out[0]["prediction_held"] is False

    def test_attach_set_predictions_also_honors_the_n_suffix(self):
        sge = _score_graph_expand()
        comparisons = [
            {"base": "ladder_s2_cos_bm25_rr", "new": "ctrl_c1_denorm_rr",
             "verdict": "no detectable difference at n=22"},
        ]
        predictions = {
            "owner": {"original_set": {"S2->C1": {"expected_verdict": "no detectable difference"}}},
            "planner": {"original_set": {}},
        }

        out = sge.attach_set_predictions(comparisons, predictions, "original")

        assert out[0]["owner_prediction_held"] is True


class TestPathSplitGroupsByQueryThenSums:
    def test_top_k_is_taken_per_query_not_globally(self):
        """5 rows for q1 (all hits) and 5 rows for q2 (only rank 5 is a hit)
        interleaved in rank order. A global top-5 across both queries' rows
        would only ever see one query's rank-5 row, undercounting hits.
        Grouping by query_id first and summing each query's own top-5 hits
        must find all 6 hits (5 from q1, 1 from q2)."""
        sge = _score_graph_expand()
        rows = []
        for rank in range(1, 6):
            rows.append(_row("graph_b1r_rr", "q1", f"q1-d{rank}", rank,
                              via={"path": "entity", "connector": "x"}))
        for rank in range(1, 6):
            rows.append(_row("graph_b1r_rr", "q2", f"q2-d{rank}", rank,
                              via={"path": "contains", "connector": "y"}))
        qrels = _qrels(
            [("q1", f"q1-d{r}", 2) for r in range(1, 6)]
            + [("q2", "q2-d1", 0), ("q2", "q2-d2", 0), ("q2", "q2-d3", 0),
               ("q2", "q2-d4", 0), ("q2", "q2-d5", 2)]
        )

        out = sge.path_split(rows, qrels, k=5)

        assert out["hits"] == 6
        assert out["shares"]["entity"] == pytest.approx(5 / 6)
        assert out["shares"]["contains"] == pytest.approx(1 / 6)

    def test_each_query_is_still_truncated_to_k(self):
        """A query with more than k candidate rows only contributes its own
        top-k, not every row it has."""
        sge = _score_graph_expand()
        rows = [_row("x", "q1", f"d{r}", r, via={"path": "entity", "connector": "z"})
                for r in range(1, 8)]  # 7 rows, k=5
        qrels = _qrels([("q1", f"d{r}", 2) for r in range(1, 8)])

        out = sge.path_split(rows, qrels, k=5)

        assert out["hits"] == 5

    def test_single_query_behaves_as_before(self):
        sge = _score_graph_expand()
        rows = [
            _row("graph_b1r_rr", "q1", "d1", 1, via={"path": "entity", "connector": "x"}),
            _row("graph_b1r_rr", "q1", "d2", 2, via={"path": "entity", "connector": "x"}),
            _row("graph_b1r_rr", "q1", "d3", 3, via={"path": "contains", "connector": None}),
        ]
        qrels = _qrels([("q1", "d1", 2), ("q1", "d2", 3), ("q1", "d3", 2)])

        out = sge.path_split(rows, qrels, k=5)

        assert out["hits"] == 3
        assert out["shares"]["entity"] == pytest.approx(2 / 3)


class TestRenderAppendixIsMarkdownTables:
    def test_no_raw_json_blob_in_the_appendix(self):
        sge = _score_graph_expand()
        result = {"sets": {}}
        text = sge.render_appendix(result).decode("utf-8")

        assert '"sets"' not in text
        assert "json.dumps" not in text

    def test_metrics_rendered_as_a_table_with_8_system_columns(self):
        sge = _score_graph_expand()
        metrics = {
            rid: {"ndcg_at_5": 0.5, "recall_pool": 0.5, "judged_at_5": 1.0, "n": 8}
            for rid in sge.SYSTEMS
        }
        result = {
            "sets": {
                "graph": {
                    "metrics": metrics,
                    "comparisons": [],
                    "decisions": {},
                    "path_split": {},
                }
            }
        }

        text = sge.render_appendix(result).decode("utf-8")

        assert "| metric |" in text
        for rid in sge.SYSTEMS:
            assert sge.SHORT_NAMES[rid] in text
        for metric_name in ("ndcg_at_5", "recall_pool", "judged_at_5"):
            assert metric_name in text

    def test_verdicts_predictions_and_path_split_sections_present(self):
        sge = _score_graph_expand()
        result = {
            "sets": {
                "original": {
                    "metrics": {rid: {"ndcg_at_5": 0.1, "recall_pool": 0.1,
                                       "judged_at_5": 1.0, "n": 22}
                                for rid in sge.SYSTEMS},
                    "comparisons": [
                        {"base": "ladder_s2_cos_bm25_rr", "new": "ctrl_c0_deeper_rr",
                         "n": 22, "mean_delta": 0.01, "ci_low": -0.01, "ci_high": 0.03,
                         "verdict": "no detectable difference at n=22",
                         "owner_prediction": "improved", "owner_prediction_held": False,
                         "planner_prediction": "no detectable difference",
                         "planner_prediction_held": True},
                    ],
                    "decisions": {
                        "r": {"branch": "GRAPH OUT", "step1_deeper_pool": False,
                              "step2_graph_over_c0": False, "step3_graph_over_shortcuts": False},
                    },
                    "path_split": {"r": {"hits": 2, "shares": {"entity": 1.0}}},
                }
            }
        }

        text = sge.render_appendix(result).decode("utf-8")

        assert "GRAPH OUT" in text
        assert "owner_prediction_held" in text or "owner_held" in text
        assert "planner_prediction_held" in text or "planner_held" in text
        assert "path split" in text.lower()
        assert "entity" in text
