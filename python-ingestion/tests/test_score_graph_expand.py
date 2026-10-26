"""U8 score_graph_expand: score sets, pick branches, write Appendix D (section 13.8).

The two query sets are always reported separately and the branch rules are
pre-registered (D16). Decision functions take no latency input (D17).
"""

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _score_graph_expand():
    from evals import score_graph_expand
    return score_graph_expand


def _row(run_id, qid, cid, rank, score=1.0, via=None, note=None):
    return {"run_id": run_id, "query_id": qid, "candidate_id": cid,
            "rank": rank, "score": score, "via": via, "note": note}


ALL_SYSTEMS = (
    "ladder_s2_cos_bm25_rr",
    "ctrl_c0_deeper_rr",
    "ctrl_c1_denorm_rr",
    "ctrl_c2_payload_rr",
    "graph_b1r_rr",
    "graph_b2r_ctx_rr",
    "graph_b1q_rr",
    "graph_b2q_ctx_rr",
)

EXPECTED_COMPARISONS = (
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


def _qrels(pairs):
    out = {}
    for qid, cid, grade in pairs:
        out.setdefault(qid, {})[cid] = {"relevance": grade, "answer_bearing": 0}
    return out


def _all_runs(qid="q1"):
    """One graded row per system: enough to exercise the comparison table."""
    return {run_id: [_row(run_id, qid, "d1", 1)] for run_id in ALL_SYSTEMS}


class TestScoreGraphExpandUnit:
    def test_U8_B1_fully_judged_before_any_metric(self, monkeypatch):
        """An unjudged top-5 pair fails before nDCG or a bootstrap is computed."""
        sge = _score_graph_expand()
        from evals import topup

        calls = []

        def must_not_run(*args, **kwargs):
            calls.append("metric")
            raise AssertionError("a metric ran before the judged check")

        monkeypatch.setattr(sge.ladder_stats, "per_query", must_not_run)
        monkeypatch.setattr(sge.ladder_stats, "paired_bootstrap", must_not_run)

        runs = {"ladder_s2_cos_bm25_rr": [
            _row("ladder_s2_cos_bm25_rr", "q1", "d1", 1),
            _row("ladder_s2_cos_bm25_rr", "q1", "ungraded", 2),
        ]}
        qrels = _qrels([("q1", "d1", 2)])

        with pytest.raises(topup.UnjudgedError, match="ungraded"):
            sge.score_sets(
                {"original": runs, "graph": {"ladder_s2_cos_bm25_rr": []}},
                {"original": qrels, "graph": {}},
                {"original": ["q1"], "graph": []},
                systems=("ladder_s2_cos_bm25_rr",),
            )
        assert calls == []

    def test_U8_B2_original_and_graph_sets_are_never_pooled(self):
        """Changing graph qrels must not move a single original-set number."""
        sge = _score_graph_expand()
        runs = {
            "original": {"ladder_s2_cos_bm25_rr": [
                _row("ladder_s2_cos_bm25_rr", "q1", "d1", 1)]},
            "graph": {"ladder_s2_cos_bm25_rr": [
                _row("ladder_s2_cos_bm25_rr", "g1", "d1", 1)]},
        }
        queries = {"original": ["q1"], "graph": ["g1"]}

        first = sge.score_sets(
            runs,
            {"original": _qrels([("q1", "d1", 3)]),
             "graph": _qrels([("g1", "d1", 3)])},
            queries,
            systems=("ladder_s2_cos_bm25_rr",),
        )
        second = sge.score_sets(
            runs,
            {"original": _qrels([("q1", "d1", 3)]),
             "graph": _qrels([("g1", "d1", 0)])},
            queries,
            systems=("ladder_s2_cos_bm25_rr",),
        )

        assert set(first) == {"original", "graph"}
        assert first["original"] == second["original"]
        assert first["graph"] != second["graph"]

    def test_U8_B3_exactly_the_eleven_comparisons_in_order_per_set(self):
        sge = _score_graph_expand()
        assert tuple(sge.COMPARISONS) == EXPECTED_COMPARISONS
        assert len(sge.COMPARISONS) == 11

        runs = {"original": _all_runs("q1"), "graph": _all_runs("g1")}
        qrels = {"original": _qrels([("q1", "d1", 2)]),
                 "graph": _qrels([("g1", "d1", 2)])}
        queries = {"original": ["q1"], "graph": ["g1"]}

        result = sge.score_sets(runs, qrels, queries)

        for set_name in ("original", "graph"):
            got = [(c["base"], c["new"]) for c in result[set_name]["comparisons"]]
            assert tuple(got) == EXPECTED_COMPARISONS, set_name

    def test_U8_B4_original_beats_needs_improved_and_mean_delta_0020(self):
        sge = _score_graph_expand()
        base = {"q1": 0.5, "q2": 0.5}
        at_threshold = {"q1": 0.52, "q2": 0.52}          # mean Δ = 0.0200...
        below = {"q1": 0.5199, "q2": 0.5199}             # mean Δ = 0.0199...

        assert sge.beats_original(base, at_threshold, "improved") is True
        assert sge.beats_original(base, below, "improved") is False
        assert sge.beats_original(base, at_threshold, "no detectable difference") is False
        assert sge.beats_original(base, at_threshold, "regressed") is False

    @pytest.mark.parametrize("n,wins,losses,expected", [
        (8, 7, 0, True), (8, 6, 0, False), (8, 7, 1, False),
        (7, 7, 0, True), (7, 6, 0, False),
        (6, 6, 0, True), (6, 5, 0, False),
        (5, 5, 0, True), (5, 4, 0, False),
        (4, 4, 0, None), (3, 3, 0, None), (0, 0, 0, None),   # n <= 4: too few to decide
    ])
    def test_U8_B5_scoreboard_d16(self, n, wins, losses, expected):
        sge = _score_graph_expand()
        assert sge.graph_beats(n, wins, losses) is expected

    def test_U8_B6_decision_branches_and_independent_step_1(self):
        sge = _score_graph_expand()

        out = sge.decide_branch(step1_deeper_pool=True,
                                step2_graph_over_c0=False,
                                step3_graph_over_shortcuts=True)
        assert out["branch"] == "GRAPH OUT"
        assert out["step1_deeper_pool"] is True
        assert set(out) >= {"step1_deeper_pool", "step2_graph_over_c0",
                            "step3_graph_over_shortcuts", "branch"}

        out = sge.decide_branch(step1_deeper_pool=False,
                                step2_graph_over_c0=True,
                                step3_graph_over_shortcuts=False)
        assert out["branch"] == "SHIP THE SHORTCUT(S)"
        # Step 1 is recorded as measured, whatever the later steps say.
        assert out["step1_deeper_pool"] is False

        out = sge.decide_branch(step1_deeper_pool=True,
                                step2_graph_over_c0=True,
                                step3_graph_over_shortcuts=True)
        assert out["branch"] == "GRAPH STAYS"

    def test_U8_B7_path_split_shares_sum_to_one(self):
        sge = _score_graph_expand()
        rows = [
            _row("graph_b1r_rr", "q1", "d1", 1, via={"path": "entity", "connector": "x"}),
            _row("graph_b1r_rr", "q1", "d2", 2, via={"path": "entity", "connector": "x"}),
            _row("graph_b1r_rr", "q1", "d3", 3, via={"path": "contains", "connector": None}),
            _row("graph_b1r_rr", "q1", "d4", 4, via=None),
            _row("graph_b1r_rr", "q1", "d5", 5, via={"path": "topic", "connector": "y"}),
        ]
        qrels = _qrels([("q1", "d1", 2), ("q1", "d2", 3), ("q1", "d3", 2),
                        ("q1", "d4", 3), ("q1", "d5", 0)])

        out = sge.path_split(rows, qrels, k=5)

        assert out["hits"] == 3
        assert out["shares"]["entity"] == pytest.approx(2 / 3)
        assert out["shares"]["contains"] == pytest.approx(1 / 3)
        assert sum(out["shares"].values()) == pytest.approx(1.0)

        empty = sge.path_split([], qrels, k=5)
        assert empty == {"hits": 0, "shares": {}}

    def test_U8_B8_recall_pool_and_judged_at_5(self):
        sge = _score_graph_expand()
        rows = [
            _row("x", "q1", "d1", 1),
            _row("x", "q1", "unjudged", 2),
            _row("x", "q1", "d3", 3),
            _row("x", "q1", "d4", 4),
            _row("x", "q1", "unjudged2", 5),
        ]
        qrels = _qrels([("q1", "d1", 2), ("q1", "d3", 3),
                        ("q1", "d4", 2), ("q1", "d5", 2)])

        assert sge.recall_pool(rows, qrels) == pytest.approx(3 / 4)
        assert sge.judged_at_5(rows, qrels) == pytest.approx(3 / 5)

    def test_U8_B9_appendix_preserves_report_bytes_and_refuses_a_second_copy(self):
        sge = _score_graph_expand()
        report = b"# Manthan report\n\nbody\n"
        appendix = sge.render_appendix({"sets": {}})

        out = sge.append_appendix(report, appendix)
        assert out[:len(report)] == report
        assert out[len(report):].lstrip().startswith(
            "## Appendix D — Graph as candidate generator".encode("utf-8"))

        with pytest.raises(ValueError, match="Appendix D"):
            sge.append_appendix(out, appendix)

    def test_U8_B10_appendix_has_the_d18_disclosure_and_d24_grader_note(self):
        sge = _score_graph_expand()
        text = sge.render_appendix({"sets": {}}).decode("utf-8")

        assert text.lstrip().startswith(
            "## Appendix D — Graph as candidate generator")
        # D18: who wrote the graph questions, from what, and what they had seen.
        assert "drafted the 8 questions" in text
        assert "documents.jsonl text only" in text
        assert "owner adjusted" in text
        assert "connector degrees" in text
        # D24: what the graph-set grader sees.
        assert "grader view" in text
        assert "shared link" in text
        assert "candidate_id order" in text

    def test_U8_A2_prediction_and_prediction_held_per_comparison(self):
        sge = _score_graph_expand()
        comparisons = [
            {"base": "a", "new": "b", "verdict": "improved"},
            {"base": "c", "new": "d", "verdict": "no detectable difference"},
            {"base": "e", "new": "f", "verdict": "regressed"},
        ]
        predictions = {
            "a->b": {"expected_verdict": "improved"},
            "c->d": {"expected_verdict": "improved"},
        }

        out = sge.attach_predictions(comparisons, predictions)

        assert [(c["prediction"], c["prediction_held"]) for c in out] == [
            ("improved", True),
            ("improved", False),
            (None, None),
        ]

    def test_U8_B11_decide_for_set_both_sets_both_starts(self):
        """Review #4 K6: decide_for_set wires step 1/2/3 per start (r, q).
        Original set: beats_original-style (bootstrap verdict + mean delta).
        Graph set: graph_beats (D16 scoreboard) via ladder_stats.win_tie_loss
        over the raw per-query ndcg in per_system."""
        sge = _score_graph_expand()

        def cmp(base, new, verdict, mean_delta):
            return {"base": base, "new": new, "verdict": verdict, "mean_delta": mean_delta,
                    "ci_low": 0.0, "ci_high": 0.0, "n": 8}

        original_result = {
            "per_system": {},
            "comparisons": [
                cmp("ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr", "improved", 0.03),
                cmp("ctrl_c0_deeper_rr", "graph_b2r_ctx_rr", "improved", 0.03),
                cmp("ctrl_c1_denorm_rr", "graph_b2r_ctx_rr", "no detectable difference", 0.00),
                cmp("ctrl_c2_payload_rr", "graph_b2r_ctx_rr", "improved", 0.03),
                cmp("ctrl_c0_deeper_rr", "graph_b2q_ctx_rr", "improved", 0.03),
                cmp("ctrl_c1_denorm_rr", "graph_b2q_ctx_rr", "improved", 0.03),
                cmp("ctrl_c2_payload_rr", "graph_b2q_ctx_rr", "improved", 0.03),
            ],
        }

        out = sge.decide_for_set("original", original_result)
        assert out["r"]["step1_deeper_pool"] is True
        assert out["r"]["branch"] == "SHIP THE SHORTCUT(S)"   # C1 doesn't beat B2-r
        assert out["q"]["step1_deeper_pool"] is True
        assert out["q"]["branch"] == "GRAPH STAYS"            # C1 and C2 both beat B2-q

        def per_query(values):
            return {qid: {"ndcg": v, "success": 0, "mrr": 0, "candidate_success": 0}
                    for qid, v in values.items()}

        qids = [f"g{i:03d}" for i in range(1, 9)]
        base_low = per_query({q: 0.5 for q in qids})

        def winning(n_wins):
            return per_query({q: (0.52 if i < n_wins else 0.5) for i, q in enumerate(qids)})

        graph_result = {
            "per_system": {
                "ladder_s2_cos_bm25_rr": base_low,
                "ctrl_c0_deeper_rr": base_low,
                "ctrl_c1_denorm_rr": base_low,
                "ctrl_c2_payload_rr": base_low,
                "graph_b2r_ctx_rr": winning(7),   # 7 wins, 0 losses -> graph_beats(8,7,0) True
                "graph_b2q_ctx_rr": winning(6),   # 6 wins -> graph_beats(8,6,0) False
            },
            "comparisons": [
                cmp("ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr", "improved", 0.0),
            ],
        }

        out2 = sge.decide_for_set("graph", graph_result)
        assert out2["r"]["step2_graph_over_c0"] is True
        assert out2["r"]["step3_graph_over_shortcuts"] is True
        assert out2["r"]["branch"] == "GRAPH STAYS"
        assert out2["q"]["step2_graph_over_c0"] is False
        assert out2["q"]["branch"] == "GRAPH OUT"


class TestScoreGraphExpandAntiCheat:
    def test_U8_A1_decision_functions_take_no_latency_input(self):
        sge = _score_graph_expand()
        for fn_name in ("decide_branch", "beats_original", "graph_beats"):
            params = inspect.signature(getattr(sge, fn_name)).parameters
            for name in params:
                lowered = name.lower()
                assert "latency" not in lowered, (fn_name, name)
                assert "p50" not in lowered and "p95" not in lowered, (fn_name, name)
                assert not lowered.endswith("_ms"), (fn_name, name)
