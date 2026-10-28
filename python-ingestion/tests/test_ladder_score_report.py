"""U10: score the ladder and append Appendix C (spec section 14).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common, run_ladder, score_ladder, topup  # noqa: E402
from evals import ladder_stats as S  # noqa: E402

S0, S1, S2, S3, S4 = run_ladder.STEPS
QUERIES = [{"query_id": "q1", "query_text": "a", "intent": "concept"},
           {"query_id": "q2", "query_text": "b", "intent": "person_lookup"},
           {"query_id": "q3", "query_text": "c", "intent": "no_answer"}]
QRELS = {
    "q1": {"m:good": {"relevance": 3, "answer_bearing": 0}, "m:bad": {"relevance": 0, "answer_bearing": 0}},
    "q2": {"m:who": {"relevance": 2, "answer_bearing": 0}, "m:bad": {"relevance": 0, "answer_bearing": 0}},
    "q3": {"m:bad": {"relevance": 1, "answer_bearing": 0}, "m:good": {"relevance": 0, "answer_bearing": 0}},
}


def rows(step, qid, ids, scores=None):
    scores = scores or [1.0 / (i + 1) for i in range(len(ids))]
    return [{"run_id": step, "query_id": qid, "candidate_id": c, "rank": i + 1, "score": s}
            for i, (c, s) in enumerate(zip(ids, scores))]


def make_runs(bad_first=(S0, S1, S3)):
    runs = {}
    for step in run_ladder.STEPS:
        q1 = ["m:bad", "m:good"] if step in bad_first else ["m:good", "m:bad"]
        runs[step] = (rows(step, "q1", q1) + rows(step, "q2", ["m:who", "m:bad"])
                      + rows(step, "q3", ["m:bad", "m:good"], [0.3, 0.2]))
    return runs


PREDICTIONS = {"registered_at": "2026-10-18", "comparisons": {
    f"{S1}->{S2}": {"expected_verdict": "improved", "reason": "r"},
    f"{S0}->{S1}": {"expected_verdict": "no detectable difference", "reason": "r"},
}}


@pytest.fixture
def result():
    return score_ladder.score(make_runs(), QRELS, QUERIES, PREDICTIONS)


def test_comparisons_are_pre_registered_in_order():
    assert score_ladder.COMPARISONS == ((S0, S1), (S1, S2), (S1, S3), (S3, S4), (S2, S4), (S0, S4))


def test_scoring_refuses_unjudged_top5_pairs():
    runs = make_runs()
    runs[S2] = rows(S2, "q1", ["m:never_graded", "m:good"]) + runs[S2][2:]
    with pytest.raises(topup.UnjudgedError):
        score_ladder.score(runs, QRELS, QUERIES, PREDICTIONS)


def test_averages_are_over_answerable_queries_only(result):
    assert result["answerable"] == ["q1", "q2"] and result["no_answer"] == ["q3"]
    assert result["n_answerable"] == 2
    per_q = S.per_query(make_runs()[S2], QRELS, ["q1", "q2"])
    expected = (per_q["q1"]["ndcg"] + per_q["q2"]["ndcg"]) / 2
    assert result["steps"][S2]["mean"]["ndcg"] == pytest.approx(expected)


def test_every_step_reports_the_four_metrics(result):
    for step in run_ladder.STEPS:
        assert set(result["steps"][step]["mean"]) == {"ndcg", "success", "mrr", "candidate_success"}


def test_comparison_verdict_comes_from_the_stats_function(result):
    comp = next(c for c in result["comparisons"] if (c["base"], c["new"]) == (S1, S2))
    b = comp["bootstrap"]
    assert comp["verdict"] == S.verdict(b["ci_low"], b["ci_high"], 2)
    assert comp["win_tie_loss"]["wins"] == 1  # q1 improved, q2 tied


def test_predictions_are_checked(result):
    by_pair = {(c["base"], c["new"]): c for c in result["comparisons"]}
    assert by_pair[(S1, S2)]["prediction"]["expected_verdict"] == "improved"
    held = by_pair[(S1, S2)]["verdict"].startswith("improved")
    assert by_pair[(S1, S2)]["prediction_held"] is held
    assert by_pair[(S1, S3)]["prediction"] is None and by_pair[(S1, S3)]["prediction_held"] is None


def test_abstention_only_for_steps_with_a_meaningful_score_scale(result):
    assert set(result["abstention"]) == {S0, S2, S4}
    assert result["abstention"][S0]["no_answer_top1"] == {"q3": 0.3}


def test_candidate_success_flagged_as_lower_bound_when_depth30_has_unjudged(result):
    runs = make_runs()
    runs[S1] = runs[S1] + [{"run_id": S1, "query_id": "q1", "candidate_id": "m:unjudged",
                            "rank": 6, "score": 0.01}]  # below top-5, inside depth 30
    out = score_ladder.score(runs, QRELS, QUERIES, PREDICTIONS)
    assert out["steps"][S1]["unjudged_at_30"] == 1
    assert out["steps"][S1]["candidate_success_is_lower_bound"] is True
    assert result["steps"][S1]["candidate_success_is_lower_bound"] is False


def test_result_is_json_serialisable(result):
    json.dumps(result)


class TestAppendix:
    def appendix(self, result):
        latency = {"machine": {"processor": "x86", "cpu_count": 16, "gpu": None},
                   "components": {"rerank_cpu": {"summary": {"n": 3, "p50": 40.0, "p95": 55.0}}}}
        return score_ladder.render_appendix(result, latency)

    def test_heading(self, result):
        assert self.appendix(result).startswith("## Appendix C — Retrieval ladder")

    def test_every_verdict_is_written_verbatim(self, result):
        text = self.appendix(result)
        for comp in result["comparisons"]:
            assert comp["verdict"] in text

    def test_states_n_and_lower_bound_and_machine_labels(self, result):
        text = self.appendix(result)
        assert "n=2" in text
        assert "lower bound" in text
        assert "estimate across machines" in text

    def test_append_preserves_existing_bytes(self, result, tmp_path):
        report = tmp_path / "manthan_report.md"
        original = "# Report\n\nOriginal text — kept byte-identical.\n".encode("utf-8")
        report.write_bytes(original)
        score_ladder.append_appendix(report, self.appendix(result))
        new = report.read_bytes()
        assert new.startswith(original) and b"## Appendix C" in new[len(original):]

    def test_second_append_is_refused(self, result, tmp_path):
        report = tmp_path / "manthan_report.md"
        report.write_text("# Report\n", encoding="utf-8")
        score_ladder.append_appendix(report, self.appendix(result))
        with pytest.raises(ValueError):
            score_ladder.append_appendix(report, self.appendix(result))


@pytest.mark.artifacts
def test_artifact_real_scoring_uses_qrels_v2_and_22_answerable_queries():
    data = common.data_dir(common.DEFAULT_CORPUS_VERSION, create=False)
    out = json.loads((data / "ladder_metrics.json").read_text(encoding="utf-8"))
    assert out["qrels_version"] == "qrels_v2.csv"
    assert out["n_answerable"] == len(out["answerable"])
    assert len(out["comparisons"]) == len(score_ladder.COMPARISONS)
