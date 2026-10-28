"""U7: per-query metrics, win/tie/loss, paired bootstrap, verdicts (spec section 11).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import ladder_stats as S  # noqa: E402
from evals import metrics as M  # noqa: E402


def q(grades):
    return {f"d{i}": {"relevance": g, "answer_bearing": 0} for i, g in enumerate(grades)}


def run(qid, ids, scores=None):
    scores = scores or [1.0 / (i + 1) for i in range(len(ids))]
    return [{"query_id": qid, "candidate_id": c, "rank": i + 1, "score": s}
            for i, (c, s) in enumerate(zip(ids, scores))]


def test_constants_are_pre_registered():
    assert S.TIE_EPS == 0.01 and S.N_RESAMPLES == 10000 and S.SEED == 42


class TestAnswerable:
    def test_answerable_means_any_grade_two_or_more(self):
        qrels = {"q1": q([0, 3]), "q2": q([1, 1]), "q3": q([2]), "q4": {}}
        assert S.answerable(qrels, ["q4", "q3", "q2", "q1", "q5"]) == ["q1", "q3"]


class TestPerQuery:
    QRELS = {"q1": q([3, 0, 2])}

    def test_uses_the_existing_metric_definitions(self):
        rows = run("q1", ["d1", "d0", "d2"])
        got = S.per_query(rows, self.QRELS, ["q1"])["q1"]
        grades = [0, 3, 2]
        assert got["ndcg"] == pytest.approx(M.ndcg_at_k(grades, [3, 0, 2], 5))
        assert got["success"] is True
        assert got["mrr"] == pytest.approx(0.5)
        assert got["candidate_success"] is True

    def test_unjudged_documents_count_as_grade_zero(self):
        rows = run("q1", ["zz", "d0"])
        got = S.per_query(rows, self.QRELS, ["q1"])["q1"]
        assert got["mrr"] == pytest.approx(0.5)

    def test_candidate_success_looks_to_depth_thirty(self):
        ids = [f"x{i}" for i in range(29)] + ["d0"]
        got = S.per_query(run("q1", ids), self.QRELS, ["q1"])["q1"]
        assert got["success"] is False and got["candidate_success"] is True
        ids = [f"x{i}" for i in range(30)] + ["d0"]
        assert S.per_query(run("q1", ids), self.QRELS, ["q1"])["q1"]["candidate_success"] is False

    def test_rows_are_ordered_by_rank_not_file_order(self):
        rows = list(reversed(run("q1", ["d0", "d1"])))
        assert S.per_query(rows, self.QRELS, ["q1"])["q1"]["mrr"] == 1.0

    def test_query_with_no_rows_scores_zero(self):
        got = S.per_query([], self.QRELS, ["q1"])["q1"]
        assert got["ndcg"] == 0.0 and got["success"] is False


class TestWinTieLoss:
    def test_counts_with_tie_epsilon(self):
        base = {"q1": 0.50, "q2": 0.50, "q3": 0.50, "q4": 0.50}
        new = {"q1": 0.70, "q2": 0.505, "q3": 0.30, "q4": 0.51}
        out = S.win_tie_loss(base, new)
        assert (out["wins"], out["ties"], out["losses"]) == (2, 1, 1)
        assert out["improved"] == ["q1", "q4"] and out["regressed"] == ["q3"]

    def test_mismatched_queries_fail(self):
        with pytest.raises(ValueError):
            S.win_tie_loss({"q1": 1.0}, {"q2": 1.0})


class TestBootstrap:
    def test_mean_delta_is_exact(self):
        out = S.paired_bootstrap({"a": 0.0, "b": 0.5}, {"a": 0.2, "b": 0.9})
        assert out["mean_delta"] == pytest.approx(0.3) and out["n"] == 2

    def test_identical_runs_give_a_zero_interval(self):
        base = {f"q{i}": i / 10 for i in range(10)}
        out = S.paired_bootstrap(base, dict(base))
        assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0

    def test_all_positive_deltas_give_an_interval_above_zero(self):
        base = {f"q{i}": 0.2 for i in range(22)}
        new = {f"q{i}": 0.2 + 0.1 + i / 100 for i in range(22)}
        out = S.paired_bootstrap(base, new)
        assert out["ci_low"] > 0

    def test_deterministic_for_a_seed(self):
        rng = random.Random(1)
        base = {f"q{i}": rng.random() for i in range(22)}
        new = {k: v + rng.uniform(-0.2, 0.2) for k, v in base.items()}
        assert S.paired_bootstrap(base, new) == S.paired_bootstrap(base, new)

    def test_documented_percentile_algorithm(self):
        base = {"q1": 0.0, "q2": 0.0, "q3": 0.0}
        new = {"q1": 0.3, "q2": -0.6, "q3": 0.9}
        deltas = [0.3, -0.6, 0.9]  # sorted query ids
        rng = random.Random(7)
        means = sorted(sum(deltas[rng.randrange(3)] for _ in range(3)) / 3 for _ in range(200))
        out = S.paired_bootstrap(base, new, n_resamples=200, seed=7)
        assert out["ci_low"] == pytest.approx(means[int(0.025 * 200)])
        assert out["ci_high"] == pytest.approx(means[int(0.975 * 200) - 1])


class TestVerdict:
    def test_three_verdicts(self):
        assert S.verdict(0.01, 0.2, 22) == "improved"
        assert S.verdict(-0.2, -0.01, 22) == "regressed"
        assert S.verdict(-0.05, 0.2, 22) == "no detectable difference at n=22"

    def test_boundary_zero_is_not_a_detection(self):
        assert S.verdict(0.0, 0.3, 22) == "no detectable difference at n=22"
        assert S.verdict(-0.3, 0.0, 22) == "no detectable difference at n=22"


class TestUnjudgedAndAbstention:
    def test_unjudged_at_depth(self):
        qrels = {"q1": q([3])}
        rows = run("q1", ["d0", "u1", "u2", "u3"])
        assert S.unjudged_at(rows, qrels, ["q1"], depth=3) == 2

    def test_abstention_reports_top1_scores(self):
        rows = run("na", ["x"], [0.31]) + run("ok", ["y"], [0.72])
        out = S.abstention(rows, no_answer_ids=["na"], answerable_ids=["ok"])
        assert out["no_answer_top1"] == {"na": 0.31}
        assert out["answerable_top1"] == {"ok": 0.72}
        assert out["median_no_answer"] == pytest.approx(0.31)
        assert out["median_answerable"] == pytest.approx(0.72)
