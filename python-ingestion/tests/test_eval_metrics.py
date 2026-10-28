"""Hand-computed fixtures for the evaluation metrics (spec acceptance test 5)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import metrics as M
from evals.bm25 import BM25


def rows(*pairs):
    """(grade, score[, truncated]) tuples -> ranked row dicts."""
    return [{"grade": p[0], "score": p[1], "truncated": p[2] if len(p) > 2 else False}
            for p in pairs]


class TestNdcg:
    def test_graded_ranking_3_0_2_0_1(self):
        # DCG  = 7/log2(2) + 0 + 3/log2(4) + 0 + 1/log2(6) = 8.88685
        # IDCG = 7/log2(2) + 3/log2(3) + 1/log2(4)         = 9.39279
        assert M.ndcg_at_k([3, 0, 2, 0, 1], [3, 2, 1], 5) == pytest.approx(0.94614, abs=1e-4)

    def test_ideal_ordering_is_one(self):
        assert M.ndcg_at_k([3, 2, 1, 0, 0], [3, 2, 1], 5) == pytest.approx(1.0)

    def test_nothing_judged_relevant_is_undefined(self):
        assert M.ndcg_at_k([0, 0, 0, 0, 0], [0, 0], 5) is None

    def test_only_top_k_counts(self):
        assert M.ndcg_at_k([0, 0, 0, 0, 0, 3], [3], 5) == 0.0

    def test_grade_one_still_contributes_gain(self):
        assert 0 < M.ndcg_at_k([1, 0, 0, 0, 0], [1], 5) <= 1.0


class TestMrr:
    def test_zero_when_first_relevant_is_rank_six(self):
        assert M.mrr_at_k([0, 0, 0, 0, 0, 3], 5) == 0.0

    def test_grade_one_is_not_relevant(self):
        assert M.mrr_at_k([0, 1, 0, 2, 0], 5) == pytest.approx(0.25)

    def test_rank_one_is_one(self):
        assert M.mrr_at_k([3, 0, 0, 0, 0], 5) == 1.0


class TestRecall:
    def test_two_of_three_relevant_retrieved(self):
        assert M.recall_at_k([3, 0, 2, 0, 0, 2], 3, 5) == pytest.approx(2 / 3, abs=1e-3)

    def test_undefined_without_relevant_docs(self):
        assert M.recall_at_k([0, 0], 0, 5) is None


class TestSuccess:
    def test_grade_one_does_not_count(self):
        assert M.success_at_k([0, 1, 1, 1, 1, 3], 5) is False

    def test_grade_two_in_top_five_counts(self):
        assert M.success_at_k([0, 0, 0, 0, 2], 5) is True

    def test_candidate_depth_is_wider_than_top_five(self):
        ranked = [0] * 10 + [3]
        assert M.success_at_k(ranked, 5) is False
        assert M.success_at_k(ranked, 30) is True


class TestAuc:
    def test_perfect_separation(self):
        assert M.roc_auc([0.9, 0.8, 0.1], [True, False, False]) == 1.0

    def test_ties_get_half_credit(self):
        assert M.roc_auc([0.5, 0.5], [True, False]) == 0.5

    def test_inverted_scores(self):
        assert M.roc_auc([0.1, 0.9], [True, False]) == 0.0

    def test_undefined_with_one_class(self):
        assert M.roc_auc([0.1, 0.2], [False, False]) is None


class TestClassifyFailure:
    def test_no_relevant_document(self):
        assert M.classify_failure(rows((0, 0.9), (1, 0.8))) == "no_corpus_answer"

    def test_success(self):
        assert M.classify_failure(rows((3, 0.6), (0, 0.5))) == "success"

    def test_threshold_miss_when_cutoff_removes_the_hit(self):
        assert M.classify_failure(rows((0, 0.9), (2, 0.2))) == "threshold_miss"

    def test_ranking_miss(self):
        ranked = rows(*([(0, 0.5)] * 6 + [(3, 0.4)]))
        assert M.classify_failure(ranked) == "ranking_miss"

    def test_cosine_retrieval_miss_beyond_candidate_depth(self):
        ranked = rows(*([(0, 0.5)] * 30 + [(3, 0.4)]))
        assert M.classify_failure(ranked) == "cosine_retrieval_miss"

    def test_long_document_miss_takes_precedence_for_truncated_relevant(self):
        ranked = rows(*([(0, 0.5)] * 30 + [(3, 0.3, True)]))
        assert M.classify_failure(ranked) == "long_document_miss"


class TestThresholdSweep:
    def test_precision_recall_and_success(self):
        q = rows((3, 0.7), (0, 0.3), (2, 0.2))
        low, high = M.threshold_sweep([q], [0.25, 0.5])
        assert (low["returned"], low["precision"], low["recall"]) == (2, 0.5, 0.5)
        assert low["success_at_k"] == 1.0
        assert (high["returned"], high["precision"], high["recall"]) == (1, 1.0, 0.5)

    def test_no_results_above_threshold_has_undefined_precision(self):
        (r,) = M.threshold_sweep([rows((3, 0.4))], [0.9])
        assert r["precision"] is None and r["success_at_k"] == 0.0


class TestBm25:
    DOCS = ["qdrant is an open source vector database",
            "my favourite pasta recipes for dinner",
            "hybrid vector search with qdrant and reranking"]

    def test_ranks_matching_documents_and_excludes_zero_scores(self):
        top = BM25(self.DOCS).top("qdrant vector search", 5)
        assert [i for i, _ in top][0] == 2
        assert 1 not in [i for i, _ in top]

    def test_deterministic_tie_break(self):
        bm = BM25(["alpha beta", "alpha beta"])
        assert [i for i, _ in bm.top("alpha", 2)] == [0, 1]

    def test_long_document_is_findable_by_a_rare_term(self):
        long_doc = "filler " * 400 + "llama cpp moe"
        docs = [long_doc, "short intro hello", "another short note"]
        assert BM25(docs).top("moe", 1)[0][0] == 0
