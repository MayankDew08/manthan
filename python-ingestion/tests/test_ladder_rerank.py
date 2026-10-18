"""U5: deterministic cross-encoder re-ranker (spec section 9).

Offline tests inject a fake scorer; tests marked ``model`` load the pinned weights
from the local Hugging Face cache (no network).
LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import rerank  # noqa: E402


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, pairs):
        self.calls.append(list(pairs))
        return [self.scores[text] for _, text in pairs]


def test_model_is_pinned():
    assert rerank.MODEL_NAME == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert rerank.MODEL_REVISION == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    assert rerank.MAX_LENGTH == 512
    assert rerank.SCORE_DECIMALS == 6


def test_orders_by_score_desc():
    r = rerank.Reranker(score_pairs=FakeScorer({"t1": 0.1, "t2": 3.0, "t3": -2.0}))
    out = r.rerank("q", [("a", "t1"), ("b", "t2"), ("c", "t3")])
    assert [c for c, _ in out] == ["b", "a", "c"]


def test_scores_are_rounded_to_six_decimals():
    r = rerank.Reranker(score_pairs=FakeScorer({"t1": 0.123456789}))
    assert r.rerank("q", [("a", "t1")]) == [("a", 0.123457)]


def test_ties_after_rounding_break_by_candidate_id():
    r = rerank.Reranker(score_pairs=FakeScorer({"t1": 1.0000001, "t2": 1.0000002}))
    out = r.rerank("q", [("z", "t1"), ("m", "t2")])
    assert [c for c, _ in out] == ["m", "z"]


def test_scorer_is_called_once_with_query_text_pairs_in_input_order():
    scorer = FakeScorer({"t1": 0.0, "t2": 0.0})
    rerank.Reranker(score_pairs=scorer).rerank("the query", [("a", "t1"), ("b", "t2")])
    assert scorer.calls == [[("the query", "t1"), ("the query", "t2")]]


def test_membership_is_never_changed():
    scorer = FakeScorer({f"t{i}": float(i % 7) for i in range(30)})
    cands = [(f"c{i:02d}", f"t{i}") for i in range(30)]
    out = rerank.Reranker(score_pairs=scorer).rerank("q", cands)
    assert sorted(c for c, _ in out) == sorted(c for c, _ in cands)


def test_empty_candidates_do_not_call_the_model():
    scorer = FakeScorer({})
    assert rerank.Reranker(score_pairs=scorer).rerank("q", []) == []
    assert scorer.calls == []


def test_scores_are_plain_floats():
    import numpy as np
    r = rerank.Reranker(score_pairs=lambda pairs: np.array([0.5], dtype=np.float32))
    (_, score), = r.rerank("q", [("a", "t")])
    assert type(score) is float


# ---- real pinned model (local cache only) ---------------------------------------

@pytest.fixture(scope="module")
def real():
    return rerank.Reranker()


@pytest.mark.model
def test_model_pinned_revision_is_in_the_local_cache():
    from huggingface_hub import snapshot_download
    path = snapshot_download(rerank.MODEL_NAME, revision=rerank.MODEL_REVISION,
                             local_files_only=True)
    assert rerank.MODEL_REVISION in path


@pytest.mark.model
def test_model_runs_on_cpu(real):
    model = rerank.load_model()
    assert str(model.model.device) == "cpu"
    assert model.model.training is False


@pytest.mark.model
def test_model_ranks_the_obvious_answer_first(real):
    out = real.rerank("how do I bake bread at home",
                      [("x", "Qdrant is a vector database written in Rust."),
                       ("y", "Knead the dough, let it rise, then bake the bread at 220C.")])
    assert out[0][0] == "y"


@pytest.mark.model
def test_model_is_deterministic_across_calls(real):
    cands = [(f"c{i}", f"document number {i} about retrieval and reranking") for i in range(10)]
    assert real.rerank("reranking", cands) == real.rerank("reranking", cands)
