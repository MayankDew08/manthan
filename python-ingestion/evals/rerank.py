"""Deterministic cross-encoder re-ranker for the retrieval ladder (spec section 9).

The weights are pinned to one revision and read from the local Hugging Face cache;
nothing here ever reaches the network. Ranking depends only on the (query, text)
pairs handed in — no wall-clock time, randomness or environment value is consulted,
so the same input always produces the same output.
"""

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
MAX_LENGTH = 512
SCORE_DECIMALS = 6

_MODEL = None


def load_model():
    """Return the pinned cross-encoder, in eval mode, loaded from the local cache.

    The import is deliberately inside the function: the offline callers that inject
    their own scorer should not pay for loading the deep-learning stack.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            MODEL_NAME,
            revision=MODEL_REVISION,
            device="cpu",
            max_length=MAX_LENGTH,
            local_files_only=True,
        )
        model.model.eval()
        _MODEL = model
    return _MODEL


class Reranker:
    """Scores (query, text) pairs and orders candidates by descending score."""

    def __init__(self, score_pairs=None):
        # An injected callable replaces the real model; the default loads it lazily.
        self._score_pairs = score_pairs

    def score_pairs(self, pairs):
        """Score a list of (query, text) pairs, in the order given."""
        if self._score_pairs is None:
            self._score_pairs = load_model().predict
        return self._score_pairs(list(pairs))

    def rerank(self, query, candidates):
        """Return [(candidate_id, score)] ordered by (-score, candidate_id).

        Membership is never altered — every input id comes back exactly once — and
        scoring happens in a single call so the model sees the input order intact.
        """
        pairs = [(query, text) for _candidate_id, text in candidates]
        if not pairs:
            return []
        raw_scores = self.score_pairs(pairs)
        scored = [
            (candidate_id, round(float(score), SCORE_DECIMALS))
            for (candidate_id, _text), score in zip(candidates, raw_scores)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored