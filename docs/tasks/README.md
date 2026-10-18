# Retrieval ladder: task cards

Read `docs/retrieval-ladder-spec.md` first, especially section 3 (rules) and 17 (escalate).
One card = one unit = one branch/worktree = one executor. **H** cards are for the owner, **U** cards
for executors. All commands run from `python-ingestion/`.

| Card | Wave | Track | Depends on | Owns |
|---|---|---|---|---|
| [U0](U0.md) | 1 | live prerequisite | — | `vector_store.py` + 5 Qdrant scripts |
| [H0](H0.md) | 1 | live (owner) | U0 merged | `docker-compose.yaml`, both `.env` |
| [U1](U1.md) | 1 | live | H0 | `evals/live_guard.py` |
| [U2](U2.md) | 1 | live | U1 | `evals/freeze_graph.py` |
| [U3](U3.md) | 1 | offline | — | `evals/run_bm25.py` |
| [U4](U4.md) | 1 | offline | — | `evals/fusion.py` |
| [U5](U5.md) | 1 | offline | — | `evals/rerank.py` |
| [U6](U6.md) | 1 | offline | — | `evals/topup.py` |
| [U7](U7.md) | 1 | offline | — | `evals/ladder_stats.py` |
| [U8](U8.md) | 1 | live | U1, U4, U5 | `evals/latency_bench.py` |
| [U9](U9.md) | 2 | integration | U2, U3, U4, U5 | `evals/run_ladder.py` |
| [H1](H1.md) | 2 | owner | U9, U6 | `annotation_topup.csv` grades |
| [U10](U10.md) | 2 | integration | H1, U7, U8 | `evals/score_ladder.py` |

Every card's definition of done also includes, on the merged branch:
`uv run pytest -q --ignore-glob='tests/test_ladder_*.py'` green (170 pre-existing tests),
`uv run pytest tests/test_ladder_lock.py -q` green, the card's own test file green, and
`git diff --name-only ladder-base...HEAD` listing only the card's owned files.

Recorded red state (2026-10-18, before any implementation): each `test_ladder_*.py` except the
lock test fails at collection with `ImportError: cannot import name '<module>' from 'evals'`;
`test_ladder_qdrant_api_key.py` collects and fails 6 tests (`assert None == 'placeholder-from-env'` and
"QdrantClient call without api_key" for the 5 scripts). The planner checked that every offline and
model test can pass, using a throwaway reference implementation that was not committed.
