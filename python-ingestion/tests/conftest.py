"""Opt-in test groups for the retrieval ladder (docs/retrieval-ladder-spec.md).

  --live       tests that read the real Qdrant and Neo4j (read-only, enforced)
  --model      tests that load the pinned cross-encoder weights from the HF cache
  --artifacts  tests that check frozen files under evals/data/ after a run

Tests in a group are *deselected* unless their flag is given, never skipped:
once a flag is passed, an unreachable store or a missing file is a failure.
LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

GROUPS = {
    "live": "reads the real Qdrant and Neo4j through the read-only harness",
    "model": "loads the pinned cross-encoder from the local Hugging Face cache",
    "artifacts": "checks frozen run files under evals/data/",
}


def pytest_addoption(parser):
    for name, help_text in GROUPS.items():
        parser.addoption(f"--{name}", action="store_true", default=False,
                         help=f"run tests marked '{name}': {help_text}")


def pytest_configure(config):
    for name, help_text in GROUPS.items():
        config.addinivalue_line("markers", f"{name}: {help_text}")


def pytest_collection_modifyitems(config, items):
    keep, drop = [], []
    for item in items:
        gated = [g for g in GROUPS if item.get_closest_marker(g)]
        if all(config.getoption(f"--{g}") for g in gated):
            keep.append(item)
        else:
            drop.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
