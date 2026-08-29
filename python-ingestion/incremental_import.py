from pathlib import Path

from parser import parse_chat
from import_state import ImportStateStore, identify_messages
from heuristic_filter import heuristic_filter
from grader import grade_messages, grade_pass2
from enrich import process_candidates, _merge_pass2
from pipeline import push_to_stores, MIN_QUALITY


def _summary(total: int, already: int, new: int,
             dropped: int, low_quality: int, stored: int) -> dict:
    return {
        "total_messages": total,
        "already_processed": already,
        "new_messages": new,
        "heuristically_dropped": dropped,
        "low_quality": low_quality,
        "stored": stored,
    }


def run_incremental_import(
    source_id: str,
    file_path: str,
    revision: int,
    db_path: str = "import_state.sqlite",
) -> dict:
    """Ingest only the messages not yet seen, recording outcomes in SQLite.

    Steps: parse -> generate IDs -> ask the store which are unseen -> return
    early if none -> grade/enrich/store the unseen subset -> mark them
    processed -> return an import summary.
    """
    store = ImportStateStore(db_path)
    source_type = source_id.split(":", 1)[0]
    file_name = Path(file_path).name

    # Register/update the source so Drive revision + status commands work.
    store.upsert_source(source_id, source_type, file_name, str(revision))

    datas = parse_chat(file_path)
    all_ids = identify_messages(source_id, datas)
    all_message_ids = [m.message_id for m in all_ids]
    unseen_ids = set(store.find_unseen_messages(source_id, all_message_ids))

    total = len(all_ids)
    new_count = len(unseen_ids)
    if new_count == 0:
        store.close()
        return _summary(total, total, 0, 0, 0, 0)

    # Map back from unseen IDs to the actual message objects.
    unseen = [m.data for m in all_ids if m.message_id in unseen_ids]
    key = lambda d: (d.datetime_iso, d.sender, d.text.strip())
    data_to_id = {key(m.data): m.message_id for m in all_ids}

    # Reuse the existing pipeline building blocks on the unseen subset.
    kept, _ = heuristic_filter(unseen)
    graded = grade_messages(kept)
    verified = grade_pass2(graded)
    final = _merge_pass2(graded, verified)

    candidates = [(d, g) for d, g in zip(kept, final)
                  if g.quality >= MIN_QUALITY]
    enriched, scraped, blocked, ask_user, _, _ = process_candidates(
        candidates
    )

    # Persist to Neo4j/Qdrant. Raise on failure so nothing is marked done.
    push_to_stores(enriched, scraped, ask_user, MIN_QUALITY, blocked=blocked)

    # Determine outcomes and record them.
    kept_set = set(map(id, kept))
    dropped_datas = [d for d in unseen if id(d) not in kept_set]
    outcomes = []
    for d in dropped_datas:
        outcomes.append((data_to_id[key(d)], "heuristic_drop"))
    for d, g in zip(kept, final):
        outcome = "stored" if g.quality >= MIN_QUALITY else "low_quality"
        outcomes.append((data_to_id[key(d)], outcome))

    store.mark_processed(source_id, outcomes)
    store.update_source_revision(source_id, str(revision))
    store.close()

    counts = {label: 0 for label in ("heuristic_drop", "low_quality", "stored")}
    for _, outcome in outcomes:
        counts[outcome] += 1

    return _summary(
        total,
        total - new_count,
        new_count,
        counts["heuristic_drop"],
        counts["low_quality"],
        counts["stored"],
    )
