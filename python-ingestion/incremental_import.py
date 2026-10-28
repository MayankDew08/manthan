from pathlib import Path
import datetime as _dt
import hashlib
import os
import re
import tempfile

from parser import parse_chat
from import_state import ImportStateStore, identify_messages
from heuristic_filter import heuristic_filter
from main import run_pipeline
from ingest import build_records_node
from pipeline import push_to_stores, MIN_QUALITY

_PLACEHOLDER_AUTHOR = "x"


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


def _serialized_text(messages) -> str:
    """WhatsApp-style text for ``messages`` that ``parser.parse_chat`` reads back identically.

    ``run_pipeline`` (the existing graph) parses a file, so the unseen subset is
    serialized here. The format and multiline/media handling match ``parser.parse_chat``
    so the graph's heuristic partition equals the one we compute locally.
    """
    lines = []
    for d in messages:
        dt = _dt.datetime.fromisoformat(d.datetime_iso)
        head = dt.strftime("%d/%m/%Y, %I:%M:%S %p")
        parts = d.text.split("\n")
        # parse_chat treats a header with no "author: " prefix as a WhatsApp
        # system event and skips it, so emit a fixed placeholder author. The
        # parser discards it again; it is not an identity.
        lines.append(f"{head} - {_PLACEHOLDER_AUTHOR}: {parts[0]}")
        lines.extend(parts[1:])
    return "".join(line + "\n" for line in lines)


def _batch_path(source_id: str, revision, content: str) -> str:
    """Temp path for a batch; run_pipeline derives its checkpoint thread from it.

    The content digest keeps a different batch from ever resuming an unfinished
    run left on the same source+revision (e.g. after revisions are reset), while
    a crash on the *same* batch still resumes its own checkpoint.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{source_id}-{revision}")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"manthan-batch-{safe}-{digest}.txt")


def _heuristic_split(messages):
    """Split into kept/discarded (order preserved) using the original Data objects."""
    kept, _ = heuristic_filter(messages)
    kept_ids = {id(m) for m in kept}
    discarded = [m for m in messages if id(m) not in kept_ids]
    return kept, discarded


def run_incremental_import(
    source_id: str,
    file_path: str,
    revision: int | str,
    db_path: str = "import_state.sqlite",
) -> dict:
    """Ingest only the messages not yet seen, via the existing LangGraph pipeline.

    Steps: parse -> generate IDs -> ask the store which are unseen -> write the
    unseen subset to a temp file -> run the existing graph (heuristic -> grade ->
    link scrape -> merge -> persist) -> build full message records -> sync to
    Neo4j/Qdrant -> record outcomes -> advance the revision.
    """
    store = ImportStateStore(db_path)
    try:
        source_type = source_id.split(":", 1)[0]
        file_name = Path(file_path).name

        # Register source metadata only; revision advances only after a successful run.
        store.upsert_source(source_id, source_type, file_name)

        datas = parse_chat(file_path)
        all_ids = identify_messages(source_id, datas)
        all_message_ids = [m.message_id for m in all_ids]
        unseen_ids = set(store.find_unseen_messages(source_id, all_message_ids))

        total = len(all_ids)
        new_count = len(unseen_ids)
        if new_count == 0:
            # Fully processed through this revision: advance it, no Gemma calls.
            if total == 0:
                print(f"WARNING: parsed 0 messages from {file_path} "
                      f"— unsupported chat format? Nothing stored.")
            store.update_source_revision(source_id, str(revision))
            return _summary(total, total, 0, 0, 0, 0)

        unseen = [m.data for m in all_ids if m.message_id in unseen_ids]
        data_to_id = {id(m.data): m.message_id for m in all_ids}

        # Heuristic filter locally so we keep the original Data objects (and their ids)
        # for outcome mapping. The graph re-derives the same partition from the temp
        # file below, in the same order.
        kept_unseen, discarded_unseen = _heuristic_split(unseen)

        content = _serialized_text(unseen)
        tmp_path = _batch_path(source_id, revision, content)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)

        try:
            # Existing graph: heuristic (first node) -> grade -> link scrape -> merge
            # -> persist JSON artifacts.
            state = run_pipeline(tmp_path)

            final = state.get("final") or []
            if len(final) != len(kept_unseen):
                # A parse/serialization mismatch would silently misalign outcomes.
                raise RuntimeError(
                    f"graph graded {len(final)} messages but {len(kept_unseen)} "
                    f"were expected after heuristic filtering"
                )
            # Counts alone cannot catch a resumed stale run that graded other
            # texts (2026-10-09: 148 empty-text grades passed as 148 == 148).
            stale = next((i for i, (g, d) in enumerate(zip(final, kept_unseen))
                          if g.original_text != d.text), None)
            if stale is not None:
                raise RuntimeError(
                    f"graph grades do not match this batch (first mismatch at "
                    f"message {stale}); refusing to record outcomes"
                )

            # Build full Message records (linkless included) via the existing node fn.
            built = build_records_node({
                "kept": kept_unseen,
                "final": final,
                "enriched": state.get("enriched") or [],
                "trusted_flags": [False] * len(kept_unseen),
            })

            # Sync to Neo4j/Qdrant. Raise on failure so nothing is marked done.
            push_to_stores(
                built["enriched"],
                state.get("scraped") or [],
                state.get("ask_user") or [],
                MIN_QUALITY,
                blocked=state.get("blocked") or [],
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Determine outcomes. Positional alignment + id() keeps duplicates correct.
        outcomes = []
        for i, g in enumerate(final):
            outcome = "stored" if g.quality >= MIN_QUALITY else "low_quality"
            outcomes.append((data_to_id[id(kept_unseen[i])], outcome))
        for d in discarded_unseen:
            outcomes.append((data_to_id[id(d)], "heuristic_drop"))

        store.mark_processed(source_id, outcomes)
        store.update_source_revision(source_id, str(revision))

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
    finally:
        store.close()
