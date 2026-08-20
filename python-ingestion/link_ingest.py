import datetime as dt
import sys
from pathlib import Path

from models import load_list, save_list, to_dict, ScrapedLink
import enhancer

ASK_USER_FILE = "ask_user_links.json"
SCRAPED_FILE = "scraped_links.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _pending() -> list:
    items = load_list(ASK_USER_FILE)
    return sorted([i for i in items if not i.get("resolved")],
                  key=lambda x: x.get("sent_at", ""))


def list_ask_user() -> None:
    pending = _pending()
    if not pending:
        print("No unresolved ask-user links.")
        return
    for idx, it in enumerate(pending):
        ctx = it.get("message_context", {})
        preview = (ctx.get("original_text") or "")[:40]
        print(f"{idx:>3}  {it.get('sent_at','')[:26]:<26} "
              f"{it.get('block_reason',''):<22} {it.get('url','')}")
        if preview:
            print(f"      context: {preview!r}")
    print(f"\n{len(pending)} unresolved. Use: link_ingest.py --ingest <#>")


def ingest(index: int, file_path: str = None) -> None:
    items = load_list(ASK_USER_FILE)
    pending = sorted([i for i in items if not i.get("resolved")],
                     key=lambda x: x.get("sent_at", ""))
    if not pending:
        print("No unresolved ask-user links.")
        return
    if index < 0 or index >= len(pending):
        print(f"index out of range (0..{len(pending) - 1})")
        return
    rec = pending[index]

    if file_path:
        try:
            content = Path(file_path).read_text(encoding="utf-8").strip()
        except OSError as e:
            print(f"cannot read {file_path}: {e}")
            return
    else:
        print("Paste the content of the link below, then press Ctrl-D/Ctrl-Z on an empty line:")
        lines = []
        try:
            for line in sys.stdin:
                lines.append(line)
        except KeyboardInterrupt:
            print("\nCancelled; link left unresolved.")
            return
        content = "".join(lines).strip()

    if not content:
        print("No content provided; link left unresolved.")
        return

    print("[ingest] summarizing pasted content...")
    res = enhancer.summarize_content(rec.get("url", ""), "manual-paste", content)
    ctx = rec.get("message_context", {})
    sl = ScrapedLink(
        url=rec.get("url", ""),
        sent_at=rec.get("sent_at", ""),
        message_context=ctx,
        status="scraped",
        final_url=rec.get("url", ""),
        title="manual-paste",
        source="manual_paste",
        raw_text=content,
        summary=res["summary"],
        link_intent=ctx.get("link_intent") or "",
        topics=res["topics"],
        entities=res["entities"],
        scraped_at=_now(),
    )
    scraped = load_list(SCRAPED_FILE)
    scraped.append(to_dict(sl))
    save_list(SCRAPED_FILE, scraped)

    rec["resolved"] = True
    rec["ingested_at"] = _now()
    save_list(ASK_USER_FILE, items)
    print(f"Ingested {rec.get('url')} -> {SCRAPED_FILE} (ask-user record marked resolved)")


def main() -> None:
    args = sys.argv[1:]
    if "--list" in args or "-l" in args:
        list_ask_user()
        return
    if args and args[0] == "--ingest":
        idx = int(args[1]) if len(args) > 1 else None
        if idx is None:
            print("usage: link_ingest.py --ingest <index> [--file path]")
            return
        fp = None
        if "--file" in args:
            fp = args[args.index("--file") + 1]
        ingest(idx, fp)
        return
    if args and args[0] == "--skip":
        print("Skipped: the link stays unresolved on the list.")
        return
    print("link_ingest.py  --list | --ingest <index> [--file path] | --skip <index>")


if __name__ == "__main__":
    main()