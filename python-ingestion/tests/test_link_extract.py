import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from link_extract import extract_links, is_github, is_x, is_youtube, parse_github_url, youtube_video_id

FAILS = []


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


urls = extract_links("See https://example.com/a?utm_source=x and http://q.com/b. "
                     "Also (https://x.com/user/status/123) and https://youtu.be/abc123!")
check("extracts all urls", len(urls) == 4, urls)
check("strips trailing punct", urls[1] == "http://q.com/b" and urls[2] == "https://x.com/user/status/123", urls)
check("keeps utm param as-is", "utm_source=x" in urls[0])
check("dedupes", extract_links("https://a.com https://a.com") == ["https://a.com"])

check("is_youtube youtube.com", is_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
check("is_youtube youtu.be", is_youtube("https://youtu.be/dQw4w9WgXcQ"))
check("is_youtube shorts", is_youtube("https://www.youtube.com/shorts/dQw4w9WgXcQ"))
check("not youtube", not is_youtube("https://github.com/x/y"))

check("video id watch", youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ")
check("video id youtu.be", youtube_video_id("https://youtu.be/dQw4w9WgXcQ?s=46") == "dQw4w9WgXcQ")
check("video id shorts", youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ")
check("video id embed", youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ")
check("video id none", youtube_video_id("https://github.com/x/y") is None)

check("is_github repo", is_github("https://github.com/qdrant/qdrant"))
check("is_github raw", is_github("https://raw.githubusercontent.com/a/b/main/f.py"))
check("is_github gist", is_github("https://gist.github.com/u/abc123"))
check("not github", not is_github("https://example.com/a"))

g = parse_github_url("https://github.com/qdrant/qdrant")
check("gh repo", g.kind == "repo" and g.owner == "qdrant" and g.repo == "qdrant", g)
g = parse_github_url("https://github.com/qdrant/qdrant.git")
check("gh repo strips .git", g.repo == "qdrant", g)
g = parse_github_url("https://github.com/qdrant/qdrant/tree/master/docs")
check("gh tree", g.kind == "tree" and g.ref == "master" and g.path == "docs", g)
g = parse_github_url("https://github.com/qdrant/qdrant/blob/main/src/main.rs")
check("gh blob", g.kind == "blob" and g.ref == "main" and g.path == "src/main.rs", g)
g = parse_github_url("https://raw.githubusercontent.com/qdrant/qdrant/main/src/main.rs")
check("gh raw", g.kind == "raw" and g.owner == "qdrant" and g.path == "src/main.rs", g)
g = parse_github_url("https://github.com/qdrant/qdrant/issues/123")
check("gh issues", g.kind == "issues" and g.owner == "qdrant", g)
g = parse_github_url("https://github.com/qdrant/qdrant/pull/7")
check("gh pull", g.kind == "pulls", g)
g = parse_github_url("https://github.com/qdrant/qdrant/discussions")
check("gh discussions", g.kind == "discussions", g)
g = parse_github_url("https://gist.github.com/u/abc123")
check("gh gist", g.kind == "gist", g)
g = parse_github_url("https://github.com/openai")
check("gh org -> other", g.kind == "other", g)
check("gh non-github -> None", parse_github_url("https://example.com/x") is None)

check("is_x x.com", is_x("https://x.com/user/status/123"))
check("is_x twitter.com", is_x("https://twitter.com/user/status/123"))
check("not x", not is_x("https://example.com/x") and not is_x("https://youtube.com/x"))

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
