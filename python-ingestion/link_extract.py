import re
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Optional

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_TRAILING = ".,;:!?)]}>*_\"'"
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


@dataclass
class GithubRef:
    kind: str
    owner: str = ""
    repo: str = ""
    ref: str = ""
    path: str = ""


def is_github(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return (host in ("github.com", "www.github.com", "raw.githubusercontent.com",
                     "gist.github.com")
            or host.endswith(".github.com") or "githubusercontent.com" in host)


def is_x(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ("x.com", "www.x.com", "twitter.com", "www.twitter.com",
                    "mobile.twitter.com") or host.endswith(".twitter.com") \
        or host.endswith(".x.com")


def parse_github_url(url: str) -> Optional[GithubRef]:
    if not is_github(url):
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    seg = [s for s in parsed.path.strip("/").split("/") if s]
    if "gist.github.com" in host:
        return GithubRef(kind="gist")
    if "raw.githubusercontent.com" in host:
        if len(seg) < 4:
            return GithubRef(kind="other")
        return GithubRef(kind="raw", owner=seg[0], repo=_strip_git(seg[1]),
                         ref=seg[2], path="/".join(seg[3:]))
    if len(seg) < 2:
        return GithubRef(kind="other")
    owner, repo = seg[0], _strip_git(seg[1])
    rest = seg[2:]
    if not rest:
        return GithubRef(kind="repo", owner=owner, repo=repo)
    sub = rest[0].lower()
    if sub in ("tree", "blob", "raw"):
        ref = rest[1] if len(rest) > 1 else ""
        return GithubRef(kind=sub, owner=owner, repo=repo, ref=ref,
                         path="/".join(rest[2:]))
    if sub == "issues":
        return GithubRef(kind="issues", owner=owner, repo=repo)
    if sub in ("pull", "pulls"):
        return GithubRef(kind="pulls", owner=owner, repo=repo)
    if sub == "discussions":
        return GithubRef(kind="discussions", owner=owner, repo=repo)
    return GithubRef(kind="other")


def _strip_git(name: str) -> str:
    return name[:-4] if name.endswith(".git") else name


def _clean_url(url: str) -> str:
    while url and url[-1] in _TRAILING:
        url = url[:-1]
    return url.strip()


def extract_links(text: str) -> List[str]:
    if not text:
        return []
    out = []
    seen = set()
    for m in _URL_RE.finditer(text):
        url = _clean_url(m.group(0))
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def is_youtube(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in _YOUTUBE_HOSTS or "youtube" in host


def youtube_video_id(url: str) -> Optional[str]:
    if not is_youtube(url):
        return None
    parsed = urlparse(url)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        seg = [s for s in parsed.path.split("/") if s]
        if seg:
            return _clean_video_id(seg[0])
    for key in ("v", "vi"):
        vid = _query_param(parsed, key)
        if vid:
            return _clean_video_id(vid)
    for marker in ("/shorts/", "/live/", "/embed/", "/v/"):
        if marker in parsed.path:
            seg = [s for s in parsed.path.split(marker, 1)[-1].split("/") if s]
            if seg:
                return _clean_video_id(seg[0])
    return None


def _query_param(parsed, key: str) -> Optional[str]:
    q = parsed.query
    for part in q.split("&"):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None


def _clean_video_id(vid: str) -> Optional[str]:
    vid = vid.strip()
    if vid and len(vid) <= 64 and re.fullmatch(r"[A-Za-z0-9_\-]+", vid):
        return vid
    return None
