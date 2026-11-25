"""Reusable embedded media extraction and recovery helpers."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit


EMBEDDED_MEDIA_HOSTS = (
    "slideslive.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "zoom.us",
    "zoom.com",
)
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".opus", ".webm", ".wav", ".aac", ".flac")
SECRET_QUERY_RE = re.compile(
    r"(?i)(token|key|secret|session|cookie|auth|bearer|signature|embed_init_token|user_uuid|pwd)"
)
SLIDESLIVE_ID_PATTERNS = (
    r"slideslive\.com/(?:embed/)?(?:presentation/)?(\d{7,})",
    r"presentationId\s*[:=]\s*['\"](\d{7,})['\"]",
    r"presentation-embed-(\d{7,})",
    r"data-slideslive-id\s*=\s*['\"](\d{7,})['\"]",
)


@dataclass(frozen=True)
class EmbeddedMediaLink:
    """A recoverable media reference found outside ordinary anchor hrefs."""

    url: str
    text: str = "Embedded media"
    tag: str = "embedded"


def sanitize_media_url(url: str) -> str:
    """Remove secret-bearing query params from a media URL.

    Only the query string is filtered (any secret-shaped param via
    ``SECRET_QUERY_RE``, keeping the small media allowlist).  The path is
    preserved verbatim: a prior "redact any 48+ opaque path run" rule
    silently rewrote legitimate long path segments -- e.g. CVPR paper
    filenames encoding the full title -- into ``REDACTED``, producing
    unreachable URLs.  A path segment is data, not a secret.
    """

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url
    safe_query = []
    for part in parsed.query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        if SECRET_QUERY_RE.search(key + value):
            continue
        if key in {"v", "id", "list", "t", "start"}:
            safe_query.append(part)
    return parsed._replace(query="&".join(safe_query), fragment="").geturl()


def _hostname(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def _host_matches(host: str, allowed: str) -> bool:
    return host == allowed or host.endswith(f".{allowed}")


def is_embedded_media_host(url: str) -> bool:
    host = _hostname(url)
    return any(_host_matches(host, marker) for marker in EMBEDDED_MEDIA_HOSTS)


def _path_parts(url: str) -> list[str]:
    try:
        return [part for part in urlsplit(url).path.split("/") if part]
    except ValueError:
        return []


def _is_youtube_video_url(url: str) -> bool:
    host = _hostname(url)
    parts = _path_parts(url)
    if _host_matches(host, "youtu.be"):
        return bool(parts and parts[0])
    if not _host_matches(host, "youtube.com"):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parts[:1] == ["watch"]:
        return bool(parse_qs(parsed.query).get("v", [""])[0])
    if parts[:1] in (["embed"], ["shorts"], ["live"]):
        return len(parts) >= 2 and bool(parts[1])
    return False


def _is_vimeo_video_url(url: str) -> bool:
    host = _hostname(url)
    if not _host_matches(host, "vimeo.com"):
        return False
    parts = _path_parts(url)
    if host == "player.vimeo.com":
        return len(parts) >= 2 and parts[0] == "video" and parts[1].isdigit()
    return bool(parts and parts[0].isdigit())


def is_recoverable_embedded_media_url(url: str) -> bool:
    """Return whether yt-dlp style recovery should be attempted for a URL."""

    if not is_embedded_media_host(url):
        return False
    host = _hostname(url)
    if _host_matches(host, "slideslive.com"):
        return bool(
            re.search(r"slideslive\.com/(?:embed/)?(?:presentation/)?\d{7,}", url, re.I)
        )
    if _host_matches(host, "youtube.com") or _host_matches(host, "youtu.be"):
        return _is_youtube_video_url(url)
    if _host_matches(host, "vimeo.com"):
        return _is_vimeo_video_url(url)
    return True


def embedded_media_links(
    text: str,
    *,
    base_url: str,
    direct_media_predicate: Callable[[str], bool] | None = None,
) -> list[EmbeddedMediaLink]:
    """Extract page-state media URLs and SlidesLive ids from HTML/JS text."""

    decoded = html.unescape(text).replace("\\/", "/")
    candidates: list[EmbeddedMediaLink] = []
    for pattern in (
        r"https?://[^\"'<>\s)]+",
        r"//(?:www\.)?(?:slideslive\.com|youtube\.com|youtu\.be|vimeo\.com|[^/]*zoom\.(?:us|com))/[^\"'<>\s)]+",
    ):
        for raw_url in re.findall(pattern, decoded, re.I):
            url = raw_url if raw_url.startswith("http") else f"https:{raw_url}"
            if is_recoverable_embedded_media_url(url) or (
                direct_media_predicate is not None and direct_media_predicate(url)
            ):
                candidates.append(EmbeddedMediaLink(url=url, text="Embedded media"))

    for pattern in SLIDESLIVE_ID_PATTERNS:
        for slideslive_id in re.findall(pattern, decoded, re.I):
            candidates.append(
                EmbeddedMediaLink(
                    url=f"https://slideslive.com/{slideslive_id}",
                    text="SlidesLive embedded media",
                )
            )

    seen: set[str] = set()
    unique: list[EmbeddedMediaLink] = []
    for link in candidates:
        absolute = urljoin(base_url, link.url)
        if absolute in seen:
            continue
        seen.add(absolute)
        unique.append(EmbeddedMediaLink(url=absolute, text=link.text, tag=link.tag))
    return unique


def existing_completed_audio(dest: Path) -> Path | None:
    """Return an existing downloaded audio artifact for a yt-dlp destination stem."""

    for extension in AUDIO_EXTENSIONS:
        candidate = dest.with_suffix(extension)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None
