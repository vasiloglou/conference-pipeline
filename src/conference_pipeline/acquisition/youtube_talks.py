"""Conference-general YouTube talk harvesting.

Program-spine-driven, bidirectional matching between a conference program and
publicly available YouTube talk videos:

- **Direction A (harvest -> program):** enumerate configured channels,
  playlists, and broad search queries, then match harvested videos back to the
  program spine by normalized title and speaker evidence.
- **Direction B (program -> search):** for every program item still unmatched
  after Direction A, run targeted per-item searches (yt-dlp ``ytsearch`` and an
  optional, runner-authorized Tavily web-search lane restricted to YouTube
  domains), highest-priority families first (keynote/invited, then oral).

The lane is **transcript-first**: matched videos get captions pulled with
yt-dlp (``--write-subs`` before ``--write-auto-subs`` so manual-vs-auto
provenance is recorded) and no audio is downloaded. Talks without usable
captions are marked ``asr_fallback_required`` for the transcription stage to
handle as an explicit per-talk decision.

Nothing in this module is conference-specific: the program spine, conference
label, channels, queries, and host allowlist all arrive as inputs. Receipts
record ``allowed_hosts`` and sanitized URLs only.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from conference_pipeline.acquisition.embedded_media import sanitize_media_url
from conference_pipeline.caption_text import (  # re-exported
    ROLLING_CAPTION_OVERLAP_WINDOW,
    collapse_rolling_caption_segments,
    parse_caption_file,
)

DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("youtube.com", "youtu.be")
DEFAULT_MATCH_THRESHOLD = 0.6
DEFAULT_PER_ITEM_MAX_RESULTS = 5
DEFAULT_TRANSCRIPT_LANGS: tuple[str, ...] = ("en",)
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Lower value = higher priority. Keynotes/invited talks and orals are the
# priority families: they are expected to have recorded videos, so a miss is a
# must-investigate gap rather than best-effort slack.
FAMILY_PRIORITY: dict[str, int] = {
    "keynote": 0,
    "invited": 0,
    "invited_talk": 0,
    "oral": 1,
    "oral_session": 1,
}
PRIORITY_FAMILIES: frozenset[str] = frozenset(
    family for family, priority in FAMILY_PRIORITY.items() if priority <= 1
)
DEFAULT_FAMILY_PRIORITY = 2

_WORD_RE = re.compile(r"[a-z0-9]+")
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


class YouTubeTalkHarvestError(RuntimeError):
    """Raised when the harvest cannot proceed at all (e.g. yt-dlp missing)."""


class YouTubeBotCheckBlocked(YouTubeTalkHarvestError):
    """YouTube answered with its "confirm you're not a bot" gate.

    This is an IP-level block, not evidence that a video lacks captions, so
    callers must stop hitting YouTube for the rest of the run instead of
    recording ``no caption`` (every further request extends the block).
    """


# Substrings yt-dlp prints when YouTube serves the sign-in/bot gate.
YOUTUBE_BOT_CHECK_MARKERS: tuple[str, ...] = (
    "confirm you're not a bot",
    "confirm you’re not a bot",
)


def is_youtube_bot_check(stderr: str | None) -> bool:
    text = (stderr or "").lower()
    return any(marker in text for marker in YOUTUBE_BOT_CHECK_MARKERS)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProgramItem:
    """One program-spine entry (keynote, oral, poster, ...)."""

    item_id: str
    title: str
    speakers: tuple[str, ...] = ()
    family: str = "other"
    session: str | None = None
    forum_id: str | None = None
    source_url: str | None = None

    @property
    def priority(self) -> int:
        return FAMILY_PRIORITY.get(self.family.strip().lower(), DEFAULT_FAMILY_PRIORITY)

    @property
    def is_priority_family(self) -> bool:
        return self.family.strip().lower() in PRIORITY_FAMILIES

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any], index: int) -> "ProgramItem":
        speakers_raw = record.get("speakers") or record.get("authors") or ()
        if isinstance(speakers_raw, str):
            speakers: tuple[str, ...] = (speakers_raw,)
        else:
            speakers = tuple(str(name) for name in speakers_raw if str(name).strip())
        item_id = str(
            record.get("item_id")
            or record.get("id")
            or record.get("forum_id")
            or f"item-{index:05d}"
        )
        return cls(
            item_id=item_id,
            title=str(record.get("title") or ""),
            speakers=speakers,
            family=str(record.get("family") or record.get("event_family") or "other"),
            session=(str(record["session"]) if record.get("session") else None),
            forum_id=(str(record["forum_id"]) if record.get("forum_id") else None),
            source_url=(str(record["source_url"]) if record.get("source_url") else None),
        )


@dataclass
class HarvestedVideo:
    """One candidate YouTube video discovered during harvest or search."""

    video_id: str
    title: str
    url: str
    channel: str | None = None
    duration_seconds: float | None = None
    source: str = "channel"  # channel | playlist | broad_search | item_search | tavily_search
    query: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "url": sanitize_media_url(self.url),
            "channel": self.channel,
            "duration_seconds": self.duration_seconds,
            "source": self.source,
            "query": self.query,
        }


@dataclass
class TalkMatch:
    """A program item matched to a YouTube video."""

    item: ProgramItem
    video: HarvestedVideo
    direction: str  # harvest_to_program | program_to_search
    confidence: float
    method: str  # title_exact | title_fuzzy | title_speaker
    transcript: dict[str, Any] | None = None


@dataclass
class HarvestConfig:
    """Inputs for one harvest run. All values are conference-supplied."""

    workspace: Path
    conference_label: str
    program_spine: Sequence[ProgramItem]
    sources: Sequence[str] = ()
    broad_queries: Sequence[str] = ()
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS
    per_item_search: bool = True
    per_item_max_results: int = DEFAULT_PER_ITEM_MAX_RESULTS
    tavily_api_key: str | None = None
    tavily_authority: Mapping[str, Any] | None = None
    transcripts_enabled: bool = True
    transcript_langs: Sequence[str] = DEFAULT_TRANSCRIPT_LANGS
    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    output_dir: Path | None = None
    receipt_path: Path | None = None
    resume: bool = False
    rate_limit_seconds: float = 1.0
    pilot_limit: int | None = None
    run_id: str | None = None
    # Optional paid tier: a ``youtube_transcripts.GeminiTranscriber`` used for
    # talks whose captions are missing or bot-gated. None keeps the lane free.
    gemini_transcriber: Any = None


# ---------------------------------------------------------------------------
# Title normalization and matching
# ---------------------------------------------------------------------------


def normalize_title(text: str, *, stop_tokens: Iterable[str] = ()) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop stop tokens."""

    tokens = _WORD_RE.findall(text.lower())
    stops = {token.lower() for token in stop_tokens}
    kept = [token for token in tokens if token not in stops]
    return " ".join(kept)


def title_tokens(text: str, *, stop_tokens: Iterable[str] = ()) -> frozenset[str]:
    return frozenset(normalize_title(text, stop_tokens=stop_tokens).split())


def title_similarity(a: str, b: str, *, stop_tokens: Iterable[str] = ()) -> float:
    """Token Jaccard with a containment boost for prefixed/suffixed uploads."""

    tokens_a = title_tokens(a, stop_tokens=stop_tokens)
    tokens_b = title_tokens(b, stop_tokens=stop_tokens)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union
    containment = intersection / min(len(tokens_a), len(tokens_b))
    # Author uploads commonly prefix "[CVPR 2026 Oral]" or suffix channel
    # branding, so containment matters as much as strict Jaccard.
    return max(jaccard, 0.9 * containment)


def _speaker_surname_tokens(item: ProgramItem) -> frozenset[str]:
    surnames: set[str] = set()
    for speaker in item.speakers:
        parts = _WORD_RE.findall(speaker.lower())
        if parts:
            surnames.add(parts[-1])
    return frozenset(surnames)


def score_video_for_item(
    item: ProgramItem,
    video: HarvestedVideo,
    *,
    stop_tokens: Iterable[str] = (),
) -> tuple[float, str]:
    """Score one candidate video against one program item."""

    item_norm = normalize_title(item.title, stop_tokens=stop_tokens)
    video_norm = normalize_title(video.title, stop_tokens=stop_tokens)
    if item_norm and item_norm == video_norm:
        return 1.0, "title_exact"
    base = title_similarity(item.title, video.title, stop_tokens=stop_tokens)
    surnames = _speaker_surname_tokens(item)
    haystack = title_tokens(video.title) | title_tokens(video.channel or "")
    if surnames and surnames & haystack:
        return min(1.0, base + 0.15), "title_speaker"
    return base, "title_fuzzy"


def match_videos_to_spine(
    spine: Sequence[ProgramItem],
    videos: Sequence[HarvestedVideo],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    stop_tokens: Iterable[str] = (),
    direction: str = "harvest_to_program",
) -> list[TalkMatch]:
    """Greedy best-first matching of harvested videos to program items."""

    candidates: list[tuple[float, str, int, int]] = []
    for item_index, item in enumerate(spine):
        for video_index, video in enumerate(videos):
            score, method = score_video_for_item(item, video, stop_tokens=stop_tokens)
            if score >= threshold:
                candidates.append((score, method, item_index, video_index))
    candidates.sort(key=lambda row: (-row[0], row[2], row[3]))
    matched_items: set[int] = set()
    matched_videos: set[int] = set()
    matches: list[TalkMatch] = []
    for score, method, item_index, video_index in candidates:
        if item_index in matched_items or video_index in matched_videos:
            continue
        matched_items.add(item_index)
        matched_videos.add(video_index)
        matches.append(
            TalkMatch(
                item=spine[item_index],
                video=videos[video_index],
                direction=direction,
                confidence=round(score, 4),
                method=method,
            )
        )
    return matches


# ---------------------------------------------------------------------------
# yt-dlp integration (injectable runner for tests)
# ---------------------------------------------------------------------------

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def resolve_yt_dlp_cmd() -> list[str]:
    """Resolve yt-dlp from PATH or the active Python environment."""

    executable = shutil.which("yt-dlp")
    if executable is not None:
        return [executable]
    try:
        __import__("yt_dlp")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise YouTubeTalkHarvestError(
            "yt-dlp not found. Install it with: pip install yt-dlp "
            "(declared in the project's 'media' extra)"
        ) from exc
    return [sys.executable, "-m", "yt_dlp"]


def _default_runner(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - fixed binary, no shell
        command,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def extract_video_id(url: str) -> str | None:
    """Extract a YouTube video id from watch/short/embed/youtu.be URLs."""

    try:
        from urllib.parse import parse_qs, urlsplit

        parsed = urlsplit(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate if _YOUTUBE_ID_RE.match(candidate) else None
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            values = parse_qs(parsed.query).get("v") or []
            candidate = values[0] if values else ""
            return candidate if _YOUTUBE_ID_RE.match(candidate) else None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            candidate = parts[1]
            return candidate if _YOUTUBE_ID_RE.match(candidate) else None
    return None


def url_host_allowed(url: str, allowed_hosts: Iterable[str]) -> bool:
    from urllib.parse import urlsplit

    try:
        host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return False
    for allowed in allowed_hosts:
        allowed_norm = allowed.lower().removeprefix("www.")
        if host == allowed_norm or host.endswith("." + allowed_norm):
            return True
    return False


def _videos_from_yt_dlp_json(
    payload: Mapping[str, Any],
    *,
    source: str,
    query: str | None,
    allowed_hosts: Iterable[str],
) -> list[HarvestedVideo]:
    entries = payload.get("entries")
    if entries is None:
        entries = [payload]
    videos: list[HarvestedVideo] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        video_id = str(entry.get("id") or "")
        url = str(entry.get("url") or entry.get("webpage_url") or "")
        if not url and video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        if not video_id:
            candidate = extract_video_id(url)
            if candidate is None:
                continue
            video_id = candidate
        if not url_host_allowed(url, allowed_hosts):
            continue
        duration = entry.get("duration")
        videos.append(
            HarvestedVideo(
                video_id=video_id,
                title=str(entry.get("title") or ""),
                url=url,
                channel=(
                    str(entry.get("channel") or entry.get("uploader") or "") or None
                ),
                duration_seconds=(float(duration) if duration else None),
                source=source,
                query=query,
            )
        )
    return videos


def enumerate_source_videos(
    source_url: str,
    *,
    runner: Runner,
    yt_dlp_cmd: Sequence[str],
    allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
) -> list[HarvestedVideo]:
    """Enumerate a channel or playlist with a flat (metadata-only) listing."""

    result = runner([*yt_dlp_cmd, "--flat-playlist", "-J", source_url])
    if result.returncode != 0 or not (result.stdout or "").strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    kind = "playlist" if "list=" in source_url or "/playlist" in source_url else "channel"
    return _videos_from_yt_dlp_json(
        payload, source=kind, query=None, allowed_hosts=allowed_hosts
    )


def search_videos(
    query: str,
    *,
    runner: Runner,
    yt_dlp_cmd: Sequence[str],
    max_results: int = DEFAULT_PER_ITEM_MAX_RESULTS,
    source: str = "item_search",
    allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
) -> list[HarvestedVideo]:
    """Search YouTube via yt-dlp's ytsearch pseudo-URL (metadata only)."""

    result = runner(
        [*yt_dlp_cmd, "--flat-playlist", "-J", f"ytsearch{max_results}:{query}"]
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return _videos_from_yt_dlp_json(
        payload, source=source, query=query, allowed_hosts=allowed_hosts
    )


# ---------------------------------------------------------------------------
# Tavily lane (optional, key from environment only)
# ---------------------------------------------------------------------------

HttpPost = Callable[[str, dict[str, Any]], dict[str, Any]]


def _default_http_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def tavily_search_videos(
    query: str,
    *,
    api_key: str,
    http_post: HttpPost = _default_http_post,
    max_results: int = DEFAULT_PER_ITEM_MAX_RESULTS,
    allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
) -> list[HarvestedVideo]:
    """Search the web for YouTube talk URLs through the Tavily API lane."""

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_domains": list(allowed_hosts),
    }
    try:
        response = http_post(TAVILY_SEARCH_URL, payload)
    except Exception:  # noqa: BLE001 - lane degrades to a recorded miss
        return []
    videos: list[HarvestedVideo] = []
    for row in response.get("results") or []:
        if not isinstance(row, Mapping):
            continue
        url = str(row.get("url") or "")
        if not url_host_allowed(url, allowed_hosts):
            continue
        video_id = extract_video_id(url)
        if video_id is None:
            continue
        videos.append(
            HarvestedVideo(
                video_id=video_id,
                title=str(row.get("title") or ""),
                url=url,
                channel=None,
                duration_seconds=None,
                source="tavily_search",
                query=query,
            )
        )
    return videos


def _authorized_tavily_item_ids(config: HarvestConfig) -> frozenset[str]:
    """Return only the non-secret, runner-bound priority frontier item IDs.

    ``HarvestConfig`` remains injectable for tests and runner adapters, but an
    API key alone is intentionally insufficient.  Generic command-line use is
    rejected separately; this guard prevents an accidental library caller from
    turning a bare key into broad paid search.
    """

    authority = config.tavily_authority
    if not config.tavily_api_key or not isinstance(authority, Mapping):
        return frozenset()
    item_ids = authority.get("priority_program_item_ids")
    if (
        authority.get("authority_kind") != "cvpr2026_tavily_priority_gap_bound"
        or not isinstance(authority.get("authority_digest"), str)
        or not isinstance(item_ids, (list, tuple))
        or not item_ids
        or not all(isinstance(item_id, str) and item_id for item_id in item_ids)
        or len(set(item_ids)) != len(item_ids)
    ):
        return frozenset()
    return frozenset(item_ids)


# ---------------------------------------------------------------------------
# Captions (transcript-first; manual before auto for provenance)
# ---------------------------------------------------------------------------


def download_captions(
    video: HarvestedVideo,
    dest_dir: Path,
    *,
    runner: Runner,
    yt_dlp_cmd: Sequence[str],
    langs: Sequence[str] = DEFAULT_TRANSCRIPT_LANGS,
) -> dict[str, Any] | None:
    """Fetch captions for one video without downloading any media.

    Tries author/manual subtitles first, then auto-generated ones, so the
    provenance (``youtube_caption_manual`` vs ``youtube_caption_auto``) is
    honest. Returns None when no caption track exists in the requested
    languages. Raises :class:`YouTubeBotCheckBlocked` when YouTube serves its
    bot gate, because that says nothing about whether captions exist.
    """

    dest_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(dest_dir / f"{video.video_id}.%(ext)s")
    lang_arg = ",".join(langs)
    attempts = (
        ("--write-subs", "youtube_caption_manual"),
        ("--write-auto-subs", "youtube_caption_auto"),
    )
    for flag, provenance in attempts:
        existing = _find_caption_file(dest_dir, video.video_id)
        if existing is None:
            result = runner(
                [
                    *yt_dlp_cmd,
                    "--skip-download",
                    flag,
                    "--sub-langs",
                    lang_arg,
                    "--sub-format",
                    "vtt",
                    "-o",
                    output_template,
                    video.url,
                ]
            )
            if is_youtube_bot_check(getattr(result, "stderr", None)):
                raise YouTubeBotCheckBlocked(
                    f"YouTube bot check while fetching captions for {video.video_id}"
                )
            existing = _find_caption_file(dest_dir, video.video_id)
        if existing is not None:
            language = _caption_language_from_name(existing.name, video.video_id)
            return {
                "caption_path": existing,
                "transcript_source": provenance,
                "language": language,
            }
    return None


def caption_sidecar_matches_video_id(path: str | Path, video_id: str) -> bool:
    """Return whether a caption sidecar is bound to exactly ``video_id``.

    yt-dlp names subtitle sidecars as ``<video-id>.<language>.<extension>``
    (or simply ``<video-id>.<extension>``).  Treating an arbitrary filename
    prefix as a match would let ``ABCDEF12345x.en.vtt`` be reused for
    ``ABCDEF12345``.  Keep this narrow binding shared by discovery and the
    conference-specific receipt validators.
    """

    name = Path(path).name
    suffix = Path(name).suffix.lower()
    return suffix in {".vtt", ".srt"} and name.startswith(f"{video_id}.")


def _find_caption_file(dest_dir: Path, video_id: str) -> Path | None:
    matches = sorted(
        path
        for pattern in ("*.vtt", "*.srt")
        for path in dest_dir.glob(pattern)
        if caption_sidecar_matches_video_id(path, video_id)
    )
    return matches[0] if matches else None


def _caption_language_from_name(name: str, video_id: str) -> str | None:
    if not caption_sidecar_matches_video_id(name, video_id):
        return None
    stem = name.rsplit(".", 1)[0]
    remainder = stem[len(video_id):].strip(".")
    return remainder or None


def write_transcript_sidecar(
    video: HarvestedVideo,
    caption_info: Mapping[str, Any],
    transcripts_dir: Path,
) -> dict[str, Any]:
    """Write the transcript JSON sidecar for one matched talk."""

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    segments = parse_caption_file(Path(caption_info["caption_path"]))
    text = " ".join(segment["text"] for segment in segments)
    record = {
        "video_id": video.video_id,
        "video_url": sanitize_media_url(video.url),
        "transcript_source": caption_info["transcript_source"],
        "language": caption_info.get("language"),
        "segment_count": len(segments),
        "word_count": len(text.split()),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "segments": segments,
        "text": text,
    }
    out_path = transcripts_dir / f"{video.video_id}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {
        "transcript_path": str(out_path),
        "transcript_source": record["transcript_source"],
        "language": record["language"],
        "segment_count": record["segment_count"],
        "word_count": record["word_count"],
        "text_sha256": record["text_sha256"],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_program_spine(path: Path) -> list[ProgramItem]:
    """Load a program spine JSON file (list or ``{"items": [...]}``)."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rows = payload.get("items") or payload.get("talks") or payload.get("entries") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        raise YouTubeTalkHarvestError(
            f"program spine {path} must be a list or an object with an 'items' list"
        )
    return [ProgramItem.from_mapping(row, index) for index, row in enumerate(rows)]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def harvest_youtube_talks(
    config: HarvestConfig,
    *,
    runner: Runner = _default_runner,
    http_post: HttpPost = _default_http_post,
    sleeper: Callable[[float], None] = time.sleep,
    yt_dlp_cmd: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the full bidirectional harvest and write inventory plus receipt."""

    if yt_dlp_cmd is None:
        yt_dlp_cmd = resolve_yt_dlp_cmd()
    stop_tokens = _WORD_RE.findall(config.conference_label.lower())
    output_dir = config.output_dir or (config.workspace / "storage" / "youtube-talks")
    captions_dir = output_dir / "captions"
    transcripts_dir = output_dir / "transcripts"
    inventory_path = output_dir / "talk_inventory.json"
    receipt_path = config.receipt_path or (output_dir / "harvest_receipt.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    previously_matched: dict[str, dict[str, Any]] = {}
    if config.resume and inventory_path.exists():
        try:
            prior = json.loads(inventory_path.read_text(encoding="utf-8"))
            for talk in prior.get("talks") or []:
                if isinstance(talk, Mapping) and talk.get("item_id"):
                    previously_matched[str(talk["item_id"])] = dict(talk)
        except (OSError, json.JSONDecodeError):
            previously_matched = {}

    spine = list(config.program_spine)
    if config.pilot_limit is not None:
        spine = sorted(spine, key=lambda item: item.priority)[: config.pilot_limit]

    resumed_item_ids = set(previously_matched)
    open_items = [item for item in spine if item.item_id not in resumed_item_ids]

    # Direction A: broad harvest.
    harvested: dict[str, HarvestedVideo] = {}
    source_stats: list[dict[str, Any]] = []
    for source_url in config.sources:
        videos = enumerate_source_videos(
            source_url,
            runner=runner,
            yt_dlp_cmd=yt_dlp_cmd,
            allowed_hosts=config.allowed_hosts,
        )
        source_stats.append(
            {
                "source": sanitize_media_url(source_url),
                "kind": "channel_or_playlist",
                "video_count": len(videos),
            }
        )
        for video in videos:
            harvested.setdefault(video.video_id, video)
        sleeper(config.rate_limit_seconds)
    for query in config.broad_queries:
        videos = search_videos(
            query,
            runner=runner,
            yt_dlp_cmd=yt_dlp_cmd,
            max_results=max(config.per_item_max_results * 10, 50),
            source="broad_search",
            allowed_hosts=config.allowed_hosts,
        )
        source_stats.append(
            {"source": query, "kind": "broad_search", "video_count": len(videos)}
        )
        for video in videos:
            harvested.setdefault(video.video_id, video)
        sleeper(config.rate_limit_seconds)

    matches = match_videos_to_spine(
        open_items,
        list(harvested.values()),
        threshold=config.match_threshold,
        stop_tokens=stop_tokens,
        direction="harvest_to_program",
    )
    matched_item_ids = {match.item.item_id for match in matches}

    # Direction B: targeted per-item search, priority families first.
    authorized_tavily_item_ids = _authorized_tavily_item_ids(config)
    tavily_lane = {
        "enabled": bool(authorized_tavily_item_ids),
        "queries": 0,
        "hits": 0,
        "status": "authorized" if authorized_tavily_item_ids else "not_authorized",
        "skipped_reason": (
            None
            if authorized_tavily_item_ids
            else ("TAVILY_API_KEY not set" if not config.tavily_api_key else "governed Tavily authority required")
        ),
    }
    if config.per_item_search:
        unmatched = sorted(
            (item for item in open_items if item.item_id not in matched_item_ids),
            key=lambda item: (item.priority, item.item_id),
        )
        for item in unmatched:
            speaker_part = f" {item.speakers[0]}" if item.speakers else ""
            query = f'"{item.title}"{speaker_part} {config.conference_label}'.strip()
            candidates = search_videos(
                query,
                runner=runner,
                yt_dlp_cmd=yt_dlp_cmd,
                max_results=config.per_item_max_results,
                source="item_search",
                allowed_hosts=config.allowed_hosts,
            )
            sleeper(config.rate_limit_seconds)
            if (
                not candidates
                and item.is_priority_family
                and item.item_id in authorized_tavily_item_ids
            ):
                tavily_lane["queries"] += 1
                candidates = tavily_search_videos(
                    query,
                    api_key=config.tavily_api_key,
                    http_post=http_post,
                    max_results=config.per_item_max_results,
                    allowed_hosts=config.allowed_hosts,
                )
                if candidates:
                    tavily_lane["hits"] += 1
                sleeper(config.rate_limit_seconds)
            item_matches = match_videos_to_spine(
                [item],
                candidates,
                threshold=config.match_threshold,
                stop_tokens=stop_tokens,
                direction="program_to_search",
            )
            if item_matches:
                matches.append(item_matches[0])
                matched_item_ids.add(item.item_id)
                harvested.setdefault(
                    item_matches[0].video.video_id, item_matches[0].video
                )

    # Transcript-first: YouTube captions, then the optional Gemini tier.
    from conference_pipeline.acquisition.youtube_transcripts import (  # noqa: PLC0415 - avoids an import cycle
        GeminiBudgetExhausted,
        YouTubeLaneState,
        acquire_transcript,
    )

    caption_failures: list[dict[str, Any]] = []
    youtube_state = YouTubeLaneState()
    transcript_tiers: dict[str, int] = {}
    gemini_halted_reason: str | None = None
    if config.transcripts_enabled:
        for match in matches:
            if gemini_halted_reason is not None:
                outcome = {"tier": None, "status": "gemini_budget_halted"}
            else:
                try:
                    outcome = acquire_transcript(
                        match.video,
                        captions_dir=captions_dir,
                        transcripts_dir=transcripts_dir,
                        runner=runner,
                        yt_dlp_cmd=yt_dlp_cmd,
                        youtube_state=youtube_state,
                        langs=config.transcript_langs,
                        gemini=config.gemini_transcriber,
                    )
                except GeminiBudgetExhausted as exc:
                    gemini_halted_reason = str(exc)
                    outcome = {"tier": None, "status": "gemini_budget_halted"}
            if outcome.get("status") == "transcribed":
                tier = str(outcome.get("tier"))
                transcript_tiers[tier] = transcript_tiers.get(tier, 0) + 1
                match.transcript = outcome["transcript"]
            else:
                caption_failures.append(
                    {
                        "item_id": match.item.item_id,
                        "video_id": match.video.video_id,
                        "reason": outcome.get("status"),
                        "youtube_outcome": outcome.get("youtube_outcome"),
                        "next_action": (
                            "resume after the YouTube block clears or enable the Gemini tier"
                            if outcome.get("status") == "youtube_blocked_gemini_disabled"
                            else "route to transcription.session_speech as asr_fallback, or enable the Gemini tier"
                        ),
                    }
                )
                match.transcript = {"transcript_source": "asr_fallback_required"}
            if not youtube_state.blocked:
                sleeper(config.rate_limit_seconds)

    # Build inventory: resumed rows first, then fresh matches.
    talks: list[dict[str, Any]] = list(previously_matched.values())
    for match in matches:
        talks.append(
            {
                "item_id": match.item.item_id,
                "item_title": match.item.title,
                "family": match.item.family,
                "priority_family": match.item.is_priority_family,
                "speakers": list(match.item.speakers),
                "forum_id": match.item.forum_id,
                "video": match.video.to_record(),
                "youtube_video_id": match.video.video_id,
                "match_direction": match.direction,
                "match_confidence": match.confidence,
                "match_method": match.method,
                "transcript": match.transcript,
            }
        )

    all_matched_ids = resumed_item_ids | matched_item_ids
    gaps = [
        {
            "item_id": item.item_id,
            "title": item.title,
            "family": item.family,
            "priority": item.priority,
            "must_investigate": item.is_priority_family,
        }
        for item in spine
        if item.item_id not in all_matched_ids
    ]
    priority_gaps = [gap for gap in gaps if gap["must_investigate"]]
    if not priority_gaps and not tavily_lane["queries"]:
        tavily_lane.update(
            {
                "enabled": False,
                "status": "not_required",
                "skipped_reason": "no unmatched priority gaps",
            }
        )

    inventory = {
        "schema_version": 1,
        "conference_label": config.conference_label,
        "run_id": config.run_id,
        "generated_at": _utc_now_iso(),
        "allowed_hosts": list(config.allowed_hosts),
        "program_item_count": len(spine),
        "matched_count": len(talks),
        "gap_count": len(gaps),
        "priority_gap_count": len(priority_gaps),
        "talks": talks,
        "gaps": gaps,
    }
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    status = "succeeded"
    if priority_gaps or caption_failures:
        status = "partial"
    receipt = {
        "schema_version": 1,
        "stage": "acquisition.youtube_talk_harvest",
        "status": status,
        "run_id": config.run_id,
        "generated_at": _utc_now_iso(),
        "conference_label": config.conference_label,
        "workspace": str(config.workspace),
        "inventory_path": str(inventory_path),
        "allowed_hosts": list(config.allowed_hosts),
        "counts": {
            "program_items": len(spine),
            "harvested_videos": len(harvested),
            "matched_talks": len(talks),
            "resumed_talks": len(resumed_item_ids),
            "direction_a_matches": sum(
                1 for match in matches if match.direction == "harvest_to_program"
            ),
            "direction_b_matches": sum(
                1 for match in matches if match.direction == "program_to_search"
            ),
            "gaps": len(gaps),
            "priority_gaps": len(priority_gaps),
            "caption_failures": len(caption_failures),
            "transcribed_by_tier": transcript_tiers,
        },
        "sources": source_stats,
        "tavily_lane": tavily_lane,
        "caption_failures": caption_failures,
        "youtube_lane": {
            "blocked": youtube_state.blocked,
            "blocked_at_video_id": youtube_state.blocked_at_video_id,
            "caption_requests": youtube_state.caption_requests,
        },
        "gemini_lane": (
            config.gemini_transcriber.lane_summary()
            if config.gemini_transcriber is not None
            else {"enabled": False}
        ),
        "gemini_halted_reason": gemini_halted_reason,
        "boundaries": {
            "audio_downloaded": False,
            "video_downloaded": False,
            "paywalled_sources_touched": False,
            "transcript_first": config.transcripts_enabled,
        },
        "pilot_limit": config.pilot_limit,
        "pilot_limit_note": (
            "pilot_limit caps the spine for development runs only; live "
            "production corpus commands must not use item-count caps"
            if config.pilot_limit is not None
            else None
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path)
    return receipt
