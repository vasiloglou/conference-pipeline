#!/usr/bin/env python3
"""Acquire ICLR 2026 workshop talk media or explicit per-workshop blockers."""

from __future__ import annotations

import argparse
import hashlib
import html
import html.parser
import json
import mimetypes
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

from conference_pipeline.acquisition.embedded_media import (
    embedded_media_links,
    existing_completed_audio,
    is_embedded_media_host,
    is_recoverable_embedded_media_url,
    sanitize_media_url,
)
from conference_pipeline.config import PipelineSettings
from conference_pipeline.downloader import DownloadError, YtDlpDownloader
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.slideslive_recovery import (
    SlidesLivePageAsset,
    merge_media_artifacts,
    sanitize_url as sanitize_slideslive_url,
)
from conference_pipeline.slideslive_slides import recover_slide_assets


DEFAULT_WORKSPACE = Path("conferences/iclr2026")
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_WORKSHOP_LIST_URL = "https://iclr.cc/virtual/2026/events/workshop"
DEFAULT_EMBEDDED_MEDIA_HOSTS = ("slideslive.com",)
WORKSHOPS_DIR = Path("storage/workshops")
WORKSHOP_INVENTORY = WORKSHOPS_DIR / "inventory.json"
WORKSHOP_DOWNLOAD_RECEIPT = WORKSHOPS_DIR / "download_receipt.json"
TALK_MEDIA_RECEIPT = WORKSHOPS_DIR / "talk_media_acquisition_receipt.json"
TALK_MEDIA_INVENTORY = WORKSHOPS_DIR / "talk_media_inventory.json"
TALK_MEDIA_DIR = WORKSHOPS_DIR / "talk-media"
DOWNLOAD_CORPUS_INVENTORY = Path("storage/download-corpus/download_corpus.json")
ICLR_CACHED_WORKSHOP_LIST = Path("cache/acquisition-downloads/source-pages/workshops.html")
HTTP_USER_AGENT = "conference-pipeline-iclr2026-workshop-talk-media/1.0"
DEFAULT_PRODUCTION_BLOCKER_REASON = (
    "Default production mode uses the cached ICLR workshop listing as a deterministic "
    "boundary because live workshop detail crawling did not complete reliably; rerun "
    "with --live-crawl to attempt direct public media downloads."
)
DIRECT_MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m3u8",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
AUTH_GATE_PHRASES = (
    "log in and register to view live content",
    "login and register to view live content",
    "please log in to view",
    "accounts/login",
)
PROJECT_LINK_HINTS = (
    "project page",
    "website",
    "workshop website",
    "home page",
    "homepage",
    "program",
    "schedule",
    "agenda",
)
MEDIA_FIELD_NAMES = (
    "media_kind",
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "source_url",
    "media_url",
    "local_path",
    "sha256",
    "byte_count",
    "status",
)
SLIDESLIVE_URL_ID_RE = re.compile(
    r"slideslive\.com/(?:embed/)?(?:presentation/)?(?P<id>\d{7,})", re.I
)


WORKSHOP_TITLE_HINTS: dict[str, tuple[str, ...]] = {
    "AFA": ("advances in financial ai", "financial ai"),
    "AFAA": ("algorithmic fairness across alignment",),
    "AI4Mat": ("ai4mat", "accelerated materials"),
    "AI4Peace": ("ai for peace",),
    "AIMS": ("mechanism design", "aims"),
    "AIWILD": ("agents in the wild", "safety, security"),
    "AI_and_PDE": ("ai&pde", "partial differential equations"),
    "CAO": ("catch, adapt, and operate",),
    "DATA-FM": ("data-fm", "data problems for foundation models"),
    "DeLTa": ("deep generative model", "theory, principle and efficacy"),
    "ES-Reasoning": ("efficient spatial reasoning",),
    "FM4Science": ("foundation models for science",),
    "GEM": ("generative and experimental platforms", "biomolecular design"),
    "GRaM": ("geometry-grounded representation",),
    "Gen2": ("generative ai in genomics", "gen^2"),
    "HCAIR": ("human cognition to ai reasoning",),
    "ICBINB": ("can't believe it's not better",),
    "LIT": ("latent", "implicit thinking"),
    "LLA": ("lifelong agents",),
    "LLM_Reasoning": ("logical reasoning of large language models",),
    "LMRL": ("learning meaningful representations of life",),
    "MALGAI": ("multi-agent learning",),
    "ML4RS_Main_Track": ("machine learning for remote sensing", "ml4rs"),
    "ML4RS_Tiny_Track": ("machine learning for remote sensing", "ml4rs"),
    "ML4RS_Tutorial_Track": ("machine learning for remote sensing", "ml4rs"),
    "MLGenX": ("machine learning for genomics explorations", "mlgenx"),
    "MM_Intelligence": ("multimodal intelligence",),
    "MemAgent": ("memory for llm-based agentic systems", "memagents"),
    "NFAM": ("new frontiers in associative memories",),
    "P-AGI": ("post-agi",),
    "RSI": ("recursive self-improvement",),
    "Re-Align": ("representational alignment",),
    "Re-Align/Challenge": ("representational alignment",),
    "Re-Align/Research": ("representational alignment",),
    "ReALM-GEN": ("realm-gen",),
    "Reliable_Autonomy": ("hallucinations to reliable autonomy",),
    "SPOT": ("scaling post-training", "spot"),
    "Sci4DL": ("scientific methods for understanding deep learning", "sci4dl"),
    "TSALM": ("time series in the age of large models",),
    "TTU_Main_Track": ("test-time updates", "ttu"),
    "TTU_Tiny_Track": ("test-time updates", "ttu"),
    "Trustworthy_AI": ("principled design for trustworthy ai",),
    "UCRL": ("unifying concept representation learning",),
    "VerifAI-2": ("verifai-2", "ai verification"),
    "World_Models": ("world models",),
}


@dataclass
class EventCard:
    event_id: str
    title: str
    href: str
    data_title: str = ""


@dataclass
class Link:
    url: str
    text: str
    tag: str


@dataclass
class FetchResult:
    url: str
    final_url: str
    ok: bool
    status_code: int | None = None
    content_type: str = ""
    text: str = ""
    body: bytes = b""
    error: str | None = None


@dataclass
class WorkshopResult:
    workshop: dict[str, Any]
    event: EventCard | None
    source_urls: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    slide_recovery_summaries: list[dict[str, Any]] = field(default_factory=list)
    access_mode: str = "public"
    access_attempts: list[str] = field(default_factory=lambda: ["public"])


class EventListParser(html.parser.HTMLParser):
    """Extract event cards from the ICLR workshop listing page."""

    def __init__(self) -> None:
        super().__init__()
        self.cards: list[EventCard] = []
        self._in_card = False
        self._card_depth = 0
        self._current: dict[str, str] = {}
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = attr.get("class", "")
        if tag == "div" and "event-card" in classes:
            self._in_card = True
            self._card_depth = 1
            self._current = {
                "event_id": attr.get("data-event-id", ""),
                "data_title": attr.get("data-event-title", ""),
                "href": "",
            }
            self._title_parts = []
            return
        if not self._in_card:
            return
        if tag == "div":
            self._card_depth += 1
        if tag == "h3" and "event-title" in classes:
            self._in_title = True
        if tag == "a" and self._in_title and attr.get("href"):
            self._current["href"] = attr["href"]

    def handle_data(self, data: str) -> None:
        if self._in_card and self._in_title and data.strip():
            self._title_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if self._in_card and tag == "h3":
            self._in_title = False
        if not (self._in_card and tag == "div"):
            return
        self._card_depth -= 1
        if self._card_depth != 0:
            return
        event_id = self._current.get("event_id", "").strip()
        title = html.unescape(" ".join(self._title_parts).strip())
        href = self._current.get("href", "").strip()
        if event_id and title and href:
            self.cards.append(
                EventCard(
                    event_id=event_id,
                    title=title,
                    href=href,
                    data_title=html.unescape(self._current.get("data_title", "")),
                )
            )
        self._in_card = False
        self._current = {}
        self._title_parts = []


class LinkParser(html.parser.HTMLParser):
    """Extract link, iframe, audio, and video source candidates from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[Link] = []
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        if tag == "a" and attr.get("href"):
            self._anchor_href = attr["href"]
            self._anchor_text = []
            return
        if tag in {"iframe", "audio", "video", "source"}:
            url = attr.get("src") or attr.get("data-src") or ""
            if url:
                self.links.append(Link(url=html.unescape(url), text="", tag=tag))

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor_href is not None:
            text = html.unescape(" ".join(part.strip() for part in self._anchor_text).strip())
            self.links.append(Link(url=html.unescape(self._anchor_href), text=text, tag="a"))
            self._anchor_href = None
            self._anchor_text = []


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_relpath(workspace: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"malformed json: {exc}"
    if not isinstance(payload, dict):
        return None, "json root is not an object"
    return payload, None


def normalize_text(value: str) -> str:
    normalized = html.unescape(value).lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug or "workshop"


def venue_slug(venueid: str) -> str:
    marker = "/Workshop/"
    if marker in venueid:
        return venueid.split(marker, 1)[1]
    return venueid.rsplit("/", 1)[-1]


def workshop_title_from_venue(venueid: str) -> str:
    return venue_slug(venueid).replace("_", " ").replace("-", " / ")


def load_cookie_file(path: Path | None) -> tuple[str | None, dict[str, object]]:
    status: dict[str, object] = {
        "attempted": path is not None,
        "exists": False,
        "loaded": False,
        "cookie_names": [],
        "secret_values_recorded": False,
    }
    if path is None:
        return None, status
    status["path"] = str(path)
    if not path.exists():
        return None, status
    status["exists"] = True
    cookies: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("cookie:"):
            line = line.split(":", 1)[1].strip()
        if "\t" in line:
            fields = line.split("\t")
            if len(fields) >= 7:
                name = fields[-2].strip()
                value = fields[-1].strip()
                if name and value:
                    cookies.append((name, value))
                continue
        for part in line.split(";"):
            item = part.strip()
            if not item or "=" not in item:
                continue
            name, value = item.split("=", 1)
            name = name.strip()
            value = value.strip()
            if name and value:
                cookies.append((name, value))
    deduped: dict[str, str] = {}
    for name, value in cookies:
        deduped[name] = value
    if not deduped:
        return None, status
    status["loaded"] = True
    status["cookie_names"] = sorted(deduped)
    return "; ".join(f"{name}={value}" for name, value in sorted(deduped.items())), status


def cookie_headers(url: str, cookie_header: str | None) -> dict[str, str]:
    if not cookie_header:
        return {}
    host = (urlsplit(url).hostname or "").lower()
    if (
        host == "iclr.cc"
        or host.endswith(".iclr.cc")
        or host == "icml.cc"
        or host.endswith(".icml.cc")
        or host in {"127.0.0.1", "localhost", "::1"}
    ):
        return {"Cookie": cookie_header}
    return {}


def request_url(
    url: str,
    *,
    timeout_seconds: float,
    binary: bool = False,
    cookie_header: str | None = None,
) -> FetchResult:
    headers = {"User-Agent": HTTP_USER_AGENT, **cookie_headers(url, cookie_header)}
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - controlled public crawl
            body = response.read()
            final_url = response.geturl()
            content_type = response.headers.get("content-type", "")
            if binary:
                return FetchResult(
                    url=url,
                    final_url=final_url,
                    ok=True,
                    status_code=response.status,
                    content_type=content_type,
                    body=body,
                )
            charset = response.headers.get_content_charset() or "utf-8"
            text = body.decode(charset, errors="replace")
            return FetchResult(
                url=url,
                final_url=final_url,
                ok=True,
                status_code=response.status,
                content_type=content_type,
                text=text,
                body=body,
            )
    except HTTPError as exc:
        return FetchResult(
            url=url,
            final_url=url,
            ok=False,
            status_code=exc.code,
            error=f"http {exc.code}",
        )
    except (URLError, TimeoutError, OSError) as exc:
        return FetchResult(url=url, final_url=url, ok=False, error=str(exc))


def load_workshop_list_html(
    workspace: Path,
    *,
    workshop_list_url: str,
    timeout_seconds: float,
    cookie_header: str | None,
) -> tuple[FetchResult, bool]:
    cached = workspace / ICLR_CACHED_WORKSHOP_LIST
    if workshop_list_url == DEFAULT_WORKSHOP_LIST_URL and cached.exists():
        text = cached.read_text(encoding="utf-8", errors="replace")
        return (
            FetchResult(
                url=workshop_list_url,
                final_url=workshop_list_url,
                ok=True,
                status_code=200,
                content_type="text/html; source=cached-phase4-page",
                text=text,
                body=text.encode("utf-8"),
            ),
            True,
        )
    return request_url(
        workshop_list_url,
        timeout_seconds=timeout_seconds,
        cookie_header=cookie_header,
    ), False


def parse_events(text: str) -> list[EventCard]:
    parser = EventListParser()
    parser.feed(text)
    return parser.cards


def parse_links(text: str) -> list[Link]:
    parser = LinkParser()
    parser.feed(text)
    return parser.links


def score_event_for_workshop(event: EventCard, venueid: str) -> int:
    slug = venue_slug(venueid)
    normalized_slug = normalize_text(slug)
    title = normalize_text(f"{event.title} {event.data_title}")
    score = 0
    if normalized_slug and normalized_slug in title:
        score += 100
    for part in normalized_slug.split():
        if len(part) >= 3 and part in title:
            score += 5
    for hint in WORKSHOP_TITLE_HINTS.get(slug, ()):
        normalized_hint = normalize_text(hint)
        if normalized_hint and normalized_hint in title:
            score += 80
    return score


def match_event(events: list[EventCard], venueid: str) -> EventCard | None:
    scored = sorted(
        ((score_event_for_workshop(event, venueid), event) for event in events),
        key=lambda item: item[0],
        reverse=True,
    )
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return None


def is_direct_media_url(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(path.endswith(ext) for ext in DIRECT_MEDIA_EXTENSIONS)


def is_unsupported_embed(url: str) -> bool:
    return is_embedded_media_host(url)


def is_project_link(link: Link) -> bool:
    text = normalize_text(link.text)
    url = link.url.lower()
    if is_direct_media_url(link.url) or is_unsupported_embed(link.url):
        return False
    return any(hint in text for hint in PROJECT_LINK_HINTS) or "sites.google.com" in url


def has_auth_wall(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in AUTH_GATE_PHRASES)


def make_talk_id(workshop_id: str, index: int, media_url: str) -> str:
    digest = hashlib.sha1(f"{workshop_id}:{index}:{media_url}".encode("utf-8")).hexdigest()[:10]
    return f"talk_media_{index:03d}_{digest}"


def media_filename(media_url: str, index: int, content_type: str) -> str:
    path_name = Path(unquote(urlsplit(media_url).path)).name
    if path_name and "." in path_name:
        return safe_slug(path_name)
    extension = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) or ".bin"
    return f"media-{index:03d}{extension}"


def empty_media_fields(
    *,
    workshop_id: str,
    workshop_title: str,
    source_url: str | None,
    media_url: str | None,
    status: str,
    media_kind: str | None = None,
    talk_id: str | None = None,
    talk_title: str | None = None,
    talk_speaker: str | None = None,
) -> dict[str, Any]:
    return {
        "media_kind": media_kind,
        "workshop_id": workshop_id,
        "workshop_title": workshop_title,
        "talk_id": talk_id,
        "talk_title": talk_title,
        "talk_speaker": talk_speaker,
        "source_url": source_url,
        "media_url": media_url,
        "local_path": None,
        "sha256": None,
        "byte_count": None,
        "status": status,
    }


def make_blocker(
    *,
    workshop_id: str,
    workshop_title: str,
    code: str,
    reason: str,
    source_url: str | None = None,
    media_url: str | None = None,
    media_kind: str | None = None,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_media_url = sanitize_media_url(media_url) if media_url else None
    blocker = {
        "workshop_id": workshop_id,
        "workshop_title": workshop_title,
        "code": code,
        "reason": reason,
        "source_url": source_url,
        "media_url": safe_media_url,
        "media_kind": media_kind,
        "retryable": retryable,
    }
    if media_url:
        blocker["download_url_sha256"] = sha256_bytes(media_url.encode("utf-8"))
    if details:
        blocker["details"] = details
    return blocker


def blocker_record(blocker: dict[str, Any]) -> dict[str, Any]:
    record = empty_media_fields(
        workshop_id=str(blocker["workshop_id"]),
        workshop_title=str(blocker["workshop_title"]),
        source_url=blocker.get("source_url") if isinstance(blocker.get("source_url"), str) else None,
        media_url=blocker.get("media_url") if isinstance(blocker.get("media_url"), str) else None,
        status="blocked",
        media_kind=str(blocker.get("media_kind") or "blocked"),
    )
    record["blocker_code"] = blocker["code"]
    record["reason"] = blocker["reason"]
    if isinstance(blocker.get("download_url_sha256"), str):
        record["download_url_sha256"] = blocker["download_url_sha256"]
    return record


def media_succeeded(result: WorkshopResult) -> bool:
    return any(record.get("status") == "succeeded" for record in result.records)


def speech_media_succeeded(result: WorkshopResult) -> bool:
    return any(
        record.get("status") == "succeeded"
        and record.get("media_kind") in {"direct_audio_video", "embedded_audio"}
        for record in result.records
    )


def should_retry_with_auth(result: WorkshopResult) -> bool:
    if speech_media_succeeded(result):
        return False
    if media_succeeded(result):
        return True
    retryable_codes = {
        "auth_wall",
        "inaccessible_agenda",
        "no_media_links",
        "unsupported_embed",
        "download_error",
    }
    return any(blocker.get("code") in retryable_codes for blocker in result.blockers)


def normalize_embedded_media_hosts(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    hosts: list[str] = []
    seen: set[str] = set()
    for value in values:
        host = value.strip().lower()
        if not host:
            continue
        if "://" in host:
            host = urlsplit(host).hostname or ""
        host = host.rstrip(".")
        if host.startswith("www."):
            host = host[4:]
        if not host or host in seen:
            continue
        seen.add(host)
        hosts.append(host)
    return tuple(hosts)


def embedded_media_host_allowed(url: str, allowed_hosts: tuple[str, ...]) -> bool:
    if not allowed_hosts:
        return False
    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if host.startswith("www."):
        host = host[4:]
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in allowed_hosts)


def set_access_metadata(result: WorkshopResult, *, access_mode: str, access_attempts: list[str]) -> WorkshopResult:
    result.access_mode = access_mode
    result.access_attempts = access_attempts
    for record in result.records:
        record["access_mode"] = access_mode
        record["access_attempts"] = access_attempts
    for blocker in result.blockers:
        blocker["access_mode"] = access_mode
        blocker["access_attempts"] = access_attempts
    return result


def download_media(
    *,
    workspace: Path,
    workshop_id: str,
    media_url: str,
    source_url: str,
    link_text: str,
    index: int,
    timeout_seconds: float,
    resume: bool,
    cookie_header: str | None,
) -> dict[str, Any]:
    workshop_slug = safe_slug(venue_slug(workshop_id))
    output_dir = workspace / TALK_MEDIA_DIR / workshop_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    guessed_name = media_filename(media_url, index, "")
    output_path = output_dir / guessed_name
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        data = output_path.read_bytes()
        content_type = mimetypes.guess_type(output_path.name)[0] or "application/octet-stream"
    else:
        fetched = request_url(
            media_url,
            timeout_seconds=timeout_seconds,
            binary=True,
            cookie_header=cookie_header,
        )
        if not fetched.ok:
            raise RuntimeError(fetched.error or "download failed")
        content_type = fetched.content_type
        if content_type.lower().startswith("text/html"):
            raise RuntimeError("media URL returned HTML, not downloadable media")
        data = fetched.body
        output_path = output_dir / media_filename(media_url, index, content_type)
        output_path.write_bytes(data)

    talk_id = make_talk_id(workshop_id, index, media_url)
    return {
        "media_kind": "direct_audio_video",
        "workshop_id": workshop_id,
        "workshop_title": "",
        "talk_id": talk_id,
        "talk_title": link_text.strip() or Path(unquote(urlsplit(media_url).path)).name or talk_id,
        "talk_speaker": None,
        "source_url": source_url,
        "media_url": sanitize_media_url(media_url),
        "download_url_sha256": sha256_bytes(media_url.encode("utf-8")),
        "local_path": workspace_relpath(workspace, output_path),
        "sha256": sha256_bytes(data),
        "byte_count": len(data),
        "content_type": content_type,
        "status": "succeeded",
    }


def download_embedded_audio(
    *,
    workspace: Path,
    workshop_id: str,
    media_url: str,
    source_url: str,
    link_text: str,
    index: int,
    resume: bool,
    downloader: YtDlpDownloader,
) -> dict[str, Any]:
    workshop_slug = safe_slug(venue_slug(workshop_id))
    output_dir = workspace / TALK_MEDIA_DIR / workshop_slug / "embedded-audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    talk_id = make_talk_id(workshop_id, index, media_url)
    dest = output_dir / f"{talk_id}-raw"

    if resume:
        existing = existing_completed_audio(dest)
        if existing is not None:
            audio_path = existing
        else:
            audio_path = downloader.download_audio(media_url, dest)
    else:
        audio_path = downloader.download_audio(media_url, dest)

    return {
        "media_kind": "embedded_audio",
        "workshop_id": workshop_id,
        "workshop_title": "",
        "talk_id": talk_id,
        "talk_title": link_text.strip() or "Embedded workshop media",
        "talk_speaker": None,
        "source_url": source_url,
        "media_url": sanitize_media_url(media_url),
        "download_url_sha256": sha256_bytes(media_url.encode("utf-8")),
        "local_path": workspace_relpath(workspace, audio_path),
        "sha256": sha256_file(audio_path),
        "byte_count": audio_path.stat().st_size,
        "content_type": mimetypes.guess_type(audio_path.name)[0] or "audio/mpeg",
        "status": "succeeded",
        "retrieval_status": "embedded_audio_recovered",
    }


def slideslive_id_from_url(url: str) -> str | None:
    match = SLIDESLIVE_URL_ID_RE.search(url)
    return match.group("id") if match else None


def workshop_slide_asset(
    *,
    workshop_id: str,
    source_url: str,
    media_url: str,
    index: int,
) -> SlidesLivePageAsset | None:
    slideslive_id = slideslive_id_from_url(media_url)
    if not slideslive_id:
        return None
    workshop_slug = safe_slug(venue_slug(workshop_id))
    return SlidesLivePageAsset(
        item_id=f"workshop-{workshop_slug}-{slideslive_id}",
        kind="workshop",
        page_url=sanitize_slideslive_url(source_url),
        page_artifact_path=sanitize_slideslive_url(source_url),
        source_family_id=f"workshop-talk-media:{workshop_slug}",
        slideslive_id=slideslive_id,
        slideslive_url=f"https://slideslive.com/{slideslive_id}",
        explicit_no_slideslive_embed=False,
    )


def workshop_slide_inventory_record(
    *,
    workspace: Path,
    workshop_id: str,
    workshop_title: str,
    source_url: str,
    link_text: str,
    media_url: str,
    artifact: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    slideslive_id = str(artifact.get("slideslive_id") or slideslive_id_from_url(media_url) or "")
    slide_index = artifact.get("slide_index")
    talk_id = make_talk_id(workshop_id, index, media_url)
    title = link_text.strip() or "SlidesLive workshop slides"
    return {
        "media_kind": "slideslive_slide_image",
        "workshop_id": workshop_id,
        "workshop_title": workshop_title,
        "talk_id": talk_id,
        "talk_title": title,
        "talk_speaker": None,
        "source_url": sanitize_slideslive_url(source_url),
        "media_url": sanitize_media_url(media_url),
        "download_url_sha256": sha256_bytes(media_url.encode("utf-8")),
        "local_path": artifact.get("local_path"),
        "sha256": artifact.get("sha256"),
        "byte_count": artifact.get("byte_count"),
        "content_type": artifact.get("content_type"),
        "status": "succeeded",
        "retrieval_status": artifact.get("retrieval_status", "slideslive_slide_image_recovered"),
        "slideslive_id": slideslive_id,
        "slide_index": slide_index,
        "slide_image_url": artifact.get("source_url"),
        "source_artifact_path": artifact.get("source_artifact_path"),
        "item_id": artifact.get("item_id"),
        "item_kind": artifact.get("item_kind"),
        "secret_values_recorded": False,
    }


def enrich_workshop_slide_artifact(
    *,
    workshop_id: str,
    workshop_title: str,
    source_url: str,
    link_text: str,
    media_url: str,
    artifact: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    enriched = dict(artifact)
    enriched.update(
        {
            "workshop_id": workshop_id,
            "workshop_title": workshop_title,
            "talk_id": make_talk_id(workshop_id, index, media_url),
            "talk_title": link_text.strip() or "SlidesLive workshop slides",
            "talk_speaker": None,
            "workshop_source_url": sanitize_slideslive_url(source_url),
            "media_url": sanitize_media_url(media_url),
            "download_url_sha256": sha256_bytes(media_url.encode("utf-8")),
            "secret_values_recorded": False,
        }
    )
    return enriched


def merge_workshop_slide_artifacts_into_download_corpus(
    *,
    workspace: Path,
    run_id: str,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    path = workspace / DOWNLOAD_CORPUS_INVENTORY
    if not artifacts:
        return {
            "attempted": False,
            "download_corpus_path": str(DOWNLOAD_CORPUS_INVENTORY),
            "merged_slide_image_count": 0,
        }
    if not path.exists():
        return {
            "attempted": True,
            "status": "blocked",
            "download_corpus_path": str(DOWNLOAD_CORPUS_INVENTORY),
            "reason": "download corpus inventory is missing; workshop slide artifacts could not be merged for Phase 5 OCR",
            "merged_slide_image_count": 0,
        }
    try:
        download_corpus = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "attempted": True,
            "status": "blocked",
            "download_corpus_path": str(DOWNLOAD_CORPUS_INVENTORY),
            "reason": f"download corpus inventory is malformed: {exc}",
            "merged_slide_image_count": 0,
        }
    if not isinstance(download_corpus, dict):
        return {
            "attempted": True,
            "status": "blocked",
            "download_corpus_path": str(DOWNLOAD_CORPUS_INVENTORY),
            "reason": "download corpus inventory root is not an object",
            "merged_slide_image_count": 0,
        }
    sha_before = sha256_file(path)
    merge_media_artifacts(download_corpus, artifacts)
    download_corpus["workshop_slideslive_slide_recovery"] = sanitize_for_receipt(
        {
            "run_id": run_id,
            "status": "succeeded",
            "updated_at": utc_now(),
            "merged_slide_image_count": len(artifacts),
            "workshop_bound_slide_images": True,
        }
    )
    download_corpus["updated_at"] = utc_now()
    atomic_write_json(path, sanitize_for_receipt(download_corpus))
    return {
        "attempted": True,
        "status": "succeeded",
        "download_corpus_path": str(DOWNLOAD_CORPUS_INVENTORY),
        "download_corpus_sha256_before": sha_before,
        "download_corpus_sha256_after": sha256_file(path),
        "merged_slide_image_count": len(artifacts),
    }


def recover_workshop_slide_images(
    *,
    workspace: Path,
    run_id: str,
    workshop_id: str,
    workshop_title: str,
    embedded_candidates: list[tuple[str, Link]],
    resume: bool,
    worker_count: int,
    height: int,
    image_format: str,
    use_ytdlp_fallback: bool,
    download_corpus_merge_lock: Lock | None = None,
    recoverer: Callable[..., dict[str, Any]] = recover_slide_assets,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    assets: list[SlidesLivePageAsset] = []
    metadata_by_slideslive_id: dict[str, tuple[str, Link, int]] = {}
    seen_ids: set[str] = set()
    for index, (source_url, link) in enumerate(embedded_candidates, start=1):
        slideslive_id = slideslive_id_from_url(link.url)
        if not slideslive_id or slideslive_id in seen_ids:
            continue
        asset = workshop_slide_asset(
            workshop_id=workshop_id,
            source_url=source_url,
            media_url=link.url,
            index=index,
        )
        if asset is None:
            continue
        assets.append(asset)
        metadata_by_slideslive_id[slideslive_id] = (source_url, link, index)
        seen_ids.add(slideslive_id)

    if not assets:
        return [], [], {"attempted": False, "slideslive_embed_count": 0}, []

    summary = recoverer(
        workspace=workspace,
        assets=assets,
        run_id=run_id,
        resume=resume,
        worker_count=worker_count,
        height=height,
        image_format=image_format,
        use_ytdlp_fallback=use_ytdlp_fallback,
    )
    inventory_records: list[dict[str, Any]] = []
    corpus_artifacts: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for artifact in summary.get("media_records", []):
        if not isinstance(artifact, dict):
            continue
        slideslive_id = str(artifact.get("slideslive_id") or "")
        metadata = metadata_by_slideslive_id.get(slideslive_id)
        if metadata is None:
            continue
        source_url, link, index = metadata
        corpus_artifacts.append(
            enrich_workshop_slide_artifact(
                workshop_id=workshop_id,
                workshop_title=workshop_title,
                source_url=source_url,
                link_text=link.text,
                media_url=link.url,
                artifact=artifact,
                index=index,
            )
        )
        inventory_records.append(
            workshop_slide_inventory_record(
                workspace=workspace,
                workshop_id=workshop_id,
                workshop_title=workshop_title,
                source_url=source_url,
                link_text=link.text,
                media_url=link.url,
                artifact=corpus_artifacts[-1],
                index=index,
            )
        )

    for blocker in summary.get("blockers", []):
        if not isinstance(blocker, dict):
            continue
        slideslive_id = str(blocker.get("slideslive_id") or "")
        metadata = metadata_by_slideslive_id.get(slideslive_id)
        source_url, link, _index = metadata if metadata else (None, None, None)
        blockers.append(
            make_blocker(
                workshop_id=workshop_id,
                workshop_title=workshop_title,
                code=str(blocker.get("code") or "slideslive_slide_download_failed"),
                reason=str(blocker.get("reason") or "SlidesLive workshop slide recovery failed"),
                source_url=source_url,
                media_url=link.url if link else None,
                media_kind="slideslive_slide_image",
                retryable=bool(blocker.get("retryable", True)),
                details={"slideslive_id": slideslive_id} if slideslive_id else None,
            )
        )

    if download_corpus_merge_lock is None:
        merge_summary = merge_workshop_slide_artifacts_into_download_corpus(
            workspace=workspace,
            run_id=run_id,
            artifacts=corpus_artifacts,
        )
    else:
        with download_corpus_merge_lock:
            merge_summary = merge_workshop_slide_artifacts_into_download_corpus(
                workspace=workspace,
                run_id=run_id,
                artifacts=corpus_artifacts,
            )
    if merge_summary.get("status") == "blocked":
        blockers.append(
            make_blocker(
                workshop_id=workshop_id,
                workshop_title=workshop_title,
                code="workshop_slide_download_corpus_merge_failed",
                reason=str(merge_summary.get("reason") or "Workshop slide artifacts could not be merged"),
                source_url=None,
                media_url=None,
                media_kind="slideslive_slide_image",
                retryable=True,
                details=merge_summary,
            )
        )
    summary["download_corpus_merge"] = merge_summary
    return inventory_records, blockers, summary, corpus_artifacts


def unique_links(links: list[Link], *, base_url: str) -> list[Link]:
    seen: set[str] = set()
    unique: list[Link] = []
    for link in links:
        absolute = urljoin(base_url, link.url)
        if absolute in seen:
            continue
        seen.add(absolute)
        unique.append(Link(url=absolute, text=link.text, tag=link.tag))
    return unique


def crawl_workshop(
    *,
    workspace: Path,
    workshop: dict[str, Any],
    event: EventCard | None,
    workshop_list_url: str,
    timeout_seconds: float,
    resume: bool,
    cookie_header: str | None,
    embedded_downloader: YtDlpDownloader | None,
    embedded_media_hosts: tuple[str, ...],
    recover_embedded_slides: bool,
    slide_worker_count: int,
    slide_height: int,
    slide_image_format: str,
    slide_use_ytdlp_fallback: bool,
    download_corpus_merge_lock: Lock | None,
    run_id: str,
    access_mode: str = "public",
) -> WorkshopResult:
    venueid = str(workshop.get("venueid") or "")
    fallback_title = workshop_title_from_venue(venueid)
    if event is None:
        blocker = make_blocker(
            workshop_id=venueid,
            workshop_title=fallback_title,
            code="no_site_url",
            reason="No matching public ICLR workshop detail page was found for this workshop inventory record.",
            source_url=None,
            retryable=False,
        )
        return set_access_metadata(
            WorkshopResult(workshop=workshop, event=None, blockers=[blocker], records=[blocker_record(blocker)]),
            access_mode=access_mode,
            access_attempts=[access_mode],
        )

    detail_url = urljoin(workshop_list_url, event.href)
    result = WorkshopResult(
        workshop=workshop,
        event=event,
        source_urls=[detail_url],
        access_mode=access_mode,
        access_attempts=[access_mode],
    )
    detail = request_url(detail_url, timeout_seconds=timeout_seconds, cookie_header=cookie_header)
    workshop_title = event.title or fallback_title
    if not detail.ok:
        blocker = make_blocker(
            workshop_id=venueid,
            workshop_title=workshop_title,
            code="inaccessible_agenda",
            reason=f"Workshop detail page could not be fetched: {detail.error or detail.status_code}",
            source_url=detail_url,
            retryable=True,
        )
        result.blockers.append(blocker)
        result.records.append(blocker_record(blocker))
        return set_access_metadata(result, access_mode=access_mode, access_attempts=[access_mode])

    page_links = unique_links(parse_links(detail.text), base_url=detail.final_url)
    pages_to_scan: list[tuple[str, str, list[Link]]] = [(detail.final_url, detail.text, page_links)]
    if has_auth_wall(detail.text):
        result.blockers.append(
            make_blocker(
                workshop_id=venueid,
                workshop_title=workshop_title,
                code="auth_wall",
                reason="ICLR detail page reports that live content requires login and registration.",
                source_url=detail.final_url,
                retryable=False,
            )
        )

    for project_link in [link for link in page_links if is_project_link(link)][:2]:
        project_url = urljoin(detail.final_url, project_link.url)
        if project_url in result.source_urls:
            continue
        result.source_urls.append(project_url)
        project_page = request_url(
            project_url,
            timeout_seconds=timeout_seconds,
            cookie_header=cookie_header,
        )
        if not project_page.ok:
            result.blockers.append(
                make_blocker(
                    workshop_id=venueid,
                    workshop_title=workshop_title,
                    code="inaccessible_agenda",
                    reason=f"Workshop project/agenda page could not be fetched: {project_page.error or project_page.status_code}",
                    source_url=project_url,
                    retryable=True,
                )
            )
            continue
        pages_to_scan.append(
            (
                project_page.final_url,
                project_page.text,
                unique_links(parse_links(project_page.text), base_url=project_page.final_url),
            )
        )

    direct_candidates: list[tuple[str, Link]] = []
    embedded_candidates: list[tuple[str, Link]] = []
    for source_url, page_text, links in pages_to_scan:
        for link in links:
            if is_direct_media_url(link.url):
                direct_candidates.append((source_url, link))
            elif is_recoverable_embedded_media_url(link.url) and embedded_media_host_allowed(
                link.url, embedded_media_hosts
            ):
                embedded_candidates.append((source_url, link))
        for link in embedded_media_links(
            page_text,
            base_url=source_url,
            direct_media_predicate=is_direct_media_url,
        ):
            if is_direct_media_url(link.url):
                direct_candidates.append((source_url, link))
            elif is_recoverable_embedded_media_url(link.url) and embedded_media_host_allowed(
                link.url, embedded_media_hosts
            ):
                embedded_candidates.append((source_url, link))

    seen_media: set[str] = set()
    for source_url, link in direct_candidates:
        if link.url in seen_media:
            continue
        seen_media.add(link.url)
        try:
            record = download_media(
                workspace=workspace,
                workshop_id=venueid,
                media_url=link.url,
                source_url=source_url,
                link_text=link.text,
                index=len(result.records) + 1,
                timeout_seconds=timeout_seconds,
                resume=resume,
                cookie_header=cookie_header,
            )
            record["workshop_title"] = workshop_title
            result.records.append(record)
        except RuntimeError as exc:
            blocker = make_blocker(
                workshop_id=venueid,
                workshop_title=workshop_title,
                code="download_error",
                reason=str(exc),
                source_url=source_url,
                media_url=link.url,
                retryable=True,
            )
            result.blockers.append(blocker)
            result.records.append(blocker_record(blocker))

    seen_embedded: set[str] = set()
    processed_embedded_candidates: list[tuple[str, Link]] = []
    for source_url, link in embedded_candidates:
        if link.url in seen_embedded or link.url in seen_media:
            continue
        seen_embedded.add(link.url)
        processed_embedded_candidates.append((source_url, link))
        if embedded_downloader is None:
            blocker = make_blocker(
                workshop_id=venueid,
                workshop_title=workshop_title,
                code="embedded_media_downloader_unavailable",
                reason="Workshop page exposes embedded media, but yt-dlp is unavailable for audio recovery.",
                source_url=source_url,
                media_url=link.url,
                retryable=True,
            )
            result.blockers.append(blocker)
            result.records.append(blocker_record(blocker))
            continue
        try:
            record = download_embedded_audio(
                workspace=workspace,
                workshop_id=venueid,
                media_url=link.url,
                source_url=source_url,
                link_text=link.text,
                index=len(result.records) + 1,
                resume=resume,
                downloader=embedded_downloader,
            )
            record["workshop_title"] = workshop_title
            result.records.append(record)
        except (DownloadError, RuntimeError) as exc:
            blocker = make_blocker(
                workshop_id=venueid,
                workshop_title=workshop_title,
                code="embedded_media_download_error",
                reason=str(exc)[:500],
                source_url=source_url,
                media_url=link.url,
                retryable=True,
            )
            result.blockers.append(blocker)
            result.records.append(blocker_record(blocker))

    if recover_embedded_slides and processed_embedded_candidates:
        slide_records, slide_blockers, slide_summary, _slide_artifacts = recover_workshop_slide_images(
            workspace=workspace,
            run_id=run_id,
            workshop_id=venueid,
            workshop_title=workshop_title,
            embedded_candidates=processed_embedded_candidates,
            resume=resume,
            worker_count=slide_worker_count,
            height=slide_height,
            image_format=slide_image_format,
            use_ytdlp_fallback=slide_use_ytdlp_fallback,
            download_corpus_merge_lock=download_corpus_merge_lock,
        )
        for record in slide_records:
            record["workshop_title"] = workshop_title
            result.records.append(record)
        for blocker in slide_blockers:
            result.blockers.append(blocker)
            result.records.append(blocker_record(blocker))
        if slide_summary.get("attempted") is not False:
            result.slide_recovery_summaries.append(slide_summary)

    if not any(record.get("status") == "succeeded" for record in result.records):
        if embedded_candidates and not any(
            blocker.get("code", "").startswith("embedded_media_") for blocker in result.blockers
        ):
            for source_url, link in embedded_candidates[:3]:
                result.blockers.append(
                    make_blocker(
                        workshop_id=venueid,
                        workshop_title=workshop_title,
                        code="unsupported_embed",
                        reason="Workshop page exposes embedded media that is not a direct downloadable audio/video URL.",
                        source_url=source_url,
                        media_url=link.url,
                        retryable=False,
                    )
                )
        if not result.blockers:
            result.blockers.append(
                make_blocker(
                    workshop_id=venueid,
                    workshop_title=workshop_title,
                    code="no_media_links",
                    reason="Workshop detail and project/agenda pages were accessible but contained no direct talk media links.",
                    source_url=detail.final_url,
                    retryable=False,
                )
            )
        elif all(blocker["code"] == "auth_wall" for blocker in result.blockers):
            result.blockers.append(
                make_blocker(
                    workshop_id=venueid,
                    workshop_title=workshop_title,
                    code="no_media_links",
                    reason="No direct talk media links were visible without authenticated live-content access.",
                    source_url=detail.final_url,
                    retryable=False,
                )
            )
        existing_blocker_records = {
            (record.get("blocker_code"), record.get("source_url"), record.get("media_url"))
            for record in result.records
            if record.get("status") == "blocked"
        }
        for blocker in result.blockers:
            key = (blocker.get("code"), blocker.get("source_url"), blocker.get("media_url"))
            if key not in existing_blocker_records:
                result.records.append(blocker_record(blocker))

    return set_access_metadata(result, access_mode=access_mode, access_attempts=[access_mode])


def crawl_workshop_with_auth_fallback(
    *,
    workspace: Path,
    workshop: dict[str, Any],
    event: EventCard | None,
    workshop_list_url: str,
    timeout_seconds: float,
    resume: bool,
    cookie_header: str | None,
    embedded_downloader: YtDlpDownloader | None,
    embedded_media_hosts: tuple[str, ...],
    recover_embedded_slides: bool,
    slide_worker_count: int,
    slide_height: int,
    slide_image_format: str,
    slide_use_ytdlp_fallback: bool,
    download_corpus_merge_lock: Lock | None,
    run_id: str,
) -> WorkshopResult:
    public_result = crawl_workshop(
        workspace=workspace,
        workshop=workshop,
        event=event,
        workshop_list_url=workshop_list_url,
        timeout_seconds=timeout_seconds,
        resume=resume,
        cookie_header=None,
        embedded_downloader=embedded_downloader,
        embedded_media_hosts=embedded_media_hosts,
        recover_embedded_slides=recover_embedded_slides,
        slide_worker_count=slide_worker_count,
        slide_height=slide_height,
        slide_image_format=slide_image_format,
        slide_use_ytdlp_fallback=slide_use_ytdlp_fallback,
        download_corpus_merge_lock=download_corpus_merge_lock,
        run_id=run_id,
        access_mode="public",
    )
    if not cookie_header or not should_retry_with_auth(public_result):
        return public_result
    authenticated_result = crawl_workshop(
        workspace=workspace,
        workshop=workshop,
        event=event,
        workshop_list_url=workshop_list_url,
        timeout_seconds=timeout_seconds,
        resume=resume,
        cookie_header=cookie_header,
        embedded_downloader=embedded_downloader,
        embedded_media_hosts=embedded_media_hosts,
        recover_embedded_slides=recover_embedded_slides,
        slide_worker_count=slide_worker_count,
        slide_height=slide_height,
        slide_image_format=slide_image_format,
        slide_use_ytdlp_fallback=slide_use_ytdlp_fallback,
        download_corpus_merge_lock=download_corpus_merge_lock,
        run_id=run_id,
        access_mode="authenticated",
    )
    return set_access_metadata(
        authenticated_result,
        access_mode="authenticated" if media_succeeded(authenticated_result) else "authenticated_retry",
        access_attempts=["public", "authenticated"],
    )


def guard_failure_payload(
    *,
    workspace: Path,
    run_id: str,
    started_at: str,
    blockers: list[dict[str, Any]],
    checks: dict[str, Any],
    cookie_file_status: dict[str, object],
    embedded_downloader_status: dict[str, object],
    embedded_slide_recovery_status: dict[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_workshop_talk_media_inventory",
        "workspace": str(workspace),
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": "blocked",
        "required_media_fields": list(MEDIA_FIELD_NAMES),
        "workshop_count": 0,
        "records": [],
        "workshops": [],
        "blockers": blockers,
        "iclr_cookie_file": cookie_file_status,
        "embedded_audio_recovery": embedded_downloader_status,
        "embedded_slide_recovery": embedded_slide_recovery_status,
    }
    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_workshop_talk_media_acquisition_receipt",
        "workspace": str(workspace),
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "blocked",
        "checks": {
            **checks,
            "iclr_cookie_file": cookie_file_status,
            "embedded_audio_recovery": embedded_downloader_status,
            "embedded_slide_recovery": embedded_slide_recovery_status,
        },
        "boundaries": boundary_payload(),
        "inventory_path": str(TALK_MEDIA_INVENTORY),
        "blockers": blockers,
    }
    return receipt, inventory


def boundary_payload() -> dict[str, Any]:
    return {
        "phase4_artifacts_redownloaded": False,
        "paper_pdfs_redownloaded": False,
        "workshop_pdfs_redownloaded": False,
        "slideslive_audio_redownloaded": False,
        "slideslive_slide_images_redownloaded": False,
        "workshop_slideslive_slide_images_recovered": False,
        "poster_images_redownloaded": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }


def build_inventory(
    *,
    workspace: Path,
    run_id: str,
    results: list[WorkshopResult],
    workshop_inventory_sha256: str,
    download_receipt_sha256: str,
    workshop_list_url: str,
    workshop_list_from_cache: bool,
    workshop_list_access_mode: str,
    deterministic_blocker_mode: bool,
    cookie_file_status: dict[str, object],
    embedded_downloader_status: dict[str, object],
    embedded_slide_recovery_status: dict[str, object],
) -> dict[str, Any]:
    flat_records: list[dict[str, Any]] = []
    flat_blockers: list[dict[str, Any]] = []
    workshops_payload: list[dict[str, Any]] = []
    for result in results:
        venueid = str(result.workshop.get("venueid") or "")
        title = result.event.title if result.event else workshop_title_from_venue(venueid)
        succeeded = [record for record in result.records if record.get("status") == "succeeded"]
        status = "succeeded" if succeeded else f"blocked_{result.blockers[0]['code']}"
        flat_records.extend(result.records)
        flat_blockers.extend(result.blockers)
        workshops_payload.append(
            {
                "workshop_id": venueid,
                "venueid": venueid,
                "workshop_slug": venue_slug(venueid),
                "workshop_title": title,
                "event_id": result.event.event_id if result.event else None,
                "source_urls": result.source_urls,
                "access_mode": result.access_mode,
                "access_attempts": result.access_attempts,
                "status": status,
                "media_record_count": len(succeeded),
                "slide_image_record_count": sum(
                    1 for record in succeeded if record.get("media_kind") == "slideslive_slide_image"
                ),
                "blocker_count": len(result.blockers),
                "records": result.records,
                "blockers": result.blockers,
                "slide_recovery_summaries": result.slide_recovery_summaries,
            }
        )
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_workshop_talk_media_inventory",
        "workspace": str(workspace),
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": "succeeded",
        "required_media_fields": list(MEDIA_FIELD_NAMES),
        "inputs": {
            "workshop_inventory_path": str(WORKSHOP_INVENTORY),
            "workshop_inventory_sha256": workshop_inventory_sha256,
            "workshop_download_receipt_path": str(WORKSHOP_DOWNLOAD_RECEIPT),
            "workshop_download_receipt_sha256": download_receipt_sha256,
            "workshop_list_url": workshop_list_url,
            "workshop_list_from_cache": workshop_list_from_cache,
            "workshop_list_access_mode": workshop_list_access_mode,
            "deterministic_blocker_mode": deterministic_blocker_mode,
            "iclr_cookie_file": cookie_file_status,
            "embedded_audio_recovery": embedded_downloader_status,
            "embedded_slide_recovery": embedded_slide_recovery_status,
        },
        "iclr_cookie_file": cookie_file_status,
        "embedded_audio_recovery": embedded_downloader_status,
        "embedded_slide_recovery": embedded_slide_recovery_status,
        "workshop_count": len(results),
        "media_record_count": sum(
            1 for record in flat_records if record.get("status") == "succeeded"
        ),
        "slide_image_record_count": sum(
            1
            for record in flat_records
            if record.get("status") == "succeeded"
            and record.get("media_kind") == "slideslive_slide_image"
        ),
        "blocked_workshop_count": sum(1 for item in workshops_payload if item["blockers"]),
        "blocker_count": len(flat_blockers),
        "records": flat_records,
        "workshops": workshops_payload,
        "blockers": flat_blockers,
    }


def build_receipt(
    *,
    workspace: Path,
    run_id: str,
    started_at: str,
    inventory: dict[str, Any],
    inventory_sha256: str,
    workshop_inventory_sha256: str,
    download_receipt_sha256: str,
    workshop_list_url: str,
    workshop_list_from_cache: bool,
    workshop_list_access_mode: str,
    deterministic_blocker_mode: bool,
    cookie_file_status: dict[str, object],
    embedded_downloader_status: dict[str, object],
    embedded_slide_recovery_status: dict[str, object],
    event_count: int,
) -> dict[str, Any]:
    workshop_count = int(inventory.get("workshop_count") or 0)
    media_record_count = int(inventory.get("media_record_count") or 0)
    slide_image_record_count = int(inventory.get("slide_image_record_count") or 0)
    blocked_workshop_count = int(inventory.get("blocked_workshop_count") or 0)
    blocker_count = int(inventory.get("blocker_count") or 0)
    covered_workshop_count = sum(
        1
        for workshop in inventory.get("workshops", [])
        if workshop.get("records") or workshop.get("blockers")
    )
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_workshop_talk_media_acquisition_receipt",
        "workspace": str(workspace),
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "succeeded" if covered_workshop_count == workshop_count else "blocked",
        "inventory_path": str(TALK_MEDIA_INVENTORY),
        "inventory_sha256": inventory_sha256,
        "checks": {
            "workshop_inventory_path": str(WORKSHOP_INVENTORY),
            "workshop_inventory_present": True,
            "workshop_inventory_sha256": workshop_inventory_sha256,
            "download_receipt_path": str(WORKSHOP_DOWNLOAD_RECEIPT),
            "download_receipt_present": True,
            "download_receipt_sha256": download_receipt_sha256,
            "workshop_list_url": workshop_list_url,
            "workshop_list_from_cache": workshop_list_from_cache,
            "workshop_list_access_mode": workshop_list_access_mode,
            "deterministic_blocker_mode": deterministic_blocker_mode,
            "live_detail_crawl_attempted": not deterministic_blocker_mode,
            "iclr_cookie_file": cookie_file_status,
            "embedded_audio_recovery": embedded_downloader_status,
            "embedded_slide_recovery": embedded_slide_recovery_status,
            "workshop_event_count": event_count,
            "workshop_count": workshop_count,
            "covered_workshop_count": covered_workshop_count,
            "media_record_count": media_record_count,
            "slide_image_record_count": slide_image_record_count,
            "blocked_workshop_count": blocked_workshop_count,
            "blocker_count": blocker_count,
            "all_workshops_have_media_or_blocker": covered_workshop_count == workshop_count,
            "required_media_fields": list(MEDIA_FIELD_NAMES),
        },
        "boundaries": {
            **boundary_payload(),
            "workshop_slideslive_slide_images_recovered": slide_image_record_count > 0,
        },
        "per_workshop": [
            {
                "workshop_id": workshop["workshop_id"],
                "workshop_title": workshop["workshop_title"],
                "status": workshop["status"],
                "media_record_count": workshop["media_record_count"],
                "slide_image_record_count": workshop.get("slide_image_record_count", 0),
                "blocker_count": workshop["blocker_count"],
            }
            for workshop in inventory.get("workshops", [])
        ],
        "blockers": inventory.get("blockers", []),
    }


def run_acquisition(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    workspace = args.workspace.resolve()
    run_id = args.run_id
    started_at = utc_now()
    cookie_header, cookie_file_status = load_cookie_file(args.iclr_cookie_file)
    embedded_downloader: YtDlpDownloader | None = None
    embedded_downloader_status: dict[str, object] = {
        "attempted": not args.skip_embedded_audio_recovery,
        "available": False,
        "workshop_worker_count": args.workshop_worker_count,
        "allowed_hosts": list(args.embedded_media_hosts),
    }
    embedded_slide_recovery_status: dict[str, object] = {
        "attempted": not args.skip_embedded_slide_recovery,
        "available": True,
        "worker_count": args.slide_worker_count,
        "height": args.slide_height,
        "image_format": args.slide_image_format,
        "use_ytdlp_fallback": not args.no_slide_ytdlp_fallback,
        "workshop_worker_count": args.workshop_worker_count,
        "allowed_hosts": list(args.embedded_media_hosts),
    }
    if not args.skip_embedded_audio_recovery:
        try:
            embedded_downloader = YtDlpDownloader(PipelineSettings())
            embedded_downloader_status["available"] = True
            embedded_downloader_status["command"] = embedded_downloader._yt_dlp_cmd[0]
        except RuntimeError as exc:
            embedded_downloader_status["error"] = str(exc)
    workshops_dir = workspace / WORKSHOPS_DIR
    workshops_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = workspace / WORKSHOP_INVENTORY
    download_receipt_path = workspace / WORKSHOP_DOWNLOAD_RECEIPT

    workshop_inventory, inventory_error = load_json(inventory_path)
    download_receipt, download_error = load_json(download_receipt_path)
    guard_blockers: list[dict[str, Any]] = []
    checks = {
        "workshop_inventory_path": str(WORKSHOP_INVENTORY),
        "workshop_inventory_present": inventory_path.exists(),
        "download_receipt_path": str(WORKSHOP_DOWNLOAD_RECEIPT),
        "download_receipt_present": download_receipt_path.exists(),
    }
    if inventory_error:
        guard_blockers.append(
            {
                "code": "missing_workshop_inventory",
                "reason": f"Workshop inventory could not be loaded: {inventory_error}",
                "retryable": inventory_error == "missing",
            }
        )
    if download_error:
        guard_blockers.append(
            {
                "code": "missing_workshop_download_receipt",
                "reason": f"Workshop download receipt could not be loaded: {download_error}",
                "retryable": download_error == "missing",
            }
        )
    if isinstance(download_receipt, dict) and download_receipt.get("status") not in {
        "succeeded",
        "succeeded_with_skips",
    }:
        guard_blockers.append(
            {
                "code": "workshop_download_receipt_not_succeeded",
                "reason": "Workshop talk media acquisition requires the workshop paper download receipt to be current and succeeded.",
                "retryable": True,
            }
        )
    if guard_blockers:
        receipt, inventory = guard_failure_payload(
            workspace=workspace,
            run_id=run_id,
            started_at=started_at,
            blockers=guard_blockers,
            checks=checks,
            cookie_file_status=cookie_file_status,
            embedded_downloader_status=embedded_downloader_status,
            embedded_slide_recovery_status=embedded_slide_recovery_status,
        )
        atomic_write_json(workspace / TALK_MEDIA_INVENTORY, inventory)
        atomic_write_json(workspace / TALK_MEDIA_RECEIPT, receipt)
        return 1, {
            "status": "blocked",
            "receipt_path": str(workspace / TALK_MEDIA_RECEIPT),
            "inventory_path": str(workspace / TALK_MEDIA_INVENTORY),
            "blockers": guard_blockers,
        }

    assert workshop_inventory is not None
    workshops = workshop_inventory.get("workshops")
    if not isinstance(workshops, list):
        blocker = {
            "code": "malformed_workshop_inventory",
            "reason": "Workshop inventory has no workshops array.",
            "retryable": True,
        }
        receipt, inventory = guard_failure_payload(
            workspace=workspace,
            run_id=run_id,
            started_at=started_at,
            blockers=[blocker],
            checks={**checks, "download_receipt_present": True},
            cookie_file_status=cookie_file_status,
            embedded_downloader_status=embedded_downloader_status,
            embedded_slide_recovery_status=embedded_slide_recovery_status,
        )
        atomic_write_json(workspace / TALK_MEDIA_INVENTORY, inventory)
        atomic_write_json(workspace / TALK_MEDIA_RECEIPT, receipt)
        return 1, {
            "status": "blocked",
            "receipt_path": str(workspace / TALK_MEDIA_RECEIPT),
            "inventory_path": str(workspace / TALK_MEDIA_INVENTORY),
            "blockers": [blocker],
        }

    list_fetch, list_from_cache = load_workshop_list_html(
        workspace,
        workshop_list_url=args.workshop_list_url,
        timeout_seconds=args.timeout_seconds,
        cookie_header=None,
    )
    list_access_mode = "public"
    if not list_fetch.ok and cookie_header and not list_from_cache:
        list_fetch, list_from_cache = load_workshop_list_html(
            workspace,
            workshop_list_url=args.workshop_list_url,
            timeout_seconds=args.timeout_seconds,
            cookie_header=cookie_header,
        )
        list_access_mode = "authenticated_retry"
    if not list_fetch.ok:
        events: list[EventCard] = []
    else:
        events = parse_events(list_fetch.text)
    deterministic_blocker_mode = (
        args.workshop_list_url == DEFAULT_WORKSHOP_LIST_URL
        and list_from_cache
        and not args.live_crawl
    )

    download_corpus_merge_lock = Lock()

    def process_workshop(workshop: dict[str, Any]) -> WorkshopResult | None:
        if not isinstance(workshop, dict):
            return None
        venueid = str(workshop.get("venueid") or "")
        event = match_event(events, venueid)
        if not list_fetch.ok:
            blocker = make_blocker(
                workshop_id=venueid,
                workshop_title=workshop_title_from_venue(venueid),
                code="inaccessible_agenda",
                reason=f"Workshop listing page could not be fetched: {list_fetch.error or list_fetch.status_code}",
                source_url=args.workshop_list_url,
                retryable=True,
            )
            return set_access_metadata(
                WorkshopResult(
                    workshop=workshop,
                    event=None,
                    source_urls=[args.workshop_list_url],
                    records=[blocker_record(blocker)],
                    blockers=[blocker],
                    access_mode="public",
                    access_attempts=["public"],
                ),
                access_mode="public",
                access_attempts=["public"],
            )
        if deterministic_blocker_mode:
            source_url = urljoin(args.workshop_list_url, event.href) if event else args.workshop_list_url
            blocker = make_blocker(
                workshop_id=venueid,
                workshop_title=event.title if event else workshop_title_from_venue(venueid),
                code="inaccessible_agenda" if event else "no_site_url",
                reason=DEFAULT_PRODUCTION_BLOCKER_REASON,
                source_url=source_url,
                retryable=True,
            )
            return set_access_metadata(
                WorkshopResult(
                    workshop=workshop,
                    event=event,
                    source_urls=[source_url],
                    records=[blocker_record(blocker)],
                    blockers=[blocker],
                    access_mode="deterministic_cached",
                    access_attempts=["deterministic_cached"],
                ),
                access_mode="deterministic_cached",
                access_attempts=["deterministic_cached"],
            )
        return crawl_workshop_with_auth_fallback(
            workspace=workspace,
            workshop=workshop,
            event=event,
            workshop_list_url=args.workshop_list_url,
            timeout_seconds=args.timeout_seconds,
            resume=args.resume,
            cookie_header=cookie_header,
            embedded_downloader=embedded_downloader,
            embedded_media_hosts=args.embedded_media_hosts,
            recover_embedded_slides=not args.skip_embedded_slide_recovery,
            slide_worker_count=args.slide_worker_count,
            slide_height=args.slide_height,
            slide_image_format=args.slide_image_format,
            slide_use_ytdlp_fallback=not args.no_slide_ytdlp_fallback,
            download_corpus_merge_lock=download_corpus_merge_lock,
            run_id=run_id,
        )

    results_by_index: dict[int, WorkshopResult] = {}
    valid_workshops = [
        (index, workshop)
        for index, workshop in enumerate(workshops)
        if isinstance(workshop, dict)
    ]
    if args.workshop_worker_count == 1:
        for index, workshop in valid_workshops:
            result = process_workshop(workshop)
            if result is not None:
                results_by_index[index] = result
            if args.request_delay_seconds > 0:
                time.sleep(args.request_delay_seconds)
    else:
        with ThreadPoolExecutor(max_workers=args.workshop_worker_count) as executor:
            futures = {}
            for index, workshop in valid_workshops:
                futures[executor.submit(process_workshop, workshop)] = index
                if args.request_delay_seconds > 0:
                    time.sleep(args.request_delay_seconds)
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results_by_index[futures[future]] = result

    results = [results_by_index[index] for index in sorted(results_by_index)]

    workshop_inventory_sha256 = sha256_file(inventory_path)
    download_receipt_sha256 = sha256_file(download_receipt_path)
    inventory = build_inventory(
        workspace=workspace,
        run_id=run_id,
        results=results,
        workshop_inventory_sha256=workshop_inventory_sha256,
        download_receipt_sha256=download_receipt_sha256,
        workshop_list_url=args.workshop_list_url,
        workshop_list_from_cache=list_from_cache,
        workshop_list_access_mode=list_access_mode,
        deterministic_blocker_mode=deterministic_blocker_mode,
        cookie_file_status=cookie_file_status,
        embedded_downloader_status=embedded_downloader_status,
        embedded_slide_recovery_status=embedded_slide_recovery_status,
    )
    atomic_write_json(workspace / TALK_MEDIA_INVENTORY, inventory)
    inventory_sha256 = sha256_file(workspace / TALK_MEDIA_INVENTORY)
    receipt = build_receipt(
        workspace=workspace,
        run_id=run_id,
        started_at=started_at,
        inventory=inventory,
        inventory_sha256=inventory_sha256,
        workshop_inventory_sha256=workshop_inventory_sha256,
        download_receipt_sha256=download_receipt_sha256,
        workshop_list_url=args.workshop_list_url,
        workshop_list_from_cache=list_from_cache,
        workshop_list_access_mode=list_access_mode,
        deterministic_blocker_mode=deterministic_blocker_mode,
        cookie_file_status=cookie_file_status,
        embedded_downloader_status=embedded_downloader_status,
        embedded_slide_recovery_status=embedded_slide_recovery_status,
        event_count=len(events),
    )
    atomic_write_json(workspace / TALK_MEDIA_RECEIPT, receipt)
    status = str(receipt["status"])
    return (0 if status == "succeeded" else 1), {
        "status": status,
        "receipt_path": str(workspace / TALK_MEDIA_RECEIPT),
        "inventory_path": str(workspace / TALK_MEDIA_INVENTORY),
        "workshop_count": receipt["checks"]["workshop_count"],
        "covered_workshop_count": receipt["checks"]["covered_workshop_count"],
        "media_record_count": receipt["checks"]["media_record_count"],
        "slide_image_record_count": receipt["checks"]["slide_image_record_count"],
        "blocked_workshop_count": receipt["checks"]["blocked_workshop_count"],
        "blocker_count": receipt["checks"]["blocker_count"],
        "iclr_cookie_file": cookie_file_status,
        "embedded_audio_recovery": embedded_downloader_status,
        "embedded_slide_recovery": embedded_slide_recovery_status,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--workshop-list-url", default=DEFAULT_WORKSHOP_LIST_URL)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--workshop-worker-count",
        type=int,
        default=1,
        help=(
            "Parallel workshop crawl workers. Values above 1 can run multiple "
            "yt-dlp audio recoveries at once; keep modest to avoid ICLR/SlidesLive throttling."
        ),
    )
    parser.add_argument(
        "--embedded-media-host",
        dest="embedded_media_hosts",
        action="append",
        default=None,
        help=(
            "Recover embedded audio/slides only from this host. Repeat to allow multiple "
            "hosts. ICLR 2026 defaults to SlidesLive-only; direct media files are unaffected."
        ),
    )
    parser.add_argument(
        "--iclr-cookie-file",
        type=Path,
        help=(
            "Optional local cookie file. The crawler tries public pages first, then "
            "uses this cookie for an authenticated retry when public access is blocked "
            "or media-less. Cookie values are never written to receipts."
        ),
    )
    parser.add_argument(
        "--skip-embedded-audio-recovery",
        action="store_true",
        help="Skip yt-dlp recovery for allowed embedded workshop media hosts.",
    )
    parser.add_argument(
        "--skip-embedded-slide-recovery",
        action="store_true",
        help="Skip SlidesLive slide-image recovery for embedded workshop presentations.",
    )
    parser.add_argument(
        "--slide-worker-count",
        type=int,
        default=2,
        help="Parallel SlidesLive slide-image downloads for workshop presentations.",
    )
    parser.add_argument("--slide-height", type=int, default=1080, help="SlidesLive slide resize height.")
    parser.add_argument(
        "--slide-image-format",
        choices=("jpeg", "webp"),
        default="jpeg",
        help="SlidesLive slide image format.",
    )
    parser.add_argument(
        "--no-slide-ytdlp-fallback",
        action="store_true",
        help="Disable yt-dlp thumbnail metadata fallback for workshop SlidesLive slides.",
    )
    parser.add_argument(
        "--live-crawl",
        action="store_true",
        help=(
            "Attempt live ICLR workshop detail/project page crawling. The default production "
            "path uses cached listing evidence and emits explicit blockers deterministically."
        ),
    )
    args = parser.parse_args(argv)
    args.embedded_media_hosts = normalize_embedded_media_hosts(
        args.embedded_media_hosts or list(DEFAULT_EMBEDDED_MEDIA_HOSTS)
    )
    if not args.embedded_media_hosts:
        parser.error("--embedded-media-host must leave at least one allowed host")
    if args.workshop_worker_count < 1:
        parser.error("--workshop-worker-count must be >= 1")
    if args.slide_worker_count < 1:
        parser.error("--slide-worker-count must be >= 1")
    if args.slide_height < 1:
        parser.error("--slide-height must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exit_code, summary = run_acquisition(args)
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif exit_code:
        print(
            f"ICLR 2026 workshop talk media acquisition blocked; receipt: {summary.get('receipt_path')}",
            file=sys.stderr,
        )
    else:
        print("ICLR 2026 workshop talk media acquisition succeeded")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
