"""Conference virtual-site crawl lanes for a frozen source-family corpus (Stage 4).

A conference virtual site (the ``/virtual/<year>/...`` platform shared by ICML,
ICLR and NeurIPS) publishes its program as family index pages (calendar,
posters, orals, invited talks, tutorials, ...) that link item pages, which in
turn link papers, posters and talk media. These lanes acquire that surface for
a frozen source-family corpus:

* :func:`source_page_lane` downloads every frozen family's source page and
  records the acquisition targets it links;
* :func:`direct_target_lane` downloads every directly downloadable target;
* :func:`item_page_expansion_lane` probes every discovered item page through a
  bounded :class:`RequestPacer`, retries auth-gated probes with a slower pacer
  and downloads the media/papers they expose;
* :func:`media_family_coverage_lane` (a ``gate`` lane) fails closed when a
  declared media-bearing program family yielded no item page.

:func:`download_http_artifact` is the shared HTTP downloader (resume reuse,
retry with backoff, auth-gate detection, no secret recorded).

Recommended for
---------------
``RECOMMENDED_FOR = ("ICML",)``.

* **ICML stage and quirk covered:** Stage 4 ``download-corpus``: the
  registered-session cookie for the ICML virtual site, the server-side
  burst throttle (``RequestPacer``) on item-page probes, the slow auth retry
  passes, and the fail-closed media-family coverage gate.
* **May also fit:** ICLR and NeurIPS (the same virtual-site platform and
  family layout); the year-pinned path patterns, cookie host and user agents
  come from :class:`CrawlPolicy`.

Conference identity (path patterns, virtual year, cookie host, user agents,
cache root, blocker wording) comes from :class:`CrawlPolicy`; this module
carries no conference literal.
"""

from __future__ import annotations

import hashlib
import html.parser
import mimetypes
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import Request

from conference_pipeline.indexing.corpus_vector_store import workspace_relpath
from conference_pipeline.production.download_corpus_stage import (
    PAPER_COUNT_FROM_DIRECT_ONLY,
    FrozenCorpusState,
    LaneContext,
)
from conference_pipeline.production.hooks import LANE_PHASE_GATE, AcquisitionLane, LaneResult
from conference_pipeline.program_coverage import (
    DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
    classify_item_page_title_family,
    classify_virtual_item_page,
    media_family_coverage,
    media_family_coverage_blockers,
)
from conference_pipeline.receipt_lineage import sha256_file

RECOMMENDED_FOR = ("ICML",)

DEFAULT_MEDIA_URL_PATTERNS = (
    "slideslive.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    ".mp4",
    ".mp3",
    ".m3u8",
    ".webm",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mov",
    ".avi",
    ".mkv",
    "/media/posterpdfs/",
)
DEFAULT_SOURCE_ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tgz", ".tar.gz", ".gz")
DEFAULT_DIRECT_DOWNLOAD_EXTENSIONS = (
    ".pdf",
    ".zip",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".gz",
    ".mp4",
    ".mp3",
    ".m3u8",
    ".webm",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mov",
    ".avi",
    ".mkv",
)
DEFAULT_AUTH_GATE_PHRASES = (
    "log in and register to view live content",
    "login and register to view live content",
    "please log in to view",
    "accounts/login",
)
SITE_COOKIE_CREDENTIAL = "site_cookie_header"


@dataclass(frozen=True)
class CrawlPolicy:
    """A conference virtual site's crawl policy.

    ``opener(request, timeout=)`` and ``now()`` are the orchestrator's HTTP
    opener and timestamp source, bound at call time; ``cookie_hosts`` are the
    hosts the operator's site session cookie is sent to.
    """

    virtual_year: int
    paper_page_patterns: tuple[str, ...]
    session_page_patterns: tuple[str, ...]
    cookie_hosts: tuple[str, ...]
    download_user_agent: str
    probe_user_agent: str
    cache_dir: Path
    opener: Callable[..., Any]
    now: Callable[[], str]
    login_required_code: str
    login_required_reason: str
    login_required_next_action: str
    media_url_patterns: tuple[str, ...] = DEFAULT_MEDIA_URL_PATTERNS
    source_archive_extensions: tuple[str, ...] = DEFAULT_SOURCE_ARCHIVE_EXTENSIONS
    direct_download_extensions: tuple[str, ...] = DEFAULT_DIRECT_DOWNLOAD_EXTENSIONS
    auth_gate_phrases: tuple[str, ...] = DEFAULT_AUTH_GATE_PHRASES
    media_scope_waiver: Path | None = None
    # ``pacer_factory(min_interval_seconds)`` builds the item-page request pacer
    # (the ``download-corpus.request_pacer`` hook).
    pacer_factory: Callable[[float], Any] | None = None

    def pacer(self, min_interval_seconds: float) -> Any:
        factory = self.pacer_factory or RequestPacer
        return factory(min_interval_seconds)


# --------------------------------------------------------------------------
# URL helpers
# --------------------------------------------------------------------------


def safe_slug(value: object, *, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip(".-")
    return text[:120] or fallback


def lower_url_path(url: str) -> str:
    return urlsplit(url).path.lower()


def url_has_extension(url: str, extensions: tuple[str, ...]) -> bool:
    path = lower_url_path(url)
    return any(path.endswith(extension) for extension in extensions)


def url_extension(url: str, *, default: str) -> str:
    path = lower_url_path(url)
    if path.endswith(".tar.gz"):
        return ".tar.gz"
    suffix = Path(path).suffix
    return suffix or default


def has_auth_gate(body: bytes | str, phrases: Sequence[str] = DEFAULT_AUTH_GATE_PHRASES) -> bool:
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


class RequestPacer:
    """Serialize live network requests to a bounded rate across worker threads.

    A bulk item-page probe against an authenticated virtual site showed an
    ~100% "log in and register to view live content" gate rate across 180
    newly-discovered pages, even though the same session succeeded on
    one-off manual requests immediately before and after. Cookie freshness,
    header shape, and worker-count (8 vs 2) were all ruled out; the signature
    matches server-side burst/velocity throttling rather than an auth
    failure. This pacer bounds the aggregate request rate across the whole
    thread pool (not per-worker), independent of the item-page worker count.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_allowed_at)
            self._next_allowed_at = start_at + self._min_interval
        delay = start_at - now
        if delay > 0:
            time.sleep(delay)


def load_cookie_file(path: Path | None) -> tuple[str | None, dict[str, object]]:
    """Read a Netscape/``Cookie:`` session file; return the header and secret-free status flags."""

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


def site_cookie_headers(url: str, cookie_header: str | None, hosts: Sequence[str]) -> dict[str, str] | None:
    """The session cookie header for ``url`` when it is on one of ``hosts`` (or a subdomain)."""

    if not cookie_header:
        return None
    host = urlsplit(url).netloc.lower()
    if any(host == allowed or host.endswith("." + allowed) for allowed in hosts):
        return {"Cookie": cookie_header}
    return None


class AcquisitionLinkParser(html.parser.HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.urls: list[dict[str, str]] = []

    def _add_url(self, tag: str, attr: str, value: str | None) -> None:
        if not value:
            return
        value = value.strip()
        if not value or value.startswith(("mailto:", "javascript:", "data:")):
            return
        self.urls.append({"url": urljoin(self.base_url, value), "tag": tag, "attribute": attr})

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs}
        for attr in ("href", "src"):
            if attr in attrs_dict:
                self._add_url(tag_name, attr, attrs_dict.get(attr))
        if tag_name == "meta":
            meta_name = str(attrs_dict.get("property") or attrs_dict.get("name") or "").lower()
            if meta_name in {"og:image", "twitter:image", "og:video", "twitter:player"}:
                self._add_url(tag_name, "content", attrs_dict.get("content"))


def extract_urls_from_html(base_url: str, body: bytes) -> list[dict[str, str]]:
    text = body.decode("utf-8", errors="replace")
    parser = AcquisitionLinkParser(base_url)
    try:
        parser.feed(text)
    except Exception:
        pass
    urls = list(parser.urls)
    for match in re.finditer(r"(?P<quote>['\"])(?P<url>/media/PosterPDFs/[^'\"]+)(?P=quote)", text):
        urls.append({"url": urljoin(base_url, match.group("url")), "tag": "html", "attribute": "media-reference"})
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in urls:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique


def classify_discovered_url(url: str, policy: CrawlPolicy) -> str | None:
    lowered = url.lower()
    path = lower_url_path(url)
    if "openreview.net/pdf" in lowered or url_has_extension(url, (".pdf",)):
        return "paper_pdf"
    if url_has_extension(url, policy.source_archive_extensions):
        return "source_archive"
    if any(pattern in lowered for pattern in policy.media_url_patterns):
        return "media_asset"
    if any(pattern in path for pattern in policy.paper_page_patterns):
        return "paper_page"
    if any(pattern in path for pattern in policy.session_page_patterns):
        return "session_page"
    # Default-include: any other virtual-site event item page (tutorials,
    # invited talks, sessions, expo, bare-ID links) is an acquisition target.
    # Family allowlists silently dropped every tutorial and keynote once.
    if classify_virtual_item_page(url, year=policy.virtual_year) is not None:
        return "paper_page"
    return None


def direct_download_kind(url: str, policy: CrawlPolicy) -> str | None:
    kind = classify_discovered_url(url, policy)
    if kind in {"paper_pdf", "source_archive", "media_asset"} and url_has_extension(url, policy.direct_download_extensions):
        return kind
    return None


def source_page_destination(workspace: Path, family_id: str, url: str, policy: CrawlPolicy) -> Path:
    extension = url_extension(url, default=".html")
    return workspace / policy.cache_dir / "source-pages" / f"{safe_slug(family_id)}{extension}"


def direct_artifact_destination(workspace: Path, family_id: str, kind: str, url: str, policy: CrawlPolicy) -> Path:
    split = urlsplit(url)
    extension = url_extension(url, default=".asset")
    stem = safe_slug(Path(split.path).stem or parse_qs(split.query).get("id", ["artifact"])[0])
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return workspace / policy.cache_dir / kind / safe_slug(family_id) / f"{stem}-{digest}{extension}"


def probe_page_destination(workspace: Path, url: str, policy: CrawlPolicy) -> Path:
    # Virtual sites canonicalize the host casing between source pages and
    # item-page links. Keep the cache identity stable across those equivalent
    # URLs so a resume does not re-fetch thousands of authenticated probes.
    cache_key = urlsplit(url)._replace(netloc=urlsplit(url).netloc.lower()).geturl()
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
    return workspace / policy.cache_dir / "item-page-probes" / f"{safe_slug(urlsplit(url).path)}-{digest}.html"


# --------------------------------------------------------------------------
# HTTP downloader
# --------------------------------------------------------------------------


def download_http_artifact(
    *,
    workspace: Path,
    source_family_id: str,
    artifact_kind: str,
    url: str,
    dest: Path,
    timeout_seconds: float,
    retry_count: int,
    resume: bool,
    policy: CrawlPolicy,
    user_agent: str | None = None,
    extra_headers: dict[str, str] | None = None,
    pacer: RequestPacer | None = None,
) -> dict[str, object]:
    """Download one artifact (resume reuse, bounded retry, auth-gate detection); record no secret."""

    now = policy.now
    started_at = now()
    headers = {"User-Agent": user_agent or policy.download_user_agent, **(extra_headers or {})}
    storage_lane = workspace_relpath(workspace, workspace / policy.cache_dir)
    if resume and dest.exists() and dest.stat().st_size > 0:
        body = dest.read_bytes() if dest.stat().st_size <= 5 * 1024 * 1024 else b""
        cached_access_gate = has_auth_gate(body, policy.auth_gate_phrases) if body else False
        if cached_access_gate and extra_headers and extra_headers.get("Cookie"):
            # A stale unauthenticated probe should not block a retry with the
            # operator's current local site session.
            pass
        else:
            return {
                "source_family_id": source_family_id,
                "artifact_kind": artifact_kind,
                "source_url": url,
                "final_url": url,
                "local_path": workspace_relpath(workspace, dest),
                "storage_lane": storage_lane,
                "retrieval_status": "resume_reused",
                "status": "succeeded",
                "http_status": None,
                "content_type": mimetypes.guess_type(dest.name)[0],
                "byte_count": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "retries_attempted": 0,
                "auth_required": False,
                "access_gate_detected": cached_access_gate,
                "started_at": started_at,
                "ended_at": now(),
                "secret_values_recorded": False,
            }

    last_error: str | None = None
    retryable_codes = {429, 500, 502, 503, 504}
    for attempt in range(1, max(1, retry_count) + 1):
        if pacer is not None:
            pacer.wait()
        request = Request(url, headers=headers)
        try:
            with policy.opener(request, timeout=timeout_seconds) as response:
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp_dest = dest.with_name(f".{dest.name}.{os.getpid()}.{attempt}.tmp")
                digest = hashlib.sha256()
                byte_count = 0
                body_sample = bytearray()
                with temp_dest.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                        if len(body_sample) < 1024 * 1024:
                            remaining = 1024 * 1024 - len(body_sample)
                            body_sample.extend(chunk[:remaining])
                os.replace(temp_dest, dest)
                content_type = response.headers.get("content-type")
                return {
                    "source_family_id": source_family_id,
                    "artifact_kind": artifact_kind,
                    "source_url": url,
                    "final_url": response.geturl(),
                    "local_path": workspace_relpath(workspace, dest),
                    "storage_lane": storage_lane,
                    "retrieval_status": "downloaded",
                    "status": "succeeded",
                    "http_status": int(response.status),
                    "content_type": content_type,
                    "byte_count": byte_count,
                    "sha256": digest.hexdigest(),
                    "retries_attempted": attempt - 1,
                    "auth_required": False,
                    "access_gate_detected": has_auth_gate(bytes(body_sample), policy.auth_gate_phrases),
                    "started_at": started_at,
                    "ended_at": now(),
                    "secret_values_recorded": False,
                }
        except HTTPError as exc:
            auth_required = int(exc.code) in {401, 403}
            last_error = f"HTTP {exc.code}"
            if auth_required or int(exc.code) not in retryable_codes or attempt >= retry_count:
                return {
                    "source_family_id": source_family_id,
                    "artifact_kind": artifact_kind,
                    "source_url": url,
                    "final_url": exc.geturl(),
                    "local_path": workspace_relpath(workspace, dest),
                    "storage_lane": storage_lane,
                    "retrieval_status": "auth_required" if auth_required else "unreachable_http_error",
                    "status": "blocked",
                    "http_status": int(exc.code),
                    "content_type": exc.headers.get("content-type") if exc.headers else None,
                    "byte_count": 0,
                    "sha256": None,
                    "retries_attempted": attempt - 1,
                    "auth_required": auth_required,
                    "access_gate_detected": False,
                    "started_at": started_at,
                    "ended_at": now(),
                    "error": last_error,
                    "secret_values_recorded": False,
                }
        except (URLError, TimeoutError, OSError) as exc:
            last_error = str(getattr(exc, "reason", exc))
            if attempt >= retry_count:
                return {
                    "source_family_id": source_family_id,
                    "artifact_kind": artifact_kind,
                    "source_url": url,
                    "final_url": None,
                    "local_path": workspace_relpath(workspace, dest),
                    "storage_lane": storage_lane,
                    "retrieval_status": "unreachable_fetch_error",
                    "status": "blocked",
                    "http_status": None,
                    "content_type": None,
                    "byte_count": 0,
                    "sha256": None,
                    "retries_attempted": attempt - 1,
                    "auth_required": False,
                    "access_gate_detected": False,
                    "started_at": started_at,
                    "ended_at": now(),
                    "error": last_error,
                    "secret_values_recorded": False,
                }
        time.sleep(2 ** (attempt - 1))

    raise AssertionError("download retry loop exited unexpectedly")


def discover_targets_from_artifact(workspace: Path, artifact: Mapping[str, Any], policy: CrawlPolicy) -> list[dict[str, Any]]:
    local_path = artifact.get("local_path")
    final_url = str(artifact.get("final_url") or artifact.get("source_url") or "")
    content_type = str(artifact.get("content_type") or "")
    if not local_path or "html" not in content_type:
        return []
    path = workspace / str(local_path)
    if not path.exists():
        return []
    discovered: list[dict[str, Any]] = []
    for item in extract_urls_from_html(final_url, path.read_bytes()):
        url = item["url"]
        target_kind = classify_discovered_url(url, policy)
        if target_kind is None:
            continue
        discovered.append(
            {
                "source_family_id": artifact.get("source_family_id"),
                "source_artifact_path": local_path,
                "target_kind": target_kind,
                "target_url": url,
                "html_tag": item.get("tag"),
                "html_attribute": item.get("attribute"),
            }
        )
    return discovered


def _site_cookie(ctx: LaneContext) -> str | None:
    credentials = ctx.auth.credentials if ctx.auth is not None else {}
    value = credentials.get(SITE_COOKIE_CREDENTIAL)
    return value if isinstance(value, str) else None


def _direct_failure(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "code": "direct_artifact_download_failed",
        "source_family_id": artifact.get("source_family_id"),
        "source_url": artifact.get("source_url"),
        "artifact_kind": artifact.get("artifact_kind"),
        "reason": artifact.get("error") or artifact.get("retrieval_status"),
        "auth_required": artifact.get("auth_required"),
        "retryable": artifact.get("retrieval_status") != "auth_required",
    }


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------


def media_scope_fields(options: Any, workspace: Path, policy: CrawlPolicy) -> dict[str, Any]:
    """The inventory's ``media_scope`` (full item-page expansion or an explicit waiver)."""

    skipped = bool(options.skip_item_page_media)
    return {
        "media_scope": {
            "item_page_media_mode": "skipped_with_explicit_scope" if skipped else "full_item_page_expansion",
            "waiver_path": workspace_relpath(workspace, policy.media_scope_waiver)
            if skipped and policy.media_scope_waiver is not None
            else None,
            "rationale": (
                "Paper acquisition is complete for the frozen corpus; item-page media probing "
                "is deferred because this run explicitly excludes slide OCR and podcast/video "
                "production."
                if skipped
                else None
            ),
        }
    }


def source_page_lane(policy: CrawlPolicy) -> AcquisitionLane:
    """Download every frozen family's source page and record the targets it links."""

    def run(ctx: LaneContext, frozen_corpus: Mapping[str, Any], _prior: Any) -> LaneResult:
        state: FrozenCorpusState = ctx.state
        options = ctx.options
        cookie = _site_cookie(ctx)
        blockers: list[dict[str, Any]] = []
        for record in frozen_corpus.get("included_records", []):
            if not isinstance(record, dict):
                continue
            family_id = str(record.get("source_family_id") or record.get("record_id") or "unknown")
            planned_acquisition = record.get("planned_acquisition")
            planned_url = planned_acquisition.get("source_url") if isinstance(planned_acquisition, dict) else None
            source_url = str(planned_url or record.get("final_url") or record.get("source_url") or "")
            if not source_url:
                state.blockers.append(
                    {
                        "code": "missing_source_url",
                        "source_family_id": family_id,
                        "reason": "included corpus record has no source URL",
                        "retryable": False,
                    }
                )
                continue
            artifact = download_http_artifact(
                workspace=ctx.workspace,
                source_family_id=family_id,
                artifact_kind="source_page",
                url=source_url,
                dest=source_page_destination(ctx.workspace, family_id, source_url, policy),
                timeout_seconds=options.download_timeout_seconds,
                retry_count=options.download_retry_count,
                resume=options.resume,
                policy=policy,
                extra_headers=site_cookie_headers(source_url, cookie, policy.cookie_hosts),
            )
            state.record_artifact(artifact, direct=False)
            if artifact["status"] != "succeeded":
                state.blockers.append(
                    {
                        "code": "source_page_download_failed",
                        "source_family_id": family_id,
                        "source_url": source_url,
                        "reason": artifact.get("error") or artifact.get("retrieval_status"),
                        "auth_required": artifact.get("auth_required"),
                        "retryable": artifact.get("retrieval_status") != "auth_required",
                    }
                )
            state.discovered_targets.extend(discover_targets_from_artifact(ctx.workspace, artifact, policy))
            state.persist()
        return LaneResult(blockers=blockers)

    return AcquisitionLane(lane_id="virtual-site-source-pages", run=run)


def direct_target_lane(policy: CrawlPolicy) -> AcquisitionLane:
    """Deduplicate the discovered targets and download every directly downloadable one."""

    def run(ctx: LaneContext, _frozen: Any, _prior: Any) -> LaneResult:
        state: FrozenCorpusState = ctx.state
        options = ctx.options
        cookie = _site_cookie(ctx)
        state.dedupe_targets()
        for target in list(state.discovered_targets):
            url = str(target.get("target_url") or "")
            kind = direct_download_kind(url, policy)
            if kind is None or url in state.downloaded_direct_urls:
                continue
            family_id = str(target.get("source_family_id") or "unknown")
            artifact = download_http_artifact(
                workspace=ctx.workspace,
                source_family_id=family_id,
                artifact_kind=kind,
                url=url,
                dest=direct_artifact_destination(ctx.workspace, family_id, kind, url, policy),
                timeout_seconds=options.download_timeout_seconds,
                retry_count=options.download_retry_count,
                resume=options.resume,
                policy=policy,
                extra_headers=site_cookie_headers(url, cookie, policy.cookie_hosts),
            )
            state.record_artifact(artifact, direct=True)
            state.downloaded_direct_urls.add(url)
            if artifact["status"] != "succeeded":
                state.blockers.append(_direct_failure(artifact))
            state.persist()
        return LaneResult()

    return AcquisitionLane(lane_id="virtual-site-direct-targets", run=run)


def item_page_expansion_lane(policy: CrawlPolicy) -> AcquisitionLane:
    """Probe every discovered item page (paced), retry auth-gated probes, download what they expose."""

    def run(ctx: LaneContext, _frozen: Any, _prior: Any) -> LaneResult:
        state: FrozenCorpusState = ctx.state
        options = ctx.options
        workspace = ctx.workspace
        cookie = _site_cookie(ctx)
        blockers: list[dict[str, Any]] = []
        item_page_targets = [
            target for target in state.discovered_targets if target.get("target_kind") in {"paper_page", "session_page"}
        ]
        probed_item_page_count = 0
        item_page_auth_required: list[dict[str, Any]] = []
        item_page_failures: list[dict[str, Any]] = []
        item_direct_targets: list[dict[str, Any]] = []
        worker_count = max(1, int(options.item_page_worker_count))

        def probe(url: str, family_id: str, pacer: RequestPacer) -> dict[str, Any]:
            return download_http_artifact(
                workspace=workspace,
                source_family_id=family_id,
                artifact_kind="item_page_auth_probe",
                url=url,
                dest=probe_page_destination(workspace, url, policy),
                timeout_seconds=options.download_timeout_seconds,
                retry_count=options.download_retry_count,
                resume=options.resume,
                policy=policy,
                user_agent=policy.probe_user_agent,
                extra_headers=site_cookie_headers(url, cookie, policy.cookie_hosts),
                pacer=pacer,
            )

        def queue_exposed_targets(artifact: Mapping[str, Any]) -> None:
            for target in discover_targets_from_artifact(workspace, artifact, policy):
                state.add_discovered_target(target)
                target_url = str(target.get("target_url") or "")
                if direct_download_kind(target_url, policy) is None or target_url in state.downloaded_direct_urls:
                    continue
                state.downloaded_direct_urls.add(target_url)
                item_direct_targets.append(target)

        if item_page_targets and not options.skip_item_page_media:
            state.flags[PAPER_COUNT_FROM_DIRECT_ONLY] = True
            sorted_item_page_targets = sorted(
                item_page_targets,
                key=lambda item: (0 if item.get("target_kind") == "paper_page" else 1, str(item.get("target_url"))),
            )
            item_page_pacer = policy.pacer(
                max(0.0, float(getattr(options, "item_page_request_delay_seconds", 0.0) or 0.0))
            )

            def download_item_page(target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                probe_url = str(target.get("target_url"))
                return target, probe(probe_url, str(target.get("source_family_id") or "item-page"), item_page_pacer)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(download_item_page, target) for target in sorted_item_page_targets]
                for future in as_completed(futures):
                    probe_target, probe_artifact = future.result()
                    state.record_artifact(probe_artifact, direct=False)
                    probed_item_page_count += 1
                    probe_url = str(probe_target.get("target_url"))
                    if probe_artifact.get("access_gate_detected") or probe_artifact.get("auth_required"):
                        item_page_auth_required.append(
                            {
                                "source_family_id": probe_target.get("source_family_id"),
                                "source_url": probe_url,
                                "reason": probe_artifact.get("error") or probe_artifact.get("retrieval_status"),
                            }
                        )
                        continue
                    if probe_artifact.get("status") != "succeeded":
                        item_page_failures.append(
                            {
                                "source_family_id": probe_target.get("source_family_id"),
                                "source_url": probe_url,
                                "reason": probe_artifact.get("error") or probe_artifact.get("retrieval_status"),
                            }
                        )
                        continue
                    queue_exposed_targets(probe_artifact)

            auth_retry_passes = max(0, int(getattr(options, "item_page_auth_retry_passes", 0) or 0))
            if item_page_auth_required and auth_retry_passes:
                auth_retry_pacer = policy.pacer(
                    max(0.0, float(getattr(options, "item_page_auth_retry_delay_seconds", 0.0) or 0.0))
                )
                for _ in range(auth_retry_passes):
                    if not item_page_auth_required:
                        break
                    pending_retries = item_page_auth_required
                    item_page_auth_required = []
                    for entry in pending_retries:
                        retry_artifact = probe(
                            str(entry.get("source_url")), str(entry.get("source_family_id") or "item-page"), auth_retry_pacer
                        )
                        state.record_artifact(retry_artifact, direct=False)
                        if retry_artifact.get("access_gate_detected") or retry_artifact.get("auth_required"):
                            item_page_auth_required.append(entry)
                            continue
                        if retry_artifact.get("status") != "succeeded":
                            item_page_failures.append(entry)
                            continue
                        queue_exposed_targets(retry_artifact)

            if item_direct_targets:

                def download_item_direct_target(target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
                    url = str(target.get("target_url") or "")
                    kind = direct_download_kind(url, policy)
                    if kind is None:
                        return target, None
                    family_id = str(target.get("source_family_id") or "unknown")
                    artifact = download_http_artifact(
                        workspace=workspace,
                        source_family_id=family_id,
                        artifact_kind=kind,
                        url=url,
                        dest=direct_artifact_destination(workspace, family_id, kind, url, policy),
                        timeout_seconds=options.download_timeout_seconds,
                        retry_count=options.download_retry_count,
                        resume=options.resume,
                        policy=policy,
                        extra_headers=site_cookie_headers(url, cookie, policy.cookie_hosts),
                    )
                    return target, artifact

                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(download_item_direct_target, target) for target in item_direct_targets]
                    for future in as_completed(futures):
                        _, artifact = future.result()
                        if artifact is None:
                            continue
                        state.record_artifact(artifact, direct=True)
                        if artifact["status"] != "succeeded":
                            blockers.append(_direct_failure(artifact))

            if item_page_auth_required:
                blockers.append(
                    {
                        "code": policy.login_required_code,
                        "reason": policy.login_required_reason,
                        "retryable": True,
                        "auth_required": True,
                        "item_page_target_count": len(item_page_targets),
                        "auth_required_count": len(item_page_auth_required),
                        "sample_pages": item_page_auth_required[:10],
                        "next_action": policy.login_required_next_action,
                    }
                )
            if item_page_failures:
                blockers.append(
                    {
                        "code": "item_page_expansion_failures",
                        "reason": "one or more discovered paper/session item pages failed during full expansion",
                        "retryable": True,
                        "auth_required": False,
                        "item_page_target_count": len(item_page_targets),
                        "failed_count": len(item_page_failures),
                        "sample_failures": item_page_failures[:10],
                    }
                )
            if state.count_succeeded("media_asset") == 0:
                blockers.append(
                    {
                        "code": "no_direct_media_artifacts_discovered",
                        "reason": "source and item pages exposed no direct public video/audio/media download targets",
                        "retryable": True,
                        "auth_required": False,
                        "item_page_target_count": len(item_page_targets),
                        "probed_item_page_count": probed_item_page_count,
                        "next_action": "add item-level media downloader or provide official media export before Phase 5",
                    }
                )

        skipped = bool(options.skip_item_page_media)
        expansion_clean = (
            probed_item_page_count == len(item_page_targets) and not item_page_auth_required and not item_page_failures
        )
        return LaneResult(
            blockers=blockers,
            receipt_fields={
                "item_page_expansion": {
                    "target_count": len(item_page_targets),
                    "probed_count": probed_item_page_count,
                    "auth_required_count": len(item_page_auth_required),
                    "failed_count": len(item_page_failures),
                    "worker_count": worker_count,
                    "status": "skipped_with_explicit_scope" if skipped else "completed" if expansion_clean else "partial",
                    "waiver_path": workspace_relpath(workspace, policy.media_scope_waiver)
                    if skipped and policy.media_scope_waiver is not None
                    else None,
                    "complete": not skipped and expansion_clean,
                }
            },
        )

    return AcquisitionLane(lane_id="virtual-site-item-page-expansion", run=run)


def media_family_coverage_lane(policy: CrawlPolicy) -> AcquisitionLane:
    """Fail closed when a declared media-bearing program family yielded no item page.

    Every declared media-bearing family (tutorials, invited talks, orals,
    workshops, expo) must yield at least one discovered item page unless the
    operator explicitly excluded it (``options.exclude_media_families``).
    """

    def run(ctx: LaneContext, _frozen: Any, _prior: Any) -> LaneResult:
        state: FrozenCorpusState = ctx.state
        excluded = sorted({str(item) for item in (getattr(ctx.options, "exclude_media_families", None) or [])})

        def resolve_bare_id_family(url: str) -> str | None:
            # An index page (e.g. the conference calendar) commonly links items
            # from other families as bare-ID URLs with no family path segment;
            # source_family_id then names the index, not the item's own family.
            # Read the already-cached item page and classify it from its title.
            cache_path = probe_page_destination(ctx.workspace, url, policy)
            if not cache_path.is_file():
                return None
            try:
                html_text = cache_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            return classify_item_page_title_family(html_text)

        coverage = media_family_coverage(
            state.discovered_targets,
            year=policy.virtual_year,
            declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
            resolve_bare_id_family=resolve_bare_id_family,
        )
        coverage["excluded_families"] = excluded
        return LaneResult(
            blockers=media_family_coverage_blockers(coverage, excluded_families=excluded),
            receipt_fields={"media_family_coverage": coverage},
        )

    return AcquisitionLane(lane_id="virtual-site-media-family-coverage", run=run, phase=LANE_PHASE_GATE)


def prior_artifact_carry_forward_lane(
    keep: Callable[[Mapping[str, Any]], bool], *, lane_id: str = "prior-media-carry-forward"
) -> AcquisitionLane:
    """Carry the prior inventory's direct artifacts that ``keep`` accepts into this run.

    A focused recovery owner (for example a SlidesLive audio recovery) is an
    acquisition owner in its own right: when Stage 4 is resumed after it
    completes, only its governed records are carried into the canonical
    inventory, never artifacts from earlier broad probes. Declare this lane
    first so carried artifacts keep their position ahead of fresh ones.
    """

    def run(ctx: LaneContext, _frozen: Any, prior_inventory: Mapping[str, Any] | None) -> LaneResult:
        from conference_pipeline.production.download_corpus_stage import carry_forward_prior_artifacts

        return LaneResult(artifacts=carry_forward_prior_artifacts(prior_inventory, keep))

    return AcquisitionLane(lane_id=lane_id, run=run)
