"""Conference-general workshop-paper acquisition from the CVF Open Access site.

CVF conferences (CVPR, ICCV, WACV) publish their archival workshop proceedings
at ``openaccess.thecvf.com/<Venue><Year>_workshops/menu``. Many workshops never
post papers on OpenReview (or post metadata only), so the OpenReview workshop
downloader leaves them as ``external_site_required`` stubs. This module fills
the archival part of that gap from one uniform, public source:

1. parse the workshop menu into per-workshop listing pages;
2. parse each listing page into papers (title, authors, pdf/supp/arXiv links);
3. annotate each CVF workshop with the OpenReview workshop venue it matches in
   ``storage/workshops/inventory.json`` (if any);
4. skip papers whose title was already downloaded from OpenReview, so the
   corpus never admits the same paper twice;
5. download the remaining PDFs with resume, rate limiting and 429 backoff,
   writing per-paper receipts in the same ``workshop_paper_receipt`` schema the
   OpenReview downloader writes, plus a stage receipt with the same
   ``per_workshop`` / ``output_dir`` shape.

Non-archival workshop tracks are, by definition, not on this site; those still
need the workshop's own website or arXiv.

The menu URL is conference-specific and comes from the workspace
``config.yaml`` (``urls.openaccess_workshops``), falling back to the CVF naming
convention applied to ``urls.openaccess``.

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)`` (listed in
:mod:`conference_pipeline.acquisition.lane_catalog` as
``cvf-openaccess-workshop-papers``).

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus`` workshop
  papers: CVPR workshop proceedings that never reach OpenReview.
* **May also fit:** ICCV and WACV (same CVF Open Access layout).
"""

from __future__ import annotations

import concurrent.futures
import difflib
import hashlib
import html
import http.client
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urljoin, urlsplit

from conference_pipeline.acquisition.workshop_pipeline import slugify_venueid
from conference_pipeline.conference_identity import load_conference_identity

RECOMMENDED_FOR = ("CVPR",)

SOURCE_ID = "cvf_openaccess"
STAGE_ID = "acquisition.cvf_openaccess_workshop_paper_download"
DEFAULT_USER_AGENT = "neurips-foundation-conf-pipeline/cvf-openaccess-workshops"
DEFAULT_RATE_LIMIT_RPS = 1.0
DEFAULT_WORKERS = 4
DEFAULT_RECEIPT_RELATIVE = Path("storage/workshops/cvf_openaccess_download_receipt.json")
DEFAULT_PAPERS_DIR_RELATIVE = Path("storage/workshops/papers")
DEFAULT_INVENTORY_RELATIVE = Path("storage/workshops/inventory.json")
DEFAULT_OPENREVIEW_RECEIPT_RELATIVE = Path("storage/workshops/download_receipt.json")
BACKOFF_SCHEDULE = (1.0, 2.0, 4.0, 8.0, 16.0)
PDF_MAGIC = b"%PDF-"
CVF_HOST = "openaccess.thecvf.com"

_MENU_HREF_RE = re.compile(r'<a\s+href="(?P<href>/[^"/]{1,80}_workshops/(?P<short>[^"/]{1,120}))"\s*>(?P<title>[^<]{1,400})</a>')
_PAPER_SPLIT_RE = re.compile(r'<dt\s+class="ptitle"\s*>')
_TITLE_RE = re.compile(r'<a\s+href="(?P<href>/content/[^"]{1,400}/html/[^"/]{1,300}\.html)"\s*>(?P<title>.{1,1000}?)</a>', re.S)
_AUTHOR_RE = re.compile(r'name="query_author"\s+value="(?P<name>[^"]{0,300})"')
_PDF_RE = re.compile(r'href="(?P<href>/content/[^"]{1,400}/papers/(?P<stem>[^"/]{1,300})\.pdf)"')
_SUPP_RE = re.compile(r'href="(?P<href>/content/[^"]{1,400}/supplemental/[^"/]{1,300})"')
_ARXIV_RE = re.compile(r'href="(?P<href>https?://arxiv\.org/abs/[^"]{1,60})"')
_TAG_RE = re.compile(r"<[^>]{0,400}>")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", value)).split())


def normalize_title(title: str) -> str:
    """Case/punctuation/whitespace-insensitive title key for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", "", html.unescape(title).casefold())


def _normalize_short_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def _short_name_keys(name: str) -> set[str]:
    """Exact key plus the key with a trailing edition year / leading ordinal removed.

    Organizers rename the same workshop between hosts (``CV4Clinic2026`` vs
    ``CV4Clinic``, ``6thAdvML@CV`` vs ``AdvML@CV``); only these two
    deterministic rewrites are applied so unrelated workshops never merge.
    """
    base = _normalize_short_name(name)
    keys = {base}
    keys.add(re.sub(r"(19|20)\d\d$", "", base))
    keys.add(re.sub(r"^\d+(st|nd|rd|th)", "", base))
    return {key for key in keys if len(key) >= 2}


# --------------------------------------------------------------------------- config


def resolve_menu_url(workspace: Path, override: str | None = None) -> str:
    """Return the CVF workshop menu URL for ``workspace``.

    ``urls.openaccess_workshops`` wins; otherwise the CVF convention
    ``<urls.openaccess>_workshops/menu`` is applied. Raises ``ValueError`` when
    neither is configured or the URL is not on the CVF open-access host.
    """
    url = (override or "").strip()
    if not url:
        identity = load_conference_identity(workspace)
        url = identity.url("openaccess_workshops")
        if not url:
            main = identity.url("openaccess").rstrip("/")
            if main:
                url = f"{main}_workshops/menu"
    if not url:
        raise ValueError(
            "no CVF open-access workshop menu configured: set urls.openaccess_workshops "
            "(or urls.openaccess) in the workspace config.yaml"
        )
    parsed = urlsplit(url)
    if parsed.scheme != "https" or (parsed.hostname or "") != CVF_HOST:
        raise ValueError(f"CVF workshop menu must be an https://{CVF_HOST} URL: {url}")
    return url


# --------------------------------------------------------------------------- parsing


@dataclass(frozen=True)
class CvfWorkshop:
    short_name: str
    title: str
    url: str

    @property
    def venueid(self) -> str:
        """Stable pseudo-venue id (``host/<Venue>_workshops/<short>``); names the output dir."""
        parsed = urlsplit(self.url)
        return f"{parsed.hostname}{parsed.path}"


@dataclass(frozen=True)
class CvfPaper:
    stem: str
    title: str
    authors: tuple[str, ...]
    pdf_url: str
    html_url: str
    supplemental_url: str | None = None
    arxiv_url: str | None = None

    @property
    def paper_id(self) -> str:
        """Filesystem-safe, deterministic id (CVF stems can exceed 200 chars)."""
        return "cvf_" + hashlib.sha256(self.pdf_url.encode("utf-8")).hexdigest()[:20]


def parse_workshop_menu(page_html: str, base_url: str) -> list[CvfWorkshop]:
    workshops: list[CvfWorkshop] = []
    seen: set[str] = set()
    for match in _MENU_HREF_RE.finditer(page_html):
        url = urljoin(base_url, match.group("href"))
        if url in seen or match.group("short") == "menu":
            continue
        seen.add(url)
        workshops.append(
            CvfWorkshop(
                short_name=html.unescape(match.group("short")),
                title=_clean_text(match.group("title")),
                url=url,
            )
        )
    return workshops


def parse_workshop_papers(page_html: str, base_url: str) -> list[CvfPaper]:
    """One record per ``<dt class="ptitle">`` block that declares its own PDF link."""
    papers: list[CvfPaper] = []
    seen: set[str] = set()
    chunks = _PAPER_SPLIT_RE.split(page_html)[1:]
    for chunk in chunks:
        title_match = _TITLE_RE.search(chunk)
        pdf_match = _PDF_RE.search(chunk)
        if not title_match or not pdf_match:
            continue
        pdf_url = urljoin(base_url, pdf_match.group("href"))
        if pdf_url in seen:
            continue
        seen.add(pdf_url)
        authors = tuple(
            name
            for name in (_clean_text(m.group("name")) for m in _AUTHOR_RE.finditer(chunk))
            if name
        )
        supp_match = _SUPP_RE.search(chunk)
        arxiv_match = _ARXIV_RE.search(chunk)
        papers.append(
            CvfPaper(
                stem=pdf_match.group("stem"),
                title=_clean_text(title_match.group("title")),
                authors=authors,
                pdf_url=pdf_url,
                html_url=urljoin(base_url, title_match.group("href")),
                supplemental_url=urljoin(base_url, supp_match.group("href")) if supp_match else None,
                arxiv_url=arxiv_match.group("href") if arxiv_match else None,
            )
        )
    return papers


# --------------------------------------------------------------------------- joins


def match_openreview_venues(
    workshops: Iterable[CvfWorkshop], inventory: Mapping[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """Map CVF short name -> matching OpenReview inventory row (unambiguous key matches only)."""
    rows = [row for row in (inventory or {}).get("workshops", []) or [] if isinstance(row, Mapping)]
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        venueid = str(row.get("venueid") or "")
        if not venueid or row.get("sub_track_parent"):
            continue
        for key in _short_name_keys(venueid.rsplit("/", 1)[-1]):
            by_key.setdefault(key, []).append(row)
    matches: dict[str, dict[str, Any]] = {}
    for workshop in workshops:
        candidates: dict[str, Mapping[str, Any]] = {}
        for key in _short_name_keys(workshop.short_name):
            for row in by_key.get(key, []):
                candidates[str(row["venueid"])] = row
        if len(candidates) == 1:
            row = next(iter(candidates.values()))
            matches[workshop.short_name] = {
                "openreview_venueid": row.get("venueid"),
                "openreview_pdf_availability_class": row.get("pdf_availability_class"),
                "openreview_decision": row.get("decision"),
            }
    return matches


@dataclass(frozen=True)
class KnownPaper:
    title_key: str
    author_keys: frozenset[str]
    ref: str


def author_key(name: str) -> str:
    """``first|last`` token key, so middle names/initials don't split one author."""
    tokens = re.findall(r"[a-z0-9]+", html.unescape(name).casefold())
    if not tokens:
        return ""
    return tokens[0] if len(tokens) == 1 else f"{tokens[0]}|{tokens[-1]}"


class KnownPaperIndex:
    """Papers already acquired elsewhere (OpenReview), for duplicate detection.

    A CVF paper is a duplicate when its normalized title matches exactly, or
    when the camera-ready was retitled but it is plainly the same paper: most
    authors shared (Jaccard >= ``AUTHOR_JACCARD``) and the titles still
    resemble each other (ratio >= ``RETITLED_MIN_RATIO``).
    """

    AUTHOR_JACCARD = 0.6
    RETITLED_MIN_RATIO = 0.5

    def __init__(self, papers: Iterable[KnownPaper]) -> None:
        self._by_title: dict[str, KnownPaper] = {}
        self._by_author: dict[str, list[KnownPaper]] = {}
        for paper in papers:
            if paper.title_key:
                self._by_title.setdefault(paper.title_key, paper)
            for key in paper.author_keys:
                self._by_author.setdefault(key, []).append(paper)

    def __len__(self) -> int:
        return len(self._by_title)

    def find(self, title: str, authors: Iterable[str]) -> tuple[KnownPaper, str] | None:
        title_key = normalize_title(title)
        exact = self._by_title.get(title_key)
        if exact is not None:
            return exact, "exact_title"
        keys = {key for key in (author_key(a) for a in authors) if key}
        candidates = {id(p): p for key in keys for p in self._by_author.get(key, [])}
        best: tuple[float, KnownPaper] | None = None
        for paper in candidates.values():
            union = keys | paper.author_keys
            if not union or len(keys & paper.author_keys) / len(union) < self.AUTHOR_JACCARD:
                continue
            ratio = difflib.SequenceMatcher(None, title_key, paper.title_key).ratio()
            if ratio >= self.RETITLED_MIN_RATIO and (best is None or ratio > best[0]):
                best = (ratio, paper)
        if best is not None:
            return best[1], f"retitled_same_authors(title_ratio={best[0]:.2f})"
        return None


def resolve_receipt_output_dir(workspace: Path, value: object) -> Path:
    """A download receipt's ``output_dir`` as a directory path.

    Receipts record it as absolute, workspace-relative, or relative to the
    repository root the downloader ran from (``conferences/<slug>/...``). All
    three must resolve, or duplicate checks silently see no known papers.
    """

    text = str(value or "").strip()
    if not text:
        return workspace / DEFAULT_PAPERS_DIR_RELATIVE
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    options = [workspace / candidate, Path.cwd() / candidate]
    # Repository-root relative: drop the workspace's own prefix when present.
    parts = candidate.parts
    workspace_parts = Path(workspace).resolve().parts
    for size in range(len(parts), 0, -1):
        if tuple(workspace_parts[-size:]) == tuple(parts[:size]):
            options.append(workspace.joinpath(*parts[size:]))
            break
    return next((option for option in options if option.is_dir()), options[0])


def iter_paper_receipts(workspace: Path, download_receipt: Mapping[str, Any] | None) -> Iterable[dict[str, Any]]:
    """Yield the succeeded per-paper receipts behind a workshop paper download receipt.

    Works for any receipt with the shared ``output_dir`` + ``per_workshop[].venueid``
    layout (OpenReview, CVF Open Access, workshop-site downloads).
    """
    if not download_receipt:
        return
    output_dir = resolve_receipt_output_dir(workspace, download_receipt.get("output_dir"))
    for entry in download_receipt.get("per_workshop", []) or []:
        venueid = str((entry or {}).get("venueid") or "")
        receipts_dir = output_dir / slugify_venueid(venueid) / "receipts"
        if not venueid or not receipts_dir.is_dir():
            continue
        for receipt_file in sorted(receipts_dir.glob("*.json")):
            try:
                receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(receipt, dict) and receipt.get("status") == "succeeded":
                yield {**receipt, "venueid": receipt.get("venueid") or venueid}


def load_known_papers(workspace: Path, download_receipts: Iterable[Mapping[str, Any] | None]) -> KnownPaperIndex:
    """Index every succeeded per-paper receipt behind the given download receipts."""
    papers: list[KnownPaper] = []
    for download_receipt in download_receipts:
        for receipt in iter_paper_receipts(workspace, download_receipt):
            authors = receipt.get("authors") if isinstance(receipt.get("authors"), list) else []
            papers.append(
                KnownPaper(
                    title_key=normalize_title(str(receipt.get("title") or "")),
                    author_keys=frozenset(k for k in (author_key(str(a)) for a in authors) if k),
                    ref=f"{receipt['venueid']}:{receipt.get('note_id')}",
                )
            )
    return KnownPaperIndex(papers)


def load_known_openreview_papers(workspace: Path, openreview_receipt: Mapping[str, Any] | None) -> KnownPaperIndex:
    """Index the succeeded per-paper receipts of the OpenReview workshop download."""
    return load_known_papers(workspace, [openreview_receipt])


# --------------------------------------------------------------------------- http


class RequestStartLimiter:
    """Space out request starts while allowing response bodies to overlap."""

    def __init__(self, rate_limit_rps: float) -> None:
        self.interval = 1.0 / max(0.1, rate_limit_rps)
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_start)
            self._next_start = start_at + self.interval
        if start_at > now:
            time.sleep(start_at - now)


Fetcher = Callable[[str], tuple[int, str, bytes]]


def urllib_fetch(url: str, *, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 120.0) -> tuple[int, str, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()


def fetch_with_backoff(
    url: str,
    *,
    fetch: Fetcher,
    limiter: RequestStartLimiter | None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    backoff_events: list[dict[str, Any]] = []
    last_status: int | None = None
    for attempt, backoff in enumerate((0.0, *BACKOFF_SCHEDULE), start=1):
        if backoff:
            sleep(backoff)
        if limiter is not None:
            limiter.wait()
        try:
            status, content_type, body = fetch(url)
            return {
                "ok": True,
                "http_status": status,
                "content_type": content_type,
                "body": body,
                "attempts": attempt,
                "backoff_events": backoff_events,
            }
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt > len(BACKOFF_SCHEDULE):
                return {
                    "ok": False,
                    "http_status": exc.code,
                    "failure_reason": f"HTTPError {exc.code} {exc.reason}",
                    "attempts": attempt,
                    "backoff_events": backoff_events,
                }
            backoff_events.append({"attempt": attempt, "http_status": exc.code, "at": now_iso()})
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, OSError) as exc:
            if attempt > len(BACKOFF_SCHEDULE):
                return {
                    "ok": False,
                    "http_status": last_status,
                    "failure_reason": f"{exc.__class__.__name__}: {exc}",
                    "attempts": attempt,
                    "backoff_events": backoff_events,
                }
            backoff_events.append({"attempt": attempt, "error_type": exc.__class__.__name__, "at": now_iso()})
    return {  # pragma: no cover - loop always returns
        "ok": False,
        "http_status": last_status,
        "failure_reason": "exceeded backoff schedule",
        "attempts": len(BACKOFF_SCHEDULE) + 1,
        "backoff_events": backoff_events,
    }


# --------------------------------------------------------------------------- download


@dataclass
class RunOptions:
    workspace: Path
    menu_url: str
    output_dir_relative: Path = DEFAULT_PAPERS_DIR_RELATIVE
    inventory_relative: Path = DEFAULT_INVENTORY_RELATIVE
    openreview_receipt_relative: Path = DEFAULT_OPENREVIEW_RECEIPT_RELATIVE
    rate_limit_rps: float = DEFAULT_RATE_LIMIT_RPS
    workers: int = DEFAULT_WORKERS
    max_papers: int = 0
    dry_run: bool = False
    only_workshops: frozenset[str] = field(default_factory=frozenset)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _already_succeeded(receipt: Mapping[str, Any] | None, pdf_path: Path) -> bool:
    if not receipt or receipt.get("status") != "succeeded" or not pdf_path.is_file():
        return False
    size = pdf_path.stat().st_size
    return size > 0 and size == receipt.get("byte_count")


def _download_one(
    paper: CvfPaper,
    *,
    workshop: CvfWorkshop,
    workshop_dir: Path,
    workspace: Path,
    fetch: Fetcher,
    limiter: RequestStartLimiter,
) -> dict[str, Any]:
    pdf_path = workshop_dir / f"{paper.paper_id}.pdf"
    record: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_paper_receipt",
        "source": SOURCE_ID,
        "note_id": paper.paper_id,
        "venueid": workshop.venueid,
        "cvf_workshop_short_name": workshop.short_name,
        "cvf_stem": paper.stem,
        "title": paper.title,
        "authors": list(paper.authors),
        "pdf_url": paper.pdf_url,
        "html_url": paper.html_url,
        "supplemental_url": paper.supplemental_url,
        "arxiv_url": paper.arxiv_url,
        "pdf_path": pdf_path.relative_to(workspace).as_posix(),
        "attempted_at": now_iso(),
    }
    outcome = fetch_with_backoff(paper.pdf_url, fetch=fetch, limiter=limiter)
    body = outcome.pop("body", b"")
    if outcome.pop("ok"):
        if not body.startswith(PDF_MAGIC):
            outcome.update({"status": "unreachable", "failure_reason": "response_not_pdf", "byte_count": len(body)})
        else:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(body)
            outcome.update(
                {"status": "succeeded", "byte_count": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            )
    else:
        outcome["status"] = "unreachable"
    outcome["last_http_status"] = outcome.pop("http_status", None)
    return {**record, **outcome}


def run(options: RunOptions, *, fetch: Fetcher = urllib_fetch) -> dict[str, Any]:
    """Discover (and unless ``dry_run``) download CVF workshop papers; return the stage receipt."""
    workspace = options.workspace
    output_dir = workspace / options.output_dir_relative
    limiter = RequestStartLimiter(options.rate_limit_rps)
    inventory = _read_json(workspace / options.inventory_relative)
    openreview_receipt = _read_json(workspace / options.openreview_receipt_relative)
    known_papers = load_known_openreview_papers(workspace, openreview_receipt)

    base = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_paper_download_receipt",
        "stage_id": STAGE_ID,
        "source": SOURCE_ID,
        "menu_url": options.menu_url,
        "dry_run": options.dry_run,
        "rate_limit_rps": options.rate_limit_rps,
        "workers": options.workers,
        "max_papers": options.max_papers,
        "output_dir": options.output_dir_relative.as_posix(),
        "inventory_path": options.inventory_relative.as_posix() if inventory else None,
        "openreview_receipt_path": options.openreview_receipt_relative.as_posix() if openreview_receipt else None,
        "openreview_papers_known": len(known_papers),
    }
    menu = fetch_with_backoff(options.menu_url, fetch=fetch, limiter=limiter)
    if not menu.get("ok"):
        return {**base, "status": "blocked", "blocked_reason": f"menu_fetch_failed: {menu.get('failure_reason')}", "generated_at": now_iso()}
    workshops = parse_workshop_menu(menu["body"].decode("utf-8", errors="replace"), options.menu_url)
    if not workshops:
        return {**base, "status": "blocked", "blocked_reason": "menu_has_no_workshops", "generated_at": now_iso()}
    if options.only_workshops:
        workshops = [w for w in workshops if w.short_name in options.only_workshops]
    openreview_matches = match_openreview_venues(workshops, inventory)

    totals = {"listed": 0, "duplicates_of_openreview": 0, "planned": 0, "attempted": 0, "succeeded": 0, "unreachable": 0, "resume_skipped": 0, "available": 0}
    per_workshop: list[dict[str, Any]] = []
    budget_hit = False
    for workshop in workshops:
        page = fetch_with_backoff(workshop.url, fetch=fetch, limiter=limiter)
        summary: dict[str, Any] = {
            "venueid": workshop.venueid,
            "cvf_short_name": workshop.short_name,
            "title": workshop.title,
            "listing_url": workshop.url,
            **openreview_matches.get(workshop.short_name, {"openreview_venueid": None}),
        }
        if not page.get("ok"):
            per_workshop.append({**summary, "status": "blocked", "blocked_reason": f"listing_fetch_failed: {page.get('failure_reason')}"})
            continue
        papers = parse_workshop_papers(page["body"].decode("utf-8", errors="replace"), workshop.url)
        workshop_dir = output_dir / slugify_venueid(workshop.venueid)
        duplicates = []
        for p in papers:
            hit = known_papers.find(p.title, p.authors)
            if hit is not None:
                duplicates.append({"paper_id": p.paper_id, "title": p.title, "duplicate_of": hit[0].ref, "match": hit[1]})
        duplicate_ids = {d["paper_id"] for d in duplicates}
        jobs: list[CvfPaper] = []
        skipped = 0
        for paper in papers:
            if paper.paper_id in duplicate_ids:
                continue
            receipt_path = workshop_dir / "receipts" / f"{paper.paper_id}.json"
            if _already_succeeded(_read_json(receipt_path), workshop_dir / f"{paper.paper_id}.pdf"):
                skipped += 1
                continue
            if options.max_papers and totals["planned"] + len(jobs) >= options.max_papers:
                budget_hit = True
                break
            jobs.append(paper)
        totals["listed"] += len(papers)
        totals["duplicates_of_openreview"] += len(duplicates)
        totals["planned"] += len(jobs)
        totals["resume_skipped"] += skipped
        succeeded = unreachable = 0
        if jobs and not options.dry_run:
            totals["attempted"] += len(jobs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, options.workers)) as pool:
                futures = [
                    pool.submit(
                        _download_one,
                        paper,
                        workshop=workshop,
                        workshop_dir=workshop_dir,
                        workspace=workspace,
                        fetch=fetch,
                        limiter=limiter,
                    )
                    for paper in jobs
                ]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    write_json(workshop_dir / "receipts" / f"{record['note_id']}.json", record)
                    if record["status"] == "succeeded":
                        succeeded += 1
                    else:
                        unreachable += 1
        totals["succeeded"] += succeeded
        totals["unreachable"] += unreachable
        if options.dry_run:
            status = "planned"
        elif unreachable or budget_hit:
            status = "partial"
        elif succeeded + skipped == 0:
            # Nothing of this workshop's own to admit (empty listing, or every
            # paper already acquired from OpenReview): no receipts directory
            # exists, and corpus builders only admit ``succeeded`` entries.
            status = "no_new_papers"
        else:
            status = "succeeded"
        per_workshop.append(
            {
                **summary,
                "paper_count_listed": len(papers),
                "duplicates_of_openreview": duplicates,
                "planned": len(jobs),
                "downloaded": succeeded,
                "resume_skipped": skipped,
                "available": succeeded + skipped,
                "unreachable": unreachable,
                "status": status,
            }
        )
        if budget_hit:
            break
    totals["available"] = totals["succeeded"] + totals["resume_skipped"]
    blocked = sum(1 for entry in per_workshop if entry["status"] == "blocked")
    if options.dry_run:
        status = "dry_run"
    elif totals["unreachable"] or blocked or budget_hit:
        status = "partial"
    else:
        status = "succeeded"
    return {
        **base,
        "status": status,
        "workshop_count": len(per_workshop),
        "workshops_matched_to_openreview": sum(1 for e in per_workshop if e.get("openreview_venueid")),
        "workshops_blocked": blocked,
        "totals": totals,
        "per_workshop": per_workshop,
        "generated_at": now_iso(),
    }
