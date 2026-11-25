"""Resolve public source links for presentation references.

Structured deck contracts often carry compact source identifiers plus local
ledger paths. Rendered presentations should expose public conference URLs
instead of those local artifacts. This module rehydrates those links from the
cached conference corpus without calling live providers.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

URL_FIELDS = (
    "url",
    "source_url",
    "openreview_url",
    "forum_url",
    "paper_url",
    "doi_url",
    "slideslive_url",
    "media_url",
    "dataset_url",
    "artifact_url",
    "pdf_url",
)
ANCHOR_RE = re.compile(r"<a\b[^>]*?href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")


POSTER_PATH_RE = re.compile(r"(?:^|/)media_asset/poster-(\d+)(?:/|$)")
POSTER_URL_RE = re.compile(r"/poster/(\d+)(?:[/?#]|$)")
POSTER_PROBE_FILENAME_RE = re.compile(r"(?:^|-)poster-(\d+)-[0-9a-f]+\.html$", re.I)
HTML_TITLE_RE = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)
PROGRAM_TITLE_PREFIX_RE = re.compile(
    r"^[A-Z][A-Za-z]{1,7}\s+(?:Poster|Oral|Spotlight|Talk|Workshop|Tutorial|Invited\s+Talk)\s+"
)
OPENREVIEW_FORUM_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_-]+)")
PAPER_URL_RE = re.compile(r"(?:openreview\.net/(?:forum|pdf)|arxiv\.org/(?:abs|pdf)|/pdf$|\.pdf$)", re.I)


def _is_paper_url(url: str) -> bool:
    return bool(PAPER_URL_RE.search(str(url or "")))


def canonical_public_url(value: object, *, site_root: str = "") -> str:
    """Return a public URL, canonicalizing conference-relative hrefs when possible.

    ``site_root`` is the conference ``scheme://host`` (from the workspace
    ``config.yaml`` via :class:`ConferenceIdentity`). A relative ``/`` href is
    only rehydrated when ``site_root`` is supplied; without it the relative link
    is dropped (returns "") rather than defaulting to a wrong-conference host.
    """

    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith("/"):
        return f"{site_root.rstrip('/')}{raw}" if site_root else ""
    return ""


def workshop_url_from_event_id(value: object, *, virtual_url: str = "") -> str:
    """Build a public workshop URL from a numeric event id and the conference
    ``urls.virtual`` root (e.g. ``https://icml.cc/virtual/2026``)."""
    event_id = str(value or "").strip()
    if not event_id or not virtual_url:
        return ""
    if not re.fullmatch(r"\d+", event_id):
        return ""
    return f"{virtual_url.rstrip('/')}/workshop/{event_id}"


def _clean_text(value: object) -> str:
    text = TAG_RE.sub(" ", str(value or ""))
    text = html.unescape(text)
    return " ".join(text.split())


def _norm_title(value: object) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.get("records") or []:
            if isinstance(nested, dict):
                yield nested
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield from _iter_records(item)


def _first_public_url(
    record: dict[str, Any], *, site_root: str = "", virtual_url: str = ""
) -> str:
    for field_name in URL_FIELDS:
        url = canonical_public_url(record.get(field_name), site_root=site_root)
        if url:
            return url
    return workshop_url_from_event_id(record.get("event_id"), virtual_url=virtual_url)


def _public_url_from_local_artifact_path(value: object, *, virtual_url: str = "") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    poster_match = re.search(r"(?:^|/)media_asset/poster-(\d+)(?:/|$)", raw)
    if poster_match and virtual_url:
        return f"{virtual_url.rstrip('/')}/poster/{poster_match.group(1)}"
    workshop_paper_match = re.search(r"(?:^|/)workshops/papers/[^/]+/([A-Za-z0-9_-]+)\.pdf$", raw)
    if workshop_paper_match:
        return f"https://openreview.net/forum?id={workshop_paper_match.group(1)}"
    return ""


def _source_id_variants(source_id: object) -> list[str]:
    raw = str(source_id or "").strip()
    if not raw:
        return []
    variants = [raw, raw.lower()]
    if raw.startswith("speech_transcript:"):
        parts = raw.split(":")
        if len(parts) >= 2:
            base = ":".join(parts[:2])
            variants.extend([base, base.lower()])
    if ":" in raw:
        suffix = raw.rsplit(":", 1)[-1].strip()
        if suffix:
            variants.extend([suffix, suffix.lower()])
    if raw.startswith("paper_abstract:"):
        suffix = raw.split(":", 1)[1].strip()
        variants.extend([f"openreview:{suffix}", f"paper:{suffix}"])
    if raw.startswith("workshop_paper:"):
        suffix = raw.split(":", 1)[1].strip()
        variants.extend([suffix, suffix.lower(), f"openreview:{suffix}", f"paper:{suffix}"])
    return list(dict.fromkeys(variants))


def _text_candidates(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("title", "name", "detail", "note", "contribution_note", "relevance_rationale"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    for key in ("works", "systems", "datasets"):
        raw = item.get(key)
        if isinstance(raw, list):
            values.extend(str(value) for value in raw if str(value).strip())
    return values


def _workshop_aliases(title: str) -> list[str]:
    aliases = [title]
    stripped = re.sub(r"^\s*the\s+\d+(?:st|nd|rd|th)?\s+workshop\s+on\s+", "", title, flags=re.I)
    if stripped != title:
        aliases.append(stripped)
    stripped = re.sub(r"^\s*workshop\s+on\s+", "", title, flags=re.I)
    if stripped != title:
        aliases.append(stripped)
    if ":" in title:
        aliases.append(title.split(":", 1)[0])
        aliases.append(title.split(":", 1)[1])
    lowered = title.lower()
    if "accelerated materials design" in lowered or "ai4mat" in lowered:
        aliases.extend(["Materials Design workshop", "AI4MAT workshop", "Accelerated Materials Design"])
    if "foundation models for science" in lowered:
        aliases.extend(["AI4Science workshop", "Foundation Models for Science"])
    return [alias.strip() for alias in aliases if alias and alias.strip()]


@dataclass
class SourceLinkResolver:
    """Workspace-local resolver for public source URLs."""

    workspace: Path
    site_root: str = ""
    virtual_url: str = ""
    by_source_id: dict[str, str] = field(default_factory=dict)
    by_title: dict[str, str] = field(default_factory=dict)
    by_local_path: dict[str, str] = field(default_factory=dict)
    title_index: list[tuple[str, str]] = field(default_factory=list)
    # Poster program pages: poster item id -> {"title", "paper_url", "poster_url"}.
    # A poster always corresponds to a paper; the cached virtual-site poster page
    # names the paper and links its OpenReview forum, so one source can carry
    # BOTH a paper link and a poster link.
    poster_pages: dict[str, dict[str, str]] = field(default_factory=dict)
    poster_id_by_source_id: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_workspace(cls, workspace: Path) -> "SourceLinkResolver":
        resolved = workspace.expanduser().resolve()
        site_root = ""
        virtual_url = ""
        try:
            from conference_pipeline.conference_identity import load_conference_identity

            identity = load_conference_identity(resolved)
            site_root = identity.site_root()
            virtual_url = identity.virtual_url
        except Exception:  # noqa: BLE001 - identity is best-effort for link rehydration
            pass
        resolver = cls(workspace=resolved, site_root=site_root, virtual_url=virtual_url)
        resolver._load_openreview_papers()
        resolver._load_workshop_inventory()
        resolver._load_extraction_source_links()
        resolver._load_program_pages()
        resolver.title_index = sorted(
            resolver.by_title.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        return resolver

    def resolve_item(self, item: dict[str, Any]) -> str:
        for field_name in URL_FIELDS:
            url = canonical_public_url(item.get(field_name), site_root=self.site_root)
            if url:
                return url
        for source_id in _source_id_variants(item.get("source_id") or item.get("id") or item.get("forum")):
            url = self.by_source_id.get(source_id) or self.by_source_id.get(source_id.lower())
            if url:
                return url
        for text in _text_candidates(item):
            url = self.resolve_title(text)
            if url:
                return url
        combined = " ".join(_text_candidates(item))
        return self.resolve_title(combined)

    def resolve_links(self, item: dict[str, Any]) -> dict[str, str]:
        """Every public link for one source: ``paper`` (OpenReview/forum/PDF),
        ``poster`` (virtual-site poster page), and ``primary`` (the single link
        ``resolve_item`` would choose). Keys are present only when resolved."""

        links: dict[str, str] = {}
        primary = self.resolve_item(item)
        poster_id = self._poster_id_for_item(item, primary_url=primary)
        poster_page = self.poster_pages.get(poster_id, {}) if poster_id else {}
        if poster_id:
            poster_url = poster_page.get("poster_url") or (
                f"{self.virtual_url.rstrip('/')}/poster/{poster_id}" if self.virtual_url else ""
            )
            if poster_url:
                links["poster"] = poster_url
        paper_url = ""
        for field_name in ("openreview_url", "forum_url", "pdf_url"):
            paper_url = canonical_public_url(item.get(field_name), site_root=self.site_root)
            if paper_url:
                break
        if not paper_url:
            paper_url = poster_page.get("paper_url", "")
        if not paper_url and primary and _is_paper_url(primary):
            paper_url = primary
        if not paper_url and poster_page.get("title"):
            candidate = self.resolve_title(poster_page["title"])
            if candidate and _is_paper_url(candidate):
                paper_url = candidate
        if paper_url:
            links["paper"] = paper_url
        if primary:
            links["primary"] = primary
        return links

    def _poster_id_for_item(self, item: dict[str, Any], *, primary_url: str = "") -> str:
        for source_id in _source_id_variants(item.get("source_id") or item.get("id") or item.get("forum")):
            poster_id = self.poster_id_by_source_id.get(source_id) or self.poster_id_by_source_id.get(source_id.lower())
            if poster_id:
                return poster_id
        for key in ("source_artifact_path", "local_path", "path", "artifact_path"):
            match = POSTER_PATH_RE.search(str(item.get(key) or ""))
            if match:
                return match.group(1)
        for url in (primary_url, *(str(item.get(key) or "") for key in URL_FIELDS)):
            match = POSTER_URL_RE.search(url)
            if match:
                return match.group(1)
        return ""

    def resolve_title(self, title: object) -> str:
        norm = _norm_title(title)
        if not norm:
            return ""
        direct = self.by_title.get(norm)
        if direct:
            return direct
        if len(norm) < 16:
            return ""
        norm_tokens = set(norm.split())
        for title_norm, url in self.title_index:
            if len(title_norm) < 16:
                continue
            if norm in title_norm or title_norm in norm:
                return url
            title_tokens = set(title_norm.split())
            if not norm_tokens or not title_tokens:
                continue
            overlap = len(norm_tokens & title_tokens) / max(min(len(norm_tokens), len(title_tokens)), 1)
            if overlap >= 0.72 and len(norm_tokens & title_tokens) >= 3:
                return url
        return ""

    def _record_source_id(self, source_id: object, url: str) -> None:
        if not url:
            return
        for variant in _source_id_variants(source_id):
            self.by_source_id.setdefault(variant, url)

    def _record_title(self, title: object, url: str) -> None:
        if not url:
            return
        norm = _norm_title(title)
        if len(norm) >= 4:
            self.by_title.setdefault(norm, url)

    def _load_openreview_papers(self) -> None:
        path = self.workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json"
        payload = _read_json(path)
        papers = payload.get("papers") if isinstance(payload, dict) else payload
        if not isinstance(papers, list):
            return
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            url = (
                canonical_public_url(paper.get("forum_url"), site_root=self.site_root)
                or canonical_public_url(paper.get("openreview_url"), site_root=self.site_root)
                or canonical_public_url(paper.get("source_url"), site_root=self.site_root)
                or canonical_public_url(paper.get("pdf_url"), site_root=self.site_root)
            )
            if not url:
                continue
            paper_id = paper.get("id") or paper.get("forum")
            if not paper_id:
                match = re.search(r"[?&]id=([^&#]+)", url)
                paper_id = match.group(1) if match else ""
            self._record_source_id(paper_id, url)
            if paper_id:
                self._record_source_id(f"paper_abstract:{paper_id}", url)
                self._record_source_id(f"openreview:{paper_id}", url)
                self._record_source_id(f"paper:{paper_id}", url)
            self._record_title(paper.get("title"), url)

    def _load_workshop_inventory(self) -> None:
        for path in (
            self.workspace / "storage" / "workshops" / "talk_media_inventory.json",
            self.workspace / "storage" / "workshops" / "inventory.json",
        ):
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            workshops = payload.get("workshops") if isinstance(payload.get("workshops"), list) else []
            records = payload.get("records") if isinstance(payload.get("records"), list) else []
            for workshop in workshops:
                if not isinstance(workshop, dict):
                    continue
                url = _first_public_url(workshop, site_root=self.site_root, virtual_url=self.virtual_url)
                title = str(workshop.get("workshop_title") or workshop.get("title") or workshop.get("name") or "").strip()
                for alias in _workshop_aliases(title):
                    self._record_title(alias, url)
                self._record_source_id(workshop.get("workshop_id") or workshop.get("venueid") or workshop.get("event_id"), url)
                for note in workshop.get("sample_notes") or []:
                    if isinstance(note, dict):
                        self._record_workshop_note(note)
                for record in workshop.get("records") or []:
                    if isinstance(record, dict):
                        self._record_media_record(record)
            for record in records:
                if isinstance(record, dict):
                    self._record_media_record(record)

    def _record_media_record(self, record: dict[str, Any]) -> None:
        url = (
            canonical_public_url(record.get("source_url"), site_root=self.site_root)
            or canonical_public_url(record.get("media_url"), site_root=self.site_root)
            or workshop_url_from_event_id(record.get("event_id"), virtual_url=self.virtual_url)
        )
        if not url:
            return
        local_path = str(record.get("local_path") or "").strip()
        if local_path:
            self.by_local_path.setdefault(local_path, url)
        workshop_title = str(record.get("workshop_title") or "").strip()
        for alias in _workshop_aliases(workshop_title):
            self._record_title(alias, url)
        talk_title = str(record.get("talk_title") or "").strip()
        if talk_title and talk_title.lower() != "slideslive embedded media":
            self._record_title(talk_title, url)
            speaker = str(record.get("talk_speaker") or "").strip()
            if speaker:
                self._record_title(f"{speaker} {talk_title}", url)
                self._record_title(f"{speaker} - {talk_title}", url)
        self._record_source_id(record.get("talk_id") or record.get("item_id"), url)

    def _record_talk_metadata(self, record: dict[str, Any], url: str) -> None:
        if not url:
            return
        talk_title = str(record.get("talk_title") or record.get("resolved_title") or record.get("title") or "").strip()
        speaker = str(record.get("talk_speaker") or record.get("speaker") or "").strip()
        workshop_title = str(record.get("workshop_title") or "").strip()
        if workshop_title:
            for alias in _workshop_aliases(workshop_title):
                self._record_title(alias, url)
        if talk_title and talk_title.lower() != "slideslive embedded media":
            self._record_title(talk_title, url)
            if speaker:
                self._record_title(f"{speaker} {talk_title}", url)
                self._record_title(f"{speaker} - {talk_title}", url)
                self._record_title(f"{speaker} keynote", url)
                self._record_title(f"{speaker} talk", url)

    def _record_extraction_record(self, record: dict[str, Any]) -> None:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        merged = {**record, **metadata}
        local_path = str(
            merged.get("source_artifact_path")
            or merged.get("local_path")
            or merged.get("source_path")
            or ""
        ).strip()
        url = _first_public_url(merged, site_root=self.site_root, virtual_url=self.virtual_url)
        if not url and local_path:
            url = self.by_local_path.get(local_path, "") or _public_url_from_local_artifact_path(local_path, virtual_url=self.virtual_url)
        if not url:
            return
        if local_path:
            self.by_local_path.setdefault(local_path, url)
        for key in ("artifact_id", "source_artifact_id", "source_id", "talk_id", "workshop_id"):
            self._record_source_id(merged.get(key), url)
        self._record_talk_metadata(merged, url)

    def _record_workshop_note(self, note: dict[str, Any]) -> None:
        note_id = str(note.get("note_id") or "").strip()
        if not note_id:
            return
        url = f"https://openreview.net/forum?id={note_id}"
        self._record_source_id(note_id, url)
        self._record_source_id(f"workshop_paper:{note_id}", url)
        self._record_source_id(f"paper:{note_id}", url)
        self._record_title(note.get("title"), url)

    def _load_extraction_source_links(self) -> None:
        root = self.workspace / "storage" / "extractions"
        for pattern in ("**/by-topic/*.json", "**/by-unit/*.json"):
            for path in root.glob(pattern):
                payload = _read_json(path)
                if payload is None:
                    continue
                self._walk_extraction_payload(payload)

    def _walk_extraction_payload(self, value: Any) -> None:
        if isinstance(value, dict):
            self._record_extraction_record(value)
            local_path = str(value.get("source_artifact_path") or value.get("local_path") or "").strip()
            url = (
                self.by_local_path.get(local_path, "")
                or _public_url_from_local_artifact_path(local_path, virtual_url=self.virtual_url)
                if local_path
                else ""
            )
            if url:
                for key in ("artifact_id", "source_artifact_id", "source_id"):
                    self._record_source_id(value.get(key), url)
            poster_match = POSTER_PATH_RE.search(local_path) if local_path else None
            if poster_match:
                for key in ("artifact_id", "source_artifact_id", "source_id", "unit_id"):
                    for variant in _source_id_variants(value.get(key)):
                        self.poster_id_by_source_id.setdefault(variant, poster_match.group(1))
            for nested in value.values():
                self._walk_extraction_payload(nested)
        elif isinstance(value, list):
            for item in value:
                self._walk_extraction_payload(item)

    def _load_program_pages(self) -> None:
        cache_root = self.workspace / "cache" / "acquisition-downloads"
        files = []
        files.extend(sorted((cache_root / "source-pages").glob("*.html")))
        files.extend(sorted((cache_root / "item-page-probes").glob("*.html")))
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            self._record_poster_page(path, text)
            for href, body in ANCHOR_RE.findall(text):
                url = canonical_public_url(href, site_root=self.site_root)
                if not url:
                    continue
                title = _clean_text(body)
                if title:
                    self._record_title(title, url)

    def _record_poster_page(self, path: Path, text: str) -> None:
        """Index a cached virtual-site poster page: the paper title it names and
        the OpenReview forum it links, keyed by the poster item id in the file
        name (``virtual-2026-poster-63306-<hash>.html``)."""

        match = POSTER_PROBE_FILENAME_RE.search(path.name)
        if not match:
            return
        poster_id = match.group(1)
        title_match = HTML_TITLE_RE.search(text)
        title = _clean_text(title_match.group(1)) if title_match else ""
        title = PROGRAM_TITLE_PREFIX_RE.sub("", title).strip()
        paper_url = ""
        paper_match = OPENREVIEW_FORUM_RE.search(text)
        if paper_match:
            paper_url = canonical_public_url(
                f"https://openreview.net/forum?id={paper_match.group(1)}", site_root=self.site_root
            )
        poster_url = f"{self.virtual_url.rstrip('/')}/poster/{poster_id}" if self.virtual_url else ""
        record = {"title": title, "paper_url": paper_url, "poster_url": poster_url}
        self.poster_pages.setdefault(poster_id, record)
        if title:
            self._record_title(title, paper_url or poster_url)
        if paper_url:
            self._record_source_id(f"poster:{poster_id}", paper_url)
        elif poster_url:
            self._record_source_id(f"poster:{poster_id}", poster_url)
