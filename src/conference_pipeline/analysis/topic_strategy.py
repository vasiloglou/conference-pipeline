"""Title-spine topic strategy generation for conference analysis.

The topic strategy stage collects conference program titles, asks multiple
frontier models for independent 12-topic taxonomies, and merges those
recommendations into the final analysis topic plan.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.conference_identity import (
    ConferenceIdentityError,
    load_conference_identity,
)
from conference_pipeline.program_coverage import classify_item_page_title_kind, item_page_title
from conference_pipeline.model_governance import (
    default_model,
    default_model_for_provider,
    default_provider,
)
from conference_pipeline.model_governance.model_families import (
    anthropic_rejects_temperature,
    uses_openai_reasoning_params,
)
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    gemini_generate_content,
    openai_chat_completions_create,
    provider_receipt_context,
)


SCHEMA_VERSION = 1
RECEIPT_TYPE = "conference_pipeline.topic_strategy_receipt"
TITLE_SPINE_RECEIPT_TYPE = "conference_pipeline.topic_strategy_title_spine"
DEFAULT_TOPIC_COUNT = 12
DEFAULT_MODEL_PROVIDERS = ("openai", "anthropic", "gemini")
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
MODEL_GOVERNANCE_TASK_ID = "analysis.topic_strategy_generation"
HTML_TITLE_RE = re.compile(
    r'<div[^>]+class="[^"]*(?:title-style|sessiontitle)[^"]*"[^>]*>\s*'
    r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.I | re.S,
)
DATA_EVENT_TITLE_RE = re.compile(r'data-event-title="(?P<title>[^"]+)"', re.I)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
FEATURED_WORKSHOP_RE = re.compile(
    r"\b(keynote|invited|oral|spotlight|plenary|tutorial|panel|opening|remarks)\b",
    re.I,
)
NOISE_TITLES = {
    "break",
    "registration desk",
    "social",
    "reception",
    "coffee break",
    "lunch",
    "mentorship session",
}


@dataclass(frozen=True)
class TitleRecord:
    title: str
    source_family: str
    source_path: str
    source_url: str | None = None
    speaker: str | None = None
    workshop_id: str | None = None
    workshop_title: str | None = None
    talk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @classmethod
    def parse(cls, raw: str) -> "ModelSpec":
        if ":" not in raw:
            raise ValueError(f"model spec must be provider:model, got {raw!r}")
        provider, model = raw.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in {"openai", "anthropic", "gemini"}:
            raise ValueError(f"unsupported provider in model spec {raw!r}")
        if not model:
            raise ValueError(f"missing model in spec {raw!r}")
        return cls(provider=provider, model=model)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def safe_key(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", self.key).strip("-")


@dataclass
class ModelRecommendation:
    model_spec: str
    status: str
    topics: list[dict[str, Any]]
    method_notes: list[str] = field(default_factory=list)
    coverage_warnings: list[str] = field(default_factory=list)
    raw_response_path: str | None = None
    parsed_response_path: str | None = None
    provider_receipt_path: str | None = None
    error: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_model_specs() -> tuple[str, ...]:
    return tuple(
        f"{provider}:{default_model_for_provider(MODEL_GOVERNANCE_TASK_ID, provider)}"
        for provider in DEFAULT_MODEL_PROVIDERS
    )


def default_merge_model_spec() -> str:
    return f"{default_provider(MODEL_GOVERNANCE_TASK_ID)}:{default_model(MODEL_GOVERNANCE_TASK_ID)}"


def _clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _slug(value: str, *, fallback: str = "topic") -> str:
    candidate = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return candidate[:72].strip("_") or fallback


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json_dumps(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        words = re.findall(r"\S+", text)
        return int(len(words) * 1.35)


def _record_source_path(workspace: Path, path: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def _append_record(records: list[TitleRecord], record: TitleRecord) -> None:
    title = _clean_text(record.title)
    if not title:
        return
    if _norm_key(title) in NOISE_TITLES:
        return
    records.append(
        TitleRecord(
            title=title,
            source_family=record.source_family,
            source_path=record.source_path,
            source_url=record.source_url,
            speaker=record.speaker,
            workshop_id=record.workshop_id,
            workshop_title=record.workshop_title,
            talk_id=record.talk_id,
            metadata=record.metadata,
        )
    )


def _collect_openreview_orals(workspace: Path, records: list[TitleRecord]) -> None:
    path = workspace / "storage/download-corpus/openreview_accepted_papers.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = payload.get("papers", []) if isinstance(payload, dict) else payload
    if not isinstance(papers, list):
        return
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = _clean_text(paper.get("title"))
        venue = _clean_text(paper.get("venue")).lower()
        if not title or "oral" not in venue:
            continue
        _append_record(
            records,
            TitleRecord(
                title=title,
                source_family="main_oral",
                source_path=_record_source_path(workspace, path),
                source_url=paper.get("forum_url") or paper.get("source_url"),
                metadata={
                    "venue": paper.get("venue"),
                    "primary_area": paper.get("primary_area"),
                    "keywords": paper.get("keywords") or [],
                    "paper_id": paper.get("id") or paper.get("forum"),
                },
            ),
        )


# Event kinds (as ``item_kind`` / talk maps name them) to title-spine families.
# The spine used to know only keynotes, sessions and workshops: tutorials were
# filed under sessions by filename and expo, affinity, town-hall and
# test-of-time events were dropped, so the ICML 2026 spine showed no tutorial
# family at all after the recovery that downloaded every tutorial recording.
EVENT_KIND_FAMILIES: dict[str, str] = {
    "keynote": "main_keynote",
    "invited": "main_keynote",
    "invited-talk": "main_keynote",
    "invited-speaker": "main_keynote",
    "tutorial": "main_tutorial",
    "expo": "expo",
    "session": "session_title",
    "oral": "session_title",
    "spotlight": "session_title",
    "workshop": "workshop_title",
    "affinity": "program_event",
    "affinity-event": "program_event",
    "town-hall": "program_event",
    "test-of-time": "program_event",
}
BARE_ITEM_HREF_RE = re.compile(r"/virtual/(?P<year>\d{4})/(?P<id>\d+)/?(?:[?#]|$)")
OVERFLOW_ROOM_RE = re.compile(r"\[\s*overflow\s+room\s*\]", re.I)


def _family_for_kind(kind: str | None) -> str | None:
    return EVENT_KIND_FAMILIES.get(str(kind or "").lower()) if kind else None


def _classify_html_title(
    path: Path,
    title: str,
    href: str | None,
    *,
    resolve_bare_id: Callable[[str], str | None] | None = None,
) -> str | None:
    name = path.name.lower()
    href_lower = (href or "").lower()
    title_lower = title.lower()
    if OVERFLOW_ROOM_RE.search(title):
        # The calendar lists an overflow-room screening of a keynote as a
        # second event with the same title; it is not a second talk.
        return None
    if "/workshop/" in href_lower:
        return "workshop_title"
    if "/invited-talk/" in href_lower or "/keynote/" in href_lower or "/invited-speaker/" in href_lower:
        return "main_keynote"
    if "/tutorial/" in href_lower:
        return "main_tutorial"
    if "/expo" in href_lower:
        return "expo"
    if "/affinity-event/" in href_lower or "/town-hall/" in href_lower or "/test-of-time" in href_lower:
        return "program_event"
    if "/session/" in href_lower or "session" in title_lower:
        return "session_title"
    if href:
        # A bare-ID link names no kind; the cached item page does.
        bare = BARE_ITEM_HREF_RE.search(href_lower)
        if bare and resolve_bare_id is not None:
            return _family_for_kind(resolve_bare_id(href))
        return None
    if "workshop" in name:
        return "workshop_title"
    if "invited" in name or "keynote" in name:
        return "main_keynote"
    if "tutorial" in name:
        return "main_tutorial"
    if "expo" in name:
        return "expo"
    if "panel" in name or "competition" in name or "conference-specific" in name:
        return "session_title"
    return None


def _bare_id_resolver(cache_root: Path) -> Callable[[str], str | None]:
    """Resolve a bare-ID item href to its kind from the cached item-page probe.

    Probes are cached as ``item-page-probes/virtual-<year>-<id>-<digest>.html``
    (``probe_page_destination`` in the conference runners).
    """

    probes = cache_root / "item-page-probes"

    def _resolve(href: str) -> str | None:
        match = BARE_ITEM_HREF_RE.search(href.lower())
        if not match or not probes.is_dir():
            return None
        for candidate in sorted(probes.glob(f"virtual-{match.group('year')}-{match.group('id')}-*.html")):
            kind = classify_item_page_title_kind(candidate.read_text(encoding="utf-8", errors="ignore"))
            if kind:
                return kind
        return None

    return _resolve


def _conference_site_root(workspace: Path) -> str:
    """Resolve ``scheme://host`` for the workspace conference (empty if unknown)."""
    try:
        return load_conference_identity(workspace).site_root()
    except ConferenceIdentityError:
        return ""


def _collect_cached_html_titles(workspace: Path, records: list[TitleRecord]) -> None:
    cache_root = workspace / "cache/acquisition-downloads"
    if not cache_root.exists():
        return
    site_root = _conference_site_root(workspace)
    resolve_bare_id = _bare_id_resolver(cache_root)
    candidates = list((cache_root / "source-pages").glob("*.html"))
    probe_root = cache_root / "item-page-probes"
    for extra in (
        "*invited*.html",
        "*keynote*.html",
        "*oral*.html",
        "*session*.html",
        "*tutorial*.html",
        "*expo*.html",
        "*affinity*.html",
        "*town-hall*.html",
        "*test-of-time*.html",
    ):
        candidates.extend(probe_root.glob(extra))
    # Bare-ID item pages (tutorials, expo items) have no typed slug in their
    # name; their own <title> says what they are.
    bare_probes = sorted(probe_root.glob("virtual-[0-9]*-[0-9]*-*.html")) if probe_root.is_dir() else []
    seen_paths: set[Path] = set()
    for path in bare_probes:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        family = _family_for_kind(classify_item_page_title_kind(text))
        title = _clean_text(item_page_title(text))
        if family is None or not title:
            continue
        _append_record(
            records,
            TitleRecord(title=title, source_family=family, source_path=_record_source_path(workspace, path)),
        )
    for path in sorted(candidates):
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in HTML_TITLE_RE.finditer(text):
            title = _clean_text(match.group("title"))
            href = html.unescape(match.group("href") or "")
            family = _classify_html_title(path, title, href, resolve_bare_id=resolve_bare_id)
            if family is None:
                continue
            if href.startswith("http") or not site_root:
                source_url = href
            else:
                source_url = f"{site_root}{href}"
            _append_record(
                records,
                TitleRecord(
                    title=title,
                    source_family=family,
                    source_path=_record_source_path(workspace, path),
                    source_url=source_url,
                ),
            )
        for match in DATA_EVENT_TITLE_RE.finditer(text):
            title = _clean_text(match.group("title"))
            family = _classify_html_title(path, title, None)
            if family is None:
                continue
            _append_record(
                records,
                TitleRecord(
                    title=title,
                    source_family=family,
                    source_path=_record_source_path(workspace, path),
                ),
            )


def _workshop_display_id(venueid: str) -> str:
    tail = venueid.rstrip("/").split("/")[-1]
    return tail or venueid


def _collect_workshop_inventory_titles(workspace: Path, records: list[TitleRecord]) -> None:
    for path in (
        workspace / "storage/workshops/talk_media_inventory.json",
        workspace / "storage/workshops/inventory.json",
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        workshops = payload.get("workshops", []) if isinstance(payload, dict) else []
        if not isinstance(workshops, list):
            continue
        for workshop in workshops:
            if not isinstance(workshop, dict):
                continue
            title = _clean_text(
                workshop.get("workshop_title")
                or workshop.get("title")
                or workshop.get("name")
                or _workshop_display_id(str(workshop.get("venueid") or ""))
            )
            venueid = str(workshop.get("workshop_id") or workshop.get("venueid") or "")
            if not title:
                continue
            _append_record(
                records,
                TitleRecord(
                    title=title,
                    source_family="workshop_title",
                    source_path=_record_source_path(workspace, path),
                    source_url=workshop.get("source_url") or workshop.get("url"),
                    workshop_id=venueid or None,
                    workshop_title=title,
                    metadata={
                        "paper_count": workshop.get("paper_count"),
                        "pdf_availability_class": workshop.get("pdf_availability_class"),
                    },
                ),
            )


def _collect_workshop_talk_titles(
    workspace: Path,
    records: list[TitleRecord],
    *,
    workshop_talk_filter: str,
) -> None:
    root = workspace / "storage/transcription-ocr/workshop-talks"
    if not root.exists():
        return
    talk_paths = {
        *root.glob("*/talks.json"),
        *root.glob("*_talks.json"),
    }
    for path in sorted(talk_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        talks = payload.get("talks", []) if isinstance(payload, dict) else []
        if not isinstance(talks, list):
            continue
        workshop_id = str(payload.get("workshop_id") or "") if isinstance(payload, dict) else ""
        workshop_title = _clean_text(payload.get("workshop_title")) if isinstance(payload, dict) else ""
        for talk in talks:
            if not isinstance(talk, dict):
                continue
            title = _clean_text(talk.get("talk_title") or talk.get("title"))
            if not title:
                continue
            description = _clean_text(talk.get("description"))
            featured = bool(FEATURED_WORKSHOP_RE.search(f"{title} {description}"))
            if workshop_talk_filter == "featured" and not featured:
                continue
            _append_record(
                records,
                TitleRecord(
                    title=title,
                    source_family="workshop_segmented_talk_featured"
                    if featured
                    else "workshop_segmented_talk",
                    source_path=_record_source_path(workspace, path),
                    speaker=_clean_text(talk.get("talk_speaker")) or None,
                    workshop_id=str(talk.get("workshop_id") or workshop_id) or None,
                    workshop_title=_clean_text(talk.get("workshop_title")) or workshop_title or None,
                    talk_id=str(talk.get("talk_id") or "") or None,
                    metadata={
                        "timestamp_start": talk.get("timestamp_start"),
                        "timestamp_end": talk.get("timestamp_end"),
                        "featured_hint": featured,
                    },
                ),
            )


def _collect_recording_talk_titles(workspace: Path, records: list[TitleRecord]) -> None:
    """Titles from talk maps over non-workshop recordings (keynotes, tutorials,
    sessions): ``storage/transcription-ocr/talk-maps/base/*_talks.json`` and
    every evidence lane's ``talks/*_talks.json``. The spine reads what was
    actually recorded and transcribed, not only what the program page listed;
    a talk whose kind is unknown is recorded as ``recorded_talk``."""

    roots = [workspace / "storage/transcription-ocr/talk-maps/base"]
    lanes_root = workspace / "storage/transcription-ocr/evidence-lanes"
    if lanes_root.is_dir():
        roots.extend(sorted(path / "talks" for path in lanes_root.iterdir() if (path / "talks").is_dir()))
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*_talks.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("workshop_association") == "event_url":
                continue  # workshop talks are collected from the workshop talk maps
            hints = payload.get("program_hints") if isinstance(payload.get("program_hints"), dict) else {}
            event_kind = hints.get("event_category") or payload.get("event_category")
            for talk in payload.get("talks") or []:
                if not isinstance(talk, dict):
                    continue
                title = _clean_text(talk.get("talk_title") or talk.get("title"))
                if not title or OVERFLOW_ROOM_RE.search(title):
                    continue
                family = _family_for_kind(talk.get("talk_kind")) or _family_for_kind(event_kind) or "recorded_talk"
                _append_record(
                    records,
                    TitleRecord(
                        title=title,
                        source_family=family,
                        source_path=_record_source_path(workspace, path),
                        speaker=_clean_text(talk.get("talk_speaker")) or None,
                        talk_id=str(talk.get("talk_id") or "") or None,
                        metadata={
                            "talk_kind": talk.get("talk_kind"),
                            "event_category": event_kind,
                            "timestamp_start": talk.get("timestamp_start"),
                            "timestamp_end": talk.get("timestamp_end"),
                        },
                    ),
                )


def _collect_workshop_paper_titles(workspace: Path, records: list[TitleRecord]) -> None:
    path = workspace / "storage/workshops/inventory.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    workshops = payload.get("workshops", []) if isinstance(payload, dict) else []
    if not isinstance(workshops, list):
        return
    for workshop in workshops:
        if not isinstance(workshop, dict):
            continue
        venueid = str(workshop.get("venueid") or "")
        workshop_title = _clean_text(
            workshop.get("workshop_title")
            or workshop.get("title")
            or _workshop_display_id(venueid)
        )
        notes = workshop.get("sample_notes") or []
        if not isinstance(notes, list):
            continue
        for note in notes:
            if not isinstance(note, dict):
                continue
            tier = _clean_text(note.get("tier")).lower()
            if tier and not any(token in tier for token in ("oral", "spotlight", "keynote", "invited")):
                continue
            title = _clean_text(note.get("title"))
            if not title:
                continue
            _append_record(
                records,
                TitleRecord(
                    title=title,
                    source_family="workshop_oral_paper",
                    source_path=_record_source_path(workspace, path),
                    source_url=note.get("forum_url") or note.get("url"),
                    workshop_id=venueid or None,
                    workshop_title=workshop_title or None,
                    metadata={"tier": note.get("tier"), "note_id": note.get("note_id")},
                ),
            )


def collect_title_spine(
    workspace: Path | str,
    *,
    workshop_talk_filter: str = "all",
    include_workshop_paper_titles: bool = False,
) -> tuple[list[TitleRecord], dict[str, Any]]:
    workspace = Path(workspace)
    records: list[TitleRecord] = []
    _collect_openreview_orals(workspace, records)
    _collect_cached_html_titles(workspace, records)
    _collect_workshop_inventory_titles(workspace, records)
    _collect_workshop_talk_titles(workspace, records, workshop_talk_filter=workshop_talk_filter)
    _collect_recording_talk_titles(workspace, records)
    if include_workshop_paper_titles:
        _collect_workshop_paper_titles(workspace, records)

    deduped: list[TitleRecord] = []
    seen: set[tuple[str, str, str | None]] = set()
    for record in records:
        key_workshop = (
            record.workshop_id
            if record.source_family.startswith("workshop_segmented_talk")
            else None
        )
        key = (_norm_key(record.title), record.source_family, key_workshop)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    family_counts = Counter(record.source_family for record in deduped)
    title_text = "\n".join(record.title for record in deduped)
    summary = {
        "workspace": str(workspace),
        "title_count": len(deduped),
        "source_family_counts": dict(sorted(family_counts.items())),
        "estimated_title_tokens": estimate_tokens(title_text),
        "workshop_talk_filter": workshop_talk_filter,
        "include_workshop_paper_titles": include_workshop_paper_titles,
    }
    return deduped, summary


def build_title_spine_payload(records: list[TitleRecord], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": TITLE_SPINE_RECEIPT_TYPE,
        "generated_at": utc_now(),
        "summary": summary,
        "records": [asdict(record) for record in records],
    }


def render_recommendation_prompt(
    *,
    conference_label: str,
    title_spine: dict[str, Any],
    topic_count: int,
) -> str:
    compact_records = [
        {
            "title": record["title"],
            "source_family": record["source_family"],
            "speaker": record.get("speaker"),
            "workshop_title": record.get("workshop_title"),
            "metadata": {
                key: value
                for key, value in (record.get("metadata") or {}).items()
                if key in {"primary_area", "keywords", "featured_hint"}
            },
        }
        for record in title_spine["records"]
    ]
    payload = {
        "conference": conference_label,
        "title_spine_summary": title_spine["summary"],
        "title_records": compact_records,
    }
    return textwrap.dedent(
        f"""
        You are designing the analysis taxonomy for {conference_label}.

        Use only the supplied title spine: invited/keynote titles, main oral
        paper titles, official session titles, workshop titles, and segmented
        workshop talk titles. Propose exactly {topic_count} analysis topics.
        The topics should be conference-level lenses: broad enough to organize
        evidence across papers and talks, specific enough to guide retrieval,
        deck planning, podcast planning, and deep narrative analysis.

        Return strict JSON only with this schema:
        {{
          "topics": [
            {{
              "rank": 1,
              "topic_id": "01_short_slug",
              "title": "Topic title",
              "rationale": "Why this is one of the {topic_count} topics.",
              "evidence_title_patterns": ["phrases seen in the title spine"],
              "source_family_coverage": {{"main_oral": 0, "main_keynote": 0, "session_title": 0, "workshop_title": 0, "workshop_segmented_talk": 0}},
              "starter_queries": ["query for retrieval and GPT-Researcher"]
            }}
          ],
          "method_notes": ["brief note on how you clustered the titles"],
          "coverage_warnings": ["important gaps or ambiguities"]
        }}

        Title spine JSON:
        {_json_dumps(payload)}
        """
    ).strip()


def render_merge_prompt(
    *,
    conference_label: str,
    title_spine_summary: dict[str, Any],
    recommendations: list[ModelRecommendation],
    topic_count: int,
) -> str:
    payload = {
        "conference": conference_label,
        "title_spine_summary": title_spine_summary,
        "recommendations": [
            {
                "model_spec": rec.model_spec,
                "topics": rec.topics,
                "method_notes": rec.method_notes,
                "coverage_warnings": rec.coverage_warnings,
            }
            for rec in recommendations
            if rec.status == "succeeded"
        ],
    }
    # Registry-backed example spec so no current model id is hardcoded here.
    example_specs = [rec.model_spec for rec in recommendations if rec.status == "succeeded"]
    example_spec = example_specs[0] if example_specs else default_merge_model_spec()
    return textwrap.dedent(
        f"""
        Merge the independent model recommendations into the final {topic_count}
        topic analysis strategy for {conference_label}.

        Requirements:
        - Return exactly {topic_count} topics.
        - Preserve high-consensus themes, but merge duplicate or overly narrow
          variants.
        - Include topics that cover workshop-only signals if they are
          substantively distinct.
        - Make titles concise and presentation-ready.
        - Provide retrieval starter queries and source-family coverage notes.
        - Return strict JSON only.

        JSON schema:
        {{
          "final_topics": [
            {{
              "rank": 1,
              "topic_id": "01_short_slug",
              "title": "Final topic title",
              "rationale": "Why this topic belongs in the final taxonomy.",
              "consensus_sources": ["{example_spec}"],
              "source_family_coverage": {{"main_oral": 0, "main_keynote": 0, "session_title": 0, "workshop_title": 0, "workshop_segmented_talk": 0}},
              "starter_queries": ["query for retrieval and GPT-Researcher"]
            }}
          ],
          "merge_notes": ["how conflicts were resolved"],
          "coverage_warnings": ["residual gaps"]
        }}

        Recommendation payload:
        {_json_dumps(payload)}
        """
    ).strip()


def _parse_json_response(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        # Some providers (observed: Gemini) occasionally emit a complete,
        # well-formed JSON document followed by trailing garbage (a stray
        # duplicated closing brace/bracket or restated commentary). Recover
        # by decoding only the first complete JSON value and requiring the
        # remainder be non-JSON trailing noise rather than a second payload.
        if exc.msg != "Extra data":
            raise
        decoder = json.JSONDecoder()
        parsed, end = decoder.raw_decode(candidate)
        if not isinstance(parsed, dict):
            raise
        return parsed


def _normalize_topics(topics: list[dict[str, Any]], *, topic_count: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, topic in enumerate(topics[:topic_count], start=1):
        title = _clean_text(topic.get("title") or f"Topic {index}")
        topic_id = _clean_text(topic.get("topic_id") or "")
        if not topic_id or not re.match(r"^\d{2}_", topic_id):
            topic_id = f"{index:02d}_{_slug(title)}"
        normalized.append(
            {
                "rank": int(topic.get("rank") or index),
                "topic_id": topic_id,
                "title": title,
                "rationale": _clean_text(topic.get("rationale")),
                "evidence_title_patterns": list(topic.get("evidence_title_patterns") or []),
                "consensus_sources": list(topic.get("consensus_sources") or []),
                "source_family_coverage": dict(topic.get("source_family_coverage") or {}),
                "starter_queries": list(topic.get("starter_queries") or []),
            }
        )
    return normalized


def _mock_topics(provider_label: str, *, topic_count: int) -> dict[str, Any]:
    base = [
        "Reasoning, post-training, and test-time adaptation",
        "Agentic systems, tool use, and autonomous workflows",
        "Multi-agent learning, cooperation, markets, and social simulation",
        "Multimodal, spatial, audio-video, and retrieval-augmented models",
        "Generative models for images, video, 3D, flows, and world models",
        "AI for science: biology, molecules, materials, physics, and open science",
        "Earth, climate, geospatial, remote sensing, and scarce-label learning",
        "Safety, privacy, trustworthiness, provenance, and alignment",
        "Interpretability, causality, representations, and alignment of internal structure",
        "Efficient architectures, inference, serving, memory, and systems",
        "Theory, optimization, scaling laws, probabilistic methods, and learning dynamics",
        "Human-centered, embodied, robotic, developmental, and neurocognitive intelligence",
    ]
    topics = []
    for idx, title in enumerate(base[:topic_count], start=1):
        topics.append(
            {
                "rank": idx,
                "topic_id": f"{idx:02d}_{_slug(title)}",
                "title": title,
                "rationale": f"Mock {provider_label} recommendation based on recurring title-spine signals.",
                "evidence_title_patterns": title.split(",")[:2],
                "source_family_coverage": {},
                "starter_queries": [title],
            }
        )
    return {
        "topics": topics,
        "method_notes": [f"Mock deterministic recommendation for {provider_label}."],
        "coverage_warnings": [],
    }


def _mock_merge(recommendations: list[ModelRecommendation], *, topic_count: int) -> dict[str, Any]:
    source = next((rec for rec in recommendations if rec.topics), None)
    topics = _normalize_topics(source.topics if source else _mock_topics("merge", topic_count=topic_count)["topics"], topic_count=topic_count)
    final_topics = []
    consensus = [rec.model_spec for rec in recommendations if rec.status == "succeeded"]
    for topic in topics:
        topic["consensus_sources"] = consensus
        final_topics.append(topic)
    return {
        "final_topics": final_topics,
        "merge_notes": ["Mock merge preserved the deterministic consensus topic set."],
        "coverage_warnings": [],
    }


def _openai_options(model: str, max_completion_tokens: int) -> dict[str, Any]:
    if uses_openai_reasoning_params(model):
        return {"max_completion_tokens": max_completion_tokens}
    return {"max_tokens": max_completion_tokens, "temperature": 0.2}


def _anthropic_options(model: str, max_output_tokens: int) -> dict[str, Any]:
    options: dict[str, Any] = {"max_tokens": max_output_tokens}
    if anthropic_rejects_temperature(model):
        return options
    options["temperature"] = 0.2
    return options


class _AnthropicRestMessages:
    def __init__(self, client: "_AnthropicRestClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> "_AnthropicRestResponse":
        return self._client.create_message(**kwargs)


class _AnthropicRestResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.id = payload.get("id")
        self.content = payload.get("content") or []
        self.usage = payload.get("usage") or {}

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]


class _AnthropicRestClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.messages = _AnthropicRestMessages(self)

    def create_message(self, **kwargs: Any) -> _AnthropicRestResponse:
        data = json.dumps(kwargs).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            method="POST",
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic HTTP {exc.code}: {body[:500]}") from exc
        return _AnthropicRestResponse(payload)


def _response_text(provider: str, response: Any) -> str:
    if provider == "openai":
        return str(response.choices[0].message.content)
    if provider == "gemini":
        return str(getattr(response, "text", "") or "")
    if provider == "anthropic":
        chunks = getattr(response, "content", []) or []
        parts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text") or ""))
            else:
                parts.append(str(getattr(chunk, "text", "") or ""))
        return "".join(parts)
    raise ValueError(f"unsupported provider: {provider}")


def _call_model(
    *,
    spec: ModelSpec,
    prompt: str,
    system: str,
    workspace: Path,
    run_id: str,
    stage_id: str,
    max_output_tokens: int,
) -> tuple[str, str | None]:
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=stage_id,
        caller=__file__,
    ):
        if spec.provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            result = openai_chat_completions_create(
                client,
                caller=__file__,
                model=spec.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                **_openai_options(spec.model, max_output_tokens),
            )
        elif spec.provider == "gemini":
            from google import genai
            from google.genai import types

            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.2,
                response_mime_type="application/json",
                system_instruction=system,
            )
            result = gemini_generate_content(
                client,
                caller=__file__,
                model=spec.model,
                contents=prompt,
                config=config,
            )
        elif spec.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic topic strategy calls")
            client = _AnthropicRestClient(api_key=api_key)
            result = anthropic_messages_create(
                client,
                caller=__file__,
                model=spec.model,
                **_anthropic_options(spec.model, max_output_tokens),
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise ValueError(f"unsupported provider: {spec.provider}")
    return _response_text(spec.provider, result.response), result.receipt.receipt_path


def _env_file_candidates(workspace: Path) -> list[Path]:
    repo_root = Path.cwd()
    return [
        DEFAULT_ENV_FILE,
        repo_root / ".env",
        repo_root.parent / "neurips-foundation-conf-pipeline" / ".env",
        workspace.parent.parent / ".env" if len(workspace.parents) > 1 else workspace / ".env",
    ]


def _load_env_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        loaded.append(key)
    return {"path": str(path), "status": "loaded", "keys_loaded": sorted(loaded)}


def ensure_live_environment(
    *,
    workspace: Path,
    model_specs: list[ModelSpec],
    env_file: Path | None,
) -> dict[str, Any]:
    env_status: dict[str, Any]
    if env_file:
        env_status = _load_env_file(env_file)
    else:
        env_status = {"status": "not_loaded"}
        needed = required_env_keys(model_specs)
        if any(not os.environ.get(key) for key in needed):
            for candidate in _env_file_candidates(workspace):
                if candidate.exists():
                    env_status = _load_env_file(candidate)
                    break
    missing = [key for key in required_env_keys(model_specs) if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            "missing provider environment variables for live topic strategy: "
            + ", ".join(sorted(missing))
        )
    return env_status


def required_env_keys(model_specs: list[ModelSpec]) -> set[str]:
    keys: set[str] = set()
    for spec in model_specs:
        if spec.provider == "openai":
            keys.add("OPENAI_API_KEY")
        elif spec.provider == "anthropic":
            keys.add("ANTHROPIC_API_KEY")
        elif spec.provider == "gemini":
            if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
                keys.add("GOOGLE_API_KEY")
    return keys


def _approval_is_present() -> bool:
    return os.environ.get("TOPIC_STRATEGY_PROVIDER_COST_APPROVED", "").lower() in {
        "1",
        "true",
        "yes",
        "approved",
    }


def generate_model_recommendations(
    *,
    mode: str,
    model_specs: list[ModelSpec],
    prompt: str,
    system: str,
    workspace: Path,
    run_id: str,
    output_dir: Path,
    topic_count: int,
    max_output_tokens: int,
) -> list[ModelRecommendation]:
    recommendations: list[ModelRecommendation] = []
    for spec in model_specs:
        raw_path = output_dir / f"{spec.safe_key}_raw.txt"
        parsed_path = output_dir / f"{spec.safe_key}_recommendation.json"
        provider_receipt_path: str | None = None
        if mode == "mock":
            parsed = _mock_topics(spec.key, topic_count=topic_count)
            _atomic_write_text(raw_path, _json_dumps(parsed) + "\n")
            _atomic_write_json(parsed_path, parsed)
            recommendations.append(
                ModelRecommendation(
                    model_spec=spec.key,
                    status="succeeded",
                    topics=_normalize_topics(parsed.get("topics", []), topic_count=topic_count),
                    method_notes=list(parsed.get("method_notes") or []),
                    coverage_warnings=list(parsed.get("coverage_warnings") or []),
                    raw_response_path=str(raw_path),
                    parsed_response_path=str(parsed_path),
                )
            )
            continue
        try:
            raw, provider_receipt_path = _call_model(
                spec=spec,
                prompt=prompt,
                system=system,
                workspace=workspace,
                run_id=run_id,
                stage_id=f"topic-strategy-{spec.safe_key}",
                max_output_tokens=max_output_tokens,
            )
            _atomic_write_text(raw_path, raw)
            parsed = _parse_json_response(raw)
            _atomic_write_json(parsed_path, parsed)
            topics = _normalize_topics(parsed.get("topics", []), topic_count=topic_count)
            if len(topics) != topic_count:
                raise RuntimeError(f"{spec.key} returned {len(topics)} topics, expected {topic_count}")
            recommendations.append(
                ModelRecommendation(
                    model_spec=spec.key,
                    status="succeeded",
                    topics=topics,
                    method_notes=list(parsed.get("method_notes") or []),
                    coverage_warnings=list(parsed.get("coverage_warnings") or []),
                    raw_response_path=str(raw_path),
                    parsed_response_path=str(parsed_path),
                    provider_receipt_path=provider_receipt_path,
                )
            )
        except Exception as exc:  # noqa: BLE001 - failures are recorded in the strategy receipt.
            recommendations.append(
                ModelRecommendation(
                    model_spec=spec.key,
                    status="failed",
                    topics=[],
                    raw_response_path=str(raw_path) if raw_path.exists() else None,
                    parsed_response_path=str(parsed_path) if parsed_path.exists() else None,
                    provider_receipt_path=provider_receipt_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return recommendations


def merge_recommendations(
    *,
    mode: str,
    merge_spec: ModelSpec,
    prompt: str,
    system: str,
    workspace: Path,
    run_id: str,
    output_dir: Path,
    recommendations: list[ModelRecommendation],
    topic_count: int,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str | None]:
    raw_path = output_dir / f"{merge_spec.safe_key}_merge_raw.txt"
    parsed_path = output_dir / f"{merge_spec.safe_key}_merge.json"
    if mode == "mock":
        parsed = _mock_merge(recommendations, topic_count=topic_count)
        _atomic_write_text(raw_path, _json_dumps(parsed) + "\n")
        _atomic_write_json(parsed_path, parsed)
        return parsed, None
    raw, provider_receipt_path = _call_model(
        spec=merge_spec,
        prompt=prompt,
        system=system,
        workspace=workspace,
        run_id=run_id,
        stage_id=f"topic-strategy-merge-{merge_spec.safe_key}",
        max_output_tokens=max_output_tokens,
    )
    _atomic_write_text(raw_path, raw)
    parsed = _parse_json_response(raw)
    _atomic_write_json(parsed_path, parsed)
    topics = _normalize_topics(parsed.get("final_topics", []), topic_count=topic_count)
    if len(topics) != topic_count:
        raise RuntimeError(f"merge model returned {len(topics)} topics, expected {topic_count}")
    parsed["final_topics"] = topics
    _atomic_write_json(parsed_path, parsed)
    return parsed, provider_receipt_path


def render_strategy_markdown(
    *,
    conference_label: str,
    result: dict[str, Any],
) -> str:
    lines = [
        f"# {conference_label} Topic Strategy",
        "",
        f"Generated at: {result['generated_at']}",
        f"Mode: {result['mode']}",
        f"Title spine: {result['title_spine']['summary']['title_count']} titles, "
        f"about {result['title_spine']['summary']['estimated_title_tokens']} tokens",
        "",
        "## Final Topics",
        "",
    ]
    for topic in result["final_topics"]:
        lines.extend(
            [
                f"### {topic['topic_id']}: {topic['title']}",
                "",
                topic.get("rationale") or "",
                "",
                "Starter queries:",
            ]
        )
        for query in topic.get("starter_queries") or []:
            lines.append(f"- {query}")
        lines.append("")
    if result.get("merge_notes"):
        lines.extend(["## Merge Notes", ""])
        lines.extend(f"- {note}" for note in result["merge_notes"])
        lines.append("")
    if result.get("coverage_warnings"):
        lines.extend(["## Coverage Warnings", ""])
        lines.extend(f"- {warning}" for warning in result["coverage_warnings"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_topic_strategy(
    *,
    workspace: Path | str,
    run_id: str,
    conference_label: str,
    mode: str,
    model_specs: list[str],
    merge_model: str,
    topic_count: int = DEFAULT_TOPIC_COUNT,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    workshop_talk_filter: str = "all",
    include_workshop_paper_titles: bool = False,
    max_output_tokens: int = 12000,
) -> dict[str, Any]:
    workspace = Path(workspace)
    if mode not in {"mock", "live"}:
        raise ValueError("mode must be mock or live")
    specs = [ModelSpec.parse(raw) for raw in model_specs]
    merge_spec = ModelSpec.parse(merge_model)
    all_live_specs = specs + [merge_spec]
    if mode == "live" and not (provider_cost_approved or _approval_is_present()):
        raise RuntimeError("--provider-cost-approved is required for live topic strategy calls")

    output_dir = workspace / "storage/analysis/topic-strategy" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    records, summary = collect_title_spine(
        workspace,
        workshop_talk_filter=workshop_talk_filter,
        include_workshop_paper_titles=include_workshop_paper_titles,
    )
    title_spine = build_title_spine_payload(records, summary)
    title_spine_path = output_dir / "title_spine.json"
    _atomic_write_json(title_spine_path, title_spine)
    title_spine_sha256 = hashlib.sha256(title_spine_path.read_bytes()).hexdigest()
    prompt = render_recommendation_prompt(
        conference_label=conference_label,
        title_spine=title_spine,
        topic_count=topic_count,
    )
    prompt_path = output_dir / "topic_strategy_prompt.txt"
    _atomic_write_text(prompt_path, prompt)
    env_file_status: dict[str, Any] = {"status": "not_required_for_mock"}
    if mode == "live":
        env_file_status = ensure_live_environment(
            workspace=workspace,
            model_specs=all_live_specs,
            env_file=env_file,
        )

    system = (
        "You are a senior conference-analysis editor. Return strict JSON only. "
        "Do not invent source titles beyond the supplied title spine."
    )
    recommendations = generate_model_recommendations(
        mode=mode,
        model_specs=specs,
        prompt=prompt,
        system=system,
        workspace=workspace,
        run_id=run_id,
        output_dir=output_dir,
        topic_count=topic_count,
        max_output_tokens=max_output_tokens,
    )
    succeeded = [rec for rec in recommendations if rec.status == "succeeded"]
    failed = [rec for rec in recommendations if rec.status != "succeeded"]
    if failed:
        raise RuntimeError(
            "topic strategy requires all recommendation models to succeed; failed: "
            + ", ".join(f"{rec.model_spec} ({rec.error})" for rec in failed)
        )
    merge_prompt = render_merge_prompt(
        conference_label=conference_label,
        title_spine_summary=summary,
        recommendations=succeeded,
        topic_count=topic_count,
    )
    merge_prompt_path = output_dir / "topic_strategy_merge_prompt.txt"
    _atomic_write_text(merge_prompt_path, merge_prompt)
    merge_payload, merge_provider_receipt_path = merge_recommendations(
        mode=mode,
        merge_spec=merge_spec,
        prompt=merge_prompt,
        system=system,
        workspace=workspace,
        run_id=run_id,
        output_dir=output_dir,
        recommendations=succeeded,
        topic_count=topic_count,
        max_output_tokens=max_output_tokens,
    )
    final_topics = _normalize_topics(merge_payload.get("final_topics", []), topic_count=topic_count)
    result = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "generated_at": utc_now(),
        "status": "succeeded",
        "mode": mode,
        "workspace": str(workspace),
        "run_id": run_id,
        "conference_label": conference_label,
        "model_governance_task_id": MODEL_GOVERNANCE_TASK_ID,
        "topic_count": topic_count,
        "model_specs": [spec.key for spec in specs],
        "merge_model": merge_spec.key,
        "title_spine": {
            "path": str(title_spine_path),
            "sha256": title_spine_sha256,
            "summary": summary,
            "prompt_path": str(prompt_path),
            "prompt_sha256": _sha256_text(prompt),
            "prompt_estimated_tokens": estimate_tokens(prompt),
        },
        "recommendations": [asdict(rec) for rec in recommendations],
        "final_topics": final_topics,
        "merge_notes": list(merge_payload.get("merge_notes") or []),
        "coverage_warnings": list(merge_payload.get("coverage_warnings") or []),
        "merge_provider_receipt_path": merge_provider_receipt_path,
        "provider_receipt_paths": [
            path
            for path in [rec.provider_receipt_path for rec in recommendations] + [merge_provider_receipt_path]
            if path
        ],
        "env_file_status": env_file_status,
        "secrets_copied": False,
    }
    analysis_dir = workspace / "analysis"
    strategy_json_path = analysis_dir / "topic_strategy.json"
    strategy_md_path = analysis_dir / "topic_strategy.md"
    result["analysis_topic_strategy_json"] = str(strategy_json_path)
    result["analysis_topic_strategy_md"] = str(strategy_md_path)
    _atomic_write_json(strategy_json_path, result)
    _atomic_write_text(strategy_md_path, render_strategy_markdown(conference_label=conference_label, result=result))
    receipt_path = output_dir / "topic_strategy_receipt.json"
    _atomic_write_json(receipt_path, result)
    result["receipt_path"] = str(receipt_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 12-topic strategy from conference title spine.")
    parser.add_argument("--workspace", required=True, help="Conference workspace, e.g. conferences/iclr2026.")
    parser.add_argument("--run-id", required=True, help="Workflow run id for receipts.")
    parser.add_argument("--conference-label", default=None, help="Human conference label.")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--model", action="append", dest="models", help="Provider model spec provider:model. Repeat three times.")
    parser.add_argument("--merge-model", default=None, help="Provider model spec for final merge.")
    parser.add_argument("--topic-count", type=int, default=DEFAULT_TOPIC_COUNT)
    parser.add_argument("--provider-cost-approved", action="store_true", help="Required for --mode live.")
    parser.add_argument("--env-file", type=Path, default=None, help="Optional .env file for live providers.")
    parser.add_argument(
        "--workshop-talk-filter",
        choices=["all", "featured"],
        default="all",
        help="Use all segmented workshop talks or only keynote/oral-like talks.",
    )
    parser.add_argument(
        "--include-workshop-paper-titles",
        action="store_true",
        help="Also include workshop OpenReview oral/spotlight/keynote paper titles when present.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--summary", action="store_true", help="Print compact JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    conference_label = args.conference_label or workspace.name.upper()
    models = args.models or list(default_model_specs())
    merge_model = args.merge_model or default_merge_model_spec()
    try:
        result = run_topic_strategy(
            workspace=workspace,
            run_id=args.run_id,
            conference_label=conference_label,
            mode=args.mode,
            model_specs=models,
            merge_model=merge_model,
            topic_count=args.topic_count,
            provider_cost_approved=args.provider_cost_approved,
            env_file=args.env_file,
            workshop_talk_filter=args.workshop_talk_filter,
            include_workshop_paper_titles=args.include_workshop_paper_titles,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must report a clear blocker.
        print(f"blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.summary:
        print(
            _json_dumps(
                {
                    "status": result["status"],
                    "mode": result["mode"],
                    "title_count": result["title_spine"]["summary"]["title_count"],
                    "estimated_title_tokens": result["title_spine"]["summary"]["estimated_title_tokens"],
                    "source_family_counts": result["title_spine"]["summary"]["source_family_counts"],
                    "final_topic_count": len(result["final_topics"]),
                    "receipt_path": result.get("receipt_path"),
                    "analysis_topic_strategy_json": result.get("analysis_topic_strategy_json"),
                }
            )
        )
    else:
        print(result.get("receipt_path", "topic strategy complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
