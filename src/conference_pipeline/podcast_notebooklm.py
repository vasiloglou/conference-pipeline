"""Per-deck NotebookLM podcast (Audio Overview) production.

One NotebookLM notebook per presentation deck (each topic deck plus each
executive deck), filled with a curated source set, steered by a
deterministic prompt, and rendered by Google NotebookLM's Audio Overview
through the ``notebooklm-py`` client.

Design rules (see the CVPR 2026 Phase 10.4 context packet):

* **Conference-neutral.** Deck ids, titles, and the conference label are read
  from ``analysis/analysis_manifest.json``, ``decks/<id>/presentation_frame.json``
  and ``config.yaml``; nothing conference-specific is embedded here.
* **Fail closed.** ``--live`` needs an explicit acknowledgement, the auth
  preflight must pass before any network call, a cited work with no on-disk
  text is a blocker (never a silent drop), an auth failure halts the whole
  run, and a quota/rate-limit failure halts *resumably*.
* **Idempotent by manifest.** ``notebook_manifest.json`` binds the notebook
  id, every staged source's sha256 and NotebookLM source id, the prompt sha256
  and the audio sha256, so a rerun reuses the notebook, adds only missing
  sources and skips generation when nothing changed.
* **Dry-run never touches the network** and never imports ``notebooklm``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from conference_pipeline.caption_text import caption_segments_payload
from conference_pipeline.conference_identity import conference_name

SCHEMA_VERSION = 1
MANIFEST_TYPE = "conference_pipeline.notebooklm_podcast_manifest"
RECEIPT_TYPE = "conference_pipeline.notebooklm_podcast_receipt"
STAGE_ID = "notebooklm-podcast"
MODEL_TASK_ID = "podcast.notebooklm_audio_overview"
OUTPUT_SUBDIR = "notebooklm_podcast"
AUDIO_ROOT = "audio"
DECKS_DIR = "decks"
ANALYSIS_DIR = "analysis"
ANALYSIS_MANIFEST = "analysis/analysis_manifest.json"
EXECUTIVE_DIGESTS = "decks/executive_topic_digests.json"
CORPUS_INVENTORY = "storage/corpus/corpus_inventory.json"
EVENT_MEDIA_INVENTORY = "storage/download-corpus/event-media/event_media_inventory.json"
CAPTION_TRANSCRIPTS_GLOB = "storage/transcription-ocr/*/caption-transcripts/*.json"
DEFAULT_EXECUTIVE_DECK_IDS: tuple[str, ...] = ("executive_summary_short", "executive_summary_long")
AUDIO_FILENAME = "podcast.m4a"
MANIFEST_FILENAME = "notebook_manifest.json"
DRY_RUN_MANIFEST_FILENAME = "notebook_manifest.dry_run.json"
PROMPT_FILENAME = "prompt.txt"
SHOW_NOTES_FILENAME = "show_notes.md"
FORMATS: tuple[str, ...] = ("deep-dive", "brief", "critique", "debate")
LENGTHS: tuple[str, ...] = ("short", "default", "long")
LONG_PROMPT_CAP = 4000
SHORT_PROMPT_CAP = 1200
PACKAGE_NAME = "notebooklm-py"
PROMPT_FALLBACK_PATTERN = re.compile(r"instruction|prompt|too long|length|limit|exceed", re.IGNORECASE)


class NotebookLMPodcastError(RuntimeError):
    """Raised when podcast production fails closed."""


class NotebookLMAuthBlocked(NotebookLMPodcastError):
    """The NotebookLM session is missing or expired; nothing can proceed."""


class NotebookLMQuotaBlocked(NotebookLMPodcastError):
    """NotebookLM refused the request for quota/rate-limit reasons; resumable."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NotebookLMPodcastError(f"JSON object expected: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` only when the on-disk content differs (keeps sha/mtime stable)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


def slugify(text: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return (slug[:max_length].rstrip("-")) or "item"


# Words that make the Drive backup's secret-file guard (scripts/sync_to_gdrive.py
# is_secret_like_path) refuse a whole sync. A staged source named after a talk
# ("Secrets in the Embeddings ...") is not a secret, but its filename must not
# look like one: the guard stays strict and the name drops the word instead.
_BACKUP_GUARD_WORDS = frozenset(
    {"cookie", "cookies", "secret", "secrets", "password", "passwords",
     "credential", "credentials", "apikey"}
)
# The guard flags "token(s)" only beside one of these ("refresh token").
_BACKUP_GUARD_TOKEN_QUALIFIERS = frozenset({"access", "auth", "bearer", "id", "oauth", "refresh", "session"})


def backup_safe_slug(text: str, *, max_length: int = 48) -> str:
    """``slugify`` without the words the backup's secret-file guard refuses."""

    words = [w for w in re.split(r"[^a-z0-9]+", str(text).lower()) if w]
    drop = set(_BACKUP_GUARD_WORDS)
    if "api" in words and "key" in words:
        drop.add("key")
    if _BACKUP_GUARD_TOKEN_QUALIFIERS.intersection(words):
        drop.update({"token", "tokens"})
    kept = [w for w in words if w not in drop]
    return slugify(" ".join(kept), max_length=max_length)


def word_count(text: str) -> int:
    return len(text.split())


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate(text: str, limit: int) -> str:
    """Trim to ``limit`` chars, preferring a sentence boundary past 60% of the limit."""

    text = _clean(text)
    if len(text) <= limit:
        return text
    window = text[: limit - 1]
    best = max(window.rfind(". "), window.rfind("; "), window.rfind("? "), window.rfind("! "))
    if best >= int(limit * 0.6):
        return window[: best + 1]
    cut = window.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


def _conference_voice(text: str, conference_label: str) -> str:
    """Rewrite corpus/analysis self-references so hosts speak about the conference."""

    text = re.sub(r"\b(?:in |from )?the source packs?\b", "in the conference material", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe " + re.escape(conference_label) + r" corpus\b", conference_label, text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe corpus\b", conference_label, text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis corpus\b", conference_label, text, flags=re.IGNORECASE)
    return text


def parse_markdown_sections(md_text: str) -> dict[str, list[str]]:
    """Split a Markdown doc into ``{lowercased H2 heading: [body lines]}``."""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in md_text.splitlines():
        heading = re.match(r"^##\s+(.*)$", line)
        if heading:
            current = heading.group(1).strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def section_bullets(sections: Mapping[str, list[str]], heading: str, limit: int) -> list[str]:
    bullets: list[str] = []
    for line in sections.get(heading.lower(), []):
        stripped = line.strip()
        if stripped.startswith("- "):
            text = _clean(stripped[2:])
            if text:
                bullets.append(text)
    return bullets[:limit]


def collapse_source_id(source_id: str) -> str:
    """``speech_transcript:<vid>:segment:00003`` -> ``speech_transcript:<vid>``."""

    parts = str(source_id).split(":")
    if len(parts) >= 4 and parts[-2] == "segment":
        return ":".join(parts[:-2])
    return str(source_id)


# --------------------------------------------------------------------------
# Deck scope
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeckRef:
    id: str
    kind: str  # "topic" | "executive"
    title: str
    number: int | None = None
    markdown_path: str | None = None


def deck_scope(
    workspace: Path,
    *,
    executive_deck_ids: Sequence[str] = DEFAULT_EXECUTIVE_DECK_IDS,
) -> list[DeckRef]:
    """All in-scope decks: every analysis-manifest topic plus present executive decks."""

    manifest = read_json(workspace / ANALYSIS_MANIFEST)
    topics = manifest.get("topics")
    if not isinstance(topics, list) or not topics:
        raise NotebookLMPodcastError("analysis manifest has no topics")
    refs: list[DeckRef] = []
    for index, topic in enumerate(topics, start=1):
        if not isinstance(topic, Mapping):
            continue
        topic_id = str(topic.get("id") or topic.get("topic_id") or "").strip()
        if not topic_id:
            raise NotebookLMPodcastError(f"analysis manifest topic {index} has no id")
        number = topic.get("number")
        refs.append(
            DeckRef(
                id=topic_id,
                kind="topic",
                title=_clean(topic.get("title") or topic_id),
                number=int(number) if isinstance(number, int) else index,
                markdown_path=str(topic.get("markdown_path") or f"{ANALYSIS_DIR}/{topic_id}.md"),
            )
        )
    for deck_id in executive_deck_ids:
        frame_path = workspace / DECKS_DIR / deck_id / "presentation_frame.json"
        if not frame_path.is_file():
            continue
        frame = read_json(frame_path)
        refs.append(
            DeckRef(
                id=deck_id,
                kind="executive",
                title=_clean(frame.get("audience_title") or frame.get("retrieval_title") or deck_id),
                number=None,
                markdown_path=str(frame.get("source_analysis_path") or f"{ANALYSIS_DIR}/executive_summary.md"),
            )
        )
    return refs


# --------------------------------------------------------------------------
# Corpus join (cited work -> on-disk text)
# --------------------------------------------------------------------------


@dataclass
class CorpusJoin:
    by_artifact: dict[str, dict[str, Any]] = field(default_factory=dict)
    event_titles: dict[str, str] = field(default_factory=dict)
    caption_json_by_video: dict[str, Path] = field(default_factory=dict)
    # Transcript-lane talks, cited as ``speech_transcript:<lane>_<video>``.
    lane_transcript_by_key: dict[str, Path] = field(default_factory=dict)
    # Workshop papers, cited as ``paper_abstract:<venue path with / and : as _>``.
    workshop_paper_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Papers keyed by OpenReview id: conferences whose inventory names papers
    # ``paper_text:<hash>`` cite them as ``paper_abstract:<openreview id>``.
    paper_by_openreview_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Fallbacks for a cited id the inventory no longer carries verbatim (a
    # corpus rebuild re-keys artifacts): the inventory hash after the family
    # prefix, then the exact title within the same family.
    paper_by_artifact_hash: dict[str, dict[str, Any]] = field(default_factory=dict)
    records_by_family_title: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    corpus_inventory_sha256: str | None = None


def _title_key(title: Any) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(title or "").casefold()).split())


def _record_titles(record: Mapping[str, Any]) -> list[str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    return [
        str(value)
        for value in (record.get("paper_title"), metadata.get("paper_title"), metadata.get("talk_title"), metadata.get("title"))
        if value
    ]


def resolve_cited_record(join: "CorpusJoin", source_id: str, title: str) -> tuple[dict[str, Any] | None, str | None]:
    """The inventory record a cited id names, and how it was found.

    Direct id forms first; then the inventory hash; then an exact title in
    the same family. ``None`` when nothing matches -- never a fuzzy guess.
    """

    family, _, key = source_id.partition(":")
    if family == "paper_abstract":
        record = join.by_artifact.get(f"paper:{key}") or join.workshop_paper_by_key.get(key)
        if record is not None:
            return record, "id"
        if key in join.paper_by_openreview_id:
            return join.paper_by_openreview_id[key], "openreview_id"
        if key in join.paper_by_artifact_hash:
            return join.paper_by_artifact_hash[key], "artifact_hash"
        families = ("paper",)
    elif family == "speech_transcript":
        record = join.by_artifact.get(source_id)
        if record is not None:
            return record, "id"
        families = ("speech_transcript",)
    else:
        return None, None
    wanted = _title_key(title)
    for record_family in families:
        record = join.records_by_family_title.get((record_family, wanted)) if wanted else None
        if record is not None:
            return record, "exact_title"
    return None, None


def _load_transcript_payload(path: Path) -> dict[str, Any] | None:
    """Transcript JSON as-is; WebVTT through the shared caption parser; else plain text."""

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    video_id = path.name.split(".", 1)[0]
    payload = caption_segments_payload(raw)
    if payload is None:
        text = " ".join(raw.split())
        payload = {"text": text} if text else None
    if payload is not None:
        payload["video_id"] = video_id
    return payload


def load_corpus_join(workspace: Path) -> CorpusJoin:
    join = CorpusJoin()
    inventory_path = workspace / CORPUS_INVENTORY
    if inventory_path.is_file():
        join.corpus_inventory_sha256 = sha256_file(inventory_path)
        payload = read_json(inventory_path)
        for record in payload.get("artifact_records") or []:
            if isinstance(record, Mapping) and record.get("artifact_id"):
                join.by_artifact[str(record["artifact_id"])] = dict(record)
                parts = str(record["artifact_id"]).split(":")
                if len(parts) == 3 and parts[0] == "event_media_caption" and record.get("transcript_path"):
                    join.lane_transcript_by_key.setdefault(f"{parts[1]}_{parts[2]}", workspace / str(record["transcript_path"]))
                artifact_id = str(record["artifact_id"])
                if artifact_id.startswith("workshop_paper:"):
                    key = artifact_id.split(":", 1)[1].replace("/", "_").replace(":", "_")
                    join.workshop_paper_by_key.setdefault(key, dict(record))
                if artifact_id.startswith(("paper_text:", "workshop_paper_text:")):
                    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
                    openreview_id = record.get("openreview_id") or metadata.get("openreview_id")
                    if openreview_id:
                        join.paper_by_openreview_id.setdefault(str(openreview_id), dict(record))
                    join.paper_by_artifact_hash.setdefault(artifact_id.split(":", 1)[1], dict(record))
                record_family = (
                    "paper"
                    if artifact_id.startswith(("paper_text:", "workshop_paper_text:", "paper:", "workshop_paper:"))
                    else "speech_transcript"
                    if artifact_id.startswith("speech_transcript:")
                    else None
                )
                if record_family and record.get("indexable_text") is not False:
                    for record_title in _record_titles(record):
                        if _title_key(record_title):
                            join.records_by_family_title.setdefault((record_family, _title_key(record_title)), dict(record))
    events_path = workspace / EVENT_MEDIA_INVENTORY
    if events_path.is_file():
        payload = read_json(events_path)
        for event in payload.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            title = _clean(event.get("title"))
            for video_id in event.get("video_ids") or []:
                if title and video_id:
                    join.event_titles.setdefault(str(video_id), title)
    for path in sorted(workspace.glob(CAPTION_TRANSCRIPTS_GLOB)):
        video_id = path.name.rsplit("-", 1)[0] if "-" in path.name else path.stem
        join.caption_json_by_video.setdefault(video_id, path)
    return join


def dedupe_rolling_captions(segments: Sequence[Mapping[str, Any]], *, window: int = 40) -> list[str]:
    """Rebuild clean speech from rolling auto-captions.

    YouTube auto-captions repeat each cue as the rolling window advances, so
    ``normalized_speech_text`` triples every phrase. For each cue we keep only
    the suffix that does not overlap the words already emitted.
    """

    words: list[str] = []
    for segment in segments:
        cue = _clean(segment.get("text")).split()
        if not cue:
            continue
        tail = words[-window:]
        overlap = 0
        for size in range(min(len(cue), len(tail)), 0, -1):
            if tail[-size:] == cue[:size]:
                overlap = size
                break
        if overlap == len(cue):
            continue
        words.extend(cue[overlap:])
    return words


def caption_json_to_text(
    payload: Mapping[str, Any],
    *,
    title: str,
    people: Sequence[str],
    conference_label: str,
) -> tuple[str, dict[str, Any]]:
    segments = payload.get("segments")
    raw_text = _clean(payload.get("normalized_speech_text") or payload.get("text"))
    if isinstance(segments, list) and segments:
        words = dedupe_rolling_captions(segments)
    else:
        words = raw_text.split()
    body = " ".join(words)
    raw_words = max(word_count(raw_text), 1)
    stats = {
        "raw_word_count": word_count(raw_text),
        "deduped_word_count": len(words),
        "dedup_ratio": round(len(words) / raw_words, 3),
    }
    header = [f"# {title}"]
    if people:
        header.append(f"Speakers/Authors: {', '.join(people)}")
    header.append(f"Source: {conference_label} talk transcript (video id {payload.get('video_id', 'unknown')})")
    return "\n".join(header) + "\n\n" + body + "\n", stats


def corpus_transcript_to_text(
    path: Path,
    *,
    title: str,
    people: Sequence[str],
    conference_label: str,
) -> str | None:
    """A corpus speech transcript (``speech_transcript:<hash>``) as source text.

    These are ASR transcripts, not rolling auto-captions, so no overlap
    de-duplication applies. An annotated multi-talk recording keeps one
    heading per talk so the hosts can tell which speaker said what.
    """

    raw = path.read_text(encoding="utf-8", errors="ignore")
    body: list[str] = []
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        segments = payload.get("segments") if isinstance(payload, Mapping) else None
        if not isinstance(segments, list):
            return None
        current_talk = None
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            talk = _clean(segment.get("talk_title"))
            if talk and talk != current_talk:
                speaker = _clean(segment.get("talk_speaker"))
                body.append(f"\n## {talk}" + (f" — {speaker}" if speaker and speaker.lower() != "unknown" else ""))
                current_talk = talk
            text = _clean(segment.get("text"))
            if text:
                body.append(text)
    else:
        text = " ".join(raw.split())
        if text:
            body.append(text)
    if not body:
        return None
    header = [f"# {title}"]
    if people:
        header.append(f"Speakers/Authors: {', '.join(people)}")
    header.append(f"Source: {conference_label} talk transcript")
    return "\n".join(header) + "\n\n" + "\n".join(body).strip() + "\n"


def paper_text_with_header(
    text: str,
    *,
    title: str,
    authors: Sequence[str],
    conference_label: str,
) -> str:
    header = [f"# {title}"]
    if authors:
        header.append(f"Authors: {', '.join(authors)}")
    header.append(f"Source: {conference_label} paper")
    return "\n".join(header) + "\n\n" + text.strip() + "\n"


# --------------------------------------------------------------------------
# Rendered spines (blueprint / executive digest / frame)
# --------------------------------------------------------------------------


def _names(values: object, limit: int = 4) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean(v) for v in values if _clean(v)][:limit]


def render_blueprint_spine(blueprint: Mapping[str, Any], *, deck_title: str) -> str:
    """Narrative outline rendered from ``presentation_blueprint.json``."""

    lines = [f"# {deck_title} — editorial spine", ""]
    if blueprint.get("deck_thesis"):
        lines += [f"Thesis: {_clean(blueprint['deck_thesis'])}", ""]
    if blueprint.get("audience_promise"):
        lines += [f"Audience promise: {_clean(blueprint['audience_promise'])}", ""]
    slides = [s for s in blueprint.get("slides") or [] if isinstance(s, Mapping)]
    by_id = {str(s.get("slide_id")): s for s in slides}
    sections = blueprint.get("telescopic_section_map") or blueprint.get("sections") or []
    lines.append("## Sections")
    ordered_slide_ids: list[str] = []
    for index, section in enumerate(sections, start=1):
        if not isinstance(section, Mapping):
            continue
        section_title = _clean(section.get("title") or section.get("section_id") or f"Section {index}")
        lines += ["", f"### {index}. {section_title}"]
        for slide_id in section.get("slide_ids") or []:
            slide = by_id.get(str(slide_id))
            if slide is None:
                continue
            ordered_slide_ids.append(str(slide_id))
            lines.extend(_render_slide(slide))
    leftovers = [s for s in slides if str(s.get("slide_id")) not in set(ordered_slide_ids)]
    if leftovers:
        lines += ["", "### Additional slides"]
        for slide in leftovers:
            lines.extend(_render_slide(slide))
    if blueprint.get("conclusion_synthesis"):
        lines += ["", "## Conclusion", _clean(blueprint["conclusion_synthesis"])]
    return "\n".join(lines) + "\n"


def _render_slide(slide: Mapping[str, Any]) -> list[str]:
    out = [f"- Slide: {_clean(slide.get('title'))}"]
    if slide.get("intent"):
        out[0] += f" — {_clean(slide['intent'])}"
    for claim in slide.get("claims") or []:
        if _clean(claim):
            out.append(f"  - Claim: {_clean(claim)}")
    beats = slide.get("narration_beats") or ([slide["narration_beat"]] if slide.get("narration_beat") else [])
    for beat in beats:
        if _clean(beat):
            out.append(f"  - Narration: {_clean(beat)}")
    mentions = slide.get("speaker_author_work_mentions") or []
    named = []
    for mention in mentions:
        if isinstance(mention, Mapping) and _clean(mention.get("title")):
            people = _names(mention.get("speakers")) or _names(mention.get("authors"))
            named.append(f"{_clean(mention['title'])}" + (f" ({', '.join(people)})" if people else ""))
    if named:
        out.append(f"  - Named works: {'; '.join(named)}")
    for ref in slide.get("source_refs") or []:
        if _clean(ref):
            out.append(f"  - Source: {_clean(ref)}")
    return out


def render_executive_digest(digests: Mapping[str, Any], *, conference_label: str) -> str:
    lines = [f"# {conference_label} — cross-topic executive digest", ""]
    central = digests.get("central_message") or {}
    if isinstance(central, Mapping):
        if central.get("headline"):
            lines.append(f"Central message: {_clean(central['headline'])}")
        if central.get("subtitle"):
            lines.append(_clean(central["subtitle"]))
        lines.append("")
    index_by_topic = {
        str(t.get("topic_id")): t for t in digests.get("topic_index") or [] if isinstance(t, Mapping)
    }
    takeaways = {
        str(t.get("topic_id")): t for t in digests.get("topics") or [] if isinstance(t, Mapping)
    }
    arcs = digests.get("arcs") or []
    if arcs:
        lines.append("## Arcs")
        for arc in arcs:
            if not isinstance(arc, Mapping):
                continue
            lines += ["", f"### {_clean(arc.get('name'))}"]
            for topic_id in arc.get("topic_ids") or []:
                entry = index_by_topic.get(str(topic_id), {})
                take = takeaways.get(str(topic_id), {})
                title = _clean(entry.get("title") or topic_id)
                one_liner = _clean(entry.get("one_liner"))
                lines.append(f"- {title}: {one_liner}")
                if _clean(take.get("central_takeaway")):
                    lines.append(f"  - Takeaway: {_clean(take['central_takeaway'])}")
    if digests.get("open_questions"):
        lines += ["", "## Open questions"]
        for item in digests["open_questions"]:
            if isinstance(item, Mapping):
                lines.append(f"- {_clean(item.get('question'))}")
                if _clean(item.get("why_it_matters")):
                    lines.append(f"  - Why it matters: {_clean(item['why_it_matters'])}")
    if digests.get("predictions"):
        lines += ["", "## Predictions"]
        for item in digests["predictions"]:
            if isinstance(item, Mapping):
                lines.append(f"- {_clean(item.get('prediction'))}")
                if _clean(item.get("basis")):
                    lines.append(f"  - Basis: {_clean(item['basis'])}")
    return "\n".join(lines) + "\n"


def render_frame_spine(frame: Mapping[str, Any], blueprint: Mapping[str, Any] | None, *, deck_title: str) -> str:
    lines = [f"# {deck_title} — framing", ""]
    if frame.get("field_thesis"):
        lines += [f"Thesis: {_clean(frame['field_thesis'])}", ""]
    if frame.get("audience_subtitle"):
        lines += [f"Audience: {_clean(frame['audience_subtitle'])}", ""]
    questions = frame.get("general_questions") or []
    if questions:
        lines.append("## Guiding questions")
        lines += [f"- {_clean(q)}" for q in questions if _clean(q)]
        lines.append("")
    if blueprint:
        lines.append(render_blueprint_spine(blueprint, deck_title=deck_title))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Source sets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    index: int
    kind: str
    title: str
    origin_id: str
    local_path: str  # workspace-relative staged file
    sha256: str
    word_count: int
    notebooklm_title: str


@dataclass
class DeckSourceSet:
    deck: DeckRef
    conference_label: str
    sources: list[SourceSpec] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    cited_works: list[dict[str, Any]] = field(default_factory=list)
    dedup_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    episode_plan: Any = None  # TopicEpisodePlan | ExecutiveEpisodePlan once built

    @property
    def source_sha_set(self) -> list[str]:
        return sorted(s.sha256 for s in self.sources)


def output_dir(workspace: Path, deck_id: str, *, audio_root: str = AUDIO_ROOT) -> Path:
    return workspace / audio_root / deck_id / OUTPUT_SUBDIR


def cited_source_ids(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Distinct works the approved blueprint cites, in first-seen order."""

    seen: dict[str, dict[str, Any]] = {}
    for key in ("speaker_author_work_mention_plan", "source_paper_plan"):
        for entry in blueprint.get(key) or []:
            if not isinstance(entry, Mapping) or not entry.get("source_id"):
                continue
            source_id = collapse_source_id(str(entry["source_id"]))
            if source_id in seen:
                continue
            seen[source_id] = {
                "source_id": source_id,
                "label": _clean(entry.get("label")),
                "title": _clean(entry.get("title")) or (_names(entry.get("works"), 1) or [""])[0],
                "authors": _names(entry.get("authors"), 6),
                "speakers": _names(entry.get("speakers"), 6),
                "contribution_note": _truncate(entry.get("contribution_note") or "", 240),
            }
    return list(seen.values())


class _Stager:
    """Stages source files under ``<out_dir>/sources`` with stable numbering."""

    def __init__(self, workspace: Path, out_dir: Path, conference_label: str) -> None:
        self.workspace = workspace
        self.sources_dir = out_dir / "sources"
        self.conference_label = conference_label
        self.specs: list[SourceSpec] = []

    def add(self, *, kind: str, title: str, origin_id: str, text: str) -> SourceSpec:
        index = len(self.specs) + 1
        filename = f"{index:02d}__{backup_safe_slug(kind)}__{backup_safe_slug(title)}.md"
        path = self.sources_dir / filename
        write_text_if_changed(path, text)
        spec = SourceSpec(
            index=index,
            kind=kind,
            title=title,
            origin_id=origin_id,
            local_path=str(path.relative_to(self.workspace)),
            sha256=sha256_text(text),
            word_count=word_count(text),
            notebooklm_title=notebooklm_source_title(index, self.conference_label, kind, title, sha256_text(text)),
        )
        self.specs.append(spec)
        return spec

    def add_file(self, *, kind: str, title: str, origin_id: str, path: Path) -> SourceSpec:
        return self.add(kind=kind, title=title, origin_id=origin_id, text=path.read_text(encoding="utf-8"))

    def prune_stale(self) -> list[str]:
        keep = {Path(spec.local_path).name for spec in self.specs}
        removed: list[str] = []
        if self.sources_dir.is_dir():
            for stale in self.sources_dir.iterdir():
                if stale.is_file() and stale.name not in keep and not stale.name.startswith("."):
                    stale.unlink()
                    removed.append(stale.name)
        return removed


def notebooklm_source_title(index: int, conference_label: str, kind: str, title: str, sha256: str) -> str:
    """``NN [Conf] Kind: Title [sha:8hex]`` — the slot prefix identifies the source,
    the sha tag identifies its content, so a changed file is re-added and the
    stale one is removed rather than silently reused."""

    return f"{index:02d} [{conference_label}] {_kind_label(kind)}: {_truncate(title, 90)} [sha:{sha256[:8]}]"


def source_slot_prefix(notebooklm_title: str) -> str:
    return notebooklm_title.split(" ", 1)[0]


def _kind_label(kind: str) -> str:
    return {
        "analysis_report": "Topic report",
        "exhaustive_summary": "Talk synthesis",
        "deep_dive": "Deep dive",
        "blueprint_spine": "Editorial spine",
        "paper_text": "Paper",
        "talk_transcript": "Talk",
        "executive_summary": "Executive summary",
        "trend_report": "Trend report",
        "topic_digest": "Topic digest",
        "frame_spine": "Framing",
        "episode_brief": "Episode brief",
        "program_map": "Program map",
    }.get(kind, kind.replace("_", " ").title())


# --------------------------------------------------------------------------
# Production facts: credit, how the material was made, program map
# --------------------------------------------------------------------------

LLM_FULL_READ_RECEIPT_GLOB = "storage/workflow-runs/*/14A-llm-full-read-extraction-receipt.json"
LLM_FULL_READ_BY_TOPIC_GLOB = "storage/extractions/*/llm-full-read/by-topic/{topic_id}.json"
PROGRAM_CATEGORY_ORDER: tuple[tuple[str, str], ...] = (
    ("keynote", "Keynotes"),
    ("workshop", "Workshops"),
    ("tutorial", "Tutorials"),
    ("oral", "Oral sessions"),
    ("poster", "Other recorded talks"),
)
EPISODE_BRIEF_KIND = "episode_brief"
PROGRAM_MAP_KIND = "program_map"
_TEMPLATED_TEXT_MARKERS = ("judged on named", "named source evidence", "clusters and", "named sources, the corpus")
_DECK_EDITORIAL_MARKERS = ("deck", "slide", "retrieve", "unsupported", "verified", "full-paper", "conference narrative", "presentation")


@dataclass(frozen=True)
class ProductionFacts:
    """Honest, receipt-derived facts the hosts state in the opening."""

    conference_label: str
    creator: str | None
    paper_count: int | None
    workshop_paper_count: int | None
    talk_count: int | None
    full_read_model: str | None
    full_read_unit_count: int | None
    topic_count: int

    def how_made(self) -> str:
        """One paragraph the hosts paraphrase: harvested, read end to end, distilled."""

        inventory: list[str] = []
        if self.paper_count:
            inventory.append(f"{self.paper_count:,} main-conference papers")
        if self.workshop_paper_count:
            inventory.append(f"{self.workshop_paper_count:,} workshop papers")
        if self.talk_count:
            inventory.append(f"{self.talk_count:,} recorded talks with transcripts")
        harvested = f" ({', '.join(inventory)})" if inventory else ""
        model = f" ({self.full_read_model})" if self.full_read_model else ""
        windows = (
            f" The talks alone were read in {self.full_read_unit_count:,} transcript windows, each one assigned to one of the {self.topic_count} topics and summarized."
            if self.full_read_unit_count
            else ""
        )
        return (
            f"The material was harvested from {self.conference_label}{harvested}, and every item was read end to end by a small language model running locally{model}.{windows} "
            "That reading was then distilled incrementally, step by step: exhaustive per-topic extraction, a searchable evidence index, targeted deep-dive research questions, "
            "topic narratives with traceable citations, and a cross-topic synthesis, all to surface what is genuinely interesting. "
            "It is a comprehensive distillation of the conference, not a single deep-research query, and the hosts did not attend the conference."
        )

    def how_made_short(self) -> str:
        counts = ", ".join(
            part
            for part in (
                f"{self.paper_count:,} papers" if self.paper_count else "",
                f"{self.workshop_paper_count:,} workshop papers" if self.workshop_paper_count else "",
                f"{self.talk_count:,} recorded talks" if self.talk_count else "",
            )
            if part
        )
        counts = f" ({counts})" if counts else ""
        return (
            f"everything was harvested from {self.conference_label}{counts}, read end to end by a small language model running locally, "
            "and distilled step by step (topic classification, exhaustive extraction, targeted deep-dive research, traceable topic narratives) "
            "to surface what is interesting; it is not a single deep-research query"
        )


def speakable_model_name(model: str | None) -> str | None:
    """``mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`` -> ``Qwen3.6-35B-A3B``."""

    if not model:
        return None
    name = str(model).rsplit("/", 1)[-1]
    parts = [p for p in name.split("-") if not re.fullmatch(r"(?i)(optiq|\d+bit|mlx|gguf|instruct|awq|gptq|fp16|bf16)", p)]
    return "-".join(parts) or name


def deck_creator(workspace: Path, deck_id: str | None = None) -> str | None:
    """The deck's recorded author; falls back to the first deck blueprint that names one."""

    candidates: list[Path] = []
    if deck_id:
        candidates.append(workspace / DECKS_DIR / deck_id / "presentation_blueprint.json")
    candidates.extend(sorted((workspace / DECKS_DIR).glob("*/presentation_blueprint.json")))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            author = _clean(read_json(path).get("author"))
        except (OSError, json.JSONDecodeError, NotebookLMPodcastError):
            continue
        if author:
            return author
    return None


def load_production_facts(workspace: Path, *, conference_label: str, deck_id: str | None = None) -> ProductionFacts:
    families: dict[str, Any] = {}
    inventory_path = workspace / CORPUS_INVENTORY
    if inventory_path.is_file():
        counts = read_json(inventory_path).get("artifact_counts") or {}
        families = dict(counts.get("by_source_family") or {}) if isinstance(counts, Mapping) else {}
    model: str | None = None
    units: int | None = None
    receipts = []
    for path in workspace.glob(LLM_FULL_READ_RECEIPT_GLOB):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError, NotebookLMPodcastError):
            continue
        if payload.get("status") == "succeeded":
            receipts.append((str(payload.get("ended_at") or ""), payload))
    if receipts:
        _ended, latest = sorted(receipts, key=lambda item: item[0])[-1]
        model = speakable_model_name(latest.get("model"))
        checks = latest.get("checks") or {}
        completed = checks.get("completed_unit_count") if isinstance(checks, Mapping) else None
        units = int(completed) if isinstance(completed, int) and completed > 0 else None
    try:
        topic_count = len(read_json(workspace / ANALYSIS_MANIFEST).get("topics") or [])
    except (OSError, json.JSONDecodeError, NotebookLMPodcastError):
        topic_count = 0

    def _count(key: str) -> int | None:
        value = families.get(key)
        return int(value) if isinstance(value, int) and value > 0 else None

    return ProductionFacts(
        conference_label=conference_label,
        creator=deck_creator(workspace, deck_id),
        paper_count=_count("paper_text"),
        workshop_paper_count=_count("workshop_paper_text"),
        talk_count=_count("speech_transcript"),
        full_read_model=model,
        full_read_unit_count=units,
        topic_count=topic_count,
    )


@dataclass
class ProgramItem:
    video_id: str
    title: str
    category: str
    source_url: str | None
    window_count: int
    summaries: list[str] = field(default_factory=list)


def clean_event_title(title: object) -> str:
    """Drop schedule brackets such as ``[9:00-10:15]`` from program titles."""

    return _clean(re.sub(r"\s*\[[^\]]*\d{1,2}:\d{2}[^\]]*\]\s*", " ", str(title or "")))


def load_event_index(workspace: Path) -> dict[str, Mapping[str, Any]]:
    path = workspace / EVENT_MEDIA_INVENTORY
    index: dict[str, Mapping[str, Any]] = {}
    if not path.is_file():
        return index
    for event in read_json(path).get("events") or []:
        if not isinstance(event, Mapping):
            continue
        for video_id in event.get("video_ids") or []:
            index.setdefault(str(video_id), event)
    return index


def topic_program(workspace: Path, topic_id: str, event_index: Mapping[str, Mapping[str, Any]]) -> dict[str, list[ProgramItem]]:
    """Rank the talks, workshops, and sessions the full read assigned to a topic.

    Ranking is by the number of transcript windows the small model judged
    important for this topic; each item keeps its distinct window summaries.
    """

    paths = sorted(workspace.glob(LLM_FULL_READ_BY_TOPIC_GLOB.format(topic_id=topic_id)))
    if not paths:
        return {}
    payload = read_json(paths[-1])
    items: dict[str, ProgramItem] = {}
    for unit in payload.get("items") or []:
        if not isinstance(unit, Mapping):
            continue
        artifact = str(unit.get("source_artifact_id") or "")
        if not artifact.startswith("event_media_caption:"):
            continue
        video_id = artifact.split(":", 1)[1]
        event = event_index.get(video_id)
        if event is None:
            continue
        item = items.get(video_id)
        if item is None:
            item = ProgramItem(
                video_id=video_id,
                title=clean_event_title(event.get("title")) or f"Talk {video_id}",
                category=_clean(event.get("event_category")) or "talk",
                source_url=_clean(event.get("source_url")) or None,
                window_count=0,
            )
            items[video_id] = item
        item.window_count += 1
        summary = _clean(unit.get("summary"))
        if summary and summary not in item.summaries:
            item.summaries.append(summary)
    grouped: dict[str, list[ProgramItem]] = {}
    for item in sorted(items.values(), key=lambda i: (-i.window_count, i.title)):
        grouped.setdefault(item.category, []).append(item)
    return grouped


PROGRAM_RELEVANCE_FLOOR = 0.25
PROGRAM_MIN_WINDOWS = 3


def program_highlights(program: Mapping[str, Sequence[ProgramItem]], *, limit: int = 4) -> list[ProgramItem]:
    """Keynotes, workshops, tutorials, then oral sessions: the named program to mention.

    An item must carry at least a quarter of the strongest item's relevant
    windows (and a small absolute minimum), so a session the small model
    touched only in passing is not presented as central to the topic.
    """

    named = [item for category in ("keynote", "workshop", "tutorial", "oral") for item in program.get(category) or []]
    if not named:
        return []
    floor = max(PROGRAM_MIN_WINDOWS, PROGRAM_RELEVANCE_FLOOR * max(item.window_count for item in named))
    picked: list[ProgramItem] = []
    for category in ("keynote", "workshop", "tutorial", "oral"):
        for item in program.get(category) or []:
            if item.window_count < floor:
                continue
            if len(picked) >= limit:
                return picked
            if category == "oral" and len([p for p in picked if p.category == "oral"]) >= 1:
                break
            picked.append(item)
    return picked


def render_program_map(
    deck_title: str,
    program: Mapping[str, Sequence[ProgramItem]],
    *,
    conference_label: str,
    per_category: int = 8,
    summaries_per_item: int = 4,
) -> str:
    lines = [
        f"# {deck_title} — talks, workshops, and sessions at {conference_label}",
        "",
        "How this list was made: a small language model read every recorded talk end to end, in transcript windows, "
        "and judged which windows matter for this topic. Items are ranked by how many of their windows it assigned here; "
        "the notes under each item are its own summaries of those windows.",
    ]
    for category, heading in PROGRAM_CATEGORY_ORDER:
        entries = list(program.get(category) or [])[:per_category]
        if not entries:
            continue
        lines += ["", f"## {heading}"]
        for item in entries:
            lines.append(f"- **{item.title}** ({item.window_count} relevant transcript windows)")
            for summary in item.summaries[:summaries_per_item]:
                lines.append(f"  - {_truncate(summary, 260)}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Report digest (the reconciled topic report, not the raw talk synthesis)
# --------------------------------------------------------------------------


@dataclass
class ReportDigest:
    summary_paragraphs: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def _strip_citations(text: str) -> str:
    return _clean(re.sub(r"\s*\[(?:T|P|S|W)\d+(?:\s*,\s*(?:T|P|S|W)?\d+)*\]", "", text))


def _first_sentences(text: str, count: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", _clean(text))
    return " ".join(sentences[:count]).strip()


def is_templated_text(text: str) -> bool:
    lowered = _clean(text).lower()
    return any(marker in lowered for marker in _TEMPLATED_TEXT_MARKERS)


def digest_topic_report(report_md: str) -> ReportDigest:
    sections = parse_markdown_sections(report_md)
    digest = ReportDigest()
    paragraph: list[str] = []
    for line in sections.get("executive summary", []) + [""]:
        if line.strip():
            paragraph.append(line.strip())
        elif paragraph:
            digest.summary_paragraphs.append(_strip_citations(" ".join(paragraph)))
            paragraph = []
    for line in sections.get("key findings", []):
        match = re.match(r"^\s*-\s+\*\*(.+?)\*\*\s*(.*)$", line)
        if match:
            headline = _clean(match.group(1)).rstrip(".")
            detail = _first_sentences(_strip_citations(match.group(2)), 1)
            digest.findings.append(f"{headline}." + (f" {detail}" if detail else ""))
    for line in sections.get("recommendations", []):
        match = re.match(r"^\s*\d+\.\s+\*\*(.+?)\*\*\s*(.*)$", line)
        if not match:
            continue
        headline = _clean(match.group(1)).rstrip(".")
        if any(marker in (headline + " " + match.group(2)).lower() for marker in _DECK_EDITORIAL_MARKERS):
            continue
        digest.recommendations.append(f"{headline}.")
    return digest


# --------------------------------------------------------------------------
# Episode briefs (the run sheet NotebookLM follows, never read on air)
# --------------------------------------------------------------------------


def _work_line(work: Mapping[str, Any]) -> str:
    people = list(work.get("speakers") or work.get("authors") or [])[:2]
    title = _clean(work.get("title"))
    return f"{title} ({', '.join(people)})" if people else title


def _exhaustive_bullets(workspace: Path, deck_id: str) -> tuple[list[str], list[str], list[str]]:
    path = workspace / ANALYSIS_DIR / "exhaustive_summaries" / f"{deck_id}__transcripts.md"
    if not path.is_file():
        return [], [], []
    sections = parse_markdown_sections(path.read_text(encoding="utf-8"))
    return (
        section_bullets(sections, "Key Points", 6),
        section_bullets(sections, "Tensions Or Contrasts", 3),
        section_bullets(sections, "Open Questions", 3),
    )


def blueprint_segments(blueprint: Mapping[str, Any], *, deck_title: str, conference_label: str) -> list[tuple[str, str]]:
    """Section titles with one real claim each; templated frame sections are dropped."""

    slides = {str(s.get("slide_id")): s for s in blueprint.get("slides") or [] if isinstance(s, Mapping)}
    segments: list[tuple[str, str]] = []
    for section in blueprint.get("telescopic_section_map") or []:
        if not isinstance(section, Mapping):
            continue
        claim = ""
        for slide_id in section.get("slide_ids") or []:
            for candidate in slides.get(str(slide_id), {}).get("claims") or []:
                text = _clean(candidate)
                if text and not is_templated_text(text) and not text.lower().startswith(deck_title.lower()):
                    claim = _conference_voice(text, conference_label)
                    break
            if claim:
                break
        if not claim:
            continue  # a frame-only section: the opening covers it
        segments.append((_clean(section.get("title") or section.get("section_id")), claim))
    if not segments:
        segments = [
            (_clean(c.get("title")), _conference_voice(_clean(c.get("claim")), conference_label))
            for c in blueprint.get("subtopic_clusters") or []
            if isinstance(c, Mapping) and _clean(c.get("title"))
        ]
    return segments


@dataclass
class TopicEpisodePlan:
    deck_title: str
    creator: str | None
    facts: ProductionFacts
    hook: str
    segments: list[tuple[str, str]]
    works: list[str]
    program: list[ProgramItem]
    findings: list[str]
    tensions: list[str]
    open_questions: list[str]
    takeaways: list[str]
    target_minutes: int


def plan_topic_episode(workspace: Path, deck: DeckRef, source_set: DeckSourceSet, *, conference_label: str) -> TopicEpisodePlan:
    blueprint = read_json(workspace / DECKS_DIR / deck.id / "presentation_blueprint.json")
    report_path = workspace / (deck.markdown_path or f"{ANALYSIS_DIR}/{deck.id}.md")
    report = digest_topic_report(report_path.read_text(encoding="utf-8")) if report_path.is_file() else ReportDigest()
    _points, tensions, open_questions = _exhaustive_bullets(workspace, deck.id)
    facts = load_production_facts(workspace, conference_label=conference_label, deck_id=deck.id)
    hook = ""
    if report.summary_paragraphs:
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", report.summary_paragraphs[0]):
            hook = f"{hook} {sentence}".strip()
            if len(hook) >= 150:
                break
    if not hook and not is_templated_text(_clean(blueprint.get("deck_thesis"))):
        hook = _clean(blueprint.get("deck_thesis"))
    budget = blueprint.get("duration_budget") or {}
    minutes = budget.get("target_minutes") if isinstance(budget, Mapping) else None
    program = topic_program(workspace, deck.id, load_event_index(workspace))
    return TopicEpisodePlan(
        deck_title=deck.title,
        creator=facts.creator,
        facts=facts,
        hook=_conference_voice(hook, conference_label),
        segments=blueprint_segments(blueprint, deck_title=deck.title, conference_label=conference_label),
        works=[_work_line(w) for w in source_set.cited_works if w.get("title")],
        program=program_highlights(program, limit=5),
        findings=report.findings[:6],
        tensions=[_conference_voice(t, conference_label) for t in tensions],
        open_questions=[_conference_voice(q, conference_label) for q in open_questions],
        takeaways=report.recommendations[:3],
        target_minutes=int(round(float(minutes))) if isinstance(minutes, (int, float)) and minutes > 0 else 25,
    )


def render_topic_brief(plan: TopicEpisodePlan) -> str:
    c = plan.facts.conference_label
    credit = f"This episode was created by {plan.creator}." if plan.creator else "Credit the creator named in the notes."
    lines = [
        f"# Episode brief — {plan.deck_title} ({c})",
        "",
        "Run sheet for the hosts. Follow it closely. Never mention this brief, the notebook, or any source by name on air.",
        "",
        "## Credit",
        credit,
        "",
        "## How the material was made (say this near the start, in your own words)",
        plan.facts.how_made(),
        "",
        "## Opening, in this order, within the first ninety seconds",
        f"1. Say this is a deep dive into {plan.deck_title} from {c}" + (f", created by {plan.creator}." if plan.creator else "."),
        "2. Explain how the material was made (above), plainly and briefly.",
        f"3. Say that the hosts did not attend {c}.",
        "4. Hook the listener with the central shift: " + (plan.hook or "the tension below."),
        "5. Give a one-sentence roadmap of the segments.",
        "Do not read titles, headings, or thesis statements aloud. Do not claim to have attended, walked the halls, or sat in sessions, and do not say the listener sent you papers.",
        "",
        "## Segments, in order",
    ]
    lines += [f"{i}. {title} — {claim}" for i, (title, claim) in enumerate(plan.segments, start=1)]
    if plan.works:
        lines += ["", "## Papers and talks to name, with their people"]
        lines += [f"- {w}" for w in plan.works]
    if plan.program:
        lines += ["", "## Workshops and sessions to bring in where they fit"]
        lines += [f"- {item.title} ({item.category})" for item in plan.program]
    if plan.findings:
        lines += ["", "## Findings to emphasize"] + [f"- {f}" for f in plan.findings]
    if plan.tensions:
        lines += ["", "## Tensions worth arguing over"] + [f"- {t}" for t in plan.tensions]
    if plan.open_questions:
        lines += ["", "## Open questions to leave with the listener"] + [f"- {q}" for q in plan.open_questions]
    lines += ["", "## Close"]
    if plan.takeaways:
        lines.append("What this changes for someone building or funding this work:")
        lines += [f"- {t}" for t in plan.takeaways]
    lines.append(f"End by crediting {plan.creator} again." if plan.creator else "End with a brief credit.")
    return "\n".join(lines) + "\n"


@dataclass
class ExecutiveTopic:
    order: int
    topic_id: str
    title: str
    arc: str
    signal: str
    takeaway: str
    named_works: list[str]
    program: list[ProgramItem]
    implication: str


@dataclass
class ExecutiveEpisodePlan:
    deck_title: str
    creator: str | None
    facts: ProductionFacts
    shift_headline: str
    shift_detail: str
    topics: list[ExecutiveTopic]
    predictions: list[str]
    open_questions: list[str]
    target_minutes: int
    depth: str  # "short" | "long"


def plan_executive_episode(workspace: Path, deck: DeckRef, *, conference_label: str) -> ExecutiveEpisodePlan:
    deck_dir = workspace / DECKS_DIR / deck.id
    blueprint = read_json(deck_dir / "presentation_blueprint.json") if (deck_dir / "presentation_blueprint.json").is_file() else {}
    digests = read_json(workspace / EXECUTIVE_DIGESTS) if (workspace / EXECUTIVE_DIGESTS).is_file() else {}
    facts = load_production_facts(workspace, conference_label=conference_label, deck_id=deck.id)
    index_by_topic = {str(t.get("topic_id")): t for t in digests.get("topic_index") or [] if isinstance(t, Mapping)}
    detail_by_topic = {str(t.get("topic_id")): t for t in digests.get("topics") or [] if isinstance(t, Mapping)}
    manifest_topics = {t.id: t for t in deck_scope(workspace, executive_deck_ids=())}
    ordered: list[tuple[str, str]] = []
    for arc in digests.get("arcs") or []:
        if isinstance(arc, Mapping):
            ordered.extend((str(tid), _clean(arc.get("name"))) for tid in arc.get("topic_ids") or [])
    placed = {tid for tid, _arc in ordered}
    ordered.extend((tid, "Other topics") for tid in manifest_topics if tid not in placed)
    event_index = load_event_index(workspace)
    topics: list[ExecutiveTopic] = []
    for order, (topic_id, arc_name) in enumerate(ordered, start=1):
        entry = index_by_topic.get(topic_id, {})
        detail = detail_by_topic.get(topic_id, {})
        short = detail.get("short") if isinstance(detail.get("short"), Mapping) else {}
        evidence = short.get("evidence_slide") if isinstance(short.get("evidence_slide"), Mapping) else {}
        bigpicture = short.get("bigpicture_slide") if isinstance(short.get("bigpicture_slide"), Mapping) else {}
        manifest_topic = manifest_topics.get(topic_id)
        topics.append(
            ExecutiveTopic(
                order=order,
                topic_id=topic_id,
                title=_clean(entry.get("title") or (manifest_topic.title if manifest_topic else topic_id)),
                arc=arc_name,
                signal=_clean(entry.get("one_liner") or detail.get("one_liner")),
                takeaway=_conference_voice(_clean(detail.get("central_takeaway")), conference_label),
                named_works=[
                    _clean(src.get("title"))
                    for src in evidence.get("named_sources") or []
                    if isinstance(src, Mapping) and _clean(src.get("title"))
                ],
                program=program_highlights(topic_program(workspace, topic_id, event_index), limit=2),
                implication=_clean(bigpicture.get("implication")),
            )
        )
    central = digests.get("central_message") if isinstance(digests.get("central_message"), Mapping) else {}
    minutes = blueprint.get("target_duration_minutes")
    return ExecutiveEpisodePlan(
        deck_title=deck.title,
        creator=facts.creator,
        facts=facts,
        shift_headline=_clean(central.get("headline")),
        shift_detail=_conference_voice(_clean(central.get("subtitle")), conference_label),
        topics=topics,
        predictions=[_clean(p.get("prediction")) for p in digests.get("predictions") or [] if isinstance(p, Mapping) and _clean(p.get("prediction"))],
        open_questions=[_clean(q.get("question")) for q in digests.get("open_questions") or [] if isinstance(q, Mapping) and _clean(q.get("question"))],
        target_minutes=int(round(float(minutes))) if isinstance(minutes, (int, float)) and minutes > 0 else 25,
        depth="short" if deck.id.endswith("short") else "long",
    )


_ORDINAL_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)


def count_word(n: int) -> str:
    return _ORDINAL_WORDS[n - 1] if 1 <= n <= len(_ORDINAL_WORDS) else str(n)


def render_executive_brief(plan: ExecutiveEpisodePlan) -> str:
    c = plan.facts.conference_label
    n = len(plan.topics)
    total = count_word(n)
    lines = [
        f"# Episode brief — {plan.deck_title} ({c})",
        "",
        "Run sheet for the hosts. Follow it closely. Never mention this brief, the notebook, or any source by name on air.",
        "",
        "## Credit",
        f"This episode was created by {plan.creator}." if plan.creator else "Credit the creator named in the notes.",
        "",
        "## How the material was made (say this near the start, in your own words)",
        plan.facts.how_made(),
        "",
        "## Opening, in this order, within the first two minutes",
        f"1. Say this is the executive readout of {c}" + (f", created by {plan.creator}." if plan.creator else "."),
        "2. Explain how the material was made (above), plainly and briefly.",
        f"3. Say that the hosts did not attend {c}.",
        f"4. State the conference-wide shift in one plain sentence: {plan.shift_headline}.",
        f"5. Name all {total} topics the episode will cover, in the order below.",
        "Do not read titles, headings, or thesis statements aloud as a list of labels. Do not claim to have attended, walked the halls, or sat in sessions.",
        "",
        "## The conference-wide shift (the through-line; return to it at every transition)",
        plan.shift_headline,
    ]
    if plan.shift_detail:
        lines.append(plan.shift_detail)
    per_topic = "about ninety seconds, one named work or workshop" if plan.depth == "short" else "three to four minutes, two or three named works and a workshop"
    lines += [
        "",
        f"## All {total} topics, in order — every one gets its own segment ({per_topic})",
        f"Before each topic, say which of the {total} it is (for example 'topic five of {total}') and connect it to the topic before it. "
        "When entering a new arc, name the arc and how it advances the conference-wide shift.",
    ]
    current_arc = None
    for topic in plan.topics:
        if topic.arc != current_arc:
            current_arc = topic.arc
            lines += ["", f"### Arc: {topic.arc}"]
        lines.append(f"#### Topic {topic.order} of {n}: {topic.title}")
        if topic.signal:
            lines.append(f"- Signal: {topic.signal}")
        if topic.takeaway:
            lines.append(f"- Takeaway: {topic.takeaway}")
        if topic.named_works:
            lines.append(f"- Named works: {'; '.join(topic.named_works)}")
        if topic.program:
            lines.append(f"- Workshops and sessions: {'; '.join(f'{p.title} ({p.category})' for p in topic.program)}")
        if topic.implication:
            lines.append(f"- For a research leader: {topic.implication}")
        if topic.order > 1:
            previous = plan.topics[topic.order - 2]
            if previous.arc != topic.arc:
                lines.append(f"- Transition in: leave the arc '{previous.arc}' and enter '{topic.arc}'; say why this arc comes next in the shift.")
            else:
                lines.append(f"- Transition in: connect '{previous.signal or previous.title}' to '{topic.signal or topic.title}'.")
    if plan.predictions:
        lines += ["", "## What to watch next"] + [f"- {p}" for p in plan.predictions]
    if plan.open_questions:
        lines += ["", "## Open questions"] + [f"- {q}" for q in plan.open_questions]
    lines += [
        "",
        "## Close",
        "Name the two or three decisions a research leader should make differently this year because of the shift, "
        + (f"then credit {plan.creator} again." if plan.creator else "then give a brief credit."),
    ]
    return "\n".join(lines) + "\n"


def render_executive_program_map(plan: ExecutiveEpisodePlan) -> str:
    c = plan.facts.conference_label
    lines = [
        f"# {plan.deck_title} — workshops and sessions behind each topic at {c}",
        "",
        "A small language model read every recorded talk end to end and assigned its transcript windows to topics; "
        "for each topic, these are the workshops and sessions it judged most relevant, with its own summaries.",
    ]
    for topic in plan.topics:
        lines += ["", f"## Topic {topic.order}: {topic.title}"]
        if not topic.program:
            lines.append("- No recorded workshop or session was assigned to this topic.")
        for item in topic.program:
            lines.append(f"- **{item.title}** ({item.category}, {item.window_count} relevant transcript windows)")
            for summary in item.summaries[:3]:
                lines.append(f"  - {_truncate(summary, 240)}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Source sets
# --------------------------------------------------------------------------


def build_topic_source_set(
    workspace: Path,
    deck: DeckRef,
    join: CorpusJoin,
    *,
    conference_label: str,
    audio_root: str = AUDIO_ROOT,
) -> DeckSourceSet:
    out = output_dir(workspace, deck.id, audio_root=audio_root)
    stager = _Stager(workspace, out, conference_label)
    result = DeckSourceSet(deck=deck, conference_label=conference_label)
    deck_dir = workspace / DECKS_DIR / deck.id

    report_path = workspace / (deck.markdown_path or f"{ANALYSIS_DIR}/{deck.id}.md")
    if not report_path.is_file():
        result.blockers.append({"code": "notebooklm_analysis_report_missing", "path": str(report_path.relative_to(workspace))})
    else:
        stager.add_file(kind="analysis_report", title=deck.title, origin_id=str(report_path.relative_to(workspace)), path=report_path)
        result.inputs["analysis_report_sha256"] = sha256_file(report_path)

    summary_path = workspace / ANALYSIS_DIR / "exhaustive_summaries" / f"{deck.id}__transcripts.md"
    if summary_path.is_file():
        stager.add_file(kind="exhaustive_summary", title=f"{deck.title} — talk synthesis", origin_id=str(summary_path.relative_to(workspace)), path=summary_path)
        result.inputs["exhaustive_summary_sha256"] = sha256_file(summary_path)

    for dd_path in sorted((workspace / ANALYSIS_DIR / "deep_dive_topics").glob(f"{deck.id}__dd*.md")):
        title = dd_path.stem.split("__", 1)[1].replace("__", " ").replace("_", " ")
        stager.add_file(kind="deep_dive", title=f"{deck.title} — {title}", origin_id=str(dd_path.relative_to(workspace)), path=dd_path)

    blueprint_path = deck_dir / "presentation_blueprint.json"
    if not blueprint_path.is_file():
        result.blockers.append({"code": "notebooklm_blueprint_missing", "path": str(blueprint_path.relative_to(workspace))})
        result.sources = stager.specs
        return result
    blueprint = read_json(blueprint_path)
    result.inputs["blueprint_sha256"] = sha256_file(blueprint_path)
    stager.add(kind="blueprint_spine", title=f"{deck.title} — editorial spine", origin_id=str(blueprint_path.relative_to(workspace)), text=render_blueprint_spine(blueprint, deck_title=deck.title))

    ledger_people: dict[str, list[str]] = {}
    ledger_path = deck_dir / "speaker_author_work_ledger.json"
    if ledger_path.is_file():
        result.inputs["ledger_sha256"] = sha256_file(ledger_path)
        for entry in read_json(ledger_path).get("entries") or []:
            if isinstance(entry, Mapping) and entry.get("source_id"):
                people = _names(entry.get("speakers"), 6) or _names(entry.get("authors"), 6)
                ledger_people.setdefault(collapse_source_id(str(entry["source_id"])), people)
    result.inputs["corpus_inventory_sha256"] = join.corpus_inventory_sha256

    works = cited_source_ids(blueprint)
    result.cited_works = works
    for work in works:
        source_id = work["source_id"]
        people = work["speakers"] or work["authors"] or ledger_people.get(source_id, [])
        if source_id.startswith("paper_abstract:"):
            sha = source_id.split(":", 1)[1]
            record, resolved_by = resolve_cited_record(join, source_id, work["title"])
            origin_id = f"paper:{sha}" if resolved_by == "id" else str((record or {}).get("artifact_id"))
            if resolved_by in ("artifact_hash", "exact_title"):
                result.inputs.setdefault("resolved_by_fallback", []).append({"source_id": source_id, "artifact_id": origin_id, "method": resolved_by})
            text_path = workspace / str(record.get("text_path")) if record and record.get("text_path") else None
            if record is None or text_path is None or not text_path.is_file() or record.get("indexable_text") is False:
                result.blockers.append({"code": "notebooklm_source_text_missing", "source_id": source_id, "title": work["title"]})
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            title = work["title"] or _clean(record.get("paper_title") or metadata.get("paper_title")) or sha[:12]
            authors = people or _names(record.get("authors"), 6)
            text = paper_text_with_header(text_path.read_text(encoding="utf-8"), title=title, authors=authors, conference_label=conference_label)
            stager.add(kind="paper_text", title=title, origin_id=origin_id, text=text)
        elif source_id.startswith("speech_transcript:"):
            video_id = source_id.split(":", 1)[1]
            caption_path = join.caption_json_by_video.get(video_id) or join.lane_transcript_by_key.get(video_id)
            payload = _load_transcript_payload(caption_path) if caption_path is not None and caption_path.is_file() else None
            if payload is None:
                # Not a caption: a corpus ASR transcript recorded on the
                # inventory under this very id.
                found, resolved_by = resolve_cited_record(join, source_id, work["title"])
                record = found or {}
                if resolved_by not in (None, "id"):
                    result.inputs.setdefault("resolved_by_fallback", []).append(
                        {"source_id": source_id, "artifact_id": str(record.get("artifact_id")), "method": resolved_by}
                    )
                transcript_path = workspace / str(record["indexed_text_path"]) if record.get("indexed_text_path") else None
                metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
                title = work["title"] or _clean(metadata.get("talk_title") or metadata.get("title")) or f"Talk {video_id[:12]}"
                text = (
                    corpus_transcript_to_text(transcript_path, title=title, people=people, conference_label=conference_label)
                    if transcript_path is not None and transcript_path.is_file() and record.get("indexable_text") is not False
                    else None
                )
                if text is None:
                    result.blockers.append({"code": "notebooklm_source_text_missing", "source_id": source_id, "title": work["title"]})
                    continue
                stager.add(kind="talk_transcript", title=title, origin_id=source_id, text=text)
                continue
            title = work["title"] or join.event_titles.get(video_id) or f"Talk {video_id}"
            text, stats = caption_json_to_text(payload, title=title, people=people, conference_label=conference_label)
            result.dedup_stats[video_id] = stats
            stager.add(kind="talk_transcript", title=title, origin_id=f"event_media_caption:{video_id}", text=text)
        elif source_id.startswith("historical_paper:"):
            # Pre-conference background (e.g. a 2015 paper behind an anecdote):
            # not conference material, so it is not uploaded; the blueprint
            # spine still carries its contribution note.
            result.inputs.setdefault("skipped_background_sources", []).append(source_id)
        else:
            result.blockers.append({"code": "notebooklm_source_kind_unsupported", "source_id": source_id, "title": work["title"]})

    # Appended last so existing notebook slots keep their titles on re-runs.
    if not result.blockers:
        program = topic_program(workspace, deck.id, load_event_index(workspace))
        if program:
            stager.add(
                kind=PROGRAM_MAP_KIND,
                title=f"{deck.title} — talks, workshops, and sessions",
                origin_id=f"llm-full-read/by-topic/{deck.id}",
                text=render_program_map(deck.title, program, conference_label=conference_label),
            )
        plan = plan_topic_episode(workspace, deck, result, conference_label=conference_label)
        result.episode_plan = plan
        stager.add(kind=EPISODE_BRIEF_KIND, title=f"Episode brief — {deck.title}", origin_id=f"episode-brief/{deck.id}", text=render_topic_brief(plan))
    stager.prune_stale()
    result.sources = stager.specs
    return result


def build_executive_source_set(
    workspace: Path,
    deck: DeckRef,
    join: CorpusJoin,
    *,
    conference_label: str,
    audio_root: str = AUDIO_ROOT,
) -> DeckSourceSet:
    out = output_dir(workspace, deck.id, audio_root=audio_root)
    stager = _Stager(workspace, out, conference_label)
    result = DeckSourceSet(deck=deck, conference_label=conference_label)
    deck_dir = workspace / DECKS_DIR / deck.id

    for kind, rel, title in (
        ("executive_summary", deck.markdown_path or f"{ANALYSIS_DIR}/executive_summary.md", f"{conference_label} executive summary"),
        ("trend_report", f"{ANALYSIS_DIR}/trend_report.md", f"{conference_label} trend report"),
    ):
        path = workspace / rel
        if not path.is_file():
            result.blockers.append({"code": "notebooklm_analysis_report_missing", "path": rel})
            continue
        stager.add_file(kind=kind, title=title, origin_id=rel, path=path)
        result.inputs[f"{kind}_sha256"] = sha256_file(path)

    digests_path = workspace / EXECUTIVE_DIGESTS
    if digests_path.is_file():
        digests = read_json(digests_path)
        result.inputs["digests_sha256"] = sha256_file(digests_path)
        stager.add(kind="topic_digest", title=f"{conference_label} cross-topic digest", origin_id=EXECUTIVE_DIGESTS, text=render_executive_digest(digests, conference_label=conference_label))
    else:
        result.blockers.append({"code": "notebooklm_executive_digests_missing", "path": EXECUTIVE_DIGESTS})

    frame_path = deck_dir / "presentation_frame.json"
    blueprint_path = deck_dir / "presentation_blueprint.json"
    frame = read_json(frame_path) if frame_path.is_file() else {}
    blueprint = read_json(blueprint_path) if blueprint_path.is_file() else None
    if blueprint_path.is_file():
        result.inputs["blueprint_sha256"] = sha256_file(blueprint_path)
    if frame or blueprint:
        stager.add(kind="frame_spine", title=f"{deck.title} — framing", origin_id=str(frame_path.relative_to(workspace)), text=render_frame_spine(frame, blueprint, deck_title=deck.title))

    for topic in deck_scope(workspace, executive_deck_ids=()):
        path = workspace / (topic.markdown_path or f"{ANALYSIS_DIR}/{topic.id}.md")
        if path.is_file():
            stager.add_file(kind="analysis_report", title=topic.title, origin_id=str(path.relative_to(workspace)), path=path)
        else:
            result.blockers.append({"code": "notebooklm_analysis_report_missing", "path": str(path.relative_to(workspace))})

    # Appended last so existing notebook slots keep their titles on re-runs.
    if not result.blockers:
        plan = plan_executive_episode(workspace, deck, conference_label=conference_label)
        result.episode_plan = plan
        if any(topic.program for topic in plan.topics):
            stager.add(kind=PROGRAM_MAP_KIND, title=f"{deck.title} — workshops and sessions by topic", origin_id=f"llm-full-read/by-topic/{deck.id}", text=render_executive_program_map(plan))
        stager.add(kind=EPISODE_BRIEF_KIND, title=f"Episode brief — {deck.title}", origin_id=f"episode-brief/{deck.id}", text=render_executive_brief(plan))
    stager.prune_stale()
    result.sources = stager.specs
    return result


def build_source_set(workspace: Path, deck: DeckRef, join: CorpusJoin, *, conference_label: str, audio_root: str = AUDIO_ROOT) -> DeckSourceSet:
    if deck.kind == "executive":
        return build_executive_source_set(workspace, deck, join, conference_label=conference_label, audio_root=audio_root)
    return build_topic_source_set(workspace, deck, join, conference_label=conference_label, audio_root=audio_root)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptBundle:
    long_text: str
    short_text: str
    target_minutes: int

    @property
    def long_sha256(self) -> str:
        return sha256_text(self.long_text)

    @property
    def short_sha256(self) -> str:
        return sha256_text(self.short_text)


NO_ATTENDANCE_RULE = (
    "Never claim to have attended, walked the halls, sat in sessions, or returned from the conference, and never say the listener sent you papers. "
    "Never read titles, headings, or thesis statements aloud as labels."
)
NO_COLD_OPEN_RULE = (
    "The very first words of the episode are the welcome and the credit in step 1: no cold open, no analogy, metaphor, or story before them."
)
RUN_SHEET_RULE = (
    'The source titled "Episode brief" is your run sheet for the opening, the segment order, the names to mention, and the close. '
    "Follow it closely, but never mention the brief, the notebook, or any source by name on air."
)
ATTRIBUTION_RULE = "Use every name exactly as given; invent no papers, people, workshops, venues, or numbers, and quote only figures that appear verbatim in the material."

# Segment claims, full findings, and topic signals live in the Episode brief
# source; the prompt carries only complete sentences so nothing is cut mid-thought.
_TOPIC_BUDGETS: tuple[dict[str, int], ...] = (
    {"segments": 8, "claim": 0, "works": 5, "program": 4, "findings": 3, "hook": 560},
    {"segments": 8, "claim": 0, "works": 4, "program": 4, "findings": 2, "hook": 560},
    {"segments": 8, "claim": 0, "works": 3, "program": 3, "findings": 1, "hook": 420},
    {"segments": 7, "claim": 0, "works": 2, "program": 3, "findings": 0, "hook": 320},
)


def compose_topic_prompt(plan: TopicEpisodePlan, budget: Mapping[str, int]) -> str:
    c, t = plan.facts.conference_label, plan.deck_title
    credit = f", created by {plan.creator}" if plan.creator else ""
    segments = "\n".join(
        f"{i}. {title}" + (f" — {_truncate(claim, budget['claim'])}" if claim and budget["claim"] else "")
        for i, (title, claim) in enumerate(plan.segments[: budget["segments"]], start=1)
    )
    parts = [
        f'Create a deep-dive conversation about "{t}" at {c} for technical leaders and researchers. Define jargon once, in one sentence.',
        "",
        RUN_SHEET_RULE,
        "",
        "Open in this order, in your own natural words, within the first ninety seconds:",
        NO_COLD_OPEN_RULE,
        f"1. Say this is a deep dive into {t} from {c}{credit}.",
        f"2. Explain how the material was made: {plan.facts.how_made_short()}.",
        f"3. Say plainly that the hosts did not attend {c}.",
        "4. Hook the listener with the central shift" + (f": {_truncate(plan.hook, budget['hook'])}" if plan.hook else "."),
        "5. Give a one-sentence roadmap.",
        NO_ATTENDANCE_RULE,
        "",
        "Then cover, in order:",
        segments,
        "",
        "Attribute every claim: papers with their authors and talks with their speakers"
        + (f", for example {'; '.join(plan.works[: budget['works']])}" if plan.works[: budget["works"]] else "")
        + ".",
    ]
    if plan.program[: budget["program"]]:
        parts.append(
            "Bring these workshops and sessions into the story where their discussion fits, naming them: "
            + "; ".join(f"{item.title} ({item.category})" for item in plan.program[: budget["program"]])
            + "."
        )
    parts.append(ATTRIBUTION_RULE)
    if plan.findings[: budget["findings"]]:
        headlines = [re.split(r"(?<=\.)\s", f, maxsplit=1)[0] for f in plan.findings[: budget["findings"]]]
        parts += ["", "Emphasize:"] + [f"- {h}" for h in headlines]
    parts += [
        "",
        "Close with what this changes for someone building or funding this work"
        + (f", then credit {plan.creator} again" if plan.creator else "")
        + f". Aim for about {plan.target_minutes} minutes.",
    ]
    return "\n".join(parts).strip() + "\n"


def compose_topic_short_prompt(plan: TopicEpisodePlan) -> str:
    c, t = plan.facts.conference_label, plan.deck_title
    credit = f" created by {plan.creator}" if plan.creator else ""
    program = "; ".join(item.title for item in plan.program[:3])
    text = (
        f'Deep dive on "{t}" at {c}{credit}, for technical leaders. Follow the "Episode brief" source as your run sheet and never mention it. '
        f"Start immediately with the welcome and the credit (no cold open), then say {plan.facts.how_made_short()}, and that the hosts did not attend. "
        f"Cover: {', '.join(title for title, _ in plan.segments[:7])}. "
        f"Name papers with authors and talks with speakers" + (f", and bring in {program}" if program else "") + ". "
        "Invent nothing. Never claim attendance or read titles aloud. "
        + (f"Close by crediting {plan.creator} again." if plan.creator else "")
    )
    return _clean(text) + "\n"


def compose_executive_prompt(plan: ExecutiveEpisodePlan, budget: Mapping[str, int]) -> str:
    c, t = plan.facts.conference_label, plan.deck_title
    n = len(plan.topics)
    total = count_word(n)
    credit = f", created by {plan.creator}" if plan.creator else ""
    arcs: list[str] = []
    current: list[str] = []
    arc_name = None
    arc_index = 0
    for topic in plan.topics:
        if topic.arc != arc_name:
            if current:
                arcs.append(f"Arc {arc_index} — {arc_name}: " + "; ".join(current))
            arc_name, current = topic.arc, []
            arc_index += 1
        label = f"{topic.order}. {topic.title}"
        if budget["signal"] and topic.signal:
            label += f" ({_truncate(topic.signal, budget['signal'])})"
        current.append(label)
    if current:
        arcs.append(f"Arc {arc_index} — {arc_name}: " + "; ".join(current))
    depth = (
        "Keep each topic to about ninety seconds: its signal, one named paper, talk, or workshop behind it, and why it matters for a research leader. This is the short readout; keep the whole episode near the target length."
        if plan.depth == "short"
        else "For each topic give its signal, two or three named papers or talks and a workshop behind it, and why it matters for a research leader."
    )
    parts = [
        f'Create a deep-dive conversation that covers all {total} topics of {c} for technical leaders and researchers: "{t}". Define jargon once, in one sentence.',
        "",
        RUN_SHEET_RULE,
        "",
        "Open in this order, in your own natural words, within the first two minutes:",
        NO_COLD_OPEN_RULE,
        f"1. Say this is the executive readout of {c}{credit}.",
        f"2. Explain how the material was made: {plan.facts.how_made_short()}.",
        f"3. Say plainly that the hosts did not attend {c}.",
        f"4. State the conference-wide shift in one plain sentence: {plan.shift_headline}.",
        f"5. Name all {total} topics you will cover, in order.",
        NO_ATTENDANCE_RULE,
        "",
        f"Cover every one of the {total} topics, one segment each, in this order:",
        "\n".join(arcs),
        "",
        f"Transitions: before each topic, say which of the {total} it is (\"topic five of {total}\") and connect it to the one before. "
        "When entering a new arc, name the arc and how it advances the conference-wide shift, and keep returning to that shift.",
        depth,
        ATTRIBUTION_RULE,
    ]
    if plan.predictions[: budget["predictions"]]:
        parts += ["", "What to watch next:"] + [f"- {_truncate(p, 320)}" for p in plan.predictions[: budget["predictions"]]]
    parts += [
        "",
        "Close with the two or three decisions a research leader should make differently this year"
        + (f", then credit {plan.creator} again" if plan.creator else "")
        + f". Aim for about {plan.target_minutes} minutes and do not drop topics to save time.",
    ]
    return "\n".join(parts).strip() + "\n"


def compose_executive_short_prompt(plan: ExecutiveEpisodePlan) -> str:
    c = plan.facts.conference_label
    total = count_word(len(plan.topics))
    credit = f" created by {plan.creator}" if plan.creator else ""
    text = (
        f"Executive readout of {c}{credit}. Follow the \"Episode brief\" source as your run sheet and never mention it. "
        f"Start immediately with the welcome and the credit (no cold open), then say {plan.facts.how_made_short()}, and that the hosts did not attend. "
        f"State the shift: {plan.shift_headline}. Cover all {total} topics in order, announcing 'topic N of {total}' with a transition each time: "
        + "; ".join(topic.title for topic in plan.topics)
        + ". Invent nothing. Never claim attendance. "
        + (f"Close by crediting {plan.creator} again." if plan.creator else "")
    )
    return _clean(text) + "\n"


_EXECUTIVE_BUDGETS: tuple[dict[str, int], ...] = (
    {"signal": 0, "predictions": 2},
    {"signal": 0, "predictions": 1},
    {"signal": 0, "predictions": 0},
)


def _fit(compose: Callable[[Any, Mapping[str, int]], str], plan: Any, budgets: Sequence[Mapping[str, int]], cap: int) -> str:
    text = compose(plan, budgets[0])
    for budget in budgets[1:]:
        if len(text) <= cap:
            break
        text = compose(plan, budget)
    return text


def build_prompt(workspace: Path, deck: DeckRef, source_set: DeckSourceSet, *, conference_label: str) -> PromptBundle:
    plan = source_set.episode_plan
    if deck.kind == "executive":
        if not isinstance(plan, ExecutiveEpisodePlan):
            plan = plan_executive_episode(workspace, deck, conference_label=conference_label)
        return PromptBundle(
            long_text=_fit(compose_executive_prompt, plan, _EXECUTIVE_BUDGETS, LONG_PROMPT_CAP),
            short_text=compose_executive_short_prompt(plan),
            target_minutes=plan.target_minutes,
        )
    if not isinstance(plan, TopicEpisodePlan):
        plan = plan_topic_episode(workspace, deck, source_set, conference_label=conference_label)
    return PromptBundle(
        long_text=_fit(compose_topic_prompt, plan, _TOPIC_BUDGETS, LONG_PROMPT_CAP),
        short_text=compose_topic_short_prompt(plan),
        target_minutes=plan.target_minutes,
    )


def default_length_for(deck: DeckRef) -> str:
    """Topic decks and the long executive readout are ``long``; the short readout is ``default``.

    At ``long`` NotebookLM ignored the short readout's time target and produced
    a 48-minute episode, as long as the extended readout. Coverage of every
    topic is enforced by the brief's per-topic budget and the listening review,
    not by stretching the episode.
    """

    if deck.kind == "executive" and deck.id.endswith("short"):
        return "default"
    return "long"


# --------------------------------------------------------------------------
# Engine seam
# --------------------------------------------------------------------------


class NotebookLMEngine(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def preflight(self) -> dict[str, Any]: ...

    def ensure_notebook(self, title: str, notebook_id: str | None, *, match_title: bool = False) -> tuple[str, bool]: ...

    def sync_sources(self, notebook_id: str, specs: Sequence[SourceSpec], workspace: Path, *, exclusive: bool = False) -> list[dict[str, Any]]: ...

    def generate_audio(
        self, notebook_id: str, *, instructions: str, audio_format: str, audio_length: str, language: str, source_ids: Sequence[str] | None = None
    ) -> dict[str, Any]: ...

    def download_audio(self, notebook_id: str, artifact_id: str | None, output_path: Path) -> Path: ...


def engine_package_version() -> str | None:
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_storage_state_path(*, profile: str | None = None, explicit: Path | None = None) -> Path:
    """Mirror notebooklm-py's storage precedence without importing it."""

    if explicit is not None:
        return explicit
    env_path = os.environ.get("NOTEBOOKLM_AUTH_JSON")
    if env_path:
        return Path(env_path)
    home = Path(os.environ.get("NOTEBOOKLM_HOME") or (Path.home() / ".notebooklm"))
    name = profile or os.environ.get("NOTEBOOKLM_PROFILE")
    if not name:
        config = home / "config.json"
        if config.is_file():
            try:
                name = str(json.loads(config.read_text(encoding="utf-8")).get("active_profile") or "")
            except (OSError, json.JSONDecodeError):
                name = ""
    name = name or "default"
    candidate = home / "profiles" / name / "storage_state.json"
    legacy = home / "storage_state.json"
    return candidate if candidate.exists() or not legacy.exists() else legacy


class RealNotebookLMEngine:
    """``notebooklm-py`` adapter. Every method opens one client session."""

    def __init__(
        self,
        *,
        profile: str | None = None,
        storage_path: Path | None = None,
        generation_timeout_s: float = 1800.0,
        source_wait_timeout_s: float = 300.0,
        upload_concurrency: int = 2,
        upload_max_attempts: int = 4,
        upload_backoff_s: float = 5.0,
    ) -> None:
        self.profile = profile
        self.storage_path = storage_path
        self.generation_timeout_s = generation_timeout_s
        self.source_wait_timeout_s = source_wait_timeout_s
        self.upload_concurrency = max(1, int(upload_concurrency))
        self.upload_max_attempts = max(1, int(upload_max_attempts))
        self.upload_backoff_s = float(upload_backoff_s)

    def describe(self) -> dict[str, Any]:
        return {"package": PACKAGE_NAME, "version": engine_package_version(), "backend": "web", "kind": "real", "profile": self.profile or os.environ.get("NOTEBOOKLM_PROFILE") or "default"}

    # -- session plumbing -------------------------------------------------
    def _run(self, fn: Callable[[Any], Any]) -> Any:
        import notebooklm  # noqa: WPS433 - deliberate lazy import
        from notebooklm import exceptions as nb_exc

        async def session() -> Any:
            async with notebooklm.NotebookLMClient.from_storage(
                path=str(self.storage_path) if self.storage_path else None,
                profile=self.profile,
            ) as client:
                return await fn(client)

        try:
            return asyncio.run(session())
        except nb_exc.RateLimitError as exc:
            raise NotebookLMQuotaBlocked(f"RateLimitError: {exc}") from exc
        except (nb_exc.AuthError, nb_exc.HeadlessReauthError) as exc:
            raise NotebookLMAuthBlocked(f"{type(exc).__name__}: {exc}") from exc
        except FileNotFoundError as exc:
            raise NotebookLMAuthBlocked(f"storage state not found: {exc}") from exc
        except nb_exc.NotebookLMError as exc:
            # ServerError (5xx), NetworkError, SourceProcessingError, timeouts, ...:
            # a deck-level failure, never a silent skip and never a crash.
            raise NotebookLMPodcastError(f"{type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        from notebooklm import exceptions as nb_exc

        return isinstance(exc, (nb_exc.ServerError, nb_exc.NetworkError, nb_exc.SourceTimeoutError))

    # -- protocol ----------------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        storage = resolve_storage_state_path(profile=self.profile, explicit=self.storage_path)
        if not storage.is_file():
            raise NotebookLMAuthBlocked(
                f"NotebookLM storage state missing for profile {self.profile or 'default'}; run `notebooklm login --browser-cookies chrome`"
            )

        async def probe(client: Any) -> dict[str, Any]:
            notebooks = await client.notebooks.list()
            return {"notebook_count": len(notebooks)}

        result = self._run(probe)
        return {"storage_state_present": True, "profile": self.profile or "default", "account_probe": "ok", **result}

    def ensure_notebook(self, title: str, notebook_id: str | None, *, match_title: bool = False) -> tuple[str, bool]:
        async def op(client: Any) -> tuple[str, bool]:
            if not notebook_id and match_title:
                # A shared notebook: every deck finds the same one by title.
                matches = [nb for nb in await client.notebooks.list() if str(getattr(nb, "title", "")) == title]
                if matches:
                    return str(matches[0].id), True
            if notebook_id:
                try:
                    existing = await client.notebooks.get_or_none(notebook_id)
                except Exception as exc:  # noqa: BLE001 - only permission-denied falls through
                    # The recorded notebook exists under another Google account.
                    if str(getattr(exc, "rpc_code", "")) != "7" and "permission denied" not in str(exc).lower():
                        raise
                    existing = None
                if existing is not None:
                    try:
                        await client.sources.list(str(existing.id))
                        return str(existing.id), True
                    except Exception as exc:  # noqa: BLE001 - only permission-denied falls through
                        # A notebook recorded under another Google account
                        # (the profile was re-logged in) is visible to the
                        # lookup but unusable: start a fresh one in this account.
                        if str(getattr(exc, "rpc_code", "")) != "7" and "permission denied" not in str(exc).lower():
                            raise
            created = await client.notebooks.create(title)
            return str(created.id), False

        return self._run(op)

    def sync_sources(self, notebook_id: str, specs: Sequence[SourceSpec], workspace: Path, *, exclusive: bool = False) -> list[dict[str, Any]]:
        async def op(client: Any) -> list[dict[str, Any]]:
            existing = await client.sources.list(notebook_id)
            by_title = {str(s.title): s for s in existing}
            wanted_titles = {spec.notebooklm_title for spec in specs}
            wanted_slots = {source_slot_prefix(spec.notebooklm_title) for spec in specs}
            for title, source in list(by_title.items()):
                # Exclusive (a notebook shared by every deck): the previous
                # deck's sources go, so the notebook never exceeds one deck's
                # source set; finished episodes stay as notebook artifacts.
                stale = title not in wanted_titles and (exclusive or (source_slot_prefix(title) in wanted_slots and "[sha:" in title))
                if stale:
                    await client.sources.delete(notebook_id, str(source.id))  # stale content in this slot
                    by_title.pop(title, None)
            rows: list[dict[str, Any]] = []
            pending: list[SourceSpec] = []
            for spec in specs:
                found = by_title.get(spec.notebooklm_title)
                if found is not None:
                    rows.append({"sha256": spec.sha256, "notebooklm_source_id": str(found.id), "status": "reused"})
                else:
                    pending.append(spec)
            semaphore = asyncio.Semaphore(self.upload_concurrency)

            async def recover_processing_failure(exc: BaseException) -> str | None:
                """A SourceProcessingError is often transient: the source turns ready
                moments later. Recheck it; if it is still broken, delete it so the
                retry re-adds a clean copy instead of a duplicate."""

                from notebooklm import exceptions as nb_exc

                if not isinstance(exc, nb_exc.SourceProcessingError):
                    return None
                source_id = str(getattr(exc, "source_id", "") or "")
                if not source_id:
                    match = re.search(r"Source ([0-9a-f-]{36})", str(exc))
                    source_id = match.group(1) if match else ""
                if not source_id:
                    return None
                await asyncio.sleep(self.upload_backoff_s * 4)
                try:
                    ready = await client.sources.wait_until_ready(notebook_id, source_id, timeout=self.source_wait_timeout_s)
                    return str(ready.id)
                except Exception:  # noqa: BLE001 - any failure means re-add from scratch
                    try:
                        await client.sources.delete(notebook_id, source_id)
                    except Exception:  # noqa: BLE001
                        pass
                    return None

            async def upload(spec: SourceSpec) -> dict[str, Any]:
                async with semaphore:
                    attempt = 0
                    while True:
                        attempt += 1
                        try:
                            added = await client.sources.add_file(
                                notebook_id,
                                str(workspace / spec.local_path),
                                "text/markdown",
                                wait=True,
                                wait_timeout=self.source_wait_timeout_s,
                                title=spec.notebooklm_title,
                            )
                            added_id = str(added.id)
                            break
                        except Exception as exc:  # noqa: BLE001 - classified below
                            recovered = await recover_processing_failure(exc)
                            if recovered:
                                added_id = recovered
                                break
                            retryable = self._is_transient(exc) or type(exc).__name__ == "SourceProcessingError"
                            if attempt >= self.upload_max_attempts or not retryable:
                                raise
                            await asyncio.sleep(self.upload_backoff_s * (3 ** (attempt - 1)))
                return {"sha256": spec.sha256, "notebooklm_source_id": added_id, "status": "added", "attempts": attempt}

            rows.extend(await asyncio.gather(*(upload(spec) for spec in pending)))
            return rows

        return self._run(op)

    def generate_audio(
        self, notebook_id: str, *, instructions: str, audio_format: str, audio_length: str, language: str, source_ids: Sequence[str] | None = None
    ) -> dict[str, Any]:
        async def op(client: Any) -> dict[str, Any]:
            from notebooklm import AudioFormat, AudioLength

            status = await client.artifacts.generate_audio(
                notebook_id,
                source_ids=list(source_ids) if source_ids else None,
                instructions=instructions,
                audio_format=AudioFormat[audio_format.replace("-", "_").upper()],
                audio_length=AudioLength[audio_length.upper()],
                language=language,
            )
            final = await client.artifacts.wait_for_completion(notebook_id, status.task_id, timeout=self.generation_timeout_s)
            return {
                "task_id": str(status.task_id),
                "artifact_id": str(status.task_id),
                "status": str(getattr(final, "status", "")),
                "error": getattr(final, "error", None),
                "error_code": getattr(final, "error_code", None),
            }

        result = self._run(op)
        status = str(result.get("status") or "").lower()
        if status != "completed":
            code = str(result.get("error_code") or "")
            message = f"generation ended with status={status!r} error={result.get('error')!r} code={code!r}"
            if code == "USER_DISPLAYABLE_ERROR":
                raise NotebookLMQuotaBlocked(message)
            raise NotebookLMPodcastError(message)
        return result

    def download_audio(self, notebook_id: str, artifact_id: str | None, output_path: Path) -> Path:
        async def op(client: Any) -> str:
            return await client.artifacts.download_audio(notebook_id, str(output_path), artifact_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = self._run(op)
        return Path(written) if written else output_path


class FakeNotebookLMEngine:
    """Deterministic stand-in for tests and ``--fake-engine`` proofs.

    ``fail_auth_on_deck``/``fail_quota_on_deck`` raise on the Nth
    ``generate_audio`` call (1-based) to exercise the halt paths;
    ``reject_long_prompt`` makes the first long prompt raise a validation-like
    error so the short-variant fallback is exercised.
    """

    def __init__(
        self,
        *,
        fail_auth_on_call: int | None = None,
        fail_quota_on_call: int | None = None,
        reject_long_prompt: bool = False,
        preflight_ok: bool = True,
    ) -> None:
        self.fail_auth_on_call = fail_auth_on_call
        self.fail_quota_on_call = fail_quota_on_call
        self.reject_long_prompt = reject_long_prompt
        self.preflight_ok = preflight_ok
        self.notebooks: dict[str, dict[str, Any]] = {}
        self.generate_calls = 0
        self.calls: list[tuple[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        return {"package": PACKAGE_NAME, "version": "fake", "backend": "fake", "kind": "fake"}

    def preflight(self) -> dict[str, Any]:
        self.calls.append(("preflight", None))
        if not self.preflight_ok:
            raise NotebookLMAuthBlocked("fake auth preflight failure")
        return {"storage_state_present": True, "profile": "fake", "account_probe": "ok", "notebook_count": len(self.notebooks)}

    def ensure_notebook(self, title: str, notebook_id: str | None, *, match_title: bool = False) -> tuple[str, bool]:
        self.calls.append(("ensure_notebook", title))
        if notebook_id and notebook_id in self.notebooks:
            return notebook_id, True
        if match_title:
            for existing_id, notebook in self.notebooks.items():
                if notebook["title"] == title:
                    return existing_id, True
        new_id = f"nb_{sha256_text(title)[:12]}"
        self.notebooks[new_id] = {"title": title, "sources": {}, "artifacts": []}
        return new_id, False

    def sync_sources(self, notebook_id: str, specs: Sequence[SourceSpec], workspace: Path, *, exclusive: bool = False) -> list[dict[str, Any]]:
        self.calls.append(("sync_sources", len(specs)))
        notebook = self.notebooks.setdefault(notebook_id, {"title": notebook_id, "sources": {}, "artifacts": []})
        wanted_titles = {spec.notebooklm_title for spec in specs}
        wanted_slots = {source_slot_prefix(spec.notebooklm_title) for spec in specs}
        for title in list(notebook["sources"]):
            if title not in wanted_titles and (exclusive or source_slot_prefix(title) in wanted_slots):
                notebook["sources"].pop(title)
                self.calls.append(("delete_source", title))
        rows = []
        for spec in specs:
            if spec.notebooklm_title in notebook["sources"]:
                rows.append({"sha256": spec.sha256, "notebooklm_source_id": notebook["sources"][spec.notebooklm_title], "status": "reused"})
                continue
            if not (workspace / spec.local_path).is_file():
                raise NotebookLMPodcastError(f"staged source missing: {spec.local_path}")
            source_id = f"src_{sha256_text(spec.notebooklm_title)[:10]}"
            notebook["sources"][spec.notebooklm_title] = source_id
            rows.append({"sha256": spec.sha256, "notebooklm_source_id": source_id, "status": "added"})
        return rows

    def generate_audio(
        self, notebook_id: str, *, instructions: str, audio_format: str, audio_length: str, language: str, source_ids: Sequence[str] | None = None
    ) -> dict[str, Any]:
        self.generate_calls += 1
        self.calls.append((
            "generate_audio",
            {"format": audio_format, "length": audio_length, "prompt_sha256": sha256_text(instructions), "source_ids": list(source_ids or [])},
        ))
        if self.fail_auth_on_call == self.generate_calls:
            raise NotebookLMAuthBlocked("fake auth failure")
        if self.fail_quota_on_call == self.generate_calls:
            raise NotebookLMQuotaBlocked("fake quota exhausted")
        if self.reject_long_prompt and len(instructions) > SHORT_PROMPT_CAP:
            raise NotebookLMPodcastError("instructions too long for customization")
        artifact_id = f"art_{sha256_text(notebook_id + instructions)[:12]}"
        self.notebooks.setdefault(notebook_id, {"title": notebook_id, "sources": {}, "artifacts": []})["artifacts"].append(artifact_id)
        return {"task_id": artifact_id, "artifact_id": artifact_id, "status": "completed", "error": None, "error_code": None}

    def download_audio(self, notebook_id: str, artifact_id: str | None, output_path: Path) -> Path:
        self.calls.append(("download_audio", artifact_id))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal MP4/M4A-looking header so container sniffing reports "m4a".
        payload = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00M4A mp42isom" + (artifact_id or "audio").encode("utf-8")
        output_path.write_bytes(payload)
        return output_path


def sniff_audio_container(path: Path) -> str:
    with path.open("rb") as handle:
        head = handle.read(16)
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "m4a"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    return "unknown"


# --------------------------------------------------------------------------
# Listening review: transcribe the episode locally and check it
# --------------------------------------------------------------------------

REVIEW_TASK_ID = "podcast.notebooklm_review_transcription"
TRANSCRIPT_FILENAME = "transcript.json"
TRANSCRIPT_TEXT_FILENAME = "transcript.txt"
REVIEW_FILENAME = "review.json"
TOPIC_OPENING_SECONDS = 240.0
EXECUTIVE_OPENING_SECONDS = 360.0
_MATCH_STOPWORDS = frozenset(
    """a an and are as at be by for from in into is it of on or the to with via vs versus beyond
    workshop workshops session sessions oral tutorial tutorials talk talks keynote challenge challenges international annual
    first second third fourth fifth sixth seventh eighth ninth tenth 1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th
    new trends using toward towards""".split()
)
_BASE_ATTENDANCE_PATTERNS = (
    r"walk(?:ed|ing) (?:the|those|these) halls",
    r"\bwe (?:were|was) (?:there|on the ground|at the conference)",
    r"\bon the ground at\b",
    r"\bsat in (?:on )?(?:the |those |these )?(?:sessions|talks|rooms)",
    r"\bwe attended\b",
    r"\b(?:just )?(?:got back|returned|came back) from the conference\b",
    r"\byou sent us\b",
    r"\bboth of us were\b",
)

Transcriber = Callable[[Path], list[dict[str, Any]]]


class NotebookLMReviewUnavailable(NotebookLMPodcastError):
    """The local transcriber cannot run, so the episode cannot be reviewed."""


def mlx_whisper_transcriber(model: str | None = None) -> Transcriber:
    """Local, zero-cost transcription; the model comes from model governance."""

    def transcribe(audio_path: Path) -> list[dict[str, Any]]:
        try:
            import mlx_whisper  # noqa: WPS433 - optional, darwin-arm64 only
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise NotebookLMReviewUnavailable(f"mlx_whisper is not installed: {exc}") from exc
        from conference_pipeline.model_governance import default_model

        repo = model or default_model(REVIEW_TASK_ID)
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=repo)
        return [
            {"start": float(seg.get("start") or 0.0), "end": float(seg.get("end") or 0.0), "text": _clean(seg.get("text"))}
            for seg in result.get("segments") or []
            if _clean(seg.get("text"))
        ]

    return transcribe


def _match_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _stem(token: str) -> str:
    return token[:6] if len(token) > 6 else token


def significant_tokens(title: str) -> list[str]:
    seen: list[str] = []
    for token in _match_tokens(title):
        if token in _MATCH_STOPWORDS or (len(token) < 3 and not re.search(r"\d", token)):
            continue
        if re.fullmatch(r"\d+", token) and len(token) != 4:
            continue
        stem = _stem(token)
        if stem not in seen:
            seen.append(stem)
    return seen


def distinctive_tokens(title: str, other_titles: Sequence[str]) -> list[str]:
    """Stems of ``title`` that no other title in the set uses."""

    others: set[str] = set()
    for other in other_titles:
        others.update(significant_tokens(other))
    own = significant_tokens(title)
    unique = [stem for stem in own if stem not in others]
    return unique or own


def mention_positions(
    transcript_tokens: Sequence[str],
    stems: Sequence[str],
    *,
    need: int,
    window: int = 60,
    required_any: Sequence[str] | None = None,
) -> list[int]:
    """Token indices where ``need`` distinct stems co-occur inside one window.

    When ``required_any`` is given, a window only counts if it contains at
    least one of those stems (a term unique to this title), so generic shared
    words such as "video" or "models" cannot satisfy a match on their own.
    Adjacent transcript tokens are also joined, so "multi modal" matches
    "multimodal" the way a transcriber might split it.
    """

    wanted = set(stems)
    required = set(required_any or ())
    if not wanted:
        return []
    need = max(1, min(need, len(wanted)))
    hits: list[tuple[int, str]] = []
    for i, token in enumerate(transcript_tokens):
        joined = token + transcript_tokens[i + 1] if i + 1 < len(transcript_tokens) else ""
        for candidate in (token, joined):
            if candidate and _stem(candidate) in wanted:
                hits.append((i, _stem(candidate)))
    positions: list[int] = []
    for a in range(len(hits)):
        found: set[str] = set()
        b = a
        while b < len(hits) and hits[b][0] - hits[a][0] <= window:
            found.add(hits[b][1])
            b += 1
        if len(found) >= need and (not required or found & required) and hits[a][0] not in positions:
            positions.append(hits[a][0])
    return positions


def count_separate_mentions(positions: Sequence[int], *, window: int = 60) -> int:
    """Collapse qualifying window starts that sit within one window of each other."""

    count, last = 0, None
    for position in sorted(positions):
        if last is None or position - last > window:
            count += 1
            last = position
    return count


def mentioned_in(transcript_tokens: Sequence[str], stems: Sequence[str], *, need: int, window: int = 60) -> bool:
    return bool(mention_positions(transcript_tokens, stems, need=need, window=window))


def topic_match_terms(title: str, other_titles: Sequence[str]) -> tuple[list[str], list[str]]:
    """(all significant stems, stems unique to this title among its siblings)."""

    return significant_tokens(title), distinctive_tokens(title, other_titles)


def _fold(text: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if ch.isalnum() or ch.isspace())


def _name_in_text(name: str, words: Sequence[str]) -> bool:
    """Fuzzy match of a spoken name that a transcriber may have mangled."""

    from difflib import SequenceMatcher

    folded_words = [_fold(w) for w in words]
    for part in _match_tokens(_fold(name)):
        if len(part) < 4:
            continue
        for word in folded_words:
            if len(word) >= 4 and SequenceMatcher(None, part, word).ratio() >= 0.72:
                return True
    return False


def _credit_phrase_names(name: str, text: str) -> bool:
    """A looser match anchored on a credit phrase ("created ... by <name>").

    Transcribers render uncommon names badly ("Nicolas Fezal-Leglou"); right
    after "created by" a whole-name similarity of 0.6 is still specific, while
    unrelated words score far lower.
    """

    from difflib import SequenceMatcher

    target = "".join(_match_tokens(_fold(name)))
    folded = _fold(text)
    for match in re.finditer(r"\b(?:created|made|produced|put together|brought to you)\b[^.?!]{0,80}?\bby\b\s+([^.?!,]{3,40})", folded):
        candidate = "".join(_match_tokens(match.group(1))[:3])
        for size in range(len(candidate), 3, -1):
            if SequenceMatcher(None, target, candidate[:size]).ratio() >= 0.6:
                return True
    return False


@dataclass
class ReviewExpectations:
    deck_kind: str
    creator: str | None
    conference_label: str
    topic_titles: list[str] = field(default_factory=list)
    program_titles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @property
    def sha256(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))


def review_expectations(deck: DeckRef, source_set: DeckSourceSet) -> ReviewExpectations:
    plan = source_set.episode_plan
    if isinstance(plan, ExecutiveEpisodePlan):
        return ReviewExpectations(
            deck_kind="executive",
            creator=plan.creator,
            conference_label=plan.facts.conference_label,
            topic_titles=[topic.title for topic in plan.topics],
        )
    if isinstance(plan, TopicEpisodePlan):
        return ReviewExpectations(
            deck_kind="topic",
            creator=plan.creator,
            conference_label=plan.facts.conference_label,
            program_titles=[item.title for item in plan.program if item.category in ("keynote", "workshop", "tutorial")],
        )
    return ReviewExpectations(deck_kind=deck.kind, creator=None, conference_label=source_set.conference_label)


def review_transcript(segments: Sequence[Mapping[str, Any]], expectations: ReviewExpectations) -> dict[str, Any]:
    """Check an episode transcript against the operator's listening requirements."""

    opening_seconds = EXECUTIVE_OPENING_SECONDS if expectations.deck_kind == "executive" else TOPIC_OPENING_SECONDS
    timed: list[tuple[str, float]] = [
        (token, float(s.get("start") or 0.0)) for s in segments for token in _match_tokens(_clean(s.get("text")))
    ]
    full_text = " ".join(_clean(s.get("text")) for s in segments)
    opening_text = " ".join(_clean(s.get("text")) for s in segments if float(s.get("start") or 0.0) <= opening_seconds)
    opening_words = _match_tokens(opening_text)
    tokens = _match_tokens(full_text)
    checks: dict[str, dict[str, Any]] = {}

    if expectations.creator:
        checks["attribution_in_opening"] = {
            "blocking": True,
            "passed": _name_in_text(expectations.creator, opening_words) or _credit_phrase_names(expectations.creator, opening_text),
            "detail": f"creator {expectations.creator!r} credited within the first {int(opening_seconds)}s (a short cold open before the credit is fine)",
        }
    lowered_opening = opening_text.lower()
    checks["disclaimer_in_opening"] = {
        "blocking": True,
        "passed": "language model" in lowered_opening and bool(re.search(r"\bread\b|distill", lowered_opening)),
        "detail": "opening says a language model read the material and it was distilled",
    }
    checks["did_not_attend_stated"] = {
        "blocking": False,
        "passed": bool(
            re.search(
                r"(?:did not|didn't|did n't|were not|weren't|never) (?:actually |physically )?(?:attend|there|physically)"
                r"|neither of us (?:actually |physically )?(?:attended|was there|were there)"
                r"|(?:we|the hosts) (?:were|was) not (?:at|there)",
                lowered_opening,
            )
        ),
        "detail": "opening says the hosts did not attend",
    }
    patterns = list(_BASE_ATTENDANCE_PATTERNS) + [
        r"\b(?:just )?(?:got back|returned|came back) from " + re.escape(expectations.conference_label.lower())
    ]
    lowered = full_text.lower()
    negation = re.compile(r"\b(?:not|never|no|didn't|did not|weren't|were not|wasn't|was not|aren't|neither|nor|without)\b[^.?!]{0,40}$")
    claims = sorted(
        {
            m.group(0)
            for pattern in patterns
            for m in re.finditer(pattern, lowered)
            # "we weren't walking the halls" is the denial the brief asks for, not a claim
            if not negation.search(lowered[max(0, m.start() - 60) : m.start()])
        }
    )
    checks["no_attendance_claims"] = {"blocking": True, "passed": not claims, "detail": claims or "none"}

    if expectations.deck_kind == "executive" and expectations.topic_titles:
        per_topic: dict[str, dict[str, Any]] = {}
        for title in expectations.topic_titles:
            stems, unique = topic_match_terms(title, [t for t in expectations.topic_titles if t != title])
            positions = mention_positions(tokens, stems, need=2, required_any=unique)
            after = [i for i in positions if timed and timed[i][1] > opening_seconds]
            per_topic[title] = {
                "in_opening": any(timed[i][1] <= opening_seconds for i in positions) if timed else False,
                "after_opening_mentions": count_separate_mentions(after),
                "first_after_opening_s": int(timed[after[0]][1]) if after else None,
            }
        missing = [title for title, info in per_topic.items() if info["after_opening_mentions"] == 0]
        checks["all_topics_covered"] = {
            "blocking": True,
            "passed": not missing,
            "detail": {"expected": len(expectations.topic_titles), "missing": missing, "per_topic": per_topic},
        }
        checks["topics_named_in_opening"] = {
            "blocking": False,
            "passed": all(info["in_opening"] for info in per_topic.values()),
            "detail": [title for title, info in per_topic.items() if not info["in_opening"]] or "all named",
        }
        checks["topics_discussed_not_just_named"] = {
            "blocking": False,
            "passed": all(info["after_opening_mentions"] >= 2 for info in per_topic.values()),
            "detail": [title for title, info in per_topic.items() if info["after_opening_mentions"] < 2] or "all discussed",
        }
        signposts = len(re.findall(r"\b(?:topic\s+)?(?:\w+|\d+)\s+of\s+(?:twelve|\d+)\b", lowered))
        checks["transitions_signposted"] = {
            "blocking": False,
            "passed": signposts >= max(1, len(expectations.topic_titles) // 2),
            "detail": {"signpost_count": signposts},
        }
    if expectations.deck_kind == "topic" and expectations.program_titles:
        named = [title for title in expectations.program_titles if mentioned_in(tokens, significant_tokens(title), need=2, window=25)]
        need = min(2, len(expectations.program_titles))
        checks["workshops_mentioned"] = {
            "blocking": True,
            "passed": len(named) >= need,
            "detail": {"named": named, "need": need, "candidates": expectations.program_titles},
        }
    failed = [name for name, check in checks.items() if check["blocking"] and not check["passed"]]
    warnings = [name for name, check in checks.items() if not check["blocking"] and not check["passed"]]
    return {
        "status": "pass" if not failed else "failed",
        "failed_checks": failed,
        "warnings": warnings,
        "checks": checks,
        "transcript_word_count": len(tokens),
        "duration_seconds": max((float(s.get("end") or 0.0) for s in segments), default=0.0),
    }


def review_episode(
    out_dir: Path,
    audio_path: Path,
    expectations: ReviewExpectations,
    *,
    transcriber: Transcriber,
) -> dict[str, Any]:
    segments = transcriber(audio_path)
    write_json_atomic(out_dir / TRANSCRIPT_FILENAME, {"segments": segments, "audio_sha256": sha256_file(audio_path)})
    write_text_if_changed(
        out_dir / TRANSCRIPT_TEXT_FILENAME,
        "\n".join(f"[{int(s['start']) // 60:02d}:{int(s['start']) % 60:02d}] {s['text']}" for s in segments) + "\n",
    )
    result = review_transcript(segments, expectations)
    result.update(
        {
            "audio_sha256": sha256_file(audio_path),
            "expectations": expectations.to_dict(),
            "expectations_sha256": expectations.sha256,
            "transcript_path": TRANSCRIPT_TEXT_FILENAME,
            "reviewed_at": utc_now(),
        }
    )
    write_json_atomic(out_dir / REVIEW_FILENAME, result)
    return result


# --------------------------------------------------------------------------
# Manifest, show notes, deck run
# --------------------------------------------------------------------------


def load_manifest(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, NotebookLMPodcastError):
        return None


def _source_rows(source_set: DeckSourceSet, prior: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    prior_by_sha: dict[str, dict[str, Any]] = {}
    for row in (prior or {}).get("sources") or []:
        if isinstance(row, Mapping) and row.get("sha256"):
            prior_by_sha[str(row["sha256"])] = dict(row)
    rows = []
    for spec in source_set.sources:
        previous = prior_by_sha.get(spec.sha256, {})
        rows.append(
            {
                "index": spec.index,
                "kind": spec.kind,
                "title": spec.title,
                "notebooklm_title": spec.notebooklm_title,
                "origin_id": spec.origin_id,
                "local_path": spec.local_path,
                "sha256": spec.sha256,
                "word_count": spec.word_count,
                "notebooklm_source_id": previous.get("notebooklm_source_id"),
                "status": "added" if previous.get("status") in ("added", "reused") else "planned",
            }
        )
    return rows


def write_show_notes(out_dir: Path, deck: DeckRef, source_set: DeckSourceSet, prompt: PromptBundle, manifest: Mapping[str, Any]) -> Path:
    lines = [f"# {deck.title}", "", f"{source_set.conference_label} — NotebookLM deep-dive podcast", ""]
    plan = source_set.episode_plan
    facts = getattr(plan, "facts", None)
    if getattr(plan, "creator", None):
        lines += [f"Created by {plan.creator}.", ""]
    if isinstance(facts, ProductionFacts):
        lines += ["## How this episode was made", "", facts.how_made(), ""]
    generation = manifest.get("generation") or {}
    audio = manifest.get("audio") or {}
    lines += [
        f"- Format: {generation.get('format')} / length: {generation.get('length')}",
        f"- Sources in the notebook: {len(source_set.sources)}",
        f"- Prompt variant used: {(manifest.get('prompt') or {}).get('variant_used')}",
    ]
    if audio.get("sha256"):
        lines.append(f"- Audio sha256: `{audio['sha256']}` ({audio.get('byte_count')} bytes, {audio.get('container')})")
    review = manifest.get("review") or {}
    if review.get("status"):
        lines.append(f"- Listening review: {review.get('status')}" + (f" (failed: {', '.join(review.get('failed_checks') or [])})" if review.get("failed_checks") else ""))
    if isinstance(plan, ExecutiveEpisodePlan):
        lines += ["", f"## The {len(plan.topics)} topics, in order"]
        lines += [f"{t.order}. {t.title}" + (f" — {t.signal}" if t.signal else "") for t in plan.topics]
    if isinstance(plan, TopicEpisodePlan) and plan.program:
        lines += ["", "## Workshops and sessions"]
        lines += [f"- {item.title} ({item.category})" for item in plan.program]
    if source_set.cited_works:
        lines += ["", "## Works discussed"]
        for work in source_set.cited_works:
            people = ", ".join(work.get("speakers") or work.get("authors") or [])
            lines.append(f"- {work.get('title')}" + (f" — {people}" if people else ""))
    lines += ["", "## Sources"]
    for spec in source_set.sources:
        lines.append(f"- {spec.index:02d}. {_kind_label(spec.kind)}: {spec.title} ({spec.word_count} words)")
    lines += ["", "## Steering prompt", "", "```", prompt.long_text.strip(), "```", ""]
    path = out_dir / SHOW_NOTES_FILENAME
    write_text_if_changed(path, "\n".join(lines))
    return path


@dataclass
class DeckOutcome:
    deck_id: str
    status: str  # succeeded | blocked | planned | halted
    out_dir: str
    manifest_path: str
    blockers: list[dict[str, Any]] = field(default_factory=list)
    notebook_id: str | None = None
    notebook_reused: bool | None = None
    source_count: int = 0
    sources_added: int = 0
    sources_reused: int = 0
    audio_sha256: str | None = None
    audio_byte_count: int | None = None
    audio_container: str | None = None
    prompt_variant: str | None = None
    prompt_sha256: str | None = None
    generation_skipped_reason: str | None = None
    network_calls_made: bool = False
    dedup_ratio_min: float | None = None
    review_status: str | None = None
    review_failed_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def run_deck(
    workspace: Path,
    deck: DeckRef,
    *,
    join: CorpusJoin,
    engine: NotebookLMEngine | None,
    mode: str,
    conference_label: str,
    audio_format: str = "deep-dive",
    audio_length: str | None = None,
    language: str = "en",
    prompt_override: str | None = None,
    prompt_variant: str = "auto",
    resume: bool = True,
    audio_root: str = AUDIO_ROOT,
    review: bool = True,
    transcriber: Transcriber | None = None,
    accept_review_findings: bool = False,
    force_regenerate: bool = False,
    accept_existing_episode: bool = False,
    acceptance_note: str | None = None,
    shared_notebook: str | None = None,
) -> DeckOutcome:
    """Produce (or plan) one deck's podcast. Never raises for per-deck problems;
    auth/quota halts propagate so the caller can stop the run.

    A live episode only succeeds after its listening review passes: the audio
    is transcribed locally and checked for the creator credit and the
    how-it-was-made disclaimer in the opening, no false attendance claims, the
    named workshops (topic decks), and every topic (executive decks). An
    unchanged, already-reviewed episode is never regenerated or re-reviewed.
    """

    if audio_format not in FORMATS:
        raise NotebookLMPodcastError(f"unsupported audio format {audio_format!r}")
    length = audio_length or default_length_for(deck)
    if length not in LENGTHS:
        raise NotebookLMPodcastError(f"unsupported audio length {length!r}")
    out_dir = output_dir(workspace, deck.id, audio_root=audio_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_FILENAME
    prior = load_manifest(out_dir) if resume else None
    outcome = DeckOutcome(deck_id=deck.id, status="planned", out_dir=str(out_dir.relative_to(workspace)), manifest_path=str(manifest_path.relative_to(workspace)))

    source_set = build_source_set(workspace, deck, join, conference_label=conference_label, audio_root=audio_root)
    outcome.source_count = len(source_set.sources)
    if source_set.dedup_stats:
        outcome.dedup_ratio_min = min(float(s["dedup_ratio"]) for s in source_set.dedup_stats.values())
    try:
        prompt = build_prompt(workspace, deck, source_set, conference_label=conference_label)
    except (OSError, NotebookLMPodcastError, KeyError, TypeError, ValueError) as exc:
        source_set.blockers.append({"code": "notebooklm_prompt_build_failed", "error": f"{type(exc).__name__}: {exc}"})
        prompt = PromptBundle(long_text="", short_text="", target_minutes=0)

    # A dry run next to a live episode must not overwrite that episode's record:
    # its manifest, the prompt it was actually generated with, and its show notes.
    protect_live_record = mode != "live" and (prior or load_manifest(out_dir) or {}).get("mode") == "live"
    write_text_if_changed(out_dir / "prompt_long.txt", prompt.long_text)
    write_text_if_changed(out_dir / "prompt_short.txt", prompt.short_text)
    if prompt_override is not None:
        candidates = [("operator_override", prompt_override)]
    elif prompt_variant == "short":
        candidates = [("short", prompt.short_text)]
    elif prompt_variant == "long":
        candidates = [("long", prompt.long_text)]
    else:
        candidates = [("long", prompt.long_text), ("short", prompt.short_text)]
    chosen_variant, chosen_text = candidates[0]
    if not protect_live_record:
        write_text_if_changed(out_dir / PROMPT_FILENAME, chosen_text)

    notebook_title = shared_notebook or f"{conference_label} — {deck.title}"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "deck_id": deck.id,
        "deck_kind": deck.kind,
        "deck_title": deck.title,
        "deck_number": deck.number,
        "conference_label": conference_label,
        "mode": mode,
        "generated_at": utc_now(),
        "engine": engine.describe() if engine is not None else {"package": PACKAGE_NAME, "version": engine_package_version(), "kind": "none"},
        # A shared notebook (one for every deck) is found by title; a prior
        # id recorded under a different notebook title is not this notebook.
        "notebook": {
            "id": (prior or {}).get("notebook", {}).get("id") if (prior or {}).get("notebook", {}).get("title") == notebook_title else None,
            "title": notebook_title,
            "shared": bool(shared_notebook),
            "reused": None,
        },
        "inputs": source_set.inputs,
        "sources": _source_rows(source_set, prior),
        "prompt": {
            "variant_used": chosen_variant,
            "long_sha256": prompt.long_sha256,
            "short_sha256": prompt.short_sha256,
            "used_sha256": sha256_text(chosen_text),
            "path": str((out_dir / PROMPT_FILENAME).relative_to(workspace)),
            "fallback_reason": None,
            "target_minutes": prompt.target_minutes,
        },
        "generation": {"format": audio_format, "length": length, "language": language, "task_id": None, "artifact_id": None, "requested_at": None, "completed_at": None, "status": "planned"},
        "audio": {"path": str((out_dir / AUDIO_FILENAME).relative_to(workspace)), "sha256": None, "byte_count": None, "container": None},
        "dedup_stats": source_set.dedup_stats,
        "cited_work_count": len(source_set.cited_works),
        "show_notes_path": str((out_dir / SHOW_NOTES_FILENAME).relative_to(workspace)),
        "blockers": list(source_set.blockers),
    }
    if protect_live_record:
        write_json_atomic(out_dir / DRY_RUN_MANIFEST_FILENAME, manifest)
    else:
        write_json_atomic(manifest_path, manifest)
        write_show_notes(out_dir, deck, source_set, prompt, manifest)
    outcome.prompt_variant = chosen_variant
    outcome.prompt_sha256 = manifest["prompt"]["used_sha256"]
    outcome.blockers = list(source_set.blockers)

    if source_set.blockers:
        outcome.status = "blocked"
        return outcome
    if mode != "live" or engine is None:
        outcome.status = "planned"
        return outcome

    audio_path = out_dir / AUDIO_FILENAME
    prior_acceptance = (prior or {}).get("operator_acceptance") or {}
    if accept_existing_episode and not force_regenerate:
        if not audio_path.is_file() or audio_path.stat().st_size <= 0:
            manifest["blockers"].append({"code": "notebooklm_accept_existing_episode_missing_audio"})
            write_json_atomic(manifest_path, manifest)
            outcome.status = "blocked"
            outcome.blockers = list(manifest["blockers"])
            return outcome
        prior_acceptance = {
            "audio_sha256": sha256_file(audio_path),
            "accepted_at": utc_now(),
            "note": acceptance_note or "operator accepted the existing episode",
        }

    # ---- resume decision ------------------------------------------------
    prior_audio = (prior or {}).get("audio") or {}
    prior_prompt = (prior or {}).get("prompt") or {}
    prior_generation = (prior or {}).get("generation") or {}
    prior_shas = sorted(str(r.get("sha256")) for r in (prior or {}).get("sources") or [] if isinstance(r, Mapping) and r.get("status") in ("added", "reused"))
    candidate_shas = {sha256_text(t) for _variant, t in candidates if t}
    reuse_audio = (
        resume
        and not force_regenerate
        and prior_generation.get("status") == "succeeded"
        and bool(prior_audio.get("sha256"))
        and audio_path.is_file()
        and sha256_file(audio_path) == prior_audio.get("sha256")
        and prior_prompt.get("used_sha256") in candidate_shas
        and prior_shas == source_set.source_sha_set
        and prior_generation.get("format") == audio_format
        and prior_generation.get("length") == length
    )
    accepted_episode = (
        not force_regenerate
        and bool(prior_acceptance.get("audio_sha256"))
        and audio_path.is_file()
        and sha256_file(audio_path) == prior_acceptance.get("audio_sha256")
    )
    if accepted_episode and not reuse_audio:
        # The operator kept this episode even though the prompt or sources have
        # moved on since it was generated; record what it actually is.
        reuse_audio = True
        manifest["operator_acceptance"] = dict(prior_acceptance)
        prior_generation = dict(prior_generation) if prior_generation.get("status") == "succeeded" else {
            "format": audio_format,
            "length": length,
            "language": language,
            "status": "succeeded",
            "task_id": None,
            "artifact_id": None,
            "requested_at": None,
            "completed_at": None,
            "note": "generation record reconstructed from operator acceptance of the existing audio",
        }
        prior_audio = {
            "path": str(audio_path.relative_to(workspace)),
            "sha256": prior_acceptance["audio_sha256"],
            "byte_count": audio_path.stat().st_size,
            "container": sniff_audio_container(audio_path),
        }
        if not prior_prompt.get("used_sha256"):
            prior_prompt = {**manifest["prompt"], "used_sha256": None, "variant_used": "unknown"}
    elif accepted_episode:
        manifest["operator_acceptance"] = dict(prior_acceptance)
    if reuse_audio:
        manifest["notebook"] = dict((prior or {}).get("notebook") or manifest["notebook"])
        manifest["notebook"]["reused"] = True
        manifest["sources"] = [dict(r) for r in (prior or {}).get("sources") or []]
        manifest["prompt"].update({"variant_used": prior_prompt.get("variant_used"), "used_sha256": prior_prompt.get("used_sha256"), "fallback_reason": prior_prompt.get("fallback_reason")})
        manifest["generation"] = dict(prior_generation)
        manifest["audio"] = dict(prior_audio)
        manifest["resume"] = {"generation_skipped_reason": "audio_and_inputs_unchanged"}
        write_json_atomic(manifest_path, manifest)
        outcome.notebook_id = manifest["notebook"].get("id")
        outcome.notebook_reused = True
        outcome.sources_reused = len(manifest["sources"])
        outcome.generation_skipped_reason = "operator_accepted_existing_episode" if accepted_episode else "audio_and_inputs_unchanged"
        manifest["resume"] = {"generation_skipped_reason": outcome.generation_skipped_reason}
    else:
        # ---- live path ----------------------------------------------------
        outcome.network_calls_made = True
        try:
            notebook_id, reused = engine.ensure_notebook(manifest["notebook"]["title"], manifest["notebook"].get("id"), match_title=bool(shared_notebook))
            manifest["notebook"].update({"id": notebook_id, "reused": reused, "created_at": manifest["notebook"].get("created_at") or utc_now()})
            outcome.notebook_id, outcome.notebook_reused = notebook_id, reused
            write_json_atomic(manifest_path, manifest)
            try:
                rows = engine.sync_sources(notebook_id, source_set.sources, workspace, exclusive=bool(shared_notebook))
            except NotebookLMPodcastError as exc:
                # A reused notebook owned by another Google account can pass the
                # lookup yet refuse writes; move this deck to a fresh notebook.
                if not (reused and "permission denied" in str(exc).lower()):
                    raise
                notebook_id, reused = engine.ensure_notebook(manifest["notebook"]["title"], None, match_title=bool(shared_notebook))
                manifest["notebook"].update({"id": notebook_id, "reused": False, "created_at": utc_now()})
                outcome.notebook_id, outcome.notebook_reused = notebook_id, False
                write_json_atomic(manifest_path, manifest)
                rows = engine.sync_sources(notebook_id, source_set.sources, workspace, exclusive=bool(shared_notebook))
        except (NotebookLMAuthBlocked, NotebookLMQuotaBlocked):
            write_json_atomic(manifest_path, manifest)
            raise
        except NotebookLMPodcastError as exc:
            manifest["blockers"].append({"code": "notebooklm_source_sync_failed", "error": _truncate(str(exc), 400)})
            write_json_atomic(manifest_path, manifest)
            outcome.status = "blocked"
            outcome.blockers = list(manifest["blockers"])
            return outcome
        by_sha = {str(r["sha256"]): r for r in rows}
        for row in manifest["sources"]:
            synced = by_sha.get(row["sha256"])
            if synced:
                row["notebooklm_source_id"] = synced.get("notebooklm_source_id")
                row["status"] = synced.get("status")
                row["added_at"] = row.get("added_at") or utc_now()
        outcome.sources_added = sum(1 for r in rows if r.get("status") == "added")
        outcome.sources_reused = sum(1 for r in rows if r.get("status") == "reused")
        write_json_atomic(manifest_path, manifest)

        generation: dict[str, Any] | None = None
        for attempt, (variant, text) in enumerate(candidates):
            manifest["generation"]["requested_at"] = utc_now()
            manifest["prompt"].update({"variant_used": variant, "used_sha256": sha256_text(text)})
            write_text_if_changed(out_dir / PROMPT_FILENAME, text)
            write_json_atomic(manifest_path, manifest)
            try:
                generation = engine.generate_audio(
                    notebook_id,
                    instructions=text,
                    audio_format=audio_format,
                    audio_length=length,
                    language=language,
                    # A shared notebook holds only this deck's sources after an
                    # exclusive sync; naming them still pins the episode to them.
                    source_ids=[str(r["notebooklm_source_id"]) for r in rows if r.get("notebooklm_source_id")] if shared_notebook else None,
                )
                break
            except (NotebookLMAuthBlocked, NotebookLMQuotaBlocked):
                manifest["generation"]["status"] = "halted"
                write_json_atomic(manifest_path, manifest)
                raise
            except NotebookLMPodcastError as exc:
                has_fallback = attempt + 1 < len(candidates)
                if has_fallback and PROMPT_FALLBACK_PATTERN.search(str(exc)):
                    manifest["prompt"]["fallback_reason"] = f"{variant} prompt rejected: {_truncate(str(exc), 200)}"
                    continue
                manifest["generation"]["status"] = "failed"
                manifest["blockers"].append({"code": "notebooklm_generation_failed", "error": _truncate(str(exc), 400)})
                write_json_atomic(manifest_path, manifest)
                outcome.status = "blocked"
                outcome.blockers = list(manifest["blockers"])
                return outcome
        assert generation is not None
        manifest["generation"].update({"task_id": generation.get("task_id"), "artifact_id": generation.get("artifact_id"), "status": "generated", "completed_at": utc_now()})
        write_json_atomic(manifest_path, manifest)

        written = engine.download_audio(notebook_id, generation.get("artifact_id"), audio_path)
        if written != audio_path and written.is_file():
            os.replace(written, audio_path)
        if not audio_path.is_file() or audio_path.stat().st_size <= 0:
            manifest["generation"]["status"] = "failed"
            manifest["blockers"].append({"code": "notebooklm_audio_download_missing"})
            write_json_atomic(manifest_path, manifest)
            outcome.status = "blocked"
            outcome.blockers = list(manifest["blockers"])
            return outcome
        manifest["audio"].update({"sha256": sha256_file(audio_path), "byte_count": audio_path.stat().st_size, "container": sniff_audio_container(audio_path)})
        manifest["generation"]["status"] = "succeeded"
        manifest.pop("resume", None)
        write_json_atomic(manifest_path, manifest)

    outcome.audio_sha256 = manifest["audio"]["sha256"]
    outcome.audio_byte_count = manifest["audio"]["byte_count"]
    outcome.audio_container = manifest["audio"]["container"]
    outcome.prompt_variant = manifest["prompt"]["variant_used"]
    outcome.prompt_sha256 = manifest["prompt"]["used_sha256"]

    # ---- listening review ---------------------------------------------------
    expectations = review_expectations(deck, source_set)
    prior_review = (prior or {}).get("review") or {}
    if not review:
        review_result: dict[str, Any] = {"status": "waived", "reason": "review disabled by operator (--no-review)", "audio_sha256": outcome.audio_sha256, "reviewed_at": utc_now()}
    elif (
        reuse_audio
        and prior_review.get("audio_sha256") == outcome.audio_sha256
        and (prior_review.get("expectations_sha256") == expectations.sha256 or accepted_episode)
        and prior_review.get("status") in ("pass", "accepted")
    ):
        review_result = dict(prior_review)
    else:
        try:
            review_result = review_episode(out_dir, audio_path, expectations, transcriber=transcriber or mlx_whisper_transcriber())
        except NotebookLMReviewUnavailable as exc:
            review_result = {"status": "unavailable", "error": _truncate(str(exc), 300), "audio_sha256": outcome.audio_sha256, "reviewed_at": utc_now()}
        if review_result.get("status") == "failed" and (accept_review_findings or accepted_episode):
            review_result["status"] = "accepted"
            review_result["accepted_findings"] = list(review_result.get("failed_checks") or [])
            review_result["accepted_at"] = utc_now()
    manifest["review"] = review_result
    outcome.review_status = review_result.get("status")
    outcome.review_failed_checks = list(review_result.get("failed_checks") or [])
    if review_result.get("status") in ("failed", "unavailable"):
        code = "notebooklm_podcast_review_failed" if review_result["status"] == "failed" else "notebooklm_podcast_review_unavailable"
        manifest["blockers"].append({"code": code, "failed_checks": outcome.review_failed_checks, "error": review_result.get("error")})
        write_json_atomic(manifest_path, manifest)
        write_show_notes(out_dir, deck, source_set, prompt, manifest)
        outcome.status = "blocked"
        outcome.blockers = list(manifest["blockers"])
        return outcome
    write_json_atomic(manifest_path, manifest)
    write_show_notes(out_dir, deck, source_set, prompt, manifest)
    outcome.status = "succeeded"
    return outcome


# --------------------------------------------------------------------------
# Multi-deck stage run
# --------------------------------------------------------------------------


def run_stage(
    workspace: Path,
    *,
    deck_ids: Sequence[str] | None,
    engine: NotebookLMEngine | None,
    mode: str,
    audio_format: str = "deep-dive",
    audio_length: str | None = None,
    language: str = "en",
    prompt_override: str | None = None,
    prompt_variant: str = "auto",
    resume: bool = True,
    audio_root: str = AUDIO_ROOT,
    executive_deck_ids: Sequence[str] = DEFAULT_EXECUTIVE_DECK_IDS,
    review: bool = True,
    transcriber: Transcriber | None = None,
    accept_review_findings: bool = False,
    force_regenerate: bool = False,
    accept_existing_episode: bool = False,
    acceptance_note: str | None = None,
    shared_notebook: str | None = None,
) -> dict[str, Any]:
    """Run every in-scope deck; returns a summary the runner turns into a receipt."""

    workspace = workspace.resolve()
    conference_label = conference_name(workspace)
    scope = deck_scope(workspace, executive_deck_ids=executive_deck_ids)
    known = {d.id: d for d in scope}
    wanted = [d.strip() for d in (deck_ids or []) if d.strip()]
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        raise NotebookLMPodcastError(f"unknown deck ids: {unknown}")
    selected = [known[d] for d in wanted] if wanted else scope
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "mode": mode,
        "conference_label": conference_label,
        "engine": engine.describe() if engine is not None else {"package": PACKAGE_NAME, "version": engine_package_version(), "kind": "none"},
        "deck_scope_full_count": len(scope),
        "deck_scope_filter": wanted,
        "deck_count": len(selected),
        "preflight": None,
        "decks": {},
        "halt": None,
        "generated_at": utc_now(),
    }
    if mode == "live" and engine is not None:
        summary["preflight"] = engine.preflight()  # raises NotebookLMAuthBlocked -> caller records
    join = load_corpus_join(workspace)
    for deck in selected:
        try:
            outcome = run_deck(
                workspace,
                deck,
                join=join,
                engine=engine,
                mode=mode,
                conference_label=conference_label,
                audio_format=audio_format,
                audio_length=audio_length,
                language=language,
                prompt_override=prompt_override,
                prompt_variant=prompt_variant,
                resume=resume,
                audio_root=audio_root,
                review=review,
                transcriber=transcriber,
                accept_review_findings=accept_review_findings,
                force_regenerate=force_regenerate,
                accept_existing_episode=accept_existing_episode,
                acceptance_note=acceptance_note,
                shared_notebook=shared_notebook,
            )
        except NotebookLMAuthBlocked as exc:
            summary["halt"] = {"code": "notebooklm_auth_failed", "deck_id": deck.id, "retryable": False, "error": _truncate(str(exc), 300)}
            summary["decks"][deck.id] = {"deck_id": deck.id, "status": "halted", "blockers": [summary["halt"]]}
            break
        except NotebookLMQuotaBlocked as exc:
            summary["halt"] = {
                "code": "notebooklm_quota_exhausted",
                "deck_id": deck.id,
                "retryable": True,
                "error": _truncate(str(exc), 300),
                "resume_hint": "rerun with --resume after the daily NotebookLM quota resets; completed decks are skipped by manifest sha",
            }
            summary["decks"][deck.id] = {"deck_id": deck.id, "status": "halted", "blockers": [summary["halt"]]}
            break
        summary["decks"][deck.id] = outcome.to_dict()
    if summary["halt"] is not None:
        for deck in selected:
            summary["decks"].setdefault(deck.id, {"deck_id": deck.id, "status": "not_attempted", "blockers": []})
    statuses = [d.get("status") for d in summary["decks"].values()]
    summary["deck_success_count"] = sum(1 for s in statuses if s == "succeeded")
    summary["deck_planned_count"] = sum(1 for s in statuses if s == "planned")
    summary["deck_blocker_count"] = sum(1 for s in statuses if s in ("blocked", "halted", "not_attempted"))
    summary["network_calls_made"] = any(bool(d.get("network_calls_made")) for d in summary["decks"].values()) or bool(summary["preflight"])
    summary["review_statuses"] = {k: v.get("review_status") for k, v in summary["decks"].items() if v.get("review_status")}
    summary["audio_sha256s"] = {k: v.get("audio_sha256") for k, v in summary["decks"].items() if v.get("audio_sha256") and v.get("status") == "succeeded"}
    summary["aggregate_sha256"] = sha256_text("\n".join(f"{k}:{v}" for k, v in sorted(summary["audio_sha256s"].items()))) if summary["audio_sha256s"] else None
    if summary["halt"] is not None:
        summary["status"] = "blocked"
    elif mode != "live":
        summary["status"] = "planned" if summary["deck_blocker_count"] == 0 else "blocked"
    else:
        summary["status"] = "succeeded" if summary["deck_blocker_count"] == 0 and summary["deck_success_count"] == len(selected) else "blocked"
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-deck NotebookLM podcasts (Audio Overviews).")
    parser.add_argument("--workspace", type=Path, required=True, help="Conference workspace, e.g. conferences/<slug>.")
    parser.add_argument("--deck-id", action="append", default=[], help="Restrict to a deck id (repeatable).")
    parser.add_argument("--deck-ids", default=None, help="Comma-separated deck ids.")
    parser.add_argument("--profile", default=None, help="notebooklm-py profile (default: NOTEBOOKLM_PROFILE or 'default').")
    parser.add_argument("--storage-path", type=Path, default=None, help="Explicit storage_state.json path.")
    parser.add_argument("--format", dest="audio_format", choices=FORMATS, default="deep-dive")
    parser.add_argument("--length", dest="audio_length", choices=LENGTHS, default=None, help="Default: long for every deck.")
    parser.add_argument("--language", default="en")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Stage sources and prompts only; no network (default).")
    mode.add_argument("--live", action="store_true", help="Generate through Google NotebookLM.")
    parser.add_argument("--ack-live-notebooklm", action="store_true", help="Required with --live: acknowledges quota-metered generation on the logged-in Google account.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Operator prompt override (recorded as operator_override).")
    parser.add_argument("--prompt-variant", choices=("auto", "long", "short"), default="auto")
    parser.add_argument("--audio-root", default=AUDIO_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--summary", action="store_true", help="Print the JSON summary.")
    parser.add_argument("--fake-engine", action="store_true", help="Test-only deterministic engine; refused with --live.")
    parser.add_argument("--no-review", action="store_true", help="Skip the local listening review (recorded as a waiver).")
    parser.add_argument("--accept-review-findings", action="store_true", help="Record failed review checks as operator-accepted instead of blocking.")
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate audio for the selected decks even when inputs are unchanged.")
    parser.add_argument(
        "--accept-existing-episode",
        action="store_true",
        help="Keep the audio already on disk for the selected decks as final, even if prompts or sources changed since; recorded as operator acceptance.",
    )
    parser.add_argument("--acceptance-note", default=None, help="Why the existing episode was accepted (recorded in the manifest).")
    parser.add_argument(
        "--shared-notebook",
        default=None,
        metavar="TITLE",
        help="Generate every deck's episode in ONE notebook with this title (found by title, created once); "
        "each deck's sync replaces the previous deck's sources, and every episode stays in the notebook.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    live = bool(args.live)
    if live and not args.ack_live_notebooklm:
        print("error: --live requires --ack-live-notebooklm", file=sys.stderr)
        return 2
    if live and args.fake_engine:
        print("error: --fake-engine cannot be combined with --live", file=sys.stderr)
        return 2
    deck_ids = list(args.deck_id) + ([d for d in str(args.deck_ids).split(",")] if args.deck_ids else [])
    prompt_override = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else None
    engine: NotebookLMEngine | None
    if args.fake_engine:
        engine = FakeNotebookLMEngine()
        mode = "fake-engine"
    elif live:
        engine = RealNotebookLMEngine(profile=args.profile, storage_path=args.storage_path)
        mode = "live"
    else:
        engine = None
        mode = "dry-run"
    exit_code = 0
    try:
        summary = run_stage(
            args.workspace,
            deck_ids=deck_ids,
            engine=engine,
            mode="live" if mode == "fake-engine" else mode,
            audio_format=args.audio_format,
            audio_length=args.audio_length,
            language=args.language,
            prompt_override=prompt_override,
            prompt_variant=args.prompt_variant,
            resume=not args.no_resume,
            audio_root=args.audio_root,
            review=not args.no_review and mode != "fake-engine",
            accept_review_findings=args.accept_review_findings,
            force_regenerate=args.force_regenerate,
            accept_existing_episode=args.accept_existing_episode,
            acceptance_note=args.acceptance_note,
            shared_notebook=args.shared_notebook,
        )
        if mode == "fake-engine":
            summary["mode"] = "fake-engine"
    except NotebookLMAuthBlocked as exc:
        summary = {"receipt_type": RECEIPT_TYPE, "status": "blocked", "mode": mode, "halt": {"code": "notebooklm_auth_preflight_failed", "retryable": False, "error": str(exc)}, "generated_at": utc_now()}
    except NotebookLMPodcastError as exc:
        summary = {"receipt_type": RECEIPT_TYPE, "status": "blocked", "mode": mode, "halt": {"code": "notebooklm_podcast_error", "retryable": False, "error": str(exc)}, "generated_at": utc_now()}
    if summary.get("status") not in ("succeeded", "planned"):
        exit_code = 2
    if args.receipt_path:
        write_json_atomic(args.receipt_path, summary)
    if args.summary or not args.receipt_path:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
