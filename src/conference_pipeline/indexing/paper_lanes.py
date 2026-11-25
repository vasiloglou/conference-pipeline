"""The paper-lane registry and the existing-corpus adapters of a workspace.

v6.0-Public-Release Phase 5 (spec section 6.4). A workspace that admits paper sets added
after its original corpus declares them in ``storage/papers/late_paper_lanes.json``:

    {
      "schema_version": 1,
      "existing_corpus": {"adapter": "<adapter name>", "path": "<workspace-relative path or glob>"},
      "lanes": [
        {"lane_id": "...", "unit_file": "storage/papers/late/<lane>/units.jsonl",
         "track": "main|workshop|...", "id_kind": "openreview|arxiv|proceedings|site",
         "content": "full_text|abstract", "authors_file": "<optional notes JSON>",
         "provenance": {"source": "...", "match_basis": "...", "auth_mode": "..."}}
      ]
    }

The adapters read a workspace's existing paper set in its own format; each is named by
format, never by conference, and chosen in the registry.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from conference_pipeline.indexing.paper_identity import ID_KINDS, identity_key, key_from_source

REGISTRY_PATH = Path("storage") / "papers" / "late_paper_lanes.json"
EXCLUSIONS_PATH = Path("storage") / "papers" / "paper_exclusions.json"
CONTENT_KINDS = frozenset({"full_text", "abstract"})
# ``auto``: an OpenReview-shaped id (10 characters of [A-Za-z0-9_-]) is an OpenReview id;
# anything else (a workshop site's or a schedule's id) is a site id.
LANE_ID_KINDS = ID_KINDS | {"auto"}
_OPENREVIEW_ID = re.compile(r"[A-Za-z0-9_-]{10}")
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


class PaperLaneError(ValueError):
    """The registry, a lane or an adapter input is missing or malformed."""


@dataclass(frozen=True)
class PaperLane:
    lane_id: str
    unit_file: str
    track: str
    id_kind: str
    content: str = "full_text"
    authors_file: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)
    track_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperSource:
    """One record from the existing corpus or a lane, before admission."""

    origin: str  # "existing" or a lane id
    unit_id: str
    keys: tuple[str, ...]
    title: str | None
    authors: tuple[str, ...]
    track: str
    content: str  # "metadata", "abstract" or "full_text" (decided by the admission)
    text: str
    source_path: Path
    text_path: Path | None
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class Registry:
    existing_adapter: str
    existing_path: str
    lanes: tuple[PaperLane, ...]


def load_registry(workspace: Path) -> Registry:
    path = workspace / REGISTRY_PATH
    if not path.is_file():
        raise PaperLaneError(f"no paper-lane registry at {REGISTRY_PATH.as_posix()}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperLaneError("the paper-lane registry is not readable JSON") from exc
    existing = payload.get("existing_corpus") if isinstance(payload, dict) else None
    if not isinstance(existing, dict) or existing.get("adapter") not in EXISTING_CORPUS_ADAPTERS:
        raise PaperLaneError("existing_corpus.adapter must name a known adapter")
    lanes: list[PaperLane] = []
    seen: set[str] = set()
    for raw in payload.get("lanes") or []:
        if not isinstance(raw, dict):
            raise PaperLaneError("a lane must be an object")
        lane = PaperLane(
            lane_id=str(raw.get("lane_id") or ""),
            unit_file=str(raw.get("unit_file") or ""),
            track=str(raw.get("track") or ""),
            id_kind=str(raw.get("id_kind") or ""),
            content=str(raw.get("content") or "full_text"),
            authors_file=raw.get("authors_file"),
            provenance={str(k): str(v) for k, v in (raw.get("provenance") or {}).items()},
            track_map={str(k): str(v) for k, v in (raw.get("track_map") or {}).items()},
        )
        if not lane.lane_id or lane.lane_id in seen:
            raise PaperLaneError(f"lane ids must be present and unique: {lane.lane_id!r}")
        if not lane.unit_file or not lane.track:
            raise PaperLaneError(f"lane {lane.lane_id} needs unit_file and track")
        if lane.id_kind not in LANE_ID_KINDS or lane.content not in CONTENT_KINDS:
            raise PaperLaneError(f"lane {lane.lane_id}: id_kind or content not recognized")
        seen.add(lane.lane_id)
        lanes.append(lane)
    return Registry(str(existing["adapter"]), str(existing.get("path") or ""), tuple(lanes))


def load_exclusions(workspace: Path) -> dict[str, str]:
    """``{identity key: reason}`` from ``storage/papers/paper_exclusions.json``, if present."""
    path = workspace / EXCLUSIONS_PATH
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for row in payload.get("exclusions") or []:
        key, reason = str(row.get("id") or ""), str(row.get("reason") or "")
        if not key or not reason or ":" not in key:
            raise PaperLaneError("each exclusion needs an identity key (<kind>:<id>) and a reason")
        out[key] = reason
    return out


# --- existing-corpus adapters --------------------------------------------------------


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def _clean_author(name: str) -> str:
    """An author name without any email address a source printed beside it (IF11)."""
    return re.sub(r"\s{2,}", " ", _EMAIL.sub("", name)).strip(" ,;()<>[]")


def _authors(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = [str(part) for part in value]
    else:
        return ()
    return tuple(cleaned for part in parts if (cleaned := _clean_author(part)))


def _openreview_acquisition_json(workspace: Path, path_text: str) -> Iterator[PaperSource]:
    """``{"papers": [{"id", "forum", "title", "authors", "abstract", "track"?}]}``."""
    path = workspace / path_text
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("papers") or []:
        forum = str(row.get("forum") or row.get("id") or "")
        if not forum:
            continue
        yield PaperSource(
            origin="existing",
            unit_id=forum,
            keys=(identity_key("openreview", forum),),
            title=row.get("title"),
            authors=_authors(row.get("authors")),
            track=str(row.get("track") or "main"),
            content="abstract",
            text=str(row.get("abstract") or ""),
            source_path=path,
            text_path=None,
            provenance={"source": path_text, "title": "existing_corpus", "authors": "existing_corpus"},
        )


_FIELD = re.compile(r"^(Title|Paper ID|Authors|Track):\s*(.*)$")


def _deep_dive_abstracts_txt(workspace: Path, path_text: str) -> Iterator[PaperSource]:
    """NotebookLM deep-dive paper files: '# Title' blocks with Title/Paper ID/Authors/Track lines."""
    files = sorted(workspace.glob(path_text))
    if not files:
        raise PaperLaneError(f"no existing-corpus files match {path_text}")
    for path in files:
        block: dict[str, str] = {}
        abstract: list[str] = []
        in_abstract = False

        def flush() -> Iterator[PaperSource]:
            paper_id = block.get("Paper ID", "").strip()
            if paper_id:
                yield PaperSource(
                    origin="existing",
                    unit_id=paper_id,
                    keys=(identity_key("openreview", paper_id),),
                    title=block.get("Title"),
                    authors=_authors(block.get("Authors", "")),
                    track=block.get("Track", "main").strip() or "main",
                    content="abstract",
                    text="\n".join(abstract).strip(),
                    source_path=path,
                    text_path=None,
                    provenance={"source": path.relative_to(workspace).as_posix(), "title": "existing_corpus", "authors": "existing_corpus"},
                )

        for line in path.read_text(encoding="utf-8").split("\n"):  # newline only (IF7)
            if line.startswith("# "):
                yield from flush()
                block, abstract, in_abstract = {}, [], False
                continue
            if in_abstract:
                abstract.append(line)
                continue
            if line.strip() == "Abstract:":
                in_abstract = True
                continue
            match = _FIELD.match(line)
            if match:
                block[match.group(1)] = match.group(2)
        yield from flush()


EXISTING_CORPUS_ADAPTERS: dict[str, Callable[[Path, str], Iterator[PaperSource]]] = {
    "openreview_acquisition_json": _openreview_acquisition_json,
    "deep_dive_abstracts_txt": _deep_dive_abstracts_txt,
}


def existing_sources(workspace: Path, registry: Registry) -> list[PaperSource]:
    return list(EXISTING_CORPUS_ADAPTERS[registry.existing_adapter](workspace, registry.existing_path))


# --- lanes ---------------------------------------------------------------------------


def _authors_index(workspace: Path, lane: PaperLane) -> dict[str, tuple[str, ...]]:
    """``{id: authors}`` from a receipt: ``{"notes": [{"id", "forum", "authors"}]}`` or
    ``{"papers": [{"key", "authors"}]}`` (a lookup receipt keyed by the unit's own id)."""
    if not lane.authors_file:
        return {}
    payload = json.loads((workspace / lane.authors_file).read_text(encoding="utf-8"))
    rows = (payload.get("notes") or payload.get("papers")) if isinstance(payload, dict) else payload
    index: dict[str, tuple[str, ...]] = {}
    for row in rows or []:
        authors = _authors(row.get("authors"))
        if not authors:
            continue
        for key in ("id", "forum", "key"):
            if row.get(key):
                index[str(row[key])] = authors
    return index


def lane_sources(workspace: Path, lane: PaperLane) -> list[PaperSource]:
    unit_file = workspace / lane.unit_file
    if not unit_file.is_file():
        raise PaperLaneError(f"lane {lane.lane_id}: no unit file {lane.unit_file}")
    authors_by_id = _authors_index(workspace, lane)
    out: list[PaperSource] = []
    # Records split on "\n" only: str.splitlines() also splits on U+2028, U+0085 and
    # others that may appear raw inside a JSON string (IF7).
    for number, line in enumerate(unit_file.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        unit = json.loads(line)
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            raise PaperLaneError(f"lane {lane.lane_id}: unit on line {number} has no unit_id")
        kind = lane.id_kind
        if kind == "auto":
            kind = "openreview" if _OPENREVIEW_ID.fullmatch(unit_id) and not unit_id.startswith("site_") else "site"
        keys = [identity_key(kind, unit_id)]
        url_match = _ARXIV_URL.search(str(unit.get("url") or ""))
        for extra in (
            key_from_source(str(unit.get("source") or "")),
            identity_key("arxiv", url_match.group(1)) if url_match else None,
        ):
            if extra and extra not in keys:
                keys.append(extra)
        text_path: Path | None = None
        if unit.get("path"):
            text_path = workspace / str(unit["path"])
            if text_path.is_symlink() or not text_path.is_file():
                raise PaperLaneError(f"lane {lane.lane_id}: unit {unit_id} text file missing")
            text = text_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(unit.get("text") or "")
        out.append(
            PaperSource(
                origin=lane.lane_id,
                unit_id=unit_id,
                keys=tuple(keys),
                title=unit.get("title"),
                authors=_authors(unit.get("authors")) or authors_by_id.get(unit_id, ()),
                track=lane.track_map.get(str(unit.get("track")), lane.track),
                content=lane.content,
                text=text,
                source_path=text_path or unit_file,
                text_path=text_path,
                provenance={**lane.provenance, "title": f"lane:{lane.lane_id}", "authors": "unit" if unit.get("authors") else ("notes" if unit_id in authors_by_id else "none")},
            )
        )
    return out
