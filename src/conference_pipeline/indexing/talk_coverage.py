"""Talk coverage: every transcript on disk is admitted to the corpus inventory or excluded by name.

v6.0-Public-Release Phase 5.1 (spec ``notes/phase-5.1-spec.md`` section 6.6). NeurIPS 2025
and ICLR 2025 held hundreds of talk transcripts that no corpus inventory admitted, and
nothing noticed. This check reads the transcripts a workspace holds, in each format the
platform has written, and compares them with the ``speech_transcript`` records of
``storage/corpus/corpus_inventory.json``. It writes nothing and pins no count: the result
is the equation on-disk = admitted + excluded, with every remainder named.

Discovery is by fixed workspace-relative locations and file shapes, never by a config the
workspace could leave out:

========================================  =========================================================
format                                    one on-disk talk per
========================================  =========================================================
``transcription_receipt_v15``             ``storage/transcription-ocr/*transcription_ocr_receipts*.json``:
                                          a ``media_artifacts`` entry of class ``speech_transcript``
                                          whose ``transcript_path`` exists
``stage_transcription_inventory``         ``storage/transcription-ocr/transcription_ocr_inventory.json``
                                          (``output_inventory.speech_transcript_outputs``) and
                                          ``storage/transcription-ocr/*/inventory.json``
                                          (``caption_transcripts[]``, by ``caption_path``; its
                                          ``output_path`` also matches): a listed file that exists
``evidence_lane``                         each receipt in ``evidence_lanes.json``: a record whose
                                          transcript exists, and each coverage gap
``deep_dive_presentations``               ``exports/deep-dive/manifest.json``: a presentation file
                                          that exists
========================================  =========================================================

A talk is **admitted** when an inventory ``speech_transcript`` record without
``index_exclusion`` names it (``transcript_path``, ``source_path``, ``indexed_text_path``
or a ``source:<path>`` provenance entry); **excluded** when such a record carries
``index_exclusion``, or when it is a lane coverage gap, or when it is a lane record whose
video an admitted record already carries (``superseded``: lane precedence skipped it);
otherwise **unaccounted**.

Exit codes (``scripts/audit_talk_coverage.py``): 0 nothing unaccounted; 1 something
unaccounted, or talks on disk with no corpus inventory; 2 an unreadable input or a missing
workspace, with nothing counted.

What it cannot see: a transcript in a format outside these four. A new transcript
producer writes one of them, or binds its output as an evidence lane.
"""

from __future__ import annotations

import fnmatch
import json
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMATS = (
    "transcription_receipt_v15",
    "stage_transcription_inventory",
    "evidence_lane",
    "deep_dive_presentations",
)
SPEECH_FAMILY = "speech_transcript"
INVENTORY_PATH = Path("storage") / "corpus" / "corpus_inventory.json"
TRANSCRIPTION_DIR = Path("storage") / "transcription-ocr"
DEEP_DIVE_MANIFEST = Path("exports") / "deep-dive" / "manifest.json"
LANE_REGISTRY = TRANSCRIPTION_DIR / "evidence_lanes.json"
EVENT_INVENTORY_GLOB = "storage/acquisition/*source_inventory*.json"
NO_CORPUS_INVENTORY = "no_corpus_inventory"
SUPERSEDED = "superseded"


class _Unreadable(Exception):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(path)
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class OnDiskTalk:
    format: str
    identity: str
    item_kind: str | None
    paths: tuple[str, ...]
    gap_reason: str | None = None
    video_id: str | None = None


def _lexical(workspace: Path, path: Path | str) -> tuple[str, bool]:
    """``(workspace-relative path, inside)`` without resolving links: a workspace may link a
    folder into another store (a Drive folder), and that folder is still the workspace's
    (finding F14). Only ``..`` can leave the workspace."""
    root = os.path.normpath(os.path.abspath(workspace))
    full = os.path.normpath(os.path.join(root, os.fspath(path)))
    inside = full == root or full.startswith(root + os.sep)
    return (Path(os.path.relpath(full, root)).as_posix() if inside else Path(full).as_posix()), inside


def _rel(workspace: Path, path: Path) -> str:
    return _lexical(workspace, path)[0]


def _load(workspace: Path, path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _Unreadable(_rel(workspace, path), "unreadable") from exc
    except UnicodeDecodeError as exc:
        raise _Unreadable(_rel(workspace, path), "invalid_json") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Unreadable(_rel(workspace, path), "invalid_json") from exc


ABSENT, FILE, FOLDER, OTHER = "absent", "file", "folder", "other"


def _probe(workspace: Path, rel: Path | str) -> str:
    """What is at a workspace-relative path, following links: ``absent``, ``file``,
    ``folder`` or ``other``. A broken link, or a path the process may not inspect, is
    unreadable, never absent (review phase-5.1-r3 F1 and F2). ``Path.is_file`` and
    ``is_dir`` turn both into "absent" (or, on Python 3.12, raise), so the guard never
    calls them."""
    rel = Path(rel)
    path = workspace / rel
    try:
        info = os.stat(path)
    except FileNotFoundError as exc:
        if os.path.islink(path):
            raise _Unreadable(rel.as_posix(), "unreadable") from exc
        return ABSENT
    except NotADirectoryError:
        return ABSENT
    except OSError as exc:
        raise _Unreadable(rel.as_posix(), "unreadable") from exc
    if stat.S_ISDIR(info.st_mode):
        return FOLDER
    if stat.S_ISREG(info.st_mode):
        return FILE
    return OTHER


def _exists(workspace: Path, rel: Any) -> bool:
    """A listed file inside the workspace that is there; one the process may not read is an
    error, never counted (review phase-5.1-r1 F4)."""
    if not isinstance(rel, str) or not rel:
        return False
    norm, inside = _lexical(workspace, rel)
    path = workspace / norm
    if not inside or _probe(workspace, norm) != FILE:
        return False
    if not os.access(path, os.R_OK):
        raise _Unreadable(rel, "unreadable")
    return True


def _object(workspace: Path, path: Path) -> dict[str, Any]:
    """A control file must be a JSON object (review phase-5.1-r1 F3)."""
    payload = _load(workspace, path)
    if not isinstance(payload, dict):
        raise _Unreadable(_rel(workspace, path), "invalid_structure")
    return payload


def _list(workspace: Path, path: Path, payload: dict[str, Any], key: str, *, required: bool) -> list[Any]:
    value = payload.get(key)
    if value is None and not required:
        return []
    if not isinstance(value, list):
        raise _Unreadable(_rel(workspace, path), "invalid_structure")
    return value


def _listdir(workspace: Path, rel_dir: Path) -> list[str]:
    """A folder's names, sorted. A folder that exists but cannot be listed, a broken link, or
    a link that lists empty is unreadable or unverified, never empty, at every depth
    (review phase-5.1-r2 F2, r3 F1). ``Path.glob`` swallows listing errors, so discovery
    never uses it."""
    kind = _probe(workspace, rel_dir)
    if kind == ABSENT:
        return []
    if kind != FOLDER:
        raise _Unreadable(rel_dir.as_posix(), "invalid_structure")
    path = workspace / rel_dir
    try:
        names = sorted(os.listdir(path))
    except OSError as exc:
        raise _Unreadable(rel_dir.as_posix(), "unreadable") from exc
    if not names and os.path.islink(path):
        # A link into another store (a Drive folder) that shows nothing (finding F14).
        raise _Unreadable(rel_dir.as_posix(), "unverified_empty_link")
    return names


def _event_kinds(workspace: Path) -> dict[str, str]:
    kinds: dict[str, str] = {}
    folder = Path(EVENT_INVENTORY_GLOB).parent
    pattern = Path(EVENT_INVENTORY_GLOB).name
    for name in _listdir(workspace, folder):
        if not fnmatch.fnmatch(name, pattern):
            continue
        path = workspace / folder / name
        payload = _object(workspace, path)
        for item in _list(workspace, path, payload, "items", required=False):
            if isinstance(item, dict) and item.get("item_id") and item.get("kind"):
                kinds[str(item["item_id"])] = str(item["kind"])
    return kinds


def _receipt_talks(workspace: Path) -> Iterable[OnDiskTalk]:
    kinds: dict[str, str] | None = None
    for name in _listdir(workspace, TRANSCRIPTION_DIR):
        if not fnmatch.fnmatch(name, "*transcription_ocr_receipts*.json"):
            continue
        path = workspace / TRANSCRIPTION_DIR / name
        payload = _object(workspace, path)
        for artifact in _list(workspace, path, payload, "media_artifacts", required=True):
            if not isinstance(artifact, dict) or artifact.get("artifact_class") != SPEECH_FAMILY:
                continue
            rel = artifact.get("transcript_path")
            if not _exists(workspace, rel):
                continue
            if kinds is None:
                kinds = _event_kinds(workspace)
            parts = str(artifact.get("artifact_id") or "").split(":")
            item_id = parts[1] if len(parts) >= 2 else ""
            yield OnDiskTalk("transcription_receipt_v15", rel, kinds.get(item_id), (rel,))


def _stage_inventory_talks(workspace: Path) -> Iterable[OnDiskTalk]:
    base = workspace / TRANSCRIPTION_DIR
    candidates = [base / "transcription_ocr_inventory.json"]
    for name in _listdir(workspace, TRANSCRIPTION_DIR):
        if _probe(workspace, TRANSCRIPTION_DIR / name) == FOLDER and "inventory.json" in _listdir(
                workspace, TRANSCRIPTION_DIR / name):
            candidates.append(base / name / "inventory.json")
    for path in candidates:
        if _probe(workspace, _rel(workspace, path)) != FILE:
            continue
        payload = _object(workspace, path)
        output_inventory = payload.get("output_inventory")
        if output_inventory is not None and not isinstance(output_inventory, dict):
            raise _Unreadable(_rel(workspace, path), "invalid_structure")
        outputs = _list(workspace, path, output_inventory or {}, "speech_transcript_outputs", required=False)
        for rel in outputs:
            if _exists(workspace, rel):
                yield OnDiskTalk("stage_transcription_inventory", rel, None, (rel,))
        for entry in _list(workspace, path, payload, "caption_transcripts", required=False):
            if not isinstance(entry, dict):
                continue
            caption, output = entry.get("caption_path"), entry.get("output_path")
            # Either file on disk is a talk: the caption, or its materialized transcript when the
            # caption is gone (review phase-5.1-r1 F2).
            present = [p for p in (caption, output) if _exists(workspace, p)]
            if present:
                yield OnDiskTalk("stage_transcription_inventory", present[0], None,
                                 tuple(p for p in (caption, output) if isinstance(p, str) and p))


def _lane_talks(workspace: Path) -> Iterable[OnDiskTalk]:
    registry_file = workspace / LANE_REGISTRY
    if _probe(workspace, LANE_REGISTRY) != FILE:
        return
    registry = _object(workspace, registry_file)
    for lane in _list(workspace, registry_file, registry, "lanes", required=True):
        if not isinstance(lane, dict):
            continue
        receipt_rel, inside = _lexical(workspace, str(lane.get("receipt_path") or ""))
        if not inside:
            # A registry entry that leaves the workspace (review phase-5.1-r2 F3).
            raise _Unreadable(str(lane.get("receipt_path")), "invalid_structure")
        receipt_file = workspace / receipt_rel
        if _probe(workspace, receipt_rel) != FILE:
            raise _Unreadable(str(lane.get("receipt_path")), "lane_receipt_missing")
        receipt = _object(workspace, receipt_file)
        for record in _list(workspace, receipt_file, receipt, "records", required=True):
            if isinstance(record, dict) and _exists(workspace, record.get("transcript_path")):
                rel = record["transcript_path"]
                yield OnDiskTalk("evidence_lane", rel, record.get("item_kind"), (rel,),
                                 video_id=str(record.get("video_id") or "") or None)
        kinds = receipt.get("gap_kinds") if isinstance(receipt.get("gap_kinds"), dict) else {}
        for gap in _list(workspace, receipt_file, receipt, "coverage_gaps", required=False):
            if isinstance(gap, dict) and gap.get("video_id"):
                video_id = str(gap["video_id"])
                yield OnDiskTalk("evidence_lane", f"{lane.get('lane_id')}:{video_id}", kinds.get(video_id), (),
                                 gap_reason=str(gap.get("reason") or "unspecified"))


def _deep_dive_talks(workspace: Path) -> Iterable[OnDiskTalk]:
    manifest = workspace / DEEP_DIVE_MANIFEST
    if _probe(workspace, DEEP_DIVE_MANIFEST) != FILE:
        return
    payload = _object(workspace, manifest)
    for entry in _list(workspace, manifest, payload, "presentations", required=True):
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        rel = _rel(workspace, manifest.parent / str(entry["file"]))
        if _exists(workspace, rel):
            yield OnDiskTalk("deep_dive_presentations", f"deep_dive:{rel}", None, (rel,))


# Every location the guard reads. One that exists (or is a link) but cannot be listed is
# unreadable, never empty: a Drive-backed symlink the process may not open, or a Drive
# folder that is not mounted, would otherwise pass as "no talks" (finding F12).
LOCATIONS = (
    TRANSCRIPTION_DIR,
    INVENTORY_PATH.parent,
    DEEP_DIVE_MANIFEST.parent,
    Path("storage") / "acquisition",
)


def _check_locations(workspace: Path) -> None:
    for rel in LOCATIONS:
        _listdir(workspace, rel)


def discover_on_disk(workspace: Path) -> list[OnDiskTalk]:
    """Every on-disk talk in the four formats; raises ``_Unreadable`` on a bad file."""
    _check_locations(workspace)
    talks: list[OnDiskTalk] = []
    for reader in (_receipt_talks, _stage_inventory_talks, _lane_talks, _deep_dive_talks):
        talks.extend(reader(workspace))
    return talks


def _exclusion_reason(value: Any) -> str | None:
    """ICML 2026 writes ``index_exclusion`` as ``{"reason", "evidence"}``; others as a string.
    No nonblank reason names nothing (review phase-5.1-r1 F5)."""
    reason = value.get("reason") if isinstance(value, dict) else value
    return reason.strip() if isinstance(reason, str) and reason.strip() else None


def _inventory_index(workspace: Path) -> tuple[set[str], dict[str, str], set[str]] | None:
    path = workspace / INVENTORY_PATH
    if _probe(workspace, INVENTORY_PATH) != FILE:
        return None
    payload = _object(workspace, path)
    records = _list(workspace, path, payload, "artifact_records", required=True)
    admitted: set[str] = set()
    excluded: dict[str, str] = {}
    admitted_videos: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("source_family_id") != SPEECH_FAMILY:
            continue
        exclusion = record.get("index_exclusion")
        if exclusion is None and record.get("video_id"):
            admitted_videos.add(str(record["video_id"]))
        names = [record.get(key) for key in ("transcript_path", "source_path", "indexed_text_path")]
        provenance = record.get("provenance")
        if isinstance(provenance, list):
            names += [item[len("source:"):] for item in provenance if isinstance(item, str) and item.startswith("source:")]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            # Only an absent exclusion admits; a present one excludes when it names a reason and
            # otherwise accounts for nothing, empty or not (review phase-5.1-r2 F4).
            if exclusion is None:
                admitted.add(name)
            else:
                reason = _exclusion_reason(exclusion)
                if reason is not None:
                    excluded.setdefault(name, reason)
    return admitted, excluded, admitted_videos


def _report(workspace: Path, *, exit_code: int, **fields: Any) -> dict[str, Any]:
    base = {"workspace": workspace.as_posix(), "exit_code": exit_code, "on_disk": 0, "admitted": 0, "excluded": {},
            "unaccounted": [], "by_format": {}, "errors": [], "reason": None}
    base.update(fields)
    return base


def measure_talk_coverage(workspace: Path) -> dict[str, Any]:
    """The coverage report; its ``exit_code`` is the script's."""
    if not workspace.is_dir():
        return _report(workspace, exit_code=2, errors=[{"path": workspace.as_posix(), "reason": "workspace_missing"}])
    given = workspace
    # Every path below is joined to an absolute workspace; a relative one doubled its prefix
    # and dropped the deep-dive export (review phase-5.1-r2 F1).
    workspace = Path(os.path.abspath(workspace))
    try:
        talks = discover_on_disk(workspace)
        index = _inventory_index(workspace)
    except _Unreadable as exc:
        return _report(given, exit_code=2, errors=[{"path": exc.path, "reason": exc.reason}])

    admitted_names, excluded_names, admitted_videos = index if index is not None else (set(), {}, set())
    admitted = 0
    excluded: dict[str, int] = {}
    unaccounted: list[dict[str, Any]] = []
    by_format: dict[str, dict[str, int]] = {}
    for talk in talks:
        row = by_format.setdefault(talk.format, {"on_disk": 0, "admitted": 0, "excluded": 0, "unaccounted": 0})
        row["on_disk"] += 1
        reason = talk.gap_reason
        if reason is None and any(path in admitted_names for path in talk.paths):
            admitted += 1
            row["admitted"] += 1
            continue
        if reason is None:
            reason = next((excluded_names[path] for path in talk.paths if path in excluded_names), None)
        if reason is None and talk.format == "evidence_lane" and talk.video_id in admitted_videos:
            # Lane precedence: an earlier lane (or the base inventory) admitted this video, so
            # load_admitted_lane_records skipped this copy (its superseded_count).
            reason = SUPERSEDED
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            row["excluded"] += 1
            continue
        row["unaccounted"] += 1
        unaccounted.append({"format": talk.format, "identity": talk.identity, "item_kind": talk.item_kind})

    no_inventory = index is None and bool(talks)
    return _report(
        given,
        exit_code=1 if unaccounted or no_inventory else 0,
        on_disk=len(talks),
        admitted=admitted,
        excluded=dict(sorted(excluded.items())),
        unaccounted=unaccounted,
        by_format=dict(sorted(by_format.items())),
        reason=NO_CORPUS_INVENTORY if no_inventory else None,
    )


__all__ = ["FORMATS", "OnDiskTalk", "discover_on_disk", "measure_talk_coverage"]
