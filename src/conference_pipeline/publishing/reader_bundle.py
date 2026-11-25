"""Read a release bundle's reader content: topics, decks, media, narratives, catalog.

v6.0-Public-Release Phase 4 (spec ``notes/phase-4-spec.md`` section 6.1, the reader
content contract). Every file this module hands to a page builder is listed in the bundle's
``CHECKSUMS.json`` and matches its sha256; a file that is not listed is never read, and a
listed file whose bytes differ stops the build (``bundle_file_not_in_checksums``).

Paths, all optional except ``RELEASE.json`` and ``CHECKSUMS.json``:

- ``decks/control/presentation_manifest.json``: topic ids, titles, order
- ``decks/payloads/<topic>/presentation.{html,pdf}``, ``*.mp4.cdn.json`` video sidecars
- ``audio/<topic>/**/*.mp3`` or ``*.m4a`` (inline), or their ``.cdn.json`` sidecars
- ``narratives/<topic>.md``
- ``catalog/papers.jsonl`` (``docs/public_catalog_row.schema.json``), ``catalog/topic_papers.json``,
  ``catalog/excluded_ids.json``, ``catalog/statistics.json``
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from conference_pipeline.publishing.public_index import validate_public_bundle_root

NOT_IN_CHECKSUMS = "bundle_file_not_in_checksums"
UNSAFE_PATH = "unsafe path in bundle"
UNSAFE_TOPIC_ID = "unsafe topic id"
TOPIC_TYPES = frozenset({"topic-deep-dive"})
_TOPIC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def is_plain_relative(rel: Any) -> bool:
    """A bundle path is a plain relative POSIX path: no ``..``, ``.``, empty part, leading
    slash, backslash or control character, so the string the copy rules inspect is the path
    that is opened (phase-4-r1 F1)."""
    if not isinstance(rel, str) or not rel or "\\" in rel or any(ord(c) < 32 for c in rel):
        return False
    parts = rel.split("/")
    return not rel.startswith("/") and all(p not in ("", ".", "..") for p in parts) and PurePosixPath(rel).as_posix() == rel


class ReaderBundleError(RuntimeError):
    """The bundle is not a public bundle, or a file it would hand on is not as checksummed."""


@dataclass(frozen=True)
class Video:
    url: str
    mime_type: str
    label: str


@dataclass(frozen=True)
class Topic:
    topic_id: str
    title: str
    deck_html: str | None
    deck_pdf: str | None
    videos: tuple[Video, ...]
    podcast: str | None  # a bundle-relative mp3, or an absolute CDN URL
    narrative: str | None
    papers: tuple[str, ...]
    deck_pptx: str | None = None  # a deck shipped as PowerPoint (6b, NeurIPS 2025)
    deck_text: str | None = None  # a deck shipped as slide Markdown (6b, NeurIPS 2025)


@dataclass
class ReaderBundle:
    root: Path
    release: dict[str, Any]
    checksums: dict[str, dict[str, Any]]
    topics: list[Topic]
    executive: list[Topic]
    catalog: list[dict[str, Any]]
    excluded_ids: set[str]
    statistics: dict[str, Any] | None
    index_manifest: dict[str, Any] | None
    ignored_not_in_checksums: list[str] = field(default_factory=list)
    rows_excluded: int = 0
    # The checksummed catalog/*.jsonl files read; empty means the release has no catalog,
    # which differs from a catalog with no rows (IF8).
    catalog_files: list[str] = field(default_factory=list)
    # index/ is written by the index stage after the export, so its files are not in
    # CHECKSUMS.json; they are verified against the index manifest's output_checksums (IF5).
    index_checksums: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def bundle_id(self) -> str:
        return str(self.release.get("bundle_id") or "")

    @property
    def conference_name(self) -> str:
        nested = self.release.get("release") if isinstance(self.release.get("release"), dict) else {}
        return str(nested.get("conference_name") or self.release.get("conference_name") or self.release.get("conference") or "")

    def listed(self, rel: str) -> bool:
        return rel in self.checksums or rel in self.index_checksums

    def verified(self, rel: str) -> Path:
        """The bundle path of a listed file whose bytes match its checksum."""
        entry = self.checksums.get(rel) or self.index_checksums.get(rel)
        if not is_plain_relative(rel):
            raise ReaderBundleError(f"{UNSAFE_PATH}: {rel!r}")
        path = self.root / rel
        if entry is None or path.is_symlink() or not path.is_file():
            raise ReaderBundleError(f"{NOT_IN_CHECKSUMS}: {rel}")
        try:  # a symlinked directory part would also leave the bundle
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            raise ReaderBundleError(f"{UNSAFE_PATH}: {rel!r}") from None
        if _sha256(path) != entry.get("sha256"):
            raise ReaderBundleError(f"{NOT_IN_CHECKSUMS}: {rel} (sha256 differs)")
        return path

    def read_text(self, rel: str) -> str:
        return self.verified(rel).read_text(encoding="utf-8")

    def read_json(self, rel: str) -> Any:
        return json.loads(self.read_text(rel))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _videos(bundle: ReaderBundle, prefix: str) -> tuple[Video, ...]:
    videos = []
    for rel in sorted(r for r in bundle.checksums if r.startswith(prefix) and r.endswith(".mp4.cdn.json")):
        sidecar = bundle.read_json(rel)
        url = str(sidecar.get("cdn_url") or "")
        if url.startswith("https://"):
            label = "Two-host conversation" if "notebooklm" in rel else "Narrated presentation"
            videos.append(Video(url=url, mime_type=str(sidecar.get("mime_type") or "video/mp4"), label=label))
    return tuple(videos)


DECK_TEXT_NAMES = ("slides.md", "5_draft_slides_with_images.md")
PODCAST_SUFFIXES = (".mp3", ".m4a")  # NotebookLM episodes are m4a (v6.0 Phase 6b)


def _podcast(bundle: ReaderBundle, topic_id: str) -> str | None:
    prefix = f"audio/{topic_id}/"
    inline = sorted(r for r in bundle.checksums if r.startswith(prefix) and r.endswith(PODCAST_SUFFIXES))
    if inline:
        return inline[0]
    sidecars = tuple(suffix + ".cdn.json" for suffix in PODCAST_SUFFIXES)
    for rel in sorted(r for r in bundle.checksums if r.startswith(prefix) and r.endswith(sidecars)):
        url = str(bundle.read_json(rel).get("cdn_url") or "")
        if url.startswith("https://"):
            return url
    return None


def _topic(bundle: ReaderBundle, entry: dict[str, Any], topic_papers: dict[str, list[str]]) -> Topic:
    topic_id = str(entry.get("id") or "")
    # The id becomes topics/<id>.html and a bundle path: one name, nothing more (F2).
    if not _TOPIC_ID.fullmatch(topic_id) or ".." in topic_id:
        raise ReaderBundleError(f"{UNSAFE_TOPIC_ID}: {topic_id!r}")
    payload = f"decks/payloads/{topic_id}/"
    html_rel, pdf_rel = payload + "presentation.html", payload + "presentation.pdf"
    pptx_rel = payload + "presentation.pptx"
    narrative = f"narratives/{topic_id}.md"
    return Topic(
        topic_id=topic_id,
        title=str(entry.get("title") or topic_id),
        deck_html=html_rel if bundle.listed(html_rel) else None,
        deck_pdf=pdf_rel if bundle.listed(pdf_rel) else None,
        videos=_videos(bundle, payload),
        podcast=_podcast(bundle, topic_id),
        narrative=narrative if bundle.listed(narrative) else None,
        papers=tuple(str(p) for p in topic_papers.get(topic_id, [])),
        deck_pptx=pptx_rel if bundle.listed(pptx_rel) else None,
        deck_text=next((r for r in (f"decks/markdown/{topic_id}/{n}" for n in DECK_TEXT_NAMES) if bundle.listed(r)), None),
    )


def load_reader_bundle(bundle_root: Path) -> ReaderBundle:
    """Read a bundle's reader content, verifying each file it hands on."""
    try:
        root = validate_public_bundle_root(bundle_root)
    except Exception as exc:
        raise ReaderBundleError(str(exc)) from exc
    checksums = json.loads((root / "CHECKSUMS.json").read_text(encoding="utf-8"))
    if not isinstance(checksums, dict):
        raise ReaderBundleError("CHECKSUMS.json must be an object")
    for rel in checksums:
        if not is_plain_relative(rel):
            raise ReaderBundleError(f"{UNSAFE_PATH}: CHECKSUMS.json lists {rel!r}")
    release = json.loads((root / "RELEASE.json").read_text(encoding="utf-8"))
    bundle = ReaderBundle(root, release, checksums, [], [], [], set(), None, None)
    bundle.verified("RELEASE.json")

    topic_papers: dict[str, list[str]] = {}
    if bundle.listed("catalog/topic_papers.json"):
        topic_papers = bundle.read_json("catalog/topic_papers.json")
    manifest_rel = "decks/control/presentation_manifest.json"
    entries = bundle.read_json(manifest_rel).get("presentations", []) if bundle.listed(manifest_rel) else []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        topic = _topic(bundle, entry, topic_papers)
        (bundle.topics if entry.get("presentation_type") in TOPIC_TYPES else bundle.executive).append(topic)

    if bundle.listed("catalog/excluded_ids.json"):
        payload = bundle.read_json("catalog/excluded_ids.json")
        bundle.excluded_ids = {str(x) for x in (payload.get("excluded") or [])}
    for rel in sorted(p.relative_to(root).as_posix() for p in (root / "catalog").glob("*.jsonl")) if (root / "catalog").is_dir() else []:
        if not bundle.listed(rel):
            bundle.ignored_not_in_checksums.append(rel)
            continue
        bundle.catalog_files.append(rel)
        for line in bundle.read_text(rel).split("\n"):
            if line.strip():
                row = json.loads(line)
                if str(row.get("paper_id")) in bundle.excluded_ids:
                    bundle.rows_excluded += 1
                else:
                    bundle.catalog.append(row)
    if bundle.listed("catalog/statistics.json"):
        bundle.statistics = bundle.read_json("catalog/statistics.json")
    _load_index(bundle)
    return bundle


def _load_index(bundle: ReaderBundle) -> None:
    """The index manifest, read only when it names this release, and its file checksums."""
    path = bundle.root / "index" / "index_manifest.json"
    if path.is_symlink() or not path.is_file():
        return
    try:  # index/ itself may be a symlink out of the bundle (phase-4-r2 F1)
        path.resolve().relative_to(bundle.root.resolve())
    except ValueError:
        raise ReaderBundleError(f"{UNSAFE_PATH}: 'index/index_manifest.json'") from None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("bundle_id") != bundle.bundle_id:
        raise ReaderBundleError("index/index_manifest.json does not name this release")
    bundle.index_manifest = manifest
    for name in manifest.get("output_checksums") or {}:
        if not is_plain_relative(name):
            raise ReaderBundleError(f"{UNSAFE_PATH}: index_manifest.json lists {name!r}")
    bundle.index_checksums = {
        f"index/{name}": {"sha256": digest}
        for name, digest in (manifest.get("output_checksums") or {}).items()
        if isinstance(digest, str)
    }


def semantic_block(bundle: ReaderBundle) -> dict[str, Any] | None:
    """The browser-embedding block, only when its blob is in the bundle as checksummed (spec 6.5)."""
    block = (bundle.index_manifest or {}).get("browser_index")
    dim = block.get("dim") if isinstance(block, dict) else None
    # A positive integer, not a truthy value int() would floor to 0 (phase-4-r2 F4).
    if not isinstance(block, dict) or not block.get("model") or not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        return None
    blob = f"index/{block.get('blob') or 'embeddings_browser.f32'}"
    if not bundle.listed(blob):
        return None
    order = f"index/{block.get('chunk_order') or 'chunk_ids.json'}"
    return {"model": str(block["model"]), "dim": dim, "blob": blob, "digest": block.get("digest"), "order": order}
