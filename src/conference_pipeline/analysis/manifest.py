"""analysis_manifest.json dataclasses and fail-closed validator."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from conference_pipeline import workflow
from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.analysis.errors import ManifestValidationError
from conference_pipeline.workflow_runs import write_json_atomic

MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
MIN_COMPATIBLE_SCHEMA_VERSION = 1  # DEPRECATED: prefer SUPPORTED_SCHEMA_VERSIONS
MANIFEST_FILENAME = "analysis_manifest.json"
ANALYSIS_TOPIC_PATTERN = re.compile(r"^(\d{2})_(.+)\.md$")
_GENERATOR_MAX_LEN = 200
_GENERATOR_VALID_RE = re.compile(r"^[\x20-\x7E]+$")


CITATION_COUNT_FIELDS: tuple[str, ...] = (
    "extraction_record_ids",
    "vector_store_chunk_ids",
    "research_ids",
)


def _validate_citation_counts(
    value: Any, *, topic_index: int
) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ManifestValidationError(
            f"topic[{topic_index}] field 'citation_counts' must be an object or null"
        )
    counts: dict[str, int] = {}
    for field_name in CITATION_COUNT_FIELDS:
        raw = value.get(field_name)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ManifestValidationError(
                f"topic[{topic_index}] citation_counts.{field_name} must be a non-negative int"
            )
        counts[field_name] = raw
    extras = sorted(set(value.keys()) - set(CITATION_COUNT_FIELDS))
    if extras:
        raise ManifestValidationError(
            f"topic[{topic_index}] citation_counts has unknown fields: {extras}"
        )
    return counts


@dataclass(frozen=True)
class TopicEntry:
    id: str
    number: int
    slug: str
    title: str
    markdown_path: str
    markdown_sha256: str
    fact_ranks_path: str | None
    fact_ranks_sha256: str | None
    figures_dir: str | None
    audio_dir: str | None
    decks_dir: str | None
    provenance_path: str | None = None
    provenance_json_path: str | None = None
    citation_counts: dict[str, int] | None = None
    deep_analysis_path: str | None = None
    slide_brief_path: str | None = None
    analysis_profile: str | None = None
    narrative_model: str | None = None
    deep_validation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "number": self.number,
            "slug": self.slug,
            "title": self.title,
            "markdown_path": self.markdown_path,
            "markdown_sha256": self.markdown_sha256,
            "fact_ranks_path": self.fact_ranks_path,
            "fact_ranks_sha256": self.fact_ranks_sha256,
            "figures_dir": self.figures_dir,
            "audio_dir": self.audio_dir,
            "decks_dir": self.decks_dir,
        }
        if self.provenance_path is not None:
            data["provenance_path"] = self.provenance_path
        if self.provenance_json_path is not None:
            data["provenance_json_path"] = self.provenance_json_path
        if self.citation_counts is not None:
            data["citation_counts"] = {
                field_name: int(self.citation_counts[field_name])
                for field_name in CITATION_COUNT_FIELDS
            }
        if self.deep_analysis_path is not None:
            data["deep_analysis_path"] = self.deep_analysis_path
        if self.slide_brief_path is not None:
            data["slide_brief_path"] = self.slide_brief_path
        if self.analysis_profile is not None:
            data["analysis_profile"] = self.analysis_profile
        if self.narrative_model is not None:
            data["narrative_model"] = self.narrative_model
        if self.deep_validation is not None:
            data["deep_validation"] = dict(self.deep_validation)
        return data

    @classmethod
    def from_json(cls, data: Any, *, topic_index: int) -> "TopicEntry":
        if not isinstance(data, dict):
            raise ManifestValidationError(f"topic[{topic_index}] must be a JSON object")
        for field_name in ("id", "slug", "title", "markdown_path", "markdown_sha256"):
            value = data.get(field_name)
            if not isinstance(value, str) or not value:
                raise ManifestValidationError(
                    f"topic[{topic_index}] is missing required string field {field_name!r}"
                )
        number = data.get("number")
        if not isinstance(number, int) or isinstance(number, bool):
            raise ManifestValidationError(f"topic[{topic_index}] field 'number' must be int")

        def opt_str(field_name: str) -> str | None:
            value = data.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ManifestValidationError(
                    f"topic[{topic_index}] field {field_name!r} must be string or null"
                )
            return value

        fact_ranks_path = opt_str("fact_ranks_path")
        fact_ranks_sha256 = opt_str("fact_ranks_sha256")
        if (fact_ranks_path is None) != (fact_ranks_sha256 is None):
            raise ManifestValidationError(
                f"topic[{topic_index}] fact_ranks_path and fact_ranks_sha256 must both be null or both set"
            )

        citation_counts = _validate_citation_counts(
            data.get("citation_counts"), topic_index=topic_index
        )
        deep_validation = data.get("deep_validation")
        if deep_validation is not None and not isinstance(deep_validation, dict):
            raise ManifestValidationError(
                f"topic[{topic_index}] field 'deep_validation' must be object or null"
            )

        return cls(
            id=data["id"],
            number=number,
            slug=data["slug"],
            title=data["title"],
            markdown_path=data["markdown_path"],
            markdown_sha256=data["markdown_sha256"],
            fact_ranks_path=fact_ranks_path,
            fact_ranks_sha256=fact_ranks_sha256,
            figures_dir=opt_str("figures_dir"),
            audio_dir=opt_str("audio_dir"),
            decks_dir=opt_str("decks_dir"),
            provenance_path=opt_str("provenance_path"),
            provenance_json_path=opt_str("provenance_json_path"),
            citation_counts=citation_counts,
            deep_analysis_path=opt_str("deep_analysis_path") or data["markdown_path"],
            slide_brief_path=opt_str("slide_brief_path"),
            analysis_profile=opt_str("analysis_profile"),
            narrative_model=opt_str("narrative_model"),
            deep_validation=deep_validation,
        )


@dataclass(frozen=True)
class AnalysisManifest:
    schema_version: int
    generated_at: str
    generator: str
    workspace: str
    content_sha256: str
    topics: tuple[TopicEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "generator": self.generator,
            "workspace": self.workspace,
            "content_sha256": self.content_sha256,
            "topics": [topic.to_dict() for topic in self.topics],
        }

    @classmethod
    def from_json(cls, data: Any) -> "AnalysisManifest":
        if not isinstance(data, dict):
            raise ManifestValidationError("manifest root must be a JSON object")
        schema_version = data.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ManifestValidationError(
                f"manifest schema_version must be int, got {type(schema_version).__name__}"
            )
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ManifestValidationError(
                f"Analysis manifest schema_version {schema_version} is not supported "
                f"(supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}); "
                "regenerate with a current analyze stage."
            )
        for field_name in ("generated_at", "generator", "workspace", "content_sha256"):
            value = data.get(field_name)
            if not isinstance(value, str) or not value:
                raise ManifestValidationError(
                    f"manifest missing required string field {field_name!r}"
                )
        topics_data = data.get("topics")
        if not isinstance(topics_data, list):
            raise ManifestValidationError("manifest 'topics' must be a list")
        topics = tuple(
            TopicEntry.from_json(topic, topic_index=index)
            for index, topic in enumerate(topics_data)
        )
        return cls(
            schema_version=schema_version,
            generated_at=data["generated_at"],
            generator=data["generator"],
            workspace=data["workspace"],
            content_sha256=data["content_sha256"],
            topics=topics,
        )


@dataclass(frozen=True)
class LoadedManifest:
    manifest: AnalysisManifest
    manifest_path: Path
    manifest_sha256: str
    content_sha256: str


def _isoformat_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _short_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workflow.PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "dev"
    return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else "dev"


def _extract_title(markdown_path: Path) -> str | None:
    try:
        for line in markdown_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                return title or None
    except OSError:
        return None
    return None


def _validate_generator(generator: str) -> str:
    if len(generator) > _GENERATOR_MAX_LEN:
        raise ManifestValidationError("manifest generator exceeds 200 characters")
    if not _GENERATOR_VALID_RE.fullmatch(generator):
        raise ManifestValidationError("manifest generator must be printable ASCII")
    return generator


def _compute_content_sha256(manifest_data: dict[str, Any]) -> str:
    stable = dict(manifest_data)
    stable.pop("generated_at", None)
    stable["content_sha256"] = ""
    canonical = json.dumps(
        stable,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _relative_to_workspace(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _optional_path(path: Path, workspace_root: Path) -> str | None:
    return _relative_to_workspace(path, workspace_root) if path.exists() else None


def _topic_entry(
    markdown_path: Path,
    workspace_root: Path,
    *,
    citation_counts: dict[str, int] | None = None,
    provenance_paths: dict[str, str | None] | None = None,
    narrative_metadata: Mapping[str, Any] | None = None,
) -> TopicEntry | None:
    match = ANALYSIS_TOPIC_PATTERN.match(markdown_path.name)
    if not match:
        return None
    topic_id = markdown_path.stem
    number = int(match.group(1))
    slug = match.group(2)
    audio_dir = workspace_root / "audio" / topic_id
    fact_ranks_path = audio_dir / "ranks.json"
    figures_dir = workspace_root / "figures" / topic_id
    decks_dir = workspace_root / "decks" / topic_id
    if not decks_dir.exists():
        decks_dir = workspace_root / "presentations" / topic_id

    fact_ranks_rel = _optional_path(fact_ranks_path, workspace_root)
    provenance_paths = provenance_paths or {}
    narrative_metadata = narrative_metadata or {}
    slide_brief_path = workspace_root / "analysis" / "slide_briefs" / markdown_path.name
    slide_brief_rel = _optional_path(slide_brief_path, workspace_root)
    return TopicEntry(
        id=topic_id,
        number=number,
        slug=slug,
        title=_extract_title(markdown_path) or slug.replace("_", " ").title(),
        markdown_path=_relative_to_workspace(markdown_path, workspace_root),
        markdown_sha256=sha256_file(markdown_path),
        fact_ranks_path=fact_ranks_rel,
        fact_ranks_sha256=sha256_file(fact_ranks_path) if fact_ranks_rel else None,
        figures_dir=_optional_path(figures_dir, workspace_root),
        audio_dir=_optional_path(audio_dir, workspace_root),
        decks_dir=_optional_path(decks_dir, workspace_root),
        provenance_path=provenance_paths.get("provenance_path"),
        provenance_json_path=provenance_paths.get("provenance_json_path"),
        citation_counts=citation_counts,
        deep_analysis_path=_relative_to_workspace(markdown_path, workspace_root),
        slide_brief_path=str(narrative_metadata.get("slide_brief_path") or slide_brief_rel or "") or None,
        analysis_profile=(
            str(narrative_metadata.get("analysis_profile"))
            if narrative_metadata.get("analysis_profile")
            else None
        ),
        narrative_model=(
            str(narrative_metadata.get("narrative_model"))
            if narrative_metadata.get("narrative_model")
            else None
        ),
        deep_validation=(
            dict(narrative_metadata.get("deep_validation"))
            if isinstance(narrative_metadata.get("deep_validation"), dict)
            else None
        ),
    )


def _normalise_citations_by_topic(
    citations_by_topic: Mapping[str, Any] | None,
) -> dict[str, dict[str, int]]:
    """Coerce a sidecar-style mapping into validated per-topic count dicts.

    Each value must declare integer counts under the
    :data:`CITATION_COUNT_FIELDS` keys. Missing keys default to 0 so callers
    can pass partial telemetry. Unknown keys raise to keep the manifest
    schema closed.
    """
    if not citations_by_topic:
        return {}
    normalised: dict[str, dict[str, int]] = {}
    for topic_id, raw in citations_by_topic.items():
        if not isinstance(raw, dict):
            raise ManifestValidationError(
                f"citations_by_topic[{topic_id!r}] must be a dict"
            )
        unknown = sorted(set(raw.keys()) - set(CITATION_COUNT_FIELDS))
        if unknown:
            raise ManifestValidationError(
                f"citations_by_topic[{topic_id!r}] has unknown fields: {unknown}"
            )
        counts: dict[str, int] = {}
        for field_name in CITATION_COUNT_FIELDS:
            value = raw.get(field_name, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ManifestValidationError(
                    f"citations_by_topic[{topic_id!r}].{field_name} must be non-negative int"
                )
            counts[field_name] = value
        normalised[str(topic_id)] = counts
    return normalised


def emit_analysis_manifest(
    workspace: Any,
    *,
    generator: str | None = None,
    citations_by_topic: Mapping[str, Any] | None = None,
    provenance_by_topic: Mapping[str, Mapping[str, str | None]] | None = None,
    narrative_metadata_by_topic: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, str, str]:
    workspace_root = workspace.root.resolve()
    analysis_dir = workflow._resolve_path_inside_workspace(
        workspace_root,
        "analysis",
        label="analysis directory",
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    generator_value = _validate_generator(generator or f"conference_pipeline.analysis@{_short_commit()}")

    citations_lookup = _normalise_citations_by_topic(citations_by_topic)
    provenance_lookup = {
        str(topic_id): {
            "provenance_path": (
                str(value.get("provenance_path"))
                if isinstance(value, Mapping) and value.get("provenance_path")
                else None
            ),
            "provenance_json_path": (
                str(value.get("provenance_json_path"))
                if isinstance(value, Mapping) and value.get("provenance_json_path")
                else None
            ),
        }
        for topic_id, value in (provenance_by_topic or {}).items()
        if isinstance(value, Mapping)
    }
    narrative_metadata_lookup = {
        str(topic_id): dict(value)
        for topic_id, value in (narrative_metadata_by_topic or {}).items()
        if isinstance(value, Mapping)
    }

    topics = tuple(
        topic
        for topic in (
            _topic_entry(
                path,
                workspace_root,
                citation_counts=citations_lookup.get(path.stem),
                provenance_paths=provenance_lookup.get(path.stem),
                narrative_metadata=narrative_metadata_lookup.get(path.stem),
            )
            for path in sorted(analysis_dir.glob("*.md"))
        )
        if topic is not None
    )
    manifest = AnalysisManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        generated_at=_isoformat_z(datetime.now(timezone.utc)),
        generator=generator_value,
        workspace=str(workspace_root),
        content_sha256="",
        topics=topics,
    )
    manifest_data = manifest.to_dict()
    content_sha256 = _compute_content_sha256(manifest_data)
    manifest_data["content_sha256"] = content_sha256

    manifest_path = analysis_dir / MANIFEST_FILENAME
    write_json_atomic(manifest_path, manifest_data)
    return manifest_path, sha256_file(manifest_path), content_sha256


def _load_manifest_json(manifest_path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError(
            "Missing analysis_manifest.json. Run `workflow run --stages analyze` or "
            "`python3 scripts/backfill_analysis_manifest.py --workspace <workspace>` first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"Malformed analysis manifest: {manifest_path}") from exc
    if not isinstance(loaded, dict):
        raise ManifestValidationError("manifest root must be a JSON object")
    return loaded


def _resolve_declared_path(
    workspace_root: Path,
    relative_path: str,
    *,
    label: str,
    require_exists: bool,
) -> Path:
    if Path(relative_path).is_absolute():
        raise ManifestValidationError(f"{label} must be workspace-relative: {relative_path}")
    try:
        resolved = workflow._resolve_path_inside_workspace(
            workspace_root,
            relative_path,
            label=label,
        )
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc
    if require_exists and not resolved.exists():
        raise ManifestValidationError(f"{label} {relative_path} does not exist")
    return resolved


def _validate_topic_paths(
    workspace_root: Path,
    topic: TopicEntry,
    *,
    require_path_existence: bool,
) -> None:
    markdown_path = _resolve_declared_path(
        workspace_root,
        topic.markdown_path,
        label="markdown_path",
        require_exists=require_path_existence,
    )
    if require_path_existence and sha256_file(markdown_path) != topic.markdown_sha256:
        raise ManifestValidationError(f"markdown_sha256 mismatch for {topic.markdown_path}")
    if topic.fact_ranks_path:
        fact_ranks_path = _resolve_declared_path(
            workspace_root,
            topic.fact_ranks_path,
            label="fact_ranks_path",
            require_exists=require_path_existence,
        )
        if (
            require_path_existence
            and topic.fact_ranks_sha256
            and sha256_file(fact_ranks_path) != topic.fact_ranks_sha256
        ):
            raise ManifestValidationError(f"fact_ranks_sha256 mismatch for {topic.fact_ranks_path}")
    for label, relative_path in (
        ("figures_dir", topic.figures_dir),
        ("audio_dir", topic.audio_dir),
        ("decks_dir", topic.decks_dir),
        ("provenance_path", topic.provenance_path),
        ("provenance_json_path", topic.provenance_json_path),
        ("deep_analysis_path", topic.deep_analysis_path),
        ("slide_brief_path", topic.slide_brief_path),
    ):
        if relative_path:
            _resolve_declared_path(
                workspace_root,
                relative_path,
                label=label,
                require_exists=require_path_existence,
            )


def load_and_validate_manifest(
    workspace: Any,
    *,
    require_path_existence: bool = True,
) -> LoadedManifest:
    workspace_root = workspace.root.resolve()
    manifest_path = workspace_root / "analysis" / MANIFEST_FILENAME
    data = _load_manifest_json(manifest_path)
    manifest = AnalysisManifest.from_json(data)
    recomputed_content_sha256 = _compute_content_sha256(data)
    if manifest.content_sha256 != recomputed_content_sha256:
        raise ManifestValidationError("content_sha256 mismatch: manifest tampered")
    for topic in manifest.topics:
        _validate_topic_paths(
            workspace_root,
            topic,
            require_path_existence=require_path_existence,
        )
    return LoadedManifest(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        content_sha256=recomputed_content_sha256,
    )
