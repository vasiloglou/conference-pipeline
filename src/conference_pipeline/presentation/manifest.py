"""presentation_manifest.json dataclasses and fail-closed validator."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from conference_pipeline.model_governance import model_for_alias
from conference_pipeline import workflow
from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.presentation.blueprint import validate_blueprint_review_file
from conference_pipeline.presentation.errors import PresentationManifestValidationError
from conference_pipeline.presentation.legacy_markdown_gate import (
    NON_PRODUCTION_SUPPORT_LEVELS,
    collect_legacy_markdown_paths,
    reject_legacy_markdown_in_production_mode,
)
from conference_pipeline.presentation.presentation_frame import validate_presentation_frame_payload
from conference_pipeline.workflow_runs import write_json_atomic

MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
MANIFEST_FILENAME = "presentation_manifest.json"
MANIFEST_RELATIVE_PATH = "decks/presentation_manifest.json"
PRESENTATION_TYPES = ("topic-deep-dive", "master", "executive-summary")
EXECUTIVE_SUMMARY_PRESENTATION_IDS = frozenset(
    {"executive_summary", "executive_summary_short", "executive_summary_long"}
)
PRESENTATION_STAGE_IDS = (
    "presentation_blueprint",
    "deck_contract_generation",
    "source_binding",
    "visual_plan_generation",
    "render_spec_generation",
    "render_export",
    "visual_qc",
    "delivery_generation",
    "presentation_review",
    "video_generation",
    "video_sync_validation",
    "human_review",
)

_GENERATOR_MAX_LEN = 200
_GENERATOR_VALID_RE = re.compile(r"^[\x20-\x7E]+$")
_MODE_READINESS_VALUES = frozenset({"ready", "planned", "unsupported"})
_TOPIC_SCOPE_MODES = frozenset({"single-topic", "all-topics"})
_ARTIFACT_KINDS = frozenset({"file", "directory"})
_STAGE_STATE_VALUES = frozenset({"missing", "partial", "complete"})
_PRESENTATION_TYPE_SET = frozenset(PRESENTATION_TYPES)
_PRESENTATION_STAGE_ID_SET = frozenset(PRESENTATION_STAGE_IDS)
_CANONICAL_TOPIC_DIR_RE = re.compile(r"^\d{2}_.+$")
_ANALYSIS_TOPIC_FILE_RE = re.compile(r"^\d{2}_.+\.md$")
_HUMAN_REVIEW_APPROVED_STATUSES = frozenset({"accepted", "approved"})
_HUMAN_REVIEW_APPROVED_DECISIONS = frozenset({"approved_for_bulk_rollout"})


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PresentationManifestValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _require_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PresentationManifestValidationError(f"{label} must be int")
    return value


def _optional_float(value: Any, *, label: str, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PresentationManifestValidationError(f"{label} must be number")
    parsed = float(value)
    if parsed < 0:
        raise PresentationManifestValidationError(f"{label} must be non-negative")
    return parsed


def _require_string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PresentationManifestValidationError(f"{label} must be a list")
    values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise PresentationManifestValidationError(
                f"{label}[{index}] must be a non-empty string"
            )
        values.append(item)
    return tuple(values)


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


def _validate_generator(generator: str) -> str:
    if len(generator) > _GENERATOR_MAX_LEN:
        raise PresentationManifestValidationError("manifest generator exceeds 200 characters")
    if not _GENERATOR_VALID_RE.fullmatch(generator):
        raise PresentationManifestValidationError("manifest generator must be printable ASCII")
    return generator


def _relative_to_workspace(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _extract_title(markdown_path: Path) -> str | None:
    try:
        for line in markdown_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                return title or None
    except OSError:
        return None
    return None


_DECK_TITLE_SUFFIX_RE = re.compile(
    r"\s+(?:slide\s+brief|presentation\s+brief|deck\s+brief)\s*$",
    flags=re.IGNORECASE,
)


def _deck_display_title(title: str) -> str:
    """Normalize report-derived headings into deck-facing display titles."""
    normalized = re.sub(r"\s+", " ", title).strip()
    normalized = _DECK_TITLE_SUFFIX_RE.sub("", normalized).strip()
    return normalized or title.strip()


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


def _resolve_declared_path(
    workspace_root: Path,
    relative_path: str,
    *,
    label: str,
    require_exists: bool,
) -> Path:
    if Path(relative_path).is_absolute():
        raise PresentationManifestValidationError(
            f"{label} must be workspace-relative: {relative_path}"
        )
    try:
        resolved = workflow._resolve_path_inside_workspace(
            workspace_root,
            relative_path,
            label=label,
        )
    except ValueError as exc:
        raise PresentationManifestValidationError(str(exc)) from exc
    if require_exists and not resolved.exists():
        raise PresentationManifestValidationError(f"{label} {relative_path} does not exist")
    return resolved


def _directory_sha256(path: Path) -> str:
    if not path.is_dir():
        raise PresentationManifestValidationError(f"declared directory artifact is not a directory: {path}")
    hasher = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        rel = child.relative_to(path).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(sha256_file(child).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _artifact_sha256(path: Path, *, kind: str) -> str:
    if kind == "file":
        if not path.is_file():
            raise PresentationManifestValidationError(f"declared file artifact is not a file: {path}")
        return sha256_file(path)
    if kind == "directory":
        return _directory_sha256(path)
    raise PresentationManifestValidationError(f"unsupported artifact kind: {kind}")


@dataclass(frozen=True)
class StageModeReadiness:
    live: str
    dry_run: str
    mock: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "live": self.live,
            "dry_run": self.dry_run,
            "mock": self.mock,
        }

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> "StageModeReadiness":
        if not isinstance(data, dict):
            raise PresentationManifestValidationError(f"{label} must be a JSON object")
        values: dict[str, str] = {}
        for key in ("live", "dry_run", "mock"):
            value = _require_string(data.get(key), label=f"{label}.{key}")
            if value not in _MODE_READINESS_VALUES:
                raise PresentationManifestValidationError(
                    f"{label}.{key} must be one of {sorted(_MODE_READINESS_VALUES)}"
                )
            values[key] = value
        return cls(**values)


@dataclass(frozen=True)
class PresentationStageSpec:
    id: str
    input_artifacts: tuple[str, ...]
    output_artifacts: tuple[str, ...]
    owner_refs: tuple[str, ...]
    provider_requirements: tuple[str, ...]
    cache_behavior: str
    support_level: str
    mode_readiness: StageModeReadiness

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_artifacts": list(self.input_artifacts),
            "output_artifacts": list(self.output_artifacts),
            "owner_refs": list(self.owner_refs),
            "provider_requirements": list(self.provider_requirements),
            "cache_behavior": self.cache_behavior,
            "support_level": self.support_level,
            "mode_readiness": self.mode_readiness.to_dict(),
        }

    @classmethod
    def from_json(cls, data: Any, *, stage_index: int) -> "PresentationStageSpec":
        if not isinstance(data, dict):
            raise PresentationManifestValidationError(
                f"stage_specs[{stage_index}] must be a JSON object"
            )
        stage_id = _require_string(data.get("id"), label=f"stage_specs[{stage_index}].id")
        if stage_id not in _PRESENTATION_STAGE_ID_SET:
            raise PresentationManifestValidationError(f"unknown stage id: {stage_id}")
        support_level = _require_string(
            data.get("support_level"),
            label=f"stage_specs[{stage_index}].support_level",
        )
        try:
            reject_legacy_markdown_in_production_mode(
                [
                    *_require_string_tuple(
                        data.get("input_artifacts"),
                        label=f"stage_specs[{stage_index}].input_artifacts",
                    ),
                    *_require_string_tuple(
                        data.get("output_artifacts"),
                        label=f"stage_specs[{stage_index}].output_artifacts",
                    ),
                ],
                context=f"stage_specs[{stage_index}]",
                support_level=support_level,
                production_evidence=support_level not in NON_PRODUCTION_SUPPORT_LEVELS,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "LegacyMarkdownStageError":
                raise PresentationManifestValidationError(str(exc)) from exc
            raise
        return cls(
            id=stage_id,
            input_artifacts=_require_string_tuple(
                data.get("input_artifacts"),
                label=f"stage_specs[{stage_index}].input_artifacts",
            ),
            output_artifacts=_require_string_tuple(
                data.get("output_artifacts"),
                label=f"stage_specs[{stage_index}].output_artifacts",
            ),
            owner_refs=_require_string_tuple(
                data.get("owner_refs"),
                label=f"stage_specs[{stage_index}].owner_refs",
            ),
            provider_requirements=_require_string_tuple(
                data.get("provider_requirements"),
                label=f"stage_specs[{stage_index}].provider_requirements",
            ),
            cache_behavior=_require_string(
                data.get("cache_behavior"),
                label=f"stage_specs[{stage_index}].cache_behavior",
            ),
            support_level=support_level,
            mode_readiness=StageModeReadiness.from_json(
                data.get("mode_readiness"),
                label=f"stage_specs[{stage_index}].mode_readiness",
            ),
        )


@dataclass(frozen=True)
class PresentationArtifactRef:
    kind: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> "PresentationArtifactRef":
        if not isinstance(data, dict):
            raise PresentationManifestValidationError(f"{label} must be a JSON object")
        kind = _require_string(data.get("kind"), label=f"{label}.kind")
        if kind not in _ARTIFACT_KINDS:
            raise PresentationManifestValidationError(
                f"{label}.kind must be one of {sorted(_ARTIFACT_KINDS)}"
            )
        return cls(
            kind=kind,
            path=_require_string(data.get("path"), label=f"{label}.path"),
            sha256=_require_string(data.get("sha256"), label=f"{label}.sha256"),
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        workspace_root: Path,
        kind: str,
    ) -> "PresentationArtifactRef":
        if kind not in _ARTIFACT_KINDS:
            raise PresentationManifestValidationError(
                f"artifact kind must be one of {sorted(_ARTIFACT_KINDS)}"
            )
        return cls(
            kind=kind,
            path=_relative_to_workspace(path, workspace_root),
            sha256=_artifact_sha256(path, kind=kind),
        )


@dataclass(frozen=True)
class PresentationTopicScope:
    mode: str
    topics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "topics": list(self.topics),
        }

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> "PresentationTopicScope":
        if not isinstance(data, dict):
            raise PresentationManifestValidationError(f"{label} must be a JSON object")
        mode = _require_string(data.get("mode"), label=f"{label}.mode")
        if mode not in _TOPIC_SCOPE_MODES:
            raise PresentationManifestValidationError(
                f"{label}.mode must be one of {sorted(_TOPIC_SCOPE_MODES)}"
            )
        topics = _require_string_tuple(data.get("topics"), label=f"{label}.topics")
        return cls(mode=mode, topics=topics)


@dataclass(frozen=True)
class PresentationStageState:
    status: str
    artifacts: tuple[PresentationArtifactRef, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "notes": list(self.notes),
        }

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> "PresentationStageState":
        if not isinstance(data, dict):
            raise PresentationManifestValidationError(f"{label} must be a JSON object")
        status = _require_string(data.get("status"), label=f"{label}.status")
        if status not in _STAGE_STATE_VALUES:
            raise PresentationManifestValidationError(
                f"{label}.status must be one of {sorted(_STAGE_STATE_VALUES)}"
            )
        artifacts_data = data.get("artifacts")
        if not isinstance(artifacts_data, list):
            raise PresentationManifestValidationError(f"{label}.artifacts must be a list")
        artifacts = tuple(
            PresentationArtifactRef.from_json(artifact, label=f"{label}.artifacts[{index}]")
            for index, artifact in enumerate(artifacts_data)
        )
        try:
            reject_legacy_markdown_in_production_mode(
                (artifact.path for artifact in artifacts),
                context=label,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "LegacyMarkdownStageError":
                raise PresentationManifestValidationError(str(exc)) from exc
            raise
        notes = _require_string_tuple(data.get("notes"), label=f"{label}.notes")
        return cls(status=status, artifacts=artifacts, notes=notes)


@dataclass(frozen=True)
class LegacyMarkdownCompatibility:
    support_level: str
    production_evidence: bool
    artifacts: tuple[PresentationArtifactRef, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_level": self.support_level,
            "production_evidence": self.production_evidence,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "notes": list(self.notes),
        }

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> "LegacyMarkdownCompatibility | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise PresentationManifestValidationError(f"{label} must be a JSON object")
        support_level = _require_string(data.get("support_level"), label=f"{label}.support_level")
        production_evidence = data.get("production_evidence")
        if not isinstance(production_evidence, bool):
            raise PresentationManifestValidationError(f"{label}.production_evidence must be bool")
        if support_level not in NON_PRODUCTION_SUPPORT_LEVELS or production_evidence is not False:
            raise PresentationManifestValidationError(
                f"{label} must be non-production legacy/debug compatibility metadata"
            )
        artifacts_data = data.get("artifacts")
        if not isinstance(artifacts_data, list):
            raise PresentationManifestValidationError(f"{label}.artifacts must be a list")
        artifacts = tuple(
            PresentationArtifactRef.from_json(artifact, label=f"{label}.artifacts[{index}]")
            for index, artifact in enumerate(artifacts_data)
        )
        reject_legacy_markdown_in_production_mode(
            (artifact.path for artifact in artifacts),
            context=label,
            support_level=support_level,
            production_evidence=production_evidence,
        )
        notes = _require_string_tuple(data.get("notes"), label=f"{label}.notes")
        return cls(
            support_level=support_level,
            production_evidence=production_evidence,
            artifacts=artifacts,
            notes=notes,
        )


@dataclass(frozen=True)
class PresentationEntry:
    id: str
    presentation_type: str
    title: str
    topic_scope: PresentationTopicScope
    target_duration_minutes: int
    target_slide_count: int
    source_analysis_artifacts: tuple[PresentationArtifactRef, ...]
    stage_states: dict[str, PresentationStageState]
    target_hyperlinks_per_slide: float = 0.0
    target_svgs_per_slide: float = 0.0
    legacy_markdown: LegacyMarkdownCompatibility | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "presentation_type": self.presentation_type,
            "title": self.title,
            "topic_scope": self.topic_scope.to_dict(),
            "target_duration_minutes": self.target_duration_minutes,
            "target_slide_count": self.target_slide_count,
            "target_hyperlinks_per_slide": self.target_hyperlinks_per_slide,
            "target_svgs_per_slide": self.target_svgs_per_slide,
            "source_analysis_artifacts": [
                artifact.to_dict() for artifact in self.source_analysis_artifacts
            ],
            "stage_states": {
                stage_id: state.to_dict()
                for stage_id, state in self.stage_states.items()
            },
        }
        if self.legacy_markdown is not None:
            payload["legacy_markdown"] = self.legacy_markdown.to_dict()
        return payload

    @classmethod
    def from_json(
        cls,
        data: Any,
        *,
        entry_index: int,
        stage_ids: tuple[str, ...],
    ) -> "PresentationEntry":
        if not isinstance(data, dict):
            raise PresentationManifestValidationError(
                f"presentations[{entry_index}] must be a JSON object"
            )
        presentation_type = _require_string(
            data.get("presentation_type"),
            label=f"presentations[{entry_index}].presentation_type",
        )
        if presentation_type not in _PRESENTATION_TYPE_SET:
            raise PresentationManifestValidationError(
                f"presentations[{entry_index}].presentation_type must be one of {sorted(_PRESENTATION_TYPE_SET)}"
            )
        source_analysis_artifacts_data = data.get("source_analysis_artifacts")
        if not isinstance(source_analysis_artifacts_data, list):
            raise PresentationManifestValidationError(
                f"presentations[{entry_index}].source_analysis_artifacts must be a list"
            )
        stage_states_data = data.get("stage_states")
        if not isinstance(stage_states_data, dict):
            raise PresentationManifestValidationError(
                f"presentations[{entry_index}].stage_states must be a JSON object"
            )
        stage_states: dict[str, PresentationStageState] = {}
        for stage_id in stage_ids:
            if stage_id not in stage_states_data:
                raise PresentationManifestValidationError(
                    f"presentations[{entry_index}].stage_states missing required stage {stage_id!r}"
                )
            stage_states[stage_id] = PresentationStageState.from_json(
                stage_states_data[stage_id],
                label=f"presentations[{entry_index}].stage_states.{stage_id}",
            )
        extra_stage_ids = set(stage_states_data) - set(stage_ids)
        if extra_stage_ids:
            raise PresentationManifestValidationError(
                f"presentations[{entry_index}].stage_states declares unknown stage ids: {sorted(extra_stage_ids)}"
            )
        return cls(
            id=_require_string(data.get("id"), label=f"presentations[{entry_index}].id"),
            presentation_type=presentation_type,
            title=_require_string(data.get("title"), label=f"presentations[{entry_index}].title"),
            topic_scope=PresentationTopicScope.from_json(
                data.get("topic_scope"),
                label=f"presentations[{entry_index}].topic_scope",
            ),
            target_duration_minutes=_require_int(
                data.get("target_duration_minutes"),
                label=f"presentations[{entry_index}].target_duration_minutes",
            ),
            target_slide_count=_require_int(
                data.get("target_slide_count"),
                label=f"presentations[{entry_index}].target_slide_count",
            ),
            target_hyperlinks_per_slide=_optional_float(
                data.get("target_hyperlinks_per_slide"),
                label=f"presentations[{entry_index}].target_hyperlinks_per_slide",
            ),
            target_svgs_per_slide=_optional_float(
                data.get("target_svgs_per_slide"),
                label=f"presentations[{entry_index}].target_svgs_per_slide",
            ),
            source_analysis_artifacts=tuple(
                PresentationArtifactRef.from_json(
                    artifact,
                    label=f"presentations[{entry_index}].source_analysis_artifacts[{index}]",
                )
                for index, artifact in enumerate(source_analysis_artifacts_data)
            ),
            stage_states=stage_states,
            legacy_markdown=LegacyMarkdownCompatibility.from_json(
                data.get("legacy_markdown"),
                label=f"presentations[{entry_index}].legacy_markdown",
            ),
        )


@dataclass(frozen=True)
class PresentationManifest:
    schema_version: int
    generated_at: str
    generator: str
    workspace: str
    content_sha256: str
    stage_specs: tuple[PresentationStageSpec, ...]
    presentations: tuple[PresentationEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "generator": self.generator,
            "workspace": self.workspace,
            "content_sha256": self.content_sha256,
            "stage_specs": [spec.to_dict() for spec in self.stage_specs],
            "presentations": [presentation.to_dict() for presentation in self.presentations],
        }

    @classmethod
    def from_json(cls, data: Any) -> "PresentationManifest":
        if not isinstance(data, dict):
            raise PresentationManifestValidationError("manifest root must be a JSON object")
        schema_version = _require_int(data.get("schema_version"), label="manifest.schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise PresentationManifestValidationError(
                f"Presentation manifest schema_version {schema_version} is not supported "
                f"(supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}); "
                "regenerate with a current presentation manifest emitter."
            )
        generated_at = _require_string(data.get("generated_at"), label="manifest.generated_at")
        generator = _validate_generator(
            _require_string(data.get("generator"), label="manifest.generator")
        )
        workspace_value = _require_string(data.get("workspace"), label="manifest.workspace")
        content_sha256 = _require_string(
            data.get("content_sha256"),
            label="manifest.content_sha256",
        )
        stage_specs_data = data.get("stage_specs")
        if not isinstance(stage_specs_data, list):
            raise PresentationManifestValidationError("manifest.stage_specs must be a list")
        stage_specs = tuple(
            PresentationStageSpec.from_json(spec, stage_index=index)
            for index, spec in enumerate(stage_specs_data)
        )
        stage_ids = tuple(spec.id for spec in stage_specs)
        if stage_ids != PRESENTATION_STAGE_IDS:
            raise PresentationManifestValidationError(
                f"manifest.stage_specs must declare the exact checkpoint ids {PRESENTATION_STAGE_IDS}"
            )
        presentations_data = data.get("presentations")
        if not isinstance(presentations_data, list):
            raise PresentationManifestValidationError("manifest.presentations must be a list")
        presentations = tuple(
            PresentationEntry.from_json(
                presentation,
                entry_index=index,
                stage_ids=stage_ids,
            )
            for index, presentation in enumerate(presentations_data)
        )
        return cls(
            schema_version=schema_version,
            generated_at=generated_at,
            generator=generator,
            workspace=workspace_value,
            content_sha256=content_sha256,
            stage_specs=stage_specs,
            presentations=presentations,
        )


@dataclass(frozen=True)
class LoadedPresentationManifest:
    manifest: PresentationManifest
    manifest_path: Path
    manifest_sha256: str
    content_sha256: str


def _stage_spec(
    stage_id: str,
    *,
    input_artifacts: Iterable[str],
    output_artifacts: Iterable[str],
    owner_refs: Iterable[str],
    provider_requirements: Iterable[str],
    cache_behavior: str,
    support_level: str = "workspace-local",
    live: str = "ready",
    dry_run: str = "planned",
    mock: str = "ready",
) -> PresentationStageSpec:
    return PresentationStageSpec(
        id=stage_id,
        input_artifacts=tuple(input_artifacts),
        output_artifacts=tuple(output_artifacts),
        owner_refs=tuple(owner_refs),
        provider_requirements=tuple(provider_requirements),
        cache_behavior=cache_behavior,
        support_level=support_level,
        mode_readiness=StageModeReadiness(
            live=live,
            dry_run=dry_run,
            mock=mock,
        ),
    )


DEFAULT_STAGE_SPECS: tuple[PresentationStageSpec, ...] = (
    _stage_spec(
        "presentation_blueprint",
        input_artifacts=(
            "analysis/NN_slug.md",
            "decks/<presentation>/presentation_frame.json",
        ),
        output_artifacts=(
            "decks/<presentation>/presentation_frame.json",
            "decks/<presentation>/presentation_blueprint.json",
            "decks/<presentation>/presentation_blueprint.md",
            "decks/<presentation>/presentation_blueprint_receipt.json",
            "decks/<presentation>/blueprint_review.json",
            "decks/<presentation>/blueprint_review.md",
        ),
        owner_refs=(
            "scripts/presentations/generate_editorial_blueprint.py",
            "scripts/presentations/review_presentation_blueprint.py",
        ),
        provider_requirements=("task:presentation.blueprint_generation", "human:blueprint_review"),
        cache_behavior="generates and reviews an audience-framed, source-grounded blueprint before deck contracts",
        live="planned",
        dry_run="ready",
        mock="ready",
    ),
    _stage_spec(
        "deck_contract_generation",
        input_artifacts=(
            "decks/<presentation>/presentation_blueprint.json",
            "decks/<presentation>/presentation_blueprint_receipt.json",
            "decks/<presentation>/blueprint_review.json",
        ),
        output_artifacts=(
            "decks/<presentation>/presentation_deck.json",
            "decks/<presentation>/presentation_deck_receipt.json",
        ),
        owner_refs=("scripts/presentations/generate_deck_contract.py",),
        provider_requirements=("python:stdlib",),
        cache_behavior="compiles an approved blueprint into the typed deck contract",
    ),
    _stage_spec(
        "source_binding",
        input_artifacts=(
            "decks/<presentation>/presentation_deck.json",
            "storage/paper-matching/<run-id>/<topic>/matched_papers.json",
            "storage/extractions/<run-id>/by-topic/<topic>.json",
        ),
        output_artifacts=(
            "decks/<presentation>/ranked_papers.json",
            "decks/<presentation>/source_pack_receipt.json",
            "decks/<presentation>/source_binding_report.json",
            "storage/paper-matching/<run-id>/<topic>/matched_papers.json",
        ),
        owner_refs=(
            "scripts/presentations/build_deck_source_pack.py",
            "scripts/presentations/bind_presentation_sources.py",
        ),
        provider_requirements=("env:ANTHROPIC_API_KEY",),
        cache_behavior="binds approved source labels and visible references to deck slides",
    ),
    _stage_spec(
        "visual_plan_generation",
        input_artifacts=(
            "decks/<presentation>/presentation_deck.json",
            "decks/<presentation>/source_binding_report.json",
            "decks/<presentation>/source_pack_receipt.json",
            "decks/<presentation>/ranked_papers.json",
        ),
        output_artifacts=(
            "decks/<presentation>/visual_plan.json",
            "decks/<presentation>/asset_inventory.json",
            "decks/<presentation>/assets/generated/figure_specs.json",
            "decks/<presentation>/assets/generated/",
        ),
        owner_refs=(
            "scripts/presentations/generate_programmatic_figures.py",
        ),
        provider_requirements=(
            "task:presentation.svg_figure_generation",
            "renderer:matplotlib|plotly|svgwrite",
        ),
        cache_behavior=(
            "generates source-grounded visual plans, figure specs, and assets "
            "from structured deck/source contracts"
        ),
        support_level="workspace-local",
        live="ready",
        dry_run="planned",
        mock="ready",
    ),
    _stage_spec(
        "render_spec_generation",
        input_artifacts=(
            "decks/<presentation>/presentation_deck.json",
            "decks/<presentation>/source_binding_report.json",
            "decks/<presentation>/visual_plan.json",
            "decks/<presentation>/asset_inventory.json",
        ),
        output_artifacts=("decks/<presentation>/render_spec.json",),
        owner_refs=("scripts/presentations/generate_render_spec.py",),
        provider_requirements=("python:stdlib",),
        cache_behavior="merges deck, source, visual, author, and delivery timing into render input",
    ),
    _stage_spec(
        "render_export",
        input_artifacts=("decks/<presentation>/render_spec.json",),
        output_artifacts=(
            "decks/<presentation>/presentation.html",
            "decks/<presentation>/presentation.pdf",
            "decks/<presentation>/presentation_slide_mapping.json",
        ),
        owner_refs=(
            "scripts/presentations/md_to_revealjs.py",
            "scripts/presentations/md_to_pptx.py",
        ),
        provider_requirements=("tool:node", "tool:playwright", "brand:RelationalAI NeurIPS 2025 reference"),  # conference-literal-ok: fixed RelationalAI brand-template lock (conference-agnostic by design)
        cache_behavior=(
            "renders HTML/PDF and slide mapping from render_spec.json rather than Markdown"
        ),
    ),
    _stage_spec(
        "visual_qc",
        input_artifacts=("decks/<presentation>/presentation.html",),
        output_artifacts=(
            "decks/<presentation>/assets/generated/visual_qc_report.md",
            "decks/<presentation>/assets/generated/visual_llm_qc_report.json",
            "decks/<presentation>/assets/generated/qc/screenshots/",
        ),
        owner_refs=(
            "scripts/presentations/inspect_rendered_slides.py",
            "scripts/presentations/inspect_rendered_slides_llm.py",
        ),
        provider_requirements=("tool:playwright", "env:OPENAI_API_KEY"),
        cache_behavior=(
            "renders slides in Chromium, stores screenshots, and reviews them "
            "with governed vision-LLM visual QC"
        ),
    ),
    _stage_spec(
        "delivery_generation",
        input_artifacts=(
            "decks/<presentation>/render_spec.json",
            "decks/<presentation>/presentation_slide_mapping.json",
        ),
        output_artifacts=(
            "decks/<presentation>/delivery_single_speaker.json",
            "decks/<presentation>/delivery_notebooklm.json",
            "decks/<presentation>/6_presenter_script.md",
        ),
        owner_refs=("scripts/presentations/generate_presenter_script.py",),
        provider_requirements=("env:GEMINI_API_KEY|GOOGLE_API_KEY",),
        cache_behavior=(
            "generates structured delivery contracts plus the human-readable narration script"
        ),
    ),
    _stage_spec(
        "presentation_review",
        input_artifacts=(
            "decks/<presentation>/presentation_blueprint.json",
            "decks/<presentation>/blueprint_review.json",
            "decks/<presentation>/presentation_deck.json",
            "decks/<presentation>/render_spec.json",
            "decks/<presentation>/source_binding_report.json",
            "decks/<presentation>/delivery_notebooklm.json",
            "decks/<presentation>/video_sync_validation_notebooklm.json",
            "decks/<presentation>/assets/generated/visual_llm_qc_report.json",
            "decks/<presentation>/render_contact_sheet.png",
            "decks/<presentation>/6_presenter_script.md",
            "decks/<presentation>/assets/generated/qc/screenshots/",
        ),
        output_artifacts=(
            "decks/<presentation>/presentation_review.json",
            "decks/<presentation>/presentation_review.md",
            "decks/<presentation>/presentation_review.prompt.md",
        ),
        owner_refs=("scripts/presentations/review_presentation.py",),
        provider_requirements=(
            "tool:claude-code",
            f"model:anthropic:{model_for_alias('opus')}",
            "tool:codex",
            f"model:openai:{model_for_alias('sol')}",
            "tool:gemini",
        ),
        cache_behavior=(
            "reviews requirements, narration, and every slide screenshot with "
            "Claude Code (opus alias), Codex CLI (sol alias), and Gemini before production handoff"
        ),
        support_level="workspace-local",
        live="ready",
        dry_run="ready",
        mock="ready",
    ),
    _stage_spec(
        "video_generation",
        input_artifacts=(
            "decks/<presentation>/delivery_single_speaker.json",
            "decks/<presentation>/delivery_notebooklm.json",
            "decks/<presentation>/presentation.html",
            "decks/<presentation>/presentation_slide_mapping.json",
        ),
        output_artifacts=(
            "decks/<presentation>/video_build/",
            "decks/<presentation>/video_build_notebooklm/",
            "decks/<presentation>/presentation_video.mp4",
            "decks/<presentation>/presentation_video_notebooklm.mp4",
        ),
        owner_refs=(
            "scripts/presentations/build_presenter_video.py",
            "scripts/presentations/build_notebooklm_style_presenter_video.py",
        ),
        provider_requirements=(
            "tool:ffmpeg",
            "tool:playwright",
            "env:GEMINI_API_KEY|GOOGLE_API_KEY",
        ),
        cache_behavior="reuses rendered video work directories when they already exist",
    ),
    _stage_spec(
        "video_sync_validation",
        input_artifacts=(
            "decks/<presentation>/render_spec.json",
            "decks/<presentation>/delivery_single_speaker.json",
            "decks/<presentation>/delivery_notebooklm.json",
            "decks/<presentation>/presentation_video.mp4",
            "decks/<presentation>/presentation_video_notebooklm.mp4",
        ),
        output_artifacts=(
            "decks/<presentation>/video_sync_validation_single_speaker.json",
            "decks/<presentation>/video_sync_validation_notebooklm.json",
        ),
        owner_refs=(
            "scripts/presentations/validate_video_sync.py",
            "scripts/presentations/validate_video_delivery.py",
        ),
        provider_requirements=("tool:ffmpeg", "python:stdlib"),
        cache_behavior="validates voice gaps, slide sync, visual source, and NotebookLM divergence",
    ),
    _stage_spec(
        "human_review",
        input_artifacts=("decks/<presentation>/presentation.html",),
        output_artifacts=("decks/<presentation>/review_marker.txt",),
        owner_refs=("conferences/neurips2025/decks/Execute_Presentation_Plan.md",),
        provider_requirements=("human:operator",),
        cache_behavior="depends on explicit operator review evidence rather than generated artifacts",
        dry_run="unsupported",
        mock="unsupported",
    ),
)


def emit_presentation_manifest(
    workspace: Any,
    presentations: Sequence[PresentationEntry],
    *,
    generator: str | None = None,
    stage_specs: Sequence[PresentationStageSpec] | None = None,
) -> tuple[Path, str, str]:
    workspace_root = workspace.root.resolve()
    decks_dir = workflow._resolve_path_inside_workspace(
        workspace_root,
        "decks",
        label="decks directory",
    )
    decks_dir.mkdir(parents=True, exist_ok=True)
    stage_specs_value = tuple(stage_specs or DEFAULT_STAGE_SPECS)
    if tuple(spec.id for spec in stage_specs_value) != PRESENTATION_STAGE_IDS:
        raise PresentationManifestValidationError(
            f"stage_specs must declare the exact checkpoint ids {PRESENTATION_STAGE_IDS}"
        )
    generator_value = _validate_generator(
        generator or f"conference_pipeline.presentation@{_short_commit()}"
    )
    manifest = PresentationManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        generated_at=_isoformat_z(datetime.now(timezone.utc)),
        generator=generator_value,
        workspace=str(workspace_root),
        content_sha256="",
        stage_specs=stage_specs_value,
        presentations=tuple(presentations),
    )
    manifest_data = manifest.to_dict()
    content_sha256 = _compute_content_sha256(manifest_data)
    manifest_data["content_sha256"] = content_sha256
    manifest_path = decks_dir / MANIFEST_FILENAME
    write_json_atomic(manifest_path, manifest_data)
    return manifest_path, sha256_file(manifest_path), content_sha256


def _existing_file_artifacts(
    workspace_root: Path,
    presentation_dir: Path,
    relative_paths: Sequence[str],
) -> tuple[PresentationArtifactRef, ...]:
    artifacts: list[PresentationArtifactRef] = []
    for relative_path in relative_paths:
        candidate = presentation_dir / relative_path
        if candidate.is_file():
            artifacts.append(
                PresentationArtifactRef.from_path(
                    candidate,
                    workspace_root=workspace_root,
                    kind="file",
                )
            )
    return tuple(artifacts)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _human_review_receipt_is_approved(receipt: dict[str, Any]) -> bool:
    status = str(receipt.get("status", "")).strip().lower()
    decision = str(receipt.get("decision", "")).strip().lower()
    return (
        status in _HUMAN_REVIEW_APPROVED_STATUSES
        or decision in _HUMAN_REVIEW_APPROVED_DECISIONS
    )


def _blueprint_receipt_is_bound(
    receipt: dict[str, Any],
    blueprint_path: Path,
) -> bool:
    blueprint_hash = str(receipt.get("blueprint_sha256", "")).strip()
    return bool(blueprint_hash) and blueprint_hash == sha256_file(blueprint_path)


def _existing_directory_artifacts(
    workspace_root: Path,
    presentation_dir: Path,
    relative_paths: Sequence[str],
) -> tuple[PresentationArtifactRef, ...]:
    artifacts: list[PresentationArtifactRef] = []
    for relative_path in relative_paths:
        candidate = presentation_dir / relative_path
        if candidate.is_dir():
            artifacts.append(
                PresentationArtifactRef.from_path(
                    candidate,
                    workspace_root=workspace_root,
                    kind="directory",
                )
            )
    return tuple(artifacts)


def _stage_state_from_required_files(
    workspace_root: Path,
    presentation_dir: Path,
    required_relative_paths: Sequence[str],
    *,
    optional_relative_paths: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> PresentationStageState:
    required_artifacts = _existing_file_artifacts(
        workspace_root,
        presentation_dir,
        required_relative_paths,
    )
    optional_artifacts = _existing_file_artifacts(
        workspace_root,
        presentation_dir,
        optional_relative_paths,
    )
    artifacts = required_artifacts + optional_artifacts
    if len(required_artifacts) == len(required_relative_paths):
        status = "complete"
    elif artifacts:
        status = "partial"
    else:
        status = "missing"
    return PresentationStageState(
        status=status,
        artifacts=artifacts,
        notes=tuple(notes),
    )


def _visual_plan_generation_state(
    workspace_root: Path,
    presentation_dir: Path,
) -> PresentationStageState:
    generated_dir = presentation_dir / "assets" / "generated"
    svg_artifacts = tuple(
        PresentationArtifactRef.from_path(
            svg_path,
            workspace_root=workspace_root,
            kind="file",
        )
        for svg_path in sorted(generated_dir.rglob("*.svg"))
        if svg_path.is_file()
    )
    inventory_artifacts = _existing_file_artifacts(
        workspace_root,
        presentation_dir,
        ("visual_plan.json", "asset_inventory.json", "assets/generated/figure_specs.json"),
    )
    artifacts = inventory_artifacts + svg_artifacts
    required = {"visual_plan.json", "asset_inventory.json"}
    present_names = {Path(artifact.path).name for artifact in inventory_artifacts}
    if required.issubset(present_names) and svg_artifacts:
        status = "complete"
    elif artifacts:
        status = "partial"
    else:
        status = "missing"
    return PresentationStageState(status=status, artifacts=artifacts, notes=())


def _delivery_generation_state(
    workspace_root: Path,
    presentation_dir: Path,
) -> PresentationStageState:
    return _stage_state_from_required_files(
        workspace_root,
        presentation_dir,
        ("delivery_single_speaker.json", "delivery_notebooklm.json", "6_presenter_script.md"),
        optional_relative_paths=(
            "6_presenter_script_notebooklm_2host.md",
        ),
        notes=(
            "Single-speaker presenter Markdown is production narration evidence for whole-presentation review; the NotebookLM Markdown remains a human-readable export.",
        ),
    )


def _video_generation_state(
    workspace_root: Path,
    presentation_dir: Path,
) -> PresentationStageState:
    directory_artifacts = _existing_directory_artifacts(
        workspace_root,
        presentation_dir,
        ("video_build", "video_build_notebooklm"),
    )
    video_artifacts = _existing_file_artifacts(
        workspace_root,
        presentation_dir,
        ("presentation_video.mp4", "presentation_video_notebooklm.mp4"),
    )
    artifacts = directory_artifacts + video_artifacts
    if len(directory_artifacts) == 2 and len(video_artifacts) == 2:
        status = "complete"
    elif artifacts:
        status = "partial"
    else:
        status = "missing"
    return PresentationStageState(status=status, artifacts=artifacts, notes=())


def _video_sync_validation_state(
    workspace_root: Path,
    presentation_dir: Path,
) -> PresentationStageState:
    return _stage_state_from_required_files(
        workspace_root,
        presentation_dir,
        (
            "video_sync_validation_single_speaker.json",
            "video_sync_validation_notebooklm.json",
        ),
        optional_relative_paths=(
            "video_sync_validation_single_speaker.md",
            "video_sync_validation_notebooklm.md",
        ),
    )


def _legacy_markdown_compatibility(
    workspace_root: Path,
    presentation_dir: Path,
) -> LegacyMarkdownCompatibility | None:
    legacy_paths = collect_legacy_markdown_paths(presentation_dir)
    if not legacy_paths:
        return None
    artifacts = tuple(
        PresentationArtifactRef.from_path(
            path,
            workspace_root=workspace_root,
            kind="file",
        )
        for path in legacy_paths
    )
    return LegacyMarkdownCompatibility(
        support_level="debug-export-only",
        production_evidence=False,
        artifacts=artifacts,
        notes=(
            "Retired numbered Markdown artifacts are compatibility/debug exports only and cannot satisfy production stage state.",
        ),
    )


def _human_review_state(
    workspace_root: Path,
    presentation_dir: Path,
) -> PresentationStageState:
    receipt_path = presentation_dir / "human_review_receipt.json"
    artifacts = _existing_file_artifacts(workspace_root, presentation_dir, ("human_review_receipt.json",))
    if not artifacts:
        return PresentationStageState(
            status="missing",
            artifacts=(),
            notes=(
                "Explicit review evidence is absent; Phase 22 does not infer human review completion.",
            ),
        )
    receipt = _read_json_object(receipt_path)
    if receipt is None:
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=(
                "Human review receipt is present but is not readable JSON; operator approval is not recorded.",
            ),
        )
    if _human_review_receipt_is_approved(receipt):
        return PresentationStageState(
            status="complete",
            artifacts=artifacts,
            notes=(
                "Explicit approved human review receipt is present; validation records the receipt without inferring subjective quality.",
            ),
        )
    return PresentationStageState(
        status="partial",
        artifacts=artifacts,
        notes=(
            "Human review receipt is present but does not record an approved decision.",
        ),
    )


def _blueprint_state(
    workspace_root: Path,
    presentation_dir: Path,
) -> PresentationStageState:
    required_paths = (
        "presentation_frame.json",
        "presentation_blueprint.json",
        "presentation_blueprint.md",
        "presentation_blueprint_receipt.json",
        "blueprint_review.json",
        "blueprint_review.md",
    )
    artifacts = _existing_file_artifacts(workspace_root, presentation_dir, required_paths)
    if not artifacts:
        return PresentationStageState(
            status="missing",
            artifacts=(),
            notes=("Presentation blueprint artifacts are absent.",),
        )
    if len(artifacts) != len(required_paths):
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("Presentation blueprint artifact set is incomplete.",),
        )

    frame_path = presentation_dir / "presentation_frame.json"
    blueprint_path = presentation_dir / "presentation_blueprint.json"
    receipt_path = presentation_dir / "presentation_blueprint_receipt.json"
    review_path = presentation_dir / "blueprint_review.json"
    frame_payload = _read_json_object(frame_path)
    if frame_payload is None:
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("presentation_frame.json is present but is not readable JSON.",),
        )
    frame_result = validate_presentation_frame_payload(frame_payload)
    if frame_result.errors:
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("presentation_frame.json does not satisfy the audience framing contract.",),
        )
    blueprint = _read_json_object(blueprint_path)
    if blueprint is None:
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("Presentation blueprint is present but is not readable JSON.",),
        )
    frame_validation = blueprint.get("presentation_frame_validation")
    if not isinstance(frame_validation, dict) or frame_validation.get("status") != "pass":
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("Presentation blueprint is missing passing presentation_frame_validation.",),
        )
    input_hashes = blueprint.get("input_hashes")
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    frame_hash = str(input_hashes.get("presentation_frame.json") or "").strip()
    if not frame_hash or frame_hash != sha256_file(frame_path):
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("Presentation blueprint is not hash-bound to the current presentation_frame.json.",),
        )
    receipt = _read_json_object(receipt_path)
    if receipt is None:
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("Presentation blueprint receipt is present but is not readable JSON.",),
        )
    if not _blueprint_receipt_is_bound(receipt, blueprint_path):
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("Presentation blueprint receipt is not bound to the current blueprint hash.",),
        )
    review_result = validate_blueprint_review_file(blueprint_path, review_path)
    if review_result.errors:
        return PresentationStageState(
            status="partial",
            artifacts=artifacts,
            notes=("Blueprint review token is present but is not approved for draft generation.",),
        )
    return PresentationStageState(
        status="complete",
        artifacts=artifacts,
        notes=("Hash-bound approved presentation blueprint review is present.",),
    )


def _topic_scope_for_presentation(
    presentation_id: str,
    presentation_type: str,
) -> PresentationTopicScope:
    if presentation_type == "topic-deep-dive":
        return PresentationTopicScope(mode="single-topic", topics=(presentation_id,))
    return PresentationTopicScope(mode="all-topics", topics=())


def _analysis_artifacts_for_presentation(
    workspace_root: Path,
    presentation_id: str,
    presentation_type: str,
) -> tuple[PresentationArtifactRef, ...]:
    analysis_dir = workspace_root / "analysis"
    slide_brief_dir = analysis_dir / "slide_briefs"
    if presentation_type == "topic-deep-dive":
        candidate = slide_brief_dir / f"{presentation_id}.md"
        if not candidate.is_file():
            candidate = analysis_dir / f"{presentation_id}.md"
        if not candidate.is_file():
            return ()
        return (
            PresentationArtifactRef.from_path(
                candidate,
                workspace_root=workspace_root,
                kind="file",
            ),
        )
    if presentation_type == "executive-summary":
        summary_paths = [
            analysis_dir / "executive_summary.md",
            analysis_dir / "trend_report.md",
        ]
        analysis_paths = [
            path for path in summary_paths if path.is_file()
        ] + sorted(
            path
            for path in analysis_dir.glob("*.md")
            if _ANALYSIS_TOPIC_FILE_RE.match(path.name)
        )
        return tuple(
            PresentationArtifactRef.from_path(
                path,
                workspace_root=workspace_root,
                kind="file",
            )
            for path in analysis_paths
        )
    analysis_paths = sorted(
        path
        for path in (
            slide_brief_dir.glob("*.md") if slide_brief_dir.is_dir() else analysis_dir.glob("*.md")
        )
        if _ANALYSIS_TOPIC_FILE_RE.match(path.name)
    )
    return tuple(
        PresentationArtifactRef.from_path(
            path,
            workspace_root=workspace_root,
            kind="file",
        )
        for path in analysis_paths
    )


def _presentation_targets(presentation_type: str, presentation_id: str = "") -> tuple[int, int]:
    if presentation_type == "topic-deep-dive":
        return 60, 60
    if presentation_type == "master":
        return 90, 90
    if presentation_type == "executive-summary":
        if presentation_id == "executive_summary_short":
            return 12, 12
        if presentation_id == "executive_summary_long":
            return 30, 30
        return 30, 30
    raise PresentationManifestValidationError(f"unsupported presentation_type: {presentation_type}")


def _backfilled_target_slide_count(
    presentation_dir: Path,
    default_slide_count: int,
) -> int:
    mapping_path = presentation_dir / "presentation_slide_mapping.json"
    if not mapping_path.is_file():
        return default_slide_count
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_slide_count
    slides = payload.get("slides") if isinstance(payload, dict) else None
    if isinstance(slides, list) and slides:
        return len(slides)
    if isinstance(payload, dict):
        numeric_keys = [key for key in payload if isinstance(key, str) and key.isdigit()]
        if numeric_keys:
            return len(numeric_keys)
    return default_slide_count


def _configured_target_slide_count(
    workspace: Any,
    presentation_type: str,
    default_slide_count: int,
) -> int | None:
    if presentation_type != "topic-deep-dive":
        return None
    workflow_config = getattr(workspace, "workflow", None)
    deck_config = getattr(workflow_config, "deck", None)
    target_slides = getattr(deck_config, "target_slides", None)
    if not isinstance(target_slides, int):
        return None
    if target_slides <= 0 or target_slides == default_slide_count:
        return None
    return target_slides


def _configured_density_targets(
    workspace: Any,
    presentation_type: str,
) -> tuple[float, float]:
    if presentation_type != "topic-deep-dive":
        return 0.0, 0.0
    workflow_config = getattr(workspace, "workflow", None)
    deck_config = getattr(workflow_config, "deck", None)
    hyperlinks = getattr(deck_config, "target_hyperlinks_per_slide", 0.0)
    svgs = getattr(deck_config, "target_svgs_per_slide", 0.0)
    return (
        float(hyperlinks) if isinstance(hyperlinks, int | float) and hyperlinks >= 0 else 0.0,
        float(svgs) if isinstance(svgs, int | float) and svgs >= 0 else 0.0,
    )


def _presentation_title(
    presentation_id: str,
    presentation_type: str,
    source_analysis_artifacts: Sequence[PresentationArtifactRef],
    workspace_root: Path,
) -> str:
    if presentation_type == "master":
        return "Master Presentation"
    if presentation_type == "executive-summary":
        if presentation_id == "executive_summary_short":
            return "Executive Summary - Short"
        if presentation_id == "executive_summary_long":
            return "Executive Summary - Long"
        return "Executive Summary"
    if source_analysis_artifacts:
        source_path = workspace_root / source_analysis_artifacts[0].path
        title = _extract_title(source_path)
        if title:
            return _deck_display_title(title)
    return presentation_id.split("_", 1)[-1].replace("_", " ").title()


def _backfill_stage_states(
    workspace_root: Path,
    presentation_dir: Path,
) -> dict[str, PresentationStageState]:
    blueprint_state = _blueprint_state(workspace_root, presentation_dir)
    stage_states: dict[str, PresentationStageState] = {
        "presentation_blueprint": blueprint_state,
        "deck_contract_generation": _stage_state_from_required_files(
            workspace_root,
            presentation_dir,
            (
                "presentation_deck.json",
                "presentation_deck_receipt.json",
            ),
        ),
        "source_binding": _stage_state_from_required_files(
            workspace_root,
            presentation_dir,
            (
                "source_binding_report.json",
                "source_pack_receipt.json",
                "ranked_papers.json",
            ),
        ),
        "visual_plan_generation": _visual_plan_generation_state(workspace_root, presentation_dir),
        "render_spec_generation": _stage_state_from_required_files(
            workspace_root,
            presentation_dir,
            (
                "render_spec.json",
            ),
        ),
        "render_export": _stage_state_from_required_files(
            workspace_root,
            presentation_dir,
            ("presentation.html", "presentation.pdf", "presentation_slide_mapping.json"),
        ),
        "visual_qc": _stage_state_from_required_files(
            workspace_root,
            presentation_dir,
            (
                "assets/generated/visual_qc_report.md",
                "assets/generated/visual_llm_qc_report.json",
            ),
        ),
        "delivery_generation": _delivery_generation_state(workspace_root, presentation_dir),
        "presentation_review": _stage_state_from_required_files(
            workspace_root,
            presentation_dir,
            (
                "presentation_review.json",
                "presentation_review.md",
                "presentation_review.prompt.md",
            ),
        ),
        "video_generation": _video_generation_state(workspace_root, presentation_dir),
        "video_sync_validation": _video_sync_validation_state(workspace_root, presentation_dir),
        "human_review": _human_review_state(workspace_root, presentation_dir),
    }
    return {stage_id: stage_states[stage_id] for stage_id in PRESENTATION_STAGE_IDS}


def build_backfilled_presentations(
    workspace: Any,
    *,
    warn: Callable[[str], None] | None = None,
) -> tuple[PresentationEntry, ...]:
    workspace_root = workspace.root.resolve()
    decks_dir = workspace_root / "decks"
    if not decks_dir.is_dir():
        return ()
    presentations: list[PresentationEntry] = []
    for presentation_dir in sorted(path for path in decks_dir.iterdir() if path.is_dir()):
        presentation_id = presentation_dir.name
        if _CANONICAL_TOPIC_DIR_RE.match(presentation_id):
            presentation_type = "topic-deep-dive"
        elif presentation_id == "master_presentation":
            presentation_type = "master"
        elif presentation_id in EXECUTIVE_SUMMARY_PRESENTATION_IDS:
            presentation_type = "executive-summary"
        else:
            if warn is not None:
                warn(
                    "Skipping non-canonical presentation directory: "
                    f"{_relative_to_workspace(presentation_dir, workspace_root)}"
                )
            continue
        source_analysis_artifacts = _analysis_artifacts_for_presentation(
            workspace_root,
            presentation_id,
            presentation_type,
        )
        target_duration_minutes, target_slide_count = _presentation_targets(
            presentation_type,
            presentation_id,
        )
        configured_target_slide_count = _configured_target_slide_count(
            workspace,
            presentation_type,
            target_slide_count,
        )
        if configured_target_slide_count is not None:
            target_slide_count = configured_target_slide_count
        else:
            target_slide_count = _backfilled_target_slide_count(
                presentation_dir,
                target_slide_count,
            )
        target_hyperlinks_per_slide, target_svgs_per_slide = _configured_density_targets(
            workspace,
            presentation_type,
        )
        presentations.append(
            PresentationEntry(
                id=presentation_id,
                presentation_type=presentation_type,
                title=_presentation_title(
                    presentation_id,
                    presentation_type,
                    source_analysis_artifacts,
                    workspace_root,
                ),
                topic_scope=_topic_scope_for_presentation(
                    presentation_id,
                    presentation_type,
                ),
                target_duration_minutes=target_duration_minutes,
                target_slide_count=target_slide_count,
                source_analysis_artifacts=source_analysis_artifacts,
                stage_states=_backfill_stage_states(workspace_root, presentation_dir),
                target_hyperlinks_per_slide=target_hyperlinks_per_slide,
                target_svgs_per_slide=target_svgs_per_slide,
                legacy_markdown=_legacy_markdown_compatibility(
                    workspace_root,
                    presentation_dir,
                ),
            )
        )
    return tuple(presentations)


def _load_manifest_json(manifest_path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PresentationManifestValidationError(
            "Missing presentation_manifest.json. Run "
            "`python3 scripts/backfill_presentation_manifest.py --workspace <workspace>` first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise PresentationManifestValidationError(
            f"Malformed presentation manifest: {manifest_path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise PresentationManifestValidationError("manifest root must be a JSON object")
    return loaded


def _validate_artifact_refs(
    workspace_root: Path,
    artifacts: Sequence[PresentationArtifactRef],
    *,
    require_path_existence: bool,
    label: str,
) -> None:
    for index, artifact in enumerate(artifacts):
        artifact_path = _resolve_declared_path(
            workspace_root,
            artifact.path,
            label=f"{label}[{index}].path",
            require_exists=require_path_existence,
        )
        if not require_path_existence:
            continue
        expected_sha = _artifact_sha256(artifact_path, kind=artifact.kind)
        if expected_sha != artifact.sha256:
            raise PresentationManifestValidationError(
                f"{label}[{index}].sha256 mismatch for {artifact.path}"
            )


def load_and_validate_manifest(
    workspace: Any,
    *,
    require_path_existence: bool = True,
) -> LoadedPresentationManifest:
    workspace_root = workspace.root.resolve()
    manifest_path = workspace_root / MANIFEST_RELATIVE_PATH
    data = _load_manifest_json(manifest_path)
    manifest = PresentationManifest.from_json(data)
    recomputed_content_sha256 = _compute_content_sha256(data)
    if manifest.content_sha256 != recomputed_content_sha256:
        raise PresentationManifestValidationError("content_sha256 mismatch: manifest tampered")
    for presentation in manifest.presentations:
        _validate_artifact_refs(
            workspace_root,
            presentation.source_analysis_artifacts,
            require_path_existence=require_path_existence,
            label=f"presentations[{presentation.id}].source_analysis_artifacts",
        )
        for stage_id, stage_state in presentation.stage_states.items():
            _validate_artifact_refs(
                workspace_root,
                stage_state.artifacts,
                require_path_existence=require_path_existence,
                label=f"presentations[{presentation.id}].stage_states.{stage_id}.artifacts",
            )
    return LoadedPresentationManifest(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        content_sha256=recomputed_content_sha256,
    )
