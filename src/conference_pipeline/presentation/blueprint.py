"""Presentation Blueprint contract validation and approval-token handling."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from conference_pipeline.workflow_runs import write_json_atomic

MAX_SLIDE_TITLE_CHARS = 66
BLUEPRINT_SCHEMA_VERSION = 1

BLUEPRINT_SCHEMA_INVALID = "BLUEPRINT_SCHEMA_INVALID"
BLUEPRINT_SOURCE_BINDING_MISSING = "BLUEPRINT_SOURCE_BINDING_MISSING"
BLUEPRINT_SOURCE_UNRESOLVED = "BLUEPRINT_SOURCE_UNRESOLVED"
BLUEPRINT_VISUAL_ROLE_GENERIC = "BLUEPRINT_VISUAL_ROLE_GENERIC"
BLUEPRINT_DURATION_TARGET_EXCEEDED = "BLUEPRINT_DURATION_TARGET_EXCEEDED"
BLUEPRINT_DURATION_OVER_BUDGET = "BLUEPRINT_DURATION_OVER_BUDGET"
BLUEPRINT_APPROVAL_INVALID = "BLUEPRINT_APPROVAL_INVALID"
BLUEPRINT_HASH_MISMATCH = "BLUEPRINT_HASH_MISMATCH"
BLUEPRINT_RECEIPT_INVALID = "BLUEPRINT_RECEIPT_INVALID"

APPROVED_FOR_DRAFT_GENERATION = "approved_for_draft_generation"
APPROVED_FOR_DECK_CONTRACT_GENERATION = "approved_for_deck_contract_generation"
APPROVED_BLUEPRINT_REVIEW_STATUSES = frozenset(
    {APPROVED_FOR_DRAFT_GENERATION, APPROVED_FOR_DECK_CONTRACT_GENERATION}
)
BLUEPRINT_TARGET_RUNTIME_WARN_MINUTES = 38.0
BLUEPRINT_RUNTIME_HARD_BUDGET_MINUTES = 40.0

GATE_TYPES = frozenset({"deterministic", "LLM-judge", "human"})
REQUIRED_APPROVAL_GATES = frozenset({"deterministic", "LLM-judge", "human"})

REQUIRED_BLUEPRINT_FIELDS: tuple[str, ...] = (
    "blueprint_id",
    "revision",
    "deck_thesis",
    "audience_promise",
    "target_runtime_minutes",
    "target_slide_count",
    "telescopic_section_map",
    "subtopic_clusters",
    "slides",
    "conclusion_synthesis",
    "provenance",
)

_REQUIRED_STRING_FIELDS = frozenset(
    {
        "blueprint_id",
        "deck_thesis",
        "audience_promise",
        "conclusion_synthesis",
    }
)
_REQUIRED_LIST_FIELDS = frozenset(
    {"telescopic_section_map", "subtopic_clusters", "slides"}
)
_GENERIC_VISUAL_ROLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*diagram\s+showing\b", re.IGNORECASE),
    re.compile(r"^\s*generic\s+diagram\s*$", re.IGNORECASE),
    re.compile(r"^\s*decorative\s*$", re.IGNORECASE),
    re.compile(r"^\s*copied\s+slide\s+bullets?\s*$", re.IGNORECASE),
)


@dataclass(frozen=True)
class PresentationBlueprintError:
    """One stable blueprint contract finding."""

    code: str
    message: str
    path: str
    gate_type: str

    def __post_init__(self) -> None:
        if self.gate_type not in GATE_TYPES:
            raise ValueError(f"gate_type must be one of {sorted(GATE_TYPES)}")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "gate_type": self.gate_type,
        }


@dataclass(frozen=True)
class PresentationBlueprint:
    """Parsed subset of the blueprint contract used by deterministic gates."""

    blueprint_id: str
    repair_of: str | None
    revision: int
    source_plan_repair_of: str | None
    deck_thesis: str
    audience_promise: str
    target_runtime_minutes: float
    target_slide_count: int
    telescopic_section_map: tuple[dict[str, Any], ...]
    subtopic_clusters: tuple[dict[str, Any], ...]
    slides: tuple[dict[str, Any], ...]
    conclusion_synthesis: str
    provenance: dict[str, Any]
    source_labels: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "repair_of": self.repair_of,
            "revision": self.revision,
            "source_plan_repair_of": self.source_plan_repair_of,
            "deck_thesis": self.deck_thesis,
            "audience_promise": self.audience_promise,
            "target_runtime_minutes": self.target_runtime_minutes,
            "target_slide_count": self.target_slide_count,
            "telescopic_section_map": list(self.telescopic_section_map),
            "subtopic_clusters": list(self.subtopic_clusters),
            "slides": list(self.slides),
            "conclusion_synthesis": self.conclusion_synthesis,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class PresentationBlueprintValidationResult:
    blueprint: PresentationBlueprint | None
    errors: tuple[PresentationBlueprintError, ...]
    warnings: tuple[PresentationBlueprintError, ...]

    @property
    def status(self) -> str:
        if self.errors:
            return "fail"
        if self.warnings:
            return "warn"
        return "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "blueprint_id": self.blueprint.blueprint_id if self.blueprint else None,
        }


@dataclass(frozen=True)
class BlueprintInput:
    kind: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ApprovedBlueprintSelection:
    """Latest hash-approved blueprint/review pair."""

    blueprint_path: Path
    review_path: Path
    blueprint_id: str
    revision: int
    blueprint_sha256: str


class PresentationBlueprintValidationError(ValueError):
    """Raised when callers request fail-closed blueprint validation."""

    def __init__(self, errors: Sequence[PresentationBlueprintError]) -> None:
        self.errors = tuple(errors)
        joined = "; ".join(f"{error.code}: {error.message}" for error in self.errors)
        super().__init__(joined or "presentation blueprint validation failed")


def validate_blueprint_file(
    blueprint_path: str | Path,
    *,
    fail_on_error: bool = False,
) -> PresentationBlueprintValidationResult:
    """Validate a ``presentation_blueprint.json`` file."""

    path = Path(blueprint_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        result = PresentationBlueprintValidationResult(
            blueprint=None,
            errors=(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    "presentation_blueprint.json is missing",
                    "$",
                    "deterministic",
                ),
            ),
            warnings=(),
        )
        if fail_on_error:
            raise PresentationBlueprintValidationError(result.errors)
        return result
    except json.JSONDecodeError as exc:
        result = PresentationBlueprintValidationResult(
            blueprint=None,
            errors=(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    f"presentation_blueprint.json is malformed JSON: {exc.msg}",
                    "$",
                    "deterministic",
                ),
            ),
            warnings=(),
        )
        if fail_on_error:
            raise PresentationBlueprintValidationError(result.errors)
        return result

    result = validate_blueprint_payload(payload)
    if fail_on_error and result.errors:
        raise PresentationBlueprintValidationError(result.errors)
    return result


def validate_blueprint_payload(payload: Any) -> PresentationBlueprintValidationResult:
    """Validate a loaded blueprint payload."""

    errors: list[PresentationBlueprintError] = []
    warnings: list[PresentationBlueprintError] = []
    if not isinstance(payload, dict):
        return PresentationBlueprintValidationResult(
            blueprint=None,
            errors=(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    "presentation blueprint root must be a JSON object",
                    "$",
                    "deterministic",
                ),
            ),
            warnings=(),
        )

    for field in REQUIRED_BLUEPRINT_FIELDS:
        if field not in payload:
            errors.append(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    f"missing required blueprint field: {field}",
                    f"$.{field}",
                    "deterministic",
                )
            )

    for field in _REQUIRED_STRING_FIELDS:
        value = payload.get(field)
        if field in payload and (not isinstance(value, str) or not value.strip()):
            errors.append(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    f"{field} must be a non-empty string",
                    f"$.{field}",
                    "deterministic",
                )
            )

    for field in _REQUIRED_LIST_FIELDS:
        value = payload.get(field)
        if field in payload and not isinstance(value, list):
            errors.append(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    f"{field} must be a list",
                    f"$.{field}",
                    "deterministic",
                )
            )

    revision = payload.get("revision")
    if "revision" in payload and (
        not isinstance(revision, int) or isinstance(revision, bool) or revision < 1
    ):
        errors.append(
            _error(
                BLUEPRINT_SCHEMA_INVALID,
                "revision must be a positive integer",
                "$.revision",
                "deterministic",
            )
        )

    target_slide_count = payload.get("target_slide_count")
    if "target_slide_count" in payload and (
        not isinstance(target_slide_count, int)
        or isinstance(target_slide_count, bool)
        or target_slide_count < 1
    ):
        errors.append(
            _error(
                BLUEPRINT_SCHEMA_INVALID,
                "target_slide_count must be a positive integer",
                "$.target_slide_count",
                "deterministic",
            )
        )

    target_runtime = _optional_number(payload.get("target_runtime_minutes"))
    if "target_runtime_minutes" in payload and target_runtime is None:
        errors.append(
            _error(
                BLUEPRINT_SCHEMA_INVALID,
                "target_runtime_minutes must be a number",
                "$.target_runtime_minutes",
                "deterministic",
            )
        )

    provenance = payload.get("provenance")
    if "provenance" in payload and not isinstance(provenance, dict):
        errors.append(
            _error(
                BLUEPRINT_SCHEMA_INVALID,
                "provenance must be a JSON object",
                "$.provenance",
                "deterministic",
            )
        )
        provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}

    source_labels = _source_metadata_labels(provenance, errors)
    slides = payload.get("slides") if isinstance(payload.get("slides"), list) else []
    clusters = (
        payload.get("subtopic_clusters")
        if isinstance(payload.get("subtopic_clusters"), list)
        else []
    )

    for index, slide in enumerate(slides):
        if isinstance(slide, dict) and len(str(slide.get("title") or "")) > MAX_SLIDE_TITLE_CHARS:
            errors.append(_error("BLUEPRINT_TITLE_TOO_LONG",
                                 f"slide title exceeds {MAX_SLIDE_TITLE_CHARS} characters; rewrite it as a complete phrase",
                                 f"$.slides[{index}].title", "deterministic"))
    errors.extend(_validate_cluster_bindings(clusters))
    errors.extend(_validate_slide_sources(slides, source_labels))
    errors.extend(_validate_visual_roles(slides))

    estimated_runtime = _estimated_runtime_minutes(payload, slides)
    if estimated_runtime is not None:
        if estimated_runtime > BLUEPRINT_RUNTIME_HARD_BUDGET_MINUTES:
            errors.append(
                _error(
                    BLUEPRINT_DURATION_OVER_BUDGET,
                    "estimated runtime exceeds 40 minutes",
                    "$.target_runtime_minutes",
                    "deterministic",
                )
            )
        elif estimated_runtime > BLUEPRINT_TARGET_RUNTIME_WARN_MINUTES:
            warnings.append(
                _error(
                    BLUEPRINT_DURATION_TARGET_EXCEEDED,
                    "estimated runtime exceeds 38 minute target; Phase 13 owns final <=40 min render validation",
                    "$.target_runtime_minutes",
                    "deterministic",
                )
            )

    blueprint: PresentationBlueprint | None = None
    if not errors:
        blueprint = PresentationBlueprint(
            blueprint_id=str(payload["blueprint_id"]),
            repair_of=_optional_string(payload.get("repair_of")),
            revision=int(payload["revision"]),
            source_plan_repair_of=_optional_string(payload.get("source_plan_repair_of")),
            deck_thesis=str(payload["deck_thesis"]),
            audience_promise=str(payload["audience_promise"]),
            target_runtime_minutes=float(payload["target_runtime_minutes"]),
            target_slide_count=int(payload["target_slide_count"]),
            telescopic_section_map=tuple(payload["telescopic_section_map"]),
            subtopic_clusters=tuple(payload["subtopic_clusters"]),
            slides=tuple(payload["slides"]),
            conclusion_synthesis=str(payload["conclusion_synthesis"]),
            provenance=provenance,
            source_labels=frozenset(source_labels),
        )

    return PresentationBlueprintValidationResult(
        blueprint=blueprint,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_blueprint_review_file(
    blueprint_path: str | Path,
    review_path: str | Path,
    *,
    fail_on_error: bool = False,
) -> PresentationBlueprintValidationResult:
    """Validate ``blueprint_review.json`` as a hash-bound approval token."""

    blueprint = Path(blueprint_path)
    review = Path(review_path)
    errors: list[PresentationBlueprintError] = []

    try:
        review_payload = json.loads(review.read_text(encoding="utf-8"))
    except FileNotFoundError:
        review_payload = None
        errors.append(
            _error(
                BLUEPRINT_APPROVAL_INVALID,
                "blueprint_review.json is missing; production draft_generation requires approved_for_draft_generation",
                "$",
                "human",
            )
        )
    except json.JSONDecodeError as exc:
        review_payload = None
        errors.append(
            _error(
                BLUEPRINT_APPROVAL_INVALID,
                f"blueprint_review.json is malformed JSON: {exc.msg}",
                "$",
                "human",
            )
        )

    if isinstance(review_payload, dict):
        errors.extend(_validate_approval_token_shape(review_payload))
        if not errors:
            expected_hash = _sha256_file(blueprint)
            actual_hash = str(review_payload.get("blueprint_sha256", ""))
            if actual_hash != expected_hash:
                errors.append(
                    _error(
                        BLUEPRINT_HASH_MISMATCH,
                        "blueprint_sha256 does not match presentation_blueprint.json",
                        "$.blueprint_sha256",
                        "deterministic",
                    )
                )
    elif review_payload is not None:
        errors.append(
            _error(
                BLUEPRINT_APPROVAL_INVALID,
                "blueprint_review.json root must be a JSON object with approved_for_draft_generation",
                "$",
                "human",
            )
        )

    result = PresentationBlueprintValidationResult(
        blueprint=None,
        errors=tuple(errors),
        warnings=(),
    )
    if fail_on_error and result.errors:
        raise PresentationBlueprintValidationError(result.errors)
    return result


def require_approved_blueprint(
    blueprint_path: str | Path,
    review_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    require_live_receipt: bool = False,
) -> dict[str, Any]:
    """Return the review token or raise on invalid/mismatched approval."""

    validate_blueprint_file(blueprint_path, fail_on_error=True)
    validate_blueprint_review_file(blueprint_path, review_path, fail_on_error=True)
    if receipt_path is None and require_live_receipt:
        raise PresentationBlueprintValidationError(
            (
                _error(
                    BLUEPRINT_RECEIPT_INVALID,
                    "presentation_blueprint_receipt.json is required for live production draft generation",
                    "$",
                    "deterministic",
                ),
            )
        )
    if receipt_path is not None:
        validate_blueprint_receipt_file(
            blueprint_path,
            receipt_path,
            # Both the LLM ``live`` path and the deterministic ``source-ledger``
            # path are valid production blueprint generators; provider receipts
            # are enforced only for ``live`` inside the validator.
            accepted_modes=frozenset({"live", "source-ledger"}) if require_live_receipt else None,
            require_provider_evidence=require_live_receipt,
            fail_on_error=True,
        )
    return json.loads(Path(review_path).read_text(encoding="utf-8"))


def validate_blueprint_receipt_file(
    blueprint_path: str | Path,
    receipt_path: str | Path,
    *,
    require_mode: str | None = None,
    accepted_modes: frozenset[str] | None = None,
    require_provider_evidence: bool = False,
    fail_on_error: bool = False,
) -> PresentationBlueprintValidationResult:
    """Validate that a blueprint generation receipt is hash-bound and production-qualified."""

    blueprint = Path(blueprint_path)
    receipt = Path(receipt_path)
    errors: list[PresentationBlueprintError] = []

    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = None
        errors.append(
            _error(
                BLUEPRINT_RECEIPT_INVALID,
                "presentation_blueprint_receipt.json is missing",
                "$",
                "deterministic",
            )
        )
    except json.JSONDecodeError as exc:
        payload = None
        errors.append(
            _error(
                BLUEPRINT_RECEIPT_INVALID,
                f"presentation_blueprint_receipt.json is malformed JSON: {exc.msg}",
                "$",
                "deterministic",
            )
        )

    if isinstance(payload, dict):
        expected_hash = _sha256_file(blueprint)
        actual_hash = str(payload.get("blueprint_sha256", ""))
        if actual_hash != expected_hash:
            errors.append(
                _error(
                    BLUEPRINT_HASH_MISMATCH,
                    "receipt blueprint_sha256 does not match presentation_blueprint.json",
                    "$.blueprint_sha256",
                    "deterministic",
                )
            )
        receipt_mode = payload.get("mode")
        if require_mode is not None and receipt_mode != require_mode:
            errors.append(
                _error(
                    BLUEPRINT_RECEIPT_INVALID,
                    f"presentation_blueprint_receipt.json mode must be {require_mode}",
                    "$.mode",
                    "deterministic",
                )
            )
        if accepted_modes is not None and receipt_mode not in accepted_modes:
            errors.append(
                _error(
                    BLUEPRINT_RECEIPT_INVALID,
                    "presentation_blueprint_receipt.json mode must be one of "
                    f"{sorted(accepted_modes)}",
                    "$.mode",
                    "deterministic",
                )
            )
        if require_provider_evidence:
            if payload.get("production_evidence") is not True:
                errors.append(
                    _error(
                        BLUEPRINT_RECEIPT_INVALID,
                        "presentation_blueprint_receipt.json must record production_evidence=true",
                        "$.production_evidence",
                        "deterministic",
                    )
                )
            # Provider receipt paths are evidence of an LLM call, so they are
            # required only for the ``live`` generation mode. The deterministic
            # ``source-ledger`` mode is a valid production path that derives the
            # blueprint from hash-bound source contracts and has no provider call.
            if receipt_mode == "live":
                provider_receipts = payload.get("provider_receipt_paths")
                if not isinstance(provider_receipts, list) or not provider_receipts:
                    errors.append(
                        _error(
                            BLUEPRINT_RECEIPT_INVALID,
                            "presentation_blueprint_receipt.json must include provider_receipt_paths for live generation",
                            "$.provider_receipt_paths",
                            "deterministic",
                        )
                    )
        if "repair" in payload:
            errors.extend(_validate_receipt_repair_chain(payload.get("repair")))
    elif payload is not None:
        errors.append(
            _error(
                BLUEPRINT_RECEIPT_INVALID,
                "presentation_blueprint_receipt.json root must be a JSON object",
                "$",
                "deterministic",
            )
        )

    result = PresentationBlueprintValidationResult(
        blueprint=None,
        errors=tuple(errors),
        warnings=(),
    )
    if fail_on_error and result.errors:
        raise PresentationBlueprintValidationError(result.errors)
    return result


def _validate_receipt_repair_chain(repair: Any) -> list[PresentationBlueprintError]:
    """A repaired blueprint's receipt must point at the archived receipt it replaced.

    The receipt of a repair (``scripts/presentations/repair_presentation_blueprint.py``)
    re-binds the generation evidence to the repaired blueprint's hash. That is only
    honest when the parent receipt is still on disk, unchanged, and was itself bound
    to the parent blueprint, so a hand-edited blueprint cannot borrow a receipt.
    """

    def invalid(message: str, path: str) -> PresentationBlueprintError:
        return _error(BLUEPRINT_RECEIPT_INVALID, message, path, "deterministic")

    if not isinstance(repair, dict):
        return [invalid("receipt repair record must be a JSON object", "$.repair")]
    parent_receipt = Path(str(repair.get("parent_receipt_path") or ""))
    if not str(repair.get("parent_receipt_path") or "").strip() or not parent_receipt.is_file():
        return [invalid("repair parent_receipt_path must name the archived parent receipt", "$.repair.parent_receipt_path")]
    errors: list[PresentationBlueprintError] = []
    if _sha256_file(parent_receipt) != str(repair.get("parent_receipt_sha256") or ""):
        errors.append(invalid("archived parent receipt changed since the repair", "$.repair.parent_receipt_sha256"))
    try:
        parent_payload = json.loads(parent_receipt.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        parent_payload = None
    if not isinstance(parent_payload, dict) or str(parent_payload.get("blueprint_sha256") or "") != str(
        repair.get("parent_blueprint_sha256") or ""
    ):
        errors.append(invalid("parent receipt is not bound to the repaired blueprint's parent", "$.repair.parent_blueprint_sha256"))
    return errors


def select_latest_approved_blueprint(
    candidates: Sequence[tuple[str | Path, str | Path]],
) -> ApprovedBlueprintSelection | None:
    """Return the latest approved blueprint/review pair by revision.

    Only pairs with a passing blueprint, approval.status
    ``approved_for_draft_generation``, and matching ``blueprint_sha256`` are
    eligible. This deliberately ignores repaired blueprints until their own
    review token is written.
    """

    approved: list[ApprovedBlueprintSelection] = []
    for blueprint_candidate, review_candidate in candidates:
        blueprint_path = Path(blueprint_candidate)
        review_path = Path(review_candidate)
        blueprint_validation = validate_blueprint_file(blueprint_path)
        if blueprint_validation.errors or blueprint_validation.blueprint is None:
            continue
        review_validation = validate_blueprint_review_file(blueprint_path, review_path)
        if review_validation.errors:
            continue
        try:
            review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        approved.append(
            ApprovedBlueprintSelection(
                blueprint_path=blueprint_path,
                review_path=review_path,
                blueprint_id=blueprint_validation.blueprint.blueprint_id,
                revision=blueprint_validation.blueprint.revision,
                blueprint_sha256=str(review_payload["blueprint_sha256"]),
            )
        )
    if not approved:
        return None
    return max(approved, key=lambda item: (item.revision, item.blueprint_path.as_posix()))


def write_blueprint_receipt(
    workspace_root: str | Path,
    blueprint_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    presentation_id: str,
    review_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``presentation_blueprint_receipt.json`` with relative inputs and hashes."""

    workspace = Path(workspace_root).expanduser().resolve()
    blueprint = _resolve_inside(workspace, blueprint_path, label="presentation_blueprint")
    output = (
        _resolve_inside(workspace, receipt_path, label="presentation_blueprint_receipt")
        if receipt_path is not None
        else _resolve_inside(
            workspace,
            blueprint.parent / "presentation_blueprint_receipt.json",
            label="presentation_blueprint_receipt",
        )
    )
    inputs = [
        BlueprintInput(
            kind="presentation_blueprint",
            path=_relative_path(blueprint, workspace),
            sha256=_sha256_file(blueprint),
        )
    ]
    if review_path is not None:
        review = _resolve_inside(workspace, review_path, label="blueprint_review")
        inputs.append(
            BlueprintInput(
                kind="blueprint_review",
                path=_relative_path(review, workspace),
                sha256=_sha256_file(review),
            )
        )
    validation = validate_blueprint_file(blueprint)
    payload: dict[str, Any] = {
        "schema_version": BLUEPRINT_SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.presentation.blueprint",
        "presentation_id": presentation_id,
        "generated_at": _utc_now(),
        "blueprint_path": _relative_path(blueprint, workspace),
        "blueprint_sha256": inputs[0].sha256,
        "inputs": [item.to_dict() for item in inputs],
        "outputs": [_relative_path(output, workspace)],
        "validation": validation.to_dict(),
    }
    write_json_atomic(output, payload)
    return payload


def write_blueprint_review(
    workspace_root: str | Path,
    review_path: str | Path,
    review_payload: dict[str, Any],
) -> dict[str, Any]:
    """Write a blueprint review token through the project atomic JSON helper."""

    workspace = Path(workspace_root).expanduser().resolve()
    output = _resolve_inside(workspace, review_path, label="blueprint_review")
    write_json_atomic(output, review_payload)
    return review_payload


def _validate_approval_token_shape(
    payload: dict[str, Any],
) -> tuple[PresentationBlueprintError, ...]:
    errors: list[PresentationBlueprintError] = []
    if payload.get("status") != "pass":
        errors.append(
            _error(
                BLUEPRINT_APPROVAL_INVALID,
                "blueprint_review.json status must be pass for approved_for_draft_generation",
                "$.status",
                "human",
            )
        )
    approval = payload.get("approval")
    if not isinstance(approval, dict) or approval.get("status") not in APPROVED_BLUEPRINT_REVIEW_STATUSES:
        errors.append(
            _error(
                BLUEPRINT_APPROVAL_INVALID,
                "blueprint_review.json approval.status must be approved_for_draft_generation or approved_for_deck_contract_generation",
                "$.approval.status",
                "human",
            )
        )
    for field in ("blueprint_sha256", "approved_at", "approved_by", "review_artifact"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                _error(
                    BLUEPRINT_APPROVAL_INVALID,
                    f"blueprint_review.json missing required approval token field {field}",
                    f"$.{field}",
                    "human",
                )
            )
    gate_types = payload.get("gate_types")
    if not isinstance(gate_types, dict):
        errors.append(
            _error(
                BLUEPRINT_APPROVAL_INVALID,
                "blueprint_review.json missing required gate_types evidence",
                "$.gate_types",
                "human",
            )
        )
    else:
        for gate_type in sorted(REQUIRED_APPROVAL_GATES):
            if gate_types.get(gate_type) != "pass":
                errors.append(
                    _error(
                        BLUEPRINT_APPROVAL_INVALID,
                        f"blueprint_review.json gate_types.{gate_type} must be pass",
                        f"$.gate_types.{gate_type}",
                        gate_type,
                    )
                )
    return tuple(errors)


def _source_metadata_labels(
    provenance: dict[str, Any],
    errors: list[PresentationBlueprintError],
) -> set[str]:
    metadata = provenance.get("source_metadata")
    if not isinstance(metadata, list) or not metadata:
        errors.append(
            _error(
                BLUEPRINT_SOURCE_UNRESOLVED,
                "source metadata could not be resolved: provenance.source_metadata must list approved source labels",
                "$.provenance.source_metadata",
                "deterministic",
            )
        )
        return set()
    labels: set[str] = set()
    for index, item in enumerate(metadata):
        if not isinstance(item, dict):
            errors.append(
                _error(
                    BLUEPRINT_SOURCE_UNRESOLVED,
                    "source metadata could not be resolved: source metadata item must be an object",
                    f"$.provenance.source_metadata[{index}]",
                    "deterministic",
                )
            )
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(
                _error(
                    BLUEPRINT_SOURCE_UNRESOLVED,
                    "source metadata could not be resolved: source metadata item missing label",
                    f"$.provenance.source_metadata[{index}].label",
                    "deterministic",
                )
            )
            continue
        labels.add(label)
    return labels


def _validate_cluster_bindings(
    clusters: Sequence[Any],
) -> tuple[PresentationBlueprintError, ...]:
    errors: list[PresentationBlueprintError] = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            errors.append(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    "subtopic cluster must be a JSON object",
                    f"$.subtopic_clusters[{index}]",
                    "deterministic",
                )
            )
            continue
        sources = cluster.get("sources")
        slide_bindings = cluster.get("slide_bindings")
        if not _has_non_empty_string_list(sources) or not _has_non_empty_string_list(slide_bindings):
            errors.append(
                _error(
                    BLUEPRINT_SOURCE_BINDING_MISSING,
                    "cluster must bind at least one source to at least one slide",
                    f"$.subtopic_clusters[{index}]",
                    "deterministic",
                )
            )
    return tuple(errors)


def _validate_slide_sources(
    slides: Sequence[Any],
    source_labels: set[str],
) -> tuple[PresentationBlueprintError, ...]:
    errors: list[PresentationBlueprintError] = []
    for index, slide in enumerate(slides):
        if not isinstance(slide, dict):
            errors.append(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    "slide must be a JSON object",
                    f"$.slides[{index}]",
                    "deterministic",
                )
            )
            continue
        source_plan = slide.get("source_plan")
        if not source_plan:
            errors.append(
                _error(
                    BLUEPRINT_SOURCE_BINDING_MISSING,
                    "slide must include a non-empty source_plan bound to approved source labels",
                    f"$.slides[{index}].source_plan",
                    "deterministic",
                )
            )
            continue
        if not isinstance(source_plan, list):
            errors.append(
                _error(
                    BLUEPRINT_SCHEMA_INVALID,
                    "slide source_plan must be a list",
                    f"$.slides[{index}].source_plan",
                    "deterministic",
                )
            )
            continue
        for source_index, source in enumerate(source_plan):
            if isinstance(source, str):
                label = source
            elif isinstance(source, dict):
                label = source.get("label")
            else:
                label = None
            if not isinstance(label, str) or label not in source_labels:
                errors.append(
                    _error(
                        BLUEPRINT_SOURCE_UNRESOLVED,
                        "source metadata could not be resolved for slide source_plan label",
                        f"$.slides[{index}].source_plan[{source_index}]",
                        "deterministic",
                    )
                )
    return tuple(errors)


def _validate_visual_roles(slides: Sequence[Any]) -> tuple[PresentationBlueprintError, ...]:
    errors: list[PresentationBlueprintError] = []
    for index, slide in enumerate(slides):
        if not isinstance(slide, dict):
            continue
        visual_role = slide.get("visual_role")
        if not isinstance(visual_role, str) or not visual_role.strip():
            errors.append(
                _error(
                    BLUEPRINT_VISUAL_ROLE_GENERIC,
                    "visual role is too generic or missing; provide a story-serving visual role",
                    f"$.slides[{index}].visual_role",
                    "LLM-judge",
                )
            )
            continue
        if any(pattern.search(visual_role) for pattern in _GENERIC_VISUAL_ROLE_PATTERNS):
            errors.append(
                _error(
                    BLUEPRINT_VISUAL_ROLE_GENERIC,
                    "visual role is too generic; use a story-serving visual role instead",
                    f"$.slides[{index}].visual_role",
                    "LLM-judge",
                )
            )
    return tuple(errors)


def _estimated_runtime_minutes(payload: dict[str, Any], slides: Sequence[Any]) -> float | None:
    candidates: list[float] = []
    target_runtime = _optional_number(payload.get("target_runtime_minutes"))
    if target_runtime is not None:
        candidates.append(target_runtime)
    slide_total = 0.0
    found_slide_estimate = False
    for slide in slides:
        if not isinstance(slide, dict):
            continue
        estimate = _optional_number(slide.get("estimated_minutes"))
        if estimate is None:
            estimate = _optional_number(slide.get("timing_estimate_minutes"))
        if estimate is not None:
            found_slide_estimate = True
            slide_total += estimate
    if found_slide_estimate:
        candidates.append(slide_total)
    return max(candidates) if candidates else None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _has_non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value)


def _error(
    code: str,
    message: str,
    path: str,
    gate_type: str,
) -> PresentationBlueprintError:
    return PresentationBlueprintError(
        code=code,
        message=message,
        path=path,
        gate_type=gate_type,
    )


def _resolve_inside(root: Path, candidate: str | Path, *, label: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _relative_path(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def drop_unauthored_sections(
    blueprint: dict[str, Any],
    *,
    reasons: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Remove sections that have no slides, and clusters bound to no slide.

    A section can end up empty when every authoring call for it fails (for
    example the model declines that section's subject). Left in place, it
    claims coverage the deck does not deliver and fails the cluster-binding
    gate. Dropping it is honest only if it is recorded: each removal lands in
    ``blueprint["dropped_sections"]`` with its reason, and the cluster
    coverage justification no longer lists it. Returns the new records.
    """

    reasons = dict(reasons or {})
    sections = [s for s in blueprint.get("telescopic_section_map") or [] if isinstance(s, dict)]
    empty = [s for s in sections if not s.get("slide_ids")]
    clusters = [c for c in blueprint.get("subtopic_clusters") or [] if isinstance(c, dict)]
    unbound = [c for c in clusters if not _has_non_empty_string_list(c.get("slide_bindings"))]
    if not empty and not unbound:
        return []
    dropped_cluster_ids = {str(c.get("cluster_id") or "") for c in unbound}
    records: list[dict[str, Any]] = []
    for section in empty:
        section_id = str(section.get("section_id") or "")
        records.append(
            {
                "section_id": section_id,
                "title": str(section.get("title") or ""),
                "reason": reasons.get(section_id) or "no slides were authored for this section",
            }
        )
    for cluster in unbound:
        records.append(
            {
                "cluster_id": str(cluster.get("cluster_id") or ""),
                "title": str(cluster.get("title") or ""),
                "source_ids": list(cluster.get("source_ids") or []),
                "reason": "cluster bound to no slide after its section was dropped",
            }
        )
    blueprint["telescopic_section_map"] = [s for s in sections if s.get("slide_ids")]
    blueprint["subtopic_clusters"] = [
        c for c in clusters if str(c.get("cluster_id") or "") not in dropped_cluster_ids
    ]
    justification = blueprint.get("cluster_coverage_justification")
    if isinstance(justification, dict):
        for key in ("covered_cluster_ids", "required"):
            if isinstance(justification.get(key), list):
                justification[key] = [c for c in justification[key] if c not in dropped_cluster_ids]
    blueprint["dropped_sections"] = list(blueprint.get("dropped_sections") or []) + records
    return records
