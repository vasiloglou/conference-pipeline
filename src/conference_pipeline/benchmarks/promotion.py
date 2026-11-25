"""v3.0 Phase 03-01 — Generic operator-gated local candidate promotion.

This module provides the only sanctioned path to flip a local candidate from
``benchmark_pending`` to ``approved`` and to populate the full ``local_proof``
block required by ``approved_local_selection()``.

Promotion requires *all four* proof references:
  - ``research_ref``           — source-research document.
  - ``hardware_fit_ref``       — hardware fitness assessment document.
  - ``receipt_path``           — passing benchmark receipt JSON (rubric threshold check).
  - ``operator_approval_ref``  — operator sign-off document.

Missing *any* of these fails before the registry is touched (T-03-01 mitigation).

The registry write uses the same atomic-tempfile + parse-before-replace pattern as
``promote_vision.py`` (T-03-02 mitigation).

Dry-run receipts (``execution_mode != "live"`` or ``selection_decision != "passed"``)
cannot satisfy the promotion gate — before the rubric threshold check, the receipt is
validated through the typed receipt model and its provenance (``execution_mode``,
``selection_decision``, ``provider``, ``runtime_id``, ``task_id``, ``model_id``) is
checked, so a non-live/non-passing receipt is rejected even when its rubric scores are
above threshold (T-03-03 mitigation).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from conference_pipeline.model_governance.registry import (
    ModelGovernanceError,
    default_registry_path,
    load_registry,
)
from conference_pipeline.benchmarks.promote_vision import (
    PromotionError,
    PromotionRequiresReceiptError,
    PromotionThresholdsNotMetError,
    PromotionCandidateMissingError,
    RegistryDiff,
    _flip_candidate_status_in_yaml,
    _atomic_write_text,
    load_receipt_metrics,
)
from conference_pipeline.benchmarks.receipt_schema import (
    receipt_from_dict,
    validate_receipt,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PromotionRequiresOperatorApprovalError(PromotionError):
    """Raised when promotion is attempted without an operator_approval_ref."""


# ---------------------------------------------------------------------------
# Default thresholds (task-family-aware)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS_TRANSCRIPTION = {
    "quality": 0.6,
}

_DEFAULT_THRESHOLDS_VISION_OCR = {
    "text_recall": 0.6,
    "figure_caption_similarity": 0.4,
    "chart_summary_similarity": 0.4,
    "parse_success": 1.0,
    "overall": 0.6,
}

_DEFAULT_THRESHOLDS_GENERIC = {
    "quality": 0.5,
}


def _default_thresholds_for_task_family(task_family: str) -> dict[str, float]:
    if task_family == "transcription":
        return dict(_DEFAULT_THRESHOLDS_TRANSCRIPTION)
    if task_family in ("vision", "ocr", "presentation"):
        return dict(_DEFAULT_THRESHOLDS_VISION_OCR)
    return dict(_DEFAULT_THRESHOLDS_GENERIC)


def _check_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[str]:
    """Return human-readable violation messages; empty list means all passed."""
    violations: list[str] = []
    for axis, floor in thresholds.items():
        observed = metrics.get(axis)
        if observed is None:
            violations.append(f"{axis}: missing in receipt")
            continue
        if observed < floor:
            violations.append(f"{axis}: {observed:.3f} < threshold {floor:.3f}")
    return violations


# ---------------------------------------------------------------------------
# YAML mutation helpers
# ---------------------------------------------------------------------------


def _ensure_approved_local_in_allowed_selections(
    yaml_text: str,
    *,
    task_id: str,
    model_id: str,
) -> str:
    """Add ``approved_local`` entry to allowed_selections if not already present.

    Operates on raw YAML text to preserve comments. If the entry already exists,
    returns the text unchanged.
    """
    # Quick check: if the exact approved_local entry is already in the text
    # for this model, do nothing.
    marker = f"model: {model_id}"
    for line in yaml_text.splitlines():
        if marker in line and "approved_local" in line:
            return yaml_text
        if "approved_local" in line and model_id in yaml_text:
            # More careful check: find the block for this task and look for
            # the approved_local entry there
            pass

    lines = yaml_text.splitlines()
    task_header = re.compile(rf"^(?P<indent>\s*){re.escape(task_id)}:\s*$")
    allowed_selections_re = re.compile(r"^(?P<indent>\s+)allowed_selections:\s*$")

    task_indent: int | None = None
    task_start_idx: int | None = None
    for idx, line in enumerate(lines):
        m = task_header.match(line)
        if m:
            task_indent = len(m.group("indent"))
            task_start_idx = idx
            break

    if task_start_idx is None:
        return yaml_text

    # Find task block end
    task_end_idx = len(lines)
    for idx in range(task_start_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip() or line.strip().startswith("#"):
            continue
        leading = len(line) - len(line.lstrip(" "))
        if task_indent is not None and leading <= task_indent:
            task_end_idx = idx
            break

    # Find allowed_selections within the task block
    as_idx: int | None = None
    as_indent: int | None = None
    for idx in range(task_start_idx + 1, task_end_idx):
        m = allowed_selections_re.match(lines[idx])
        if m:
            as_idx = idx
            as_indent = len(m.group("indent"))
            break

    if as_idx is None:
        return yaml_text

    # Check if approved_local for this model_id already exists in the block
    entry_indent = " " * (as_indent + 2) if as_indent is not None else "    "
    new_entry = (
        f"{entry_indent}"
        f"- {{provider: local, model: {model_id}, role: approved_local}}"
    )
    # Check in the block following allowed_selections
    block_lines = lines[as_idx + 1 : task_end_idx]
    for bl in block_lines:
        stripped = bl.strip()
        if not stripped or stripped.startswith("#"):
            continue
        leading = len(bl) - len(bl.lstrip(" "))
        if as_indent is not None and leading <= as_indent:
            break
        if model_id in bl and "approved_local" in bl:
            return yaml_text

    # Find the last line of the allowed_selections block to insert after
    insert_after: int = as_idx
    for idx in range(as_idx + 1, task_end_idx):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        leading = len(line) - len(line.lstrip(" "))
        if as_indent is not None and leading <= as_indent:
            break
        insert_after = idx

    new_lines = lines[:insert_after + 1] + [new_entry] + lines[insert_after + 1:]
    return "\n".join(new_lines) + ("\n" if yaml_text.endswith("\n") else "")


def _upsert_local_proof(
    yaml_text: str,
    *,
    task_id: str,
    research_ref: str,
    hardware_fit_ref: str,
    benchmark_receipt_ref: str,
    operator_approval_ref: str,
    approved_date: str,
    scope: str,
) -> str:
    """Write or replace the local_proof block for task_id.

    If a ``local_proof:`` key already exists, replace its value block in-place.
    Otherwise insert it before ``allowed_selections:``.
    """
    lines = yaml_text.splitlines()
    task_header = re.compile(rf"^(?P<indent>\s*){re.escape(task_id)}:\s*$")

    task_indent: int | None = None
    task_start_idx: int | None = None
    for idx, line in enumerate(lines):
        m = task_header.match(line)
        if m:
            task_indent = len(m.group("indent"))
            task_start_idx = idx
            break

    if task_start_idx is None:
        return yaml_text

    task_end_idx = len(lines)
    for idx in range(task_start_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip() or line.strip().startswith("#"):
            continue
        leading = len(line) - len(line.lstrip(" "))
        if task_indent is not None and leading <= task_indent:
            task_end_idx = idx
            break

    # task_indent is the indent of the task key itself (e.g. 2 for "  task_id:").
    # Body fields are one level deeper (task_indent + 2).
    # local_proof values are two levels deeper (task_indent + 4).
    body_indent = " " * ((task_indent or 0) + 2)
    proof_indent = body_indent
    proof_value_indent = " " * ((task_indent or 0) + 4)

    proof_block_lines = [
        f"{proof_indent}local_proof:",
        f'{proof_value_indent}research_ref: "{research_ref}"',
        f'{proof_value_indent}hardware_fit_ref: "{hardware_fit_ref}"',
        f'{proof_value_indent}benchmark_receipt_ref: "{benchmark_receipt_ref}"',
        f'{proof_value_indent}operator_approval_ref: "{operator_approval_ref}"',
        f'{proof_value_indent}approved_date: "{approved_date}"',
        f'{proof_value_indent}scope: "{scope}"',
    ]

    # Find existing local_proof: within the task block
    local_proof_re = re.compile(r"^(?P<indent>\s+)local_proof:\s*(?:\{\})?\s*$")
    lp_idx: int | None = None
    for idx in range(task_start_idx + 1, task_end_idx):
        if local_proof_re.match(lines[idx]):
            lp_idx = idx
            break

    if lp_idx is not None:
        # Remove the existing local_proof block (header + value sub-lines)
        lp_line_indent = len(lines[lp_idx]) - len(lines[lp_idx].lstrip(" "))
        lp_end = lp_idx + 1
        while lp_end < task_end_idx:
            line = lines[lp_end]
            if not line.strip():
                lp_end += 1
                continue
            leading = len(line) - len(line.lstrip(" "))
            if leading > lp_line_indent:
                lp_end += 1
            else:
                break
        new_lines = lines[:lp_idx] + proof_block_lines + lines[lp_end:]
    else:
        # Insert before allowed_selections or at the end of the task block
        allowed_sel_re = re.compile(r"^\s+allowed_selections:\s*$")
        insert_before: int = task_end_idx
        for idx in range(task_start_idx + 1, task_end_idx):
            if allowed_sel_re.match(lines[idx]):
                insert_before = idx
                break
        new_lines = lines[:insert_before] + proof_block_lines + lines[insert_before:]

    return "\n".join(new_lines) + ("\n" if yaml_text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def promote_local_candidate(
    *,
    task_id: str,
    model_id: str,
    receipt_path: Path | str | None,
    research_ref: str,
    hardware_fit_ref: str,
    operator_approval_ref: str | None,
    registry_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
    dry_run: bool = False,
) -> RegistryDiff:
    """Promote a local candidate to ``approved`` with full operator proof.

    This is the generic promotion path supporting all task families
    (transcription, vision, OCR, etc.).  It:

    1. Validates ``operator_approval_ref`` is present (fails closed if absent).
    2. Loads and threshold-checks the benchmark receipt's ``rubric_scores``.
    3. Flips the candidate's ``status`` to ``approved`` and sets its
       ``benchmark_receipt_ref``.
    4. Ensures ``allowed_selections`` contains an ``approved_local`` entry for
       this model.
    5. Populates ``local_proof`` with all six required fields.

    After a non-dry-run promotion, ``approved_local_selection(task_id)`` will
    return a :class:`ModelSelection` with ``provider='local'`` and
    ``role='approved_local'``.

    Args:
        task_id:               Registry task identifier.
        model_id:              Candidate model ID to promote.
        receipt_path:          Path to a *passing live* benchmark receipt JSON.
        research_ref:          Proof path for source research.
        hardware_fit_ref:      Proof path for hardware fitness assessment.
        operator_approval_ref: Proof path for operator sign-off (required).
        registry_path:         Override the default registry.yaml path.
        thresholds:            Per-metric floor overrides.  Defaults are
                               task-family-aware (transcription: quality≥0.6;
                               vision/OCR: text_recall, parse_success, overall;
                               generic: quality≥0.5).
        dry_run:               When True, return the diff without writing any file.

    Returns:
        :class:`RegistryDiff` describing the mutation applied (or that would be applied).

    Raises:
        PromotionRequiresOperatorApprovalError: ``operator_approval_ref`` is None/empty.
        PromotionRequiresReceiptError:          ``receipt_path`` is None or missing.
        PromotionThresholdsNotMetError:         Receipt rubric_scores below thresholds.
        PromotionCandidateMissingError:         ``model_id`` not in local_candidates.
        PromotionError:                         Any other promotion-time failure.
    """
    # 1. Gate: operator_approval_ref must be present before touching anything.
    if not operator_approval_ref:
        raise PromotionRequiresOperatorApprovalError(
            f"{task_id}/{model_id}: operator_approval_ref is required for promotion; "
            "manual promotion fails closed by design (D-03)"
        )

    # 2. Gate: receipt_path must be provided.
    if receipt_path is None:
        raise PromotionRequiresReceiptError(
            f"{task_id}/{model_id}: cannot promote without a benchmark receipt"
        )
    receipt_path = Path(receipt_path)

    registry_path = Path(registry_path) if registry_path else default_registry_path()

    # Validate task + candidate exist.
    registry = load_registry(registry_path)
    policy = registry.get(task_id)
    candidate = next(
        (c for c in policy.local_candidates if c.model_id == model_id),
        None,
    )
    if candidate is None:
        raise PromotionCandidateMissingError(
            f"{task_id}: candidate {model_id!r} is not registered in local_candidates"
        )

    # 3. Receipt-provenance gate (T-03-03): the receipt must be a live, passing,
    # local benchmark proof BEFORE the rubric threshold check and before any
    # mutation. A dry-run / skipped receipt with high rubric scores must NOT be
    # able to flip the registry — mirrors registry._validate_approved_local_receipt
    # so the promotion gate and the selection-time gate agree.
    if not receipt_path.exists():
        raise PromotionRequiresReceiptError(
            f"{task_id}/{model_id}: benchmark receipt does not exist: {receipt_path}"
        )
    try:
        raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt = receipt_from_dict(raw_receipt)
    except Exception as exc:  # noqa: BLE001 — surface as a typed promotion failure
        raise PromotionError(
            f"{task_id}/{model_id}: benchmark receipt is not a valid BenchmarkReceipt: "
            f"{receipt_path}"
        ) from exc
    try:
        validate_receipt(receipt, task_policy=policy)
    except ValueError as exc:
        raise PromotionError(
            f"{task_id}/{model_id}: benchmark receipt failed schema validation: {exc}"
        ) from exc

    expected_provenance = {
        "execution_mode": "live",
        "selection_decision": "passed",
        "provider": "local",
        "runtime_id": "mlx",
        "task_id": task_id,
        "model_id": model_id,
    }
    provenance_mismatches = [
        f"{key} expected {value!r} got {getattr(receipt, key)!r}"
        for key, value in expected_provenance.items()
        if getattr(receipt, key) != value
    ]
    if provenance_mismatches:
        raise PromotionError(
            f"{task_id}/{model_id}: receipt is not a passing live local proof; "
            "promotion fails closed (T-03-03): "
            + "; ".join(provenance_mismatches)
        )

    # 4. Threshold check.
    effective_thresholds = thresholds or _default_thresholds_for_task_family(policy.task_family)
    metrics = load_receipt_metrics(receipt_path)
    violations = _check_thresholds(metrics, effective_thresholds)
    if violations:
        raise PromotionThresholdsNotMetError(
            f"{task_id}/{model_id}: receipt does not meet thresholds: "
            + "; ".join(violations)
        )

    # 5. Apply mutations to YAML text.
    yaml_text = registry_path.read_text(encoding="utf-8")
    new_text, previous_status = _flip_candidate_status_in_yaml(
        yaml_text,
        task_id=task_id,
        model_id=model_id,
        new_status="approved",
        receipt_path=str(receipt_path),
    )

    if previous_status != "approved":
        # Ensure allowed_selections has an approved_local entry.
        new_text = _ensure_approved_local_in_allowed_selections(
            new_text,
            task_id=task_id,
            model_id=model_id,
        )

    # Populate local_proof with all required proof fields.
    approved_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    scope = f"{task_id}/{model_id} promoted by promote-local-candidate"
    new_text = _upsert_local_proof(
        new_text,
        task_id=task_id,
        research_ref=research_ref,
        hardware_fit_ref=hardware_fit_ref,
        benchmark_receipt_ref=str(receipt_path),
        operator_approval_ref=operator_approval_ref,
        approved_date=approved_date,
        scope=scope,
    )

    if previous_status == "approved":
        # Idempotent: already approved — return diff, skip write.
        return RegistryDiff(
            task_id=task_id,
            model_id=model_id,
            previous_status="approved",
            new_status="approved",
            receipt_path=str(receipt_path),
            thresholds=dict(effective_thresholds),
            receipt_metrics=dict(metrics),
        )

    if not dry_run:
        # Parse-before-replace guard (T-03-02).
        try:
            yaml.safe_load(new_text)
        except yaml.YAMLError as exc:
            raise PromotionError(
                f"post-edit registry YAML is not parseable: {exc}"
            ) from exc
        _atomic_write_text(registry_path, new_text)
        # Clear lru_cache so future loads see the new state.
        load_registry.cache_clear()

    return RegistryDiff(
        task_id=task_id,
        model_id=model_id,
        previous_status=previous_status,
        new_status="approved",
        receipt_path=str(receipt_path),
        thresholds=dict(effective_thresholds),
        receipt_metrics=dict(metrics),
    )
