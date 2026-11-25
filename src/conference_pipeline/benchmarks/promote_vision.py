"""v1.61 Phase 69 — promote-vision-candidate: registry status flip gated on receipts.

A local vision candidate stays at ``status: benchmark_pending`` in
``registry.yaml`` until a passing benchmark receipt exists. This module
provides the only sanctioned path to flip a candidate to ``approved``.

Manual promotion without a receipt fails closed with
:class:`PromotionRequiresReceiptError`. Promotion with a receipt below
configured thresholds fails closed with :class:`PromotionThresholdsNotMetError`.
The registry write is atomic (``tempfile + os.replace``) so a partial
edit cannot leave the file in an invalid state.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from conference_pipeline.model_governance.registry import (
    ModelGovernanceError,
    default_registry_path,
    load_registry,
)


_DEFAULT_THRESHOLDS = {
    "text_recall": 0.6,
    "figure_caption_similarity": 0.4,
    "chart_summary_similarity": 0.4,
    "parse_success": 1.0,
    "overall": 0.6,
}


class PromotionError(ModelGovernanceError):
    """Base class for promotion failures."""


class PromotionRequiresReceiptError(PromotionError):
    """Raised when promotion is attempted without a receipt path."""


class PromotionThresholdsNotMetError(PromotionError):
    """Raised when the receipt fails one or more configured thresholds."""


class PromotionCandidateMissingError(PromotionError):
    """Raised when the requested model_id is not in local_candidates for the task."""


@dataclass(frozen=True)
class RegistryDiff:
    """Summary of the registry mutation applied (or that would be applied in dry-run)."""

    task_id: str
    model_id: str
    previous_status: str
    new_status: str
    receipt_path: str
    thresholds: dict[str, float]
    receipt_metrics: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "receipt_path": self.receipt_path,
            "thresholds": dict(self.thresholds),
            "receipt_metrics": dict(self.receipt_metrics),
        }


def load_thresholds(thresholds_path: Path | None = None) -> dict[str, float]:
    """Return the threshold map; falls back to ``_DEFAULT_THRESHOLDS``."""
    if thresholds_path is None:
        return dict(_DEFAULT_THRESHOLDS)
    payload = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PromotionError(
            f"thresholds file must be a JSON object: {thresholds_path}"
        )
    merged = dict(_DEFAULT_THRESHOLDS)
    for key, value in payload.items():
        merged[str(key)] = float(value)
    return merged


def load_receipt_metrics(receipt_path: Path) -> dict[str, float]:
    """Return the per-metric scores recorded in a vision-benchmark receipt.

    Receipts are expected to be JSON with a ``rubric_scores`` block (the same
    convention as the v1.6 benchmark receipts) carrying numeric metrics.
    Missing keys fall through to None and surface as threshold failures.
    """
    if not receipt_path.exists():
        raise PromotionRequiresReceiptError(
            f"benchmark receipt does not exist: {receipt_path}"
        )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PromotionError(
            f"benchmark receipt must be a JSON object: {receipt_path}"
        )
    rubric = payload.get("rubric_scores") or {}
    if not isinstance(rubric, dict):
        raise PromotionError(
            f"benchmark receipt rubric_scores is not an object: {receipt_path}"
        )
    metrics: dict[str, float] = {}
    for key, value in rubric.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if "overall" not in metrics and metrics:
        # Fall back to mean of the four named axes when overall is absent.
        named = [
            metrics[k]
            for k in ("text_recall", "figure_caption_similarity",
                      "chart_summary_similarity", "parse_success")
            if k in metrics
        ]
        if named:
            metrics["overall"] = sum(named) / len(named)
    return metrics


def check_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[str]:
    """Return a list of human-readable threshold-violation messages.

    An empty list means all thresholds passed.
    """
    violations: list[str] = []
    for axis, floor in thresholds.items():
        observed = metrics.get(axis)
        if observed is None:
            violations.append(f"{axis}: missing in receipt")
            continue
        if observed < floor:
            violations.append(
                f"{axis}: {observed:.3f} < threshold {floor:.3f}"
            )
    return violations


def _atomic_write_text(path: Path, body: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".registry-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _flip_candidate_status_in_yaml(
    yaml_text: str,
    *,
    task_id: str,
    model_id: str,
    new_status: str,
    receipt_path: str,
) -> tuple[str, str]:
    """Flip a candidate's ``status`` field while preserving comments/formatting.

    Returns ``(new_yaml_text, previous_status)``. Operates on the YAML *text*
    rather than parsing/re-emitting via PyYAML so that comments — which the
    registry uses heavily for default-pick guidance — survive the rewrite.

    Raises :class:`PromotionCandidateMissingError` when the candidate is not
    found inside the named task block.
    """
    lines = yaml_text.splitlines()

    task_header = re.compile(rf"^(?P<indent>\s*){re.escape(task_id)}:\s*$")
    candidate_marker = re.compile(r"^(?P<indent>\s+)-\s+")
    # `model_id` may be the first key after `- ` on the same line, OR on a
    # following line at the candidate body indent. Match both shapes.
    model_id_pat = re.compile(
        rf'^\s+(?:-\s+)?model_id:\s*[\'\"]?{re.escape(model_id)}[\'\"]?\s*$'
    )
    status_re = re.compile(r"^(?P<lead>\s+status:\s*)(?P<value>\S+.*?)\s*$")
    receipt_re = re.compile(r"^(?P<lead>\s+benchmark_receipt_ref:\s*)(?P<value>.+?)\s*$")
    local_candidates_re = re.compile(r"^(?P<indent>\s+)local_candidates:\s*$")

    # Locate the task header.
    task_indent = None
    task_start_idx = None
    for idx, line in enumerate(lines):
        m = task_header.match(line)
        if m:
            task_indent = len(m.group("indent"))
            task_start_idx = idx
            break
    if task_start_idx is None:
        raise PromotionCandidateMissingError(
            f"task {task_id!r} not found in registry"
        )

    # Find the boundary of the task block: first non-comment line indented
    # at or below task_indent that starts a new task entry.
    task_end_idx = len(lines)
    for idx in range(task_start_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip() or line.strip().startswith("#"):
            continue
        leading = len(line) - len(line.lstrip(" "))
        if leading <= task_indent:
            task_end_idx = idx
            break

    # Locate local_candidates: within the task block.
    lc_idx = None
    lc_indent = None
    for idx in range(task_start_idx + 1, task_end_idx):
        m = local_candidates_re.match(lines[idx])
        if m:
            lc_idx = idx
            lc_indent = len(m.group("indent"))
            break
    if lc_idx is None:
        raise PromotionCandidateMissingError(
            f"{task_id}: local_candidates: block not found"
        )

    # Each candidate begins with a line whose first non-space character is
    # `-` and whose indent is greater than the local_candidates: indent.
    candidate_starts: list[int] = []
    for idx in range(lc_idx + 1, task_end_idx):
        line = lines[idx]
        m = candidate_marker.match(line)
        if not m:
            continue
        cand_indent = len(m.group("indent"))
        if cand_indent <= lc_indent:
            # Out of the local_candidates list.
            break
        candidate_starts.append(idx)
    candidate_starts.append(task_end_idx)  # sentinel for slicing the last block

    # Walk each candidate block and find the one matching model_id.
    new_lines = list(lines)
    previous_status: str | None = None
    found = False
    for n in range(len(candidate_starts) - 1):
        block_start = candidate_starts[n]
        block_end = candidate_starts[n + 1]
        block_indices = list(range(block_start, block_end))

        # Does this block carry our target model_id?
        if not any(model_id_pat.match(lines[bi]) for bi in block_indices):
            continue

        # Apply status flip + receipt rewrite within the matched block.
        applied_status = False
        applied_receipt = False
        for bi in block_indices:
            line_text = new_lines[bi]
            s_match = status_re.match(line_text)
            if s_match and not applied_status:
                previous_status = s_match.group("value").strip().strip('"').strip("'")
                new_lines[bi] = f"{s_match.group('lead')}{new_status}"
                applied_status = True
                continue
            r_match = receipt_re.match(line_text)
            if r_match and not applied_receipt:
                new_lines[bi] = (
                    f'{r_match.group("lead")}"{receipt_path}"'
                )
                applied_receipt = True
        if not applied_status:
            raise PromotionError(
                f"{task_id}: candidate {model_id!r} has no `status:` field"
            )
        if not applied_receipt:
            # Insert benchmark_receipt_ref just after the status line.
            cand_marker_match = candidate_marker.match(lines[block_start])
            assert cand_marker_match is not None
            body_indent = " " * (len(cand_marker_match.group("indent")) + 2)
            insert_after = None
            for bi in block_indices:
                if status_re.match(new_lines[bi]):
                    insert_after = bi
                    break
            if insert_after is not None:
                new_lines.insert(
                    insert_after + 1,
                    f'{body_indent}benchmark_receipt_ref: "{receipt_path}"',
                )
        found = True
        break

    if not found:
        raise PromotionCandidateMissingError(
            f"{task_id}: candidate {model_id!r} not found in local_candidates"
        )
    if previous_status is None:
        raise PromotionError(
            f"{task_id}: candidate {model_id!r} previous status not captured"
        )
    return "\n".join(new_lines) + ("\n" if yaml_text.endswith("\n") else ""), previous_status


def promote_vision_candidate(
    *,
    task_id: str,
    model_id: str,
    receipt_path: Path | None,
    registry_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
    dry_run: bool = False,
) -> RegistryDiff:
    """Promote a candidate to ``approved`` if its receipt clears thresholds.

    Args:
        task_id: One of ``vision.slide_analysis``, ``vision.figure_caption``,
            ``vision.chart_summary``, ``presentation.visual_qc``,
            ``ocr.slide_image_parse``.
        model_id: The candidate ``model_id`` to flip.
        receipt_path: Path to the benchmark receipt JSON. ``None`` fails
            closed with :class:`PromotionRequiresReceiptError`.
        registry_path: Override the default ``registry.yaml`` path.
        thresholds: Override the default per-metric floors.
        dry_run: When True, return the diff without writing the file.

    Returns: :class:`RegistryDiff` describing the mutation.
    """
    if receipt_path is None:
        raise PromotionRequiresReceiptError(
            f"{task_id}/{model_id}: cannot promote without a benchmark receipt; "
            "manual promotion fails closed by design"
        )
    receipt_path = Path(receipt_path)
    registry_path = Path(registry_path) if registry_path else default_registry_path()
    thresholds = thresholds or dict(_DEFAULT_THRESHOLDS)

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

    metrics = load_receipt_metrics(receipt_path)
    violations = check_thresholds(metrics, thresholds)
    if violations:
        raise PromotionThresholdsNotMetError(
            f"{task_id}/{model_id}: receipt does not meet thresholds: "
            + "; ".join(violations)
        )

    yaml_text = registry_path.read_text(encoding="utf-8")
    new_text, previous_status = _flip_candidate_status_in_yaml(
        yaml_text,
        task_id=task_id,
        model_id=model_id,
        new_status="approved",
        receipt_path=str(receipt_path),
    )
    if previous_status == "approved":
        # Already approved; this is idempotent — return a diff but skip write.
        return RegistryDiff(
            task_id=task_id,
            model_id=model_id,
            previous_status="approved",
            new_status="approved",
            receipt_path=str(receipt_path),
            thresholds=dict(thresholds),
            receipt_metrics=dict(metrics),
        )
    if not dry_run:
        # Re-parse to ensure the YAML is still valid before writing.
        try:
            yaml.safe_load(new_text)
        except yaml.YAMLError as exc:
            raise PromotionError(
                f"post-edit registry YAML is not parseable: {exc}"
            )
        _atomic_write_text(registry_path, new_text)
        # Bust the lru_cache so future loads see the new status.
        load_registry.cache_clear()
    return RegistryDiff(
        task_id=task_id,
        model_id=model_id,
        previous_status=previous_status,
        new_status="approved",
        receipt_path=str(receipt_path),
        thresholds=dict(thresholds),
        receipt_metrics=dict(metrics),
    )
