"""Reusable stage-owner inventory and planning-registry checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.registry_diagnostics import attach_registry_tool_diagnostic
from conference_pipeline.stage_script_registry import (
    DEFAULT_STAGE_SCRIPT_REGISTRY_PATH,
    legacy_owner_registry_view,
)

DEFAULT_STAGE_OWNER_REGISTRY_PATH = DEFAULT_STAGE_SCRIPT_REGISTRY_PATH
PLAN_AUDIT_FILE_SUFFIXES = frozenset({".md", ".json", ".yaml", ".yml"})
PLAN_AUDIT_NEGATIVE_POLICY_TERMS = (
    "reject",
    "rejected",
    "forbidden",
    "banned",
    "must not",
    "do not",
    "cannot",
    "not use",
    "not approved",
    "deprecated",
    "legacy",
)

STAGE_OWNER_CATEGORIES = frozenset(
    {
        "conference_adapter",
        "reusable_owner",
        "claim_aggregator",
        "legacy_scaffold",
    }
)


@dataclass(frozen=True)
class StageOwnerRecord:
    """Production ownership classification for one conference runner stage."""

    stage_id: str
    category: str
    owner_refs: tuple[str, ...]
    production_output: bool = False
    explicit_blocker: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "category": self.category,
            "owner_refs": list(self.owner_refs),
            "production_output": self.production_output,
            "explicit_blocker": self.explicit_blocker,
            "notes": self.notes,
        }


def audit_stage_owner_inventory(
    records: Iterable[StageOwnerRecord | Mapping[str, Any]],
    *,
    required_owner_refs: Mapping[str, Iterable[str]] | None = None,
    owner_registry: Mapping[str, Any] | None = None,
    registry_task_ids_by_stage: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, object]:
    """Audit stage owner records and fail closed on scaffold production owners."""

    normalized = [_coerce_record(record) for record in records]
    blockers: list[dict[str, object]] = []
    by_stage = {record.stage_id: record for record in normalized}
    registry = dict(owner_registry or load_stage_owner_registry())

    for record in normalized:
        if record.category not in STAGE_OWNER_CATEGORIES:
            blockers.append(
                {
                    "code": "stage_owner_category_invalid",
                    "stage_id": record.stage_id,
                    "observed_category": record.category,
                    "allowed_categories": sorted(STAGE_OWNER_CATEGORIES),
                    "retryable": False,
                }
            )
        if (
            record.production_output
            and record.category == "legacy_scaffold"
            and not record.explicit_blocker
        ):
            blockers.append(
                {
                    "code": "production_output_uses_legacy_scaffold",
                    "stage_id": record.stage_id,
                    "owner_refs": list(record.owner_refs),
                    "retryable": False,
                }
            )

    merged_required_owner_refs: dict[str, list[str]] = {
        stage_id: [str(item) for item in refs]
        for stage_id, refs in (required_owner_refs or {}).items()
    }
    for stage_id, task_ids in (registry_task_ids_by_stage or {}).items():
        merged_required_owner_refs.setdefault(stage_id, []).extend(
            required_owner_refs_for_tasks(registry, task_ids)
        )

    for stage_id, required_refs in merged_required_owner_refs.items():
        record = by_stage.get(stage_id)
        missing: list[str] = []
        for required_ref in required_refs:
            if not _owner_ref_present(required_ref, record.owner_refs if record else ()):
                missing.append(required_ref)
        if missing:
            blockers.append(
                {
                    "code": "stage_owner_required_ref_missing",
                    "stage_id": stage_id,
                    "missing_owner_refs": missing,
                    "retryable": False,
                }
            )

    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "records": [record.to_dict() for record in normalized],
    }
    return attach_registry_tool_diagnostic(
        report,
        script_id="verification.stage_owner_audit",
        checked_inputs={
            "record_count": len(normalized),
            "required_owner_stage_count": len(merged_required_owner_refs),
        },
        accepted_evidence=[
            "Stage owner records were classified.",
            "Required owner refs were compared against runner stage records.",
        ],
        pass_next_actions=[
            "Stage owner inventory is compatible with the registry requirements."
        ],
        blocked_next_actions=[
            "Repair the cited runner stage owner refs before accepting the stage inventory."
        ],
    )


def load_stage_owner_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load the approved owner-command registry compatibility view."""

    return legacy_owner_registry_view(path)


def required_owner_refs_for_tasks(
    registry: Mapping[str, Any],
    task_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return approved owner refs for the requested registry task ids."""

    refs: list[str] = []
    for task_id in task_ids:
        task = _registry_task(registry, str(task_id))
        for owner_ref in task.get("approved_owner_refs") or ():
            owner_ref_str = str(owner_ref)
            if owner_ref_str and owner_ref_str not in refs:
                refs.append(owner_ref_str)
    return tuple(refs)


def audit_plan_owner_registry(
    plan_root: str | Path,
    *,
    task_ids: Iterable[str] | None = None,
    owner_registry: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, object]:
    """Audit generated planning docs against the approved owner registry."""

    root = Path(plan_root)
    registry = dict(owner_registry or load_stage_owner_registry(registry_path))
    selected_task_ids = tuple(
        str(task_id)
        for task_id in (
            task_ids
            if task_ids is not None
            else registry.get("planning_required_task_ids") or sorted(registry["tasks"])
        )
    )
    text = _collect_plan_text(root)
    blockers: list[dict[str, object]] = []
    rejected_owner_refs: list[str] = []

    for task_id in selected_task_ids:
        try:
            task = _registry_task(registry, task_id)
        except KeyError:
            blockers.append(
                {
                    "code": "plan_owner_registry_task_missing",
                    "task_id": task_id,
                    "retryable": False,
                }
            )
            continue

        missing_owner_refs = [
            str(owner_ref)
            for owner_ref in task.get("approved_owner_refs") or ()
            if str(owner_ref) not in text
        ]
        if missing_owner_refs:
            blockers.append(
                {
                    "code": "plan_owner_registry_approved_owner_missing",
                    "task_id": task_id,
                    "missing_owner_refs": missing_owner_refs,
                    "retryable": False,
                }
            )

        missing_terms = [
            str(term)
            for term in task.get("required_command_terms") or ()
            if str(term) not in text
        ]
        if missing_terms:
            blockers.append(
                {
                    "code": "plan_owner_registry_required_term_missing",
                    "task_id": task_id,
                    "missing_terms": missing_terms,
                    "retryable": False,
                }
            )

        for rejected_ref in task.get("rejected_owner_refs") or ():
            rejected_ref_str = str(rejected_ref)
            if rejected_ref_str and rejected_ref_str not in rejected_owner_refs:
                rejected_owner_refs.append(rejected_ref_str)

    rejected_hits = _rejected_owner_hits(root, rejected_owner_refs)
    if rejected_hits:
        blockers.append(
            {
                "code": "plan_owner_registry_rejected_owner_used",
                "hits": rejected_hits,
                "retryable": False,
            }
        )

    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "registry_path": str(registry_path or DEFAULT_STAGE_OWNER_REGISTRY_PATH),
        "checked_task_ids": list(selected_task_ids),
    }
    return attach_registry_tool_diagnostic(
        report,
        script_id="verification.stage_owner_audit",
        registry_path=registry_path,
        checked_inputs={
            "plan_root": str(plan_root),
            "registry_path": report["registry_path"],
            "checked_task_ids": list(selected_task_ids),
        },
        accepted_evidence=[
            "Generated plan text was scanned for approved owner refs.",
            "Generated plan text was scanned for required command terms.",
            "Rejected owner refs were checked for improper completion usage.",
        ],
        pass_next_actions=[
            "Compatibility owner audit passed; continue to unified stage script verification."
        ],
        blocked_next_actions=[
            "Update the generated plan to use approved owners, then rerun the owner audit."
        ],
    )


def _coerce_record(record: StageOwnerRecord | Mapping[str, Any]) -> StageOwnerRecord:
    if isinstance(record, StageOwnerRecord):
        return record
    return StageOwnerRecord(
        stage_id=str(record.get("stage_id") or ""),
        category=str(record.get("category") or ""),
        owner_refs=tuple(str(item) for item in record.get("owner_refs") or ()),
        production_output=bool(record.get("production_output")),
        explicit_blocker=(
            str(record["explicit_blocker"]) if record.get("explicit_blocker") else None
        ),
        notes=str(record.get("notes") or ""),
    )


def _owner_ref_present(required_ref: str, owner_refs: Iterable[str]) -> bool:
    return any(required_ref == owner_ref or required_ref in owner_ref for owner_ref in owner_refs)


def _registry_task(registry: Mapping[str, Any], task_id: str) -> Mapping[str, Any]:
    tasks = registry.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("Stage owner registry missing tasks mapping")
    task = tasks[task_id]
    if not isinstance(task, Mapping):
        raise ValueError(f"Malformed stage owner registry task: {task_id}")
    return task


def _collect_plan_text(root: Path) -> str:
    if root.is_file():
        return root.read_text(encoding="utf-8")
    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in PLAN_AUDIT_FILE_SUFFIXES:
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _rejected_owner_hits(root: Path, rejected_owner_refs: Iterable[str]) -> list[str]:
    """Rejected-owner mentions that are not inside their own negative-policy unit.

    The unit is a SENTENCE or a LIST ITEM, not a physical line. A wrapped
    sentence ("...rejects citation-invalid heuristic\nreports.") carries its
    negative term on its first line; a JSON array under a key named
    ``rejected_..._commands`` carries it in the key. Both are the same policy
    statement whichever line the rejected name lands on. What a unit still
    cannot do is inherit rejection from a heading above it: a bare bullet
    "- `script.py`" under "Reject these:" is judged on its own text, so a
    later executable command in a policy list is caught, as CR-03 intended.
    """

    hits: list[str] = []
    refs = tuple(ref for ref in rejected_owner_refs if ref)
    if not refs:
        return hits
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if not path.is_file() or path.suffix not in PLAN_AUDIT_FILE_SUFFIXES:
            continue
        file_text = path.read_text(encoding="utf-8")
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        if path.suffix == ".json":
            units = _json_policy_units(file_text)
        else:
            units = _text_policy_units(file_text)
        for unit_text, negative in units:
            if not any(ref in unit_text for ref in refs):
                continue
            if negative or _line_is_negative_owner_policy(unit_text.lower()):
                continue
            hits.append(f"{relative}: {' '.join(unit_text.split())[:200]}")
    return hits


# A line starting with an inline-code token (`audio_ready_package.json`
# emitters, ...) is wrapped prose, not a command: commands announce themselves
# with a bullet, "python", ".venv/", "PYTHONPATH=" or a bare "scripts/" path.
_UNIT_BREAK_RE = re.compile(r"^\s*(?:#|[-*+]\s|\d+[.)]\s|python\b|python3\b|\.venv/|PYTHONPATH=|scripts/)")


def _text_policy_units(
    text: str, *, is_metadata_line: Callable[[str], bool] | None = None
) -> list[tuple[str, bool]]:
    """Split markdown/yaml into policy units.

    A unit is a physical line, EXCEPT that a line which merely continues the
    previous line's prose (the previous line did not end a sentence, and this
    line is not a heading, bullet, or command) is joined to it. So a wrapped
    sentence is one unit, while a heading, a bullet, or an executable command
    on its own line is always judged by its own text -- a "deprecated" heading
    above a live command still cannot bless it. Registry-metadata lines are
    dropped as before.
    """

    is_metadata_line = is_metadata_line or _line_is_registry_metadata_line
    units: list[tuple[str, bool]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            units.append((" ".join(line.strip() for line in current), False))
            current.clear()

    for line in text.splitlines():
        if is_metadata_line(line.lower()):
            continue
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        previous_open = bool(current) and not current[-1].rstrip().endswith((".", "!", "?", ":"))
        starts_unit = bool(_UNIT_BREAK_RE.match(line))
        if current and previous_open and not starts_unit:
            current.append(line)
        else:
            flush()
            current.append(line)
    flush()
    return units


def _json_policy_units(
    text: str, *, is_metadata_line: Callable[[str], bool] | None = None
) -> list[tuple[str, bool]]:
    """Every string value in a JSON document, with whether any enclosing key
    is itself a negative-policy statement (``rejected_owner_commands``)."""

    is_metadata_line = is_metadata_line or _line_is_registry_metadata_line
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _text_policy_units(text, is_metadata_line=is_metadata_line)
    units: list[tuple[str, bool]] = []

    def walk(node: object, negative: bool) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                # Same exemption the line scanner applies: a registry entry's
                # own path/script_id fields describe a tool, they do not use it.
                if is_metadata_line(f'"{str(key).lower()}"'):
                    continue
                key_negative = negative or _line_is_negative_owner_policy(str(key).lower())
                walk(value, key_negative)
        elif isinstance(node, list):
            for value in node:
                walk(value, negative)
        elif isinstance(node, str):
            units.append((node, negative))

    walk(payload, False)
    return units


def _line_is_negative_owner_policy(lower_line: str) -> bool:
    return any(term in lower_line for term in PLAN_AUDIT_NEGATIVE_POLICY_TERMS)


def _line_is_registry_metadata_line(lower_line: str) -> bool:
    metadata_keys = (
        '"path"',
        '"refs"',
        '"usage_doc_refs"',
        '"script_id"',
        '"status"',
        '"scope"',
        '"stage_bindings"',
        '"task_bindings"',
    )
    return any(key in lower_line for key in metadata_keys)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Registry task id to require. Repeat for multiple tasks.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_plan_owner_registry(
        args.plan_root,
        task_ids=args.task_id,
        registry_path=args.registry_path,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
