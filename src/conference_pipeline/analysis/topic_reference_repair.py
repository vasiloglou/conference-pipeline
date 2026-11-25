"""LLM-assisted repair for topic reconciliation reference bindings.

The validator in :mod:`conference_pipeline.analysis.topic_reference_validation`
must remain deterministic: it decides whether a reconciled brief cites real
paper/transcript ids and whether titles are attached to the right ids. This
module owns the optional repair loop after validation blocks. A local model may
propose a repaired brief, but the deterministic validator remains the judge
before the repaired text can replace the primary reconciliation artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from conference_pipeline.analysis.topic_reconciliation import (
    DEFAULT_INPUT_TOKEN_CEILING,
    DEFAULT_MAX_OUTPUT_TOKENS,
    collect_reconciliation_input,
    estimate_tokens,
)
from conference_pipeline.analysis.topic_reference_validation import (
    OWNER_TOPIC_REFERENCE_VALIDATION,
    ReferenceEntry,
    ReferenceValidationResult,
    SOURCE_ID_TOKEN_RE,
    TopicReferenceLedger,
    TopicReferenceValidationError,
    build_reference_ledger,
    format_reference_ledger_prompt_section,
    render_validation_markdown,
    sha256_text,
    validate_reconciliation_references,
    workspace_relpath,
)
from conference_pipeline.receipt_lineage import refresh_substep_outputs
from conference_pipeline.research.deep_dive_lanes import DEFAULT_QWEN36_MODEL
from conference_pipeline.research.local_researcher import LocalLLMResearcher
from conference_pipeline.workflow_runs import write_json_atomic

OWNER_TOPIC_REFERENCE_REPAIR = "analysis.topic_reference_repair"
RECEIPT_TYPE_TOPIC_REFERENCE_REPAIR = "conference_pipeline.topic_reference_repair"
SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "iclr2026-live-production"


class TopicReferenceRepairError(RuntimeError):
    """Raised when reference repair cannot safely continue."""


class AsyncResearcherLike(Protocol):
    provider_receipt_paths: list[str]
    model: str

    async def run(self, query: str, *, context_path: Path | None = None) -> str: ...

    def usage(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RepairInput:
    topic_id: str
    title: str
    original_markdown_path: Path
    original_markdown: str
    initial_validation: ReferenceValidationResult
    ledger: TopicReferenceLedger
    prompt: str
    input_digest_sha256: str
    input_token_estimate: int


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicReferenceRepairError(f"json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise TopicReferenceRepairError(f"json_not_object:{path}")
    return payload


def _reconciliation_markdown_path(workspace: Path, topic_id: str) -> Path:
    return workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md"


def _repair_paths(workspace: Path, topic_id: str) -> tuple[Path, Path, Path, Path]:
    root = workspace / "analysis" / "topic_reference_repairs"
    return (
        root / f"{topic_id}.md",
        root / f"{topic_id}.json",
        root / f"{topic_id}__validation.json",
        root / f"{topic_id}__validation.md",
    )


def _receipt_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / run_id / "07F-topic-reference-repair.json"


def _topic_ids_from_source_packs(workspace: Path) -> list[str]:
    root = workspace / "analysis" / "narrative_source_packs"
    return [path.stem for path in sorted(root.glob("*.json"))]


def _line_at(text: str, lineno: int | None) -> str:
    if lineno is None or lineno <= 0:
        return ""
    lines = text.splitlines()
    if lineno > len(lines):
        return ""
    return lines[lineno - 1]


def _entry_row(entry: ReferenceEntry) -> dict[str, Any]:
    return {
        "canonical_id": entry.canonical_id,
        "source_family": entry.source_family,
        "titles_or_labels": sorted(entry.titles | entry.source_labels)[:6],
        "source_paths": sorted(entry.source_paths)[:4],
        "metadata": entry.metadata,
    }


def _candidate_entries(
    *,
    ledger: TopicReferenceLedger,
    validation: ReferenceValidationResult,
) -> list[dict[str, Any]]:
    candidates: dict[str, ReferenceEntry] = {}
    canonical_ids = sorted(ledger.entries)
    bare_ids = {entry.bare_id: entry.canonical_id for entry in ledger.entries.values()}

    def add(reference_id: str | None) -> None:
        if reference_id and reference_id in ledger.entries:
            candidates[reference_id] = ledger.entries[reference_id]

    for reference_id in validation.unknown_reference_ids:
        if reference_id.startswith("segment:"):
            suffix = f":{reference_id}"
            for candidate_id, entry in ledger.entries.items():
                if candidate_id.endswith(suffix):
                    candidates[candidate_id] = entry
        elif ":" in reference_id:
            prefix, bare = reference_id.split(":", 1)
            for bare_id, candidate_id in bare_ids.items():
                if bare_id.lower() == bare.lower():
                    candidates[candidate_id] = ledger.entries[candidate_id]
            close = difflib.get_close_matches(
                f"{prefix}:{bare}",
                canonical_ids,
                n=8,
                cutoff=0.76,
            )
            for candidate_id in close:
                candidates[candidate_id] = ledger.entries[candidate_id]

    for issue in validation.bare_reference_ids:
        if isinstance(issue, Mapping):
            add(str(issue.get("canonical_id") or ""))

    for issue in validation.title_mismatch_errors:
        if not isinstance(issue, Mapping):
            continue
        add(str(issue.get("canonical_id") or ""))
        foreign_aliases = issue.get("foreign_title_aliases")
        if isinstance(foreign_aliases, Mapping):
            for ids in foreign_aliases.values():
                if isinstance(ids, list):
                    for candidate_id in ids:
                        if isinstance(candidate_id, str):
                            add(candidate_id)

    return [_entry_row(entry) for entry in sorted(candidates.values(), key=lambda item: item.canonical_id)]


def _validation_issue_payload(
    *,
    validation: ReferenceValidationResult,
    text: str,
) -> dict[str, Any]:
    return {
        "status": validation.status,
        "unknown_reference_ids": list(validation.unknown_reference_ids),
        "bare_reference_ids": list(validation.bare_reference_ids),
        "title_mismatch_errors": list(validation.title_mismatch_errors),
        "blockers": list(validation.blockers),
        "error_lines": [
            {
                "line": issue.get("line"),
                "canonical_id": issue.get("canonical_id"),
                "line_text": issue.get("line_text") or _line_at(text, issue.get("line")),
                "foreign_title_aliases": issue.get("foreign_title_aliases"),
            }
            for issue in validation.title_mismatch_errors
            if isinstance(issue, Mapping)
        ]
        + [
            {
                "line": issue.get("line"),
                "canonical_id": issue.get("canonical_id"),
                "line_text": _line_at(text, issue.get("line")),
                "bare_id": issue.get("bare_id"),
            }
            for issue in validation.bare_reference_ids
            if isinstance(issue, Mapping)
        ],
    }


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return stripped + ("\n" if stripped else "")


def _replace_unknown_reference(
    *,
    line: str,
    unknown_id: str,
    ledger: TopicReferenceLedger,
) -> tuple[str, bool]:
    replacement: str | None = None
    if unknown_id.startswith("segment:"):
        full_id_matches = re.findall(
            r"speech_transcript:[A-Za-z0-9._-]+:segment:\d{1,8}",
            line,
        )
        for full_id in full_id_matches:
            prefix = full_id.rsplit(":segment:", 1)[0]
            candidate = f"{prefix}:{unknown_id}"
            if candidate in ledger.entries:
                replacement = candidate
                break
        if replacement is None:
            candidates = [
                reference_id
                for reference_id in ledger.entries
                if reference_id.endswith(f":{unknown_id}")
            ]
            if len(candidates) == 1:
                replacement = candidates[0]
    elif ":" in unknown_id:
        prefix, bare_id = unknown_id.split(":", 1)
        exact_case_matches = [
            entry.canonical_id
            for entry in ledger.entries.values()
            if entry.canonical_id.startswith(f"{prefix}:")
            and entry.bare_id.lower() == bare_id.lower()
        ]
        if len(exact_case_matches) == 1:
            replacement = exact_case_matches[0]
        else:
            close = difflib.get_close_matches(
                unknown_id,
                sorted(ledger.entries),
                n=1,
                cutoff=0.88,
            )
            if close:
                replacement = close[0]
    if replacement is None:
        return _drop_unresolved_reference_id(line=line, reference_id=unknown_id)
    patterns = [
        (f"`{unknown_id}`", f"`{replacement}`"),
        (unknown_id, replacement),
    ]
    updated = line
    changed = False
    backtick_before, backtick_after = patterns[0]
    if backtick_before in updated:
        updated = updated.replace(backtick_before, backtick_after)
        changed = True
    else:
        before, after = patterns[1]
        if before in updated:
            updated = updated.replace(before, after)
            changed = True
    return updated, changed


def _drop_unresolved_reference_id(*, line: str, reference_id: str) -> tuple[str, bool]:
    if re.fullmatch(r"[A-Z]:[A-Z]", reference_id) and reference_id in line:
        replacement = reference_id.replace(":", "-to-")
        return line.replace(reference_id, replacement), True
    updated = line
    patterns = [
        rf"`{re.escape(reference_id)}`\s*(?:[,;/]\s*)?",
        rf"(?:[,;/]\s*)?`{re.escape(reference_id)}`",
        rf"(?<![A-Za-z0-9._:-]){re.escape(reference_id)}(?![A-Za-z0-9._:-])",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", updated)
    updated = re.sub(r"\(\s*(?:[,;/]\s*)*\)", "", updated)
    updated = re.sub(r"\(\s*[,;/]\s*", "(", updated)
    updated = re.sub(r"\s*[,;/]\s*\)", ")", updated)
    updated = re.sub(r"\s{2,}", " ", updated).rstrip()
    return updated, updated != line


def _canonicalize_bare_reference(
    *,
    line: str,
    bare_id: str,
    canonical_id: str,
) -> tuple[str, bool]:
    if canonical_id in line:
        return line, False
    backticked = f"`{bare_id}`"
    if backticked in line:
        return line.replace(backticked, f"`{canonical_id}`"), True
    token_pattern = re.compile(
        rf"(?<![A-Za-z0-9._:-]){re.escape(bare_id)}(?![A-Za-z0-9._:-])"
    )
    if not token_pattern.search(line):
        return line, False
    if ":segment:" in bare_id:
        return token_pattern.sub(canonical_id, line), True
    return _append_missing_reference_ids_to_line(
        line=line,
        missing_reference_ids={canonical_id},
    )


def _strip_unsafe_parenthetical_labels(line: str) -> tuple[str, bool]:
    """Remove local title/speaker labels attached to cited ids on blocked lines.

    The validator blocks when a line cites a canonical id while also naming an
    alias owned by a different ledger row. The safest minimal repair is to keep
    the id and remove the adjacent human label, letting downstream synthesis
    look up titles from the ledger instead of trusting the local model's label.
    """

    pattern = re.compile(
        r"((?:`(?:paper_abstract|speech_transcript):[^`]+`)"
        r"(?:\s*/\s*`(?:paper_abstract|speech_transcript):[^`]+`)*)"
        r"\s+\([^)]{1,120}\)"
    )
    updated = pattern.sub(r"\1", line)
    return updated, updated != line


MAX_MISSING_REFERENCE_IDS_PER_LINE = 5


def _append_missing_reference_ids_to_line(
    *,
    line: str,
    missing_reference_ids: set[str],
) -> tuple[str, bool]:
    existing_ids = {match.group("id") for match in SOURCE_ID_TOKEN_RE.finditer(line)}
    additions = [
        reference_id
        for reference_id in sorted(missing_reference_ids)
        if reference_id not in existing_ids
    ]
    if not additions:
        return line, False
    if len(additions) > MAX_MISSING_REFERENCE_IDS_PER_LINE:
        # A "foreign alias" resolving to more than a handful of ids is itself
        # evidence the alias is non-discriminating (e.g. a generic corpus-family
        # placeholder label shared by many chunks), not that this one line is
        # genuinely missing that many citations. Leave the line -- and its
        # title_mismatch blocker -- untouched rather than flooding it, so the
        # topic correctly stays blocked for a human/LLM to look at instead of a
        # deterministic pass silently reporting false success over it.
        return line, False

    addition_text = ", ".join(f"`{reference_id}`" for reference_id in additions)
    citation_parenthetical = re.compile(
        r"\((?P<body>[^)]*(?:paper_abstract|speech_transcript):[^)]*)\)"
    )
    matches = list(citation_parenthetical.finditer(line))
    if not matches:
        return f"{line.rstrip()} ({addition_text})", True

    match = matches[-1]
    body = match.group("body").rstrip()
    separator = " " if body.endswith((",", ";")) else ", "
    updated_body = f"{body}{separator}{addition_text}" if body else addition_text
    updated = line[: match.start("body")] + updated_body + line[match.end("body") :]
    return updated, updated != line


def _collapse_duplicated_speech_prefix(line: str) -> tuple[str, bool]:
    pattern = re.compile(
        r"speech_transcript:([A-Za-z0-9._-]+):speech_transcript:\1:segment:(\d{1,8})"
    )
    updated = pattern.sub(r"speech_transcript:\1:segment:\2", line)
    return updated, updated != line


def apply_deterministic_reference_safety_fixes(
    *,
    text: str,
    ledger: TopicReferenceLedger,
) -> tuple[str, int]:
    """Apply conservative line-level fixes before final validation.

    This does not make new claims. It only canonicalizes obvious shorthand or
    near-miss ids and removes unsafe adjacent labels on lines that the
    deterministic validator has already rejected.
    """

    current = text
    total_changes = 0
    for _ in range(3):
        validation = validate_reconciliation_references(
            text=current,
            ledger=ledger,
            require_citations=True,
        )
        if validation.status == "succeeded":
            break
        lines = current.splitlines()
        changed = False
        for issue in validation.bare_reference_ids:
            if not isinstance(issue, Mapping):
                continue
            lineno = issue.get("line")
            bare_id = issue.get("bare_id")
            canonical_id = issue.get("canonical_id")
            if not (
                isinstance(lineno, int)
                and 1 <= lineno <= len(lines)
                and isinstance(bare_id, str)
                and isinstance(canonical_id, str)
            ):
                continue
            updated, line_changed = _canonicalize_bare_reference(
                line=lines[lineno - 1],
                bare_id=bare_id,
                canonical_id=canonical_id,
            )
            if line_changed:
                lines[lineno - 1] = updated
                changed = True
                total_changes += 1
        for unknown_id in validation.unknown_reference_ids:
            for index, line in enumerate(lines):
                if unknown_id not in line:
                    continue
                updated, line_changed = _replace_unknown_reference(
                    line=line,
                    unknown_id=unknown_id,
                    ledger=ledger,
                )
                if line_changed:
                    lines[index] = updated
                    changed = True
                    total_changes += 1
        for index, line in enumerate(lines):
            updated, line_changed = _collapse_duplicated_speech_prefix(line)
            if line_changed:
                lines[index] = updated
                changed = True
                total_changes += 1
        mismatch_lines = {
            int(issue["line"])
            for issue in validation.title_mismatch_errors
            if isinstance(issue, Mapping) and isinstance(issue.get("line"), int)
        }
        missing_ids_by_line: dict[int, set[str]] = {}
        for issue in validation.title_mismatch_errors:
            if not isinstance(issue, Mapping) or not isinstance(issue.get("line"), int):
                continue
            foreign_aliases = issue.get("foreign_title_aliases")
            if not isinstance(foreign_aliases, Mapping):
                continue
            missing_ids = missing_ids_by_line.setdefault(int(issue["line"]), set())
            for ids in foreign_aliases.values():
                if not isinstance(ids, list):
                    continue
                for reference_id in ids:
                    if isinstance(reference_id, str) and reference_id in ledger.entries:
                        missing_ids.add(reference_id)
        for lineno, missing_ids in missing_ids_by_line.items():
            if not (1 <= lineno <= len(lines)):
                continue
            updated, line_changed = _append_missing_reference_ids_to_line(
                line=lines[lineno - 1],
                missing_reference_ids=missing_ids,
            )
            if line_changed:
                lines[lineno - 1] = updated
                changed = True
                total_changes += 1
        for lineno in mismatch_lines:
            if not (1 <= lineno <= len(lines)):
                continue
            updated, line_changed = _strip_unsafe_parenthetical_labels(lines[lineno - 1])
            if line_changed:
                lines[lineno - 1] = updated
                changed = True
                total_changes += 1
        current = "\n".join(lines) + ("\n" if current.endswith("\n") else "")
        if not changed:
            break
    return current, total_changes


def build_reference_repair_prompt(
    *,
    repair_input: RepairInput,
    original_reconciliation_prompt: str,
    max_ledger_entries: int = 1000,
) -> str:
    validation_payload = _validation_issue_payload(
        validation=repair_input.initial_validation,
        text=repair_input.original_markdown,
    )
    focused_candidates = _candidate_entries(
        ledger=repair_input.ledger,
        validation=repair_input.initial_validation,
    )
    return (
        "You are repairing citation/reference bindings in one conference topic "
        "reconciliation brief. The deterministic validator has already found "
        "bad references. Repair the markdown so that every cited paper or "
        "transcript id is a canonical id from the ledger and every nearby title "
        "or speaker label belongs to that same ledger row.\n\n"
        "Do not add new claims. Prefer minimal edits: canonicalize ids, replace "
        "wrong ids with correct canonical ids found in the original evidence, or "
        "remove an unsafe parenthetical title/speaker label when the id is valid "
        "but the label is not bound to that id. If no safe id exists, remove the "
        "citation rather than invent one. Return the full repaired markdown only, "
        "with no code fence and no explanation.\n\n"
        f"Topic id: {repair_input.topic_id}\n"
        f"Topic title: {repair_input.title}\n\n"
        "## Validator Errors To Repair\n\n"
        f"{json.dumps(validation_payload, ensure_ascii=True, indent=2, sort_keys=True)}\n\n"
        "## Focused Candidate Ledger Rows\n\n"
        f"{json.dumps(focused_candidates, ensure_ascii=True, indent=2, sort_keys=True)}\n\n"
        f"{format_reference_ledger_prompt_section(repair_input.ledger, max_entries=max_ledger_entries)}\n\n"
        "## Original Evidence Bundle Used For Reconciliation\n\n"
        f"{original_reconciliation_prompt}\n\n"
        "## Markdown To Repair\n\n"
        f"{repair_input.original_markdown}\n"
    )


def collect_repair_input(
    workspace: Path,
    *,
    topic_id: str,
    input_token_ceiling: int = DEFAULT_INPUT_TOKEN_CEILING,
) -> RepairInput:
    workspace = workspace.resolve()
    markdown_path = _reconciliation_markdown_path(workspace, topic_id)
    if not markdown_path.exists():
        raise TopicReferenceRepairError(f"reconciliation_markdown_missing:{topic_id}")
    original_markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    ledger = build_reference_ledger(workspace, topic_id)
    validation = validate_reconciliation_references(
        text=original_markdown,
        ledger=ledger,
        require_citations=True,
    )
    reconciliation_input = collect_reconciliation_input(
        workspace,
        topic_id=topic_id,
        input_token_ceiling=input_token_ceiling,
    )
    prompt = build_reference_repair_prompt(
        repair_input=RepairInput(
            topic_id=topic_id,
            title=ledger.title,
            original_markdown_path=markdown_path,
            original_markdown=original_markdown,
            initial_validation=validation,
            ledger=ledger,
            prompt="",
            input_digest_sha256="",
            input_token_estimate=0,
        ),
        original_reconciliation_prompt=reconciliation_input.prompt,
    )
    token_estimate = estimate_tokens(prompt)
    if token_estimate > input_token_ceiling:
        raise TopicReferenceRepairError(
            f"topic_reference_repair_input_too_large:{topic_id}:{token_estimate}>{input_token_ceiling}"
        )
    digest = hashlib.sha256(
        json.dumps(
            {
                "original_markdown_sha256": sha256_text(original_markdown),
                "ledger_digest": ledger.input_digest_sha256,
                "reconciliation_input_digest": reconciliation_input.input_digest_sha256,
                "initial_validation": validation.to_dict(),
                "prompt_sha256": sha256_text(prompt),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return RepairInput(
        topic_id=topic_id,
        title=ledger.title,
        original_markdown_path=markdown_path,
        original_markdown=original_markdown,
        initial_validation=validation,
        ledger=ledger,
        prompt=prompt,
        input_digest_sha256=digest,
        input_token_estimate=token_estimate,
    )


def _enrich_source_pack(
    *,
    workspace: Path,
    topic_id: str,
    repair_summary: dict[str, Any],
) -> bool:
    source_pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    if not source_pack_path.exists():
        return False
    source_pack = _load_json_object(source_pack_path)
    reconciliation = source_pack.get("topic_reconciliation")
    if not isinstance(reconciliation, dict):
        reconciliation = {}
    reconciliation["reference_repair"] = repair_summary
    if repair_summary.get("applied") and repair_summary.get("status") == "succeeded":
        markdown_path = workspace / str(repair_summary["applied_markdown_path"])
        if markdown_path.exists():
            text = markdown_path.read_text(encoding="utf-8", errors="replace")
            reconciliation["reconciliation_text"] = text
            reconciliation["markdown_sha256"] = sha256_text(text)
    source_pack["topic_reconciliation"] = reconciliation
    write_json_atomic(source_pack_path, source_pack)
    return True


def run_topic_reference_repair(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    mode: str = "dry-run",
    model: str = DEFAULT_QWEN36_MODEL,
    topics: list[str] | None = None,
    input_token_ceiling: int = DEFAULT_INPUT_TOKEN_CEILING,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    apply_repaired: bool = False,
    enrich_source_packs: bool = True,
    researcher: AsyncResearcherLike | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if mode not in {"dry-run", "live"}:
        raise ValueError("mode must be dry-run or live")
    started = time.monotonic()
    generated_at = utc_now()
    topic_ids = topics or _topic_ids_from_source_packs(workspace)
    if not topic_ids:
        raise TopicReferenceRepairError("narrative_source_packs_missing")
    if mode == "live" and researcher is None:
        researcher = LocalLLMResearcher(smart_model=model, fast_model=model, max_tokens=max_output_tokens)

    outputs: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    enriched_count = 0
    for topic_id in topic_ids:
        try:
            repair_input = collect_repair_input(
                workspace,
                topic_id=topic_id,
                input_token_ceiling=input_token_ceiling,
            )
            repair_md_path, repair_json_path, validation_json_path, validation_md_path = _repair_paths(
                workspace, topic_id
            )
            repair_md_path.parent.mkdir(parents=True, exist_ok=True)
            if repair_input.initial_validation.status == "succeeded":
                repaired_text = repair_input.original_markdown
                repair_action = "skipped_already_valid"
            elif mode == "live":
                if researcher is None:
                    raise TopicReferenceRepairError("local_researcher_missing")
                repaired_text = asyncio.run(researcher.run(repair_input.prompt))
                repair_action = "llm_repair"
            else:
                repaired_text = repair_input.original_markdown
                repair_action = "dry_run_no_mutation"
            repaired_text = _strip_code_fence(repaired_text)
            repaired_text, deterministic_fix_count = apply_deterministic_reference_safety_fixes(
                text=repaired_text,
                ledger=repair_input.ledger,
            )
            repair_md_path.write_text(repaired_text, encoding="utf-8")
            repaired_validation = validate_reconciliation_references(
                text=repaired_text,
                ledger=repair_input.ledger,
                require_citations=True,
            )
            validation_payload = repaired_validation.to_dict()
            validation_payload.update(
                {
                    "generated_at": generated_at,
                    "run_id": run_id,
                    "repair_markdown_path": workspace_relpath(workspace, repair_md_path),
                    "repair_markdown_sha256": sha256_text(repaired_text),
                    "validator_owner_id": OWNER_TOPIC_REFERENCE_VALIDATION,
                }
            )
            write_json_atomic(validation_json_path, validation_payload)
            validation_md_path.write_text(
                render_validation_markdown(repaired_validation),
                encoding="utf-8",
            )
            applied = False
            if apply_repaired and repaired_validation.status == "succeeded":
                repair_input.original_markdown_path.write_text(repaired_text, encoding="utf-8")
                applied = True
            sidecar = {
                "schema_version": SCHEMA_VERSION,
                "owner_id": OWNER_TOPIC_REFERENCE_REPAIR,
                "status": repaired_validation.status,
                "generated_at": generated_at,
                "run_id": run_id,
                "mode": mode,
                "topic_id": topic_id,
                "title": repair_input.title,
                "model": model,
                "repair_action": repair_action,
                "deterministic_reference_fix_count": deterministic_fix_count,
                "input_digest_sha256": repair_input.input_digest_sha256,
                "input_token_estimate": repair_input.input_token_estimate,
                "output_token_estimate": estimate_tokens(repaired_text),
                "original_markdown_path": workspace_relpath(
                    workspace, repair_input.original_markdown_path
                ),
                "original_markdown_sha256": sha256_text(repair_input.original_markdown),
                "repair_markdown_path": workspace_relpath(workspace, repair_md_path),
                "repair_markdown_sha256": sha256_text(repaired_text),
                "validation_json_path": workspace_relpath(workspace, validation_json_path),
                "validation_markdown_path": workspace_relpath(workspace, validation_md_path),
                "initial_validation": repair_input.initial_validation.to_dict(),
                "repaired_validation": repaired_validation.to_dict(),
                "applied": applied,
                "applied_markdown_path": workspace_relpath(
                    workspace, repair_input.original_markdown_path
                ) if applied else None,
            }
            write_json_atomic(repair_json_path, sidecar)
            if enrich_source_packs:
                repair_summary = {
                    "status": sidecar["status"],
                    "owner_id": OWNER_TOPIC_REFERENCE_REPAIR,
                    "generated_at": generated_at,
                    "model": model,
                    "repair_action": repair_action,
                    "deterministic_reference_fix_count": deterministic_fix_count,
                    "repair_json_path": workspace_relpath(workspace, repair_json_path),
                    "repair_markdown_path": sidecar["repair_markdown_path"],
                    "validation_json_path": sidecar["validation_json_path"],
                    "invalid_reference_count_before": repair_input.initial_validation.invalid_reference_count,
                    "invalid_reference_count_after": repaired_validation.invalid_reference_count,
                    "bare_reference_warning_count_after": repaired_validation.bare_reference_warning_count,
                    "applied": applied,
                    "applied_markdown_path": sidecar["applied_markdown_path"],
                }
                if _enrich_source_pack(
                    workspace=workspace,
                    topic_id=topic_id,
                    repair_summary=repair_summary,
                ):
                    enriched_count += 1
                sidecar["source_pack_enriched"] = True
                write_json_atomic(repair_json_path, sidecar)
            outputs.append(sidecar)
            blockers.extend(repaired_validation.blockers)
        except Exception as exc:  # noqa: BLE001 - receipt boundary.
            blockers.append(
                {
                    "code": str(exc).split(":", 1)[0] or "topic_reference_repair_failed",
                    "topic_id": topic_id,
                    "reason": str(exc),
                    "retryable": True,
                }
            )

    usage = researcher.usage() if researcher is not None else {}
    receipt_path = _receipt_path(workspace, run_id)
    checks = {
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "applied_topic_count": sum(1 for item in outputs if item.get("applied")),
        "initial_invalid_reference_count": sum(
            int(item.get("initial_validation", {}).get("invalid_reference_count", 0))
            for item in outputs
        ),
        "repaired_invalid_reference_count": sum(
            int(item.get("repaired_validation", {}).get("invalid_reference_count", 0))
            for item in outputs
        ),
        "repaired_unknown_reference_count": sum(
            int(item.get("repaired_validation", {}).get("unknown_reference_count", 0))
            for item in outputs
        ),
        "repaired_title_mismatch_count": sum(
            int(item.get("repaired_validation", {}).get("title_mismatch_count", 0))
            for item in outputs
        ),
        "repaired_bare_reference_warning_count": sum(
            int(item.get("repaired_validation", {}).get("bare_reference_warning_count", 0))
            for item in outputs
        ),
        "source_packs_enriched_count": enriched_count,
    }
    status = "blocked" if blockers else "succeeded"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_TOPIC_REFERENCE_REPAIR,
        "owner_id": OWNER_TOPIC_REFERENCE_REPAIR,
        "status": status,
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "output_paths": [item["repair_markdown_path"] for item in outputs],
        "sidecar_paths": [
            workspace_relpath(workspace, _repair_paths(workspace, str(item["topic_id"]))[1])
            for item in outputs
        ],
        "validation_paths": [
            item["validation_json_path"] for item in outputs
        ],
        "apply_repaired": apply_repaired,
        "source_packs_enriched": enrich_source_packs and enriched_count == len(outputs),
        "provider_calls_made": mode == "live",
        "live_provider_calls_made": mode == "live",
        "paid_provider_calls_made": False,
        "usage": usage,
        "local_provider_receipt_paths": list(usage.get("provider_receipt_paths") or []),
        "checks": checks,
        "blockers": blockers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "receipt_path": workspace_relpath(workspace, receipt_path),
    }
    if apply_repaired:
        # Applied repairs rewrote 07D's Markdown: refresh its output digest so
        # the next 07E binds what it validates.
        refresh_substep_outputs(workspace, run_id, "topic-reconciliation")
    write_json_atomic(receipt_path, receipt)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair topic reconciliation references and revalidate candidates."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--model", default=DEFAULT_QWEN36_MODEL)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--input-token-ceiling", type=int, default=DEFAULT_INPUT_TOKEN_CEILING)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--apply", action="store_true", dest="apply_repaired")
    parser.add_argument("--no-enrich-source-packs", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def run_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        receipt = run_topic_reference_repair(
            workspace=args.workspace,
            run_id=args.run_id,
            mode=args.mode,
            model=args.model,
            topics=args.topics,
            input_token_ceiling=args.input_token_ceiling,
            max_output_tokens=args.max_output_tokens,
            apply_repaired=args.apply_repaired,
            enrich_source_packs=not args.no_enrich_source_packs,
        )
    except (TopicReferenceRepairError, TopicReferenceValidationError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2, sort_keys=True))
        return 1
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "topic_count": receipt.get("topic_count"),
                    "completed_topic_count": receipt.get("completed_topic_count"),
                    "checks": receipt.get("checks"),
                    "receipt_path": receipt.get("receipt_path"),
                    "blockers": receipt.get("blockers"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if receipt.get("status") == "succeeded" else 1


__all__ = [
    "OWNER_TOPIC_REFERENCE_REPAIR",
    "TopicReferenceRepairError",
    "build_reference_repair_prompt",
    "apply_deterministic_reference_safety_fixes",
    "collect_repair_input",
    "run_main",
    "run_topic_reference_repair",
]
