#!/usr/bin/env python3
"""Validate the Phase 29 analysis reconciliation spec against repo docs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_LABELS = {
    "canonical",
    "NeurIPS-local",
    "historical",
    "legacy",
    "planned",
    "experimental",
    "unsupported",
}
VALID_KINDS = {"command", "path", "script", "support-label", "contract"}
HELP_CHECK_ALLOWLIST = {
    "scripts/package_deep_research_docs.py",
    "scripts/traceability_recovery.py",
    "scripts/inline_traceability_references.py",
    "scripts/generate_slides_agentic.py",
    "scripts/phase8_paper_analysis.py",
}
ANALYSIS_SCRIPT_REFERENCES = {
    "scripts/setup_vector_store_enhanced.py",
    "scripts/setup_vector_store.py",
    "scripts/setup_gpt_researcher_with_vectorstore.py",
    "scripts/query_gpt_researcher_hybrid.py",
    "scripts/package_deep_research_docs.py",
    "scripts/qa_analysis_docs.py",
    "scripts/traceability_recovery.py",
    "scripts/inline_traceability_references.py",
    "scripts/generate_focused_podcast.py",
    "scripts/rank_facts.py",
    "scripts/generate_slides_agentic.py",
    "scripts/generate_podcast_gemini.py",
    "scripts/phase8_paper_analysis.py",
}
SCRIPT_RE = re.compile(r"scripts/[A-Za-z0-9_./-]+\.py")
PHASE28_DIRS = (
    REPO_ROOT
    / ".planning"
    / "phases"
    / "28-analysis-workflow-contract-and-script-alignment",
    REPO_ROOT
    / ".planning"
    / "milestones"
    / "v1.3-phases"
    / "28-analysis-workflow-contract-and-script-alignment",
)


@dataclass(frozen=True)
class Failure:
    item_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.item_id, "reason": self.reason}


@dataclass(frozen=True)
class LimitedCheck:
    item_id: str
    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.item_id, "path": self.path, "reason": self.reason}


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def resolve_repo_path(value: object) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, "missing path"
    candidate = Path(value)
    resolved = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    try:
        resolved = resolved.resolve()
    except OSError:
        return None, "path cannot be resolved"
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        return None, "path escapes repository"
    return resolved, None


def write_report(
    output_path: Path,
    *,
    spec_path: Path,
    checked_items: list[str],
    failures: list[Failure],
    limited_checks: list[LimitedCheck],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "spec_path": display_path(spec_path),
        "report_path": display_path(output_path),
        "checked_items": checked_items,
        "failures": [failure.to_dict() for failure in failures],
        "limited_checks": [limited.to_dict() for limited in limited_checks],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_spec(spec_path: Path) -> tuple[dict[str, object] | None, list[Failure]]:
    try:
        return json.loads(spec_path.read_text(encoding="utf-8")), []
    except (OSError, json.JSONDecodeError) as exc:
        return None, [Failure("spec", f"malformed spec: {exc}")]


def read_docs(paths: list[Path], item_id: str, failures: list[Failure]) -> dict[str, str]:
    docs: dict[str, str] = {}
    for doc_path in paths:
        try:
            docs[repo_relative(doc_path)] = doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(Failure(item_id, f"missing expected doc {repo_relative(doc_path)}: {exc}"))
    return docs


def validate_doc_paths(item_id: str, raw_docs: object, failures: list[Failure]) -> list[Path]:
    if not isinstance(raw_docs, list) or not raw_docs:
        failures.append(Failure(item_id, "expected_docs must be a non-empty list"))
        return []

    resolved_docs: list[Path] = []
    for raw_doc in raw_docs:
        doc_path, error = resolve_repo_path(raw_doc)
        if error:
            failures.append(Failure(item_id, f"{error}: {raw_doc}"))
            continue
        assert doc_path is not None
        if doc_path.suffix != ".md":
            failures.append(Failure(item_id, f"expected_doc is not markdown: {raw_doc}"))
            continue
        if not doc_path.exists():
            failures.append(Failure(item_id, f"missing expected doc: {raw_doc}"))
            continue
        resolved_docs.append(doc_path)
    return resolved_docs


def validate_item(
    item: object,
    *,
    failures: list[Failure],
    limited_checks: list[LimitedCheck],
    represented_scripts: set[str],
    coverage_docs: set[Path],
    checked_items: list[str],
) -> None:
    if not isinstance(item, dict):
        failures.append(Failure("item", "item must be an object"))
        return

    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        failures.append(Failure("item", "item id must be a non-empty string"))
        return
    checked_items.append(item_id)

    kind = item.get("kind")
    if kind not in VALID_KINDS:
        failures.append(Failure(item_id, f"unknown kind: {kind}"))

    path = item.get("path")
    resolved_path, path_error = resolve_repo_path(path)
    if path_error:
        failures.append(Failure(item_id, f"{path_error}: {path}"))
    elif isinstance(path, str) and path.startswith("scripts/"):
        represented_scripts.add(path)

    expected_docs = validate_doc_paths(item_id, item.get("expected_docs"), failures)
    coverage_docs.update(expected_docs)
    docs = read_docs(expected_docs, item_id, failures)
    combined_docs = "\n".join(docs.values())

    expected_phrases = item.get("expected_phrases", [])
    if not isinstance(expected_phrases, list):
        failures.append(Failure(item_id, "expected_phrases must be a list"))
    else:
        for phrase in expected_phrases:
            if not isinstance(phrase, str) or not phrase:
                failures.append(Failure(item_id, "expected_phrase must be a non-empty string"))
            elif phrase not in combined_docs:
                failures.append(Failure(item_id, f"missing expected phrase: {phrase}"))

    limited_reason = item.get("limited_reason")
    if limited_reason is not None and not isinstance(limited_reason, str):
        failures.append(Failure(item_id, "limited_reason must be a string when present"))
    if isinstance(limited_reason, str) and limited_reason:
        limited_checks.append(
            LimitedCheck(item_id=item_id, path=str(path), reason=limited_reason)
        )

    expected_labels = item.get("expected_labels", [])
    if not isinstance(expected_labels, list):
        failures.append(Failure(item_id, "expected_labels must be a list"))
    else:
        for label in expected_labels:
            if label not in VALID_LABELS:
                failures.append(Failure(item_id, f"unknown label: {label}"))
            elif label not in combined_docs and not limited_reason:
                failures.append(Failure(item_id, f"missing expected_label: {label}"))

    safe_checks = item.get("safe_checks", [])
    if not isinstance(safe_checks, list):
        failures.append(Failure(item_id, "safe_checks must be a list"))
        return

    if "exists" in safe_checks and (resolved_path is None or not resolved_path.exists()):
        failures.append(Failure(item_id, f"missing required path: {path}"))

    if "help" in safe_checks:
        path_string = path if isinstance(path, str) else ""
        if path_string not in HELP_CHECK_ALLOWLIST:
            failures.append(Failure(item_id, f"help check not allowlisted: {path}"))
        elif resolved_path is not None and resolved_path.exists():
            result = subprocess.run(
                [sys.executable, str(resolved_path), "--help"],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                failures.append(Failure(item_id, f"help check failed for {path}"))


def validate_spec(spec: dict[str, object] | None) -> tuple[list[str], list[Failure], list[LimitedCheck]]:
    failures: list[Failure] = []
    limited_checks: list[LimitedCheck] = []
    checked_items: list[str] = []
    represented_scripts: set[str] = set()
    coverage_docs: set[Path] = set()

    if not any(list(phase_dir.glob("28-*-SUMMARY.md")) for phase_dir in PHASE28_DIRS):
        failures.append(
            Failure(
                "phase28_completion_precondition",
                "phase28_completion_precondition: Phase 28 summary evidence is missing",
            )
        )

    if spec is None:
        return checked_items, failures, limited_checks

    if spec.get("schema_version") != SCHEMA_VERSION:
        failures.append(Failure("spec", "schema_version must be 1"))

    items = spec.get("items")
    if not isinstance(items, list):
        failures.append(Failure("spec", "items must be a list"))
        return checked_items, failures, limited_checks

    seen_ids: set[str] = set()
    for item in items:
        item_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(item_id, str):
            if item_id in seen_ids:
                failures.append(Failure(item_id, "duplicate item id"))
            seen_ids.add(item_id)
        validate_item(
            item,
            failures=failures,
            limited_checks=limited_checks,
            represented_scripts=represented_scripts,
            coverage_docs=coverage_docs,
            checked_items=checked_items,
        )

    discovered_scripts: set[str] = set()
    for doc_path in sorted(coverage_docs):
        try:
            doc_text = doc_path.read_text(encoding="utf-8")
        except OSError:
            continue
        discovered_scripts.update(
            script
            for script in SCRIPT_RE.findall(doc_text)
            if script in ANALYSIS_SCRIPT_REFERENCES
        )
    missing_coverage = sorted(discovered_scripts - represented_scripts)
    if missing_coverage:
        failures.append(
            Failure(
                "spec",
                "missing script reference coverage: " + ", ".join(missing_coverage),
            )
        )

    return checked_items, failures, limited_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Phase 29 analysis reconciliation spec against docs."
    )
    parser.add_argument("--spec", required=True, help="Path to reconciliation spec JSON")
    parser.add_argument("--output", required=True, help="Path to write report JSON")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec).expanduser()
    output_path = Path(args.output).expanduser()
    if not spec_path.is_absolute():
        spec_path = REPO_ROOT / spec_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    spec_path = spec_path.resolve()
    output_path = output_path.resolve()
    initial_failures: list[Failure] = []
    spec, load_failures = load_spec(spec_path)
    checked_items, validation_failures, limited_checks = validate_spec(spec)
    failures = initial_failures + load_failures + validation_failures
    write_report(
        output_path,
        spec_path=spec_path,
        checked_items=checked_items,
        failures=failures,
        limited_checks=limited_checks,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
