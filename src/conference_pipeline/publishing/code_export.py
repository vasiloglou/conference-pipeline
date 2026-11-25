"""Fresh-history export of the platform code (v6.0 Phase 7, D2 (a)).

The export holds exactly the tracked files that ``code_export.toml`` selects **at one
source commit**, extracted with ``git archive`` so ignored files, the working tree and
``.env`` cannot reach it. It becomes a one-commit repository whose author, committer and
dates are fixed, so the same commit and identity always give the same export commit.

Three gates run over the export: private markers over its tree, gitleaks over its whole
history, and the docs-link check run inside it. Nothing here pushes, adds a remote or
changes a repository's visibility: publishing is the owner's, per release (R3).
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from conference_pipeline._security.private_markers import (
    builtin_rules,
    extra_rules_from_env,
    has_publishing_marker,
    has_publishing_marker_bytes,
    load_allowlist,
    redact_text,
    scan_paths,
)

MANIFEST_NAME = "code_export.toml"
SCHEMA_VERSION = 1


class CodeExportError(RuntimeError):
    """A refusal: the export was not built (exit 2)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class CodeExportConfig:
    project_root: Path
    source_commit: str
    out_dir: Path
    author_name: str
    author_email: str
    dry_run: bool = False


def _git(root: Path, *args: str, env: Mapping[str, str] | None = None, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], input=data, capture_output=True, check=False, env=dict(env) if env else None
    )
    if result.returncode != 0:
        raise CodeExportError("git_failed", f"git {args[0]}: {result.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return result.stdout


def resolve_commit(project_root: Path, commit: str) -> str:
    """The full sha of ``commit``, which must be ``HEAD`` or one of its ancestors (spec §6.2)."""
    try:
        sha = _git(project_root, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}").decode().strip()
    except CodeExportError:
        raise CodeExportError("unknown_commit", commit) from None
    try:
        _git(project_root, "merge-base", "--is-ancestor", sha, "HEAD")
    except CodeExportError:
        raise CodeExportError("not_an_ancestor", f"{sha[:12]} is not HEAD or an ancestor of it") from None
    return sha


def load_manifest(project_root: Path, commit: str) -> dict[str, list[str]]:
    """``code_export.toml`` as it is **at the commit**, never as the working tree has it."""
    try:
        raw = _git(project_root, "show", f"{commit}:{MANIFEST_NAME}")
    except CodeExportError:
        raise CodeExportError("manifest_missing", f"{MANIFEST_NAME} is not in {commit[:12]}") from None
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CodeExportError("manifest_unreadable", str(exc)) from None
    export = data.get("export")
    if data.get("schema_version") != SCHEMA_VERSION or not isinstance(export, dict):
        raise CodeExportError("manifest_unreadable", "schema_version 1 with an [export] table is required")
    include, exclude = export.get("include"), export.get("exclude", [])
    for name, value in (("include", include), ("exclude", exclude)):
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise CodeExportError("manifest_unreadable", f"export.{name} must be a list of non-empty strings")
    if not include:
        raise CodeExportError("manifest_unreadable", "export.include is empty")
    return {"include": list(include), "exclude": list(exclude)}


def _matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern or fnmatch.fnmatchcase(path, pattern)


def selected_files(project_root: Path, commit: str) -> list[str]:
    """Sorted tracked paths at ``commit`` that the manifest includes and does not exclude."""
    manifest = load_manifest(project_root, commit)
    tracked = _git(project_root, "ls-tree", "-r", "--name-only", "-z", commit).decode("utf-8").split("\0")
    tracked = [path for path in tracked if path]
    for pattern in manifest["include"]:
        if not any(_matches(path, pattern) for path in tracked):
            raise CodeExportError("include_missing", f"{pattern} matches no tracked file at {commit[:12]}")
    return sorted(
        path
        for path in tracked
        if any(_matches(path, pattern) for pattern in manifest["include"])
        and not any(_matches(path, pattern) for pattern in manifest["exclude"])
    )


def tree_sha256(root: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode("utf-8") + b"\0" + hashlib.sha256((root / rel).read_bytes()).hexdigest().encode() + b"\n")
    return digest.hexdigest()


def _marker_rules(env: Mapping[str, str] | None):
    return builtin_rules() + extra_rules_from_env(env if env is not None else os.environ)


_GITHUB_NOREPLY = re.compile(r"(?:\d+\+)?[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})@users\.noreply\.github\.com")


def _check_identity(config: CodeExportConfig) -> None:
    """The commit's author must be public. A GitHub noreply alias is public by design, so the
    generic email rule does not apply to it; the operator's own literals still do."""
    for value in (config.author_name, config.author_email):
        rules = extra_rules_from_env(os.environ) if _GITHUB_NOREPLY.fullmatch(value) else _marker_rules(None)
        if not value.strip() or redact_text(value, rules) != value:
            raise CodeExportError("identity_refused", "the export identity must be public and not a private marker")


def build_code_export(config: CodeExportConfig) -> dict[str, Any]:
    """Build the export (or, with ``dry_run``, list it) and return the receipt payload."""
    root = config.project_root.resolve()
    commit = resolve_commit(root, config.source_commit)
    files = selected_files(root, commit)
    _check_identity(config)
    payload: dict[str, Any] = {
        "receipt_type": "conference_pipeline.code_export",
        "schema_version": SCHEMA_VERSION,
        "source_commit": commit,
        "file_count": len(files),
        "dry_run": config.dry_run,
        "secret_values_recorded": False,
    }
    if config.dry_run:
        payload["files"] = files
        return payload
    out = config.out_dir
    if out.exists() or out.is_symlink():
        raise CodeExportError("out_exists", "the export directory must not exist (write-once)")
    archive = _git(root, "archive", "--format=tar", commit, "--", *files)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".code-export-", dir=out.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(staging, filter="data")
        extracted = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file() or p.is_symlink())
        if extracted != files:
            raise CodeExportError("archive_mismatch", "git archive did not give exactly the selected files")
        date = _git(root, "show", "-s", "--format=%cI", commit).decode().strip()
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": config.author_name,
            "GIT_AUTHOR_EMAIL": config.author_email,
            "GIT_COMMITTER_NAME": config.author_name,
            "GIT_COMMITTER_EMAIL": config.author_email,
            "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_DATE": date,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        _git(staging, "init", "-q", "-b", "main", env=env)
        # --force: the export's own .gitignore must not drop a selected, tracked file (r1 F1).
        _git(staging, "add", "--force", "-A", env=env)
        staged = sorted(p for p in _git(staging, "ls-files", "-z", env=env).decode("utf-8").split("\0") if p)
        if staged != files:
            raise CodeExportError("commit_mismatch", "the export commit would not hold exactly the selected files")
        message = f"Initial public release of conference-pipeline (source {commit[:12]})\n"
        _git(staging, "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-F", "-", env=env, data=message.encode())
        export_commit = _git(staging, "rev-parse", "HEAD").decode().strip()
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    payload.update(
        export_commit=export_commit,
        tree_sha256=tree_sha256(out, files),
        commit_date=date,
    )
    return payload


def _gitleaks_history(out_dir: Path) -> list[dict[str, Any]]:
    binary = shutil.which("gitleaks")
    if binary is None:
        return [{"gate": "secrets", "detail": "gitleaks_not_installed"}]
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        args = [binary, "git", "--no-banner", "--report-format", "json", "--report-path", str(report), "--exit-code", "0"]
        config = out_dir / ".gitleaks.toml"
        if config.is_file():
            args += ["--config", str(config)]
        result = subprocess.run([*args, str(out_dir)], capture_output=True, text=True, check=False)
        # A scan that ran exits 0 with --exit-code 0; anything else never counts as clean.
        if result.returncode != 0:
            return [{"gate": "secrets", "detail": f"gitleaks_failed_exit_{result.returncode}"}]
        try:
            raw = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return [{"gate": "secrets", "detail": "gitleaks_report_unreadable"}]
    # Every entry must be a finding object: one that is not means the report cannot be
    # read, never that it is clean (pr-43-r1 F3).
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        return [{"gate": "secrets", "detail": "gitleaks_report_unreadable"}]
    return [
        {"gate": "secrets", "path": str(item.get("File", "")), "detail": f"{item.get('RuleID', '?')} line {item.get('StartLine', '?')}"}
        for item in raw
    ]


def _skipped_file_findings(out_dir: Path, skipped: list[tuple[str, str]], rules) -> list[dict[str, Any]]:
    """The files the text scan did not read, checked by the byte rules (pr-43-r1 F1).

    A binary or oversized file meets has_publishing_marker_bytes (text, UTF-16, zip, PDF
    and binary rules); a symlink's target text meets the text rules; anything else that
    could not be read is a finding, never a pass."""
    findings: list[dict[str, Any]] = []
    for rel, reason in skipped:
        if Path(rel).parts[:1] == (".git",):
            continue
        path = out_dir / rel
        try:
            if path.is_symlink():
                hit = has_publishing_marker(os.readlink(path), rules=rules)
                detail = "symlink target holds a private marker"
            elif path.is_file():
                hit = has_publishing_marker_bytes(path.read_bytes(), rules=rules)
                detail = f"{reason} file holds a private marker"
            else:
                hit, detail = True, f"unscannable ({reason})"
        except OSError as exc:
            hit, detail = True, f"unreadable ({type(exc).__name__})"
        if hit:
            findings.append({"gate": "private_markers", "path": rel, "detail": detail})
    return findings


def check_code_export(out_dir: Path, *, env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Run the three gates over a built export; an empty list means every gate is clean."""
    findings: list[dict[str, Any]] = []
    rules = _marker_rules(env)
    allowlist_path = out_dir / ".private-markers-allowlist.toml"
    report = scan_paths(
        out_dir, ["."], rules, load_allowlist(allowlist_path if allowlist_path.is_file() else None), tracked_only=False
    )
    findings += [
        {"gate": "private_markers", "path": f.path, "detail": f"{f.rule} line {f.line}"}
        for f in report.findings
        if not Path(f.path).parts or Path(f.path).parts[0] != ".git"
    ]
    findings += _skipped_file_findings(out_dir, report.skipped_paths, rules)
    findings += _gitleaks_history(out_dir)
    links = subprocess.run(
        [sys.executable, "scripts/check_docs_links.py"], cwd=out_dir, capture_output=True, text=True, check=False
    )
    if links.returncode != 0:
        broken = [line[2:] for line in links.stdout.splitlines() if line.startswith("- ")] or [f"exit {links.returncode}"]
        findings += [{"gate": "docs_links", "detail": redact_text(item, rules)} for item in broken]
    return findings
