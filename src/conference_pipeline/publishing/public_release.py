"""Reusable public-release orchestration for conference workspaces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from conference_pipeline.publishing.receipt_paths import (
    ReceiptPathError,
    confine_workflow_receipt_path,
    write_json_atomic_secure,
)


RECEIPT_TYPE = "conference_pipeline.public_release"
PUBLIC_RELEASE_JOURNAL_RECEIPT_TYPE = "conference_pipeline.public_release_journal"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class PublicReleaseError(RuntimeError):
    """Raised when public-release orchestration cannot be planned or run."""

    def __init__(
        self,
        reason: str,
        *,
        code: str | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.label = label


class PublicReleaseStage(str, Enum):
    VALIDATE_MANIFEST = "validate_manifest"
    EXPORT_DRY_RUN = "export_dry_run"
    EXPORT_WRITE = "export_write"
    BUILD_INDEX = "build_index"
    GENERATE_SITE = "generate_site"
    STAGE_REPO = "stage_repo"
    PACKAGE_DATASET = "package_dataset"
    PREFLIGHT_LIVE = "preflight_live"
    CHECK_LIVENESS = "check_liveness"
    CHECK_METADATA = "check_metadata"
    LIVE_GITHUB = "live_github"
    LIVE_R2 = "live_r2"
    LIVE_PAGES = "live_pages"
    LIVE_HF = "live_hf"
    LIVE_ZENODO = "live_zenodo"


PUBLIC_RELEASE_STAGES: tuple[PublicReleaseStage, ...] = tuple(PublicReleaseStage)


@dataclass(frozen=True)
class PublicReleaseCommand:
    stage: PublicReleaseStage
    argv: list[str]
    receipt_path: Path
    cwd: Path
    live: bool = False
    shell: bool = False


@dataclass(frozen=True)
class PublicReleaseConfig:
    conference_dir: Path | None = None
    conference_slug: str | None = None
    bundle_id: str | None = None
    receipt_path: Path | None = None
    env_file: Path = DEFAULT_ENV_FILE
    project_root: Path | None = None
    validate_manifest: bool = False
    dry_run: bool = False
    write_bundle: bool = False
    build_index: bool = False
    generate_site: bool = False
    stage_repo: bool = False
    package_dataset: bool = False
    preflight_live: bool = False
    check_liveness: bool = False
    check_metadata: bool = False
    live_github: bool = False
    live_r2: bool = False
    live_pages: bool = False
    live_hf: bool = False
    live_zenodo: bool = False
    reviewed_dry_run_receipt: Path | None = None
    export_write_receipt_path: Path | None = None
    public_repo_dry_run_receipt: Path | None = None
    ack_caution_families: tuple[str, ...] = ()
    ack_live_github: bool = False
    ack_live_r2: bool = False
    ack_live_pages: bool = False
    ack_live_hf: bool = False
    ack_live_zenodo: bool = False
    github_repo: str | None = None
    r2_base_url: str | None = None
    r2_bucket: str | None = None
    r2_prefix: str | None = None
    pages_project: str | None = None
    hf_repo_id: str | None = None
    zenodo_sandbox: bool = False
    public_repo_secret_scanner: Callable[[list[Path]], list[dict]] | None = None
    # v6.0 Phase 3: the release CLI's mock mode passes the index's named offline
    # embedding provider; None keeps the index's own default.
    index_embedding_provider: str | None = None
    index_embedding_cache: Path | None = None
    # v6.0 Phase 3: the owner's approval for a live publish, recorded verbatim in the
    # top-level receipt ({"approved_by", "surfaces", "recorded_at"}).
    owner_approval: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PublicReleaseResult:
    status: str
    exit_code: int
    commands: tuple[PublicReleaseCommand, ...]
    blocked_stages: tuple[dict[str, Any], ...]
    receipt_path: Path
    journal_path: Path | None = None
    secret_values_recorded: bool = False


Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_receipt_exception_code(
    prefix: str,
    stage: str,
    exc: Exception,
) -> str:
    """Return an allow-listed receipt code without serializing exception text."""
    for component in (prefix, stage):
        if not component.isascii() or not component.isidentifier():
            raise ValueError("receipt exception code labels must be safe identifiers")
    exception_type = type(exc).__name__
    if (
        not exception_type.isascii()
        or not exception_type.isidentifier()
        or len(exception_type) > 128
    ):
        exception_type = "Exception"
    return f"{prefix}:{stage}:exception:{exception_type}"


def _stable_exception_type(exc: Exception) -> str:
    return stable_receipt_exception_code(
        "public_release",
        "exception_type",
        exc,
    ).rsplit(":", 1)[-1]


def write_json(path: Path, payload: Any) -> None:
    write_json_atomic_secure(path, payload)


def resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parents[3]


def ensure_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PublicReleaseError(f"{label} escapes project root: {resolved}") from exc
    return resolved


def resolve_conference_dir(config: PublicReleaseConfig) -> Path:
    project_root = resolve_project_root(config.project_root)
    if config.conference_dir is not None and config.conference_slug is not None:
        raise PublicReleaseError("pass either conference_dir or conference_slug, not both")
    if config.conference_slug is not None:
        slug = config.conference_slug.strip()
        if not slug or "/" in slug or ".." in slug:
            raise PublicReleaseError("conference_slug must be a simple conferences/<slug> name")
        return ensure_under(project_root / "conferences" / slug, project_root, label="conference_dir")
    if config.conference_dir is not None:
        return ensure_under(config.conference_dir, project_root, label="conference_dir")
    raise PublicReleaseError("conference_dir or conference_slug is required")


def default_bundle_id(conference_dir: Path) -> str:
    return f"{conference_dir.name}-v1.0.0"


def default_receipt_dir(conference_dir: Path) -> Path:
    return conference_dir / "storage" / "workflow-runs"


def default_bundle_root(conference_dir: Path, bundle_id: str) -> Path:
    return conference_dir / "exports" / "public-release" / bundle_id


def repo_name_from_github(config: PublicReleaseConfig, conference_slug: str) -> str:
    if config.github_repo and "/" in config.github_repo:
        return config.github_repo.split("/", 1)[1]
    if config.github_repo:
        return config.github_repo
    return f"{conference_slug}-data"


def selected_any_stage(config: PublicReleaseConfig) -> bool:
    return any(
        (
            config.validate_manifest,
            config.dry_run,
            config.write_bundle,
            config.build_index,
            config.generate_site,
            config.stage_repo,
            config.package_dataset,
            config.preflight_live,
            config.check_liveness,
            config.check_metadata,
            config.live_github,
            config.live_r2,
            config.live_pages,
            config.live_hf,
            config.live_zenodo,
        )
    )


def export_review_boundary_blocker(
    config: PublicReleaseConfig,
) -> dict[str, Any] | None:
    if not config.write_bundle or not (config.validate_manifest or config.dry_run):
        return None
    return {
        "stage": PublicReleaseStage.EXPORT_WRITE.value,
        "reason": "export_review_boundary_required",
        "missing_approvals": [],
        "missing_receipts": [],
        "missing_targets": [],
    }


def _base_context(config: PublicReleaseConfig) -> tuple[Path, Path, str, Path, Path]:
    project_root = resolve_project_root(config.project_root)
    conference_dir = resolve_conference_dir(config)
    bundle_id = config.bundle_id or default_bundle_id(conference_dir)
    receipt_dir = default_receipt_dir(conference_dir)
    bundle_root = default_bundle_root(conference_dir, bundle_id)
    return project_root, conference_dir, bundle_id, receipt_dir, bundle_root


def _confine_receipt_path(
    config: PublicReleaseConfig,
    candidate: Path,
    *,
    label: str,
    must_exist: bool = False,
) -> Path:
    project_root = resolve_project_root(config.project_root)
    conference_dir = resolve_conference_dir(config)
    return confine_workflow_receipt_path(
        conference_dir,
        candidate,
        project_root=project_root,
        label=label,
        must_exist=must_exist,
    )


def _path_error_blocker(exc: ReceiptPathError) -> dict[str, Any]:
    return {
        "stage": "receipt_paths",
        "reason": "unsafe_receipt_path",
        "label": exc.label,
        "reason_code": exc.code,
        "missing_approvals": [],
        "missing_receipts": [],
        "missing_targets": [],
    }


def _public_release_error(exc: ReceiptPathError) -> PublicReleaseError:
    return PublicReleaseError(
        "unsafe_receipt_path",
        code=exc.code,
        label=exc.label,
    )


def _confined_planning_path(
    config: PublicReleaseConfig,
    candidate: Path,
    *,
    label: str,
    must_exist: bool = False,
) -> Path:
    try:
        return _confine_receipt_path(
            config,
            candidate,
            label=label,
            must_exist=must_exist,
        )
    except ReceiptPathError as exc:
        raise _public_release_error(exc) from exc


def _receipt(receipt_dir: Path, name: str, bundle_id: str) -> Path:
    return receipt_dir / name.format(bundle=bundle_id)


def _python_script(script: str, *args: str | Path | None) -> list[str]:
    # The script comes from this code's checkout, never from --project-root: a data
    # checkout on an older branch would otherwise run its own, older scripts.
    argv = [sys.executable, str(resolve_project_root(None) / script)]
    argv.extend(str(arg) for arg in args if arg is not None)
    return argv


def _append_option(argv: list[str], flag: str, value: str | Path | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _append_ack_caution(argv: list[str], config: PublicReleaseConfig) -> None:
    for family_id in config.ack_caution_families:
        argv.extend(["--ack-caution-family", family_id])


def _export_command(
    *,
    stage: PublicReleaseStage,
    config: PublicReleaseConfig,
    project_root: Path,
    conference_dir: Path,
    bundle_id: str,
    receipt_path: Path,
    dry_run: bool,
    reviewed: bool = False,
) -> PublicReleaseCommand:
    argv = _python_script(
        "scripts/export_for_public.py",
        "--conference-dir",
        conference_dir.relative_to(project_root),
        "--bundle-id",
        bundle_id,
    )
    argv.extend(["--orchestrator-stage", stage.value])
    if dry_run:
        argv.append("--dry-run")
    if reviewed and config.reviewed_dry_run_receipt is not None:
        argv.extend(["--reviewed-dry-run-receipt", str(config.reviewed_dry_run_receipt)])
    _append_ack_caution(argv, config)
    argv.extend(["--receipt-path", str(receipt_path)])
    return PublicReleaseCommand(stage=stage, argv=argv, receipt_path=receipt_path, cwd=project_root)


def _confine_command_receipt(
    config: PublicReleaseConfig,
    command: PublicReleaseCommand,
) -> PublicReleaseCommand:
    safe_path = _confined_planning_path(
        config,
        command.receipt_path,
        label=f"{command.stage.value}_receipt_path",
    )
    argv = list(command.argv)
    for index, value in enumerate(argv[:-1]):
        if value == "--receipt-path":
            argv[index + 1] = str(safe_path)
    return replace(command, argv=argv, receipt_path=safe_path)


def build_public_release_plan(config: PublicReleaseConfig) -> tuple[PublicReleaseCommand, ...]:
    if export_review_boundary_blocker(config) is not None:
        raise PublicReleaseError("export_review_boundary_required")
    project_root, conference_dir, bundle_id, receipt_dir, bundle_root = _base_context(config)
    if config.receipt_path is not None:
        config = replace(
            config,
            receipt_path=_confined_planning_path(
                config,
                config.receipt_path,
                label="receipt_path",
            ),
        )
    if config.reviewed_dry_run_receipt is not None:
        config = replace(
            config,
            reviewed_dry_run_receipt=_confined_planning_path(
                config,
                config.reviewed_dry_run_receipt,
                label="reviewed_dry_run_receipt",
                must_exist=True,
            ),
        )
    if config.public_repo_dry_run_receipt is not None:
        config = replace(
            config,
            public_repo_dry_run_receipt=_confined_planning_path(
                config,
                config.public_repo_dry_run_receipt,
                label="public_repo_dry_run_receipt",
                must_exist=True,
            ),
        )
    slug = conference_dir.name
    default_run = not selected_any_stage(config)
    commands: list[PublicReleaseCommand] = []

    if config.validate_manifest or config.dry_run or default_run:
        commands.append(
            _export_command(
                stage=PublicReleaseStage.VALIDATE_MANIFEST,
                config=config,
                project_root=project_root,
                conference_dir=conference_dir,
                bundle_id=bundle_id,
                receipt_path=_receipt(receipt_dir, "public-export-{bundle}-validation-receipt.json", bundle_id),
                dry_run=True,
            )
        )
    if config.dry_run or default_run:
        commands.append(
            _export_command(
                stage=PublicReleaseStage.EXPORT_DRY_RUN,
                config=config,
                project_root=project_root,
                conference_dir=conference_dir,
                bundle_id=bundle_id,
                receipt_path=_receipt(receipt_dir, "public-export-{bundle}-dry-run-receipt.json", bundle_id),
                dry_run=True,
            )
        )
    if config.write_bundle:
        write_receipt_path = config.export_write_receipt_path
        if write_receipt_path is None:
            write_receipt_path = _receipt(
                receipt_dir,
                "public-export-{bundle}-write-receipt.json",
                bundle_id,
            )
        else:
            write_receipt_path = _confined_planning_path(
                config,
                write_receipt_path,
                label="export_write_receipt_path",
            )
        commands.append(
            _export_command(
                stage=PublicReleaseStage.EXPORT_WRITE,
                config=config,
                project_root=project_root,
                conference_dir=conference_dir,
                bundle_id=bundle_id,
                receipt_path=write_receipt_path,
                dry_run=False,
                reviewed=True,
            )
        )
    if config.build_index:
        receipt_path = _receipt(receipt_dir, "public-index-{bundle}-receipt.json", bundle_id)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.BUILD_INDEX,
                argv=_python_script(
                    "scripts/build_public_index.py",
                    "--bundle-root",
                    bundle_root.relative_to(project_root),
                    "--receipt-path",
                    receipt_path,
                    *(
                        ("--embedding-provider", config.index_embedding_provider)
                        if config.index_embedding_provider
                        else ()
                    ),
                    *(("--embedding-cache", config.index_embedding_cache) if config.index_embedding_cache else ()),
                ),
                receipt_path=receipt_path,
                cwd=project_root,
            )
        )
    if config.generate_site:
        receipt_path = _receipt(receipt_dir, "public-site-{bundle}-receipt.json", bundle_id)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.GENERATE_SITE,
                argv=_python_script(
                    "scripts/generate_public_site.py",
                    "--bundle-root",
                    bundle_root.relative_to(project_root),
                    "--output-dir",
                    bundle_root.relative_to(project_root) / "site",
                    "--receipt-path",
                    receipt_path,
                ),
                receipt_path=receipt_path,
                cwd=project_root,
            )
        )
    # package_dataset runs before stage_repo: the staged repository copies metadata/ (v6.0
    # Phase 6 D15; Phase 3 IF6 had the CLI order them, a direct run staged a stale repo).
    if config.package_dataset:
        receipt_path = _receipt(receipt_dir, "public-dataset-{bundle}-receipt.json", bundle_id)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.PACKAGE_DATASET,
                argv=_python_script(
                    "scripts/package_public_dataset.py",
                    "--bundle-root",
                    bundle_root.relative_to(project_root),
                    "--metadata-dir",
                    bundle_root.relative_to(project_root) / "metadata",
                    "--receipt-path",
                    receipt_path,
                ),
                receipt_path=receipt_path,
                cwd=project_root,
            )
        )
    if config.stage_repo:
        receipt_path = _receipt(receipt_dir, "public-repo-{bundle}-dry-run-receipt.json", bundle_id)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.STAGE_REPO,
                argv=_python_script(
                    "scripts/push_public_repo.py",
                    "--bundle-root",
                    bundle_root.relative_to(project_root),
                    "--target-dir",
                    bundle_root.relative_to(project_root) / "public-repo",
                    "--repo-name",
                    repo_name_from_github(config, slug),
                    "--dry-run",
                    "--receipt-path",
                    receipt_path,
                ),
                receipt_path=receipt_path,
                cwd=project_root,
            )
        )
    if config.preflight_live:
        receipt_path = _receipt(receipt_dir, "public-deployment-preflight-{bundle}-receipt.json", bundle_id)
        argv = _python_script(
            "scripts/preflight_public_deployment.py",
            "--bundle-root",
            bundle_root.relative_to(project_root),
            "--receipt-path",
            receipt_path,
            "--canonical-env-path",
            config.env_file,
            "--public-repo-dry-run-receipt",
            _receipt(receipt_dir, "public-repo-{bundle}-dry-run-receipt.json", bundle_id),
            "--site-receipt",
            _receipt(receipt_dir, "public-site-{bundle}-receipt.json", bundle_id),
            "--dataset-receipt",
            _receipt(receipt_dir, "public-dataset-{bundle}-receipt.json", bundle_id),
            "--r2-upload-receipt",
            _receipt(receipt_dir, "public-cdn-{bundle}-r2-live-upload-receipt.json", bundle_id),
            "--r2-liveness-receipt",
            _receipt(receipt_dir, "public-url-liveness-{bundle}-r2-live-receipt.json", bundle_id),
        )
        _append_option(argv, "--github-repo", config.github_repo)
        _append_option(argv, "--r2-base-url", config.r2_base_url)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.PREFLIGHT_LIVE,
                argv=argv,
                receipt_path=receipt_path,
                cwd=project_root,
            )
        )
    if config.check_liveness:
        receipt_path = _receipt(receipt_dir, "public-url-liveness-{bundle}-receipt.json", bundle_id)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.CHECK_LIVENESS,
                argv=_python_script(
                    "scripts/check_public_urls.py",
                    "--bundle-root",
                    bundle_root.relative_to(project_root),
                    "--receipt-path",
                    receipt_path,
                ),
                receipt_path=receipt_path,
                cwd=project_root,
            )
        )
    if config.check_metadata:
        receipt_path = _receipt(receipt_dir, "public-release-metadata-agreement-{bundle}-receipt.json", bundle_id)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.CHECK_METADATA,
                argv=_python_script(
                    "scripts/check_public_release_metadata.py",
                    "--bundle-root",
                    bundle_root.relative_to(project_root),
                    "--receipt-path",
                    receipt_path,
                ),
                receipt_path=receipt_path,
                cwd=project_root,
            )
        )
    if config.live_github:
        receipt_path = _receipt(receipt_dir, "github-{bundle}-live-receipt.json", bundle_id)
        argv = _python_script(
            "scripts/push_public_repo.py",
            "--bundle-root",
            bundle_root.relative_to(project_root),
            "--target-dir",
            bundle_root.relative_to(project_root) / "public-repo",
            "--repo-name",
            repo_name_from_github(config, slug),
            "--live",
            "--receipt-path",
            receipt_path,
        )
        _append_option(argv, "--reviewed-dry-run-receipt", config.public_repo_dry_run_receipt)
        _append_option(argv, "--github-repo", config.github_repo)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.LIVE_GITHUB,
                argv=argv,
                receipt_path=receipt_path,
                cwd=project_root,
                live=True,
            )
        )
    if config.live_r2:
        receipt_path = _receipt(receipt_dir, "public-cdn-{bundle}-r2-live-upload-receipt.json", bundle_id)
        argv = _python_script(
            "scripts/upload_public_media.py",
            "--bundle-root",
            bundle_root.relative_to(project_root),
            "--provider",
            "r2",
            "--prefix",
            config.r2_prefix or f"conferences/{slug}",
            "--live",
            "--receipt-path",
            receipt_path,
        )
        _append_option(argv, "--bucket", config.r2_bucket)
        if _named_owner_approval(config.owner_approval):
            _append_option(argv, "--approved-by", str(config.owner_approval["approved_by"]).strip())
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.LIVE_R2,
                argv=argv,
                receipt_path=receipt_path,
                cwd=project_root,
                live=True,
            )
        )
    if config.live_pages:
        receipt_path = _receipt(receipt_dir, "cloudflare-pages-{bundle}-live-receipt.json", bundle_id)
        argv = _python_script(
            "scripts/deploy_public_pages.py",
            "--bundle-root",
            bundle_root.relative_to(project_root),
            "--site-dir",
            bundle_root.relative_to(project_root) / "site",
            "--live",
            "--receipt-path",
            receipt_path,
        )
        _append_option(argv, "--project-name", config.pages_project)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.LIVE_PAGES,
                argv=argv,
                receipt_path=receipt_path,
                cwd=project_root,
                live=True,
            )
        )
    if config.live_hf:
        receipt_path = _receipt(receipt_dir, "hf-{bundle}-live-receipt.json", bundle_id)
        argv = _python_script(
            "scripts/push_hf_dataset.py",
            "--metadata-dir",
            bundle_root.relative_to(project_root) / "metadata",
            "--live",
            "--ack-live-push",
            "--receipt-path",
            receipt_path,
        )
        _append_option(argv, "--repo-id", config.hf_repo_id)
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.LIVE_HF,
                argv=argv,
                receipt_path=receipt_path,
                cwd=project_root,
                live=True,
            )
        )
    if config.live_zenodo:
        receipt_path = _receipt(receipt_dir, "zenodo-{bundle}-live-receipt.json", bundle_id)
        argv = _python_script(
            "scripts/deposit_zenodo.py",
            "--metadata-dir",
            bundle_root.relative_to(project_root) / "metadata",
            "--live",
            "--ack-live-doi",
            "--receipt-path",
            receipt_path,
        )
        if config.zenodo_sandbox:
            argv.append("--sandbox")
        commands.append(
            PublicReleaseCommand(
                stage=PublicReleaseStage.LIVE_ZENODO,
                argv=argv,
                receipt_path=receipt_path,
                cwd=project_root,
                live=True,
            )
        )
    return tuple(_confine_command_receipt(config, command) for command in commands)


def _live_gate_specs(config: PublicReleaseConfig) -> list[tuple[PublicReleaseStage, bool, str, str | None]]:
    return [
        (PublicReleaseStage.LIVE_GITHUB, config.live_github, "ack_live_github", config.github_repo),
        (PublicReleaseStage.LIVE_R2, config.live_r2, "ack_live_r2", config.r2_bucket),
        (PublicReleaseStage.LIVE_PAGES, config.live_pages, "ack_live_pages", config.pages_project),
        (PublicReleaseStage.LIVE_HF, config.live_hf, "ack_live_hf", config.hf_repo_id),
        (PublicReleaseStage.LIVE_ZENODO, config.live_zenodo, "ack_live_zenodo", "zenodo"),
    ]


def _load_export_script(project_root: Path | None = None):
    # Always this code's own script; project_root names where the data lives, not the code.
    del project_root
    script_path = resolve_project_root(None) / "scripts" / "export_for_public.py"
    spec = importlib.util.spec_from_file_location("_public_release_export_for_public", script_path)
    if spec is None or spec.loader is None:
        raise PublicReleaseError(f"cannot load export script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sorted_artifact_contract(artifacts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(artifact) for artifact in artifacts),
        key=lambda artifact: (
            str(artifact.get("family_id", "")),
            str(artifact.get("path", "")),
            str(artifact.get("delivery", "")),
            str(artifact.get("cdn_url", "")),
        ),
    )


def _current_export_contract(config: PublicReleaseConfig) -> dict[str, Any]:
    project_root, conference_dir, bundle_id, _receipt_dir, _bundle_root = _base_context(config)
    export_module = _load_export_script(project_root)
    export_config = export_module.ExportConfig(
        conference_dir=conference_dir,
        export_manifest=conference_dir / "storage" / "public_export_manifest.json",
        output_dir=conference_dir / "exports" / "public-release",
        bundle_id=bundle_id,
        dry_run=True,
        receipt_path=None,
        reviewed_dry_run_receipt=None,
        ack_caution_families=config.ack_caution_families,
        source_sha=None,
        orchestrator_stage=None,
    )
    plan = export_module.build_export_plan(export_config)
    return {
        "conference": plan.conference,
        "bundle_id": plan.bundle_id,
        "source_sha": plan.source_sha,
        "export_manifest_sha256": plan.manifest_sha256,
        "output_dir": str(plan.config.output_dir),
        "planned_artifacts": export_module.planned_artifacts(plan),
        "cdn_stubs": export_module.cdn_stub_artifacts(plan),
    }


_CHILD_RECEIPT_TYPES: dict[PublicReleaseStage, str] = {
    PublicReleaseStage.VALIDATE_MANIFEST: "conference_pipeline.public_export",
    PublicReleaseStage.EXPORT_DRY_RUN: "conference_pipeline.public_export",
    PublicReleaseStage.EXPORT_WRITE: "conference_pipeline.public_export",
    PublicReleaseStage.BUILD_INDEX: "conference_pipeline.public_index_build",
    PublicReleaseStage.GENERATE_SITE: "conference_pipeline.public_site",
    PublicReleaseStage.STAGE_REPO: "conference_pipeline.public_repo_push",
    PublicReleaseStage.PACKAGE_DATASET: "conference_pipeline.public_dataset_package",
    PublicReleaseStage.PREFLIGHT_LIVE: "conference_pipeline.public_deployment_preflight",
    PublicReleaseStage.CHECK_LIVENESS: "conference_pipeline.public_url_liveness",
    PublicReleaseStage.CHECK_METADATA: "conference_pipeline.public_release_metadata_agreement",
    PublicReleaseStage.LIVE_GITHUB: "conference_pipeline.public_repo_push",
    PublicReleaseStage.LIVE_R2: "conference_pipeline.public_cdn_upload",
    PublicReleaseStage.LIVE_PAGES: "conference_pipeline.cloudflare_pages_deploy",
    PublicReleaseStage.LIVE_HF: "conference_pipeline.hf_dataset_push",
    PublicReleaseStage.LIVE_ZENODO: "conference_pipeline.zenodo_deposit",
}

_CHILD_REQUIRED_FIELDS: dict[PublicReleaseStage, tuple[str, ...]] = {
    PublicReleaseStage.BUILD_INDEX: ("source_sha", "chunk_count", "output_checksums"),
    PublicReleaseStage.GENERATE_SITE: ("output_dir", "output_checksums"),
    PublicReleaseStage.STAGE_REPO: ("repo_name", "public_safety", "secret_gate"),
    PublicReleaseStage.PACKAGE_DATASET: ("resource_count", "metadata_dir"),
    PublicReleaseStage.PREFLIGHT_LIVE: ("public_repo_safety", "credentials", "tools"),
    PublicReleaseStage.CHECK_LIVENESS: ("checked_count", "failed_count", "results"),
    PublicReleaseStage.CHECK_METADATA: ("checked_fields", "mismatch_count", "issues"),
}


def child_receipt_sha256(path: Path) -> str:
    """Hash the exact child receipt bytes used for contract validation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _child_receipt_failure(reason: str, *, label: str) -> PublicReleaseError:
    return PublicReleaseError(
        "child_receipt_contract_failed",
        code=f"child_receipt:{reason}",
        label=label,
    )


def _validate_export_child_contract(
    config: PublicReleaseConfig,
    command: PublicReleaseCommand,
    payload: dict[str, Any],
) -> None:
    expected = _current_export_contract(config)
    expected_dry_run = command.stage is not PublicReleaseStage.EXPORT_WRITE
    if payload.get("orchestrator_stage") != command.stage.value:
        raise _child_receipt_failure("orchestrator_stage", label=command.stage.value)
    if payload.get("dry_run") is not expected_dry_run:
        raise _child_receipt_failure("dry_run", label=command.stage.value)
    for key in ("source_sha", "export_manifest_sha256", "output_dir"):
        if payload.get(key) != expected[key]:
            raise _child_receipt_failure(key, label=command.stage.value)
    if _sorted_artifact_contract(payload.get("planned_artifacts", [])) != _sorted_artifact_contract(
        expected["planned_artifacts"]
    ):
        raise _child_receipt_failure("planned_artifacts", label=command.stage.value)
    if _sorted_artifact_contract(payload.get("cdn_stubs", [])) != _sorted_artifact_contract(
        expected["cdn_stubs"]
    ):
        raise _child_receipt_failure("cdn_stubs", label=command.stage.value)


def validate_child_receipt(
    config_or_command: PublicReleaseConfig | PublicReleaseCommand,
    command_or_config: PublicReleaseCommand | PublicReleaseConfig,
) -> dict[str, Any]:
    """Validate and project one confined, regular child receipt.

    Both argument orders are accepted to keep this seam convenient for callers
    that naturally start from either the command or the release configuration.
    The returned projection is deliberately bounded and is safe to persist in
    aggregate/journal evidence.
    """
    if isinstance(config_or_command, PublicReleaseConfig):
        config = config_or_command
        command = command_or_config
    else:
        command = config_or_command
        config = command_or_config
    if not isinstance(command, PublicReleaseCommand) or not isinstance(
        config, PublicReleaseConfig
    ):
        raise TypeError("validate_child_receipt expects a PublicReleaseConfig and PublicReleaseCommand")

    label = f"{command.stage.value}_receipt_path"
    try:
        path = _confine_receipt_path(config, command.receipt_path, label=label, must_exist=True)
    except ReceiptPathError as exc:
        raise _child_receipt_failure(exc.code, label=label) from None
    if path.is_symlink() or not path.is_file():
        raise _child_receipt_failure("receipt_not_regular", label=label)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise _child_receipt_failure("unreadable", label=label) from None
    if not isinstance(payload, dict):
        raise _child_receipt_failure("not_object", label=label)
    expected_type = _CHILD_RECEIPT_TYPES[command.stage]
    if payload.get("receipt_type") != expected_type:
        raise _child_receipt_failure("receipt_type", label=label)
    if payload.get("status") != "succeeded":
        raise _child_receipt_failure("status", label=label)
    if payload.get("exit_code") != 0:
        raise _child_receipt_failure("exit_code", label=label)
    _project_root, conference_dir, bundle_id, _receipt_dir, _bundle_root = _base_context(config)
    if payload.get("conference") not in (None, conference_dir.name):
        raise _child_receipt_failure("conference", label=label)
    if payload.get("bundle_id") not in (None, bundle_id):
        raise _child_receipt_failure("bundle_id", label=label)
    if command.stage in {
        PublicReleaseStage.VALIDATE_MANIFEST,
        PublicReleaseStage.EXPORT_DRY_RUN,
        PublicReleaseStage.EXPORT_WRITE,
    }:
        _validate_export_child_contract(config, command, payload)
    for field in _CHILD_REQUIRED_FIELDS.get(command.stage, ()):
        if field not in payload:
            raise _child_receipt_failure(f"missing_{field}", label=label)
    return {
        "stage": command.stage.value,
        "receipt_path": path.as_posix(),
        "receipt_type": expected_type,
        "conference": conference_dir.name,
        "bundle_id": bundle_id,
        "status": "succeeded",
        "exit_code": 0,
        "contract_result": "passed",
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _reviewed_export_receipt_errors(config: PublicReleaseConfig) -> list[str]:
    if config.reviewed_dry_run_receipt is None:
        return ["reviewed_dry_run_receipt"]
    try:
        path = _confine_receipt_path(
            config,
            config.reviewed_dry_run_receipt,
            label="reviewed_dry_run_receipt",
            must_exist=True,
        )
    except ReceiptPathError as exc:
        return [f"reviewed_dry_run_receipt:{exc.code}"]
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ["reviewed_dry_run_receipt:unreadable"]
    try:
        current = _current_export_contract(config)
    except Exception as exc:
        return [
            stable_receipt_exception_code(
                "reviewed_dry_run_receipt",
                "current_plan",
                exc,
            )
        ]
    errors: list[str] = []
    expected = {
        "receipt_type": "conference_pipeline.public_export",
        "status": "succeeded",
        "exit_code": 0,
        "dry_run": True,
        "conference": current["conference"],
        "bundle_id": current["bundle_id"],
        "source_sha": current["source_sha"],
        "export_manifest_sha256": current["export_manifest_sha256"],
        "output_dir": current["output_dir"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            errors.append(f"reviewed_dry_run_receipt:{key}")
    if not current["planned_artifacts"]:
        errors.append("reviewed_dry_run_receipt:current_plan_empty")
    elif _sorted_artifact_contract(receipt.get("planned_artifacts", [])) != _sorted_artifact_contract(
        current["planned_artifacts"]
    ):
        errors.append("reviewed_dry_run_receipt:planned_artifacts")
    if _sorted_artifact_contract(receipt.get("cdn_stubs", [])) != _sorted_artifact_contract(
        current["cdn_stubs"]
    ):
        errors.append("reviewed_dry_run_receipt:cdn_stubs")
    return errors


def _public_repo_receipt_errors(config: PublicReleaseConfig) -> list[str]:
    if config.public_repo_dry_run_receipt is None:
        return ["public_repo_dry_run_receipt"]
    _project_root, conference_dir, bundle_id, _receipt_dir, _bundle_root = _base_context(config)
    try:
        path = _confine_receipt_path(
            config,
            config.public_repo_dry_run_receipt,
            label="public_repo_dry_run_receipt",
            must_exist=True,
        )
    except ReceiptPathError as exc:
        return [f"public_repo_dry_run_receipt:{exc.code}"]
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ["public_repo_dry_run_receipt:unreadable"]
    errors: list[str] = []
    expected = {
        "receipt_type": "conference_pipeline.public_repo_push",
        "status": "succeeded",
        "exit_code": 0,
        "dry_run": True,
        "repo_name": repo_name_from_github(config, conference_dir.name),
        "bundle_id": bundle_id,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            errors.append(f"public_repo_dry_run_receipt:{key}")
    public_safety = receipt.get("public_safety")
    if isinstance(public_safety, dict) and public_safety.get("status") != "clean":
        errors.append("public_repo_dry_run_receipt:public_safety")
    elif public_safety is None:
        errors.append("public_repo_dry_run_receipt:public_safety")
    secret_gate = receipt.get("secret_gate")
    if (
        not isinstance(secret_gate, dict)
        or secret_gate.get("gate_id") != "secrets"
        or secret_gate.get("status") != "passed"
        or secret_gate.get("findings") != []
    ):
        errors.append("public_repo_dry_run_receipt:secret_gate")
    output_checksums = receipt.get("output_checksums")
    if not isinstance(output_checksums, dict) or not output_checksums:
        errors.append("public_repo_dry_run_receipt:output_checksums")
    try:
        current = _current_public_repo_contract(config)
    except Exception as exc:
        errors.append(
            stable_receipt_exception_code(
                "public_repo_dry_run_receipt",
                "current_stage",
                exc,
            )
        )
        return errors
    if receipt.get("source_sha") != current["source_sha"]:
        errors.append("public_repo_dry_run_receipt:source_sha")
    if output_checksums != current["output_checksums"]:
        try:
            from conference_pipeline.publishing.public_repo import reviewable_checksums

            reviewed_checksums = reviewable_checksums(
                {str(path): str(digest) for path, digest in output_checksums.items()}
            )
        except Exception:
            reviewed_checksums = output_checksums
        if reviewed_checksums != current["output_checksums"]:
            errors.append("public_repo_dry_run_receipt:output_checksums")
    return errors


def _current_public_repo_contract(config: PublicReleaseConfig) -> dict[str, Any]:
    from conference_pipeline.publishing.public_repo import (
        PublicRepoConfig,
        reviewable_checksums,
        stage_public_repo,
    )

    project_root, conference_dir, bundle_id, _receipt_dir, bundle_root = _base_context(config)
    repo_name = repo_name_from_github(config, conference_dir.name)
    with tempfile.TemporaryDirectory(prefix="public-release-repo-gate-") as tmp:
        temp_root = Path(tmp)
        result = stage_public_repo(
            PublicRepoConfig(
                bundle_root=bundle_root,
                target_dir=temp_root / "public-repo",
                repo_name=repo_name,
                dry_run=True,
                receipt_path=temp_root / "public-repo-dry-run.json",
                source_repo_root=project_root,
                secret_scanner=config.public_repo_secret_scanner,
            )
        )
        release = json.loads((bundle_root / "RELEASE.json").read_text(encoding="utf-8"))
        return {
            "bundle_id": bundle_id,
            "source_sha": release.get("source_sha"),
            "repo_name": repo_name,
            "output_checksums": reviewable_checksums(result.output_checksums),
        }


def _live_gate_blockers(config: PublicReleaseConfig) -> tuple[dict[str, Any], ...]:
    blocked: list[dict[str, Any]] = []
    ack_values = {
        "ack_live_github": config.ack_live_github,
        "ack_live_r2": config.ack_live_r2,
        "ack_live_pages": config.ack_live_pages,
        "ack_live_hf": config.ack_live_hf,
        "ack_live_zenodo": config.ack_live_zenodo,
    }
    for stage, enabled, ack_name, target in _live_gate_specs(config):
        if not enabled:
            continue
        missing_approvals: list[str] = []
        missing_receipts: list[str] = []
        missing_targets: list[str] = []
        if not ack_values[ack_name]:
            missing_approvals.append(ack_name)
        if config.dry_run:
            # A dry run never publishes; a live stage beside it is a misuse, refused here
            # rather than scheduled (phase-7.1-r2 F2).
            missing_approvals.append("not_dry_run")
        if stage is PublicReleaseStage.LIVE_GITHUB:
            missing_receipts.extend(_public_repo_receipt_errors(config))
        else:
            missing_receipts.extend(_reviewed_export_receipt_errors(config))
        if not target:
            missing_targets.append(f"{stage.value}_target")
        # R2 needed a verified live path before it could be enabled; v6.0 Phase 7.1 tests
        # the upload end to end against a fake S3 endpoint (test_release_publish_paths.py).
        # It also needs the owner's named approval, whichever entry point runs it
        # (phase-7.1-r1 F3).
        if stage is PublicReleaseStage.LIVE_R2 and not _named_owner_approval(config.owner_approval):
            missing_approvals.append("owner_approval")
        if missing_approvals or missing_receipts or missing_targets:
            blocked.append(
                {
                    "stage": stage.value,
                    "missing_approvals": missing_approvals,
                    "missing_receipts": missing_receipts,
                    "missing_targets": missing_targets,
                }
            )
    return tuple(blocked)


def _write_gate_blockers(config: PublicReleaseConfig) -> tuple[dict[str, Any], ...]:
    if not config.write_bundle or export_review_boundary_blocker(config) is not None:
        return ()
    missing_receipts = _reviewed_export_receipt_errors(config)
    if not missing_receipts:
        return ()
    return (
        {
            "stage": PublicReleaseStage.EXPORT_WRITE.value,
            "missing_approvals": [],
            "missing_receipts": missing_receipts,
            "missing_targets": [],
        },
    )


def _output_receipt_context(
    config: PublicReleaseConfig,
) -> tuple[Path, Path, tuple[dict[str, Any], ...]]:
    default_path = default_public_release_receipt_path(config)
    try:
        safe_default = _confine_receipt_path(
            config,
            default_path,
            label="receipt_path",
        )
    except ReceiptPathError as exc:
        raise _public_release_error(exc) from exc

    selected_top = safe_default
    blockers: list[dict[str, Any]] = []
    if config.receipt_path is not None:
        try:
            selected_top = _confine_receipt_path(
                config,
                config.receipt_path,
                label="receipt_path",
            )
        except ReceiptPathError as exc:
            blockers.append(_path_error_blocker(exc))
    if config.export_write_receipt_path is not None:
        try:
            _confine_receipt_path(
                config,
                config.export_write_receipt_path,
                label="export_write_receipt_path",
            )
        except ReceiptPathError as exc:
            blockers.append(_path_error_blocker(exc))
    return selected_top, safe_default, tuple(blockers)


def _receipt_alias_blocker(
    config: PublicReleaseConfig,
    receipt_path: Path,
    commands: Sequence[PublicReleaseCommand],
) -> dict[str, Any] | None:
    entries: list[tuple[str, Path]] = [
        ("receipt_path", receipt_path),
        ("journal_path", public_release_journal_path(receipt_path)),
    ]
    entries.extend(
        (f"{command.stage.value}_receipt_path", command.receipt_path)
        for command in commands
    )
    read_candidates = (
        ("reviewed_dry_run_receipt", config.reviewed_dry_run_receipt),
        ("public_repo_dry_run_receipt", config.public_repo_dry_run_receipt),
    )
    for label, candidate in read_candidates:
        if candidate is None:
            continue
        try:
            path = _confine_receipt_path(
                config,
                candidate,
                label=label,
                must_exist=True,
            )
        except ReceiptPathError:
            continue
        entries.append((label, path))

    labels_by_path: dict[Path, list[str]] = {}
    for label, path in entries:
        labels_by_path.setdefault(path, []).append(label)
    aliases = [sorted(labels) for labels in labels_by_path.values() if len(labels) > 1]
    if not aliases:
        return None
    return {
        "stage": "receipt_paths",
        "reason": "receipt_path_alias",
        "aliases": sorted(aliases),
        "missing_approvals": [],
        "missing_receipts": [],
        "missing_targets": [],
    }


def _preflight_gate_blockers(config: PublicReleaseConfig) -> tuple[dict[str, Any], ...]:
    if not config.preflight_live:
        return ()
    _project_root, _conference_dir, bundle_id, receipt_dir, _bundle_root = _base_context(config)
    required_receipts = {
        "public_repo_dry_run_receipt": _receipt(receipt_dir, "public-repo-{bundle}-dry-run-receipt.json", bundle_id),
        "site_receipt": _receipt(receipt_dir, "public-site-{bundle}-receipt.json", bundle_id),
        "dataset_receipt": _receipt(receipt_dir, "public-dataset-{bundle}-receipt.json", bundle_id),
        "r2_upload_receipt": _receipt(receipt_dir, "public-cdn-{bundle}-r2-live-upload-receipt.json", bundle_id),
        "r2_liveness_receipt": _receipt(receipt_dir, "public-url-liveness-{bundle}-r2-live-receipt.json", bundle_id),
    }
    missing_targets = []
    if not config.github_repo:
        missing_targets.append("github_repo")
    if not config.r2_base_url:
        missing_targets.append("r2_base_url")
    missing_receipts = [name for name, path in required_receipts.items() if not path.is_file()]
    if not missing_targets and not missing_receipts:
        return ()
    return (
        {
            "stage": PublicReleaseStage.PREFLIGHT_LIVE.value,
            "missing_approvals": [],
            "missing_receipts": missing_receipts,
            "missing_targets": missing_targets,
        },
    )


def _env_names_from_file(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").strip()
        if "=" in stripped:
            names.append(stripped.split("=", 1)[0].strip())
    return tuple(sorted(set(name for name in names if name)))


def _named_owner_approval(approval: Any) -> bool:
    return isinstance(approval, Mapping) and isinstance(approval.get("approved_by"), str) and bool(approval["approved_by"].strip())


def _load_env_file(path: Path, base: Mapping[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    env = dict(base)
    loaded: list[str] = []
    if not path.is_file():
        return env, ()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").strip()
        if "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not name:
            continue
        env[name] = value.strip().strip("'\"")
        loaded.append(name)
    return env, tuple(sorted(set(loaded)))


def default_public_release_receipt_path(config: PublicReleaseConfig) -> Path:
    _project_root, conference_dir, bundle_id, receipt_dir, _bundle_root = _base_context(config)
    return receipt_dir / f"public-release-{bundle_id}-receipt.json"


def public_release_journal_path(receipt_path: Path) -> Path:
    """Return the deterministic adjacent journal for one top-level receipt."""
    return receipt_path.with_name(f"{receipt_path.stem}.journal.json")


def write_public_release_journal(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a journal transition through the shared hardened JSON writer."""
    write_json_atomic_secure(path, payload)
    return path


def _journal_payload(
    *,
    config: PublicReleaseConfig,
    status: str,
    receipt_path: Path,
    commands: Sequence[PublicReleaseCommand],
    command_results: Sequence[dict[str, Any]],
    failure_code: str | None = None,
    exception_type: str | None = None,
    terminal_status: str | None = None,
    terminal_exit_code: int | None = None,
    aggregate_path: Path | None = None,
    aggregate_sha256: str | None = None,
    reconciliation: str | None = None,
) -> dict[str, Any]:
    _project_root, conference_dir, bundle_id, _receipt_dir, _bundle_root = _base_context(config)
    return {
        "schema_version": 1,
        "receipt_type": PUBLIC_RELEASE_JOURNAL_RECEIPT_TYPE,
        "status": status,
        "terminal_status": terminal_status,
        "terminal_exit_code": terminal_exit_code,
        "aggregate_path": aggregate_path.as_posix() if aggregate_path is not None else None,
        "aggregate_sha256": aggregate_sha256,
        "reconciliation": reconciliation,
        "conference": conference_dir.name,
        "bundle_id": bundle_id,
        "top_level_receipt_path": receipt_path.as_posix(),
        "planned_commands": [
            {
                "stage": command.stage.value,
                "receipt_path": command.receipt_path.as_posix(),
                "live": command.live,
                "shell": command.shell,
            }
            for command in commands
        ],
        "command_results": list(command_results),
        "last_completed_stage": (
            command_results[-1]["stage"] if command_results else None
        ),
        "failure_code": failure_code,
        "exception_type": exception_type,
        "secret_values_recorded": False,
        "updated_at": utc_now(),
    }


def _receipt_payload(
    *,
    config: PublicReleaseConfig,
    status: str,
    exit_code: int,
    commands: Sequence[PublicReleaseCommand],
    blocked_stages: Sequence[dict[str, Any]],
    command_results: Sequence[dict[str, Any]] = (),
    env_loaded_names: Sequence[str] = (),
    journal_path: Path | None = None,
) -> dict[str, Any]:
    _project_root, conference_dir, bundle_id, _receipt_dir, _bundle_root = _base_context(config)
    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "status": status,
        "exit_code": exit_code,
        "conference": conference_dir.name,
        "conference_dir": conference_dir.as_posix(),
        "bundle_id": bundle_id,
        "commands": [
            {
                "stage": command.stage.value,
                "argv": command.argv,
                "receipt_path": command.receipt_path.as_posix(),
                "live": command.live,
                "shell": command.shell,
            }
            for command in commands
        ],
        "child_receipt_paths": [command.receipt_path.as_posix() for command in commands],
        "command_results": list(command_results),
        "blocked_stages": list(blocked_stages),
        "env_file": config.env_file.as_posix(),
        "env_file_exists": config.env_file.is_file(),
        "env_loaded_for_live_stages": bool(env_loaded_names),
        "loaded_env_var_names": sorted(env_loaded_names),
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    if journal_path is not None:
        payload["journal_path"] = journal_path.as_posix()
    if config.owner_approval is not None:
        payload["owner_approval"] = dict(config.owner_approval)
    return payload


def write_public_release_receipt(path: Path, payload: dict[str, Any]) -> Path:
    write_json(path, payload)
    return path


def run_public_release(
    config: PublicReleaseConfig,
    *,
    runner: Runner = subprocess.run,
) -> PublicReleaseResult:
    receipt_path, safe_default_receipt, output_blockers = _output_receipt_context(
        config
    )
    if output_blockers:
        payload = _receipt_payload(
            config=config,
            status="operator_blocked",
            exit_code=2,
            commands=(),
            blocked_stages=output_blockers,
        )
        write_public_release_receipt(safe_default_receipt, payload)
        return PublicReleaseResult(
            status="operator_blocked",
            exit_code=2,
            commands=(),
            blocked_stages=output_blockers,
            receipt_path=safe_default_receipt,
            secret_values_recorded=False,
        )

    review_boundary = export_review_boundary_blocker(config)
    blocked_stages = (
        *((review_boundary,) if review_boundary is not None else ()),
        *_write_gate_blockers(config),
        *_preflight_gate_blockers(config),
        *_live_gate_blockers(config),
    )
    if blocked_stages:
        payload = _receipt_payload(
            config=config,
            status="operator_blocked",
            exit_code=2,
            commands=(),
            blocked_stages=blocked_stages,
        )
        write_public_release_receipt(receipt_path, payload)
        return PublicReleaseResult(
            status="operator_blocked",
            exit_code=2,
            commands=(),
            blocked_stages=blocked_stages,
            receipt_path=receipt_path,
            secret_values_recorded=False,
        )

    try:
        commands = build_public_release_plan(config)
    except PublicReleaseError as exc:
        if exc.code is None or exc.label is None:
            raise
        blocker = {
            "stage": "receipt_paths",
            "reason": "unsafe_receipt_path",
            "label": exc.label,
            "reason_code": exc.code,
            "missing_approvals": [],
            "missing_receipts": [],
            "missing_targets": [],
        }
        payload = _receipt_payload(
            config=config,
            status="operator_blocked",
            exit_code=2,
            commands=(),
            blocked_stages=(blocker,),
        )
        write_public_release_receipt(safe_default_receipt, payload)
        return PublicReleaseResult(
            status="operator_blocked",
            exit_code=2,
            commands=(),
            blocked_stages=(blocker,),
            receipt_path=safe_default_receipt,
            secret_values_recorded=False,
        )

    journal_candidate = public_release_journal_path(receipt_path)
    try:
        journal_path = _confine_receipt_path(
            config,
            journal_candidate,
            label="journal_path",
        )
    except ReceiptPathError as exc:
        blocker = _path_error_blocker(exc)
        payload = _receipt_payload(
            config=config,
            status="operator_blocked",
            exit_code=2,
            commands=commands,
            blocked_stages=(blocker,),
        )
        write_public_release_receipt(safe_default_receipt, payload)
        return PublicReleaseResult(
            status="operator_blocked",
            exit_code=2,
            commands=commands,
            blocked_stages=(blocker,),
            receipt_path=safe_default_receipt,
            secret_values_recorded=False,
        )

    alias_blocker = _receipt_alias_blocker(config, receipt_path, commands)
    if alias_blocker is not None:
        payload = _receipt_payload(
            config=config,
            status="operator_blocked",
            exit_code=2,
            commands=commands,
            blocked_stages=(alias_blocker,),
        )
        write_public_release_receipt(safe_default_receipt, payload)
        return PublicReleaseResult(
            status="operator_blocked",
            exit_code=2,
            commands=commands,
            blocked_stages=(alias_blocker,),
            receipt_path=safe_default_receipt,
            secret_values_recorded=False,
        )
    live_env: dict[str, str] | None = None
    loaded_env_names: tuple[str, ...] = ()
    if any(command.live for command in commands):
        live_env, loaded_env_names = _load_env_file(config.env_file, os.environ)

    command_results: list[dict[str, Any]] = []
    status = "succeeded"
    exit_code = 0
    try:
        write_public_release_journal(
            journal_path,
            _journal_payload(
                config=config,
                status="running",
                receipt_path=receipt_path,
                commands=commands,
                command_results=command_results,
            ),
        )
    except Exception:
        raise PublicReleaseError(
            "public_release_journal_persistence_failed",
            code="public_release_journal:initial:persistence_failed",
            label="journal_path",
        ) from None

    for command in commands:
        try:
            completed = runner(
                command.argv,
                cwd=command.cwd,
                env=live_env if command.live else None,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            status = "failed"
            exit_code = 1
            command_results.append(
                {
                    "stage": command.stage.value,
                    "receipt_path": command.receipt_path.as_posix(),
                    "status": "failed",
                    "exit_code": 1,
                    "contract_result": "failed",
                    "failure_code": stable_receipt_exception_code(
                        "public_release", command.stage.value, exc
                    ),
                    "exception_type": _stable_exception_type(exc),
                }
            )
            try:
                write_public_release_journal(
                    journal_path,
                    _journal_payload(
                        config=config,
                        status="failed",
                        receipt_path=receipt_path,
                        commands=commands,
                        command_results=command_results,
                        failure_code="child_runner_failed",
                        exception_type=_stable_exception_type(exc),
                    ),
                )
            except Exception:
                pass
            break

        if completed.returncode != 0:
            status = "operator_blocked" if completed.returncode == 2 and command.live else "failed"
            exit_code = int(completed.returncode)
            command_results.append(
                {
                    "stage": command.stage.value,
                    "receipt_path": command.receipt_path.as_posix(),
                    "status": status,
                    "exit_code": exit_code,
                    "contract_result": "not_run",
                    "failure_code": "child_nonzero_exit",
                }
            )
        else:
            try:
                command_results.append(validate_child_receipt(config, command))
            except (PublicReleaseError, OSError, ValueError, TypeError) as exc:
                status = "failed"
                exit_code = 1
                failure_code = getattr(exc, "code", None) or "child_receipt_contract_failed"
                command_results.append(
                    {
                        "stage": command.stage.value,
                        "receipt_path": command.receipt_path.as_posix(),
                        "status": "failed",
                        "exit_code": 1,
                        "contract_result": "failed",
                        "failure_code": str(failure_code),
                        "exception_type": _stable_exception_type(exc),
                    }
                )

        try:
            write_public_release_journal(
                journal_path,
                _journal_payload(
                    config=config,
                    status="failed" if status == "failed" else "running",
                    receipt_path=receipt_path,
                    commands=commands,
                    command_results=command_results,
                    failure_code=(
                        command_results[-1].get("failure_code")
                        if status == "failed"
                        else None
                    ),
                ),
            )
        except Exception:
            raise PublicReleaseError(
                "public_release_journal_persistence_failed",
                code="public_release_journal:child_result:persistence_failed",
                label="journal_path",
            ) from None
        if status != "succeeded":
            break

    payload = _receipt_payload(
        config=config,
        status=status,
        exit_code=exit_code,
        commands=commands,
        blocked_stages=(),
        command_results=command_results,
        env_loaded_names=loaded_env_names,
        journal_path=journal_path,
    )
    try:
        write_public_release_journal(
            journal_path,
            _journal_payload(
                config=config,
                status="finalizing",
                receipt_path=receipt_path,
                commands=commands,
                command_results=command_results,
                terminal_status=status,
                terminal_exit_code=exit_code,
                aggregate_path=receipt_path,
            ),
        )
        write_public_release_receipt(receipt_path, payload)
        aggregate_sha = child_receipt_sha256(receipt_path)
        write_public_release_journal(
            journal_path,
            _journal_payload(
                config=config,
                status="finalized",
                receipt_path=receipt_path,
                commands=commands,
                command_results=command_results,
                terminal_status=status,
                terminal_exit_code=exit_code,
                aggregate_path=receipt_path,
                aggregate_sha256=aggregate_sha,
            ),
        )
    except Exception as exc:
        failure_code = "top_level_receipt_persistence_failed"
        failed_payload = _receipt_payload(
            config=config,
            status="failed",
            exit_code=1,
            commands=commands,
            blocked_stages=(),
            command_results=command_results,
            env_loaded_names=loaded_env_names,
            journal_path=journal_path,
        )
        try:
            write_public_release_receipt(receipt_path, failed_payload)
        except Exception:
            pass
        try:
            failed_sha = child_receipt_sha256(receipt_path) if receipt_path.is_file() else None
            write_public_release_journal(
                journal_path,
                _journal_payload(
                    config=config,
                    status="failed",
                    receipt_path=receipt_path,
                    commands=commands,
                    command_results=command_results,
                    failure_code=failure_code,
                    exception_type=_stable_exception_type(exc),
                    terminal_status="failed",
                    terminal_exit_code=1,
                    aggregate_path=receipt_path,
                    aggregate_sha256=failed_sha,
                    reconciliation="failed",
                ),
            )
        except Exception:
            pass
        return PublicReleaseResult(
            status="failed",
            exit_code=1,
            commands=commands,
            blocked_stages=(),
            receipt_path=receipt_path,
            journal_path=journal_path,
            secret_values_recorded=False,
        )
    return PublicReleaseResult(
        status=status,
        exit_code=exit_code,
        commands=commands,
        blocked_stages=(),
        receipt_path=receipt_path,
        journal_path=journal_path,
        secret_values_recorded=False,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reusable conference public-release stages")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--conference-dir", type=Path)
    target.add_argument("--conference-slug")
    parser.add_argument("--bundle-id")
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-bundle", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--generate-site", action="store_true")
    parser.add_argument("--stage-repo", action="store_true")
    parser.add_argument("--package-dataset", action="store_true")
    parser.add_argument("--preflight-live", action="store_true")
    parser.add_argument("--check-liveness", action="store_true")
    parser.add_argument("--check-metadata", action="store_true")
    parser.add_argument("--live-github", action="store_true")
    parser.add_argument("--live-r2", action="store_true")
    parser.add_argument("--live-pages", action="store_true")
    parser.add_argument("--live-hf", action="store_true")
    parser.add_argument("--live-zenodo", action="store_true")
    parser.add_argument("--reviewed-dry-run-receipt", type=Path)
    parser.add_argument("--export-write-receipt-path", type=Path)
    parser.add_argument("--public-repo-dry-run-receipt", type=Path)
    parser.add_argument("--ack-caution-family", action="append", default=[])
    parser.add_argument("--ack-live-github", action="store_true")
    parser.add_argument("--ack-live-r2", action="store_true")
    parser.add_argument("--ack-live-pages", action="store_true")
    parser.add_argument("--ack-live-hf", action="store_true")
    parser.add_argument("--ack-live-zenodo", action="store_true")
    parser.add_argument("--github-repo")
    parser.add_argument("--r2-base-url")
    parser.add_argument("--r2-bucket")
    parser.add_argument("--r2-prefix")
    parser.add_argument("--pages-project")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--zenodo-sandbox", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PublicReleaseConfig:
    return PublicReleaseConfig(
        conference_dir=args.conference_dir,
        conference_slug=args.conference_slug,
        bundle_id=args.bundle_id,
        receipt_path=args.receipt_path,
        env_file=args.env_file,
        validate_manifest=args.validate_manifest,
        dry_run=args.dry_run,
        write_bundle=args.write_bundle,
        build_index=args.build_index,
        generate_site=args.generate_site,
        stage_repo=args.stage_repo,
        package_dataset=args.package_dataset,
        preflight_live=args.preflight_live,
        check_liveness=args.check_liveness,
        check_metadata=args.check_metadata,
        live_github=args.live_github,
        live_r2=args.live_r2,
        live_pages=args.live_pages,
        live_hf=args.live_hf,
        live_zenodo=args.live_zenodo,
        reviewed_dry_run_receipt=args.reviewed_dry_run_receipt,
        export_write_receipt_path=args.export_write_receipt_path,
        public_repo_dry_run_receipt=args.public_repo_dry_run_receipt,
        ack_caution_families=tuple(args.ack_caution_family),
        ack_live_github=args.ack_live_github,
        ack_live_r2=args.ack_live_r2,
        ack_live_pages=args.ack_live_pages,
        ack_live_hf=args.ack_live_hf,
        ack_live_zenodo=args.ack_live_zenodo,
        github_repo=args.github_repo,
        r2_base_url=args.r2_base_url,
        r2_bucket=args.r2_bucket,
        r2_prefix=args.r2_prefix,
        pages_project=args.pages_project,
        hf_repo_id=args.hf_repo_id,
        zenodo_sandbox=args.zenodo_sandbox,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_public_release(config_from_args(args))
    except PublicReleaseError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Public release status: {result.status}")
    print(f"Receipt: {result.receipt_path}")
    if result.journal_path is not None:
        print(f"Journal: {result.journal_path}")
    return result.exit_code
