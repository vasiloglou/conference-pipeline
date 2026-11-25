"""``conference-pipeline release``: one command group for a conference's public release.

v6.0-Public-Release Phase 3 (spec ``notes/phase-3-spec.md`` section 6). The commands sit
over the existing orchestrator (``publishing.public_release``) and the export script
(``scripts/export_for_public.py``); they add no stage of their own.

Exit codes, for every command: 0 done; 1 ``check`` found a finding, or an isolated
build's digest differs from the expected one; 2 refused or blocked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import typer
import yaml

from conference_pipeline._security.private_markers import (
    MAX_FILE_BYTES,
    builtin_rules,
    extra_rules_from_env,
    has_publishing_marker_bytes,
    load_allowlist,
    publishing_literal_rules,
    redact_text,
    scan_paths,
    scan_text,
)
from conference_pipeline.publishing.public_release import (
    PublicReleaseConfig,
    PublicReleaseError,
    _load_export_script,
    default_bundle_id,
    default_bundle_root,
    default_receipt_dir,
    resolve_project_root,
    run_public_release,
)
from conference_pipeline.publishing.receipt_paths import write_json_atomic_secure
from conference_pipeline.publishing.release_layout import HF_GENERATED_FILES, ReleaseTargetError, hf_upload_paths, stamped_urls
from conference_pipeline.publishing.surface_limits import check_sized, check_tree
from conference_pipeline.publishing.release_manifest_draft import (
    refresh_public_export_manifest,
    ManifestDraftError,
    draft_public_export_manifest,
    public_export_manifest_path,
    resolve_source_sha,
)

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_REFUSED = 2

SURFACES = ("github", "r2", "pages", "hf", "zenodo")
MOCK_EMBEDDING_PROVIDER = "deterministic-hash"
PLAN_REPORT = "plan-report.json"
BUILD_REPORT = "build-report.json"

Runner = Callable[..., subprocess.CompletedProcess]
Echo = Callable[[str], None]


class ReleaseRefused(Exception):
    """A command refuses to run; the message names the blocker."""


def _echo(message: str) -> None:
    print(message)


def _err(message: str) -> None:
    print(message, file=sys.stderr)


# --- workspace -----------------------------------------------------------------------


def _workspace(project_root: Path, slug: str) -> Path:
    slug = slug.strip()
    if not slug or "/" in slug or "\\" in slug or slug.startswith("."):
        raise ReleaseRefused("the workspace slug must be a plain conferences/<slug> name")
    conference_dir = project_root / "conferences" / slug
    if not conference_dir.is_dir():
        raise ReleaseRefused(f"no workspace conferences/{slug}")
    return conference_dir


def release_settings(conference_dir: Path) -> dict[str, Any]:
    """The optional ``release:`` block of the workspace ``config.yaml`` (spec D7)."""
    path = conference_dir / "config.yaml"
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseRefused("config.yaml is not readable YAML") from exc
    block = payload.get("release") if isinstance(payload, dict) else None
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ReleaseRefused("config.yaml release: must be a mapping")
    # Every target must give a URL before any command uses it: build stamps them into
    # RELEASE.json (v6.0 Phase 7.1, spec 6.7).
    try:
        stamped_urls(block)
    except ReleaseTargetError as exc:
        raise ReleaseRefused(str(exc)) from exc
    return block


def _bundle_id(conference_dir: Path, bundle_id: str | None, settings: Mapping[str, Any]) -> str:
    return bundle_id or settings.get("bundle_id") or default_bundle_id(conference_dir)


def _dry_run_receipt(conference_dir: Path, bundle_id: str) -> Path:
    return default_receipt_dir(conference_dir) / f"public-export-{bundle_id}-dry-run-receipt.json"


def _write_receipt(conference_dir: Path, bundle_id: str) -> Path:
    return default_receipt_dir(conference_dir) / f"public-export-{bundle_id}-write-receipt.json"


def _repo_dry_run_receipt(conference_dir: Path, bundle_id: str) -> Path:
    return default_receipt_dir(conference_dir) / f"public-repo-{bundle_id}-dry-run-receipt.json"


def command_receipt(conference_dir: Path, bundle_id: str, command: str) -> Path:
    """One top-level receipt per release command, so no command overwrites another's (IF5)."""
    return default_receipt_dir(conference_dir) / f"public-release-{bundle_id}-{command}-receipt.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace ``path`` atomically; a symlink at ``path`` is replaced, never followed (F1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic_secure(path, payload)


def _report_result(result: Any, echo: Echo) -> int:
    echo(f"status: {result.status}; receipt: {result.receipt_path}")
    for blocked in result.blocked_stages:
        missing = [
            *blocked.get("missing_approvals", ()),
            *blocked.get("missing_receipts", ()),
            *blocked.get("missing_targets", ()),
        ]
        echo(f"blocked: {blocked.get('stage')}: {', '.join(missing) or blocked.get('reason', '')}")
    return EXIT_OK if result.exit_code == 0 else EXIT_REFUSED


def _run(config: PublicReleaseConfig, runner: Runner, echo: Echo) -> int:
    try:
        result = run_public_release(config, runner=runner)
    except PublicReleaseError as exc:
        raise ReleaseRefused(str(exc)) from exc
    return _report_result(result, echo)


def _guarded(body: Callable[[], int], err: Echo) -> int:
    try:
        return body()
    except ReleaseRefused as exc:
        # A refusal may name a path or bundle id; it is redacted like a finding (F5).
        err(redact_text(f"refused: {exc}", builtin_rules() + extra_rules_from_env()))
        return EXIT_REFUSED


# --- init ----------------------------------------------------------------------------


def release_init(
    slug: str,
    *,
    project_root: Path | None = None,
    draft_path: Path | None = None,
    source_sha: str | None = None,
    refresh: bool = False,
    echo: Echo = _echo,
    err: Echo = _err,
) -> int:
    """Draft ``storage/public_export_manifest.json`` with every family excluded.

    With ``refresh``, an existing manifest gains an ``excluded_families`` row for every
    storage family it does not mention yet; promoted families and existing rows are never
    changed, and nothing is promoted (v6.0 Phase 6, spec section 6.7).
    """

    def body() -> int:
        root = resolve_project_root(project_root)
        conference_dir = _workspace(root, slug)
        target = public_export_manifest_path(conference_dir)
        if refresh:
            if not target.is_file():
                raise ReleaseRefused(f"{target.relative_to(root)} does not exist; run init without --refresh")
            try:
                manifest = json.loads(target.read_text(encoding="utf-8"))
                added = refresh_public_export_manifest(conference_dir, manifest)
            except (ManifestDraftError, json.JSONDecodeError) as exc:
                raise ReleaseRefused(str(exc)) from exc
            _write_json(target, manifest)
            echo(f"refreshed {target.relative_to(root)}: {added} storage families added to excluded_families")
            return EXIT_OK
        if draft_path is not None:
            target = draft_path if draft_path.is_absolute() else root / draft_path
            try:
                target.resolve().relative_to(conference_dir.resolve())
            except ValueError as exc:
                raise ReleaseRefused("--force-draft-path must be inside the workspace") from exc
        if target.exists():
            raise ReleaseRefused(
                f"{target.relative_to(root)} exists; pass --force-draft-path to draft beside it"
            )
        try:
            draft = draft_public_export_manifest(
                conference_dir,
                source_sha=source_sha if source_sha is not None else resolve_source_sha(root),
            )
        except ManifestDraftError as exc:
            raise ReleaseRefused(str(exc)) from exc
        _write_json(target, draft)
        echo(
            f"drafted {target.relative_to(root)}: 0 families published, "
            f"{len(draft['excluded_families'])} excluded; promote a family to publish it"
        )
        return EXIT_OK

    return _guarded(body, err)


def release_families(
    slug: str,
    *,
    project_root: Path | None = None,
    bundle_id: str | None = None,
    echo: Echo = _echo,
    err: Echo = _err,
) -> int:
    """Build the reader families (catalog, statistics, narratives, limitations) for a bundle."""
    from conference_pipeline.publishing.reader_families import ReaderFamiliesError, build_reader_families

    def body() -> int:
        root = resolve_project_root(project_root)
        conference_dir = _workspace(root, slug)
        bundle = _bundle_id(conference_dir, bundle_id, release_settings(conference_dir))
        try:
            result = build_reader_families(conference_dir, bundle_id=bundle, project_root=root)
        except ReaderFamiliesError as exc:
            raise ReleaseRefused(str(exc)) from exc
        echo(
            f"reader families for {bundle}: {result.rows} catalog rows, {result.excluded} excluded ids, "
            f"{result.unresolved_topic_references} unresolved topic references, {len(result.files)} files"
        )
        for rel in result.excluded_pdfs:
            echo(f"excluded PDF (a marker survived regeneration, D8): {rel}")
        return EXIT_OK

    return _guarded(body, err)


# --- isolated mode (the reproduction path) -------------------------------------------


def _isolated_dir(conference_dir: Path, isolated: Path, *, must_be_empty: bool) -> Path:
    target = isolated.resolve()
    try:
        target.relative_to(conference_dir.resolve())
    except ValueError:
        pass
    else:
        raise ReleaseRefused("--isolated must be outside the workspace")
    if target.exists() and not target.is_dir():
        raise ReleaseRefused("--isolated must be a directory")
    if must_be_empty and target.exists() and any(target.iterdir()):
        raise ReleaseRefused("--isolated must be an empty or absent directory")
    return target


def _export_plan(conference_dir: Path, bundle_id: str, output_dir: Path, ack: Sequence[str]):
    # The export script is loaded from the running code, not from the data checkout:
    # --project-root may name another checkout whose scripts predate this code (IF10).
    export = _load_export_script(resolve_project_root(None))
    plan = export.build_export_plan(
        export.ExportConfig(
            conference_dir=conference_dir,
            export_manifest=public_export_manifest_path(conference_dir),
            output_dir=output_dir,
            bundle_id=bundle_id,
            dry_run=True,
            receipt_path=None,
            reviewed_dry_run_receipt=None,
            ack_caution_families=tuple(ack),
            source_sha=None,
        )
    )
    return export, plan


def _plan_isolated(conference_dir: Path, bundle_id: str, isolated: Path, ack: Sequence[str], echo: Echo) -> int:
    target = _isolated_dir(conference_dir, isolated, must_be_empty=True)
    try:
        export, plan = _export_plan(conference_dir, bundle_id, target, ack)
        export.run_plan_level_gate(plan)
        export.enforce_caution_ack(plan)
    except Exception as exc:  # the export script's own PublicExportBlocked and friends
        raise ReleaseRefused(f"isolated plan blocked: {exc}") from exc
    report = {
        "status": "succeeded",
        "dry_run": True,
        "bundle_id": plan.bundle_id,
        "conference": plan.conference,
        "source_sha": plan.source_sha,
        "export_manifest_sha256": plan.manifest_sha256,
        "ack_caution_families": sorted(ack),
        "planned_artifact_count": len(plan.items),
        "planned_artifacts": [
            {"path": item.release_path.as_posix(), "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in plan.items
        ],
    }
    _write_json(target / PLAN_REPORT, report)
    echo(f"isolated plan: {len(plan.items)} artifacts; report {target / PLAN_REPORT}")
    return EXIT_OK


def _reference_artifacts(conference_dir: Path, bundle_id: str) -> dict[str, dict[str, Any]] | None:
    receipt = _write_receipt(conference_dir, bundle_id)
    if not receipt.is_file():
        return None
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    return {row["path"]: row for row in payload.get("actual_artifacts", []) if isinstance(row, dict)}


def _build_isolated(
    conference_dir: Path,
    bundle_id: str,
    isolated: Path,
    expect_tree_sha256: str | None,
    ack: Sequence[str],
    echo: Echo,
) -> int:
    target = _isolated_dir(conference_dir, isolated, must_be_empty=False)
    report_path = target / PLAN_REPORT
    if not report_path.is_file():
        raise ReleaseRefused(f"no {PLAN_REPORT} in --isolated; run release plan --isolated first")
    planned = json.loads(report_path.read_text(encoding="utf-8"))
    bundle_dir = target / "bundle"
    if bundle_dir.exists():
        raise ReleaseRefused("--isolated already holds a bundle/")
    try:
        export, plan = _export_plan(conference_dir, bundle_id, target, ack)
    except Exception as exc:
        raise ReleaseRefused(f"isolated build blocked: {exc}") from exc
    identity = ("status", "bundle_id", "source_sha", "export_manifest_sha256", "ack_caution_families")
    current = {
        "status": "succeeded",
        "bundle_id": plan.bundle_id,
        "source_sha": plan.source_sha,
        "export_manifest_sha256": plan.manifest_sha256,
        "ack_caution_families": sorted(ack),
    }
    mismatched = [key for key in identity if planned.get(key) != current[key]]
    if mismatched:
        raise ReleaseRefused(f"{PLAN_REPORT} does not match the current plan: {', '.join(mismatched)}")
    try:
        built = export.materialize_isolated(plan, bundle_dir)
    except Exception as exc:
        raise ReleaseRefused(f"isolated build blocked: {exc}") from exc
    report: dict[str, Any] = {key: value for key, value in built.items() if key != "artifacts"}
    report["artifacts"] = built["artifacts"]
    exit_code = EXIT_OK
    if expect_tree_sha256 is not None:
        report["expected_tree_sha256"] = expect_tree_sha256
        report["tree_sha256_matches"] = built["bundle_tree_sha256"] == expect_tree_sha256
        reference = _reference_artifacts(conference_dir, bundle_id)
        # Always state whether paths could be compared (F7): without a write receipt for
        # this bundle id there is no per-file reference, only the expected digest.
        report["reference_receipt"] = None
        report["differing_paths"] = None
        if reference is not None:
            ours = {row["path"]: row for row in built["artifacts"]}
            differing = sorted(
                path
                for path in set(ours) | set(reference)
                if (ours.get(path) or {}).get("sha256") != (reference.get(path) or {}).get("sha256")
                or (ours.get(path) or {}).get("size_bytes") != (reference.get(path) or {}).get("size_bytes")
            )
            report["reference_receipt"] = _write_receipt(conference_dir, bundle_id).name
            report["differing_paths"] = differing
        if not report["tree_sha256_matches"]:
            exit_code = EXIT_FINDINGS
    _write_json(target / BUILD_REPORT, report)
    echo(
        f"isolated build: {built['file_count']} files, tree sha256 {built['bundle_tree_sha256']}"
        + ("" if expect_tree_sha256 is None else f"; matches expected: {report['tree_sha256_matches']}")
    )
    return exit_code


# --- plan / build ----------------------------------------------------------------------


def _require_promoted_manifest(conference_dir: Path) -> None:
    path = public_export_manifest_path(conference_dir)
    if not path.is_file():
        raise ReleaseRefused("no storage/public_export_manifest.json; run release init first")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseRefused("public_export_manifest.json is not readable JSON") from exc
    if not isinstance(manifest, dict) or not manifest.get("families"):
        raise ReleaseRefused("no_family_promoted: move a family from excluded_families into families")


def release_plan(
    slug: str,
    *,
    project_root: Path | None = None,
    bundle_id: str | None = None,
    ack_caution_families: Sequence[str] = (),
    isolated: Path | None = None,
    runner: Runner = subprocess.run,
    echo: Echo = _echo,
    err: Echo = _err,
) -> int:
    """Validate the manifest and dry-run the export; write no bundle."""

    def body() -> int:
        root = resolve_project_root(project_root)
        conference_dir = _workspace(root, slug)
        settings = release_settings(conference_dir)
        bundle = _bundle_id(conference_dir, bundle_id, settings)
        _require_promoted_manifest(conference_dir)
        if isolated is not None:
            return _plan_isolated(conference_dir, bundle, isolated, ack_caution_families, echo)
        config = PublicReleaseConfig(
            project_root=root,
            conference_slug=conference_dir.name,
            bundle_id=bundle,
            receipt_path=command_receipt(conference_dir, bundle, "plan"),
            validate_manifest=True,
            dry_run=True,
            ack_caution_families=tuple(ack_caution_families),
        )
        return _run(config, runner, echo)

    return _guarded(body, err)


def release_build(
    slug: str,
    *,
    project_root: Path | None = None,
    bundle_id: str | None = None,
    bundle_only: bool = False,
    mock: bool = False,
    isolated: Path | None = None,
    expect_tree_sha256: str | None = None,
    ack_caution_families: Sequence[str] = (),
    embedding_cache: Path | None = None,
    runner: Runner = subprocess.run,
    echo: Echo = _echo,
    err: Echo = _err,
) -> int:
    """Write the bundle (after ``plan``), then the non-live derived stages."""

    def body() -> int:
        root = resolve_project_root(project_root)
        conference_dir = _workspace(root, slug)
        settings = release_settings(conference_dir)
        bundle = _bundle_id(conference_dir, bundle_id, settings)
        _require_promoted_manifest(conference_dir)
        if isolated is not None:
            return _build_isolated(conference_dir, bundle, isolated, expect_tree_sha256, ack_caution_families, echo)
        if expect_tree_sha256 is not None:
            raise ReleaseRefused("--expect-tree-sha256 needs --isolated")
        if embedding_cache is not None and mock:
            raise ReleaseRefused("--embedding-cache and --mock are two different embedding sources; pass one")
        export, plan = _export_plan(
            conference_dir, bundle, conference_dir / "exports" / "public-release", ack_caution_families
        )
        if export.published_bundle_evidence(plan):
            raise ReleaseRefused(
                f"{export.ERROR_IMMUTABLE_BUNDLE}: {bundle} has live publication receipts; "
                "use --isolated to rebuild it for comparison"
            )
        config = PublicReleaseConfig(
            project_root=root,
            conference_slug=conference_dir.name,
            bundle_id=bundle,
            receipt_path=command_receipt(conference_dir, bundle, "build"),
            write_bundle=True,
            reviewed_dry_run_receipt=_dry_run_receipt(conference_dir, bundle),
            ack_caution_families=tuple(ack_caution_families),
            build_index=not bundle_only,
            generate_site=not bundle_only,
            package_dataset=not bundle_only,
            github_repo=settings.get("github_repo"),
            index_embedding_provider=MOCK_EMBEDDING_PROVIDER if mock else None,
            index_embedding_cache=embedding_cache.resolve() if embedding_cache is not None else None,
        )
        code = _run(config, runner, echo)
        if code != EXIT_OK or bundle_only:
            return code
        # The repository is staged last, over the finished bundle: in one orchestrator
        # run it would be staged before the dataset package writes metadata/ (IF6).
        repo = PublicReleaseConfig(
            project_root=root,
            conference_slug=conference_dir.name,
            bundle_id=bundle,
            receipt_path=command_receipt(conference_dir, bundle, "build-repo"),
            stage_repo=True,
            github_repo=settings.get("github_repo"),
        )
        return _run(repo, runner, echo)

    return _guarded(body, err)


# --- check -----------------------------------------------------------------------------


def _check_bundle(
    bundle_root: Path,
    export_files: list[str] | None,
    recorded_tree: str | None,
    env: Mapping[str, str] | None,
    secret_scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> list[dict[str, str]]:
    from conference_pipeline.publishing.export_gate import GateClosed, no_secrets

    export = _load_export_script(resolve_project_root(None))
    findings: list[dict[str, str]] = []
    if export_files is not None and recorded_tree is not None:
        try:
            recomputed = export.tree_sha256_for(bundle_root, export_files)
        except Exception as exc:
            findings.append({"check": "tree_digest", "path": "", "detail": str(exc)})
        else:
            if recomputed != recorded_tree:
                findings.append({"check": "tree_digest", "path": "", "detail": "export tree digest differs from the write receipt"})
    checksums_path = bundle_root / "CHECKSUMS.json"
    try:
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        findings.append({"check": "checksums", "path": "CHECKSUMS.json", "detail": "unreadable"})
        checksums = {}
    for relative, entry in sorted(checksums.items()):
        path = bundle_root / relative
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if path.is_symlink() or not path.is_file():
            findings.append({"check": "checksums", "path": relative, "detail": "missing"})
        elif export.sha256_file(path) != expected:
            findings.append({"check": "checksums", "path": relative, "detail": "sha256 mismatch"})
    rules = builtin_rules() + extra_rules_from_env(env) + publishing_literal_rules()
    report = scan_paths(bundle_root, ["."], rules, load_allowlist(None), tracked_only=False)
    for finding in report.findings:
        # A PDF or zip has one verdict, the container check's below, which reads its text,
        # streams, comments and members with every rule; a raw text scan of it would meet
        # opaque stream bytes with text rules (pr-49-r7 F3). No rule is dropped for it.
        shown = bundle_root / finding.path
        if shown.is_file() and not shown.is_symlink() and _is_container(shown):
            continue
        findings.append({"check": "private_marker", "path": finding.path, "detail": f"{finding.rule} {finding.excerpt}"})
    # The scanner skips files over its size cap and non-regular nodes (F3). A bundle holds
    # no symlinks, and a large text file is streamed here line by line; a large binary
    # (video, audio, PDF) has no text to scan.
    for path in sorted(bundle_root.rglob("*")):
        relative = path.relative_to(bundle_root).as_posix()
        if ".git" in Path(relative).parts and path.is_file() and not path.is_symlink() and not _is_binary(path):
            # The tree scanner never enters .git; a staged repository inside the bundle does
            # hold text files there (review phase-6a-r3 F2).
            findings.extend(_scan_large_text(path, relative, rules))
            continue
        if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir()):
            findings.append({"check": "private_marker", "path": relative, "detail": "not a regular file; not scanned"})
        elif path.is_file() and (_is_binary(path) or _is_container(path)):
            # The text scanner skips binaries of any size; their literal tokens are checked
            # here, as the site build checks them (review phase-6a-r1 F3, r2 F2). A PDF or zip
            # is read here whatever its bytes: one with no NUL hides its text in streams the
            # text scanner cannot read (review phase-6b-r2 F1).
            # The byte check's own default rules, with the env file's extra markers in place
            # of the process's (pr-49-r3); the literal tokens are checked there already.
            if has_publishing_marker_bytes(path.read_bytes(), rules=builtin_rules() + extra_rules_from_env(env)):
                findings.append({"check": "private_marker", "path": relative, "detail": "publishing marker in a binary file"})
        elif path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            findings.extend(_scan_large_text(path, relative, rules))
    findings.extend(_check_licenses(bundle_root))
    findings.extend(surface_findings(bundle_root))
    files = [path for path in sorted(bundle_root.rglob("*")) if path.is_file() and not path.is_symlink()]
    try:
        no_secrets(files, _scanner=secret_scanner)
    except GateClosed as exc:
        findings.append({"check": "secrets", "path": "", "detail": f"secrets gate closed: {exc.reason}"})
    return findings


def surface_findings(bundle_root: Path, surfaces: Sequence[str] = ("github", "pages", "hf", "r2")) -> list[dict[str, str]]:
    """Every provider limit the bundle's surface trees break (v6.0 Phase 7.1, spec 6.2)."""
    found: list[dict[str, str]] = []
    if "github" in surfaces and (bundle_root / "public-repo").is_dir():
        found.extend(_prefixed(check_tree("github", bundle_root / "public-repo"), "public-repo"))
    if "pages" in surfaces and (bundle_root / "site").is_dir():
        found.extend(_prefixed(check_tree("pages", bundle_root / "site"), "site"))
    if "hf" in surfaces:
        relatives = [
            path.relative_to(bundle_root).as_posix()
            for path in sorted(bundle_root.rglob("*"))
            if path.is_file() and not path.is_symlink() and ".git" not in path.relative_to(bundle_root).parts
        ]
        # The remote tree, with the files the upload adds itself (pr-49-r1 F5).
        planned = [(rel, (bundle_root / rel).lstat().st_size) for rel in hf_upload_paths(bundle_root, relatives)]
        found.extend(check_sized("hf", planned + [(name, 0) for name in HF_GENERATED_FILES]))
    if "r2" in surfaces:
        found.extend(r2_findings(bundle_root))
    return found


def r2_findings(bundle_root: Path) -> list[dict[str, str]]:
    """Media the sidecars name that one PUT cannot upload (phase-7.1-r1 F2).

    The objects are not in the bundle, so their sizes come from the sidecars, which the
    uploader re-verifies against the source before each PUT. Every sidecar the uploader
    can read is checked, the staged repository's copies included (phase-7.1-r2 F1).
    """
    from conference_pipeline.publishing.surface_limits import check_files

    sizes: list[tuple[str, int]] = []
    for sidecar in sorted(bundle_root.rglob("*.cdn.json")):
        relative = sidecar.relative_to(bundle_root).as_posix()
        try:
            size = int(json.loads(sidecar.read_text(encoding="utf-8")).get("size_bytes"))
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            sizes.append((relative, -1))
            continue
        sizes.append((relative, size))
    found = [
        {"check": "surface_limit", "path": relative, "detail": "r2: the sidecar states no readable size"}
        for relative, size in sizes
        if size < 0
    ]
    return found + check_files("r2", [(r, s) for r, s in sizes if s >= 0], {})


def _prefixed(findings: list[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    return [
        {**item, "path": prefix if item["path"] == "." else f"{prefix}/{item['path']}"} for item in findings
    ]


def _check_licenses(bundle_root: Path) -> list[dict[str, str]]:
    """Every family has a license in RELEASE.json, and the staged LICENSE states exactly those.

    A bundle written by an older exporter carries neither, and every other check passes it
    (CVPR 2026's MIT rebuild, 2026-11-19).
    """
    from conference_pipeline.publishing.release_licenses import ReleaseLicenseError, license_text

    try:
        release = json.loads((bundle_root / "RELEASE.json").read_text(encoding="utf-8"))
        expected = license_text(release)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
        return [{"check": "licenses", "path": "RELEASE.json", "detail": f"unreadable: {type(exc).__name__}"}]
    except ReleaseLicenseError as exc:
        return [{"check": "licenses", "path": "RELEASE.json", "detail": str(exc)}]
    license_path = bundle_root / "public-repo" / "LICENSE"
    if license_path.is_file() and license_path.read_text(encoding="utf-8") != expected:
        return [{"check": "licenses", "path": "public-repo/LICENSE", "detail": "does not state RELEASE.json family_licenses"}]
    return []


def _is_container(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(5)
    return head == b"%PDF-" or head[:4] == b"PK\x03\x04"


def _is_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\0" in handle.read(8192)


def _scan_large_text(path: Path, relative: str, rules: Sequence[Any]) -> list[dict[str, str]]:
    with path.open("rb") as handle:
        if b"\0" in handle.read(8192):
            return []
    found: list[dict[str, str]] = []
    empty = load_allowlist(None)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            for finding in scan_text(line, relative, rules, empty):
                found.append(
                    {"check": "private_marker", "path": relative, "detail": f"{finding.rule} line {number} {finding.excerpt}"}
                )
    return found


def release_check(
    slug: str,
    *,
    project_root: Path | None = None,
    bundle_id: str | None = None,
    isolated: Path | None = None,
    json_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    secret_scanner: Callable[[list[Path]], list[dict]] | None = None,
    echo: Echo = _echo,
    err: Echo = _err,
) -> int:
    """Re-verify a built bundle: digest, checksums, private markers, secrets."""

    def body() -> int:
        root = resolve_project_root(project_root)
        conference_dir = _workspace(root, slug)
        settings = release_settings(conference_dir)
        bundle = _bundle_id(conference_dir, bundle_id, settings)
        export_files: list[str] | None = None
        recorded_tree: str | None = None
        if isolated is not None:
            target = _isolated_dir(conference_dir, isolated, must_be_empty=False)
            bundle_root = target / "bundle"
        else:
            bundle_root = default_bundle_root(conference_dir, bundle)
            receipt = _write_receipt(conference_dir, bundle)
            if receipt.is_file():
                payload = json.loads(receipt.read_text(encoding="utf-8"))
                export_files = [row["path"] for row in payload.get("actual_artifacts", [])]
                recorded_tree = (payload.get("bundle_verification") or {}).get("bundle_tree_sha256")
        if not bundle_root.is_dir():
            raise ReleaseRefused(f"no bundle at {bundle_root}")
        findings = _check_bundle(bundle_root, export_files, recorded_tree, env, secret_scanner)
        rules = builtin_rules() + extra_rules_from_env(env)
        for finding in findings:
            echo(redact_text(f"finding: {finding['check']} {finding['path']} {finding['detail']}", rules))
        echo(f"release check: {len(findings)} finding(s)")
        if json_path is not None:
            _write_json(
                json_path,
                {
                    "bundle_id": redact_text(bundle, rules),
                    "findings": [
                        {key: redact_text(value, rules) for key, value in finding.items()} for finding in findings
                    ],
                },
            )
        return EXIT_FINDINGS if findings else EXIT_OK

    return _guarded(body, err)


# --- publish ---------------------------------------------------------------------------


def release_publish(
    slug: str,
    *,
    approved_by: str | None,
    surfaces: Sequence[str],
    project_root: Path | None = None,
    bundle_id: str | None = None,
    env_file: Path | None = None,
    with_preflight: bool = False,
    github_repo: str | None = None,
    pages_project: str | None = None,
    hf_repo_id: str | None = None,
    zenodo_sandbox: bool | None = None,
    r2_base_url: str | None = None,
    runner: Runner = subprocess.run,
    echo: Echo = _echo,
    err: Echo = _err,
) -> int:
    """Push the chosen surfaces live, with the owner's named approval, then verify."""

    def body() -> int:
        if approved_by is None or not approved_by.strip():
            raise ReleaseRefused("owner_approval_missing: pass --approved-by with the approver's name")
        chosen = tuple(dict.fromkeys(surfaces))
        if not chosen:
            raise ReleaseRefused("at least one --surface is required")
        unknown = [surface for surface in chosen if surface not in SURFACES]
        if unknown:
            raise ReleaseRefused(f"unknown --surface: {', '.join(unknown)} (one of {', '.join(SURFACES)})")
        root = resolve_project_root(project_root)
        conference_dir = _workspace(root, slug)
        settings = release_settings(conference_dir)
        bundle = _bundle_id(conference_dir, bundle_id, settings)
        bundle_root = default_bundle_root(conference_dir, bundle)
        if bundle_root.is_dir():
            # A file over a provider's limit would fail half-way through a live push.
            oversized = surface_findings(bundle_root, [s for s in chosen if s in ("github", "pages", "hf", "r2")])
            if oversized:
                first = oversized[0]
                raise ReleaseRefused(
                    f"surface_limit: {len(oversized)} finding(s), first {first['path']}: {first['detail']}"
                )
        approval = {
            "approved_by": approved_by.strip(),
            "surfaces": list(chosen),
            "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        base = PublicReleaseConfig(
            project_root=root,
            conference_slug=conference_dir.name,
            bundle_id=bundle,
            env_file=env_file if env_file is not None else root / ".env",
            reviewed_dry_run_receipt=_dry_run_receipt(conference_dir, bundle),
            public_repo_dry_run_receipt=_repo_dry_run_receipt(conference_dir, bundle),
            github_repo=github_repo or settings.get("github_repo"),
            pages_project=pages_project or settings.get("pages_project"),
            hf_repo_id=hf_repo_id or settings.get("hf_repo_id"),
            zenodo_sandbox=bool(settings.get("zenodo_sandbox", False) if zenodo_sandbox is None else zenodo_sandbox),
            r2_base_url=r2_base_url or settings.get("r2_base_url") or settings.get("cdn_base_url"),
            r2_bucket=settings.get("r2_bucket"),
            r2_prefix=settings.get("r2_prefix"),
            owner_approval=approval,
            receipt_path=command_receipt(conference_dir, bundle, "publish"),
        )
        live = replace(
            base,
            preflight_live=with_preflight,
            live_github="github" in chosen,
            ack_live_github="github" in chosen,
            live_r2="r2" in chosen,
            ack_live_r2="r2" in chosen,
            live_pages="pages" in chosen,
            ack_live_pages="pages" in chosen,
            live_hf="hf" in chosen,
            ack_live_hf="hf" in chosen,
            live_zenodo="zenodo" in chosen,
            ack_live_zenodo="zenodo" in chosen,
        )
        code = _run(live, runner, echo)
        if code != EXIT_OK:
            return code
        verify = replace(
            base,
            check_liveness=True,
            check_metadata=True,
            receipt_path=command_receipt(conference_dir, bundle, "verify"),
        )
        return _run(verify, runner, echo)

    return _guarded(body, err)


# --- Typer wiring ----------------------------------------------------------------------

release_app = typer.Typer(help="Draft, plan, build, check and publish a conference's public release.")


def _add_hub() -> None:
    from conference_pipeline.publishing.release_hub import hub_app

    release_app.add_typer(hub_app, name="hub")


_add_hub()


@release_app.command("init")
def init_command(
    slug: str = typer.Argument(..., help="The workspace, conferences/<slug>."),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/ (default: this one)."),
    force_draft_path: Path | None = typer.Option(None, "--force-draft-path", help="Draft beside an existing manifest, inside the workspace."),
    refresh: bool = typer.Option(False, "--refresh", help="Add missing storage families to excluded_families of an existing manifest."),
) -> None:
    raise typer.Exit(release_init(slug, project_root=project_root, draft_path=force_draft_path, refresh=refresh))


@release_app.command("families")
def families_command(
    slug: str = typer.Argument(...),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/ (default: this one)."),
    bundle_id: str | None = typer.Option(None, "--bundle-id"),
) -> None:
    raise typer.Exit(release_families(slug, project_root=project_root, bundle_id=bundle_id))


@release_app.command("plan")
def plan_command(
    slug: str = typer.Argument(...),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/ (default: this one)."),
    bundle_id: str | None = typer.Option(None, "--bundle-id"),
    ack_caution_family: list[str] = typer.Option([], "--ack-caution-family"),
    isolated: Path | None = typer.Option(None, "--isolated", help="Plan into this directory; write nothing in the workspace."),
) -> None:
    raise typer.Exit(
        release_plan(
            slug, project_root=project_root, bundle_id=bundle_id, ack_caution_families=ack_caution_family, isolated=isolated
        )
    )


@release_app.command("build")
def build_command(
    slug: str = typer.Argument(...),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/ (default: this one)."),
    bundle_id: str | None = typer.Option(None, "--bundle-id"),
    bundle_only: bool = typer.Option(False, "--bundle-only", help="Write the bundle only, no derived stages."),
    mock: bool = typer.Option(False, "--mock", help="Build the index with the offline deterministic-hash provider."),
    isolated: Path | None = typer.Option(None, "--isolated", help="Materialize into this directory for comparison; never publish."),
    expect_tree_sha256: str | None = typer.Option(None, "--expect-tree-sha256"),
    ack_caution_family: list[str] = typer.Option([], "--ack-caution-family"),
    embedding_cache: Path | None = typer.Option(None, "--embedding-cache", help="Vectors made elsewhere (scripts/public_embedding_cache.py)."),
) -> None:
    raise typer.Exit(
        release_build(
            slug,
            project_root=project_root,
            bundle_id=bundle_id,
            bundle_only=bundle_only,
            mock=mock,
            isolated=isolated,
            expect_tree_sha256=expect_tree_sha256,
            ack_caution_families=ack_caution_family,
            embedding_cache=embedding_cache,
        )
    )


@release_app.command("check")
def check_command(
    slug: str = typer.Argument(...),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/ (default: this one)."),
    bundle_id: str | None = typer.Option(None, "--bundle-id"),
    isolated: Path | None = typer.Option(None, "--isolated"),
    json_path: Path | None = typer.Option(None, "--json"),
) -> None:
    raise typer.Exit(
        release_check(slug, project_root=project_root, bundle_id=bundle_id, isolated=isolated, json_path=json_path)
    )


@release_app.command("publish")
def publish_command(
    slug: str = typer.Argument(...),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/ (default: this one)."),
    approved_by: str | None = typer.Option(None, "--approved-by", help="The owner who approved this publish (required)."),
    surface: list[str] = typer.Option([], "--surface", help="github, r2, pages, hf or zenodo; repeat for several."),
    bundle_id: str | None = typer.Option(None, "--bundle-id"),
    env_file: Path | None = typer.Option(None, "--env-file"),
    with_preflight: bool = typer.Option(False, "--with-preflight"),
    github_repo: str | None = typer.Option(None, "--github-repo"),
    pages_project: str | None = typer.Option(None, "--pages-project"),
    hf_repo_id: str | None = typer.Option(None, "--hf-repo-id"),
    r2_base_url: str | None = typer.Option(None, "--r2-base-url"),
) -> None:
    raise typer.Exit(
        release_publish(
            slug,
            project_root=project_root,
            approved_by=approved_by,
            surfaces=surface,
            bundle_id=bundle_id,
            env_file=env_file,
            with_preflight=with_preflight,
            github_repo=github_repo,
            pages_project=pages_project,
            hf_repo_id=hf_repo_id,
            r2_base_url=r2_base_url,
        )
    )
