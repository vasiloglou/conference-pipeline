"""``conference-pipeline release hub``: build, check and publish the cross-conference hub.

v6.0 Phase 7.1 (spec section 6.5, D5). The hub is built from release bundles only
(``hub_site.build_hub_site``), checked like a conference site plus the Pages limits, and
deployed to its own Cloudflare Pages project, named in ``config/public_hub.yaml``.
Publishing needs ``--approved-by`` (R3), like every other surface.

Exit codes: 0 done; 1 ``check`` found a finding; 2 refused.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import typer
import yaml

from conference_pipeline.publishing.hub_site import HubSiteError, build_hub_site
from conference_pipeline.publishing.live_deployment import (
    CloudflarePagesConfig,
    LiveDeploymentError,
    deploy_cloudflare_pages,
)
from conference_pipeline.publishing.public_release import _load_env_file, resolve_project_root
from conference_pipeline.publishing.receipt_paths import write_json_atomic_secure
from conference_pipeline.publishing.site_checks import check_links, lint_accessibility
from conference_pipeline.publishing.surface_limits import check_tree

HUB_CONFIG = Path("config") / "public_hub.yaml"
Echo = Callable[[str], None]


class HubRefused(Exception):
    pass


def hub_settings(project_root: Path) -> dict[str, Any]:
    path = project_root / HUB_CONFIG
    if not path.is_file():
        raise HubRefused(f"no hub config at {HUB_CONFIG}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("pages_project"), str) or not payload.get("title"):
        raise HubRefused(f"{HUB_CONFIG} needs a title and a pages_project")
    return payload


def _marker_findings(hub_dir: Path, environ: Mapping[str, str] | None = None) -> list[str]:
    """The same private-marker rules ``release check`` applies to a bundle, over the hub.

    Text files go through the scanner with no allowlist; binary and oversized files are
    read for literal publishing markers; anything not a regular file is a finding. Extra
    markers come from ``environ`` when given: the environment the publish loaded, env file
    included (pr-49-r1 F4).
    """
    from conference_pipeline._security.private_markers import (
        builtin_rules,
        extra_rules_from_env,
        has_publishing_marker_bytes,
        load_allowlist,
        publishing_literal_rules,
        scan_paths,
    )

    rules = builtin_rules() + extra_rules_from_env(environ) + publishing_literal_rules()
    # The byte check's own rules, with the env file's extra markers (pr-49-r2 F1, r3 F1);
    # the literal tokens are checked there already.
    byte_rules = builtin_rules() + extra_rules_from_env(environ)
    report = scan_paths(hub_dir, ["."], rules, load_allowlist(None), tracked_only=False)
    containers = {
        p.relative_to(hub_dir).as_posix()
        for p in hub_dir.rglob("*")
        if p.is_file() and not p.is_symlink() and _is_container(p)
    }
    # A PDF or zip has one verdict, the container check's, which reads its text, streams,
    # comments and members with every rule; a raw text scan of it would meet opaque stream
    # bytes with text rules (pr-49-r7 F3). No rule is dropped for a container.
    found = [f"private_marker: {f.path}: {f.rule}" for f in report.findings if f.path not in containers]
    for relative in sorted(containers):
        if has_publishing_marker_bytes((hub_dir / relative).read_bytes(), rules=byte_rules):
            found.append(f"private_marker: {relative}: container holds a publishing marker")
    for relative, reason in report.skipped_paths:
        path = hub_dir / relative
        if relative in containers:
            continue
        if path.is_symlink() or not path.is_file():
            found.append(f"private_marker: {relative}: not a regular file ({reason})")
        elif has_publishing_marker_bytes(path.read_bytes(), rules=byte_rules):
            found.append(f"private_marker: {relative}: {reason} file holds a publishing marker")
    return found


def _is_container(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(5)
    return head == b"%PDF-" or head[:4] == b"PK\x03\x04"


def hub_findings(hub_dir: Path, *, secret_scanner: Any = None, environ: Mapping[str, str] | None = None) -> list[str]:
    """Links, accessibility, the Pages limits, private markers and secrets.

    Empty when the hub can be deployed. The marker and secrets gates are the ones a
    bundle passes before publication (phase-7.1-r3 F1).
    """
    from conference_pipeline.publishing.export_gate import GateClosed, no_secrets

    findings = [f"links: {f}" for f in check_links(hub_dir).findings]
    findings += [f"accessibility: {f}" for f in lint_accessibility(hub_dir).findings]
    findings += [f"surface_limit: {f['path']}: {f['detail']}" for f in check_tree("pages", hub_dir)]
    if not (hub_dir / "hub_manifest.json").is_file():
        findings.append("hub_manifest.json is missing: not a built hub")
    findings += _marker_findings(hub_dir, environ)
    files = [p for p in sorted(hub_dir.rglob("*")) if p.is_file() and not p.is_symlink()]
    try:
        no_secrets(files, _scanner=secret_scanner)
    except GateClosed as exc:
        findings.append(f"secrets: gate closed: {exc.reason}")
    return findings


def hub_build(
    bundles: Sequence[Path],
    out: Path,
    *,
    project_root: Path | None = None,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    echo: Echo = print,
    secret_scanner: Any = None,
) -> int:
    """Build the hub, then run the same gates ``check`` runs; exit 1 on any finding.

    The output stays on disk for inspection; ``publish`` refuses it while a finding stands
    (pr-49-r1 F6).
    """
    root = resolve_project_root(project_root)
    settings = hub_settings(root)
    result = build_hub_site(
        list(bundles), out, hub_title=str(settings["title"]), code_repo_url=settings.get("code_repo_url")
    )
    echo(f"hub written: {result.output_dir} ({result.rows_total} papers)")
    env = _hub_environ(root, env_file, environ)
    findings = hub_findings(Path(result.output_dir), secret_scanner=secret_scanner, environ=env)
    for finding in findings:
        echo(f"finding: {finding}")
    echo(f"hub check: {len(findings)} finding(s)")
    return 1 if findings else 0


def _hub_environ(root: Path, env_file: Path | None, environ: Mapping[str, str] | None) -> Mapping[str, str]:
    """The environment publish scans with: the process (or ``environ``) plus the env file.

    build and check load it too, so they pass only what publish would pass (pr-49-r3 F2).
    """
    env, _loaded = _load_env_file(env_file or root / ".env", environ if environ is not None else os.environ)
    return env


def hub_check(
    hub_dir: Path,
    *,
    project_root: Path | None = None,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    echo: Echo = print,
    secret_scanner: Any = None,
) -> int:
    env = _hub_environ(resolve_project_root(project_root), env_file, environ)
    findings = hub_findings(hub_dir, secret_scanner=secret_scanner, environ=env)
    for finding in findings:
        echo(f"finding: {finding}")
    echo(f"hub check: {len(findings)} finding(s)")
    return 1 if findings else 0


def hub_publish(
    hub_dir: Path,
    *,
    approved_by: str | None,
    project_root: Path | None = None,
    env_file: Path | None = None,
    live: bool = True,
    receipt_path: Path | None = None,
    request_json: Any = None,
    environ: Mapping[str, str] | None = None,
    echo: Echo = print,
    secret_scanner: Any = None,
) -> int:
    if approved_by is None or not approved_by.strip():
        raise HubRefused("owner_approval_missing: pass --approved-by with the approver's name")
    root = resolve_project_root(project_root)
    settings = hub_settings(root)
    hub_dir = hub_dir.resolve()
    # The environment is loaded first, so the marker scan applies the extra markers the
    # env file names (pr-49-r1 F4).
    env = _hub_environ(root, env_file, environ)
    findings = hub_findings(hub_dir, secret_scanner=secret_scanner, environ=env)
    if findings:
        raise HubRefused(f"the hub has {len(findings)} finding(s); first: {findings[0]}")
    receipt = receipt_path or root / "storage" / "workflow-runs" / "public-hub-pages-receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = deploy_cloudflare_pages(
        CloudflarePagesConfig(
            bundle_root=hub_dir,
            site_dir=hub_dir,
            receipt_path=receipt,
            project_name=str(settings["pages_project"]),
            live=live,
        ),
        request_json=request_json,
        environ=env,
    )
    payload["owner_approval"] = {
        "approved_by": approved_by.strip(),
        "surfaces": ["hub-pages"],
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    write_json_atomic_secure(receipt, payload)
    echo(f"hub publish: {payload.get('status')}; receipt {receipt}")
    return 0 if payload.get("status") == "succeeded" else 2


hub_app = typer.Typer(help="Build, check and publish the cross-conference hub.")


def _guard(body: Callable[[], int]) -> int:
    try:
        return body()
    except (HubRefused, HubSiteError, LiveDeploymentError) as exc:
        print(f"refused: {exc}")
        return 2


@hub_app.command("build")
def build_command(
    bundle: list[Path] = typer.Option(..., "--bundle", help="A release bundle root; repeat for each conference."),
    out: Path = typer.Option(..., "--out"),
    project_root: Path | None = typer.Option(None, "--project-root"),
    env_file: Path | None = typer.Option(None, "--env-file"),
) -> None:
    raise typer.Exit(_guard(lambda: hub_build(bundle, out, project_root=project_root, env_file=env_file)))


@hub_app.command("check")
def check_command(
    hub_dir: Path = typer.Argument(...),
    project_root: Path | None = typer.Option(None, "--project-root"),
    env_file: Path | None = typer.Option(None, "--env-file"),
) -> None:
    raise typer.Exit(_guard(lambda: hub_check(hub_dir, project_root=project_root, env_file=env_file)))


@hub_app.command("publish")
def publish_command(
    hub_dir: Path = typer.Argument(...),
    approved_by: str | None = typer.Option(None, "--approved-by", help="The owner who approved this publish (required)."),
    project_root: Path | None = typer.Option(None, "--project-root"),
    env_file: Path | None = typer.Option(None, "--env-file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write the receipt; deploy nothing."),
) -> None:
    raise typer.Exit(
        _guard(lambda: hub_publish(hub_dir, approved_by=approved_by, project_root=project_root, env_file=env_file, live=not dry_run))
    )
