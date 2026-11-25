"""Live public deployment closeout helpers for redacted receipts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from conference_pipeline.publishing.cdn import source_label, validate_public_url
from conference_pipeline.publishing.public_index import PublicIndexError, validate_public_bundle_root


RECEIPT_TYPE_PREFLIGHT = "conference_pipeline.public_deployment_preflight"
RECEIPT_TYPE_OPERATOR_BLOCKED = "conference_pipeline.provider_operator_blocked"
RECEIPT_TYPE_PROVIDER_SUCCESS = "conference_pipeline.provider_success"
RECEIPT_TYPE_METADATA_AGREEMENT = "conference_pipeline.public_release_metadata_agreement"
RECEIPT_TYPE_CLOUDFLARE_PAGES = "conference_pipeline.cloudflare_pages_deploy"

DEFAULT_REQUIRED_ENV_VARS = (
    "CLOUDFLARE_API_TOKEN",
    "ZENODO_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_API_KEY",
)
DEFAULT_CANONICAL_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


class LiveDeploymentError(RuntimeError):
    """Raised when live deployment closeout checks fail closed."""


@dataclass(frozen=True)
class DeploymentPreflightConfig:
    bundle_root: Path
    receipt_path: Path
    github_repo: str
    r2_base_url: str
    public_repo_dry_run_receipt: Path
    site_receipt: Path
    dataset_receipt: Path
    r2_upload_receipt: Path
    r2_liveness_receipt: Path
    canonical_env_path: Path = DEFAULT_CANONICAL_ENV_PATH
    repo_env_path: Path = DEFAULT_CANONICAL_ENV_PATH
    required_env_vars: Sequence[str] = DEFAULT_REQUIRED_ENV_VARS
    source_repo_root: Path = Path(".")


@dataclass(frozen=True)
class ProviderReceipt:
    provider: str
    target: str
    receipt_path: Path
    public_url: str | None = None
    commit_sha: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MetadataAgreementConfig:
    bundle_root: Path
    receipt_path: Path
    preflight_receipt: Path | None = None
    github_receipt: Path | None = None
    pages_receipt: Path | None = None
    zenodo_receipt: Path | None = None
    hf_receipt: Path | None = None


@dataclass(frozen=True)
class CloudflarePagesConfig:
    bundle_root: Path
    site_dir: Path
    receipt_path: Path
    project_name: str | None = None
    live: bool = False
    branch: str = "main"
    account_id: str | None = None
    api_token: str | None = None
    api_base_url: str = "https://api.cloudflare.com/client/v4"


Runner = Callable[..., subprocess.CompletedProcess[str]]
RequestJson = Callable[..., dict[str, Any]]
_UNSET = object()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiveDeploymentError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def path_text(path: Path) -> str:
    return path.as_posix()


def require_success_receipt(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LiveDeploymentError(f"{label} receipt missing: {path}")
    data = read_json(path)
    if data.get("status") != "succeeded":
        raise LiveDeploymentError(f"{label} receipt is not succeeded: {path}")
    return data


def split_github_repo(repo: str) -> tuple[str, str]:
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise LiveDeploymentError("--github-repo must be OWNER/REPO")
    return parts[0], parts[1]


def env_names_from_file(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").strip()
        if "=" not in stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name:
            names.add(name)
    return names


def credential_facts(
    *,
    canonical_env_path: Path,
    repo_env_path: Path,
    required_env_vars: Sequence[str],
) -> dict[str, Any]:
    canonical_names = env_names_from_file(canonical_env_path)
    repo_names = env_names_from_file(repo_env_path)
    process_names = {name for name in required_env_vars if os.getenv(name)}
    available = canonical_names | repo_names | process_names
    required = list(required_env_vars)
    present = sorted(name for name in required if name in available)
    missing = [name for name in required if name not in available]
    canonical_exists = canonical_env_path.is_file()
    repo_exists = repo_env_path.is_file()
    return {
        "required_env_vars": required,
        "canonical_env_path": path_text(canonical_env_path),
        "repo_env_path": path_text(repo_env_path),
        "canonical_env_path_exists": canonical_exists,
        "repo_env_path_exists": repo_exists,
        "env_path_discrepancy": canonical_env_path.resolve() != repo_env_path.resolve()
        or canonical_exists != repo_exists,
        "present_env_vars": present,
        "missing_env_vars": missing,
        "canonical_env_var_names": sorted(name for name in canonical_names if name in required),
        "repo_env_var_names": sorted(name for name in repo_names if name in required),
        "process_env_var_names": sorted(name for name in process_names if name in required),
        "hf_token_alias_mismatch": ("HF_TOKEN" in available) != ("HUGGINGFACE_API_KEY" in available),
        "secret_values_recorded": False,
    }


def tool_availability() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for name in ("gh", "wrangler"):
        path = shutil.which(name)
        tools[name] = {"available": bool(path)}
    return tools


def preflight_public_deployment(
    config: DeploymentPreflightConfig,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    try:
        bundle_root = validate_public_bundle_root(config.bundle_root)
    except PublicIndexError as exc:
        raise LiveDeploymentError(str(exc)) from exc
    release = read_json(bundle_root / "RELEASE.json")
    owner, repo_name = split_github_repo(config.github_repo)

    public_repo_receipt = require_success_receipt(config.public_repo_dry_run_receipt, label="public repo dry-run")
    site_receipt = require_success_receipt(config.site_receipt, label="static site")
    dataset_receipt = require_success_receipt(config.dataset_receipt, label="dataset package")
    upload_receipt = require_success_receipt(config.r2_upload_receipt, label="R2 live upload")
    liveness_receipt = require_success_receipt(config.r2_liveness_receipt, label="R2 live liveness")

    if public_repo_receipt.get("dry_run") is not True:
        raise LiveDeploymentError("public repo receipt must be a reviewed dry-run receipt")
    if upload_receipt.get("live") is not True:
        raise LiveDeploymentError("R2 upload receipt must be live")
    if upload_receipt.get("uploaded_count") != upload_receipt.get("planned_upload_count"):
        raise LiveDeploymentError("R2 upload receipt count mismatch")
    if not upload_receipt.get("uploaded_objects"):
        raise LiveDeploymentError("R2 upload receipt missing uploaded object proof")
    if liveness_receipt.get("failed_count") not in {0, None}:
        raise LiveDeploymentError("R2 liveness receipt has failed URLs")
    validate_public_url(config.r2_base_url)

    public_repo_safety = public_repo_receipt.get("public_safety")
    if not isinstance(public_repo_safety, dict):
        raise LiveDeploymentError("public repo dry-run receipt missing public_safety")
    if public_repo_safety.get("status") != "clean":
        raise LiveDeploymentError("public repo dry-run receipt did not pass public-safety checks")
    public_repo_secret_gate = public_repo_receipt.get("secret_gate")
    if (
        not isinstance(public_repo_secret_gate, dict)
        or public_repo_secret_gate.get("gate_id") != "secrets"
        or public_repo_secret_gate.get("status") != "passed"
        or public_repo_secret_gate.get("findings") != []
    ):
        raise LiveDeploymentError("public repo dry-run receipt missing passed secret gate")

    credentials = credential_facts(
        canonical_env_path=config.canonical_env_path,
        repo_env_path=config.repo_env_path,
        required_env_vars=config.required_env_vars,
    )
    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE_PREFLIGHT,
        "status": "succeeded",
        "exit_code": 0,
        "bundle_id": release.get("bundle_id") or bundle_root.name,
        "bundle_root": path_text(config.bundle_root),
        "github": {
            "owner": owner,
            "repo_name": repo_name,
            "repo": config.github_repo,
            "target": config.github_repo,
            "explicit_owner_required": True,
        },
        "r2": {
            "base_url": config.r2_base_url,
            "live_upload": {
                "receipt_path": path_text(config.r2_upload_receipt),
                "status": upload_receipt.get("status"),
                "live": upload_receipt.get("live"),
                "planned_upload_count": upload_receipt.get("planned_upload_count"),
                "uploaded_count": upload_receipt.get("uploaded_count"),
            },
            "live_liveness": {
                "receipt_path": path_text(config.r2_liveness_receipt),
                "status": liveness_receipt.get("status"),
                "checked_count": liveness_receipt.get("checked_count"),
                "failed_count": liveness_receipt.get("failed_count"),
            },
        },
        "existing_receipts": {
            "public_repo_dry_run": path_text(config.public_repo_dry_run_receipt),
            "static_site": path_text(config.site_receipt),
            "dataset_package": path_text(config.dataset_receipt),
        },
        "credentials": credentials,
        "tools": tool_availability(),
        "public_repo_safety": {
            "receipt_path": path_text(config.public_repo_dry_run_receipt),
            "status": public_repo_safety.get("status"),
            "scope": public_repo_safety.get("scope"),
            "private_marker_count": public_repo_safety.get("private_marker_count"),
            "source_repo_history_published": public_repo_safety.get(
                "source_repo_history_published", False
            ),
        },
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    write_json(config.receipt_path, payload)
    return payload


def write_provider_success_receipt(receipt: ProviderReceipt) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE_PROVIDER_SUCCESS,
        "status": "succeeded",
        "exit_code": 0,
        "provider": receipt.provider,
        "target": receipt.target,
        "public_url": receipt.public_url,
        "commit_sha": receipt.commit_sha,
        "metadata": receipt.metadata or {},
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    write_json(receipt.receipt_path, payload)
    return payload


def write_operator_blocked_receipt(
    path: Path,
    *,
    provider: str,
    target: str,
    blocked_by: Sequence[str],
    missing_env_vars: Sequence[str] = (),
    missing_approvals: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE_OPERATOR_BLOCKED,
        "status": "operator_blocked",
        "exit_code": 2,
        "provider": provider,
        "target": target,
        "blocked_by": list(blocked_by),
        "missing_env_vars": list(missing_env_vars),
        "missing_approvals": list(missing_approvals),
        "metadata": metadata or {},
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    write_json(path, payload)
    return payload


def required_cloudflare_values(
    config: CloudflarePagesConfig,
    *,
    environ: Mapping[str, str] = os.environ,
) -> tuple[str | None, str | None, str | None, list[str]]:
    api_token = config.api_token or environ.get("CLOUDFLARE_API_TOKEN")
    account_id = config.account_id or environ.get("CLOUDFLARE_ACCOUNT_ID")
    project_name = config.project_name or environ.get("CLOUDFLARE_PAGES_PROJECT")
    missing = []
    if not api_token:
        missing.append("CLOUDFLARE_API_TOKEN")
    if not account_id:
        missing.append("CLOUDFLARE_ACCOUNT_ID")
    if not project_name:
        missing.append("CLOUDFLARE_PAGES_PROJECT")
    return api_token, account_id, project_name, missing


HUB_MANIFEST = "hub_manifest.json"


def is_hub_site_dir(bundle_root: Path, site_dir: Path) -> bool:
    """A cross-conference hub is deployed from its own directory (v6.0 Phase 7.1, 6.5)."""
    return bundle_root.resolve() == site_dir.resolve() and (site_dir / HUB_MANIFEST).is_file()


def validate_pages_site_dir(bundle_root: Path, site_dir: Path) -> tuple[Path, Path]:
    if is_hub_site_dir(bundle_root, site_dir):
        source = site_dir.resolve()
        if source.is_symlink() or not source.is_dir():
            raise LiveDeploymentError("the hub directory must be a real directory")
        return source, source
    try:
        root = validate_public_bundle_root(bundle_root)
    except PublicIndexError as exc:
        raise LiveDeploymentError(str(exc)) from exc
    source = site_dir.resolve()
    expected = (root / "site").resolve()
    if source != expected:
        raise LiveDeploymentError("Cloudflare Pages source directory must be site/ under the public bundle")
    if not source.is_dir():
        raise LiveDeploymentError(f"Cloudflare Pages source directory missing: {site_dir}")
    if not (source / "site_manifest.json").is_file():
        raise LiveDeploymentError("Cloudflare Pages source directory must contain site_manifest.json")
    return root, source


def collect_pages_assets(site_dir: Path) -> tuple[dict[str, dict[str, str]], int]:
    files: dict[str, dict[str, str]] = {}
    total_bytes = 0
    for path in sorted(site_dir.rglob("*"), key=lambda item: item.relative_to(site_dir).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(site_dir).as_posix()
        data = path.read_bytes()
        total_bytes += len(data)
        files[rel] = {
            "content_type": mimetypes.guess_type(rel)[0] or "application/octet-stream",
            "sha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode("ascii"),
        }
    if not files:
        raise LiveDeploymentError("Cloudflare Pages source directory has no files")
    return files, total_bytes


def public_site_url_from_env(project_name: str, environ: Mapping[str, str] = os.environ) -> str:
    for name in ("PUBLIC_SITE_URL", "CLOUDFLARE_PUBLIC_SITE_URL", "CLOUDFLARE_PAGES_URL"):
        value = environ.get(name)
        if value and value.strip():
            validate_public_url(value.strip())
            return value.strip()
    fallback = f"https://{project_name}.pages.dev"
    validate_public_url(fallback)
    return fallback


def extract_pages_deployment_url(output: str, project_name: str, environ: Mapping[str, str]) -> str:
    candidates = re.findall(r"https://[^\s\"'<>`]+", output)
    for candidate in candidates:
        url = candidate.rstrip(".,;:)]}`")
        if "pages.dev" in url:
            validate_public_url(url)
            return url
    return public_site_url_from_env(project_name, environ=environ)


def run_wrangler_pages_deploy(
    source_dir: Path,
    *,
    project_name: str,
    branch: str,
    runner: Runner,
    environ: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({key: value for key, value in environ.items() if isinstance(value, str)})
    cmd = [
        "npx",
        "--yes",
        "wrangler",
        "pages",
        "deploy",
        str(source_dir),
        "--project-name",
        project_name,
        "--branch",
        branch,
    ]
    return runner(cmd, capture_output=True, text=True, check=False, env=env)


def default_request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    import httpx

    headers = kwargs.pop("headers", None)
    with httpx.Client(timeout=120.0) as client:
        response = client.request(method, url, headers=headers, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise LiveDeploymentError(f"Cloudflare API returned non-JSON response: HTTP {response.status_code}") from exc
    if response.status_code >= 400:
        errors = payload.get("errors") if isinstance(payload, dict) else None
        raise LiveDeploymentError(f"Cloudflare API HTTP {response.status_code}: {redacted_provider_errors(errors)}")
    if not isinstance(payload, dict):
        raise LiveDeploymentError("Cloudflare API JSON object expected")
    return payload


def redacted_provider_errors(errors: Any) -> list[dict[str, Any]]:
    if not isinstance(errors, list):
        return []
    redacted: list[dict[str, Any]] = []
    for error in errors[:5]:
        if not isinstance(error, dict):
            continue
        redacted.append(
            {
                "code": error.get("code"),
                "message": str(error.get("message", ""))[:240],
            }
        )
    return redacted


def cloudflare_success(payload: dict[str, Any]) -> bool:
    return payload.get("success") is True


def cloudflare_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    return result if isinstance(result, dict) else {}


def write_cloudflare_pages_blocked_receipt(
    config: CloudflarePagesConfig,
    *,
    bundle_id: str | None,
    source_dir: Path,
    project_name: str | None,
    blocked_by: Sequence[str],
    missing_env_vars: Sequence[str] = (),
    cloudflare_errors: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE_CLOUDFLARE_PAGES,
        "status": "operator_blocked",
        "exit_code": 2,
        "provider": "cloudflare_pages",
        "bundle_id": bundle_id,
        "project_name": project_name,
        "deployment_url": None,
        "deployment_id": None,
        "source_dir": path_text(source_dir),
        "live": config.live,
        "deploy_method": "cloudflare_pages_api_direct_upload",
        "blocked_by": list(blocked_by),
        "missing_env_vars": list(missing_env_vars),
        "cloudflare_errors": list(cloudflare_errors),
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    write_json(config.receipt_path, payload)
    return payload


def deploy_cloudflare_pages(
    config: CloudflarePagesConfig,
    *,
    request_json: RequestJson | None = None,
    runner: Runner = subprocess.run,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    bundle_root, source_dir = validate_pages_site_dir(config.bundle_root, config.site_dir)
    if (source_dir / HUB_MANIFEST).is_file() and bundle_root == source_dir:
        bundle_id = "hub"
    else:
        release = read_json(bundle_root / "RELEASE.json")
        bundle_id = release.get("bundle_id") or bundle_root.name
    api_token, account_id, project_name, missing = required_cloudflare_values(config, environ=environ)
    if missing:
        return write_cloudflare_pages_blocked_receipt(
            config,
            bundle_id=bundle_id,
            source_dir=source_dir,
            project_name=project_name,
            blocked_by=["missing_cloudflare_configuration"],
            missing_env_vars=missing,
        )
    assert api_token is not None
    assert account_id is not None
    assert project_name is not None

    if config.live:
        from conference_pipeline.publishing.surface_limits import check_tree

        if check_tree("pages", source_dir):
            # Every Pages deploy meets the limits, not only release publish (pr-49-r1 F3).
            return write_cloudflare_pages_blocked_receipt(
                config, bundle_id=bundle_id, source_dir=source_dir, project_name=project_name, blocked_by=["surface_limit"]
            )
    assets, total_bytes = collect_pages_assets(source_dir)
    if not config.live:
        payload = {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE_CLOUDFLARE_PAGES,
            "status": "succeeded",
            "exit_code": 0,
            "provider": "cloudflare_pages",
            "bundle_id": bundle_id,
            "project_name": project_name,
            "deployment_url": None,
            "deployment_id": None,
            "source_dir": path_text(source_dir),
            "live": False,
            "operator_live_step_required": True,
            "deploy_method": "cloudflare_pages_api_direct_upload",
            "source_file_count": len(assets),
            "source_total_bytes": total_bytes,
            "secret_values_recorded": False,
            "generated_at": utc_now(),
        }
        write_json(config.receipt_path, payload)
        return payload

    if request_json is not None:
        request = request_json
        headers = {"Authorization": "Bearer <redacted>", "Content-Type": "application/json"}
        actual_headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        project_url = f"{config.api_base_url}/accounts/{account_id}/pages/projects/{project_name}"
        try:
            project_payload = request("GET", project_url, headers=actual_headers)
            if not cloudflare_success(project_payload):
                return write_cloudflare_pages_blocked_receipt(
                    config,
                    bundle_id=bundle_id,
                    source_dir=source_dir,
                    project_name=project_name,
                    blocked_by=["cloudflare_project_lookup_failed"],
                    cloudflare_errors=redacted_provider_errors(project_payload.get("errors")),
                )
            deployment_response = request(
                "POST",
                f"{project_url}/deployments",
                headers=actual_headers,
                data={
                    "branch": config.branch,
                    "message": f"Deploy {bundle_id} public site",
                    "manifest": json.dumps({path: meta["sha256"] for path, meta in assets.items()}, sort_keys=True),
                },
            )
        except LiveDeploymentError as exc:
            return write_cloudflare_pages_blocked_receipt(
                config,
                bundle_id=bundle_id,
                source_dir=source_dir,
                project_name=project_name,
                blocked_by=["cloudflare_api_error"],
                cloudflare_errors=[{"message": str(exc)[:240]}],
            )

        if not cloudflare_success(deployment_response):
            return write_cloudflare_pages_blocked_receipt(
                config,
                bundle_id=bundle_id,
                source_dir=source_dir,
                project_name=project_name,
                blocked_by=["cloudflare_deployment_failed"],
                cloudflare_errors=redacted_provider_errors(deployment_response.get("errors")),
            )
        deployment = cloudflare_result(deployment_response)
        deployment_url = (
            deployment.get("url")
            or deployment.get("deployment_url")
            or deployment.get("canonical_deployment_url")
            or public_site_url_from_env(project_name, environ=environ)
        )
        deployment_id = deployment.get("id")
        deploy_method = "cloudflare_pages_api_form_manifest"
    else:
        result = run_wrangler_pages_deploy(
            source_dir,
            project_name=project_name,
            branch=config.branch,
            runner=runner,
            environ=environ,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "wrangler pages deploy failed")[:240]
            return write_cloudflare_pages_blocked_receipt(
                config,
                bundle_id=bundle_id,
                source_dir=source_dir,
                project_name=project_name,
                blocked_by=["cloudflare_pages_deploy_failed"],
                cloudflare_errors=[{"message": stderr}],
            )
        deployment_url = extract_pages_deployment_url(
            f"{result.stdout or ''}\n{result.stderr or ''}",
            project_name,
            environ,
        )
        deployment_id = None
        deploy_method = "wrangler_pages_deploy"

    try:
        validate_public_url(str(deployment_url))
    except LiveDeploymentError as exc:
        return write_cloudflare_pages_blocked_receipt(
            config,
            bundle_id=bundle_id,
            source_dir=source_dir,
            project_name=project_name,
            blocked_by=["cloudflare_api_error"],
            cloudflare_errors=[{"message": str(exc)[:240]}],
        )
    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE_CLOUDFLARE_PAGES,
        "status": "succeeded",
        "exit_code": 0,
        "provider": "cloudflare_pages",
        "bundle_id": bundle_id,
        "project_name": project_name,
        "public_url": public_site_url_from_env(project_name, environ=environ),
        "deployment_url": str(deployment_url),
        "deployment_id": deployment_id,
        "source_dir": path_text(source_dir),
        "live": True,
        "deploy_method": deploy_method,
        "branch": config.branch,
        "source_file_count": len(assets),
        "source_total_bytes": total_bytes,
        "account_id_present": True,
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    if request_json is not None:
        payload["auth_headers_recorded"] = headers
    write_json(config.receipt_path, payload)
    return payload


def text_for(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def json_text_for(path: Path) -> str:
    if not path.is_file():
        return ""
    payload = read_json(path)
    return json.dumps(payload, sort_keys=True)


def metadata_expected_fields(release: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in (
        "public_repo_url",
        "public_site_url",
        "cdn_base_url",
        "zenodo_doi",
        "zenodo_record_url",
        "hf_dataset_url",
    ):
        value = release.get(key)
        if isinstance(value, str) and value.strip():
            fields[key] = value.strip()
    return fields


def agreement_sources(bundle_root: Path) -> dict[str, list[Path]]:
    return {
        "public_repo_url": [
            bundle_root / "public-repo" / "README.md",
            bundle_root / "public-repo" / "CITATION.cff",
            bundle_root / "public-repo" / "public_repo_manifest.json",
            bundle_root / "metadata" / "croissant.json",
            bundle_root / "metadata" / "datapackage.json",
            bundle_root / "metadata" / "dataset_manifest.json",
        ],
        "public_site_url": [
            bundle_root / "public-repo" / "README.md",
            bundle_root / "site" / "site_manifest.json",
            bundle_root / "metadata" / "croissant.json",
            bundle_root / "metadata" / "datapackage.json",
            bundle_root / "metadata" / "dataset_manifest.json",
        ],
        "cdn_base_url": [
            bundle_root / "public-repo" / "README.md",
            bundle_root / "public-repo" / "public_repo_manifest.json",
            bundle_root / "site" / "site_manifest.json",
            bundle_root / "metadata" / "croissant.json",
            bundle_root / "metadata" / "datapackage.json",
            bundle_root / "metadata" / "dataset_manifest.json",
            *sorted(bundle_root.rglob("*.cdn.json")),
        ],
        "zenodo_doi": [
            bundle_root / "public-repo" / "README.md",
            bundle_root / "public-repo" / "CITATION.cff",
            bundle_root / "metadata" / "CITATION.cff",
            bundle_root / "metadata" / "croissant.json",
            bundle_root / "metadata" / "dataset_manifest.json",
        ],
        "zenodo_record_url": [
            bundle_root / "public-repo" / "README.md",
            bundle_root / "metadata" / "croissant.json",
            bundle_root / "metadata" / "datapackage.json",
            bundle_root / "metadata" / "dataset_manifest.json",
        ],
        "hf_dataset_url": [
            bundle_root / "public-repo" / "README.md",
            bundle_root / "metadata" / "croissant.json",
            bundle_root / "metadata" / "dataset_manifest.json",
        ],
    }


def value_from_receipt(receipt: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def add_release_receipt_issue(
    issues: list[dict[str, Any]],
    *,
    field: str,
    release_value: Any,
    expected: str | None,
    receipt_path: Path,
    status: str | None,
) -> None:
    if status == "succeeded":
        if not expected:
            issues.append(
                {
                    "field": field,
                    "expected": None,
                    "actual": release_value,
                    "source_path": path_text(receipt_path),
                    "reason": "succeeded_receipt_missing_public_identifier",
                }
            )
        elif release_value != expected:
            issues.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": release_value,
                    "source_path": path_text(receipt_path),
                    "reason": "release_field_mismatch",
                }
            )
    elif status == "operator_blocked" and release_value not in {None, ""}:
        issues.append(
            {
                "field": field,
                "expected": None,
                "actual": release_value,
                "source_path": path_text(receipt_path),
                "reason": "operator_blocked_receipt_has_release_identifier",
            }
        )


def check_receipt_agreement(
    release: dict[str, Any],
    config: MetadataAgreementConfig,
    issues: list[dict[str, Any]],
    checked_sources: set[str],
    bundle_root: Path,
) -> None:
    if config.preflight_receipt:
        checked_sources.add(source_label(bundle_root, config.preflight_receipt))
        preflight = require_success_receipt(config.preflight_receipt, label="public deployment preflight")
        expected = None
        r2 = preflight.get("r2")
        if isinstance(r2, dict):
            expected = value_from_receipt(r2, "base_url")
        add_release_receipt_issue(
            issues,
            field="cdn_base_url",
            release_value=release.get("cdn_base_url"),
            expected=expected,
            receipt_path=config.preflight_receipt,
            status=preflight.get("status"),
        )

    receipt_specs = [
        (config.github_receipt, "public_repo_url", ("public_url", "url")),
        (config.pages_receipt, "public_site_url", ("public_url", "deployment_url", "url")),
        (config.zenodo_receipt, "zenodo_doi", ("doi", "zenodo_doi")),
        (config.zenodo_receipt, "zenodo_record_url", ("record_url", "zenodo_record_url")),
        (config.hf_receipt, "hf_dataset_url", ("dataset_url", "hf_dataset_url", "public_url", "url")),
    ]
    for receipt_path, field, keys in receipt_specs:
        if receipt_path is None:
            continue
        checked_sources.add(source_label(bundle_root, receipt_path))
        receipt = read_json(receipt_path)
        add_release_receipt_issue(
            issues,
            field=field,
            release_value=release.get(field),
            expected=value_from_receipt(receipt, *keys),
            receipt_path=receipt_path,
            status=receipt.get("status"),
        )


def check_metadata_agreement(config: MetadataAgreementConfig) -> dict[str, Any]:
    bundle_root = validate_public_bundle_root(config.bundle_root)
    release = read_json(bundle_root / "RELEASE.json")
    fields = metadata_expected_fields(release)
    sources = agreement_sources(bundle_root)
    issues: list[dict[str, Any]] = []
    checked_sources: set[str] = {source_label(bundle_root, bundle_root / "RELEASE.json")}
    check_receipt_agreement(release, config, issues, checked_sources, bundle_root)
    for field, expected in fields.items():
        if field.endswith("_url") or field == "cdn_base_url" or field == "zenodo_record_url":
            validate_public_url(expected)
        for source in sources.get(field, []):
            if not source.exists():
                continue
            checked_sources.add(source_label(bundle_root, source))
            text = json_text_for(source) if source.suffix == ".json" else text_for(source)
            if expected not in text:
                issues.append(
                    {
                        "field": field,
                        "expected": expected,
                        "source_path": path_text(source),
                        "reason": "expected_value_absent",
                    }
                )
    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE_METADATA_AGREEMENT,
        "status": "failed" if issues else "succeeded",
        "exit_code": 1 if issues else 0,
        "bundle_id": release.get("bundle_id") or bundle_root.name,
        "checked_fields": sorted(fields),
        "checked_sources": sorted(checked_sources),
        "issues": issues,
        "mismatch_count": len(issues),
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    write_json(config.receipt_path, payload)
    if issues:
        raise LiveDeploymentError(f"{len(issues)} release metadata agreement issue(s)")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restamp_release_metadata(
    *,
    bundle_root: Path,
    public_repo_url: str | None | object = _UNSET,
    public_site_url: str | None | object = _UNSET,
    cdn_base_url: str | None | object = _UNSET,
    zenodo_doi: str | None | object = _UNSET,
    zenodo_record_url: str | None | object = _UNSET,
    hf_dataset_url: str | None | object = _UNSET,
) -> dict[str, Any]:
    root = validate_public_bundle_root(bundle_root)
    release_path = root / "RELEASE.json"
    checksums_path = root / "CHECKSUMS.json"
    release = read_json(release_path)
    updates = {
        "public_repo_url": public_repo_url,
        "public_site_url": public_site_url,
        "cdn_base_url": cdn_base_url,
        "zenodo_doi": zenodo_doi,
        "zenodo_record_url": zenodo_record_url,
        "hf_dataset_url": hf_dataset_url,
    }
    updates = {key: value for key, value in updates.items() if value is not _UNSET}
    changed = {key: value for key, value in updates.items() if release.get(key) != value}
    for key, value in updates.items():
        if value and isinstance(value, str) and (key.endswith("_url") or key == "cdn_base_url"):
            validate_public_url(str(value))
        release[key] = value
    write_json(release_path, release)

    checksums = read_json(checksums_path)
    for path in [
        release_path,
        root / "public-repo" / "README.md",
        root / "public-repo" / "CITATION.cff",
        root / "public-repo" / "public_repo_manifest.json",
        root / "public-repo" / "RELEASE.json",
        root / "public-repo" / "metadata" / "croissant.json",
        root / "public-repo" / "metadata" / "datapackage.json",
        root / "public-repo" / "metadata" / "dataset_manifest.json",
        root / "public-repo" / "metadata" / "CITATION.cff",
        root / "public-repo" / "site" / "site_manifest.json",
        root / "metadata" / "croissant.json",
        root / "metadata" / "datapackage.json",
        root / "metadata" / "dataset_manifest.json",
        root / "metadata" / "CITATION.cff",
        root / "site" / "site_manifest.json",
    ]:
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            checksums[rel] = {"sha256": sha256_file(path)}
    write_json(checksums_path, checksums)
    return {"updated_fields": sorted(changed), "checksum_entries_updated": sorted(checksums)}


def parse_preflight_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run redacted public deployment preflight checks")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--r2-base-url", required=True)
    parser.add_argument("--canonical-env-path", type=Path, default=DEFAULT_CANONICAL_ENV_PATH)
    parser.add_argument("--repo-env-path", type=Path, default=DEFAULT_CANONICAL_ENV_PATH)
    parser.add_argument("--public-repo-dry-run-receipt", type=Path, required=True)
    parser.add_argument("--site-receipt", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--r2-upload-receipt", type=Path, required=True)
    parser.add_argument("--r2-liveness-receipt", type=Path, required=True)
    return parser.parse_args(argv)


def preflight_main(argv: Sequence[str] | None = None) -> int:
    args = parse_preflight_args(argv)
    try:
        receipt = preflight_public_deployment(
            DeploymentPreflightConfig(
                bundle_root=args.bundle_root,
                receipt_path=args.receipt_path,
                github_repo=args.github_repo,
                r2_base_url=args.r2_base_url,
                public_repo_dry_run_receipt=args.public_repo_dry_run_receipt,
                site_receipt=args.site_receipt,
                dataset_receipt=args.dataset_receipt,
                r2_upload_receipt=args.r2_upload_receipt,
                r2_liveness_receipt=args.r2_liveness_receipt,
                canonical_env_path=args.canonical_env_path,
                repo_env_path=args.repo_env_path,
                source_repo_root=Path.cwd(),
            )
        )
    except LiveDeploymentError as exc:
        print(str(exc))
        return 2
    print(f"Preflight receipt: {args.receipt_path}")
    print(f"Bundle: {receipt['bundle_id']}")
    print(f"GitHub target: {receipt['github']['repo']}")
    return 0


def parse_metadata_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check or restamp public release metadata agreement")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--public-repo-url", default=argparse.SUPPRESS)
    parser.add_argument("--public-site-url", default=argparse.SUPPRESS)
    parser.add_argument("--cdn-base-url", default=argparse.SUPPRESS)
    parser.add_argument("--zenodo-doi", default=argparse.SUPPRESS)
    parser.add_argument("--zenodo-record-url", default=argparse.SUPPRESS)
    parser.add_argument("--hf-dataset-url", default=argparse.SUPPRESS)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--github-receipt", type=Path)
    parser.add_argument("--pages-receipt", type=Path)
    parser.add_argument("--zenodo-receipt", type=Path)
    parser.add_argument("--hf-receipt", type=Path)
    parser.add_argument("--restamp", action="store_true")
    return parser.parse_args(argv)


def metadata_main(argv: Sequence[str] | None = None) -> int:
    args = parse_metadata_args(argv)
    try:
        if args.restamp:
            kwargs: dict[str, Any] = {"bundle_root": args.bundle_root}
            for key in (
                "public_repo_url",
                "public_site_url",
                "cdn_base_url",
                "zenodo_doi",
                "zenodo_record_url",
                "hf_dataset_url",
            ):
                if hasattr(args, key):
                    kwargs[key] = getattr(args, key)
            restamp_release_metadata(**kwargs)
        check_metadata_agreement(
            MetadataAgreementConfig(
                bundle_root=args.bundle_root,
                receipt_path=args.receipt_path,
                preflight_receipt=args.preflight_receipt,
                github_receipt=args.github_receipt,
                pages_receipt=args.pages_receipt,
                zenodo_receipt=args.zenodo_receipt,
                hf_receipt=args.hf_receipt,
            )
        )
    except LiveDeploymentError as exc:
        print(str(exc))
        return 1
    print(f"Metadata agreement receipt: {args.receipt_path}")
    return 0


def parse_pages_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a generated public site to Cloudflare Pages")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--project-name")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def pages_main(argv: Sequence[str] | None = None) -> int:
    args = parse_pages_args(argv)
    try:
        receipt = deploy_cloudflare_pages(
            CloudflarePagesConfig(
                bundle_root=args.bundle_root,
                site_dir=args.site_dir,
                receipt_path=args.receipt_path,
                project_name=args.project_name,
                branch=args.branch,
                live=args.live,
            )
        )
    except LiveDeploymentError as exc:
        print(str(exc))
        return 2
    print(f"Cloudflare Pages receipt: {args.receipt_path}")
    print(f"Status: {receipt['status']}")
    if receipt.get("deployment_url"):
        print(f"Deployment URL: {receipt['deployment_url']}")
    return 0 if receipt.get("status") == "succeeded" else 2
