"""Dataset metadata and publication-gate helpers for public releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote

from conference_pipeline._security.private_markers import has_publishing_marker, publishing_marker_hits
from conference_pipeline.publishing.public_index import validate_public_bundle_root
from conference_pipeline.publishing.release_layout import valid_tag
from conference_pipeline.publishing.release_licenses import (
    ReleaseLicenseError,
    family_licenses,
    license_groups,
    license_summary,
    license_url,
)


class DatasetPackageError(RuntimeError):
    """Raised when dataset package or publication gates fail closed."""




@dataclass(frozen=True)
class DatasetPackageConfig:
    bundle_root: Path
    metadata_dir: Path
    receipt_path: Path | None = None


@dataclass(frozen=True)
class ZenodoLiveConfig:
    metadata_dir: Path
    receipt_path: Path
    sandbox: bool
    bundle_root: Path | None = None
    target: str = "zenodo.org"


@dataclass(frozen=True)
class HuggingFaceLiveConfig:
    metadata_dir: Path
    repo_id: str
    receipt_path: Path
    target: str


@dataclass(frozen=True)
class UploadFile:
    path: Path
    filename: str


RequestJson = Callable[..., dict[str, Any]]
UploadBytes = Callable[..., dict[str, Any]]
HfApiFactory = Callable[[str], Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetPackageError(f"JSON object expected: {path}")
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


def resource_rows(bundle_root: Path) -> list[dict[str, Any]]:
    checksums = read_json(bundle_root / "CHECKSUMS.json")
    rows: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob("*"), key=lambda item: item.relative_to(bundle_root).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle_root).as_posix()
        if rel.startswith("metadata/"):
            continue
        if has_publishing_marker(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if has_publishing_marker(text):
            continue
        checksum_value = checksums.get(rel)
        if isinstance(checksum_value, dict):
            checksum_value = checksum_value.get("sha256")
        rows.append(
            {
                "name": rel,
                "path": rel,
                "encodingFormat": path.suffix.lstrip(".") or "file",
                "sha256": checksum_value,
            }
        )
    return rows


PUBLIC_RELEASE_FIELDS = (
    "public_repo_url",
    "public_site_url",
    "cdn_base_url",
    "zenodo_doi",
    "zenodo_record_url",
    "hf_dataset_url",
)


def release_public_fields(release: dict[str, Any]) -> dict[str, Any]:
    return {field: release.get(field) for field in PUBLIC_RELEASE_FIELDS}


def citation_text(release: dict[str, Any]) -> str:
    bundle_id = release.get("bundle_id", "unknown")
    conference = release.get("conference", "Conference")
    text = (
        "cff-version: 1.2.0\n"
        f"title: \"{conference} public data release {bundle_id}\"\n"
        "message: \"If you use this dataset, please cite this release.\"\n"
        "type: dataset\n"
        f"date-released: \"{datetime.now(UTC).date().isoformat()}\"\n"
        "authors:\n"
        "  - name: \"Conference Pipeline Team\"\n"
        f"version: \"{bundle_id}\"\n"
    )
    if release.get("public_repo_url"):
        text += f"url: \"{release['public_repo_url']}\"\n"
    if release.get("zenodo_doi"):
        text += f"doi: \"{release['zenodo_doi']}\"\n"
    return text


def assert_no_private_markers(metadata_dir: Path) -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in metadata_dir.rglob("*")
        if path.is_file()
    )
    hits = publishing_marker_hits(combined)
    if hits:
        raise DatasetPackageError(f"private marker found in metadata: {hits[0]}")


def package_public_dataset(config: DatasetPackageConfig) -> Path:
    bundle_root = validate_public_bundle_root(config.bundle_root)
    metadata_dir = config.metadata_dir.resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    release = read_json(bundle_root / "RELEASE.json")
    try:
        groups = license_groups(release)
    except ReleaseLicenseError as exc:
        raise DatasetPackageError(str(exc)) from exc
    rows = resource_rows(bundle_root)
    public_fields = release_public_fields(release)
    same_as = [
        value
        for value in (
            release.get("public_site_url"),
            release.get("zenodo_record_url"),
            release.get("hf_dataset_url"),
            release.get("cdn_base_url"),
        )
        if value
    ]
    croissant = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": f"{release.get('conference', 'Conference')} public data release {release.get('bundle_id')}",
        "license": [license_url(license_id) for license_id, _ in groups],
        "identifier": release.get("zenodo_doi") or release.get("bundle_id"),
        "distribution": rows,
    }
    if release.get("public_repo_url"):
        croissant["url"] = release.get("public_repo_url")
    if same_as:
        croissant["sameAs"] = same_as
    if release.get("cdn_base_url"):
        croissant["distributionBaseUrl"] = release.get("cdn_base_url")
    datapackage = {
        "profile": "data-package",
        "name": str(release.get("bundle_id", "public-release")),
        "licenses": [
            {"name": license_id, "path": license_url(license_id), "families": families}
            for license_id, families in groups
        ],
        "resources": rows,
        "public_identifiers": public_fields,
        "bundle_id": release.get("bundle_id"),
        "source_sha": release.get("source_sha"),
        "export_manifest_sha256": release.get("export_manifest_sha256"),
        "index_identifier": "index/index_manifest.json",
        "dataset_identifier": "metadata/dataset_manifest.json",
    }
    if release.get("public_repo_url"):
        datapackage["homepage"] = release.get("public_repo_url")
    sources = [value for value in same_as if value != release.get("cdn_base_url")]
    if sources:
        datapackage["sources"] = sources
    if release.get("cdn_base_url"):
        datapackage["cdn_base_url"] = release.get("cdn_base_url")
    manifest = {
        "schema_version": "1.0.0",
        "bundle_id": release.get("bundle_id"),
        "conference": release.get("conference"),
        "source_sha": release.get("source_sha"),
        "export_manifest_sha256": release.get("export_manifest_sha256"),
        "index_identifier": "index/index_manifest.json",
        "dataset_identifier": "metadata/dataset_manifest.json",
        "family_licenses": family_licenses(release),
        **public_fields,
        "resource_count": len(rows),
        "resources": rows,
        "generated_at": utc_now(),
    }
    write_json(metadata_dir / "croissant.json", croissant)
    write_json(metadata_dir / "datapackage.json", datapackage)
    write_text(metadata_dir / "CITATION.cff", citation_text(release))
    write_json(metadata_dir / "dataset_manifest.json", manifest)
    assert_no_private_markers(metadata_dir)
    receipt_path = config.receipt_path or metadata_dir / "dataset_package_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.public_dataset_package",
            "status": "succeeded",
            "exit_code": 0,
            "bundle_id": release.get("bundle_id"),
            "resource_count": len(rows),
            "metadata_dir": str(metadata_dir),
            "secret_values_recorded": False,
            "generated_at": utc_now(),
        },
    )
    return metadata_dir


def redacted_env_present(name: str) -> bool:
    return bool(os.getenv(name))


def env_value(name: str, environ: Mapping[str, str] = os.environ) -> str | None:
    value = environ.get(name)
    return value if value else None


def accepted_hf_token(environ: Mapping[str, str] = os.environ) -> tuple[str | None, str | None]:
    token = env_value("HF_TOKEN", environ)
    if token:
        return "HF_TOKEN", token
    token = env_value("HUGGINGFACE_API_KEY", environ)
    if token:
        return "HUGGINGFACE_API_KEY", token
    return None, None


def write_publication_receipt(
    *,
    path: Path,
    receipt_type: str,
    dry_run: bool,
    live: bool,
    target: str,
    provider: str | None = None,
    token_env_var: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "receipt_type": receipt_type,
        "status": "succeeded",
        "exit_code": 0,
        "dry_run": dry_run,
        "live": live,
        "target": target,
        "token_present": bool(token_env_var) if live else False,
        "token_env_var": token_env_var,
        "token_value": "REDACTED" if live and token_env_var else None,
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    if provider:
        payload["provider"] = provider
    if extra:
        payload.update(extra)
    write_json(path, payload)
    return payload


def write_operator_blocked_publication_receipt(
    *,
    path: Path,
    receipt_type: str,
    provider: str,
    target: str,
    blocker_code: str,
    live: bool,
    dry_run: bool,
    missing_env_vars: Sequence[str] = (),
    missing_approvals: Sequence[str] = (),
    token_env_var: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "receipt_type": receipt_type,
        "status": "operator_blocked",
        "exit_code": 2,
        "dry_run": dry_run,
        "live": live,
        "provider": provider,
        "target": target,
        "blocker_code": blocker_code,
        "blocked_by": [blocker_code],
        "missing_env_vars": list(missing_env_vars),
        "missing_approvals": list(missing_approvals),
        "token_present": bool(token_env_var),
        "token_env_var": token_env_var,
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)
    return payload


def infer_bundle_root(metadata_dir: Path, bundle_root: Path | None) -> Path | None:
    if bundle_root is not None:
        return validate_public_bundle_root(bundle_root)
    candidate = metadata_dir.resolve().parent
    if (candidate / "RELEASE.json").is_file() and (candidate / "CHECKSUMS.json").is_file():
        return validate_public_bundle_root(candidate)
    return None


def zenodo_upload_filename(path_name: str) -> str:
    return path_name.replace("/", "__")


def collect_cdn_sidecars(bundle_root: Path | None) -> list[Path]:
    if bundle_root is None:
        return []
    return sorted(bundle_root.rglob("*.cdn.json"), key=lambda item: item.relative_to(bundle_root).as_posix())


def write_cdn_sidecar_archive(bundle_root: Path, sidecars: Sequence[Path], archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for sidecar in sidecars:
            archive.write(sidecar, sidecar.relative_to(bundle_root).as_posix())


def collect_zenodo_upload_files(bundle_root: Path | None, metadata_dir: Path) -> list[UploadFile]:
    files: dict[str, Path] = {}

    def add(path: Path, filename: str | None = None) -> None:
        if path.is_file():
            provider_filename = zenodo_upload_filename(filename or path.name)
            existing = files.get(provider_filename)
            if existing is not None and existing != path:
                raise DatasetPackageError(f"Zenodo upload filename collision: {provider_filename}")
            files[provider_filename] = path

    for path in sorted(metadata_dir.rglob("*"), key=lambda item: item.relative_to(metadata_dir).as_posix()):
        if path.is_file():
            add(path, f"metadata/{path.relative_to(metadata_dir).as_posix()}")

    if bundle_root is not None:
        add(bundle_root / "RELEASE.json")
        add(bundle_root / "CHECKSUMS.json")
        index_dir = bundle_root / "index"
        if index_dir.is_dir():
            for path in sorted(index_dir.rglob("*manifest*.json"), key=lambda item: item.relative_to(bundle_root).as_posix()):
                add(path, path.relative_to(bundle_root).as_posix())

    return [UploadFile(path=path, filename=filename) for filename, path in sorted(files.items())]


def zenodo_base_url(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"


def redacted_provider_message(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("error") or payload.get("errors")
    else:
        message = payload
    if not message:
        return "provider error"
    text = re.sub(r"https?://\S+", "<provider-url>", str(message))
    text = re.sub(r"\b(hf|api|tok)_[A-Za-z0-9_-]+", "<redacted-token>", text)
    return text[:240]


def default_zenodo_request_json(method: str, url: str, *, token: str, **kwargs: Any) -> dict[str, Any]:
    import httpx

    headers = dict(kwargs.pop("headers", {}) or {})
    headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=120.0) as client:
        response = client.request(method, url, headers=headers, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise DatasetPackageError(f"Zenodo API returned non-JSON response: HTTP {response.status_code}") from exc
    if response.status_code >= 400:
        raise DatasetPackageError(f"Zenodo API HTTP {response.status_code}: {redacted_provider_message(payload)}")
    if not isinstance(payload, dict):
        raise DatasetPackageError("Zenodo API JSON object expected")
    return payload


def default_zenodo_upload_bytes(url: str, filename: str, data: bytes, *, token: str, **kwargs: Any) -> dict[str, Any]:
    import httpx

    headers = dict(kwargs.pop("headers", {}) or {})
    headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=120.0) as client:
        response = client.put(url, headers=headers, content=data, **kwargs)
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code >= 400:
        raise DatasetPackageError(f"Zenodo file upload HTTP {response.status_code}: {redacted_provider_message(payload)}")
    return payload if isinstance(payload, dict) else {"filename": filename, "size": len(data)}


def zenodo_metadata_payload(metadata_dir: Path) -> dict[str, Any]:
    manifest = read_json(metadata_dir / "dataset_manifest.json") if (metadata_dir / "dataset_manifest.json").is_file() else {}
    bundle_id = manifest.get("bundle_id") or "public-release"
    conference = manifest.get("conference") or manifest.get("conference_name") or "Conference"
    release = {"family_licenses": manifest.get("family_licenses")}
    try:
        groups = license_groups(release)
    except ReleaseLicenseError as exc:
        raise DatasetPackageError(f"dataset_manifest.json: {exc}") from exc
    if len(groups) != 1:
        # A Zenodo record takes one license id. Which one a mixed release declares is
        # the owner's decision at publication, not something to pick here.
        raise DatasetPackageError(
            f"zenodo_license_mixed: a Zenodo record takes one license; this release has {len(groups)}: "
            f"{license_summary(release)}"
        )
    return {
        "metadata": {
            "title": f"{conference} public data release {bundle_id}",
            "upload_type": "dataset",
            "description": f"Public metadata and release contract files for the {conference} conference pipeline public data release.",
            "creators": [{"name": "Conference Pipeline Team"}],
            "license": groups[0][0].lower(),
        }
    }


def publish_zenodo_live(
    *,
    metadata_dir: Path,
    sandbox: bool,
    bundle_root: Path | None,
    token: str,
    request_json: RequestJson | None = None,
    upload_bytes: UploadBytes | None = None,
) -> dict[str, Any]:
    request = request_json or default_zenodo_request_json
    upload = upload_bytes or default_zenodo_upload_bytes
    base_url = zenodo_base_url(sandbox)
    deposition = request("POST", f"{base_url}/api/deposit/depositions", token=token, json={})
    deposition_id = deposition.get("id")
    if deposition_id is None:
        raise DatasetPackageError("Zenodo deposition response missing id")
    deposition_id_text = str(deposition_id)
    links = deposition.get("links") if isinstance(deposition.get("links"), dict) else {}
    bucket_url = links.get("bucket")
    if not isinstance(bucket_url, str) or not bucket_url:
        raise DatasetPackageError("Zenodo deposition response missing bucket URL")

    uploaded_files: list[dict[str, Any]] = []
    sidecar_archive: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory() as temp_dir:
        upload_files = collect_zenodo_upload_files(bundle_root, metadata_dir)
        sidecars = collect_cdn_sidecars(bundle_root)
        if bundle_root is not None and sidecars:
            archive_path = Path(temp_dir) / "cdn-sidecars.zip"
            write_cdn_sidecar_archive(bundle_root, sidecars, archive_path)
            upload_files.append(UploadFile(path=archive_path, filename="cdn-sidecars.zip"))
            sidecar_archive = {
                "path": "cdn-sidecars.zip",
                "source_file_count": len(sidecars),
                "bytes": archive_path.stat().st_size,
            }
        for item in upload_files:
            data = item.path.read_bytes()
            upload(f"{bucket_url.rstrip('/')}/{quote(item.filename, safe='')}", item.filename, data, token=token)
            uploaded_files.append({"path": item.filename, "bytes": len(data)})

    request(
        "PUT",
        f"{base_url}/api/deposit/depositions/{deposition_id_text}",
        token=token,
        json=zenodo_metadata_payload(metadata_dir),
    )
    published = request(
        "POST",
        f"{base_url}/api/deposit/depositions/{deposition_id_text}/actions/publish",
        token=token,
    )
    published_links = published.get("links") if isinstance(published.get("links"), dict) else {}
    doi = published.get("doi") or deposition.get("doi")
    record_url = published_links.get("html") or published_links.get("record_html")
    if not record_url and not sandbox:
        record_url = f"https://zenodo.org/records/{deposition_id_text}"
    if not isinstance(doi, str) or not doi:
        raise DatasetPackageError("Zenodo publish response missing DOI")
    if not isinstance(record_url, str) or not record_url.startswith("https://"):
        raise DatasetPackageError("Zenodo publish response missing record URL")
    return {
        "deposition_id": deposition_id_text,
        "doi": doi,
        "record_url": record_url,
        "uploaded_files": uploaded_files,
        "uploaded_file_count": len(uploaded_files),
        "sidecar_archive": sidecar_archive,
    }


def default_hf_api_factory(token: str) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def hf_dataset_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def commit_attr(commit_info: Any, key: str) -> str | None:
    if isinstance(commit_info, dict):
        value = commit_info.get(key)
    else:
        value = getattr(commit_info, key, None)
    return str(value) if value else None


def classify_hf_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if any(marker in text for marker in ("namespace", "403", "forbidden", "permission", "scope")):
        return "hf_namespace_scope_missing"
    return "hf_api_error"


HF_COMMIT_OPERATIONS = 100  # the Hub's recommended operations per commit (storage-limits page)
HF_CARD_CONFIGS = (
    ("catalog", "catalog/papers.jsonl"),
    ("chunks", "index/corpus_chunks.parquet"),
    ("embeddings", "index/embeddings.parquet"),
    ("transcripts", "transcripts/index.jsonl"),
)


def _yaml_text(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def hf_dataset_card(release: Mapping[str, Any], relatives: Sequence[str]) -> str:
    """The dataset card (README.md) the HF dataset shows; spec D4 for the licence field.

    A release licensed per family states ``license: other`` with the per-family table;
    a release under one licence states that licence (as an SPDX id, lower-cased).
    """
    from conference_pipeline.publishing.release_licenses import license_groups

    bundle_id = str(release.get("bundle_id") or "unknown")
    conference = str(release.get("conference") or "Conference")
    groups = license_groups(release)
    lines = ["---", f"pretty_name: {_yaml_text(f'{conference} public release {bundle_id}')}"]
    if len(groups) == 1:
        lines.append(f"license: {groups[0][0].lower()}")
    else:
        lines += ["license: other", "license_name: per-family", "license_link: LICENSE"]
    present = set(relatives)
    configs = [(name, path) for name, path in HF_CARD_CONFIGS if path in present]
    if configs:
        lines.append("configs:")
        for name, path in configs:
            lines += [f"- config_name: {name}", f"  data_files: {_yaml_text(path)}"]
    lines += ["---", "", f"# {conference} public release `{bundle_id}`", ""]
    lines.append("Files are pinned to this release by the git tag named after the bundle id.")
    lines += ["", "## Licenses by family", "", "| License | Families |", "|---|---|"]
    lines += [f"| {license_id} | {', '.join(families)} |" for license_id, families in groups]
    links = [
        ("Reading site", release.get("public_site_url")),
        ("Data repository", release.get("public_repo_url")),
        ("Media", release.get("cdn_base_url")),
    ]
    present_links = [(label, url) for label, url in links if url]
    if present_links or release.get("previous_version_doi") or release.get("zenodo_doi"):
        lines += ["", "## Links", ""]
        lines += [f"- {label}: {url}" for label, url in present_links]
        if release.get("zenodo_doi"):
            lines.append(f"- DOI: {release['zenodo_doi']}")
        if release.get("previous_version_doi"):
            lines.append(f"- Previous version DOI: {release['previous_version_doi']}")
    return "\n".join(lines) + "\n"


def _bundle_root_of(metadata_dir: Path) -> Path | None:
    parent = metadata_dir.resolve().parent
    return parent if (parent / "RELEASE.json").is_file() and (parent / "CHECKSUMS.json").is_file() else None


def hf_upload_relatives(bundle_root: Path) -> list[str]:
    from conference_pipeline.publishing.release_layout import hf_upload_paths

    relatives = [
        path.relative_to(bundle_root).as_posix()
        for path in sorted(bundle_root.rglob("*"))
        if path.is_file() and not path.is_symlink() and ".git" not in path.relative_to(bundle_root).parts
    ]
    return hf_upload_paths(bundle_root, relatives)


HF_METADATA_FILES = (
    "metadata/CITATION.cff",
    "metadata/croissant.json",
    "metadata/datapackage.json",
    "metadata/dataset_manifest.json",
)


def verified_hf_upload(
    bundle_root: Path, *, secret_scanner: Any = None, generated: Mapping[str, bytes] | None = None
) -> list[str]:
    """The HF tree, refusing any file the bundle's own listings do not name and vouch for.

    Every uploaded file is listed, with its sha256, by ``CHECKSUMS.json`` (the export) or
    by ``index/index_manifest.json`` (the index), or is one of the fixed control and
    metadata files; each is scanned again for private markers, and the secrets gate runs
    over the exact upload (phase-7.1-r1 F1). A file added after the review never uploads.

    Every listed file the HF tree carries must be on disk: a missing one would become a
    deletion of the published copy (pr-49-r1 F2). ``generated`` holds the files the upload
    writes itself (the card and the licence); they get the same marker and secrets scans
    (pr-49-r1 F1).
    """

    from conference_pipeline.publishing.release_layout import hf_upload_paths

    from conference_pipeline._security.private_markers import has_publishing_marker_bytes
    from conference_pipeline.publishing.export_gate import GateClosed, no_secrets

    relatives = hf_upload_relatives(bundle_root)
    checksums = json.loads((bundle_root / "CHECKSUMS.json").read_text(encoding="utf-8"))
    index_path = bundle_root / "index" / "index_manifest.json"
    index = json.loads(index_path.read_text(encoding="utf-8")).get("output_checksums", {}) if index_path.is_file() else {}
    fixed = {"RELEASE.json", "CHECKSUMS.json", "index/index_manifest.json", *HF_METADATA_FILES}
    problems: list[str] = []
    listed = [*checksums, *(f"index/{name}" for name in index)]
    present = set(relatives)
    for rel in hf_upload_paths(bundle_root, listed):
        if rel not in present:
            problems.append(f"{rel}: listed but missing from the bundle")
    for name, data in sorted((generated or {}).items()):
        if has_publishing_marker_bytes(data):
            problems.append(f"{name}: the generated file holds a private marker")
    for rel in relatives:
        path = bundle_root / rel
        entry = checksums.get(rel)
        if rel == "index/index_manifest.json" or (rel in fixed and rel not in checksums):
            # Control and metadata files no listing covers. The index manifest is never
            # held to a row of its own, which would be a stale self-checksum
            # (phase-7.1-r2 F5); RELEASE.json is held to its CHECKSUMS.json row whenever
            # it has one (phase-7.1-r3 F2).
            expected = None
        elif rel in checksums:
            expected = entry.get("sha256") if isinstance(entry, dict) else None
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                # A listing without a digest vouches for nothing (phase-7.1-r2 F3).
                problems.append(f"{rel}: its CHECKSUMS.json entry has no sha256")
                continue
        elif rel.startswith("index/") and rel.removeprefix("index/") in index:
            expected = index[rel.removeprefix("index/")]
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                problems.append(f"{rel}: its index manifest entry has no sha256")
                continue
        else:
            problems.append(f"{rel}: listed by neither CHECKSUMS.json nor the index manifest")
            continue
        data = path.read_bytes()
        if expected is not None and hashlib.sha256(data).hexdigest() != expected:
            problems.append(f"{rel}: sha256 differs from its listing")
        if has_publishing_marker_bytes(data):
            problems.append(f"{rel}: holds a private marker")
    if problems:
        raise DatasetPackageError(f"hf_upload_unverified: {len(problems)} file(s); first {problems[0]}")
    with tempfile.TemporaryDirectory(prefix="hf-generated-") as staging:
        extra = []
        for name, data in sorted((generated or {}).items()):
            (Path(staging) / name).write_bytes(data)
            extra.append(Path(staging) / name)
        try:
            no_secrets([*(bundle_root / rel for rel in relatives), *extra], _scanner=secret_scanner)
        except GateClosed as exc:
            raise DatasetPackageError(f"hf_upload_unverified: secrets gate closed: {exc.reason}") from exc
    return relatives


def _remote_bundle_id(api: Any, repo_id: str, remote_files: set[str]) -> str | None:
    """The bundle id of the dataset's current version; refuses when it cannot be read.

    A remote that has a RELEASE.json the push cannot read would otherwise lose its
    version tag silently (phase-7.1-r1 F5).
    """
    if "RELEASE.json" not in remote_files:
        return None
    try:
        local = api.hf_hub_download(repo_id=repo_id, filename="RELEASE.json", repo_type="dataset")
        value = json.loads(Path(local).read_text(encoding="utf-8")).get("bundle_id")
    except Exception as exc:
        raise DatasetPackageError(f"hf_previous_release_unreadable: the dataset's RELEASE.json could not be read ({type(exc).__name__})") from exc
    if not valid_tag(value):
        raise DatasetPackageError("hf_previous_release_unreadable: the dataset's RELEASE.json names no valid bundle id")
    return value


def publish_hf_dataset_live(
    *,
    metadata_dir: Path,
    repo_id: str,
    token: str,
    hf_api_factory: HfApiFactory | None = None,
    bundle_root: Path | None = None,
    secret_scanner: Any = None,
) -> dict[str, Any]:
    """Publish the bundle's HF tree (v6.0 Phase 7.1, spec 6.5) and tag it with its bundle id.

    The dataset holds exactly the new tree: files the new version dropped are deleted in
    the same push. The version it replaces is tagged first when it has no tag yet, so it
    stays reachable (spec 6.6). A metadata directory that is not inside a bundle is
    uploaded alone, as before Phase 7.1.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    from conference_pipeline.publishing.release_licenses import license_text

    from conference_pipeline.publishing.release_layout import HF_GENERATED_FILES
    from conference_pipeline.publishing.surface_limits import check_sized

    bundle_root = bundle_root or _bundle_root_of(metadata_dir)
    if bundle_root is not None:
        release = json.loads((bundle_root / "RELEASE.json").read_text(encoding="utf-8"))
        new_id = release.get("bundle_id") if isinstance(release.get("bundle_id"), str) else None
        if not valid_tag(new_id):
            raise DatasetPackageError("RELEASE.json bundle_id is not a valid tag name")
        # Everything the push writes is built, scanned and sized before any provider call
        # (pr-49-r1 F1, F5).
        relatives = hf_upload_relatives(bundle_root)
        generated = {
            "README.md": hf_dataset_card(release, relatives).encode("utf-8"),
            "LICENSE": license_text(release).encode("utf-8"),
        }
        relatives = verified_hf_upload(bundle_root, secret_scanner=secret_scanner, generated=generated)
        planned = [(rel, (bundle_root / rel).lstat().st_size) for rel in relatives]
        planned += [(name, len(generated.get(name, b""))) for name in HF_GENERATED_FILES]
        limits = check_sized("hf", planned)
        if limits:
            raise DatasetPackageError(f"hf_upload_unverified: surface_limit: {limits[0]['path']}: {limits[0]['detail']}")
    api = (hf_api_factory or default_hf_api_factory)(token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    if bundle_root is None:
        return _publish_metadata_only(api, metadata_dir, repo_id)
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    tags = {ref.name for ref in getattr(api.list_repo_refs(repo_id, repo_type="dataset"), "tags", [])}
    if new_id in tags:
        # A bundle id names one set of bytes; a tag already holding it may point at other
        # content, and the push would report a tag it did not make (phase-7.1-r1 F6).
        raise DatasetPackageError(f"hf_tag_exists: {new_id} is already a tag on {repo_id}; bundle ids are write-once")
    tagged_previous = None
    old_id = _remote_bundle_id(api, repo_id, remote_files)
    if old_id is not None and old_id != new_id and old_id not in tags:
        api.create_tag(repo_id, tag=old_id, revision="main", repo_type="dataset")
        tagged_previous = old_id
    keep = set(relatives) | set(generated) | {".gitattributes"}
    operations: list[Any] = [CommitOperationDelete(path_in_repo=path) for path in sorted(remote_files - keep)]
    operations += [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(bundle_root / rel)) for rel in relatives]
    operations += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in sorted(generated.items())]
    commit_info = None
    total = (len(operations) + HF_COMMIT_OPERATIONS - 1) // HF_COMMIT_OPERATIONS
    for number in range(total):
        part = operations[number * HF_COMMIT_OPERATIONS : (number + 1) * HF_COMMIT_OPERATIONS]
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=part,
            commit_message=f"Publish {new_id} ({number + 1}/{total})",
        )
    commit_sha = commit_attr(commit_info, "oid") or commit_attr(commit_info, "commit_hash")
    api.create_tag(repo_id, tag=new_id, revision=commit_sha or "main", repo_type="dataset")
    commit_url = commit_attr(commit_info, "commit_url")
    if not commit_url and commit_sha:
        commit_url = f"{hf_dataset_url(repo_id)}/commit/{commit_sha}"
    return {
        "dataset_url": hf_dataset_url(repo_id),
        "commit_sha": commit_sha,
        "commit_url": commit_url,
        "tag": new_id,
        "tagged_previous": tagged_previous,
        "deleted_files": sorted(remote_files - keep),
        "uploaded_files": sorted([*relatives, *generated]),
        "uploaded_file_count": len(relatives) + len(generated),
        "commit_count": total,
    }


def _publish_metadata_only(api: Any, metadata_dir: Path, repo_id: str) -> dict[str, Any]:
    commit_info = api.upload_folder(
        folder_path=metadata_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Publish {repo_id} dataset metadata",
    )
    commit_sha = commit_attr(commit_info, "commit_hash") or commit_attr(commit_info, "oid")
    commit_url = commit_attr(commit_info, "commit_url")
    if not commit_url and commit_sha:
        commit_url = f"{hf_dataset_url(repo_id)}/commit/{commit_sha}"
    files = [
        path.relative_to(metadata_dir).as_posix()
        for path in sorted(metadata_dir.rglob("*"), key=lambda item: item.relative_to(metadata_dir).as_posix())
        if path.is_file()
    ]
    return {
        "dataset_url": hf_dataset_url(repo_id),
        "commit_sha": commit_sha,
        "commit_url": commit_url,
        "uploaded_files": files,
        "uploaded_file_count": len(files),
    }


def deposit_zenodo(
    *,
    metadata_dir: Path,
    receipt_path: Path,
    sandbox: bool,
    live: bool,
    dry_run: bool,
    ack_live_doi: bool,
    bundle_root: Path | None = None,
    request_json: RequestJson | None = None,
    upload_bytes: UploadBytes | None = None,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    target = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    if live and not ack_live_doi:
        return write_operator_blocked_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.zenodo_deposit",
            provider="zenodo",
            target=target,
            blocker_code="missing_live_ack",
            live=live,
            dry_run=dry_run,
            missing_approvals=["ack_live_doi"],
            extra={"doi": None, "record_url": None},
        )
    token = env_value("ZENODO_TOKEN", environ)
    if live and not token:
        return write_operator_blocked_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.zenodo_deposit",
            provider="zenodo",
            target=target,
            blocker_code="missing_zenodo_token",
            live=live,
            dry_run=dry_run,
            missing_env_vars=["ZENODO_TOKEN"],
            extra={"doi": None, "record_url": None},
        )
    if not (metadata_dir / "croissant.json").exists():
        raise DatasetPackageError("metadata_dir must contain croissant.json")
    if not live or dry_run:
        return write_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.zenodo_deposit",
            dry_run=dry_run,
            live=live,
            target=target,
            provider="zenodo",
            token_env_var="ZENODO_TOKEN" if live else None,
        )
    assert token is not None
    try:
        result = publish_zenodo_live(
            metadata_dir=metadata_dir,
            sandbox=sandbox,
            bundle_root=infer_bundle_root(metadata_dir, bundle_root),
            token=token,
            request_json=request_json,
            upload_bytes=upload_bytes,
        )
    except Exception as exc:
        return write_operator_blocked_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.zenodo_deposit",
            provider="zenodo",
            target=target,
            blocker_code="zenodo_api_error",
            live=live,
            dry_run=dry_run,
            token_env_var="ZENODO_TOKEN",
            extra={"doi": None, "record_url": None, "provider_error": redacted_provider_message(exc)},
        )
    return write_publication_receipt(
        path=receipt_path,
        receipt_type="conference_pipeline.zenodo_deposit",
        dry_run=dry_run,
        live=live,
        target=target,
        provider="zenodo",
        token_env_var="ZENODO_TOKEN",
        extra=result,
    )


def push_hf_dataset(
    *,
    metadata_dir: Path,
    repo_id: str,
    receipt_path: Path,
    dry_run: bool,
    live: bool,
    ack_live_push: bool,
    hf_api_factory: HfApiFactory | None = None,
    environ: Mapping[str, str] = os.environ,
    secret_scanner: Any = None,
) -> dict[str, Any]:
    if live and not ack_live_push:
        return write_operator_blocked_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.hf_dataset_push",
            provider="huggingface",
            target=repo_id,
            blocker_code="missing_live_ack",
            live=live,
            dry_run=dry_run,
            missing_approvals=["ack_live_push"],
            extra={"dataset_url": None, "commit_sha": None, "commit_url": None},
        )
    token_env_var, token = accepted_hf_token(environ)
    if live and not token:
        return write_operator_blocked_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.hf_dataset_push",
            provider="huggingface",
            target=repo_id,
            blocker_code="missing_hf_token",
            live=live,
            dry_run=dry_run,
            missing_env_vars=["HF_TOKEN", "HUGGINGFACE_API_KEY"],
            extra={"dataset_url": None, "commit_sha": None, "commit_url": None},
        )
    if not (metadata_dir / "datapackage.json").exists():
        raise DatasetPackageError("metadata_dir must contain datapackage.json")
    if not live or dry_run:
        return write_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.hf_dataset_push",
            dry_run=dry_run,
            live=live,
            target=repo_id,
            provider="huggingface",
            token_env_var=token_env_var if live else None,
        )
    assert token is not None
    try:
        result = publish_hf_dataset_live(
            metadata_dir=metadata_dir,
            repo_id=repo_id,
            token=token,
            hf_api_factory=hf_api_factory,
            secret_scanner=secret_scanner,
        )
    except Exception as exc:
        return write_operator_blocked_publication_receipt(
            path=receipt_path,
            receipt_type="conference_pipeline.hf_dataset_push",
            provider="huggingface",
            target=repo_id,
            blocker_code=classify_hf_exception(exc),
            live=live,
            dry_run=dry_run,
            token_env_var=token_env_var,
            extra={
                "dataset_url": None,
                "commit_sha": None,
                "commit_url": None,
                "provider_error": redacted_provider_message(exc),
            },
        )
    return write_publication_receipt(
        path=receipt_path,
        receipt_type="conference_pipeline.hf_dataset_push",
        dry_run=dry_run,
        live=live,
        target=repo_id,
        provider="huggingface",
        token_env_var=token_env_var,
        extra=result,
    )


def package_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package public dataset metadata")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    return parser.parse_args(argv)


def package_main(argv: Sequence[str] | None = None) -> int:
    args = package_args(argv)
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(
            bundle_root=args.bundle_root,
            metadata_dir=args.metadata_dir,
            receipt_path=args.receipt_path,
        )
    )
    print(f"Dataset metadata written: {metadata_dir}")
    return 0


def zenodo_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run, sandbox, or live Zenodo deposit")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ack-live-doi", action="store_true")
    return parser.parse_args(argv)


def zenodo_main(argv: Sequence[str] | None = None) -> int:
    args = zenodo_args(argv)
    try:
        receipt = deposit_zenodo(
            metadata_dir=args.metadata_dir,
            receipt_path=args.receipt_path,
            sandbox=args.sandbox,
            live=args.live,
            dry_run=args.dry_run or not args.live,
            ack_live_doi=args.ack_live_doi,
        )
    except DatasetPackageError as exc:
        print(str(exc))
        return 2
    print(f"Receipt: {args.receipt_path}")
    return int(receipt.get("exit_code", 0))


def hf_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or live Hugging Face dataset push")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ack-live-push", action="store_true")
    return parser.parse_args(argv)


def hf_main(argv: Sequence[str] | None = None) -> int:
    args = hf_args(argv)
    try:
        receipt = push_hf_dataset(
            metadata_dir=args.metadata_dir,
            repo_id=args.repo_id,
            receipt_path=args.receipt_path,
            dry_run=args.dry_run or not args.live,
            live=args.live,
            ack_live_push=args.ack_live_push,
        )
    except DatasetPackageError as exc:
        print(str(exc))
        return 2
    print(f"Receipt: {args.receipt_path}")
    return int(receipt.get("exit_code", 0))
