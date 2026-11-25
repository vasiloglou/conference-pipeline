"""CDN upload planning and public URL liveness checks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import quote, urlparse

from conference_pipeline.publishing.public_index import validate_public_bundle_root


class PublicCdnError(RuntimeError):
    """Raised when CDN planning or liveness checks fail closed."""


PUBLIC_LIVENESS_USER_AGENT = (
    "conference-pipeline-public-liveness/1.0 "
    "(+https://github.com/relationalai/conference-pipeline)"
)


@dataclass(frozen=True)
class CdnUploadConfig:
    bundle_root: Path
    receipt_path: Path
    bundle_id: str | None = None
    provider: str = "r2"
    bucket: str | None = None
    prefix: str = ""
    dry_run: bool = True
    live: bool = False
    approved_by: str | None = None


@dataclass(frozen=True)
class CdnUploadItem:
    sidecar_path: str
    cdn_url: str
    object_key: str
    sha256: str
    size_bytes: int
    mime_type: str


@dataclass(frozen=True)
class UrlCheckResult:
    url: str
    status: str
    http_status: int | None
    reason: str | None = None


def _source_path_for_sidecar(bundle_root: Path, sidecar: Path) -> Path:
    """Map a public sidecar back to its manifest-approved source file."""
    conference_root = bundle_root.parents[2]
    parts = list(sidecar.relative_to(bundle_root).parts)
    if parts and parts[0] == "public-repo":
        parts.pop(0)
    if "payloads" in parts:
        parts.remove("payloads")
    name = parts.pop()
    if not name.endswith(".cdn.json"):
        raise PublicCdnError(f"invalid CDN sidecar name: {sidecar}")
    parts.append(name.removesuffix(".cdn.json"))
    return conference_root.joinpath(*parts)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aws_signature(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload_hash: str,
    access_key: str,
    secret_key: str,
    timestamp: dt.datetime,
) -> str:
    parsed = urlparse(url)
    canonical_uri = quote(parsed.path or "/", safe="/-_.~")
    canonical_query = parsed.query
    canonical_headers = "".join(
        f"{name.lower()}:{' '.join(value.strip().split())}\n"
        for name, value in sorted(headers.items(), key=lambda item: item[0].lower())
    )
    signed_headers = ";".join(name.lower() for name in sorted(headers, key=str.lower))
    canonical_request = "\n".join(
        (method, canonical_uri, canonical_query, canonical_headers, signed_headers, payload_hash)
    )
    date = timestamp.strftime("%Y%m%d")
    amz_date = timestamp.strftime("%Y%m%dT%H%M%SZ")
    scope = f"{date}/auto/s3/aws4_request"
    string_to_sign = "\n".join(
        ("AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest())
    )

    def sign(key: bytes, value: str) -> bytes:
        return hmac.new(key, value.encode(), hashlib.sha256).digest()

    signing_key = sign(sign(sign(sign(("AWS4" + secret_key).encode(), date), "auto"), "s3"), "aws4_request")
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicCdnError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def validate_public_url(url: str) -> None:
    if url.startswith("file://") or "Google Drive" in url or "drive.google.com" in url:
        raise PublicCdnError(f"non-public URL is not allowed: {url}")
    if url.startswith("/") or "/Users/" in url or "/Volumes/" in url:
        raise PublicCdnError(f"local path is not allowed as public URL: {url}")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PublicCdnError(f"HTTP(S) URL expected: {url}")


def object_key_for(url: str, prefix: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    normalized_prefix = prefix.strip("/")
    if normalized_prefix and (path == normalized_prefix or path.startswith(normalized_prefix + "/")):
        return path
    return "/".join(part.strip("/") for part in (normalized_prefix, path) if part.strip("/"))


def build_cdn_upload_plan(config: CdnUploadConfig) -> list[CdnUploadItem]:
    bundle_root = validate_public_bundle_root(config.bundle_root)
    items: list[CdnUploadItem] = []
    by_object_key: dict[str, CdnUploadItem] = {}
    for sidecar in sorted(bundle_root.rglob("*.cdn.json"), key=lambda path: path.relative_to(bundle_root).as_posix()):
        data = read_json(sidecar)
        missing = [
            field
            for field in ("cdn_url", "sha256", "size_bytes", "mime_type")
            if data.get(field) in {None, ""}
        ]
        if missing:
            raise PublicCdnError(f"{sidecar.relative_to(bundle_root).as_posix()} missing {', '.join(missing)}")
        url = str(data["cdn_url"])
        validate_public_url(url)
        item = CdnUploadItem(
            sidecar_path=sidecar.relative_to(bundle_root).as_posix(),
            cdn_url=url,
            object_key=object_key_for(url, config.prefix),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
            mime_type=str(data["mime_type"]),
        )
        prior = by_object_key.get(item.object_key)
        if prior is not None and (prior.sha256, prior.size_bytes) != (item.sha256, item.size_bytes):
            raise PublicCdnError(f"duplicate CDN object key has conflicting content: {item.object_key}")
        by_object_key[item.object_key] = item
    return [by_object_key[key] for key in sorted(by_object_key)]


def _r2_upload_item(item: CdnUploadItem, *, bundle_root: Path, bucket: str) -> dict[str, Any]:
    source = _source_path_for_sidecar(bundle_root, bundle_root / item.sidecar_path)
    if not source.is_file():
        raise PublicCdnError(f"CDN source file missing: {item.sidecar_path}")
    if source.stat().st_size != item.size_bytes or _sha256_path(source) != item.sha256:
        raise PublicCdnError(f"CDN source identity mismatch: {item.sidecar_path}")
    endpoint = os.getenv("CLOUDFLARE_R2_ENDPOINT")
    access_key = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID")
    secret_key = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
    missing = [
        name
        for name, value in (
            ("CLOUDFLARE_R2_ENDPOINT", endpoint),
            ("CLOUDFLARE_R2_ACCESS_KEY_ID", access_key),
            ("CLOUDFLARE_R2_SECRET_ACCESS_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise PublicCdnError(f"missing R2 configuration: {', '.join(missing)}")
    assert endpoint and access_key and secret_key
    import requests

    url = endpoint.rstrip("/") + "/" + quote(bucket, safe="/-_.~") + "/" + quote(item.object_key, safe="/-_.~")
    timestamp = dt.datetime.now(dt.UTC)
    payload_hash = item.sha256
    headers = {
        "host": urlparse(url).netloc,
        "content-type": item.mime_type,
        "content-length": str(item.size_bytes),
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": timestamp.strftime("%Y%m%dT%H%M%SZ"),
    }
    headers["authorization"] = _aws_signature(
        method="PUT",
        url=url,
        headers=headers,
        payload_hash=payload_hash,
        access_key=access_key,
        secret_key=secret_key,
        timestamp=timestamp,
    )
    with source.open("rb") as handle:
        response = requests.put(url, data=handle, headers=headers, timeout=600)
    if response.status_code < 200 or response.status_code >= 300:
        raise PublicCdnError(f"R2 upload failed for {item.object_key}: HTTP {response.status_code}")
    return {
        "object_key": item.object_key,
        "sha256": item.sha256,
        "size_bytes": item.size_bytes,
        "etag": response.headers.get("ETag", "").strip('"'),
    }


def write_cdn_upload_receipt(config: CdnUploadConfig, items: list[CdnUploadItem]) -> None:
    payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.public_cdn_upload",
        "status": "succeeded",
        "exit_code": 0,
        "dry_run": config.dry_run,
        "live": config.live,
        "provider": config.provider,
        "bucket": config.bucket,
        "prefix": config.prefix,
        "planned_upload_count": len(items),
        "planned_uploads": [asdict(item) for item in items],
        "generated_at": utc_now(),
    }
    if config.bundle_id is not None:
        payload["bundle_id"] = config.bundle_id
    write_json(config.receipt_path, payload)


def is_completed_live_upload_receipt(
    payload: dict[str, Any],
    bundle_id: str,
) -> bool:
    """Return whether a CDN receipt proves a durable, complete live upload.

    Planning rows describe intent only.  The completion predicate requires an
    explicit non-dry live receipt and one uploaded-object row for every planned
    row, with exact key, digest, and size agreement.  It performs no I/O.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("receipt_type") != "conference_pipeline.public_cdn_upload":
        return False
    if payload.get("status") != "succeeded" or payload.get("exit_code") != 0:
        return False
    if payload.get("live") is not True or type(payload.get("dry_run")) is not bool:
        return False
    if payload.get("dry_run") is not False or payload.get("bundle_id") != bundle_id:
        return False
    planned = payload.get("planned_uploads")
    uploaded = payload.get("uploaded_objects")
    planned_count = payload.get("planned_upload_count")
    uploaded_count = payload.get("uploaded_count")
    if (
        not isinstance(planned, list)
        or not isinstance(uploaded, list)
        or type(planned_count) is not int
        or type(uploaded_count) is not int
        or planned_count <= 0
        or uploaded_count <= 0
        or planned_count != len(planned)
        or uploaded_count != len(uploaded)
        or planned_count != uploaded_count
    ):
        return False

    def row_key(row: Any) -> str | None:
        if not isinstance(row, dict):
            return None
        value = row.get("object_key")
        return value if isinstance(value, str) and value else None

    planned_by_key: dict[str, dict[str, Any]] = {}
    uploaded_by_key: dict[str, dict[str, Any]] = {}
    for row in planned:
        key = row_key(row)
        if key is None or key in planned_by_key:
            return False
        planned_by_key[key] = row
    for row in uploaded:
        key = row_key(row)
        if key is None or key in uploaded_by_key:
            return False
        uploaded_by_key[key] = row
    if set(planned_by_key) != set(uploaded_by_key):
        return False
    for key, planned_row in planned_by_key.items():
        uploaded_row = uploaded_by_key[key]
        if (
            uploaded_row.get("object_key") != planned_row.get("object_key")
            or uploaded_row.get("sha256") != planned_row.get("sha256")
            or uploaded_row.get("size_bytes") != planned_row.get("size_bytes")
        ):
            return False
    return True


def plan_public_media_upload(config: CdnUploadConfig) -> list[CdnUploadItem]:
    if config.live and config.dry_run:
        # A dry run never uploads; the two flags together are a misuse (phase-7.1-r1 F4).
        raise PublicCdnError("--live and --dry-run conflict: a dry run uploads nothing")
    if config.live and not config.bucket:
        raise PublicCdnError("--live requires --bucket")
    if config.live and not (config.approved_by or "").strip():
        # A live upload is publication: the owner's name is required on every path, not
        # only through release publish (pr-49-r1 F3).
        raise PublicCdnError("--live requires --approved-by with the approver's name")
    items = build_cdn_upload_plan(config)
    if config.live:
        from conference_pipeline.publishing.surface_limits import limit_for

        ceiling = limit_for("r2").max_file_bytes
        over = [item for item in items if item.size_bytes > ceiling]
        if over:
            raise PublicCdnError(
                f"surface_limit: {over[0].sidecar_path}: {over[0].size_bytes} bytes is over the {ceiling}-byte r2 file limit"
            )
    if config.live:
        uploaded = [_r2_upload_item(item, bundle_root=validate_public_bundle_root(config.bundle_root), bucket=config.bucket or "") for item in items]
    else:
        uploaded = []
    write_cdn_upload_receipt(config, items)
    if config.live:
        payload = read_json(config.receipt_path)
        payload["uploaded_count"] = len(uploaded)
        payload["uploaded_objects"] = uploaded
        write_json(config.receipt_path, payload)
    return items


URL_PATTERN = re.compile(r"(?:https?|file)://[^\s\"'<>`]+")
PROJECT_RELATIVE_SOURCE_PREFIXES = ("docs/",)


def source_label(root: Path, path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    try:
        project_relative = resolved_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name
    if project_relative.startswith(PROJECT_RELATIVE_SOURCE_PREFIXES):
        return project_relative
    return path.name


def normalized_url(url: str) -> str:
    return url.rstrip("/")


def liveness_skip_urls(root: Path) -> set[str]:
    skip: set[str] = set()
    release_path = root / "RELEASE.json"
    if release_path.is_file():
        release = read_json(release_path)
        value = release.get("cdn_base_url")
        if isinstance(value, str) and value.strip():
            skip.add(normalized_url(value.strip()))
    return skip


def collect_public_url_entries(bundle_root: Path, extra_metadata: Sequence[Path] = ()) -> dict[str, set[str]]:
    root = validate_public_bundle_root(bundle_root)
    skip_urls = liveness_skip_urls(root)
    urls: dict[str, set[str]] = {}
    for sidecar in root.rglob("*.cdn.json"):
        data = read_json(sidecar)
        if data.get("cdn_url"):
            urls.setdefault(str(data["cdn_url"]), set()).add(source_label(root, sidecar))
    for metadata in extra_metadata:
        if not metadata.exists():
            continue
        text = metadata.read_text(encoding="utf-8", errors="ignore")
        for match in URL_PATTERN.findall(text):
            url = match.rstrip(".,;:)]}`")
            if normalized_url(url) in skip_urls:
                continue
            urls.setdefault(url, set()).add(source_label(root, metadata))
    return urls


def collect_public_urls(bundle_root: Path, extra_metadata: Sequence[Path] = ()) -> list[str]:
    return sorted(collect_public_url_entries(bundle_root, extra_metadata))


def checked_source_labels(
    root: Path,
    url_sources: dict[str, set[str]],
    extra_metadata: Sequence[Path],
) -> list[str]:
    sources = {source for source_set in url_sources.values() for source in source_set}
    for metadata in extra_metadata:
        if metadata.exists():
            sources.add(source_label(root, metadata))
    return sorted(sources)


def default_http_get(url: str) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": PUBLIC_LIVENESS_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - public URL liveness probe
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def check_public_urls(
    bundle_root: Path,
    *,
    receipt_path: Path,
    extra_metadata: Sequence[Path] = (),
    http_get: Callable[[str], int] = default_http_get,
) -> list[UrlCheckResult]:
    results: list[UrlCheckResult] = []
    root = validate_public_bundle_root(bundle_root)
    url_sources = collect_public_url_entries(root, extra_metadata)
    for url in sorted(url_sources):
        validate_public_url(url)
        try:
            status_code = http_get(url)
            reason = None if status_code == 200 else "http_status_not_200"
        except Exception:
            status_code = None
            reason = "transport_error"
        results.append(
            UrlCheckResult(
                url=url,
                status="passed" if status_code == 200 else "failed",
                http_status=status_code,
                reason=reason,
            )
        )
    failed = [result for result in results if result.status == "failed"]
    sources = checked_source_labels(root, url_sources, extra_metadata)
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.public_url_liveness",
            "status": "failed" if failed else "succeeded",
            "exit_code": 1 if failed else 0,
            "checked_count": len(results),
            "passed_count": len(results) - len(failed),
            "failed_count": len(failed),
            "blocked_count": 0,
            "checked_sources": sources,
            "checked_urls": [result.url for result in results],
            "results": [asdict(result) for result in results],
            "secret_values_recorded": False,
            "generated_at": utc_now(),
        },
    )
    if failed:
        raise PublicCdnError(f"{len(failed)} public URL(s) failed liveness")
    return results


def parse_upload_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run public media CDN uploads")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--provider", default="r2")
    parser.add_argument("--bucket")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approved-by", help="The owner who approved this live upload (required with --live).")
    return parser.parse_args(argv)


def upload_main(argv: Sequence[str] | None = None) -> int:
    args = parse_upload_args(argv)
    items = plan_public_media_upload(
        CdnUploadConfig(
            bundle_root=args.bundle_root,
            receipt_path=args.receipt_path,
            bundle_id=args.bundle_root.name,
            provider=args.provider,
            bucket=args.bucket,
            prefix=args.prefix,
            dry_run=args.dry_run or not args.live,
            live=args.live,
            approved_by=args.approved_by,
        )
    )
    print(f"Planned uploads: {len(items)}")
    print(f"Receipt: {args.receipt_path}")
    return 0


def parse_check_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check public release URLs")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--extra-metadata", type=Path, action="append", default=[])
    parser.add_argument("--mock-status", type=int, help="Use a fixed HTTP status for preview/offline proof")
    return parser.parse_args(argv)


def check_main(argv: Sequence[str] | None = None) -> int:
    args = parse_check_args(argv)
    http_get = (lambda _url: args.mock_status) if args.mock_status is not None else default_http_get
    try:
        check_public_urls(
            args.bundle_root,
            receipt_path=args.receipt_path,
            extra_metadata=args.extra_metadata,
            http_get=http_get,
        )
    except PublicCdnError as exc:
        print(str(exc))
        return 1
    print(f"Receipt: {args.receipt_path}")
    return 0
