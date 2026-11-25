"""Record a redacted, fail-closed public release closeout."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline._security.private_markers import has_publishing_marker, publishing_marker_hits
from conference_pipeline.publishing.public_index import sha256_file, validate_public_bundle_root


RECEIPT_TYPE = "conference_pipeline.public_release_closeout"


class PublicCloseoutError(RuntimeError):
    """Raised when closeout evidence is malformed."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicCloseoutError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_text(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    return None if "\x00" in text else text


def scan_public_tree(root: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    if not root.is_dir():
        raise PublicCloseoutError(f"public payload directory missing: {root}")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if has_publishing_marker(rel):
            findings.append({"path": rel, "reason": "private_marker_in_path"})
            continue
        if not path.is_file():
            continue
        text = _safe_text(path)
        if text is None:
            continue
        for marker in publishing_marker_hits(text):
            findings.append({"path": rel, "reason": f"private_marker:{marker}"})
    return {
        "status": "clean" if not findings else "failed",
        "scope": "staged_public_payload",
        "private_marker_count": len(findings),
        "findings": findings,
    }


def _provider_name(stage: str) -> str:
    return {
        "live_github": "github",
        "live_r2": "r2",
        "live_pages": "pages",
        "live_hf": "huggingface",
        "live_zenodo": "zenodo",
    }.get(stage, stage.removeprefix("live_"))


def _blocked_providers(paths: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for provider, path in paths:
        payload = read_json(path)
        for stage in payload.get("blocked_stages", []):
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage", provider))
            blocked.append(
                {
                    "provider": _provider_name(stage_name),
                    "stage": stage_name,
                    "receipt": path.name,
                    "missing_approvals": sorted(str(item) for item in stage.get("missing_approvals", [])),
                    "missing_receipts": sorted(str(item) for item in stage.get("missing_receipts", [])),
                }
            )
    return sorted(blocked, key=lambda item: (item["provider"], item["stage"]))


def record_closeout(
    *,
    bundle_root: Path,
    public_payload_root: Path,
    liveness_receipt: Path,
    metadata_receipt: Path,
    provider_receipts: Sequence[tuple[str, Path]],
    receipt_path: Path,
    focused_tests: Sequence[str] = (),
) -> dict[str, Any]:
    root = validate_public_bundle_root(bundle_root)
    release = read_json(root / "RELEASE.json")
    checksums = root / "CHECKSUMS.json"
    if not checksums.is_file():
        raise PublicCloseoutError(f"checksum manifest missing: {checksums}")
    liveness = read_json(liveness_receipt)
    metadata = read_json(metadata_receipt)
    safety = scan_public_tree(public_payload_root)
    blocked = _blocked_providers(provider_receipts)
    liveness_failed = int(liveness.get("failed_count", 0) or 0)
    liveness_passed = liveness.get("status") == "succeeded" and liveness_failed == 0
    metadata_passed = metadata.get("status") == "succeeded"
    status = "succeeded" if liveness_passed and metadata_passed and safety["status"] == "clean" and not blocked else "operator_blocked"

    payload = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "status": status,
        "exit_code": 0 if status == "succeeded" else 2,
        "bundle_id": release.get("bundle_id") or root.name,
        "source_sha": release.get("source_sha"),
        "export_manifest_sha256": release.get("export_manifest_sha256"),
        "bundle_checksum_manifest_sha256": sha256_file(checksums),
        "public_surfaces": {
            "public_repo": "public-repo",
            "site": "site",
            "index": "index",
            "dataset_metadata": "metadata",
            "release_identifiers": {
                key: release.get(key)
                for key in (
                    "public_repo_url",
                    "public_site_url",
                    "cdn_base_url",
                    "hf_dataset_url",
                    "zenodo_doi",
                    "zenodo_record_url",
                )
            },
        },
        "liveness": {
            "receipt": liveness_receipt.name,
            "status": liveness.get("status"),
            "checked_count": liveness.get("checked_count", 0),
            "passed_count": liveness.get("passed_count", 0),
            "failed_count": liveness_failed,
            "blocked_count": liveness.get("blocked_count", 0),
        },
        "metadata_agreement": {
            "receipt": metadata_receipt.name,
            "status": metadata.get("status"),
            "mismatch_count": metadata.get("mismatch_count", 0),
        },
        "safety": safety,
        "provider_blockers": blocked,
        "deferred_items": sorted(
            {
                *(["provider_live_publication"] if blocked else []),
                *(["public_url_liveness"] if not liveness_passed else []),
            }
        ),
        "focused_tests": list(focused_tests),
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    write_json(receipt_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a redacted public release closeout")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--public-payload-root", type=Path, required=True)
    parser.add_argument("--liveness-receipt", type=Path, required=True)
    parser.add_argument("--metadata-receipt", type=Path, required=True)
    parser.add_argument("--provider-receipt", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--focused-test", action="append", default=[])
    parser.add_argument("--receipt-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provider_receipts: list[tuple[str, Path]] = []
    for value in args.provider_receipt:
        if "=" not in value:
            raise SystemExit("--provider-receipt must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        provider_receipts.append((name, Path(raw_path)))
    payload = record_closeout(
        bundle_root=args.bundle_root,
        public_payload_root=args.public_payload_root,
        liveness_receipt=args.liveness_receipt,
        metadata_receipt=args.metadata_receipt,
        provider_receipts=provider_receipts,
        receipt_path=args.receipt_path,
        focused_tests=args.focused_test,
    )
    print(f"Closeout receipt: {args.receipt_path}")
    print(f"Status: {payload['status']}")
    return 0 if payload["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
