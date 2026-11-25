"""Fail-closed secrets gate for conference export pipelines."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

import jsonschema


class GateClosed(RuntimeError):
    """Raised when the export gate blocks an export."""

    def __init__(self, reason: str, receipt: dict) -> None:
        super().__init__(reason)
        self.reason = reason
        self.receipt = receipt


@dataclasses.dataclass
class GateReceipt:
    gate_id: str
    status: Literal["passed", "failed"]
    timestamp_utc: str
    findings: list[dict]
    scanner_version: str | None = None
    manifest_sha256: str | None = None
    notes: str | None = None


def write_gate_receipt(receipt: GateReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataclasses.asdict(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_secret_like_path(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    if name in {
        "cookies.json",
        "cookie.json",
        "token.json",
        "tokens.json",
        "secrets.json",
        "secret.json",
        "credentials.json",
        "credential.json",
        "password.json",
        "passwords.json",
        "api_key.json",
        "apikey.json",
    }:
        return True

    parts = [part for part in re.split(r"[^a-z0-9]+", stem) if part]
    if any(part in {"cookie", "cookies", "secret", "secrets", "password", "passwords"} for part in parts):
        return True
    if any(part in {"credential", "credentials"} for part in parts):
        return True
    if "apikey" in parts or ("api" in parts and "key" in parts):
        return True
    if any(part in {"token", "tokens"} for part in parts):
        return any(
            part in {"access", "auth", "bearer", "id", "oauth", "refresh", "session"}
            for part in parts
        )
    return False


def _path_scan(paths: list[Path]) -> list[dict]:
    findings: list[dict] = []
    for path in paths:
        if _is_secret_like_path(path):
            findings.append({"path": str(path), "reason": "secret_like_filename"})
    return findings


_DIGEST_MANIFESTS = frozenset({"public_repo_manifest.json", "CHECKSUMS.json", "site_manifest.json"})


def _gitleaks_content_scan(
    paths: list[Path],
    *,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> list[dict]:
    if _scanner is not None:
        return _scanner(paths)

    binary = shutil.which("gitleaks")
    if binary is None:
        raise GateClosed(reason="gitleaks_not_installed", receipt={})

    if not paths:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        originals: dict[str, Path] = {}
        for index, path in enumerate(paths):
            dest = tmp / f"{index:06d}_{path.name}"
            dest.write_bytes(path.read_bytes())
            originals[dest.name] = path

        report_path = tmp / "gitleaks_report.json"
        result = subprocess.run(
            [
                binary,
                "detect",
                "--source", str(tmp),
                "--no-git",
                "--report-format", "json",
                "--report-path", str(report_path),
                "--exit-code", "0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # With --exit-code 0 a scan that ran exits 0 whatever it found, and writes a list.
        # Anything else is a scan that did not run, never a clean one (6b review r3 F3).
        if result.returncode != 0:
            raise GateClosed(reason=f"gitleaks_failed_exit_{result.returncode}", receipt={})
        try:
            raw = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise GateClosed(reason="gitleaks_report_unreadable", receipt={}) from None
        if not isinstance(raw, list):
            raise GateClosed(reason="gitleaks_report_unreadable", receipt={})
        # The generated manifests map paths to sha256 digests; a path holding "api" (a
        # YouTube id, 6b) makes a digest look like a key. A value is a digest only when it is
        # the sha256 of a file this scan reads, never by its spelling (6b review r1 F3), and
        # only in a file named exactly as a generated manifest (r3 F9).
        hexes = [f for f in raw if isinstance(f, dict) and re.fullmatch(r"[0-9a-fA-F]{64}", str(f.get("Secret", "")))]
        digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} if hexes else set()

        def _manifest(finding: dict) -> bool:
            original = originals.get(Path(str(finding.get("File", ""))).name)
            return original is not None and original.name in _DIGEST_MANIFESTS

        return [
            finding
            for finding in raw
            if not (isinstance(finding, dict) and str(finding.get("Secret", "")).lower() in digests and _manifest(finding))
        ]


def no_secrets(
    paths: list[Path],
    *,
    receipt_path: Path | None = None,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> GateReceipt:
    timestamp = _utc_now()

    path_findings = _path_scan(paths)
    if path_findings:
        receipt = GateReceipt(
            gate_id="secrets",
            status="failed",
            timestamp_utc=timestamp,
            findings=path_findings,
            notes="Gate blocked by secret-like filename(s); content scan skipped.",
        )
        if receipt_path is not None:
            write_gate_receipt(receipt, receipt_path)
        raise GateClosed(reason="secret_like_path", receipt=dataclasses.asdict(receipt))

    content_findings = _gitleaks_content_scan(paths, _scanner=_scanner)
    status: Literal["passed", "failed"] = "failed" if content_findings else "passed"
    receipt = GateReceipt(
        gate_id="secrets",
        status=status,
        timestamp_utc=timestamp,
        findings=content_findings,
    )
    if receipt_path is not None:
        write_gate_receipt(receipt, receipt_path)
    if content_findings:
        raise GateClosed(reason="content_scan_findings", receipt=dataclasses.asdict(receipt))

    return receipt


_SCHEMA_CACHE: dict | None = None
_VALID_REDISTRIBUTION_STATUSES = frozenset({"allowed", "caution", "prohibited"})


def _load_schema() -> dict:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema_path = (
            Path(__file__).parents[3] / "docs" / "public_export_manifest.schema.json"
        )
        _SCHEMA_CACHE = json.loads(schema_path.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def validate_public_export_manifest(
    manifest_path: Path,
    *,
    receipt_path: Path | None = None,
) -> GateReceipt:
    """Validate a public export manifest: schema shape + redistribution policy.

    Raises GateClosed if any family is prohibited/missing/unknown status, or if the
    manifest fails schema validation.  Caution families add non-blocking findings.
    """
    timestamp = _utc_now()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema = _load_schema()
    try:
        jsonschema.validate(instance=manifest_data, schema=schema)
    except jsonschema.ValidationError as exc:
        receipt = GateReceipt(
            gate_id="redistribution",
            status="failed",
            timestamp_utc=timestamp,
            findings=[{"reason": "schema_validation_failed", "detail": exc.message}],
        )
        if receipt_path is not None:
            write_gate_receipt(receipt, receipt_path)
        raise GateClosed(
            reason="schema_validation_failed",
            receipt=dataclasses.asdict(receipt),
        ) from exc

    blocking_findings: list[dict] = []
    caution_findings: list[dict] = []

    for family in manifest_data.get("families", []):
        family_id = family.get("id", "<unknown>")
        raw_status = family.get("redistribution_status")

        if raw_status not in _VALID_REDISTRIBUTION_STATUSES:
            effective = "prohibited"
        else:
            effective = raw_status

        if effective == "prohibited":
            blocking_findings.append(
                {
                    "family_id": family_id,
                    "reason": "redistribution_status is prohibited or missing/unknown",
                    "status": effective,
                }
            )
        elif effective == "caution":
            entry: dict = {
                "family_id": family_id,
                "severity": "caution",
                "reason": "redistribution_status is caution; operator acknowledgement required",
            }
            notes = family.get("public_export_notes")
            if notes:
                entry["public_export_notes"] = notes
            caution_findings.append(entry)

    all_findings = blocking_findings + caution_findings

    if blocking_findings:
        receipt = GateReceipt(
            gate_id="redistribution",
            status="failed",
            timestamp_utc=timestamp,
            findings=all_findings,
        )
        if receipt_path is not None:
            write_gate_receipt(receipt, receipt_path)
        raise GateClosed(
            reason="redistribution_blocked",
            receipt=dataclasses.asdict(receipt),
        )

    receipt = GateReceipt(
        gate_id="redistribution",
        status="passed",
        timestamp_utc=timestamp,
        findings=caution_findings,
    )
    if receipt_path is not None:
        write_gate_receipt(receipt, receipt_path)
    return receipt


def run_gate(
    manifest_path: Path,
    candidate_paths: list[Path],
    *,
    receipt_dir: Path | None = None,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> dict[str, GateReceipt]:
    """Run the secrets gate then the redistribution gate.

    Raises GateClosed on the first failing gate; the second gate is not run.
    Returns a dict mapping gate_id → GateReceipt for both passing gates.
    """
    secrets_receipt_path = (
        receipt_dir / "secrets_gate.json" if receipt_dir is not None else None
    )
    redistribution_receipt_path = (
        receipt_dir / "redistribution_gate.json" if receipt_dir is not None else None
    )

    secrets_receipt = no_secrets(
        candidate_paths,
        receipt_path=secrets_receipt_path,
        _scanner=_scanner,
    )
    redistribution_receipt = validate_public_export_manifest(
        manifest_path,
        receipt_path=redistribution_receipt_path,
    )
    return {"secrets": secrets_receipt, "redistribution": redistribution_receipt}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed secrets gate for export candidates",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Paths (or globs) to scan",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help="Write secrets gate receipt JSON to this path",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Public export manifest path; when supplied, run_gate() is called instead of no_secrets()",
    )
    args = parser.parse_args(argv)

    import glob as _glob
    from pathlib import Path as _Path

    candidate_paths: list[Path] = []
    for pattern in args.paths:
        matches = _glob.glob(pattern, recursive=True)
        if matches:
            candidate_paths.extend(_Path(m) for m in matches)
        else:
            candidate_paths.append(_Path(pattern))

    try:
        if args.manifest_path is not None:
            receipt_dir = args.receipt_path.parent if args.receipt_path is not None else None
            run_gate(args.manifest_path, candidate_paths, receipt_dir=receipt_dir)
            print("export-gate: PASSED — secrets + redistribution clean", flush=True)
        else:
            no_secrets(candidate_paths, receipt_path=args.receipt_path)
            print("export-gate: PASSED — zero findings", flush=True)
        return 0
    except GateClosed as exc:
        print(f"export-gate: BLOCKED — {exc.reason}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
