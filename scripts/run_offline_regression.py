#!/usr/bin/env python3
"""Run the offline regression suite and update the quality baseline receipt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_PATH = REPO_ROOT / "docs" / "quality" / "quality_baseline_receipt.json"

LIVE_CREDENTIAL_GATES = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "EXTERNAL_TRANSCRIPTION_API_KEY",
]
DRIVE_MOUNT_GATES = ["GOOGLE_APPLICATION_CREDENTIALS", "GDRIVE_MOUNT_PATH"]
NETWORK_CALL_GATES = ["ENABLE_LIVE_SEARCH", "ENABLE_TAVILY"]
SURFACES_COVERED = [
    "costs",
    "research",
    "local_model",
    "export_adjacent",
    "workflow",
    "presentation",
    "public_byok_tools",
]


def preflight_fail_closed() -> None:
    present = [
        key
        for key in LIVE_CREDENTIAL_GATES + DRIVE_MOUNT_GATES + NETWORK_CALL_GATES
        if os.getenv(key)
    ]
    if present:
        print(f"FAIL-CLOSED: live environment variables detected: {', '.join(sorted(present))}")
        print("Unset provider credentials, Drive mount settings, and network toggles first.")
        raise SystemExit(2)
    print("OFFLINE CHECK: environment clean")


def run_pytest(extra_args: tuple[str, ...] = ()) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--ignore=tests/live_llm",
        "-q",
        "--no-header",
        *extra_args,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def run_gate_proof() -> int:
    """[gate] export gate: secrets + redistribution both-ways proof."""
    print("[gate] export gate: secrets + redistribution both-ways proof", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_export_gate_secrets.py",
        "tests/test_export_gate_redistribution.py",
        "-q",
        "--no-header",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def run_public_export_proof() -> int:
    """[publishing] public export: dry-run, write, and fail-closed proof."""
    print("[publishing] public export: dry-run, write, and fail-closed proof", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_export_for_public.py",
        "-q",
        "--no-header",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def run_public_index_proof() -> int:
    """[publishing] public index: bundle-root, provenance, and artifact proof."""
    print("[publishing] public index: bundle-root, provenance, and artifact proof", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_build_public_index.py",
        "-q",
        "--no-header",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def run_public_publishing_proof() -> int:
    """[publishing] public repo/site/cdn/dataset: dry-run and fail-closed proof."""
    print("[publishing] public repo/site/cdn/dataset: dry-run and fail-closed proof", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_push_public_repo.py",
        "tests/test_generate_public_site.py",
        "tests/test_public_cdn_liveness.py",
        "tests/test_public_dataset_package.py",
        "-q",
        "--no-header",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def run_public_byok_tools_proof() -> int:
    """[publishing] public BYOK tools: browser/self-host/audio safety proof."""
    print("[publishing] public BYOK tools: browser/self-host/audio safety proof", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_byok_tools.py",
        "tests/test_self_host_tooling.py",
        "tests/test_notebooklm_audio.py",
        "tests/test_docs_public_byok_safety.py",
        "-q",
        "--no-header",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def source_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"


def load_receipt() -> dict[str, Any]:
    if not RECEIPT_PATH.exists():
        return {}
    try:
        return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def update_receipt(pytest_exit_code: int) -> None:
    receipt = load_receipt()
    receipt.setdefault("schema_version", 1)
    receipt.setdefault("receipt_type", "conference_pipeline.quality_baseline_receipt")
    receipt.setdefault("review_date", datetime.now(timezone.utc).date().isoformat())
    receipt.setdefault("reviewed_modules", [])
    receipt.setdefault(
        "findings_summary",
        {"total": 0, "by_severity": {}, "by_status": {}},
    )
    receipt.setdefault("bugs_fixed", [])
    receipt.setdefault("tests_added", [])
    receipt.setdefault("deferred_findings", [])
    receipt["offline_command"] = "python scripts/run_offline_regression.py"
    receipt["offline_command_result"] = "passed" if pytest_exit_code == 0 else "failed"
    receipt["offline_command_exit_code"] = pytest_exit_code
    receipt["offline_command_timestamp"] = datetime.now(timezone.utc).isoformat()
    receipt["source_git_sha"] = source_git_sha()
    receipt["surfaces_covered"] = SURFACES_COVERED
    write_json_atomic(RECEIPT_PATH, receipt)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline pytest suite and update the quality baseline receipt."
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional extra pytest arguments appended after the offline defaults.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra_args = tuple(arg for arg in args.pytest_args if arg != "--")
    preflight_fail_closed()
    pytest_exit_code = run_pytest(extra_args)
    gate_exit_code = run_gate_proof()
    public_export_exit_code = run_public_export_proof()
    public_index_exit_code = run_public_index_proof()
    public_publishing_exit_code = run_public_publishing_proof()
    public_byok_tools_exit_code = run_public_byok_tools_proof()
    combined_exit_code = next(
        (
            code
            for code in (
                pytest_exit_code,
                gate_exit_code,
                public_export_exit_code,
                public_index_exit_code,
                public_publishing_exit_code,
                public_byok_tools_exit_code,
            )
            if code != 0
        ),
        0,
    )
    update_receipt(combined_exit_code)
    print(
        "OFFLINE SUMMARY: "
        f"pytest_exit_code={pytest_exit_code}; "
        f"gate_proof_exit_code={gate_exit_code}; "
        f"public_export_proof_exit_code={public_export_exit_code}; "
        f"public_index_proof_exit_code={public_index_exit_code}; "
        f"public_publishing_proof_exit_code={public_publishing_exit_code}; "
        f"public_byok_tools_proof_exit_code={public_byok_tools_exit_code}; "
        f"receipt={RECEIPT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0 if combined_exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
