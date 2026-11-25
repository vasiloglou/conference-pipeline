"""Regression coverage for conference_pipeline.acquisition.workshop_pipeline.

The wrapped download script serializes whatever ``output_dir`` Path object
it is handed, verbatim, into the download receipt. Every consumer of that
receipt (``resolve_workspace_member``, ``classify_workshop_pdf_sources``, the
CVPR runner's ``_build_workshop_paper_records``) expects that stored value to
be relative to the workspace, not the repo root. A live CVPR run stored the
repo-root-relative form (``conferences/cvpr2026/storage/workshops/papers``),
which silently double-prefixed on read and made every workshop paper vanish
from the corpus with zero blockers raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from conference_pipeline.acquisition import workshop_pipeline


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fake_discovery_module() -> SimpleNamespace:
    def discover(parent_venue: str, sleep_between_calls: float = 0.0) -> dict:
        return {
            "workshops": [],
            "workshop_count": 0,
            "paper_count_total": 0,
            "pdf_present_count_total": 0,
            "missing_pdf_count_total": 0,
            "availability_breakdown": {},
            "generated_at": "2026-01-01T00:00:00+00:00",
            "inventory_digest": "deadbeef",
        }

    return SimpleNamespace(
        DEFAULT_USER_AGENT="test-agent",
        OPENREVIEW_AUTH=None,
        OPENREVIEW_API_BASE="https://api2.openreview.net",
        discover=discover,
        write_json=_write_json,
    )


def _fake_downloader_module() -> SimpleNamespace:
    def run_download(args) -> dict:
        # Mirrors the real script: it stores whatever output_dir Path it was
        # given verbatim -- the wrapper is responsible for making that
        # workspace-relative before persisting the receipt.
        return {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.workshop_paper_download_receipt",
            "stage_id": "acquisition.workshop_paper_download",
            "status": "succeeded",
            "workspace": str(args.workspace) if args.workspace else None,
            "output_dir": str(args.output_dir),
            "per_workshop": [],
            "totals": {
                "attempted": 0,
                "succeeded": 0,
                "unreachable": 0,
                "resume_skipped": 0,
                "available": 0,
            },
            "skipped_external_workshops": [],
        }

    return SimpleNamespace(OPENREVIEW_AUTH=None, run_download=run_download, write_json=_write_json)


def test_download_receipt_output_dir_is_workspace_relative(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)

    def fake_load_script_module(name: str, path: Path) -> SimpleNamespace:
        if name == "_workshop_discovery_inproc":
            return _fake_discovery_module()
        if name == "_workshop_download_inproc":
            return _fake_downloader_module()
        raise AssertionError(f"unexpected script module requested: {name}")

    monkeypatch.setattr(workshop_pipeline, "_load_script_module", fake_load_script_module)

    summary = workshop_pipeline.run_workshop_discovery_and_download(
        workspace, parent_venue="thecvf.com/CVPR/2026/Workshop"
    )

    assert summary["status"] == "succeeded"
    # This is the exact field a repo-root-relative caller (workspace passed
    # as e.g. Path("conferences/cvpr2026")) would otherwise double-prefix:
    # workspace / "conferences/cvpr2026/storage/workshops/papers".
    assert summary["download"]["output_dir"] == "storage/workshops/papers"

    on_disk = json.loads((workspace / "storage/workshops/download_receipt.json").read_text())
    assert on_disk["output_dir"] == "storage/workshops/papers"
