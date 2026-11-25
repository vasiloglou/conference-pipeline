"""Conference-general workshop discovery + paper-download pipeline.

This module wraps the two registry-owned acquisition scripts
(`scripts/discover_openreview_workshops.py`,
`scripts/download_openreview_workshop_papers.py`) into a single in-process
callable suitable for per-conference production runners. The two scripts
remain the source of truth — this module just orchestrates them.

Typical use from a conference runner:

    from conference_pipeline.acquisition import run_workshop_discovery_and_download

    workshop_summary = run_workshop_discovery_and_download(
        workspace=Path("conferences/iclr2026"),
        parent_venue="ICLR.cc/2026/Workshop",
        rate_limit_rps=1.4,
    )
    # workshop_summary['totals'] feeds the stage-04 receipt; failures show up
    # in workshop_summary['status'] == 'blocked' with workshop_summary['error'].
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from conference_pipeline.openreview_auth import load_openreview_auth

DEFAULT_OPENREVIEW_API_BASE = "https://api2.openreview.net"
DEFAULT_RATE_LIMIT_RPS = 1.4
DEFAULT_WORKERS = 6
DEFAULT_WORKSHOP_INVENTORY_RELATIVE = Path("storage/workshops/inventory.json")
DEFAULT_DISCOVERY_RECEIPT_RELATIVE = Path("storage/workshops/discovery_receipt.json")
DEFAULT_DOWNLOAD_RECEIPT_RELATIVE = Path("storage/workshops/download_receipt.json")
DEFAULT_PAPERS_DIR_RELATIVE = Path("storage/workshops/papers")
DEFAULT_TRACKER_RELATIVE = Path("storage/workshops/workshop_site_tracker.json")
DEFAULT_PRIORITY_MANIFEST_RELATIVE = Path("storage/source-intake/physics_priority.json")

# Resolved at import time so the module can find the registry scripts without
# requiring the caller to set sys.path. scripts/ is not a Python package, so
# importlib.util.spec_from_file_location is the canonical loader.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_DISCOVERY_SCRIPT_PATH = _SCRIPTS_ROOT / "discover_openreview_workshops.py"
_DOWNLOAD_SCRIPT_PATH = _SCRIPTS_ROOT / "download_openreview_workshop_papers.py"


def _load_script_module(name: str, path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"workshop acquisition script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build importlib spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slugify_venueid(venueid: str) -> str:
    """Return the filesystem-safe slug used for per-workshop output directories.

    This is the canonical slug helper for any acquisition code that needs to
    name a directory or stub script after a workshop venue id. Both the
    download script's per-paper directory layout and the analysis scaffold's
    `external_site_required` stub naming use this slugifier.
    """
    keep: list[str] = []
    for ch in venueid.replace("/", "_"):
        if ch.isalnum() or ch in "-_":
            keep.append(ch.lower())
        else:
            keep.append("-")
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:80] or "workshop"


def run_workshop_discovery_and_download(
    workspace: Path,
    *,
    parent_venue: str,
    rate_limit_rps: float = DEFAULT_RATE_LIMIT_RPS,
    workers: int = DEFAULT_WORKERS,
    max_papers: int = 0,
    inventory_relative: Path = DEFAULT_WORKSHOP_INVENTORY_RELATIVE,
    discovery_receipt_relative: Path = DEFAULT_DISCOVERY_RECEIPT_RELATIVE,
    download_receipt_relative: Path = DEFAULT_DOWNLOAD_RECEIPT_RELATIVE,
    papers_dir_relative: Path = DEFAULT_PAPERS_DIR_RELATIVE,
    priority_manifest_relative: Path = DEFAULT_PRIORITY_MANIFEST_RELATIVE,
) -> dict[str, Any]:
    """Run workshop discovery and OpenReview-PDF download for a workspace.

    Returns a summary dict suitable for merging into a stage receipt:

    - ``status``: ``succeeded``, ``partial``, ``blocked``, or ``not_run``
    - ``parent_venue``, ``rate_limit_rps``
    - ``discovery``: full discovery receipt payload (workshop_count,
      paper_count_total, availability_breakdown, inventory_digest, …)
    - ``download``: full download receipt payload (totals, per_workshop,
      skipped_external_workshops, …)
    - ``inventory_path``, ``discovery_receipt_path``, ``download_receipt_path``
    - ``error`` (only present when status == 'blocked')

    Conference-specific arguments are the workspace path and the parent
    OpenReview venue prefix; the rate-limit and output layout default to the
    values recorded in the production-workflow contract.
    """
    summary: dict[str, Any] = {
        "parent_venue": parent_venue,
        "rate_limit_rps": rate_limit_rps,
        "status": "not_run",
        "priority_manifest_path": str(workspace / priority_manifest_relative),
        "priority_order_applied": False,
    }

    discovery_module = _load_script_module(
        "_workshop_discovery_inproc", _DISCOVERY_SCRIPT_PATH
    )
    inventory_path = workspace / inventory_relative
    discovery_receipt_path = workspace / discovery_receipt_relative
    try:
        auth = load_openreview_auth(discovery_module.DEFAULT_USER_AGENT, mode="auto")
        discovery_module.OPENREVIEW_AUTH = auth
        payload = discovery_module.discover(parent_venue, sleep_between_calls=0.0)
    except Exception as exc:  # pragma: no cover - surfaced as receipt failure
        summary["status"] = "blocked"
        summary["error"] = f"workshop_discovery_failed: {exc}"
        return summary

    priority_path = workspace / priority_manifest_relative
    priority_venueids: list[str] = []
    if priority_path.exists():
        try:
            priority_payload = json.loads(priority_path.read_text(encoding="utf-8"))
            raw_priority = priority_payload.get("workshop_venueids", [])
            if isinstance(raw_priority, list):
                priority_venueids = [str(item) for item in raw_priority if str(item)]
        except (OSError, json.JSONDecodeError):
            # Priority is an ordering enhancement. Discovery remains valid if
            # the optional manifest is unreadable; the receipt records that no
            # priority reorder was applied.
            priority_venueids = []
    priority_rank = {venueid: index for index, venueid in enumerate(priority_venueids)}
    workshops = payload.get("workshops", [])
    if isinstance(workshops, list) and priority_rank:
        payload["workshops"] = sorted(
            workshops,
            key=lambda item: (
                0 if str(item.get("venueid") or "") in priority_rank else 1,
                priority_rank.get(str(item.get("venueid") or ""), len(priority_rank)),
                str(item.get("venueid") or ""),
            ),
        )
        summary["priority_order_applied"] = True
        summary["priority_venueids"] = priority_venueids
    discovery_module.write_json(inventory_path, payload)
    discovery_receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_discovery_receipt",
        "stage_id": "acquisition.workshop_discovery",
        "status": "succeeded",
        "openreview_api_base": discovery_module.OPENREVIEW_API_BASE,
        "parent_venue_prefix": parent_venue,
        "workspace": str(workspace),
        "inventory_path": str(inventory_path),
        "inventory_digest": payload["inventory_digest"],
        "workshop_count": payload["workshop_count"],
        "paper_count_total": payload["paper_count_total"],
        "pdf_present_count_total": payload["pdf_present_count_total"],
        "missing_pdf_count_total": payload["missing_pdf_count_total"],
        "availability_breakdown": payload["availability_breakdown"],
        "generated_at": payload["generated_at"],
    }
    discovery_module.write_json(discovery_receipt_path, discovery_receipt)

    summary["discovery"] = discovery_receipt
    summary["inventory_path"] = str(inventory_path)
    summary["discovery_receipt_path"] = str(discovery_receipt_path)

    downloader_module = _load_script_module(
        "_workshop_download_inproc", _DOWNLOAD_SCRIPT_PATH
    )
    downloader_module.OPENREVIEW_AUTH = auth
    download_receipt_path = workspace / download_receipt_relative
    output_dir = workspace / papers_dir_relative
    download_args = argparse.Namespace(
        inventory=inventory_path,
        output_dir=output_dir,
        receipt_path=download_receipt_path,
        rate_limit_rps=rate_limit_rps,
        workers=workers,
        max_papers=max_papers,
        workspace=workspace,
    )
    try:
        download_receipt = downloader_module.run_download(download_args)
    except Exception as exc:  # pragma: no cover - surfaced as receipt failure
        summary["status"] = "blocked"
        summary["error"] = f"workshop_download_failed: {exc}"
        return summary
    # The download script serializes whatever output_dir Path it was given
    # verbatim, which is workspace-prefixed here (`workspace / papers_dir_relative`).
    # Every other consumer (resolve_workspace_member, classify_workshop_pdf_sources)
    # expects receipt paths relative to workspace, not repo-root-relative, so
    # overwrite with the value that's actually relative to `workspace`.
    download_receipt["output_dir"] = str(papers_dir_relative)
    downloader_module.write_json(download_receipt_path, download_receipt)
    summary["download"] = download_receipt
    summary["download_receipt_path"] = str(download_receipt_path)
    summary["openreview_auth"] = auth.receipt_fields()
    summary["status"] = download_receipt.get("status", "succeeded")
    return summary
