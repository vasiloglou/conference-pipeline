#!/usr/bin/env python3
"""Refresh ``download_corpus.json``'s stale source-discovery lineage binding
(Phase 10.1-10 Finding B, ORCHESTRATOR AMENDMENT Task 6).

Root cause: a real (non-dry-run) ``scripts/rederive_source_discovery_authors.py``
run rewrites ``storage/source-discovery/source_discovery_inventory.json`` and
changes its own ``inventory_sha256``. ``download_corpus.json``'s
``lineage.phase_02_inventory_sha256`` field was recorded once, at the original
Stage 3/4 run, to name the exact source-discovery inventory digest that ledger
was built against -- and Stage 6's own
``_load_source_discovery_inventory_for_index`` gate
(``scripts/run_cvpr2026_production.py``) fails closed with
``source_discovery_inventory_stale`` the moment that recorded field no longer
matches the live inventory file's ``inventory_sha256``.

This is a pure lineage-binding refresh, not a ledger rebuild:

- Only ``lineage.phase_02_inventory_sha256`` is updated on ``download_corpus.json``,
  to the live, post-rederive ``source_discovery_inventory.json``'s own
  ``inventory_sha256``. No other ledger field (``artifacts``, ``counts``,
  ``program_item_results``, ``request_results``, ...) is touched.
- ``download_corpus.json``'s own ``manifest_sha256`` is a canonical digest of
  the ledger's durable state (``canonical_ledger_digest``, the SAME helper
  ``run_download_corpus_stage`` itself uses to compute it -- not
  reimplemented here), which necessarily changes once ``lineage`` changes.
  This script recomputes it with that same helper, so the ledger stays
  internally self-consistent.
- Every existing Stage-4 receipt (``stage-04-download-corpus.json``, across
  every ``storage/workflow-runs/*/`` directory) whose
  ``checks.download_corpus_ledger_sha256`` currently equals the ledger's OLD
  ``manifest_sha256`` is updated to the NEW value, and nothing else on that
  receipt (``status``, ``mode``, ``stage_order``, ``blockers``, ...) is
  touched. This is required because
  ``scripts/run_cvpr2026_production.py``'s ``_validate_download_corpus_admission``
  / ``_find_cross_run_download_corpus_receipt`` bind Stage 6 admission to a
  Stage-4 receipt whose declared ``download_corpus_ledger_sha256`` matches the
  LIVE ledger's current ``manifest_sha256`` -- if only the ledger were
  updated, the admission would break with ``download_corpus_stage_not_succeeded``
  instead of clearing ``source_discovery_inventory_stale``.

Safety preconditions (verified by the CALLER, per Task 5/Task 6's own
acceptance criteria, before this script is ever invoked in non-dry-run mode):
``PINNED_CORPUS_SOURCE_RUN_ID`` unchanged, the rederive receipt's own
``records_authors_added`` matches the expected count, and ``source_records``
count / the full ``source_record_id`` set are unchanged pre/post rederive.
This script additionally asserts ``download_corpus.json``'s own
``lineage.run_id`` still equals the source-discovery run id supplied on the
command line, as a final belt-and-braces check that it is binding against the
inventory it thinks it is.

Usage::

    .venv/bin/python scripts/refresh_download_corpus_lineage_digest.py \\
        --workspace conferences/cvpr2026 \\
        --source-discovery-run-id cvpr2026-live-production-r10-main-paper-corpus

Add ``--dry-run`` to print the before/after digest diff without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.acquisition.cvpr_corpus_download import (  # noqa: E402
    canonical_ledger_digest,
)
from conference_pipeline.workflow_runs import write_json_atomic  # noqa: E402

DOWNLOAD_CORPUS_RELATIVE_PATH = Path("storage/download-corpus/download_corpus.json")
SOURCE_DISCOVERY_INVENTORY_RELATIVE_PATH = Path(
    "storage/source-discovery/source_discovery_inventory.json"
)
WORKFLOW_RUNS_RELATIVE_ROOT = Path("storage/workflow-runs")
STAGE4_RECEIPT_FILENAME = "stage-04-download-corpus.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workspace", type=Path, required=True, help="conference workspace root, e.g. conferences/cvpr2026"
    )
    parser.add_argument(
        "--source-discovery-run-id",
        required=True,
        help="the source-discovery run id download_corpus.json's lineage is bound to (e.g. PINNED_CORPUS_SOURCE_RUN_ID)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report the diff; write nothing")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = args.workspace.resolve()

    ledger_path = workspace_root / DOWNLOAD_CORPUS_RELATIVE_PATH
    inventory_path = workspace_root / SOURCE_DISCOVERY_INVENTORY_RELATIVE_PATH
    if not ledger_path.is_file():
        print(f"ERROR: ledger not found: {ledger_path}", file=sys.stderr)
        return 2
    if not inventory_path.is_file():
        print(f"ERROR: source-discovery inventory not found: {inventory_path}", file=sys.stderr)
        return 2

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    lineage = ledger.get("lineage")
    if not isinstance(lineage, dict):
        print("ERROR: download_corpus.json has no 'lineage' object", file=sys.stderr)
        return 2

    if lineage.get("run_id") != args.source_discovery_run_id:
        print(
            "ERROR: download_corpus.json's lineage.run_id "
            f"({lineage.get('run_id')!r}) does not match the supplied "
            f"--source-discovery-run-id ({args.source_discovery_run_id!r}); "
            "refusing to bind against a different inventory than intended.",
            file=sys.stderr,
        )
        return 1

    old_phase_02_inventory_sha256 = lineage.get("phase_02_inventory_sha256")
    new_phase_02_inventory_sha256 = inventory.get("inventory_sha256")
    if not isinstance(new_phase_02_inventory_sha256, str) or not new_phase_02_inventory_sha256:
        print("ERROR: source-discovery inventory has no inventory_sha256", file=sys.stderr)
        return 2

    old_manifest_sha256 = ledger.get("manifest_sha256")

    if old_phase_02_inventory_sha256 == new_phase_02_inventory_sha256:
        print("Nothing to refresh; lineage.phase_02_inventory_sha256 already matches the live inventory.")
        return 0

    lineage["phase_02_inventory_sha256"] = new_phase_02_inventory_sha256
    new_manifest_sha256 = canonical_ledger_digest(ledger)
    ledger["manifest_sha256"] = new_manifest_sha256

    # Find every Stage-4 receipt currently bound to the OLD ledger digest, so
    # Stage 6/7 admission (_validate_download_corpus_admission /
    # _find_cross_run_download_corpus_receipt) keeps resolving a receipt whose
    # declared download_corpus_ledger_sha256 matches the LIVE ledger.
    runs_root = workspace_root / WORKFLOW_RUNS_RELATIVE_ROOT
    bound_receipt_paths: list[Path] = []
    if runs_root.is_dir():
        for entry in sorted(runs_root.iterdir()):
            if not entry.is_dir():
                continue
            receipt_path = entry / STAGE4_RECEIPT_FILENAME
            if not receipt_path.is_file():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            checks = receipt.get("checks") if isinstance(receipt.get("checks"), dict) else {}
            if checks.get("download_corpus_ledger_sha256") == old_manifest_sha256:
                bound_receipt_paths.append(receipt_path)

    print("=== download_corpus.json lineage.phase_02_inventory_sha256 ===")
    print(f"  old: {old_phase_02_inventory_sha256}")
    print(f"  new: {new_phase_02_inventory_sha256}")
    print("=== download_corpus.json manifest_sha256 ===")
    print(f"  old: {old_manifest_sha256}")
    print(f"  new: {new_manifest_sha256}")
    print("=== bound Stage-4 receipts to update (checks.download_corpus_ledger_sha256) ===")
    if bound_receipt_paths:
        for path in bound_receipt_paths:
            print(f"  {path.relative_to(workspace_root)}")
    else:
        print("  (none found -- no receipt currently declares the old ledger digest)")

    summary = {
        "status": "success",
        "dry_run": bool(args.dry_run),
        "workspace": str(workspace_root),
        "source_discovery_run_id": args.source_discovery_run_id,
        "old_phase_02_inventory_sha256": old_phase_02_inventory_sha256,
        "new_phase_02_inventory_sha256": new_phase_02_inventory_sha256,
        "old_manifest_sha256": old_manifest_sha256,
        "new_manifest_sha256": new_manifest_sha256,
        "bound_receipt_paths": [str(path.relative_to(workspace_root)) for path in bound_receipt_paths],
        "wrote_files": [] if args.dry_run else (
            [str(ledger_path.relative_to(workspace_root))]
            + [str(path.relative_to(workspace_root)) for path in bound_receipt_paths]
        ),
    }

    if args.dry_run:
        if args.json:
            print(json.dumps(summary, indent=2))
        print("Dry-run: not writing any file.")
        return 0

    write_json_atomic(ledger_path, ledger)
    print(f"Wrote {ledger_path}")
    for receipt_path in bound_receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checks = receipt.get("checks")
        if isinstance(checks, dict):
            checks["download_corpus_ledger_sha256"] = new_manifest_sha256
        write_json_atomic(receipt_path, receipt)
        print(f"Wrote {receipt_path}")

    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
