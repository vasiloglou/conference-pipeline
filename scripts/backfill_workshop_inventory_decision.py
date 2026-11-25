#!/usr/bin/env python3
"""Backfill the per-row `decision` field on an existing workshop inventory.

Fix option 9 (backfill): populates `decision` on every workshop row that
lacks it (or whose `decision` is empty), using the same deterministic mapping
future discovery runs use. `decision` is derived purely from
`pdf_availability_class`, which is already an input to `inventory_digest`, so
this backfill NEVER changes `inventory_digest` and NEVER opens or mutates
`discovery_receipt.json` -- it writes its own annotation receipt instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # allow running both as ``python scripts/...`` and via package import
    from scripts.discover_openreview_workshops import (
        decision_for_availability,
        deterministic_digest,
    )
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for direct runs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.discover_openreview_workshops import (
        decision_for_availability,
        deterministic_digest,
    )

DEFAULT_INVENTORY_PATH = Path("storage/workshops/inventory.json")
DEFAULT_RECEIPT_PATH = Path("storage/workshops/inventory_decision_backfill_receipt.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def backfill_decisions(inventory: dict[str, Any]) -> tuple[dict[str, Any], int, dict[str, int]]:
    """Populate `decision` on rows lacking it. Returns (inventory, rows_updated, breakdown)."""

    workshops = inventory.get("workshops")
    if not isinstance(workshops, list):
        raise ValueError("inventory.json missing a 'workshops' list")

    rows_updated = 0
    breakdown: Counter[str] = Counter()
    for row in workshops:
        if not isinstance(row, dict):
            continue
        existing = row.get("decision")
        if not existing:
            row["decision"] = decision_for_availability(row["pdf_availability_class"])
            rows_updated += 1
        breakdown[row["decision"]] += 1

    inventory["workshops"] = workshops
    return inventory, rows_updated, dict(sorted(breakdown.items()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the per-row decision field on an existing workshop inventory."
    )
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--inventory-path", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--receipt-path", type=Path, default=DEFAULT_RECEIPT_PATH)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def resolve(workspace: Path | None, path: Path) -> Path:
    if path.is_absolute() or workspace is None:
        return path
    return workspace / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory_path = resolve(args.workspace, args.inventory_path)
    receipt_path = resolve(args.workspace, args.receipt_path)

    if not inventory_path.is_file():
        print(f"inventory not found: {inventory_path}", file=sys.stderr)
        return 2

    inventory = load_json(inventory_path)
    inventory_digest_before = inventory.get("inventory_digest")

    inventory, rows_updated, decision_breakdown = backfill_decisions(inventory)

    # decision is derived from pdf_availability_class, which is already an
    # input to deterministic_digest; recomputing must equal the pre-backfill
    # digest exactly. This is the same source-of-truth function future
    # discovery runs use, so recompute rather than merely copy the old value
    # forward, to positively prove the digest basis is unaffected.
    recomputed_digest = deterministic_digest(inventory["workshops"])
    inventory_digest_after = inventory.get("inventory_digest")
    if inventory_digest_before is not None and recomputed_digest != inventory_digest_before:
        print(
            "refusing to write: recomputed inventory_digest does not match "
            "the pre-backfill digest even though `decision` is not a digest "
            f"input (before={inventory_digest_before} recomputed={recomputed_digest})",
            file=sys.stderr,
        )
        return 3
    inventory["inventory_digest"] = inventory_digest_before

    write_json(inventory_path, inventory)

    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_inventory_decision_backfill",
        "status": "succeeded",
        "generated_at": utc_now(),
        "workspace": str(args.workspace) if args.workspace else None,
        "inventory_path": str(inventory_path),
        "rows_updated": rows_updated,
        "decision_breakdown": decision_breakdown,
        "inventory_digest_before": inventory_digest_before,
        "inventory_digest_after": inventory_digest_after,
        "discovery_receipt_inventory_digest_handling": (
            "inventory_digest is intentionally unchanged: `decision` is a "
            "derived field outside the digest basis (deterministic_digest "
            "only hashes venueid, paper_count, pdf_present_count, tiers, and "
            "pdf_availability_class), so the discovery receipt's "
            "inventory_digest remains valid historical evidence and is "
            "deliberately left untouched. This script never opens or "
            "mutates discovery_receipt.json."
        ),
    }
    write_json(receipt_path, receipt)

    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "rows_updated": rows_updated,
                    "decision_breakdown": decision_breakdown,
                    "inventory_digest_before": inventory_digest_before,
                    "inventory_digest_after": inventory_digest_after,
                    "receipt_path": str(receipt_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
