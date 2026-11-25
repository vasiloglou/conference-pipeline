"""Tests for the workshop inventory `decision` field (fix option 9).

Covers the deterministic decision_for_availability mapping, digest stability
when decision is added, and the standalone backfill script + its receipt.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.backfill_workshop_inventory_decision import backfill_decisions, main as backfill_main
from scripts.discover_openreview_workshops import (
    decision_for_availability,
    deterministic_digest,
    summarize_workshop,
)


def _note(note_id: str, *, pdf: str | None, venue: str = "ICLR 2026 Workshop Poster") -> dict:
    content: dict[str, object] = {"venue": venue, "title": f"Paper {note_id}"}
    if pdf is not None:
        content["pdf"] = pdf
    return {"id": note_id, "content": content}


def test_decision_for_availability_maps_all_three_classes():
    assert decision_for_availability("openreview_with_pdf") == "approved"
    assert decision_for_availability("openreview_metadata_only") == "metadata_only_acceptable"
    assert decision_for_availability("no_openreview_record") == "external_site_required"


def test_summarize_workshop_populates_decision_for_each_availability_class():
    with_pdf = summarize_workshop("v/with-pdf", [_note("n1", pdf="/pdf/n1.pdf")])
    metadata_only = summarize_workshop("v/metadata-only", [_note("n2", pdf=None)])
    no_record = summarize_workshop("v/no-record", [])

    assert with_pdf["decision"] == "approved"
    assert metadata_only["decision"] == "metadata_only_acceptable"
    assert no_record["decision"] == "external_site_required"


def test_deterministic_digest_unchanged_by_adding_decision():
    workshops = [
        summarize_workshop("v/with-pdf", [_note("n1", pdf="/pdf/n1.pdf")]),
        summarize_workshop("v/metadata-only", [_note("n2", pdf=None)]),
        summarize_workshop("v/no-record", []),
    ]
    digest_with_decision = deterministic_digest(workshops)

    workshops_without_decision = []
    for row in workshops:
        stripped = dict(row)
        del stripped["decision"]
        workshops_without_decision.append(stripped)
    digest_without_decision = deterministic_digest(workshops_without_decision)

    assert digest_with_decision == digest_without_decision


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _legacy_inventory_without_decisions() -> dict:
    workshops = [
        {
            "venueid": "v/with-pdf",
            "paper_count": 1,
            "pdf_present_count": 1,
            "missing_pdf_count": 0,
            "tiers": {"Poster": 1},
            "pdf_availability_class": "openreview_with_pdf",
            "sub_track_parent": None,
            "sample_notes": [],
        },
        {
            "venueid": "v/metadata-only",
            "paper_count": 1,
            "pdf_present_count": 0,
            "missing_pdf_count": 1,
            "tiers": {"Poster": 1},
            "pdf_availability_class": "openreview_metadata_only",
            "sub_track_parent": None,
            "sample_notes": [],
        },
        {
            "venueid": "v/no-record",
            "paper_count": 0,
            "pdf_present_count": 0,
            "missing_pdf_count": 0,
            "tiers": {},
            "pdf_availability_class": "no_openreview_record",
            "sub_track_parent": None,
            "sample_notes": [],
        },
    ]
    digest = deterministic_digest(workshops)
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_discovery",
        "workshop_count": len(workshops),
        "inventory_digest": digest,
        "workshops": workshops,
    }


def test_backfill_decisions_populates_legacy_rows_and_preserves_digest():
    inventory = _legacy_inventory_without_decisions()
    digest_before = inventory["inventory_digest"]

    updated, rows_updated, breakdown = backfill_decisions(inventory)

    assert rows_updated == 3
    assert breakdown == {
        "approved": 1,
        "external_site_required": 1,
        "metadata_only_acceptable": 1,
    }
    assert all(row.get("decision") for row in updated["workshops"])
    digest_after = deterministic_digest(updated["workshops"])
    assert digest_after == digest_before


def test_backfill_script_writes_inventory_and_receipt_without_touching_discovery_receipt(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    inventory_path = workspace / "storage" / "workshops" / "inventory.json"
    discovery_receipt_path = workspace / "storage" / "workshops" / "discovery_receipt.json"
    receipt_path = workspace / "storage" / "workshops" / "inventory_decision_backfill_receipt.json"

    inventory = _legacy_inventory_without_decisions()
    digest_before = inventory["inventory_digest"]
    _write_json(inventory_path, inventory)
    discovery_receipt_before = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_discovery_receipt",
        "status": "succeeded",
        "inventory_digest": digest_before,
    }
    _write_json(discovery_receipt_path, discovery_receipt_before)
    discovery_receipt_mtime_before = discovery_receipt_path.stat().st_mtime_ns

    exit_code = backfill_main(
        [
            "--workspace",
            str(workspace),
            "--inventory-path",
            "storage/workshops/inventory.json",
            "--receipt-path",
            "storage/workshops/inventory_decision_backfill_receipt.json",
            "--summary",
        ]
    )

    assert exit_code == 0

    updated_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert updated_inventory["inventory_digest"] == digest_before
    assert all(row.get("decision") for row in updated_inventory["workshops"])

    # discovery_receipt.json must never be opened/mutated by the backfill.
    assert discovery_receipt_path.stat().st_mtime_ns == discovery_receipt_mtime_before
    assert json.loads(discovery_receipt_path.read_text(encoding="utf-8")) == discovery_receipt_before

    backfill_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert backfill_receipt["status"] == "succeeded"
    assert backfill_receipt["receipt_type"] == (
        "conference_pipeline.workshop_inventory_decision_backfill"
    )
    assert backfill_receipt["rows_updated"] == 3
    assert backfill_receipt["decision_breakdown"] == {
        "approved": 1,
        "external_site_required": 1,
        "metadata_only_acceptable": 1,
    }
    assert backfill_receipt["inventory_digest_before"] == digest_before
    assert backfill_receipt["inventory_digest_after"] == digest_before
    assert "deliberately left untouched" in (
        backfill_receipt["discovery_receipt_inventory_digest_handling"]
    )


def test_backfill_script_skips_rows_that_already_have_a_decision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    inventory_path = workspace / "storage" / "workshops" / "inventory.json"
    receipt_path = workspace / "storage" / "workshops" / "inventory_decision_backfill_receipt.json"

    inventory = _legacy_inventory_without_decisions()
    inventory["workshops"][0]["decision"] = "excluded"  # operator override, keep as-is
    digest_before = inventory["inventory_digest"]
    _write_json(inventory_path, inventory)

    exit_code = backfill_main(
        [
            "--workspace",
            str(workspace),
            "--inventory-path",
            "storage/workshops/inventory.json",
            "--receipt-path",
            "storage/workshops/inventory_decision_backfill_receipt.json",
        ]
    )

    assert exit_code == 0
    updated_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert updated_inventory["workshops"][0]["decision"] == "excluded"
    backfill_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert backfill_receipt["rows_updated"] == 2
    assert backfill_receipt["decision_breakdown"]["excluded"] == 1
