"""Phase 11 (release bundle) and Phase 12 (final proof) contracts.

Neither stage had ever been implemented, for any conference -- all three
production runners carried ``implemented=False`` and a "implement this before
dispatch" next action. These tests pin the behaviour that matters: what the
gates refuse, and why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.final_proof import (
    build_proof_packet,
    evaluate_support_label_promotions,
    map_artifacts_to_receipts,
    proof_packet_blockers,
    residual_risks,
)
from conference_pipeline.release_bundle import (
    build_release_inventory,
    directory_digest,
    publication_status,
    rejected_evidence_markers,
    release_inventory_blockers,
    storage_lanes,
)


# ---------------------------------------------------------------------------
# Phase 11
# ---------------------------------------------------------------------------


def test_publication_is_blocked_unless_target_and_approval_are_both_given() -> None:
    """Done gate 2. Assembling a bundle is not permission to publish it."""

    assert publication_status(target=None, approved_by=None)["blocked"]
    assert publication_status(target="drive", approved_by=None)["blocked"]
    assert publication_status(target=None, approved_by="operator")["blocked"]
    granted = publication_status(target="drive", approved_by="operator")
    assert not granted["blocked"]
    assert granted["target"] == "drive"
    assert granted["approved_by"] == "operator"
    # Whitespace is not an approval.
    assert publication_status(target="  ", approved_by="  ")["blocked"]


def test_rejected_evidence_is_detected_by_field_never_by_free_text() -> None:
    """A substring scan blocks the release on a paper title.

    CVPR 2026's corpus contains a paper called "Scaffold: Elastic ...", whose
    title appears verbatim in the source-discovery receipt, and receipts
    legitimately record ``"dry_run": false``. Both must stay clean.
    """

    assert rejected_evidence_markers({"mode": "mock"}) == ["mode=mock"]
    assert rejected_evidence_markers({"dry_run": True}) == ["dry_run=true"]
    assert rejected_evidence_markers({"checks": {"steps": [{"scaffolded": True}]}}) == [
        "scaffolded=true"
    ]
    assert rejected_evidence_markers({"dry_run": False, "mode": "live"}) == []
    assert rejected_evidence_markers({"title": "Scaffold: Elastic Something", "mode": "live"}) == []
    assert rejected_evidence_markers({"note": "not a dry-run, real work"}) == []


def test_directory_digest_is_order_independent_and_content_bound(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "b.txt").write_text("second", encoding="utf-8")
    (root / "sub" / "a.txt").write_text("first", encoding="utf-8")
    digest, count, size = directory_digest(root)
    assert count == 2
    assert size == len("second") + len("first")
    # Same content, rebuilt in the other order, digests the same.
    other = tmp_path / "tree2"
    (other / "sub").mkdir(parents=True)
    (other / "sub" / "a.txt").write_text("first", encoding="utf-8")
    (other / "b.txt").write_text("second", encoding="utf-8")
    assert directory_digest(other)[0] == digest
    # Changed content digests differently.
    (other / "b.txt").write_text("changed", encoding="utf-8")
    assert directory_digest(other)[0] != digest


def _write_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conf"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "analysis").mkdir(parents=True)
    (workspace / "decks" / "01_topic").mkdir(parents=True)
    (workspace / "storage" / "storage_manifest.json").write_text(
        json.dumps(
            {
                "artifact_families": [
                    {"id": "analysis", "path": "analysis", "storage_class": "git",
                     "share_policy": "curated", "producer": "analysis", "rebuildable": False},
                    {"id": "decks", "path": "decks", "storage_class": "backup",
                     "share_policy": "private", "producer": "presentation", "rebuildable": True},
                    {"id": "absent", "path": "nowhere", "storage_class": "cache",
                     "share_policy": "none", "producer": "x", "rebuildable": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    (workspace / "analysis" / "report.md").write_text("report", encoding="utf-8")
    (workspace / "decks" / "01_topic" / "presentation.pdf").write_bytes(b"%PDF fixture")
    return workspace


def test_storage_lanes_group_families_and_record_presence(tmp_path: Path) -> None:
    """Done gate 1, storage-lane half."""

    workspace = _write_workspace(tmp_path)
    manifest = json.loads((workspace / "storage" / "storage_manifest.json").read_text())
    result = storage_lanes(workspace, manifest)
    lanes = result["lanes"]
    assert lanes["git"]["family_count"] == 1
    assert lanes["git"]["present_count"] == 1
    assert lanes["cache"]["present_count"] == 0, "a declared-but-absent family is recorded, not dropped"
    assert {row["id"] for row in result["families"]} == {"analysis", "decks", "absent"}


def test_release_inventory_digests_deliverables_and_blocks_on_a_missing_one(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)
    receipt = tmp_path / "stage-07-analysis.json"
    receipt.write_text(json.dumps({"status": "succeeded", "mode": "live"}), encoding="utf-8")

    inventory = build_release_inventory(
        workspace=workspace,
        run_id="run-1",
        conference_name="Test 2026",
        stage_receipts={"analysis": receipt},
        deliverables=[
            {"id": "report", "path": "analysis/report.md", "kind": "report", "produced_by": "analysis"},
            {"id": "gone", "path": "analysis/missing.md", "kind": "report", "produced_by": "analysis"},
        ],
        decks_dir="decks",
        deck_ids=["01_topic"],
    )
    present = {row["path"]: row for row in inventory["deliverables"]}
    assert present["analysis/report.md"]["sha256"]
    assert not present["analysis/missing.md"]["present"]
    assert inventory["publication"]["blocked"]

    codes = {item["code"] for item in release_inventory_blockers(inventory)}
    assert "release_deliverable_missing" in codes


def test_release_blocks_when_an_upstream_receipt_did_not_succeed(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)
    receipt = tmp_path / "stage-07-analysis.json"
    receipt.write_text(json.dumps({"status": "blocked", "mode": "live"}), encoding="utf-8")
    inventory = build_release_inventory(
        workspace=workspace,
        run_id="run-1",
        conference_name="Test 2026",
        stage_receipts={"analysis": receipt},
        deliverables=[
            {"id": "report", "path": "analysis/report.md", "kind": "report", "produced_by": "analysis"}
        ],
        decks_dir="decks",
        deck_ids=[],
    )
    codes = {item["code"] for item in release_inventory_blockers(inventory)}
    assert "release_upstream_stage_not_succeeded" in codes


def test_release_blocks_when_an_upstream_receipt_carries_rejected_evidence(tmp_path: Path) -> None:
    """Done gate 3: readiness-only and mock evidence cannot satisfy release."""

    workspace = _write_workspace(tmp_path)
    receipt = tmp_path / "stage-07-analysis.json"
    receipt.write_text(
        json.dumps({"status": "succeeded", "mode": "mock"}), encoding="utf-8"
    )
    inventory = build_release_inventory(
        workspace=workspace,
        run_id="run-1",
        conference_name="Test 2026",
        stage_receipts={"analysis": receipt},
        deliverables=[
            {"id": "report", "path": "analysis/report.md", "kind": "report", "produced_by": "analysis"}
        ],
        decks_dir="decks",
        deck_ids=[],
    )
    assert inventory["rejected_evidence"]
    codes = {item["code"] for item in release_inventory_blockers(inventory)}
    assert "release_rejected_evidence_cited" in codes


# ---------------------------------------------------------------------------
# Phase 12
# ---------------------------------------------------------------------------


def _inventory_with(deliverables: list[dict], provenance: list[dict]) -> dict:
    return {
        "generated_at": "2026-09-12T00:00:00Z",
        "deliverables": deliverables,
        "provenance": provenance,
        "review_status": [],
        "publication": {"blocked": True},
        "rejected_evidence": [],
    }


def test_every_artifact_maps_to_the_receipt_that_produced_it() -> None:
    """Done gate 1."""

    inventory = _inventory_with(
        deliverables=[
            {"id": "a", "path": "x.pdf", "present": True, "sha256": "aa", "produced_by": "presentation"},
            {"id": "b", "path": "y.pdf", "present": False, "produced_by": "presentation"},
        ],
        provenance=[{"stage_id": "presentation", "receipt_path": "r.json", "status": "succeeded", "sha256": "rr"}],
    )
    rows, unmapped = map_artifacts_to_receipts(inventory)
    assert len(rows) == 1, "an absent deliverable is not claimed as proven"
    assert rows[0]["receipt_path"] == "r.json"
    assert not unmapped


def test_an_artifact_whose_stage_did_not_succeed_is_a_blocker() -> None:
    inventory = _inventory_with(
        deliverables=[
            {"id": "a", "path": "x.pdf", "present": True, "sha256": "aa", "produced_by": "presentation"}
        ],
        provenance=[{"stage_id": "presentation", "receipt_path": "r.json", "status": "blocked"}],
    )
    packet = build_proof_packet(
        workspace=Path("/tmp"),
        run_id="r",
        conference_name="Test",
        inventory=inventory,
        stage_receipts={},
        support_label_candidates=[],
    )
    assert packet["unmapped_artifacts"] == ["x.pdf"]
    codes = {item["code"] for item in proof_packet_blockers(packet)}
    assert "proof_artifact_without_receipt" in codes


def test_a_support_label_is_promoted_only_on_a_local_live_succeeded_receipt(tmp_path: Path) -> None:
    """Done gate 2. The label claims "proven HERE", so another conference's
    evidence, a blocked run, or a dry-run mode can never grant it."""

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"status": "succeeded", "mode": "live"}), encoding="utf-8")
    blocked = tmp_path / "blocked.json"
    blocked.write_text(json.dumps({"status": "blocked", "mode": "live"}), encoding="utf-8")
    dry = tmp_path / "dry.json"
    dry.write_text(json.dumps({"status": "succeeded", "mode": "dry-run"}), encoding="utf-8")

    candidates = [
        {"capability": "proven", "current_label": "x", "proposed_label": "y", "requires_stages": ["a"]},
        {"capability": "blocked-stage", "current_label": "x", "proposed_label": "y", "requires_stages": ["b"]},
        {"capability": "dry-run-stage", "current_label": "x", "proposed_label": "y", "requires_stages": ["c"]},
        {"capability": "absent-receipt", "current_label": "x", "proposed_label": "y", "requires_stages": ["d"]},
        {"capability": "no-requirement", "current_label": "x", "proposed_label": "y", "requires_stages": []},
    ]
    decisions = {
        row["capability"]: row
        for row in evaluate_support_label_promotions(
            candidates=candidates, stage_receipts={"a": good, "b": blocked, "c": dry}
        )
    }
    assert decisions["proven"]["promoted"]
    assert decisions["proven"]["evidence"][0]["stage_id"] == "a"
    for name in ("blocked-stage", "dry-run-stage", "absent-receipt", "no-requirement"):
        assert not decisions[name]["promoted"], name
        assert decisions[name]["refused_because"] is not None or name == "no-requirement"


def test_residual_risks_carry_accepted_findings_forward() -> None:
    """Accepted is not the same as absent: a reader of the proof packet sees
    every slide finding that was accepted rather than fixed."""

    inventory = _inventory_with(deliverables=[], provenance=[])
    inventory["review_status"] = [
        {"deck_id": "01", "visual_qc": {"pass": 24, "fail": 3},
         "video_single_speaker": True, "video_sync_single_speaker": "pass",
         "video_notebooklm": True, "video_sync_notebooklm": None},
    ]
    risks = residual_risks(inventory)
    kinds = {risk["kind"] for risk in risks}
    assert "visual_qc_findings_accepted" in kinds
    assert "video_sync_not_passing" in kinds, "a video whose sync did not pass is a risk"
    accepted = next(r for r in risks if r["kind"] == "visual_qc_findings_accepted")
    assert "3 of 27" in accepted["detail"]
