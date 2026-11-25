"""Local contract tests for the CVPR Stage-4 talk-spine adapter."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.acquisition.cvpr_program_spine import (
    CVPR_YOUTUBE_ALLOWED_HOSTS,
    assess_cached_cvpr_slideslive_recovery,
    build_cvpr_program_spine,
    canonical_program_spine_digest,
)


def _record(
    record_id: str,
    *,
    family: str,
    title: str | None,
    url: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "source_record_id": record_id,
        "family_id": family,
        "snapshot_sha256": "a" * 64,
        "aliases": [url],
        "source_evidence": {
            "final_url": "https://cvpr.thecvf.com/virtual/2026/calendar",
            "metadata_path": "cache/source-discovery/frozen/snapshot.json",
            "snapshot_sha256": "a" * 64,
        },
    }
    if title is not None:
        record["title"] = title
    return record


def _selected(item_id: str, record_id: str) -> dict[str, object]:
    return {
        "program_item_id": item_id,
        "selection_status": "included",
        "source_record_ids": [record_id],
    }


def test_program_spine_is_deterministic_source_bound_and_keeps_colliding_titles() -> None:
    records = [
        _record(
            "cvpr2026:source:program-schedule:oral-a",
            family="program-schedule",
            title="A Shared Title",
            url="https://cvpr.thecvf.com/virtual/2026/oral/101",
        ),
        _record(
            "cvpr2026:source:program-schedule:oral-b",
            family="program-schedule",
            title="A Shared Title",
            url="https://cvpr.thecvf.com/virtual/2026/oral/102",
        ),
        _record(
            "cvpr2026:source:program-schedule:break",
            family="program-schedule",
            title="Coffee break",
            url="https://cvpr.thecvf.com/virtual/2026/break/103",
        ),
        _record(
            "cvpr2026:source:youtube:generic",
            family="youtube",
            title="CVPR", 
            url="https://www.youtube.com/",
        ),
    ]
    selection = [
        _selected("cvpr2026:program-item:oral-a", records[0]["source_record_id"]),
        _selected("cvpr2026:program-item:oral-b", records[1]["source_record_id"]),
        _selected("cvpr2026:program-item:break", records[2]["source_record_id"]),
        _selected("cvpr2026:program-item:youtube", records[3]["source_record_id"]),
    ]

    first = build_cvpr_program_spine(
        selection_rows=selection,
        source_records=records,
        lineage={"phase_03_selection_sha256": "b" * 64},
    )
    second = build_cvpr_program_spine(
        selection_rows=list(reversed(selection)),
        source_records=list(reversed(records)),
        lineage={"phase_03_selection_sha256": "b" * 64},
    )

    assert first["program_spine_sha256"] == second["program_spine_sha256"]
    assert canonical_program_spine_digest(first) == first["program_spine_sha256"]
    assert [row["item_id"] for row in first["items"]] == [
        "cvpr2026:program-item:oral-a",
        "cvpr2026:program-item:oral-b",
    ]
    assert {row["source_record_id"] for row in first["items"]} == {
        "cvpr2026:source:program-schedule:oral-a",
        "cvpr2026:source:program-schedule:oral-b",
    }
    assert all(row["family"] == "oral" for row in first["items"])
    assert all("speakers" in row["metadata_blockers"] for row in first["items"])
    assert all("session" in row["metadata_blockers"] for row in first["items"])
    dispositions = {row["program_item_id"]: row for row in first["non_talk_rows"]}
    assert dispositions["cvpr2026:program-item:break"]["reason"] == "schedule_event_not_talk_capable"
    assert dispositions["cvpr2026:program-item:youtube"]["reason"] == "youtube_source_is_not_authoritative_talk_evidence"
    assert first["counts"] == {"selected_rows": 4, "talk_spine_items": 2, "non_talk_rows": 2}


def test_program_spine_records_missing_title_without_inventing_a_talk() -> None:
    record = _record(
        "cvpr2026:source:program-schedule:missing-title",
        family="program-schedule",
        title=None,
        url="https://cvpr.thecvf.com/virtual/2026/keynote/77",
    )

    spine = build_cvpr_program_spine(
        selection_rows=[_selected("cvpr2026:program-item:missing-title", record["source_record_id"])],
        source_records=[record],
        lineage={"phase_03_selection_sha256": "b" * 64},
    )

    assert spine["items"] == []
    assert spine["metadata_blockers"] == [
        {
            "code": "talk_metadata_missing",
            "fields": ["title"],
            "program_item_id": "cvpr2026:program-item:missing-title",
            "source_record_id": "cvpr2026:source:program-schedule:missing-title",
        }
    ]
    assert spine["non_talk_rows"][0]["reason"] == "talk_metadata_missing"


def test_cached_cvpr_slideslive_assessment_uses_only_existing_item_page_probes(tmp_path: Path) -> None:
    workspace = tmp_path / "cvpr2026"
    probe = workspace / "cache" / "item-page-probes" / "cached.html"
    probe.parent.mkdir(parents=True)
    probe.write_text("<html>no embedded provider</html>", encoding="utf-8")
    ignored = workspace / "cache" / "source-pages" / "ignored.html"
    ignored.parent.mkdir(parents=True)
    ignored.write_text('<iframe src="https://slideslive.com/12345"></iframe>', encoding="utf-8")
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "item_page_auth_probe",
                "status": "succeeded",
                "local_path": probe.relative_to(workspace).as_posix(),
                "program_item_id": "cvpr2026:program-item:one",
                "source_record_id": "cvpr2026:source:program-schedule:one",
                "source_url": "https://cvpr.thecvf.com/virtual/2026/poster/1",
            },
            {
                "artifact_kind": "source_page",
                "status": "succeeded",
                "local_path": ignored.relative_to(workspace).as_posix(),
            },
        ]
    }

    assessment = assess_cached_cvpr_slideslive_recovery(workspace=workspace, ledger=ledger)

    assert assessment == {
        "status": "not_applicable",
        "reason": "no_cached_cvpr_item_page_auth_probe_with_slideslive_embed",
        "inspected_probe_count": 1,
        "embedded_assets": [],
        "allowed_hosts": list(CVPR_YOUTUBE_ALLOWED_HOSTS),
        "probes_created": False,
    }
    assert "slideslive.com" not in assessment["allowed_hosts"]


def test_cached_cvpr_slideslive_assessment_marks_existing_embed_for_focused_recovery(tmp_path: Path) -> None:
    workspace = tmp_path / "cvpr2026"
    probe = workspace / "cache" / "item-page-probes" / "cached.html"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        '<iframe src="https://slideslive.com/embed/presentation/39056835?token=secret"></iframe>',
        encoding="utf-8",
    )

    assessment = assess_cached_cvpr_slideslive_recovery(
        workspace=workspace,
        ledger={
            "artifacts": [
                {
                    "artifact_kind": "item_page_auth_probe",
                    "status": "succeeded",
                    "local_path": probe.relative_to(workspace).as_posix(),
                    "program_item_id": "cvpr2026:program-item:one",
                    "source_record_id": "cvpr2026:source:program-schedule:one",
                    "source_url": "https://cvpr.thecvf.com/virtual/2026/poster/1",
                }
            ]
        },
    )

    assert assessment["status"] == "eligible_cached_recovery"
    assert assessment["embedded_assets"] == [
        {
            "page_artifact_path": probe.relative_to(workspace).as_posix(),
            "page_url": "https://cvpr.thecvf.com/virtual/2026/poster/1",
            "program_item_id": "cvpr2026:program-item:one",
            "slideslive_id": "39056835",
            "slideslive_url": "https://slideslive.com/39056835",
            "source_record_id": "cvpr2026:source:program-schedule:one",
        }
    ]
    assert "secret" not in json.dumps(assessment)
