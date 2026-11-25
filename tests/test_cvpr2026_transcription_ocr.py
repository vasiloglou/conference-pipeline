"""Offline contract tests for the CVPR 2026 r5 caption-only Stage 5 adapter."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

from conference_pipeline.acquisition import cvpr_transcription_ocr as adapter
from conference_pipeline.acquisition.cvpr_event_media import canonical_event_media_inventory_digest
from conference_pipeline.transcription_ocr import sha256_file


def _caption_payload(video_id: str) -> str:
    return f"""WEBVTT

00:00:00.000 --> 00:00:01.000
caption for {video_id}

00:00:01.000 --> 00:00:02.000
second line
"""


def _write_caption(root: Path, video_id: str) -> dict[str, object]:
    path = root / "storage" / "download-corpus" / "event-media" / "captions" / f"{video_id}.en.vtt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_caption_payload(video_id), encoding="utf-8")
    return {
        "local_path": path.relative_to(root).as_posix(),
        "mime": "text/vtt",
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        "transcript_source": "youtube_caption_manual",
        "language": "en",
    }


def _caption_candidate(root: Path, index: int, *, available: bool) -> dict[str, object]:
    video_id = f"v{index:010d}"
    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "source_event_urls": [f"https://cvpr.thecvf.com/virtual/2026/poster/{index}"],
        "provenance": ["r5_official_event_page"],
        "caption_status": "youtube_caption_available" if available else "youtube_caption_unavailable",
        "caption": _write_caption(root, video_id) if available else None,
        "audio_downloaded": False,
        "video_downloaded": False,
    }


def _event_rows(candidate_count: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    categories = (
        ("poster", "poster", 5_023),
        ("invited-talk", "keynote", 3),
        ("tutorial", "tutorial", 19),
        ("workshop", "workshop", 153),
        ("session", "oral", 24),
        ("session", "session", 19),
        ("events", "main_event", 17),
    )
    rows: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    global_index = 0
    poster_index = 0
    for route_family, category, count in categories:
        category_counts[category] = count
        for local_index in range(1, count + 1):
            global_index += 1
            if route_family == "session":
                title = "Oral Session" if category == "oral" else "Session"
            else:
                title = f"Synthetic {category} {local_index}"
            direct = global_index <= candidate_count
            outcome_origin = "r4_registration_media" if route_family == "poster" and poster_index < 2_565 else "r5_authenticated_event_page"
            if route_family == "poster":
                poster_index += 1
            video_id = f"v{global_index:010d}"
            rows.append(
                {
                    "source_record_id": f"source-{global_index}",
                    "source_url": f"https://cvpr.thecvf.com/virtual/2026/{route_family}/{global_index}",
                    "route_family": route_family,
                    "event_category": category,
                    "title": title,
                    "page_status": "direct_youtube_recovered" if direct else "official_video_not_published",
                    "outcome_origin": outcome_origin,
                    "video_ids": [video_id] if direct else [],
                    "audio_downloaded": False,
                    "video_downloaded": False,
                }
            )
    return rows, category_counts


def _event_media_inventory(root: Path, *, candidate_count: int = 2) -> dict[str, object]:
    rows, category_counts = _event_rows(candidate_count)
    candidates = [
        _caption_candidate(root, index, available=index <= min(350, candidate_count))
        for index in range(1, candidate_count + 1)
    ]
    coverage = {
        category: {
            "event_route_count": count,
            "terminal_route_count": count,
            "direct_youtube_route_count": sum(
                row["event_category"] == category and row["page_status"] == "direct_youtube_recovered"
                for row in rows
            ),
            "status": "complete",
        }
        for category, count in category_counts.items()
    }
    inventory: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_event_media_inventory",
        "run_id": adapter.R5_EVENT_MEDIA_RUN_ID,
        "source_lineage": {
            "source_inventory_sha256": "a" * 64,
            "r4_registration_media_overlay_sha256": adapter.R4_OVERLAY_DIGEST,
        },
        "events": rows,
        "caption_candidates": candidates,
        "coverage": coverage,
        "counts": {
            "event_route_count": 5_258,
            "event_categories": category_counts,
            "oral_session_route_count": 24,
            "terminal_event_route_count": 5_258,
            "direct_youtube_route_count": candidate_count,
            "r4_reused_poster_route_count": 2_565,
            "caption_candidate_count": candidate_count,
        },
        "boundaries": {
            "audio_downloaded": False,
            "video_downloaded": False,
            "asr_dispatched": False,
            "ocr_dispatched": False,
            "provider_calls_made": 0,
        },
    }
    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)
    return inventory


def _pin_published_handoff(
    *,
    receipt_path: Path,
    inventory_path: Path,
    inventory: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind an isolated fixture to bytes rather than producer self-reporting."""

    monkeypatch.setattr(adapter, "R5_EVENT_MEDIA_RECEIPT_SHA256", sha256_file(receipt_path))
    monkeypatch.setattr(adapter, "R5_EVENT_MEDIA_FILE_SHA256", sha256_file(inventory_path))
    monkeypatch.setattr(adapter, "R5_EVENT_MEDIA_CANONICAL_DIGEST", canonical_event_media_inventory_digest(inventory))


def _write_published_handoff(root: Path, inventory: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> Path:
    (root / "config.yaml").write_text(
        "conference:\n"
        "  name: CVPR 2026 fixture\n"
        "urls:\n"
        "  virtual: https://cvpr.thecvf.com/virtual/2026\n",
        encoding="utf-8",
    )
    inventory_path = root / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    inventory_sha = sha256_file(inventory_path)
    canonical_sha = canonical_event_media_inventory_digest(inventory)
    receipt = {
        "run_id": adapter.R5_EVENT_MEDIA_RUN_ID,
        "stage_id": "download-corpus",
        "stage_order": 4,
        "mode": "live",
        "status": "succeeded",
        "event_media_inventory": {
            "path": inventory_path.relative_to(root).as_posix(),
            "inventory_sha256": canonical_sha,
        },
        "checks": {
            "event_media_inventory_path": inventory_path.relative_to(root).as_posix(),
            "event_media_inventory_sha256": inventory_sha,
            "event_media_inventory_canonical_sha256": canonical_sha,
            "r4_registration_media_overlay_sha256": adapter.R4_OVERLAY_DIGEST,
        },
        "stage_contract": {"status": "pass", "stage_id": "download-corpus"},
    }
    path = root / "storage" / "workflow-runs" / adapter.R5_EVENT_MEDIA_RUN_ID / "stage-04-download-corpus.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    _pin_published_handoff(
        receipt_path=path,
        inventory_path=inventory_path,
        inventory=inventory,
        monkeypatch=monkeypatch,
    )
    return path


@pytest.fixture
def r5_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    return tmp_path, _write_published_handoff(tmp_path, _event_media_inventory(tmp_path), monkeypatch)


def test_production_authority_constants_are_pinned() -> None:
    assert adapter.R5_EVENT_MEDIA_RUN_ID == "cvpr2026-live-production-r5-event-media"
    assert adapter.R5_EVENT_MEDIA_RECEIPT_SHA256 == "d6b0e6ac3dcea684ae699021d90c9ef737e239325b2fa496177efc25349038bc"
    assert adapter.R5_EVENT_MEDIA_FILE_SHA256 == "e6c7e00c3919e5652433a94ade336b121c83297dbd8780500576532d7e945a09"
    assert adapter.R5_EVENT_MEDIA_CANONICAL_DIGEST == "09e2d6ec1d8150b5e78f7d263018aeb227d73ee3a4109d13d9be162bf37bc55e"
    assert adapter.R4_OVERLAY_DIGEST == "7f622be554f054425f6dff6ae0911b3a0b83acb6699fe5a15ed37e70cd0d0d12"


@pytest.mark.parametrize(
    "mutation",
    [
        ("run_id", "substituted-run"),
        ("stage_id", "source-discovery"),
        ("stage_order", 3),
        ("mode", "dry-run"),
        ("status", "failed"),
        ("stage_contract.status", "blocked"),
        ("stage_contract.stage_id", "source-discovery"),
        ("event_media_inventory.path", "storage/download-corpus/event-media/substituted.json"),
        ("event_media_inventory.inventory_sha256", "0" * 64),
        ("checks.event_media_inventory_path", "storage/download-corpus/event-media/substituted.json"),
        ("checks.event_media_inventory_sha256", "0" * 64),
        ("checks.event_media_inventory_canonical_sha256", "0" * 64),
        ("checks.r4_registration_media_overlay_sha256", "0" * 64),
    ],
)
def test_published_receipt_admission_rejects_tampering_before_inventory_or_output(
    r5_workspace: tuple[Path, Path], mutation: tuple[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, receipt_path = r5_workspace
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    dotted_key, value = mutation
    target = receipt
    *parents, leaf = dotted_key.split(".")
    for parent in parents:
        target = target[parent]
    target[leaf] = value
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    inventory_path = workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    _pin_published_handoff(
        receipt_path=receipt_path,
        inventory_path=inventory_path,
        inventory=json.loads(inventory_path.read_text(encoding="utf-8")),
        monkeypatch=monkeypatch,
    )
    opened_labels: list[str] = []
    load_object = adapter._load_object

    def _only_receipt(path: Path, *, label: str) -> dict[str, object]:
        opened_labels.append(label)
        if label == "event-media inventory":
            raise AssertionError("invalid receipt must stop before opening the declared inventory")
        return load_object(path, label=label)

    monkeypatch.setattr(adapter, "_load_object", _only_receipt)

    with pytest.raises(adapter.CvprTranscriptionOcrAdmissionError):
        adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)

    assert opened_labels in ([], ["stage-04 receipt"])
    assert not (workspace / "storage" / "transcription-ocr" / "cvpr2026").exists()
    assert not (workspace / "storage" / "transcription-ocr" / "cvpr2026" / "inventory.json").exists()


def test_published_receipt_bytes_are_pinned_before_receipt_json_is_opened(
    r5_workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, receipt_path = r5_workspace
    receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    opened_labels: list[str] = []
    load_object = adapter._load_object

    def _record_open(path: Path, *, label: str) -> dict[str, object]:
        opened_labels.append(label)
        return load_object(path, label=label)

    monkeypatch.setattr(adapter, "_load_object", _record_open)

    with pytest.raises(adapter.CvprTranscriptionOcrAdmissionError):
        adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)

    assert opened_labels == []
    assert not (workspace / "storage" / "transcription-ocr" / "cvpr2026").exists()


def test_published_receipt_success_binds_computed_receipt_digest(
    r5_workspace: tuple[Path, Path]
) -> None:
    workspace, receipt_path = r5_workspace

    result = adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)

    expected_receipt_sha = sha256_file(receipt_path)
    assert result["compatibility"]["stage4_receipt_sha256"] == expected_receipt_sha
    assert result["input_digests"]["stage4_receipt_sha256"] == expected_receipt_sha
    assert result["counts"]["caption_used_no_asr_count"] == 2
    assert result["counts"]["youtube_caption_unavailable_count"] == 0
    assert result["asr_queue"] == []
    assert result["ocr_queue"] == []
    assert result["provider_calls"] == []


def test_read_only_handoff_passes_its_configured_boundary_to_terminal_inventory_validation(
    r5_workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, receipt_path = r5_workspace
    captured: dict[str, object] = {}
    original = adapter.validate_event_media_inventory

    def _require_boundary(inventory: object, *, boundary: object, **kwargs: object) -> None:
        captured["boundary"] = boundary
        original(inventory, boundary=boundary, **kwargs)

    monkeypatch.setattr(adapter, "validate_event_media_inventory", _require_boundary)

    adapter.validate_cvpr_r5_handoff_read_only(workspace=workspace, receipt_path=receipt_path)

    boundary = captured["boundary"]
    assert getattr(boundary, "origin") == "https://cvpr.thecvf.com"
    assert getattr(boundary, "root_segments") == ("virtual", "2026")


@pytest.mark.parametrize(
    "config_text",
    [
        None,
        "conference:\n  name: CVPR 2026 fixture\nurls:\n  virtual: http://cvpr.thecvf.com/virtual/2026\n",
        "conference:\n  name: CVPR 2026 fixture\nurls:\n  virtual: https://foreign.example/virtual/2027\n",
    ],
)
def test_read_only_handoff_translates_workspace_boundary_failures_before_stage5_output(
    r5_workspace: tuple[Path, Path], config_text: str | None
) -> None:
    workspace, receipt_path = r5_workspace
    inventory_path = workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    receipt_before = sha256_file(receipt_path)
    inventory_before = sha256_file(inventory_path)
    config_path = workspace / "config.yaml"
    if config_text is None:
        config_path.unlink()
    else:
        config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(adapter.CvprTranscriptionOcrAdmissionError):
        adapter.validate_cvpr_r5_handoff_read_only(workspace=workspace, receipt_path=receipt_path)

    assert sha256_file(receipt_path) == receipt_before
    assert sha256_file(inventory_path) == inventory_before
    assert not (workspace / adapter.STAGE5_DIR).exists()


def test_read_only_handoff_verifier_never_opens_stage5_output(
    r5_workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, receipt_path = r5_workspace
    script_path = Path(__file__).parents[1] / "scripts" / "verify_cvpr2026_r5_handoff_read_only.py"
    spec = importlib.util.spec_from_file_location("cvpr_r5_handoff_verifier", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.adapter, "R5_EVENT_MEDIA_RECEIPT_SHA256", sha256_file(receipt_path))
    monkeypatch.setattr(module.adapter, "R5_EVENT_MEDIA_FILE_SHA256", adapter.R5_EVENT_MEDIA_FILE_SHA256)
    monkeypatch.setattr(module.adapter, "R5_EVENT_MEDIA_CANONICAL_DIGEST", adapter.R5_EVENT_MEDIA_CANONICAL_DIGEST)
    monkeypatch.setattr(module.adapter, "R4_OVERLAY_DIGEST", adapter.R4_OVERLAY_DIGEST)

    assert module.main(["--workspace", str(workspace), "--require-absent-stage5-output"]) == 0

    verified = json.loads(capsys.readouterr().out)
    assert verified["stage4_receipt_sha256"] == sha256_file(receipt_path)
    assert verified["event_media_file_sha256"] == adapter.R5_EVENT_MEDIA_FILE_SHA256
    assert verified["event_media_canonical_sha256"] == adapter.R5_EVENT_MEDIA_CANONICAL_DIGEST
    assert verified["r4_overlay_sha256"] == adapter.R4_OVERLAY_DIGEST
    assert not (workspace / adapter.STAGE5_DIR).exists()


def test_caption_only_inventory_preserves_provenance_and_no_asr_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _event_media_inventory(tmp_path, candidate_count=2_352)
    receipt_path = _write_published_handoff(tmp_path, inventory, monkeypatch)

    result = adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=tmp_path, receipt_path=receipt_path)

    assert result["counts"] == {
        "terminal_route_count": 5_258,
        "caption_candidate_count": 2_352,
        "caption_used_no_asr_count": 350,
        "youtube_caption_unavailable_count": 2_002,
        "asr_queue_count": 0,
        "ocr_queue_count": 0,
        "provider_call_count": 0,
    }
    assert len(result["caption_transcripts"]) == 350
    assert len(result["blockers"]) == 2_002
    assert {record["source_class"] for record in result["caption_transcripts"]} == {"caption_transcript"}
    assert {record["outcome"] for record in result["caption_transcripts"]} == {"caption_used_no_asr"}
    assert {record["code"] for record in result["blockers"]} == {"youtube_caption_unavailable"}
    assert {record["asr_eligible"] for record in result["blockers"]} == {False}


@pytest.mark.parametrize(
    "field,value",
    [
        ("caption.local_path", "../escape.vtt"),
        ("caption.mime", "text/plain"),
        ("caption.byte_count", 1),
        ("caption.sha256", "0" * 64),
        ("video_id", "wrongvideoid"),
        ("caption_status", "pending"),
    ],
)
def test_caption_integrity_and_terminal_class_boundaries_are_fail_closed(
    r5_workspace: tuple[Path, Path], field: str, value: object
) -> None:
    workspace, receipt_path = r5_workspace
    inventory_path = workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidate = inventory["caption_candidates"][0]
    target = candidate
    *parents, leaf = field.split(".")
    for parent in parents:
        target = target[parent]
    target[leaf] = value
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(adapter.CvprTranscriptionOcrAdmissionError):
        adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)


def test_duplicate_source_identity_and_class_reclassification_are_rejected(
    r5_workspace: tuple[Path, Path]
) -> None:
    workspace, receipt_path = r5_workspace
    inventory_path = workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    duplicate = dict(inventory["caption_candidates"][0])
    duplicate["caption_status"] = "youtube_caption_unavailable"
    duplicate["caption"] = None
    inventory["caption_candidates"].append(duplicate)
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(adapter.CvprTranscriptionOcrAdmissionError):
        adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)


def test_bounded_workshop_receipt_rows_produce_only_class_correct_outputs(
    r5_workspace: tuple[Path, Path]
) -> None:
    workspace, receipt_path = r5_workspace
    pdf_path = workspace / "storage" / "workshops" / "papers" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\nSynthetic PDF")
    zip_path = workspace / "storage" / "workshops" / "papers" / "zip-content.pdf"
    zip_path.write_bytes(b"PK\x03\x04not-a-pdf")
    valid_row = {
        "source_identity": "workshop:paper-1",
        "predecessor_receipt_sha256": "a" * 64,
        "receipt_index_sha256": "b" * 64,
        "declared_path": pdf_path.relative_to(workspace).as_posix(),
        "mime_type": "application/pdf",
        "byte_count": pdf_path.stat().st_size,
        "sha256": sha256_file(pdf_path),
    }
    rows = [
        valid_row,
        {**valid_row, "source_identity": "workshop:missing-lineage", "predecessor_receipt_sha256": None},
        {**valid_row, "source_identity": "workshop:non-pdf", "mime_type": "text/plain"},
        {
            **valid_row,
            "source_identity": "workshop:zip",
            "declared_path": zip_path.relative_to(workspace).as_posix(),
            "byte_count": zip_path.stat().st_size,
            "sha256": sha256_file(zip_path),
        },
    ]

    result = adapter.build_cvpr_r5_transcription_ocr_inventory(
        workspace=workspace,
        receipt_path=receipt_path,
        workshop_pdf_receipt_index={
            "sha256": "b" * 64,
            "predecessor_receipt_sha256": "a" * 64,
            "rows": rows,
        },
    )

    by_identity = {row["source_identity"]: row for row in result["workshop_pdf_provenance_table"]}
    assert by_identity["workshop:paper-1"]["outcome_class"] == "workshop_paper_text_extraction"
    assert by_identity["workshop:missing-lineage"]["outcome_class"] == "workshop_pdf_lineage_unproven"
    assert by_identity["workshop:non-pdf"]["outcome_class"] == "workshop_pdf_mime_invalid"
    assert by_identity["workshop:zip"]["outcome_class"] == "workshop_pdf_zip_content"


def test_resume_requires_full_compatibility_key_and_keeps_superseded_attempt(
    r5_workspace: tuple[Path, Path]
) -> None:
    workspace, receipt_path = r5_workspace
    first = adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)
    reused = adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)

    assert reused["resume"]["reused_output_digest"] == first["inventory_sha256"]
    output_path = workspace / "storage" / "transcription-ocr" / "cvpr2026" / "inventory.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["processor_version"] = "0"
    output_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    reprocessed = adapter.build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)
    assert reprocessed["resume"]["superseded_attempt_count"] == 1
    assert reprocessed["processor_identity"] == "cvpr2026.caption_materializer"
    assert reprocessed["processor_version"] == "1"
