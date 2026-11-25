from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.workshop_talk_segmentation import (
    PROVIDER_DISCOVERY_FAILED_CODE,
    PROVIDER_NOT_CONFIGURED_CODE,
    PROVIDER_COST_NOT_APPROVED_CODE,
    WORKSHOP_TALK_DISCOVERY_COST_TASK_ID,
    WORKSHOP_TALK_DISCOVERY_OWNER_ID,
    WORKSHOP_TALK_DISCOVERY_RECEIPT_TYPE,
    WORKSHOP_TALK_SEGMENTATION_RECEIPT_TYPE,
    annotate_segments_with_talks,
    build_talk_context_batches,
    find_overlapping_talks,
    load_transcript_segments,
    load_talks_json,
    parse_talk_discovery_response,
    run_workshop_talk_boundary_discovery,
    run_workshop_talk_segmentation,
    write_talk_segmentation_receipt,
)


def test_loads_neurips_style_talks_and_overlaps(tmp_path: Path) -> None:
    talks_path = tmp_path / "talks.json"
    talks_path.write_text(
        json.dumps(
            {
                "workshop_id": "workshop_demo",
                "workshop_title": "Workshop Demo",
                "talks": [
                    {
                        "talk_id": "talk_001",
                        "title": "Talk Demo",
                        "speaker": "Dr Demo",
                        "start_sec": 10,
                        "end_sec": 40,
                        "description": "Demo abstract",
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    talks = load_talks_json(talks_path)
    overlaps = find_overlapping_talks(talks, timestamp_start=12, timestamp_end=20)

    assert talks[0].talk_id == "talk_001"
    assert talks[0].talk_title == "Talk Demo"
    assert talks[0].talk_speaker == "Dr Demo"
    assert talks[0].workshop_id == "workshop_demo"
    assert talks[0].timestamp_start == 10
    assert talks[0].timestamp_end == 40
    assert overlaps == talks


def test_annotates_segments_and_writes_receipt(tmp_path: Path) -> None:
    talks_path = tmp_path / "talks.json"
    talks_path.write_text(
        json.dumps(
            {
                "workshop_id": "workshop_demo",
                "workshop_title": "Workshop Demo",
                "talks": [
                    {
                        "talk_id": "talk_001",
                        "title": "Talk Demo",
                        "speaker": "Dr Demo",
                        "start_sec": 0,
                        "end_sec": 30,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    talks = load_talks_json(talks_path)
    annotated = annotate_segments_with_talks(
        [{"start": 5.0, "end": 8.0, "text": "hello"}],
        talks,
    )

    assert annotated[0]["talk_id"] == "talk_001"
    assert annotated[0]["talk_title"] == "Talk Demo"
    assert annotated[0]["workshop_id"] == "workshop_demo"

    receipt_path = tmp_path / "receipt.json"
    receipt = write_talk_segmentation_receipt(
        path=receipt_path,
        workspace=tmp_path,
        status="succeeded",
        talks=talks,
        output_path=tmp_path / "output.json",
    )

    assert receipt["receipt_type"] == WORKSHOP_TALK_SEGMENTATION_RECEIPT_TYPE
    assert receipt["status"] == "succeeded"
    assert receipt["provider_calls_made"] is False
    assert receipt["secrets_copied"] is False


def test_load_transcript_segments_synthesizes_from_transcript_text_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    transcript_text = (
        workspace / "storage" / "transcription-ocr" / "speech-transcripts" / "talk.txt"
    )
    transcript_json = transcript_text.with_suffix(".json")
    transcript_text.parent.mkdir(parents=True)
    transcript_text.write_text(
        "first section\n\nsecond section\n\nthird section\n",
        encoding="utf-8",
    )
    transcript_json.write_text(
        json.dumps(
            {
                "transcript_path": str(transcript_text.relative_to(workspace)),
                "source_duration_seconds": 900,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    segments = load_transcript_segments(transcript_json, workspace=workspace)

    assert len(segments) == 3
    assert segments[0]["start"] == 0
    assert segments[0]["end"] == 300
    assert segments[-1]["end"] == 900
    assert all(segment["synthetic_timestamp"] is True for segment in segments)


def test_parse_talk_discovery_response_repairs_near_json() -> None:
    talks = parse_talk_discovery_response(
        """
        ```json
        {
          "talks": [
            {
              "talk_id": "talk_001",
              "talk_title": "Opening",
              "talk_speaker": "Unknown",
              "timestamp_start": 0,
              "timestamp_end": 120,
            }
          ]
        }
        ```
        """,
        default_workshop_id="workshop_demo",
        default_workshop_title="Workshop Demo",
    )

    assert talks[0].talk_id == "talk_001"
    assert talks[0].timestamp_end == 120


def test_annotation_preserves_segment_timestamps_when_talk_metadata_merges(tmp_path: Path) -> None:
    """Regression: talk metadata must not overwrite segment timestamps.

    Earlier code merged the entire talk record (including timestamp_start /
    timestamp_end) onto each segment, corrupting downstream segment timing.
    """
    talks_path = tmp_path / "talks.json"
    talks_path.write_text(
        json.dumps(
            {
                "workshop_id": "workshop_demo",
                "workshop_title": "Workshop Demo",
                "talks": [
                    {
                        "talk_id": "talk_001",
                        "title": "Talk Demo",
                        "speaker": "Dr Demo",
                        "start_sec": 0,
                        "end_sec": 300,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    talks = load_talks_json(talks_path)
    segment = {
        "timestamp_start": 5.0,
        "timestamp_end": 8.0,
        "text": "short utterance",
    }
    annotated = annotate_segments_with_talks([segment], talks)

    assert annotated[0]["timestamp_start"] == 5.0
    assert annotated[0]["timestamp_end"] == 8.0
    assert annotated[0]["talk_id"] == "talk_001"
    assert annotated[0]["talk_title"] == "Talk Demo"
    assert annotated[0]["talk_speaker"] == "Dr Demo"
    assert annotated[0]["workshop_id"] == "workshop_demo"


def test_run_workshop_talk_segmentation_offline_and_blocked(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    talks_json = workspace / "talks.json"
    transcript_json = workspace / "transcript.json"
    output_path = workspace / "annotated.json"
    receipt_path = workspace / "receipt.json"
    talks_json.write_text(
        json.dumps(
            {
                "workshop_id": "workshop_demo",
                "talks": [{"talk_id": "talk_001", "title": "Talk Demo", "start_sec": 0, "end_sec": 20}],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    transcript_json.write_text(
        json.dumps({"segments": [{"start": 1, "end": 2, "text": "hi"}]}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    receipt = run_workshop_talk_segmentation(
        workspace=workspace,
        input_transcript_json=transcript_json,
        talks_json=talks_json,
        output_path=output_path,
        receipt_path=receipt_path,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "succeeded"
    assert output["segments"][0]["talk_id"] == "talk_001"

    blocked = run_workshop_talk_segmentation(
        workspace=workspace,
        output_path=workspace / "blocked.json",
        receipt_path=workspace / "blocked-receipt.json",
    )

    assert blocked["status"] == "blocked"
    assert blocked["blocker_code"] == PROVIDER_NOT_CONFIGURED_CODE


def test_parses_gemini_talk_discovery_response_and_builds_batches() -> None:
    talks = parse_talk_discovery_response(
        """```json
        {"talks": [
          {"talk_id": "talk_002", "talk_title": "Second", "talk_speaker": "B", "timestamp_start": 300, "timestamp_end": 600},
          {"talk_id": "talk_001", "talk_title": "First", "talk_speaker": "A", "timestamp_start": 0, "timestamp_end": 300}
        ]}
        ```""",
        default_workshop_id="ws",
        default_workshop_title="Workshop",
    )
    batches = build_talk_context_batches(
        [
            {"start": 0, "end": 100, "text": "first part"},
            {"start": 320, "end": 350, "text": "second part"},
        ],
        talks,
    )

    assert [talk.talk_id for talk in talks] == ["talk_001", "talk_002"]
    assert batches[0]["talk_ids"] == ["talk_001"]
    assert batches[1]["talk_ids"] == ["talk_002"]


class _FakeGeminiResponse:
    text = json.dumps(
        {
            "workshop_id": "ws",
            "workshop_title": "Workshop",
            "talks": [
                {
                    "talk_id": "talk_001",
                    "talk_title": "Opening",
                    "talk_speaker": "Dr One",
                    "timestamp_start": 0,
                    "timestamp_end": 300,
                    "description": "Opening talk",
                },
                {
                    "talk_id": "talk_002",
                    "talk_title": "Methods",
                    "talk_speaker": "Dr Two",
                    "timestamp_start": 300,
                    "timestamp_end": 620,
                    "description": "Methods talk",
                },
            ],
        }
    )

    class usage_metadata:
        prompt_token_count = 1200
        candidates_token_count = 200
        total_token_count = 1400


class _FakeGeminiModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeGeminiResponse()


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.models = _FakeGeminiModels()


def test_run_workshop_talk_boundary_discovery_writes_cost_receipts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transcript_json = workspace / "transcript.json"
    transcript_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 80, "text": "welcome to the workshop"},
                    {"start": 320, "end": 390, "text": "now the methods talk begins"},
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_talks = workspace / "workshop_talks" / "ws_talks.json"
    output_batches = workspace / "workshop_talks" / "ws_batches.json"
    annotated = workspace / "workshop_talks" / "ws_annotated.json"
    receipt_path = workspace / "workshop_talks" / "receipt.json"
    client = _FakeGeminiClient()

    receipt = run_workshop_talk_boundary_discovery(
        workspace=workspace,
        input_transcript_json=transcript_json,
        output_talks_json=output_talks,
        output_batches_json=output_batches,
        annotated_transcript_json=annotated,
        receipt_path=receipt_path,
        workshop_id="ws",
        workshop_title="Workshop",
        provider_cost_approved=True,
        run_id="unit-run",
        client=client,
    )
    talks_payload = json.loads(output_talks.read_text(encoding="utf-8"))
    batches_payload = json.loads(output_batches.read_text(encoding="utf-8"))
    annotated_payload = json.loads(annotated.read_text(encoding="utf-8"))

    assert receipt["receipt_type"] == WORKSHOP_TALK_DISCOVERY_RECEIPT_TYPE
    assert receipt["owner_id"] == WORKSHOP_TALK_DISCOVERY_OWNER_ID
    assert receipt["status"] == "succeeded"
    assert receipt["provider"] == "gemini"
    assert receipt["model"] == "gemini-2.5-pro"
    assert receipt["cost_task_id"] == WORKSHOP_TALK_DISCOVERY_COST_TASK_ID
    assert receipt["provider_calls_made"] is True
    assert receipt["estimated_cost_usd"] is not None
    assert receipt["provider_receipt_paths"]
    assert Path(receipt["provider_receipt_paths"][0]).exists()
    assert talks_payload["talks"][0]["talk_title"] == "Opening"
    assert batches_payload["batches"][0]["talk_ids"] == ["talk_001"]
    assert annotated_payload["segments"][0]["talk_id"] == "talk_001"
    assert client.models.calls[0]["model"] == "gemini-2.5-pro"


def test_workshop_talk_boundary_discovery_blocks_invalid_provider_boundaries(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transcript_json = workspace / "transcript.json"
    transcript_json.write_text(
        json.dumps({"segments": [{"start": 0, "end": 80, "text": "welcome"}]}) + "\n",
        encoding="utf-8",
    )
    output_talks = workspace / "workshop_talks" / "ws_talks.json"
    output_batches = workspace / "workshop_talks" / "ws_batches.json"
    annotated = workspace / "workshop_talks" / "ws_annotated.json"
    receipt_path = workspace / "workshop_talks" / "receipt.json"

    class InvalidBoundaryResponse:
        text = json.dumps(
            {
                "talks": [
                    {
                        "talk_id": "talk_001",
                        "talk_title": "Broken",
                        "timestamp_start": 100,
                    }
                ]
            }
        )

        class usage_metadata:
            prompt_token_count = 100
            candidates_token_count = 20
            total_token_count = 120

    class InvalidBoundaryModels:
        def generate_content(self, **_: object):
            return InvalidBoundaryResponse()

    class InvalidBoundaryClient:
        models = InvalidBoundaryModels()

    receipt = run_workshop_talk_boundary_discovery(
        workspace=workspace,
        input_transcript_json=transcript_json,
        output_talks_json=output_talks,
        output_batches_json=output_batches,
        annotated_transcript_json=annotated,
        receipt_path=receipt_path,
        workshop_id="ws",
        workshop_title="Workshop",
        provider_cost_approved=True,
        run_id="unit-run",
        client=InvalidBoundaryClient(),
    )

    assert receipt["status"] == "blocked"
    assert receipt["blocker_code"] == PROVIDER_DISCOVERY_FAILED_CODE
    assert receipt["error"] == "invalid_talk_boundaries:1"
    assert receipt["talk_count"] == 1
    assert not output_talks.exists()
    assert not output_batches.exists()
    assert not annotated.exists()


def test_workshop_talk_boundary_discovery_sanitizes_provider_exception(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transcript_json = workspace / "transcript.json"
    transcript_json.write_text(
        json.dumps({"segments": [{"start": 0, "end": 80, "text": "welcome"}]}) + "\n",
        encoding="utf-8",
    )

    class FailingModels:
        def generate_content(self, **_: object):
            raise RuntimeError(
                "provider failed Authorization: Bearer secret-provider-token "
                "url=https://provider.test/generate?token=secret-query-token"
            )

    class FailingClient:
        models = FailingModels()

    receipt = run_workshop_talk_boundary_discovery(
        workspace=workspace,
        input_transcript_json=transcript_json,
        output_talks_json=workspace / "talks.json",
        output_batches_json=workspace / "batches.json",
        receipt_path=workspace / "receipt.json",
        workshop_id="ws",
        workshop_title="Workshop",
        provider_cost_approved=True,
        run_id="unit-run",
        client=FailingClient(),
    )
    serialized = json.dumps(receipt)

    assert receipt["status"] == "blocked"
    assert receipt["blocker_code"] == PROVIDER_DISCOVERY_FAILED_CODE
    assert "secret-provider-token" not in serialized
    assert "secret-query-token" not in serialized
    assert "Bearer REDACTED" in receipt["error"]
    assert "token=REDACTED" in receipt["error"]


def test_workshop_talk_boundary_discovery_blocks_without_cost_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transcript_json = workspace / "transcript.json"
    transcript_json.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}), encoding="utf-8")

    receipt = run_workshop_talk_boundary_discovery(
        workspace=workspace,
        input_transcript_json=transcript_json,
        output_talks_json=workspace / "talks.json",
        output_batches_json=workspace / "batches.json",
        receipt_path=workspace / "receipt.json",
        workshop_id="ws",
        workshop_title="Workshop",
    )

    assert receipt["status"] == "blocked"
    assert receipt["blocker_code"] == PROVIDER_COST_NOT_APPROVED_CODE
    assert receipt["provider_calls_made"] is False
