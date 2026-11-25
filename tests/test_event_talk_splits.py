"""Event recordings -> named talk maps: selection, program hints, verifiable output."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.indexing import event_talk_splits as splits
from conference_pipeline.indexing import talk_maps
from conference_pipeline.workshop_talk_segmentation import build_workshop_talk_discovery_prompt


def _page(title: str, body: str = "") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "conf"
    probes = ws / "cache" / "probes"
    probes.mkdir(parents=True)
    (probes / "session.html").write_text(
        _page(
            "Track: Oral 1A Agents &amp; Tools",
            '<a href="/virtual/2031/oral/11">a</a><a href="/virtual/2031/oral/12">b</a>'
            '<a href="/virtual/2031/oral/11">dup</a><a href="/virtual/2031/session/5">self</a>',
        ),
        encoding="utf-8",
    )
    (probes / "oral11.html").write_text(_page("SITE Oral Planning with Tools"), encoding="utf-8")
    (probes / "oral12.html").write_text(_page("SITE Oral Agents &amp; Memory"), encoding="utf-8")
    (probes / "expo.html").write_text(_page("SITE Expo Workshop Agent Harnesses"), encoding="utf-8")
    (probes / "poster.html").write_text(_page("SITE Poster Something"), encoding="utf-8")
    assets = [
        {"item_id": "virtual-2031-session-5", "kind": None, "page_url": "https://site/virtual/2031/session/5",
         "page_artifact_path": "cache/probes/session.html", "slideslive_id": "900"},
        {"item_id": "oral-11", "kind": "oral", "page_url": "https://site/virtual/2031/oral/11",
         "page_artifact_path": "cache/probes/oral11.html", "slideslive_id": None},
        {"item_id": "oral-12", "kind": "oral", "page_url": "https://site/virtual/2031/oral/12",
         "page_artifact_path": "cache/probes/oral12.html", "slideslive_id": None},
        {"item_id": "virtual-2031-77", "kind": None, "page_url": "https://site/virtual/2031/77",
         "page_artifact_path": "cache/probes/expo.html", "slideslive_id": "901"},
        {"item_id": "poster-3", "kind": "poster", "page_url": "https://site/virtual/2031/poster/3",
         "page_artifact_path": "cache/probes/poster.html", "slideslive_id": "902"},
    ]
    recovery = ws / splits.RECOVERY_RELATIVE_PATH
    recovery.parent.mkdir(parents=True)
    recovery.write_text(json.dumps({"page_assets": assets}), encoding="utf-8")
    transcripts = ws / "storage" / "transcription-ocr" / "speech-transcripts"
    transcripts.mkdir(parents=True)
    records = []
    for video_id, text in (("900", "chair intro. planning with tools talk. agents and memory talk."), ("901", "expo"), ("902", "poster")):
        (transcripts / f"{video_id}.txt").write_text(text * 40, encoding="utf-8")
        (transcripts / f"{video_id}.json").write_text(
            json.dumps({
                "transcript_path": f"storage/transcription-ocr/speech-transcripts/{video_id}.txt",
                "duration_seconds": 900,
            }),
            encoding="utf-8",
        )
        records.append({
            "slideslive_id": video_id, "status": "succeeded",
            "transcript_json_path": f"storage/transcription-ocr/speech-transcripts/{video_id}.json",
            "transcript_path": f"storage/transcription-ocr/speech-transcripts/{video_id}.txt",
        })
    inventory = ws / splits.TRANSCRIPTION_INVENTORY_RELATIVE_PATH
    inventory.write_text(json.dumps({"lanes": {splits.SPEECH_LANE: {"records": records}}}), encoding="utf-8")
    return ws


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    class usage_metadata:
        prompt_token_count = 1000
        candidates_token_count = 100
        total_token_count = 1100


class _Models:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_content(self, **kwargs):
        self.prompts.append(str(kwargs.get("contents")))
        return _Response(json.dumps({"talks": [
            {"talk_id": "talk_001", "talk_title": "Session opening", "talk_speaker": "Unknown",
             "timestamp_start": 0, "timestamp_end": 120},
            {"talk_id": "talk_002", "talk_title": "Planning with Tools", "talk_speaker": "A",
             "timestamp_start": 120, "timestamp_end": 540},
            {"talk_id": "talk_003", "talk_title": "Agents & Memory", "talk_speaker": "B",
             "timestamp_start": 540, "timestamp_end": 900},
        ]}))


class _Client:
    def __init__(self) -> None:
        self.models = _Models()


def test_selection_by_kind_reads_the_program_in_page_order(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    recordings, skipped = splits.load_event_recordings(ws, kinds=["oral"], item_ids=["virtual-2031-77"])
    assert skipped == []
    by_id = {r.item_id: r for r in recordings}
    assert set(by_id) == {"virtual-2031-session-5", "virtual-2031-77"}  # the poster is not selected
    session = by_id["virtual-2031-session-5"]
    assert session.kind == "oral"
    assert session.title == "Oral 1A Agents & Tools"
    assert [t["title"] for t in session.scheduled_talks] == ["Planning with Tools", "Agents & Memory"]
    assert by_id["virtual-2031-77"].title == "Expo Workshop Agent Harnesses"
    assert by_id["virtual-2031-77"].scheduled_talks == ()


def test_unknown_item_ids_are_reported(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _recordings, skipped = splits.load_event_recordings(ws, item_ids=["nope"])
    assert skipped == [{"item_id": "nope", "reason": "item_not_found_or_no_recording"}]


def test_prompt_carries_the_program_only_when_given() -> None:
    segments = [{"start": 0, "end": 10, "text": "hello"}]
    plain = build_workshop_talk_discovery_prompt(workshop_id="w", workshop_title="W", segments=segments)
    assert "PROGRAM" not in plain
    with_program = build_workshop_talk_discovery_prompt(
        workshop_id="w", workshop_title="W", segments=segments,
        scheduled_talks=[{"title": "Planning with Tools"}, {"title": "Agents & Memory", "speakers": "B"}],
    )
    assert "1. Planning with Tools" in with_program
    assert "2. Agents & Memory -- B" in with_program
    assert "EXACTLY" in with_program


def test_split_writes_a_talk_map_the_indexer_verifies(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (recording,) = [r for r in splits.load_event_recordings(ws, kinds=["oral"])[0]]
    client = _Client()
    summary = splits.split_recording(ws, recording, provider_cost_approved=True, run_id="t", client=client)
    assert summary["status"] == "succeeded", summary
    assert summary["talk_count"] == 3
    assert summary["scheduled_titles_matched"] == 2
    assert "1. Planning with Tools" in client.models.prompts[0]
    files = talk_maps.talk_map_files(ws, talk_maps.BASE_SCOPE, "900")
    talks = json.loads(files["talks"].read_text(encoding="utf-8"))
    assert talks["video_id"] == "900"
    assert talks["discovery_method"] == splits.DISCOVERY_METHOD_WITH_PROGRAM
    assert talks["program_hints"]["event_kind"] == "oral"
    binding, code = talk_maps.verify_talk_map_chain(
        ws, talks_path=files["talks"], annotated_path=files["annotated"], video_id="900",
        transcript_sha256=recording.transcript_sha256,
    )
    assert code is None, code
    annotated = json.loads(files["annotated"].read_text(encoding="utf-8"))
    # One-minute segments: a 900 s session is 15 segments, each attributed to one talk.
    assert len(annotated["segments"]) == 15
    by_title = [s.get("talk_title") for s in annotated["segments"]]
    assert by_title.count("Session opening") == 2
    assert by_title.count("Planning with Tools") == 7
    assert by_title.count("Agents & Memory") == 6
    # An oral session is not a workshop: no workshop identity is claimed.
    assert talk_maps.authoritative_workshop(talks) == {}
    assert splits.talk_map_is_current(ws, recording)
    # A changed transcript makes the map stale, never silently attached.
    recording.transcript_path.write_text("different words", encoding="utf-8")
    _binding, code = talk_maps.verify_talk_map_chain(
        ws, talks_path=files["talks"], annotated_path=files["annotated"], video_id="900",
        transcript_sha256=splits.sha256_file(recording.transcript_path),
    )
    assert code == "talk_map_stale_for_transcript"


def test_split_without_approval_calls_nothing(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (recording,) = splits.load_event_recordings(ws, kinds=["oral"])[0]
    client = _Client()
    summary = splits.split_recording(ws, recording, provider_cost_approved=False, client=client)
    assert summary["status"] == "blocked"
    assert client.models.prompts == []


def test_cli_estimate_writes_an_estimate_and_calls_nothing(tmp_path: Path) -> None:
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "split_event_recordings.py"
    spec = importlib.util.spec_from_file_location("split_event_recordings", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ws = _workspace(tmp_path)
    client = _Client()
    assert module.main(["--workspace", str(ws), "--kind", "oral", "--estimate"], client=client) == 0
    estimate = json.loads((talk_maps.talk_map_dir(ws, "base").parent / module.RECEIPTS_DIR_NAME / module.ESTIMATE_NAME).read_text(encoding="utf-8"))
    assert estimate["recording_count"] == 1
    assert estimate["recordings"][0]["scheduled_talk_count"] == 2
    assert client.models.prompts == []
    assert module.main(["--workspace", str(ws), "--kind", "oral", "--provider-cost-approved"], client=client) == 0
    batch = json.loads((talk_maps.talk_map_dir(ws, "base").parent / module.RECEIPTS_DIR_NAME / module.BATCH_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert batch["succeeded_count"] == 1 and batch["talk_count"] == 3
    # --resume skips a recording whose map is still current.
    assert module.main(["--workspace", str(ws), "--kind", "oral", "--provider-cost-approved", "--resume"], client=client) == 0
    assert len(client.models.prompts) == 1


def _load_icml_runner():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_icml2026_production.py"
    spec = importlib.util.spec_from_file_location("icml_runner_for_talk_maps", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_icml_indexer_attaches_a_verified_map_and_blocks_a_stale_one(tmp_path: Path) -> None:
    runner = _load_icml_runner()
    ws = _workspace(tmp_path).resolve()
    (recording,) = splits.load_event_recordings(ws, kinds=["oral"])[0]
    assert runner.event_talk_map_binding(ws, "900", recording.transcript_path) == (None, None)
    splits.split_recording(ws, recording, provider_cost_approved=True, client=_Client())
    binding, code = runner.event_talk_map_binding(ws, "900", recording.transcript_path)
    assert code is None
    assert binding["annotated_transcript_json_path"].endswith("900_annotated.json")
    assert binding["talk_summary"]["talk_count"] == 3
    recording.transcript_path.write_text("re-transcribed", encoding="utf-8")
    assert runner.event_talk_map_binding(ws, "900", recording.transcript_path) == (None, "talk_map_stale_for_transcript")


class _LazyModels(_Models):
    def generate_content(self, **kwargs):
        self.prompts.append(str(kwargs.get("contents")))
        return _Response(json.dumps({"talks": [
            {"talk_id": "talk_001", "talk_title": "Session opening", "talk_speaker": "Unknown",
             "timestamp_start": 0, "timestamp_end": 60},
        ]}))


def test_a_split_that_stops_early_is_set_aside_and_retried(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (recording,) = splits.load_event_recordings(ws, kinds=["oral"])[0]
    lazy = _Client()
    lazy.models = _LazyModels()
    summary = splits.split_recording(ws, recording, provider_cost_approved=True, client=lazy)
    assert summary["status"] == "incomplete"
    assert summary["coverage"] < splits.MIN_TALK_COVERAGE
    files = talk_maps.talk_map_files(ws, talk_maps.BASE_SCOPE, "900")
    assert not files["talks"].exists()  # Stage 6 cannot attach it
    assert files["talks"].with_name(files["talks"].name + splits.INCOMPLETE_SUFFIX).exists()  # kept
    assert not splits.talk_map_is_current(ws, recording)  # --resume retries it
    summary = splits.split_recording(ws, recording, provider_cost_approved=True, client=_Client())
    assert summary["status"] == "succeeded" and summary["coverage"] == 1.0


def test_a_run_id_writes_the_lineage_substep_receipt(tmp_path: Path) -> None:
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "split_event_recordings.py"
    spec = importlib.util.spec_from_file_location("split_event_recordings_substep", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ws = _workspace(tmp_path)
    assert module.main(
        ["--workspace", str(ws), "--kind", "oral", "--provider-cost-approved", "--run-id", "run-1"], client=_Client()
    ) == 0
    receipt_path = ws / "storage" / "workflow-runs" / "run-1" / splits.SUBSTEP_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"
    # Stamped: either bound lineage or an explicit unbound reason, never silent.
    assert "lineage" in receipt or "lineage_unbound_reason" in receipt or "lineage_error" in receipt
    assert receipt["talk_maps_path"] == "storage/transcription-ocr/talk-maps/base"


def test_resume_restamps_current_maps_without_approval_and_refuses_real_splits(tmp_path: Path) -> None:
    """Re-stamping current talk maps is free; a recording that needs a split still needs approval."""
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "split_event_recordings.py"
    spec = importlib.util.spec_from_file_location("split_event_recordings_restamp", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ws = _workspace(tmp_path)
    client = _Client()

    # Nothing split yet: --resume alone must refuse before any provider call.
    assert module.main(["--workspace", str(ws), "--kind", "oral", "--resume", "--run-id", "run-1"], client=client) == 1
    assert client.models.prompts == []

    assert module.main(["--workspace", str(ws), "--kind", "oral", "--provider-cost-approved"], client=client) == 0
    calls = len(client.models.prompts)
    receipt_path = ws / "storage" / "workflow-runs" / "run-1" / splits.SUBSTEP_RECEIPT_NAME
    assert not receipt_path.exists()

    # Every map current: --resume alone re-stamps the batch and sub-step receipts, no call.
    assert module.main(["--workspace", str(ws), "--kind", "oral", "--resume", "--run-id", "run-1"], client=client) == 0
    assert len(client.models.prompts) == calls
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded" and receipt["succeeded_count"] == 1
    assert receipt["last_run"] == {"recording_count": 0, "succeeded_count": 0, "estimated_cost_usd": 0}


def test_talk_splits_depend_on_speech_transcripts_not_the_whole_transcription_stage() -> None:
    from conference_pipeline.stage_script_registry import load_stage_script_registry

    registry = load_stage_script_registry()
    substep = registry["lineage_substeps"]["event-talk-splits"] if isinstance(registry, dict) else None
    if substep is None:
        substep = json.loads(
            (Path(__file__).resolve().parents[1] / "src/conference_pipeline/stage_owner_registry.json").read_text()
        )["lineage_substeps"]["event-talk-splits"]
    assert substep["required_upstream_stages"] == []
    assert {"name": "speech_transcripts", "path": "storage/transcription-ocr/speech-transcripts"} in substep["input_artifacts"]
