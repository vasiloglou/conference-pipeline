"""Tests for the conference-general YouTube talk harvest lane."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.acquisition.youtube_talks import (
    HarvestConfig,
    HarvestedVideo,
    ProgramItem,
    download_captions,
    extract_video_id,
    harvest_youtube_talks,
    load_program_spine,
    match_videos_to_spine,
    normalize_title,
    parse_caption_file,
    score_video_for_item,
    tavily_search_videos,
    title_similarity,
    url_host_allowed,
)

YT_DLP_CMD = ["yt-dlp-test"]
REPO_ROOT = Path(__file__).resolve().parents[1]
HARVEST_SCRIPT = REPO_ROOT / "scripts" / "harvest_youtube_talks.py"


def load_harvest_script() -> object:
    spec = importlib.util.spec_from_file_location("harvest_youtube_talks_under_test", HARVEST_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


class FakeRunner:
    """Scripted yt-dlp runner keyed by substring of the command line."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: list[tuple[str, subprocess.CompletedProcess]] = []
        self.on_caption_call = None

    def add(self, needle: str, result: subprocess.CompletedProcess) -> None:
        self.responses.append((needle, result))

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(command)
        joined = " ".join(command)
        if ("--write-subs" in command or "--write-auto-subs" in command) and self.on_caption_call:
            self.on_caption_call(command)
        for needle, result in self.responses:
            if needle in joined:
                return result
        return completed("")


def test_caption_download_does_not_reuse_a_longer_video_ids_sidecar(tmp_path: Path) -> None:
    """A filename prefix is not provenance for a different YouTube video."""

    video_id = "ABCDEF12345"
    destination = tmp_path / "captions"
    destination.mkdir()
    wrong_sidecar = destination / f"{video_id}x.en.vtt"
    wrong_sidecar.write_text("WEBVTT\n", encoding="utf-8")
    runner = FakeRunner()

    result = download_captions(
        HarvestedVideo(
            video_id=video_id,
            title="",
            url=f"https://www.youtube.com/watch?v={video_id}",
            source="fixture",
        ),
        destination,
        runner=runner,
        yt_dlp_cmd=YT_DLP_CMD,
    )

    assert result is None
    assert len(runner.calls) == 2
    assert {"--write-subs", "--write-auto-subs"} == {
        next(flag for flag in ("--write-subs", "--write-auto-subs") if flag in command)
        for command in runner.calls
    }


def flat_json(entries: list[dict]) -> str:
    return json.dumps({"entries": entries})


# ---------------------------------------------------------------------------
# Normalization + matching
# ---------------------------------------------------------------------------


def test_normalize_title_strips_punctuation_and_stop_tokens() -> None:
    assert (
        normalize_title("[CVPR 2026 Oral] SceneDiff: Diffusion for 3D Scenes!",
                        stop_tokens=("cvpr", "2026"))
        == "oral scenediff diffusion for 3d scenes"
    )


def test_title_similarity_exact_and_prefixed_upload() -> None:
    paper = "SceneDiff: Diffusion Models for 3D Scene Synthesis"
    upload = "[CVPR 2026] SceneDiff: Diffusion Models for 3D Scene Synthesis"
    assert title_similarity(paper, upload, stop_tokens=("cvpr", "2026")) > 0.95
    assert title_similarity(paper, "Cooking pasta at home") < 0.2


def test_score_video_speaker_boost() -> None:
    item = ProgramItem(
        item_id="k1",
        title="The Future of Visual Intelligence",
        speakers=("Jane Nakamura",),
        family="keynote",
    )
    video = HarvestedVideo(
        video_id="abc123def45",
        title="Jane Nakamura - The Future of Visual Intelligence",
        url="https://www.youtube.com/watch?v=abc123def45",
    )
    score, method = score_video_for_item(item, video)
    assert score == 1.0 or score > 0.9
    assert method in {"title_exact", "title_speaker"}


def test_match_videos_to_spine_greedy_unique() -> None:
    spine = [
        ProgramItem(item_id="a", title="Neural Radiance Caching", family="oral"),
        ProgramItem(item_id="b", title="Robust Depth Estimation", family="poster"),
    ]
    videos = [
        HarvestedVideo(
            video_id="vid00000001",
            title="Neural Radiance Caching (oral talk)",
            url="https://www.youtube.com/watch?v=vid00000001",
        ),
        HarvestedVideo(
            video_id="vid00000002",
            title="Totally unrelated vlog",
            url="https://www.youtube.com/watch?v=vid00000002",
        ),
    ]
    matches = match_videos_to_spine(spine, videos, threshold=0.6)
    assert len(matches) == 1
    assert matches[0].item.item_id == "a"
    assert matches[0].video.video_id == "vid00000001"
    assert matches[0].direction == "harvest_to_program"


def test_priority_families_flagged() -> None:
    keynote = ProgramItem(item_id="k", title="t", family="Keynote")
    oral = ProgramItem(item_id="o", title="t", family="oral")
    poster = ProgramItem(item_id="p", title="t", family="poster")
    assert keynote.is_priority_family and keynote.priority == 0
    assert oral.is_priority_family and oral.priority == 1
    assert not poster.is_priority_family and poster.priority == 2


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://vimeo.com/12345", None),
        ("not a url", None),
    ],
)
def test_extract_video_id(url: str, expected: str | None) -> None:
    assert extract_video_id(url) == expected


def test_url_host_allowed_rejects_other_hosts() -> None:
    allowed = ("youtube.com", "youtu.be")
    assert url_host_allowed("https://www.youtube.com/watch?v=x1234567", allowed)
    assert url_host_allowed("https://youtu.be/x1234567", allowed)
    assert not url_host_allowed("https://vimeo.com/999", allowed)
    assert not url_host_allowed("https://slideslive.com/39012345", allowed)


def test_public_only_cli_contract_rejects_tavily_caps_and_host_widening() -> None:
    script = load_harvest_script()
    parser = script.build_parser()
    args = parser.parse_args(
        [
            "--workspace", "fixture-workspace",
            "--run-id", "fixture-run",
            "--conference-label", "CVPR 2026",
            "--program-spine", "fixture-spine.json",
            "--public-only",
            "--tavily",
            "--pilot-limit", "1",
            "--allowed-host", "youtube.com",
            "--allowed-host", "vimeo.com",
        ]
    )

    assert script.public_only_argument_errors(args) == [
        "--public-only requires allowed hosts exactly youtube.com,youtu.be",
        "--public-only rejects --pilot-limit",
        "--public-only rejects --tavily; use the governed paid-search lane",
    ]


# ---------------------------------------------------------------------------
# Caption parsing
# ---------------------------------------------------------------------------


def test_parse_caption_file_dedupes_rolling_auto_captions(tmp_path: Path) -> None:
    vtt = tmp_path / "vid.en.vtt"
    vtt.write_text(
        "WEBVTT\n"
        "Kind: captions\n"
        "Language: en\n"
        "\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "<c>hello everyone</c>\n"
        "\n"
        "00:00:02.000 --> 00:00:04.000\n"
        "hello everyone\n"
        "\n"
        "00:00:04.000 --> 00:00:06.500\n"
        "welcome to the talk\n",
        encoding="utf-8",
    )
    segments = parse_caption_file(vtt)
    assert [segment["text"] for segment in segments] == [
        "hello everyone",
        "welcome to the talk",
    ]
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 4.0  # merged across the duplicate
    assert segments[1]["end"] == 6.5


# ---------------------------------------------------------------------------
# Tavily lane
# ---------------------------------------------------------------------------


def test_tavily_search_filters_to_allowed_hosts() -> None:
    def fake_post(url: str, payload: dict) -> dict:
        assert payload["include_domains"] == ["youtube.com", "youtu.be"]
        return {
            "results": [
                {"url": "https://www.youtube.com/watch?v=good1234567", "title": "talk"},
                {"url": "https://vimeo.com/123", "title": "nope"},
            ]
        }

    videos = tavily_search_videos(
        "some talk", api_key="k", http_post=fake_post,
        allowed_hosts=("youtube.com", "youtu.be"),
    )
    assert [video.video_id for video in videos] == ["good1234567"]
    assert videos[0].source == "tavily_search"


def test_cli_rejects_bare_tavily_dispatch_without_a_runner_authority(tmp_path: Path) -> None:
    """The generic CLI must not turn TAVILY_API_KEY into a paid-call switch."""

    script = load_harvest_script()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spine = tmp_path / "spine.json"
    spine.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit):
        script.main(
            [
                "--workspace", str(workspace), "--run-id", "run-1",
                "--conference-label", "CVPR 2026", "--program-spine", str(spine),
                "--tavily",
            ]
        )


# ---------------------------------------------------------------------------
# Spine loading
# ---------------------------------------------------------------------------


def test_load_program_spine_list_and_items(tmp_path: Path) -> None:
    path = tmp_path / "spine.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {"title": "A talk", "family": "keynote", "speakers": ["X Y"]},
                    {"id": "or1", "title": "B talk", "family": "oral"},
                ]
            }
        ),
        encoding="utf-8",
    )
    spine = load_program_spine(path)
    assert len(spine) == 2
    assert spine[0].family == "keynote"
    assert spine[1].item_id == "or1"


# ---------------------------------------------------------------------------
# End-to-end orchestration (mocked yt-dlp)
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, spine: list[ProgramItem], **overrides) -> HarvestConfig:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    defaults = dict(
        workspace=workspace,
        conference_label="ExampleConf 2099",
        program_spine=spine,
        rate_limit_seconds=0.0,
        run_id="test-run",
    )
    defaults.update(overrides)
    return HarvestConfig(**defaults)


def test_harvest_direction_a_and_b_with_priority_and_gaps(tmp_path: Path) -> None:
    spine = [
        ProgramItem(
            item_id="key1",
            title="Vision Beyond Pixels",
            speakers=("Ada Example",),
            family="keynote",
        ),
        ProgramItem(item_id="oral1", title="Fast Volumetric Rendering", family="oral"),
        ProgramItem(item_id="pos1", title="A Poster Nobody Uploaded", family="poster"),
    ]
    runner = FakeRunner()
    # Direction A: channel enumeration returns the keynote video.
    runner.add(
        "--flat-playlist -J https://www.youtube.com/@conf",
        completed(
            flat_json(
                [
                    {
                        "id": "keyvid00001",
                        "title": "Vision Beyond Pixels | ExampleConf 2099 Keynote",
                        "url": "https://www.youtube.com/watch?v=keyvid00001",
                        "channel": "ExampleConf",
                        "duration": 3000,
                    },
                    {
                        "id": "noisevid001",
                        "title": "Conference venue tour vlog",
                        "url": "https://www.youtube.com/watch?v=noisevid001",
                    },
                ]
            )
        ),
    )
    # Direction B: per-item search finds the oral; nothing for the poster.
    runner.add(
        'ytsearch5:"Fast Volumetric Rendering"',
        completed(
            flat_json(
                [
                    {
                        "id": "oralvid0001",
                        "title": "Fast Volumetric Rendering (ExampleConf 2099 oral)",
                        "url": "https://www.youtube.com/watch?v=oralvid0001",
                    }
                ]
            )
        ),
    )

    # Captions: keynote has manual subs; oral only auto subs.
    def write_caption(command: list[str]) -> None:
        output_template = command[command.index("-o") + 1]
        video_url = command[-1]
        dest = Path(output_template.replace("%(ext)s", "en.vtt"))
        if "keyvid00001" in video_url and "--write-subs" in command:
            dest.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello keynote\n",
                encoding="utf-8",
            )
        if "oralvid0001" in video_url and "--write-auto-subs" in command:
            dest.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello oral\n",
                encoding="utf-8",
            )

    runner.on_caption_call = write_caption

    config = make_config(
        tmp_path,
        spine,
        sources=["https://www.youtube.com/@conf"],
    )
    receipt = harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)

    inventory = json.loads(
        (config.workspace / "storage" / "youtube-talks" / "talk_inventory.json").read_text()
    )
    by_item = {talk["item_id"]: talk for talk in inventory["talks"]}

    assert by_item["key1"]["match_direction"] == "harvest_to_program"
    assert by_item["key1"]["transcript"]["transcript_source"] == "youtube_caption_manual"
    assert by_item["oral1"]["match_direction"] == "program_to_search"
    assert by_item["oral1"]["transcript"]["transcript_source"] == "youtube_caption_auto"

    # Poster remains a gap but is NOT must_investigate; no priority gap.
    assert inventory["gaps"] == [
        {
            "item_id": "pos1",
            "title": "A Poster Nobody Uploaded",
            "family": "poster",
            "priority": 2,
            "must_investigate": False,
        }
    ]
    assert receipt["counts"]["priority_gaps"] == 0
    assert receipt["status"] == "succeeded"
    assert receipt["allowed_hosts"] == ["youtube.com", "youtu.be"]
    assert receipt["boundaries"] == {
        "audio_downloaded": False,
        "video_downloaded": False,
        "paywalled_sources_touched": False,
        "transcript_first": True,
    }

    # Direction B searched the oral (priority 1) but never the noise video.
    search_calls = [
        call for call in runner.calls if any("ytsearch" in part for part in call)
    ]
    assert len(search_calls) >= 1


def test_harvest_priority_gap_marks_partial_and_must_investigate(tmp_path: Path) -> None:
    spine = [
        ProgramItem(item_id="key1", title="An Unfindable Keynote", family="keynote"),
    ]
    runner = FakeRunner()  # every lookup returns nothing
    config = make_config(tmp_path, spine, transcripts_enabled=False)
    receipt = harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert receipt["status"] == "partial"
    assert receipt["counts"]["priority_gaps"] == 1
    inventory = json.loads(
        (config.workspace / "storage" / "youtube-talks" / "talk_inventory.json").read_text()
    )
    assert inventory["gaps"][0]["must_investigate"] is True


def test_harvest_direction_b_priority_order(tmp_path: Path) -> None:
    spine = [
        ProgramItem(item_id="pos1", title="Zeta Poster Work", family="poster"),
        ProgramItem(item_id="key1", title="Alpha Keynote Talk", family="keynote"),
        ProgramItem(item_id="oral1", title="Beta Oral Result", family="oral"),
    ]
    runner = FakeRunner()
    config = make_config(tmp_path, spine, transcripts_enabled=False)
    harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    queries = [
        " ".join(call) for call in runner.calls if any("ytsearch" in part for part in call)
    ]
    assert "Alpha Keynote Talk" in queries[0]
    assert "Beta Oral Result" in queries[1]
    assert "Zeta Poster Work" in queries[2]


def test_harvest_tavily_fallback_used_when_ytsearch_empty(tmp_path: Path) -> None:
    spine = [ProgramItem(item_id="key1", title="Hidden Gem Keynote", family="keynote")]
    runner = FakeRunner()

    def fake_post(url: str, payload: dict) -> dict:
        return {
            "results": [
                {
                    "url": "https://www.youtube.com/watch?v=tavilyvid01",
                    "title": "Hidden Gem Keynote (recording)",
                }
            ]
        }

    config = make_config(
        tmp_path,
        spine,
        tavily_api_key="secret-key",
        tavily_authority={
            "authority_kind": "cvpr2026_tavily_priority_gap_bound",
            "authority_digest": "a" * 64,
            "priority_program_item_ids": ["key1"],
        },
        transcripts_enabled=False,
    )
    receipt = harvest_youtube_talks(
        config, runner=runner, http_post=fake_post, yt_dlp_cmd=YT_DLP_CMD
    )
    assert receipt["counts"]["direction_b_matches"] == 1
    assert receipt["tavily_lane"] == {
        "enabled": True,
        "queries": 1,
        "hits": 1,
        "status": "authorized",
        "skipped_reason": None,
    }
    # The key must never leak into the receipt file.
    receipt_text = Path(receipt["receipt_path"]).read_text(encoding="utf-8")
    assert "secret-key" not in receipt_text


def test_harvest_tavily_disabled_records_skip(tmp_path: Path) -> None:
    spine = [ProgramItem(item_id="key1", title="Some Keynote", family="keynote")]
    runner = FakeRunner()
    config = make_config(tmp_path, spine, transcripts_enabled=False)
    receipt = harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert receipt["tavily_lane"]["enabled"] is False
    assert receipt["tavily_lane"]["skipped_reason"] == "TAVILY_API_KEY not set"


def test_harvest_caption_failure_marks_asr_fallback(tmp_path: Path) -> None:
    spine = [ProgramItem(item_id="key1", title="Captionless Keynote", family="keynote")]
    runner = FakeRunner()
    runner.add(
        'ytsearch5:"Captionless Keynote"',
        completed(
            flat_json(
                [
                    {
                        "id": "nocapvid001",
                        "title": "Captionless Keynote",
                        "url": "https://www.youtube.com/watch?v=nocapvid001",
                    }
                ]
            )
        ),
    )
    config = make_config(tmp_path, spine)
    receipt = harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert receipt["status"] == "partial"
    assert receipt["counts"]["caption_failures"] == 1
    inventory = json.loads(
        (config.workspace / "storage" / "youtube-talks" / "talk_inventory.json").read_text()
    )
    assert (
        inventory["talks"][0]["transcript"]["transcript_source"]
        == "asr_fallback_required"
    )
    assert receipt["caption_failures"][0]["next_action"].startswith(
        "route to transcription.session_speech"
    )


def test_harvest_resume_skips_matched_items(tmp_path: Path) -> None:
    spine = [ProgramItem(item_id="key1", title="Resumable Keynote", family="keynote")]
    runner = FakeRunner()
    runner.add(
        'ytsearch5:"Resumable Keynote"',
        completed(
            flat_json(
                [
                    {
                        "id": "resumevid01",
                        "title": "Resumable Keynote",
                        "url": "https://www.youtube.com/watch?v=resumevid01",
                    }
                ]
            )
        ),
    )
    config = make_config(tmp_path, spine, transcripts_enabled=False)
    first = harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert first["counts"]["matched_talks"] == 1
    first_call_count = len(runner.calls)

    resume_config = make_config(
        tmp_path, spine, transcripts_enabled=False, resume=True
    )
    second = harvest_youtube_talks(resume_config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert second["counts"]["resumed_talks"] == 1
    assert second["counts"]["matched_talks"] == 1
    assert second["counts"]["gaps"] == 0
    # No per-item search should re-run for the already matched item.
    search_calls_second = [
        call
        for call in runner.calls[first_call_count:]
        if any("ytsearch" in part for part in call)
    ]
    assert search_calls_second == []


def test_harvest_pilot_limit_recorded_and_priority_kept(tmp_path: Path) -> None:
    spine = [
        ProgramItem(item_id="pos1", title="P1", family="poster"),
        ProgramItem(item_id="key1", title="K1", family="keynote"),
    ]
    runner = FakeRunner()
    config = make_config(
        tmp_path, spine, transcripts_enabled=False, pilot_limit=1,
        per_item_search=False,
    )
    receipt = harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert receipt["pilot_limit"] == 1
    assert "must not use item-count caps" in receipt["pilot_limit_note"]
    assert receipt["counts"]["program_items"] == 1
    inventory = json.loads(
        (config.workspace / "storage" / "youtube-talks" / "talk_inventory.json").read_text()
    )
    # The keynote (priority 0) is kept, not the poster.
    assert inventory["gaps"][0]["item_id"] == "key1"


def test_harvest_filters_non_allowlisted_hosts(tmp_path: Path) -> None:
    spine = [ProgramItem(item_id="key1", title="Host Filter Keynote", family="keynote")]
    runner = FakeRunner()
    runner.add(
        'ytsearch5:"Host Filter Keynote"',
        completed(
            flat_json(
                [
                    {
                        "id": "vimeoescape",
                        "title": "Host Filter Keynote",
                        "url": "https://vimeo.com/987654",
                    }
                ]
            )
        ),
    )
    config = make_config(tmp_path, spine, transcripts_enabled=False)
    receipt = harvest_youtube_talks(config, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert receipt["counts"]["matched_talks"] == 0
    assert receipt["counts"]["priority_gaps"] == 1
