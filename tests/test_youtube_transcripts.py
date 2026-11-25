"""Tests for the tiered YouTube transcript lane (captions -> Gemini)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conference_pipeline.acquisition.youtube_talks import (
    HarvestConfig,
    HarvestedVideo,
    ProgramItem,
    YouTubeBotCheckBlocked,
    download_captions,
    harvest_youtube_talks,
    is_youtube_bot_check,
)
from conference_pipeline.acquisition.youtube_transcripts import (
    GEMINI_TRANSCRIPT_SOURCE,
    NO_SPEECH_MARKER,
    GeminiBudgetExhausted,
    GeminiTranscriber,
    GeminiWindowResult,
    VideoListConfig,
    YouTubeLaneState,
    acquire_transcript,
    fetch_watch_page_metadata,
    plan_windows,
    transcribe_video_list,
)

YT_DLP_CMD = ["yt-dlp-test"]
BOT_STDERR = (
    "ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies-from-browser"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
HARVEST_SCRIPT = REPO_ROOT / "scripts" / "harvest_youtube_talks.py"

VTT = (
    "WEBVTT\n\n"
    "00:00:00.000 --> 00:00:02.000\nhello everyone\n\n"
    "00:00:02.000 --> 00:00:04.000\nwelcome to the tutorial\n"
)


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class ScriptedRunner:
    """yt-dlp stand-in: per-video behaviour 'caption' | 'none' | 'bot'."""

    def __init__(self, behaviour: dict[str, str]) -> None:
        self.behaviour = behaviour
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(command)
        url = command[-1]
        video_id = url.rsplit("=", 1)[-1]
        mode = self.behaviour.get(video_id, "none")
        if mode == "bot":
            return completed(stderr=BOT_STDERR, returncode=1)
        if mode == "caption" and "--write-auto-subs" in command:
            template = command[command.index("-o") + 1]
            Path(template.replace("%(ext)s", "en.vtt")).write_text(VTT, encoding="utf-8")
        return completed()


class FakeWindows:
    """Gemini window stand-in that records calls and returns scripted text."""

    def __init__(
        self,
        texts=None,
        truncate_over: int | None = None,
        fail_times: int = 0,
        sparse_over: int | None = None,
    ) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.texts = texts or {}
        self.truncate_over = truncate_over
        self.fail_times = fail_times
        self.sparse_over = sparse_over

    def __call__(self, url: str, start: int | None, end: int | None) -> GeminiWindowResult:
        self.calls.append((url, start, end))
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("500 INTERNAL")
        if start is None or end is None:  # whole-video call
            span = 300
            return GeminiWindowResult(
                text=self.texts.get("whole", "whole video speech"),
                input_tokens=span * 27,
                output_tokens=span * 3,
                truncated=bool(self.truncate_over == "whole"),
                estimated_cost_usd=None,
            )
        truncated = self.truncate_over is not None and self.truncate_over != "whole" and (end - start) > self.truncate_over
        sparse = self.sparse_over is not None and (end - start) > self.sparse_over
        return GeminiWindowResult(
            text=self.texts.get((start, end), f"speech {start}-{end}"),
            input_tokens=(end - start) * 27,
            output_tokens=int((end - start) * (0.5 if sparse else 3)),
            truncated=truncated,
            estimated_cost_usd=None,
        )


def price(input_tokens: int, output_tokens: int) -> float:
    # Flash-lite-like rates: $0.25 / $1.50 per 1M tokens.
    return input_tokens * 0.25 / 1e6 + output_tokens * 1.5 / 1e6


def make_gemini(windows: FakeWindows, *, budget: float = 5.0, window_seconds: int = 1800) -> GeminiTranscriber:
    return GeminiTranscriber(
        model="gemini-test-model",  # model-literal-ok: test fixture
        budget_usd=budget,
        window_transcriber=windows,
        window_seconds=window_seconds,
        sleeper=lambda _s: None,
        price_estimator=price,
    )


def video(video_id: str, duration: float | None = None) -> HarvestedVideo:
    return HarvestedVideo(
        video_id=video_id,
        title=f"Talk {video_id}",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration_seconds=duration,
    )


def no_http(url: str) -> str:
    raise AssertionError(f"unexpected watch-page fetch {url}")


# ---------------------------------------------------------------------------
# Tier 1: bot-check detection
# ---------------------------------------------------------------------------


def test_prompt_asks_for_speech_only_never_on_screen_text() -> None:
    """Teaser videos returned slide text as if it were a transcript."""

    from conference_pipeline.acquisition.youtube_transcripts import GEMINI_PROMPT

    assert "Never read out on-screen text" in GEMINI_PROMPT
    assert NO_SPEECH_MARKER in GEMINI_PROMPT


def test_bot_check_marker_matches_both_apostrophes() -> None:
    assert is_youtube_bot_check(BOT_STDERR)
    assert is_youtube_bot_check("Sign in to confirm you're not a bot")
    assert not is_youtube_bot_check("ERROR: This video is not available")
    assert not is_youtube_bot_check(None)


def test_download_captions_raises_on_bot_gate_instead_of_reporting_no_caption(tmp_path: Path) -> None:
    runner = ScriptedRunner({"blockedvid01": "bot"})
    with pytest.raises(YouTubeBotCheckBlocked):
        download_captions(video("blockedvid01"), tmp_path, runner=runner, yt_dlp_cmd=YT_DLP_CMD)
    assert len(runner.calls) == 1  # stops at the first gated request


# ---------------------------------------------------------------------------
# Gemini windowing, budget, retries
# ---------------------------------------------------------------------------


def test_plan_windows_covers_duration_contiguously() -> None:
    # The last window stops 2 s short of the listed duration.
    assert plan_windows(4000, 1800) == [(0, 1800), (1800, 3600), (3600, 3998)]
    assert plan_windows(600, 1800) == [(0, 598)]
    with pytest.raises(ValueError):
        plan_windows(0, 1800)


def test_plan_windows_folds_a_tiny_tail_into_the_previous_window() -> None:
    # A listed 1811 s video once produced an 11 s tail past the real end (HTTP 500).
    assert plan_windows(1811, 1800) == [(0, 1809)]
    assert plan_windows(1850, 900) == [(0, 900), (900, 1848)]  # 48 s tail folded
    assert plan_windows(1900, 900) == [(0, 900), (900, 1800), (1800, 1898)]  # 98 s tail kept


def test_gemini_transcribes_windows_and_writes_provenance_sidecar(tmp_path: Path) -> None:
    windows = FakeWindows(texts={(1800, 3600): NO_SPEECH_MARKER})
    gemini = make_gemini(windows)
    info = gemini.transcribe(video("longtalk001"), duration_seconds=4000, transcripts_dir=tmp_path)

    assert [(start, end) for _url, start, end in windows.calls] == [(0, 1800), (1800, 3600), (3600, 3998)]
    sidecar = json.loads(Path(info["transcript_path"]).read_text())
    assert sidecar["transcript_source"] == GEMINI_TRANSCRIPT_SOURCE
    assert sidecar["model"] == "gemini-test-model"
    assert sidecar["task_id"] == "transcription.youtube_url"
    assert sidecar["text"] == "speech 0-1800 speech 3600-3998"
    assert sidecar["segments"][1]["no_speech"] is True
    assert sidecar["char_count"] == len(sidecar["text"])
    assert info["estimated_cost_usd"] == pytest.approx(price(3998 * 27, 3998 * 3), abs=1e-6)
    assert gemini.lane_summary()["calls"] == 3


def test_gemini_halts_before_a_window_that_would_cross_the_budget(tmp_path: Path) -> None:
    windows = FakeWindows()
    gemini = make_gemini(windows, budget=0.0001)
    with pytest.raises(GeminiBudgetExhausted):
        gemini.transcribe(video("pricey00001"), duration_seconds=1800, transcripts_dir=tmp_path)
    assert windows.calls == []  # halted before spending anything
    assert gemini.lane_summary()["halted_on_budget"] is True


def test_gemini_refuses_unpriced_calls(tmp_path: Path) -> None:
    gemini = make_gemini(FakeWindows())
    gemini.price_estimator = None
    with pytest.raises(Exception, match="unpriced"):
        gemini.transcribe(video("unpriced001"), duration_seconds=600, transcripts_dir=tmp_path)


def test_truncated_window_is_split_in_half(tmp_path: Path) -> None:
    windows = FakeWindows(truncate_over=1000)
    gemini = make_gemini(windows)
    info = gemini.transcribe(video("splitme0001"), duration_seconds=1800, transcripts_dir=tmp_path)
    assert [(start, end) for _u, start, end in windows.calls] == [(0, 1798), (0, 899), (899, 1798)]
    sidecar = json.loads(Path(info["transcript_path"]).read_text())
    assert sidecar["usage"]["split_windows"] == 1
    assert sidecar["text"] == "speech 0-899 speech 899-1798"


def test_sparse_window_is_reasked_in_halves_and_flagged_if_still_sparse(tmp_path: Path) -> None:
    # A keynote came back with only its closing Q&A: far too few tokens per second.
    windows = FakeWindows(sparse_over=500)
    gemini = make_gemini(windows, window_seconds=900)
    info = gemini.transcribe(video("sparse00001"), duration_seconds=902, transcripts_dir=tmp_path)
    assert [(start, end) for _u, start, end in windows.calls] == [(0, 900), (0, 450), (450, 900)]
    sidecar = json.loads(Path(info["transcript_path"]).read_text())
    assert sidecar["usage"]["sparse_splits"] == 1
    assert sidecar["sparse_windows"] == 0

    stubborn = FakeWindows(sparse_over=0)
    info = make_gemini(stubborn, window_seconds=900).transcribe(
        video("sparse00002"), duration_seconds=402, transcripts_dir=tmp_path
    )
    assert len(stubborn.calls) == 1  # too short to split further
    assert info["sparse_windows"] == 1


def test_transient_gemini_errors_are_retried(tmp_path: Path) -> None:
    windows = FakeWindows(fail_times=2)
    gemini = make_gemini(windows)
    info = gemini.transcribe(video("flaky000001"), duration_seconds=600, transcripts_dir=tmp_path)
    assert info["word_count"] > 0
    assert gemini.lane_summary()["failed_calls"] == 2


# ---------------------------------------------------------------------------
# Tier orchestration
# ---------------------------------------------------------------------------


def test_caption_tier_wins_when_youtube_has_captions(tmp_path: Path) -> None:
    windows = FakeWindows()
    outcome = acquire_transcript(
        video("captioned01"),
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=ScriptedRunner({"captioned01": "caption"}),
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=YouTubeLaneState(),
        gemini=make_gemini(windows),
        http_get=no_http,
    )
    assert outcome["tier"] == "youtube_caption"
    assert outcome["transcript"]["transcript_source"] == "youtube_caption_auto"
    assert windows.calls == []


def test_bot_gate_blocks_youtube_for_the_rest_of_the_run_and_falls_to_gemini(tmp_path: Path) -> None:
    runner = ScriptedRunner({"gated000001": "bot", "gated000002": "caption"})
    windows = FakeWindows()
    gemini = make_gemini(windows)
    state = YouTubeLaneState()
    kwargs = dict(
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=runner,
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=state,
        gemini=gemini,
        http_get=no_http,
    )
    first = acquire_transcript(video("gated000001", 600), **kwargs)
    calls_after_block = len(runner.calls)
    second = acquire_transcript(video("gated000002", 600), **kwargs)

    assert state.blocked and state.blocked_at_video_id == "gated000001"
    assert first["tier"] == second["tier"] == "gemini"
    assert first["youtube_outcome"] == "bot_check_blocked"
    assert second["youtube_outcome"] == "skipped_lane_blocked"
    assert len(runner.calls) == calls_after_block  # YouTube not touched again


def test_missing_duration_is_read_from_watch_page(tmp_path: Path) -> None:
    windows = FakeWindows()
    outcome = acquire_transcript(
        video("nodur000001"),
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=ScriptedRunner({}),
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=YouTubeLaneState(),
        gemini=make_gemini(windows),
        http_get=lambda url: '..."lengthSeconds":"2400"...',
    )
    assert outcome["status"] == "transcribed"
    assert outcome["duration_source"] == "watch_page"
    assert windows.calls[-1][2] == 2398


def test_unknown_duration_is_a_gap_not_a_blind_gemini_call(tmp_path: Path) -> None:
    windows = FakeWindows()
    outcome = acquire_transcript(
        video("nodur000002"),
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=ScriptedRunner({}),
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=YouTubeLaneState(),
        gemini=make_gemini(windows),
        http_get=lambda url: "<html>no metadata</html>",
    )
    assert outcome["status"] == "duration_unknown"
    assert windows.calls == []


def test_watch_page_metadata_is_best_effort() -> None:
    def boom(url: str) -> str:
        raise OSError("offline")

    assert fetch_watch_page_metadata("x", http_get=boom) == {
        "duration_seconds": None,
        "caption_tracks_listed": None,
        "playability_status": None,
        "playability_reason": None,
    }
    meta = fetch_watch_page_metadata(
        "x",
        http_get=lambda url: '"playabilityStatus":{"status":"OK"},"lengthSeconds":"93","captionTracks":[{"a":1}]',
    )
    assert meta == {
        "duration_seconds": 93,
        "caption_tracks_listed": True,
        "playability_status": "OK",
        "playability_reason": None,
    }


@pytest.mark.parametrize(
    ("page", "status"),
    [
        ('"playabilityStatus":{"status":"ERROR","reason":"This video is unavailable"}', "ERROR"),
        ('"playabilityStatus":{"status":"LOGIN_REQUIRED","reason":"Private video"}', "LOGIN_REQUIRED"),
    ],
)
def test_taken_down_or_private_videos_are_labelled_not_sent_to_gemini(
    tmp_path: Path, page: str, status: str
) -> None:
    windows = FakeWindows()
    outcome = acquire_transcript(
        video("gone0000001"),
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=ScriptedRunner({}),
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=YouTubeLaneState(),
        gemini=make_gemini(windows),
        http_get=lambda url: page,
    )
    assert outcome["status"] == "video_unavailable"
    assert outcome["playability_status"] == status
    assert windows.calls == []


def test_without_gemini_no_caption_routes_to_asr_fallback(tmp_path: Path) -> None:
    outcome = acquire_transcript(
        video("nocap000001"),
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=ScriptedRunner({}),
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=YouTubeLaneState(),
        http_get=no_http,
    )
    assert outcome["status"] == "no_caption_gemini_disabled"
    assert outcome["transcript"]["transcript_source"] == "asr_fallback_required"


# ---------------------------------------------------------------------------
# Video-list mode
# ---------------------------------------------------------------------------


def list_config(tmp_path: Path, rows, gemini=None) -> VideoListConfig:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return VideoListConfig(
        workspace=workspace,
        videos=rows,
        output_dir=tmp_path / "out",
        run_id="run-test",
        min_sleep_seconds=0,
        max_sleep_seconds=0,
        gemini=gemini,
    )


def test_video_list_mixes_tiers_and_resume_skips_transcribed(tmp_path: Path) -> None:
    rows = [
        {"video_id": "listcap0001", "title": "A", "duration_seconds": 300},
        {"video_id": "listgate001", "title": "B", "duration_seconds": 600},
        {"video_id": "listgate002", "title": "C", "duration_seconds": 600},
    ]
    runner = ScriptedRunner({"listcap0001": "caption", "listgate001": "bot"})
    windows = FakeWindows()
    receipt = transcribe_video_list(
        list_config(tmp_path, rows, make_gemini(windows)),
        runner=runner, yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http,
    )
    assert receipt["status"] == "succeeded"
    assert receipt["counts"]["transcribed_by_tier"] == {"youtube_caption": 1, "gemini": 2}
    assert receipt["youtube_lane"]["blocked"] is True
    assert receipt["boundaries"]["audio_downloaded"] is False

    runner.calls.clear()
    windows.calls.clear()
    again = transcribe_video_list(
        list_config(tmp_path, rows, make_gemini(windows)),
        runner=runner, yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http,
    )
    assert again["status"] == "succeeded"
    assert runner.calls == [] and windows.calls == []


def test_unknown_duration_uses_one_whole_video_call_without_touching_youtube(tmp_path: Path) -> None:
    """Watch-page reads get 429-ed under concurrency, so duration is optional."""

    windows = FakeWindows()
    gemini = make_gemini(windows)
    info = gemini.transcribe(video("wholevid001"), duration_seconds=None, transcripts_dir=tmp_path)
    assert windows.calls == [("https://www.youtube.com/watch?v=wholevid001", None, None)]
    sidecar = json.loads(Path(info["transcript_path"]).read_text())
    assert sidecar["whole_video_call"] is True
    assert sidecar["duration_seconds"] is None
    assert sidecar["segments"][0]["whole_video"] is True
    assert sidecar["text"] == "whole video speech"


def test_truncated_whole_video_call_is_a_failure_not_a_short_transcript(tmp_path: Path) -> None:
    gemini = make_gemini(FakeWindows(truncate_over="whole"))
    with pytest.raises(Exception, match="whole_video_incomplete"):
        gemini.transcribe(video("wholevid002"), duration_seconds=None, transcripts_dir=tmp_path)


def test_gemini_only_skips_the_duration_lookup_entirely(tmp_path: Path) -> None:
    windows = FakeWindows()
    outcome = acquire_transcript(
        video("nodurpar01"),  # no duration in the row
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=ScriptedRunner({}),
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=YouTubeLaneState(),
        gemini=make_gemini(windows),
        http_get=no_http,  # asserts if the watch page is read
        gemini_only=True,
    )
    assert outcome["status"] == "transcribed"
    assert outcome["duration_source"] == "whole_video_call"
    assert windows.calls == [("https://www.youtube.com/watch?v=nodurpar01", None, None)]


def test_gemini_only_never_contacts_youtube(tmp_path: Path) -> None:
    rows = [{"video_id": f"gemonly{i:04d}", "duration_seconds": 300} for i in range(4)]  # duration known -> windowed
    runner = ScriptedRunner({r["video_id"]: "caption" for r in rows})  # captions exist, ignored
    windows = FakeWindows()
    config = list_config(tmp_path, rows, make_gemini(windows))
    config.gemini_only = True
    receipt = transcribe_video_list(
        config, runner=runner, yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http
    )
    assert receipt["status"] == "succeeded"
    assert receipt["counts"]["transcribed_by_tier"] == {"gemini": 4}
    assert runner.calls == []  # the caption tier was never asked
    assert receipt["youtube_lane"]["caption_requests"] == 0


@pytest.mark.parametrize(
    "message",
    ["ClientError: 403 PERMISSION_DENIED. The caller does not have permission",
     "ServerError: this video is not available"],
)
def test_gemini_refusal_to_fetch_is_a_gap_not_a_lane_failure(tmp_path: Path, message: str) -> None:
    class Refusing(FakeWindows):
        def __call__(self, url, start, end):
            raise RuntimeError(message)

    outcome = acquire_transcript(
        video("denied00001", 300),
        captions_dir=tmp_path / "captions",
        transcripts_dir=tmp_path / "transcripts",
        runner=ScriptedRunner({}),
        yt_dlp_cmd=YT_DLP_CMD,
        youtube_state=YouTubeLaneState(),
        gemini=make_gemini(Refusing()),
        http_get=no_http,
        gemini_only=True,
    )
    assert outcome["status"] == "video_unavailable"


def test_resume_does_not_retry_videos_proven_gone(tmp_path: Path) -> None:
    rows = [{"video_id": "goneagain01"}]  # no duration -> watch-page playability check
    config = list_config(tmp_path, rows, make_gemini(FakeWindows()))
    first = transcribe_video_list(
        config, runner=ScriptedRunner({}), yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None,
        http_get=lambda url: '"playabilityStatus":{"status":"ERROR","reason":"This video is unavailable"}',
    )
    assert first["counts"]["by_status"] == {"video_unavailable": 1}
    windows = FakeWindows()
    again = transcribe_video_list(
        list_config(tmp_path, rows, make_gemini(windows)),
        runner=ScriptedRunner({}), yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http,
    )
    assert windows.calls == []
    assert again["counts"]["by_status"] == {"video_unavailable": 1}


def test_concurrency_requires_gemini_only(tmp_path: Path) -> None:
    config = list_config(tmp_path, [{"video_id": "concur00001"}], make_gemini(FakeWindows()))
    config.concurrency = 4
    with pytest.raises(ValueError, match="gemini_only"):
        transcribe_video_list(config, runner=ScriptedRunner({}), yt_dlp_cmd=YT_DLP_CMD, http_get=no_http)


def test_concurrent_run_transcribes_every_row_and_respects_the_budget(tmp_path: Path) -> None:
    import threading

    rows = [{"video_id": f"parallel{i:03d}", "duration_seconds": 300} for i in range(12)]
    seen_together: list[int] = []
    live = {"n": 0}
    guard = threading.Lock()

    class SlowWindows(FakeWindows):
        def __call__(self, url: str, start: int, end: int) -> GeminiWindowResult:
            with guard:
                live["n"] += 1
                seen_together.append(live["n"])
            time.sleep(0.05)
            try:
                return super().__call__(url, start, end)
            finally:
                with guard:
                    live["n"] -= 1

    windows = SlowWindows()
    config = list_config(tmp_path, rows, make_gemini(windows, budget=5.0))
    config.gemini_only = True
    config.concurrency = 4
    receipt = transcribe_video_list(
        config, runner=ScriptedRunner({}), yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http
    )
    assert receipt["counts"]["transcribed_by_tier"] == {"gemini": 12}
    assert max(seen_together) > 1  # actually ran in parallel
    assert receipt["gemini_lane"]["spent_usd"] <= 5.0


def test_concurrent_run_stops_at_the_budget_without_overspending(tmp_path: Path) -> None:
    rows = [{"video_id": f"cap{i:08d}", "duration_seconds": 600} for i in range(20)]
    # Each 600 s video is estimated at 600*40*0.25e-6 + 600*5*1.5e-6 = $0.0105.
    gemini = make_gemini(FakeWindows(), budget=0.03)
    config = list_config(tmp_path, rows, gemini)
    config.gemini_only = True
    config.concurrency = 5
    receipt = transcribe_video_list(
        config, runner=ScriptedRunner({}), yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http
    )
    assert receipt["status"] == "halted"
    assert receipt["gemini_lane"]["spent_usd"] <= 0.03
    assert 0 < receipt["counts"]["transcribed_by_tier"].get("gemini", 0) < 20


def test_video_list_halts_on_budget_and_checkpoints_progress(tmp_path: Path) -> None:
    rows = [
        {"video_id": "budget00001", "duration_seconds": 600},
        {"video_id": "budget00002", "duration_seconds": 36000},
    ]
    runner = ScriptedRunner({"budget00001": "bot"})
    receipt = transcribe_video_list(
        list_config(tmp_path, rows, make_gemini(FakeWindows(), budget=0.05)),
        runner=runner, yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http,
    )
    assert receipt["status"] == "halted"
    assert "budget" in receipt["halted_reason"]
    inventory = json.loads(Path(receipt["inventory_path"]).read_text())
    assert [row["video_id"] for row in inventory["videos"]] == ["budget00001"]


def test_video_list_without_gemini_stops_at_the_bot_gate(tmp_path: Path) -> None:
    rows = [{"video_id": "stopgate001"}, {"video_id": "stopgate002"}]
    runner = ScriptedRunner({"stopgate001": "bot", "stopgate002": "caption"})
    receipt = transcribe_video_list(
        list_config(tmp_path, rows),
        runner=runner, yt_dlp_cmd=YT_DLP_CMD, sleeper=lambda _s: None, http_get=no_http,
    )
    assert receipt["status"] == "halted"
    assert len(runner.calls) == 1


def test_harvest_uses_gemini_tier_for_captionless_matches(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spine = [ProgramItem(item_id="key1", title="Gated Keynote", family="keynote")]

    def runner(command: list[str]) -> subprocess.CompletedProcess:
        joined = " ".join(command)
        if 'ytsearch5:"Gated Keynote"' in joined:
            row = {"id": "gatedkey001", "title": "Gated Keynote", "duration": 900,
                   "url": "https://www.youtube.com/watch?v=gatedkey001"}
            return completed(json.dumps(row) + "\n")
        if "--write-subs" in command or "--write-auto-subs" in command:
            return completed(stderr=BOT_STDERR, returncode=1)
        return completed()

    windows = FakeWindows()
    config = HarvestConfig(
        workspace=workspace,
        conference_label="Test Conf 2026",
        program_spine=spine,
        rate_limit_seconds=0,
        gemini_transcriber=make_gemini(windows),
    )
    receipt = harvest_youtube_talks(config, runner=runner, sleeper=lambda _s: None, yt_dlp_cmd=YT_DLP_CMD)
    assert receipt["counts"]["transcribed_by_tier"] == {"gemini": 1}
    assert receipt["youtube_lane"]["blocked"] is True
    assert receipt["gemini_lane"]["calls"] == 1
    assert receipt["counts"]["caption_failures"] == 0


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def load_script():
    spec = importlib.util.spec_from_file_location("harvest_youtube_talks_tiers", HARVEST_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_gemini_flags_are_validated_before_any_provider() -> None:
    script = load_script()
    parser = script.build_parser()
    args = parser.parse_args(
        ["--workspace", "ws", "--run-id", "r", "--video-list", "v.json",
         "--gemini-fallback", "--gemini-window-seconds", "60", "--gemini-fps", "2"]
    )
    assert script.gemini_argument_errors(args, environ={}) == [
        "--gemini-fallback requires a positive --gemini-budget-usd",
        "--gemini-fallback requires GOOGLE_API_KEY (source the repository .env)",
        "--gemini-window-seconds must be at least 300",
        "--gemini-fps must be in (0, 1]",
    ]
    args.public_only = True
    assert "--public-only rejects --gemini-fallback; it is a paid provider tier" in (
        script.public_only_argument_errors(args)
    )


def test_cli_requires_exactly_one_input_mode(tmp_path: Path) -> None:
    script = load_script()
    with pytest.raises(SystemExit):
        script.main(["--workspace", str(tmp_path), "--run-id", "r"])
    with pytest.raises(SystemExit):
        script.main(["--workspace", str(tmp_path), "--run-id", "r",
                     "--program-spine", "s.json", "--video-list", "v.json"])


def test_no_speech_markers_are_stripped_and_loops_detected() -> None:
    from conference_pipeline.acquisition.youtube_transcripts import (
        collapse_repeated_sentences,
        repetition_loop_count,
        strip_no_speech_markers,
    )

    assert strip_no_speech_markers("[NO SPEECH]\nThe first task is simple.\n[NO SPEECH]") == "The first task is simple."
    assert strip_no_speech_markers("[NO SPEECH]") == ""
    looped = "Intro sentence about model merging. " + "The merging combines the parameters of models. " * 30
    assert repetition_loop_count(looped) == 30
    collapsed = collapse_repeated_sentences(looped)
    assert collapsed.count("The merging combines") == 1 and collapsed.startswith("Intro sentence")


def test_looping_window_is_re_asked_in_halves(tmp_path) -> None:  # noqa: ANN001
    loop = "Intro. " + "The merging combines the parameters of models. " * 40
    windows = FakeWindows(texts={(0, 398): loop})  # the planner stops 2 s short of the end
    gemini = make_gemini(windows, window_seconds=400)
    record = gemini.transcribe(video("loop1", 400.0), duration_seconds=400.0, transcripts_dir=tmp_path)
    assert [(start, end) for _url, start, end in windows.calls] == [(0, 398), (0, 199), (199, 398)]
    payload = __import__("json").loads((tmp_path / "loop1.json").read_text())
    assert payload["text"] == "speech 0-199 speech 199-398"
    assert payload["usage"]["loop_splits"] == 1
    assert record["segment_count"] == 2
