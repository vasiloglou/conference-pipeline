"""Path-based video delivery checks and their owner script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.presentation import validation as presentation_validation
from conference_pipeline.presentation import video_delivery

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "presentations" / "validate_video_delivery.py"


def _mp4_bytes(duration_seconds: float, timescale: int = 1000) -> bytes:
    """A minimal MP4 whose ``moov/mvhd`` atom declares ``duration_seconds`` --
    enough for ``_mp4_duration_seconds`` when no ffprobe is on PATH."""

    mvhd = (
        b"\x00\x00\x00\x38mvhd"
        + b"\x00"  # version 0
        + b"\x00\x00\x00"  # flags
        + (0).to_bytes(4, "big")  # creation time
        + (0).to_bytes(4, "big")  # modification time
        + timescale.to_bytes(4, "big")
        + int(duration_seconds * timescale).to_bytes(4, "big")
        + bytes(32)
    )
    moov = (len(mvhd) + 8).to_bytes(4, "big") + b"moov" + mvhd
    return b"\x00\x00\x00\x10ftypmp42\x00\x00\x00\x00" + moov


def _deck_with_videos(tmp_path: Path, *, audio_seconds: float = 120.0) -> tuple[Path, list[Path]]:
    deck_dir = tmp_path / "decks" / "01_demo"
    videos = []
    for name, build_dir in (
        ("presentation_video.mp4", "video_build"),
        ("presentation_video_notebooklm.mp4", "video_build_notebooklm"),
    ):
        video = deck_dir / name
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(_mp4_bytes(audio_seconds))
        build = deck_dir / build_dir
        build.mkdir(exist_ok=True)
        (build / "narration_track.m4a").write_bytes(b"audio")
        (build / "slides_concat.txt").write_text(
            f"file 'slide-001.png'\nduration {audio_seconds / 2:.3f}\n"
            f"file 'slide-002.png'\nduration {audio_seconds / 2:.3f}\n",
            encoding="utf-8",
        )
        videos.append(video)
    return deck_dir, videos


def _probe_factory(*, video_seconds: float, audio_seconds: float, track_seconds: float):
    def probe(path: Path):
        if path.suffix == ".m4a":
            return video_delivery._MediaDurations(format_duration=track_seconds, audio_duration=track_seconds)
        return video_delivery._MediaDurations(
            format_duration=max(video_seconds, audio_seconds),
            video_duration=video_seconds,
            audio_duration=audio_seconds,
            video_frame_rate="25/1",
            video_frame_count=int(video_seconds * 25),
        )

    return probe


def test_all_four_checks_pass_on_aligned_videos(tmp_path: Path) -> None:
    deck_dir, videos = _deck_with_videos(tmp_path)
    checks = video_delivery.run_video_delivery_checks(
        videos,
        probe=_probe_factory(video_seconds=120.0, audio_seconds=120.2, track_seconds=120.0),
        detect_black_tail=lambda *_args: 0.0,
    )
    assert [check.check_id for check in checks] == list(video_delivery.VIDEO_DELIVERY_CHECK_IDS)
    assert {check.status for check in checks} == {"pass"}
    report = video_delivery.video_delivery_report(
        videos,
        deck_dir=deck_dir,
        probe=_probe_factory(video_seconds=120.0, audio_seconds=120.2, track_seconds=120.0),
        detect_black_tail=lambda *_args: 0.0,
    )
    assert report["status"] == "pass"
    assert report["missing_videos"] == []
    assert report["videos"] == ["presentation_video.mp4", "presentation_video_notebooklm.mp4"]
    receipt_check = next(c for c in report["checks"] if c["check_id"] == "video_duration_matches_build_receipts")
    assert "video_build/narration_track.m4a" in receipt_check["evidence_paths"]


def test_audio_outliving_video_fails_stream_alignment(tmp_path: Path) -> None:
    _, videos = _deck_with_videos(tmp_path)
    checks = {
        check.check_id: check
        for check in video_delivery.run_video_delivery_checks(
            videos,
            probe=_probe_factory(video_seconds=100.0, audio_seconds=120.0, track_seconds=120.0),
            detect_black_tail=lambda *_args: 0.0,
        )
    }
    assert checks["video_stream_durations_match"].status == "fail"
    assert "delta 20.000s" in checks["video_stream_durations_match"].message


def test_over_budget_and_black_tail_fail(tmp_path: Path) -> None:
    _, videos = _deck_with_videos(tmp_path, audio_seconds=2500.0)
    checks = {
        check.check_id: check
        for check in video_delivery.run_video_delivery_checks(
            videos,
            probe=_probe_factory(video_seconds=2500.0, audio_seconds=2500.0, track_seconds=2500.0),
            detect_black_tail=lambda *_args: 6.5,
        )
    }
    assert checks["video_duration_under_40_minutes"].status == "fail"
    assert checks["video_tail_not_black"].status == "fail"
    assert checks["video_duration_matches_build_receipts"].status == "pass"


def test_receipt_drift_fails_and_missing_tools_warn(tmp_path: Path) -> None:
    _, videos = _deck_with_videos(tmp_path, audio_seconds=120.0)
    checks = {
        check.check_id: check
        for check in video_delivery.run_video_delivery_checks(
            videos,
            probe=_probe_factory(video_seconds=150.0, audio_seconds=150.0, track_seconds=120.0),
            detect_black_tail=lambda *_args: None,
        )
    }
    assert checks["video_duration_matches_build_receipts"].status == "fail"
    assert checks["video_tail_not_black"].status == "warn"
    assert video_delivery.overall_status([c.status for c in checks.values()]) == "fail"


def test_report_fails_on_a_missing_video(tmp_path: Path) -> None:
    deck_dir, videos = _deck_with_videos(tmp_path)
    videos[1].unlink()
    report = video_delivery.video_delivery_report(
        videos,
        deck_dir=deck_dir,
        probe=_probe_factory(video_seconds=120.0, audio_seconds=120.0, track_seconds=120.0),
        detect_black_tail=lambda *_args: 0.0,
    )
    assert report["status"] == "fail"
    assert report["missing_videos"] == ["presentation_video_notebooklm.mp4"]
    assert report["checks"][0]["check_id"] == "video_file_present"
    # The present video is still checked.
    assert {c["check_id"] for c in report["checks"][1:]} == set(video_delivery.VIDEO_DELIVERY_CHECK_IDS)


def test_validation_stage_wrappers_still_honour_patches_on_the_validation_module(tmp_path: Path, monkeypatch) -> None:
    """The manifest-bound checks delegate here but keep reading their probe from
    ``presentation.validation`` so the historical monkeypatch seam holds."""

    _, videos = _deck_with_videos(tmp_path)
    monkeypatch.setattr(
        presentation_validation,
        "_probe_media_durations",
        _probe_factory(video_seconds=90.0, audio_seconds=90.0, track_seconds=90.0),
    )
    monkeypatch.setattr(presentation_validation, "_detect_black_tail_seconds", lambda *_args: 0.0)
    verdict = presentation_validation.stream_alignment_verdict(
        videos, probe=presentation_validation._probe_media_durations
    )
    assert verdict.status == "pass"
    # The unpatched module would have no ffprobe fixture and must not be consulted.
    assert presentation_validation.VIDEO_MAX_DURATION_SECONDS == video_delivery.VIDEO_MAX_DURATION_SECONDS


@pytest.mark.parametrize("missing_second", [False, True])
def test_owner_script_writes_a_report_and_exits_on_failures(tmp_path: Path, monkeypatch, missing_second: bool) -> None:
    deck_dir, videos = _deck_with_videos(tmp_path)
    if missing_second:
        videos[1].unlink()
    report_out = deck_dir / "video_delivery_validation.json"
    # Without ffprobe/ffmpeg on PATH the probes return None: warn, never a false pass.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--deck-dir",
            str(deck_dir),
            "--presentation-id",
            "01_demo",
            "--video",
            str(videos[0]),
            "--video",
            str(videos[1]),
            "--report-out",
            str(report_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["deck_id"] == "01_demo"
    assert payload["generator"].endswith("validate_video_delivery.py")
    if missing_second:
        assert completed.returncode == 1, completed.stderr
        assert payload["status"] == "fail"
        assert payload["missing_videos"] == ["presentation_video_notebooklm.mp4"]
    else:
        # No ffprobe/ffmpeg on PATH: the container's own mvhd duration (120 s)
        # satisfies the budget and matches slides_concat.txt, while the stream
        # alignment and black-tail checks cannot run and warn -- never a false
        # pass, never a spurious fail.
        assert completed.returncode == 0, completed.stderr
        assert payload["status"] == "warn"
        by_id = {check["check_id"]: check["status"] for check in payload["checks"]}
        assert by_id["video_duration_under_40_minutes"] == "pass"
        assert by_id["video_duration_matches_build_receipts"] == "pass"
        assert by_id["video_stream_durations_match"] == "warn"
        assert by_id["video_tail_not_black"] == "warn"
