"""Path-based delivery checks for rendered presentation videos.

The manifest-driven validation stage (``presentation.validation``) runs four
timing/tail checks on the video artifacts a presentation manifest declares.
The shared production chain (``conference_pipeline.production``) builds its
two MP4s per deck without ever writing that manifest, so the checks live here
on plain paths: ``run_video_delivery_checks`` takes the video files and
returns one verdict per check; ``validation.py`` keeps its manifest-bound
wrappers and delegates to these functions.

Checks (ids match the validation stage so reports read the same everywhere):

* ``video_duration_under_40_minutes`` -- the final MP4 stays under the
  40-minute budget.
* ``video_stream_durations_match`` -- the container's audio and video streams
  end together, so a player cannot keep speaking over an ended picture.
* ``video_duration_matches_build_receipts`` -- the final duration agrees with
  the build evidence next to it (``video_build*/narration_track.m4a``,
  ``slides_concat.txt``, duration metadata JSON).
* ``video_tail_not_black`` -- the last seconds are not a sustained black run.

``ffprobe`` / ``ffmpeg`` are located at call time; when they are missing the
checks report ``warn`` (unreadable), never a false ``pass``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIDEO_STREAM_DURATION_TOLERANCE_SECONDS = 1.0
VIDEO_RECEIPT_DURATION_TOLERANCE_SECONDS = 1.0
VIDEO_MAX_DURATION_SECONDS = 40 * 60
VIDEO_BLACK_TAIL_LOOKBACK_SECONDS = 75.0
VIDEO_BLACK_TAIL_FAIL_SECONDS = 3.0
_VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")
_BUILD_DURATION_JSON_KEYS = (
    "duration_seconds",
    "duration_s",
    "total_duration_seconds",
    "audio_duration_seconds",
    "video_duration_seconds",
)

VIDEO_DELIVERY_CHECK_IDS: tuple[str, ...] = (
    "video_duration_under_40_minutes",
    "video_stream_durations_match",
    "video_duration_matches_build_receipts",
    "video_tail_not_black",
)


@dataclass(frozen=True)
class _MediaDurations:
    format_duration: float | None = None
    video_duration: float | None = None
    audio_duration: float | None = None
    video_frame_rate: str | None = None
    video_frame_count: int | None = None


@dataclass(frozen=True)
class VideoDeliveryCheck:
    """One path-based verdict; ``evidence_paths`` are the files it looked at."""

    check_id: str
    status: str
    severity: str
    message: str
    evidence_paths: tuple[Path, ...]

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "evidence_paths": [_display_path(path, relative_to) for path in self.evidence_paths],
        }


ProbeFn = Callable[[Path], "_MediaDurations | None"]
BlackTailFn = Callable[[Path, float], "float | None"]


def _display_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is not None:
        try:
            return path.resolve().relative_to(relative_to.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def _parse_float(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _parse_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        result = int(str(value))
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _probe_media_durations(path: Path) -> _MediaDurations | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        output = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,duration,nb_frames,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        payload = json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None

    format_duration = _parse_float(payload.get("format", {}).get("duration"))
    video_duration: float | None = None
    audio_duration: float | None = None
    video_frame_rate: str | None = None
    video_frame_count: int | None = None
    for stream in payload.get("streams", []):
        if not isinstance(stream, dict):
            continue
        codec_type = stream.get("codec_type")
        duration = _parse_float(stream.get("duration"))
        if codec_type == "video" and video_duration is None:
            video_duration = duration
            video_frame_rate = (
                str(stream.get("avg_frame_rate"))
                if stream.get("avg_frame_rate") not in (None, "0/0", "N/A")
                else None
            )
            video_frame_count = _parse_int(stream.get("nb_frames"))
        elif codec_type == "audio" and audio_duration is None:
            audio_duration = duration
    return _MediaDurations(
        format_duration=format_duration,
        video_duration=video_duration,
        audio_duration=audio_duration,
        video_frame_rate=video_frame_rate,
        video_frame_count=video_frame_count,
    )


def _sum_concat_durations(path: Path) -> float | None:
    try:
        total = 0.0
        found = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("duration "):
                continue
            value = _parse_float(stripped.split(None, 1)[1])
            if value is None:
                return None
            total += value
            found = True
    except OSError:
        return None
    return total if found else None


def _duration_json_values(path: Path) -> tuple[float, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, dict):
        return ()
    values: list[float] = []
    for key in _BUILD_DURATION_JSON_KEYS:
        value = _parse_float(payload.get(key))
        if value is not None:
            values.append(value)
    return tuple(values)


def _build_dir_for_video(video_path: Path) -> Path:
    if "notebooklm" in video_path.stem:
        return video_path.parent / "video_build_notebooklm"
    return video_path.parent / "video_build"


def _expected_duration_sources(
    video_path: Path, *, probe: ProbeFn | None = None
) -> tuple[tuple[Path, float], ...]:
    probe_fn = probe or _probe_media_durations
    build_dir = _build_dir_for_video(video_path)
    sources: list[tuple[Path, float]] = []
    for audio_name in ("narration_track.m4a", "narration_track_raw.m4a"):
        audio_path = build_dir / audio_name
        if not audio_path.exists():
            continue
        probed = probe_fn(audio_path)
        duration = None if probed is None else probed.audio_duration or probed.format_duration
        if duration is not None:
            sources.append((audio_path, duration))
            break
    slides_concat = build_dir / "slides_concat.txt"
    if slides_concat.exists():
        duration = _sum_concat_durations(slides_concat)
        if duration is not None:
            sources.append((slides_concat, duration))
    for json_name in ("metadata.json", "video_metadata.json"):
        metadata_path = build_dir / json_name
        if not metadata_path.exists():
            continue
        for duration in _duration_json_values(metadata_path):
            sources.append((metadata_path, duration))
    return tuple(sources)


def _mp4_duration_seconds(path: Path) -> float | None:
    data = path.read_bytes()
    moov_index = data.find(b"moov")
    if moov_index < 4:
        return None
    mvhd_index = data.find(b"mvhd", moov_index)
    if mvhd_index < 4:
        return None
    version = data[mvhd_index + 4]
    if version == 0 and len(data) >= mvhd_index + 28:
        timescale = int.from_bytes(data[mvhd_index + 16 : mvhd_index + 20], "big")
        duration = int.from_bytes(data[mvhd_index + 20 : mvhd_index + 24], "big")
    elif version == 1 and len(data) >= mvhd_index + 40:
        timescale = int.from_bytes(data[mvhd_index + 24 : mvhd_index + 28], "big")
        duration = int.from_bytes(data[mvhd_index + 28 : mvhd_index + 36], "big")
    else:
        return None
    if timescale <= 0 or duration <= 0:
        return None
    return duration / timescale


_BLACKDETECT_RE = re.compile(
    r"black_start:(?P<start>[0-9.]+)\s+black_end:(?P<end>[0-9.]+)\s+black_duration:(?P<duration>[0-9.]+)"
)


def _detect_black_tail_seconds(video_path: Path, media_duration: float) -> float | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    tail_window = min(VIDEO_BLACK_TAIL_LOOKBACK_SECONDS, media_duration)
    seek_start = max(0.0, media_duration - tail_window)
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-ss",
                f"{seek_start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{tail_window:.3f}",
                "-vf",
                (
                    "blackdetect="
                    f"d={VIDEO_BLACK_TAIL_FAIL_SECONDS:.3f}:"
                    "pix_th=0.10:pic_th=0.98"
                ),
                "-an",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    durations = [
        float(match.group("duration"))
        for match in _BLACKDETECT_RE.finditer(result.stderr)
        if _parse_float(match.group("duration")) is not None
    ]
    return max(durations, default=0.0)


def _final_duration(path: Path, probe_fn: ProbeFn) -> float | None:
    probed = probe_fn(path)
    duration = None if probed is None else probed.format_duration
    if duration is None and path.suffix.lower() == ".mp4":
        duration = _mp4_duration_seconds(path)
    return duration


def duration_budget_verdict(
    video_paths: Sequence[Path],
    *,
    probe: ProbeFn | None = None,
    detect_black_tail: BlackTailFn | None = None,
) -> VideoDeliveryCheck:
    """``video_duration_under_40_minutes``: every final MP4 is within budget."""

    probe_fn = probe or _probe_media_durations
    durations: list[str] = []
    failures: list[str] = []
    unreadable: list[str] = []
    for path in video_paths:
        duration = _final_duration(path, probe_fn)
        if duration is None:
            unreadable.append(path.name)
            continue
        durations.append(f"{path.name}={duration:.3f}s")
        if duration > VIDEO_MAX_DURATION_SECONDS:
            failures.append(
                f"{path.name}: {duration:.3f}s exceeds {VIDEO_MAX_DURATION_SECONDS}s"
            )

    if failures:
        status = "fail"
    elif unreadable:
        status = "warn"
    else:
        status = "pass"
    message = (
        "video duration budget check: "
        + ("; ".join(durations) if durations else "no readable durations")
    )
    if failures:
        message += "; failures: " + "; ".join(failures[:4])
    if unreadable:
        message += "; unreadable durations: " + ", ".join(unreadable[:8])
    return VideoDeliveryCheck(
        check_id="video_duration_under_40_minutes",
        status=status,
        severity="content",
        message=message,
        evidence_paths=tuple(video_paths),
    )


def stream_alignment_verdict(
    video_paths: Sequence[Path],
    *,
    probe: ProbeFn | None = None,
    detect_black_tail: BlackTailFn | None = None,
) -> VideoDeliveryCheck:
    """``video_stream_durations_match``: audio and video streams end together."""

    probe_fn = probe or _probe_media_durations
    failures: list[str] = []
    probe_missing: list[Path] = []
    details: list[str] = []
    for path in video_paths:
        probed = probe_fn(path)
        if probed is None:
            probe_missing.append(path)
            continue
        video_duration = probed.video_duration
        audio_duration = probed.audio_duration
        if video_duration is None or audio_duration is None:
            failures.append(f"{path.name}: missing audio/video stream duration")
            continue
        delta = abs(video_duration - audio_duration)
        details.append(
            f"{path.name}: video={video_duration:.3f}s audio={audio_duration:.3f}s "
            f"delta={delta:.3f}s frame_rate={probed.video_frame_rate or 'unknown'} "
            f"frames={probed.video_frame_count or 'unknown'}"
        )
        if delta > VIDEO_STREAM_DURATION_TOLERANCE_SECONDS:
            failures.append(
                f"{path.name}: audio/video delta {delta:.3f}s exceeds "
                f"{VIDEO_STREAM_DURATION_TOLERANCE_SECONDS:.3f}s"
            )

    if failures:
        status = "fail"
    elif probe_missing:
        status = "warn"
    else:
        status = "pass"
    missing_message = (
        f"; ffprobe unavailable/unreadable for {len(probe_missing)} artifact(s)"
        if probe_missing
        else ""
    )
    message = (
        "audio/video stream duration alignment: "
        + ("; ".join(details[:6]) if details else "no readable stream durations")
        + missing_message
    )
    if failures:
        message += "; failures: " + "; ".join(failures[:4])
    return VideoDeliveryCheck(
        check_id="video_stream_durations_match",
        status=status,
        severity="structural",
        message=message,
        evidence_paths=tuple(video_paths),
    )


def build_receipt_agreement_verdict(
    video_paths: Sequence[Path],
    *,
    probe: ProbeFn | None = None,
    detect_black_tail: BlackTailFn | None = None,
) -> VideoDeliveryCheck:
    """``video_duration_matches_build_receipts``: final duration equals build evidence."""

    probe_fn = probe or _probe_media_durations
    evidence: list[Path] = list(video_paths)
    failures: list[str] = []
    details: list[str] = []
    missing_sources: list[str] = []
    for video_path in video_paths:
        probed = probe_fn(video_path)
        final_duration = None if probed is None else probed.format_duration or probed.audio_duration
        if final_duration is None:
            final_duration = _mp4_duration_seconds(video_path)
        sources = _expected_duration_sources(video_path, probe=probe_fn)
        evidence.extend(source_path for source_path, _duration in sources)
        if final_duration is None:
            failures.append(f"{video_path.name}: final duration unreadable")
            continue
        if not sources:
            missing_sources.append(video_path.name)
            continue
        for source_path, expected_duration in sources:
            delta = abs(final_duration - expected_duration)
            details.append(
                f"{video_path.name} vs {source_path.name}: final={final_duration:.3f}s "
                f"expected={expected_duration:.3f}s delta={delta:.3f}s"
            )
            if delta > VIDEO_RECEIPT_DURATION_TOLERANCE_SECONDS:
                failures.append(
                    f"{video_path.name} vs {source_path.name}: delta {delta:.3f}s exceeds "
                    f"{VIDEO_RECEIPT_DURATION_TOLERANCE_SECONDS:.3f}s"
                )

    if failures:
        status = "fail"
    elif missing_sources:
        status = "skipped_missing"
    else:
        status = "pass"
    message = (
        "final MP4 duration matches build audio/slide duration evidence: "
        + ("; ".join(details[:8]) if details else "no build duration evidence found")
    )
    if missing_sources:
        message += "; missing build duration evidence for " + ", ".join(missing_sources[:8])
    if failures:
        message += "; failures: " + "; ".join(failures[:4])
    return VideoDeliveryCheck(
        check_id="video_duration_matches_build_receipts",
        status=status,
        severity="structural",
        message=message,
        evidence_paths=tuple(dict.fromkeys(evidence)),
    )


def black_tail_verdict(
    video_paths: Sequence[Path],
    *,
    probe: ProbeFn | None = None,
    detect_black_tail: BlackTailFn | None = None,
) -> VideoDeliveryCheck:
    """``video_tail_not_black``: no sustained black run at the end of the MP4."""

    probe_fn = probe or _probe_media_durations
    detect_fn = detect_black_tail or _detect_black_tail_seconds
    details: list[str] = []
    failures: list[str] = []
    unreadable: list[str] = []
    for path in video_paths:
        probed = probe_fn(path)
        media_duration = None if probed is None else probed.format_duration or probed.video_duration
        if media_duration is None:
            unreadable.append(path.name)
            continue
        black_tail_seconds = detect_fn(path, media_duration)
        if black_tail_seconds is None:
            unreadable.append(path.name)
            continue
        details.append(
            f"{path.name}: max black run in final "
            f"{min(VIDEO_BLACK_TAIL_LOOKBACK_SECONDS, media_duration):.0f}s="
            f"{black_tail_seconds:.3f}s"
        )
        if black_tail_seconds >= VIDEO_BLACK_TAIL_FAIL_SECONDS:
            failures.append(
                f"{path.name}: black tail run {black_tail_seconds:.3f}s exceeds "
                f"{VIDEO_BLACK_TAIL_FAIL_SECONDS:.3f}s"
            )

    if failures:
        status = "fail"
    elif unreadable:
        status = "warn"
    else:
        status = "pass"
    message = (
        "final MP4 tail frames are not black: "
        + ("; ".join(details[:8]) if details else "no readable tail-frame evidence")
    )
    if unreadable:
        message += "; ffmpeg tail check unavailable/unreadable for " + ", ".join(unreadable[:8])
    if failures:
        message += "; failures: " + "; ".join(failures[:4])
    return VideoDeliveryCheck(
        check_id="video_tail_not_black",
        status=status,
        severity="structural",
        message=message,
        evidence_paths=tuple(video_paths),
    )


VIDEO_DELIVERY_VERDICTS: dict[str, Callable[..., VideoDeliveryCheck]] = {
    "video_duration_under_40_minutes": duration_budget_verdict,
    "video_stream_durations_match": stream_alignment_verdict,
    "video_duration_matches_build_receipts": build_receipt_agreement_verdict,
    "video_tail_not_black": black_tail_verdict,
}


def run_video_delivery_checks(
    video_paths: Sequence[Path],
    *,
    probe: ProbeFn | None = None,
    detect_black_tail: BlackTailFn | None = None,
) -> list[VideoDeliveryCheck]:
    """Run all four checks over ``video_paths`` (in ``VIDEO_DELIVERY_CHECK_IDS`` order)."""

    paths = tuple(Path(path) for path in video_paths)
    return [
        VIDEO_DELIVERY_VERDICTS[check_id](paths, probe=probe, detect_black_tail=detect_black_tail)
        for check_id in VIDEO_DELIVERY_CHECK_IDS
    ]


def overall_status(statuses: Sequence[str]) -> str:
    """``fail`` beats ``warn`` beats ``pass``; ``skipped_missing`` counts as a warning."""

    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status in {"warn", "skipped_missing"} for status in statuses):
        return "warn"
    return "pass"


def video_delivery_report(
    video_paths: Sequence[Path],
    *,
    deck_dir: Path,
    deck_id: str | None = None,
    probe: ProbeFn | None = None,
    detect_black_tail: BlackTailFn | None = None,
) -> dict[str, Any]:
    """The JSON payload ``validate_video_delivery.py`` writes.

    A named video that does not exist is a ``fail`` in its own right
    (``video_file_present``): the chain calls this right after building both
    videos, so an absent file is a broken build, not an out-of-scope style.
    """

    paths = tuple(Path(path) for path in video_paths)
    present = tuple(path for path in paths if path.is_file() and path.stat().st_size > 0)
    missing = tuple(path for path in paths if path not in present)
    checks: list[dict[str, Any]] = []
    if missing:
        checks.append(
            VideoDeliveryCheck(
                check_id="video_file_present",
                status="fail",
                severity="structural",
                message="missing or empty video files: " + ", ".join(path.name for path in missing),
                evidence_paths=missing,
            ).to_dict(relative_to=deck_dir)
        )
    if present:
        checks.extend(
            check.to_dict(relative_to=deck_dir)
            for check in run_video_delivery_checks(present, probe=probe, detect_black_tail=detect_black_tail)
        )
    status = overall_status([check["status"] for check in checks]) if checks else "fail"
    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    return {
        "schema_version": 1,
        "generator": "scripts/presentations/validate_video_delivery.py",
        "deck_id": deck_id or deck_dir.name,
        "deck_dir": str(deck_dir),
        "videos": [_display_path(path, deck_dir) for path in paths],
        "missing_videos": [_display_path(path, deck_dir) for path in missing],
        "status": status,
        "check_counts": counts,
        "checks": checks,
    }


__all__ = [
    "VIDEO_BLACK_TAIL_FAIL_SECONDS",
    "VIDEO_BLACK_TAIL_LOOKBACK_SECONDS",
    "VIDEO_DELIVERY_CHECK_IDS",
    "VIDEO_MAX_DURATION_SECONDS",
    "VIDEO_RECEIPT_DURATION_TOLERANCE_SECONDS",
    "VIDEO_STREAM_DURATION_TOLERANCE_SECONDS",
    "VideoDeliveryCheck",
    "black_tail_verdict",
    "build_receipt_agreement_verdict",
    "duration_budget_verdict",
    "overall_status",
    "run_video_delivery_checks",
    "stream_alignment_verdict",
    "video_delivery_report",
]
