"""Deterministic video-sync receipt checks for structured presentation delivery."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRESENTATION_AUDIO_GAP_DETECTED = "PRESENTATION_AUDIO_GAP_DETECTED"
PRESENTATION_SLIDE_SYNC_MISMATCH = "PRESENTATION_SLIDE_SYNC_MISMATCH"
PRESENTATION_VISUAL_SOURCE_MISMATCH = "PRESENTATION_VISUAL_SOURCE_MISMATCH"
NOTEBOOKLM_TIMING_DIVERGENCE_INCOHERENT = "NOTEBOOKLM_TIMING_DIVERGENCE_INCOHERENT"
VIDEO_FRAME_PHASH_MISMATCH = "VIDEO_FRAME_PHASH_MISMATCH"
VIDEO_FRAME_PIXEL_MISMATCH = "VIDEO_FRAME_PIXEL_MISMATCH"
VIDEO_FLAT_FRAME_LOW_ENTROPY = "VIDEO_FLAT_FRAME_LOW_ENTROPY"
AUDIO_SILENCE_SPAN_EXCEEDED = "AUDIO_SILENCE_SPAN_EXCEEDED"
AUDIO_TOTAL_SILENCE_EXCEEDED = "AUDIO_TOTAL_SILENCE_EXCEEDED"

MAX_FRAME_PHASH_DISTANCE = 12
MAX_MEAN_RGB_PIXEL_DIFFERENCE = 18.0
MIN_FRAME_LUMINANCE_STDDEV = 2.0
MIN_FRAME_UNIQUE_COLORS = 17
MAX_SILENCE_SPAN_SECONDS = 5.0
# Total-silence ceiling, measured with silencedetect at noise=-35dB and a 0.25s
# minimum span. At that sensitivity the detector counts ordinary inter-clause
# breathing room, so the ceiling has to sit above natural narration cadence and
# still catch dead air. Calibration (ICML 2026 topic 02, reviewer-approved
# conversational narration): the single-speaker track measured 0.160 over
# 1590s and the two-host track 0.163 over 1432s, both with 500-odd spans and a
# longest span of 1.15s -- no dead air anywhere. The separate
# MAX_SILENCE_SPAN_SECONDS guard is what actually catches a dropped or silent
# segment, and it is unchanged.
MAX_TOTAL_SILENCE_RATIO = 0.20


@dataclass(frozen=True)
class VideoSyncFinding:
    code: str
    message: str
    path: str
    evidence: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message, "path": self.path}
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload


@dataclass(frozen=True)
class VideoSyncValidationResult:
    style: str
    errors: tuple[VideoSyncFinding, ...]
    warnings: tuple[VideoSyncFinding, ...] = ()

    @property
    def status(self) -> str:
        return "fail" if self.errors else "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "style": self.style,
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


def validate_delivery_sync_payload(
    *,
    style: str,
    render_spec: dict[str, Any],
    delivery_contract: dict[str, Any],
    video_path: Path,
) -> VideoSyncValidationResult:
    errors: list[VideoSyncFinding] = []
    if not video_path.exists():
        errors.append(
            VideoSyncFinding(
                code=PRESENTATION_VISUAL_SOURCE_MISMATCH,
                message="video artifact is missing",
                path=str(video_path),
            )
        )
    render_slides = {
        str(slide.get("slide_id"))
        for slide in render_spec.get("slides", [])
        if isinstance(slide, dict)
    }
    delivery_slides = {
        str(segment.get("slide_id"))
        for segment in delivery_contract.get("segments", [])
        if isinstance(segment, dict)
    }
    if delivery_slides and render_slides and not delivery_slides.issubset(render_slides):
        errors.append(
            VideoSyncFinding(
                code=PRESENTATION_SLIDE_SYNC_MISMATCH,
                message="delivery contract references slide ids absent from render_spec.json",
                path="delivery.segments",
            )
        )
    if delivery_contract.get("max_voice_gap_seconds", 0) and delivery_contract["max_voice_gap_seconds"] > 5:
        errors.append(
            VideoSyncFinding(
                code=PRESENTATION_AUDIO_GAP_DETECTED,
                message="delivery contract records a voice gap above five seconds",
                path="delivery.max_voice_gap_seconds",
            )
        )
    if style == "notebooklm" and delivery_contract.get("timing_divergence_seconds", 0) > 8:
        errors.append(
            VideoSyncFinding(
                code=NOTEBOOKLM_TIMING_DIVERGENCE_INCOHERENT,
                message="NotebookLM timing divergence requires a separate coherent delivery plan",
                path="delivery.timing_divergence_seconds",
            )
        )
    return VideoSyncValidationResult(style=style, errors=tuple(errors))


def validate_rendered_frame_truth(
    frame_path: Path,
    *,
    reference_frame: Path | None = None,
    path_label: str | None = None,
) -> list[VideoSyncFinding]:
    """Validate frame entropy and optional rendered-frame similarity thresholds."""

    path = Path(frame_path)
    label = path_label or str(path)
    if not path.is_file():
        return [
            VideoSyncFinding(
                code=PRESENTATION_VISUAL_SOURCE_MISMATCH,
                message="rendered frame is missing",
                path=label,
            )
        ]
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:  # pragma: no cover - environment dependent
        return [
            VideoSyncFinding(
                code=VIDEO_FLAT_FRAME_LOW_ENTROPY,
                message="Pillow is required for rendered-frame truth checks",
                path=label,
                evidence={"error": str(exc)},
            )
        ]

    findings: list[VideoSyncFinding] = []
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        thumbnail = rgb.copy()
        thumbnail.thumbnail((320, 180))
        grayscale = thumbnail.convert("L")
        luminance_stddev = float(ImageStat.Stat(grayscale).stddev[0])
        unique_colors = len(thumbnail.getcolors(maxcolors=320 * 180 + 1) or [])
        if luminance_stddev <= MIN_FRAME_LUMINANCE_STDDEV or unique_colors < MIN_FRAME_UNIQUE_COLORS:
            findings.append(
                VideoSyncFinding(
                    code=VIDEO_FLAT_FRAME_LOW_ENTROPY,
                    message="sampled frame is flat or low entropy",
                    path=label,
                    evidence={
                        "luminance_stddev": round(luminance_stddev, 4),
                        "unique_colors": unique_colors,
                        "min_luminance_stddev": MIN_FRAME_LUMINANCE_STDDEV,
                        "min_unique_colors": MIN_FRAME_UNIQUE_COLORS,
                    },
                )
            )
        if reference_frame is not None:
            reference = Path(reference_frame)
            if reference.is_file():
                with Image.open(reference) as reference_image:
                    phash_distance = _phash_distance(rgb, reference_image.convert("RGB"))
                    mean_rgb_difference = _mean_rgb_difference(rgb, reference_image.convert("RGB"))
                if phash_distance > MAX_FRAME_PHASH_DISTANCE:
                    findings.append(
                        VideoSyncFinding(
                            code=VIDEO_FRAME_PHASH_MISMATCH,
                            message="sampled frame pHash differs from rendered reference frame",
                            path=label,
                            evidence={
                                "phash_distance": phash_distance,
                                "max_phash_distance": MAX_FRAME_PHASH_DISTANCE,
                                "reference_frame": str(reference),
                            },
                        )
                    )
                if mean_rgb_difference > MAX_MEAN_RGB_PIXEL_DIFFERENCE:
                    findings.append(
                        VideoSyncFinding(
                            code=VIDEO_FRAME_PIXEL_MISMATCH,
                            message="sampled frame mean RGB difference exceeds rendered reference frame threshold",
                            path=label,
                            evidence={
                                "mean_rgb_difference": round(mean_rgb_difference, 4),
                                "max_mean_rgb_difference": MAX_MEAN_RGB_PIXEL_DIFFERENCE,
                                "reference_frame": str(reference),
                            },
                        )
                    )
    return findings


def validate_audio_silence_profile(
    *,
    duration_seconds: float,
    silence_spans: list[tuple[float, float]],
    path_label: str = "audio",
) -> list[VideoSyncFinding]:
    """Validate that silence spans do not exceed the Phase 15 thresholds."""

    findings: list[VideoSyncFinding] = []
    normalized_spans = [(max(0.0, start), max(0.0, end)) for start, end in silence_spans]
    span_lengths = [max(0.0, end - start) for start, end in normalized_spans]
    longest = max(span_lengths, default=0.0)
    total = sum(span_lengths)
    ratio = total / duration_seconds if duration_seconds > 0 else 1.0
    if longest > MAX_SILENCE_SPAN_SECONDS:
        findings.append(
            VideoSyncFinding(
                code=AUDIO_SILENCE_SPAN_EXCEEDED,
                message="audio contains a silence span over five seconds",
                path=path_label,
                evidence={
                    "longest_silence_seconds": round(longest, 4),
                    "max_silence_span_seconds": MAX_SILENCE_SPAN_SECONDS,
                },
            )
        )
    if ratio > MAX_TOTAL_SILENCE_RATIO:
        findings.append(
            VideoSyncFinding(
                code=AUDIO_TOTAL_SILENCE_EXCEEDED,
                message=(
                    "audio total silence exceeds "
                    f"{MAX_TOTAL_SILENCE_RATIO:.0%} of duration"
                ),
                path=path_label,
                evidence={
                    "duration_seconds": round(duration_seconds, 4),
                    "total_silence_seconds": round(total, 4),
                    "total_silence_ratio": round(ratio, 4),
                    "max_total_silence_ratio": MAX_TOTAL_SILENCE_RATIO,
                },
            )
        )
    return findings


def validate_audio_silence_spans(audio_path: Path) -> list[VideoSyncFinding]:
    path = Path(audio_path)
    if not path.is_file():
        return [
            VideoSyncFinding(
                code=AUDIO_TOTAL_SILENCE_EXCEEDED,
                message="audio artifact is missing",
                path=str(path),
            )
        ]
    duration = _media_duration_seconds(path)
    silence_spans = _detect_silence_spans(path)
    if duration is None:
        return [
            VideoSyncFinding(
                code=AUDIO_TOTAL_SILENCE_EXCEEDED,
                message="audio duration is unreadable",
                path=str(path),
            )
        ]
    if not silence_spans and _has_audio_stream(path) is False:
        silence_spans = [(0.0, duration)]
    return validate_audio_silence_profile(
        duration_seconds=duration,
        silence_spans=silence_spans,
        path_label=str(path),
    )


def validate_video_artifact_truth(video_path: Path) -> list[VideoSyncFinding]:
    """Sample MP4 frames and audio to enforce Phase 15 direct media thresholds."""

    path = Path(video_path)
    if not path.is_file():
        return [
            VideoSyncFinding(
                code=PRESENTATION_VISUAL_SOURCE_MISMATCH,
                message="video artifact is missing",
                path=str(path),
            )
        ]
    findings: list[VideoSyncFinding] = []
    duration = _media_duration_seconds(path)
    sample_times = _sample_times(duration)
    with tempfile.TemporaryDirectory(prefix="video-sync-frames-") as temp_dir:
        for index, timestamp in enumerate(sample_times, start=1):
            frame_path = Path(temp_dir) / f"sample_{index:02d}.png"
            if _extract_video_frame(path, timestamp, frame_path):
                findings.extend(validate_rendered_frame_truth(frame_path, path_label=f"{path}#t={timestamp:.2f}"))
            else:
                findings.append(
                    VideoSyncFinding(
                        code=VIDEO_FLAT_FRAME_LOW_ENTROPY,
                        message="could not extract sampled video frame",
                        path=str(path),
                        evidence={"timestamp_seconds": timestamp},
                    )
                )
    findings.extend(validate_audio_silence_spans(path))
    return findings


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _phash_distance(left: Any, right: Any) -> int:
    left_hash = _phash_bits(left)
    right_hash = _phash_bits(right)
    return sum(1 for a, b in zip(left_hash, right_hash, strict=True) if a != b)


def _phash_bits(image: Any) -> tuple[int, ...]:
    import numpy as np

    grayscale = image.convert("L").resize((32, 32))
    pixels = np.asarray(grayscale, dtype=np.float32)
    try:
        import cv2

        transformed = cv2.dct(pixels)
        low_freq = transformed[:8, :8]
    except ImportError:  # pragma: no cover - cv2 is available in project env
        low_freq = pixels[:8, :8]
    values = low_freq.flatten()
    median = float(np.median(values[1:])) if len(values) > 1 else float(np.median(values))
    return tuple(1 if value > median else 0 for value in values)


def _mean_rgb_difference(left: Any, right: Any) -> float:
    import numpy as np

    left_rgb = left.convert("RGB").resize((320, 180))
    right_rgb = right.convert("RGB").resize((320, 180))
    diff = np.abs(np.asarray(left_rgb, dtype=np.int16) - np.asarray(right_rgb, dtype=np.int16))
    return float(diff.mean())


def _media_duration_seconds(path: Path) -> float | None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def _has_audio_stream(path: Path) -> bool | None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _detect_silence_spans(path: Path) -> list[tuple[float, float]]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "silencedetect=noise=-35dB:d=0.25",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = completed.stdout + "\n" + completed.stderr
    starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", output)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", output)]
    return list(zip(starts, ends, strict=False))


def _sample_times(duration: float | None) -> list[float]:
    if duration is None or duration <= 0:
        return [0.0]
    if duration < 3:
        return [max(0.0, duration / 2)]
    return [0.5, max(0.5, duration / 2), max(0.5, duration - 0.5)]


def _extract_video_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.returncode == 0 and output_path.is_file()
