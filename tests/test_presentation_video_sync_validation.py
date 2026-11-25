from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from conference_pipeline.presentation.video_sync_validation import (
    AUDIO_SILENCE_SPAN_EXCEEDED,
    AUDIO_TOTAL_SILENCE_EXCEEDED,
    VIDEO_FLAT_FRAME_LOW_ENTROPY,
    VIDEO_FRAME_PHASH_MISMATCH,
    VIDEO_FRAME_PIXEL_MISMATCH,
    validate_audio_silence_profile,
    validate_delivery_sync_payload,
    validate_rendered_frame_truth,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_validate_video_sync_script():
    script_path = PROJECT_ROOT / "scripts" / "presentations" / "validate_video_sync.py"
    spec = importlib.util.spec_from_file_location("phase15_validate_video_sync", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_build_presenter_video_script():
    script_dir = PROJECT_ROOT / "scripts" / "presentations"
    script_path = script_dir / "build_presenter_video.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("phase15_build_presenter_video", script_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(script_dir))
        except ValueError:
            pass


def _write_reference_frame(path: Path) -> None:
    image = Image.new("RGB", (96, 54), (20, 80, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 8, 84, 46), fill=(240, 230, 40))
    draw.line((0, 0, 95, 53), fill=(220, 20, 20), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def test_frame_truth_detects_flat_and_mismatched_frames(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    actual = tmp_path / "actual.png"
    _write_reference_frame(reference)
    Image.new("RGB", (96, 54), (255, 255, 255)).save(actual)

    findings = validate_rendered_frame_truth(actual, reference_frame=reference)
    codes = {finding.code for finding in findings}

    assert VIDEO_FLAT_FRAME_LOW_ENTROPY in codes
    assert VIDEO_FRAME_PHASH_MISMATCH in codes
    assert VIDEO_FRAME_PIXEL_MISMATCH in codes


def test_audio_silence_profile_rejects_long_or_excessive_silence() -> None:
    findings = validate_audio_silence_profile(
        duration_seconds=20.0,
        silence_spans=[(0.0, 6.1), (10.0, 14.0)],
    )
    codes = {finding.code for finding in findings}

    assert AUDIO_SILENCE_SPAN_EXCEEDED in codes
    assert AUDIO_TOTAL_SILENCE_EXCEEDED in codes


def test_delivery_sync_allows_slide_number_alignment_when_render_ids_are_rewritten(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "presentation_video.mp4"
    video_path.write_bytes(b"placeholder mp4 path for contract-only validation")
    render_spec = {
        "slides": [
            {"slide_id": "sl_01_source_grounded_rewrite"},
            {"slide_id": "sl_02_source_grounded_rewrite"},
        ]
    }
    delivery_contract = {
        "max_voice_gap_seconds": 0.5,
        "segments": [
            {"slide_id": "sl_01_legacy_delivery", "slide_number": 1},
            {"slide_id": "sl_02_legacy_delivery", "slide_number": 2},
        ],
    }
    validator = _load_validate_video_sync_script()
    normalized_delivery = validator._delivery_contract_for_sync(render_spec, delivery_contract)

    result = validate_delivery_sync_payload(
        style="single_speaker",
        render_spec=render_spec,
        delivery_contract=normalized_delivery,
        video_path=video_path,
    )

    assert result.status == "pass"
    assert normalized_delivery["phase15_slide_id_alignment"]["mode"] == "slide_number_fallback"


def test_notebooklm_delivery_contract_expands_host_turns_for_tts(tmp_path: Path) -> None:
    delivery_contract = tmp_path / "delivery_notebooklm.json"
    delivery_contract.write_text(
        json.dumps(
            {
                "presentation_id": "04_safety_alignment_privacy",
                "segments": [
                    {
                        "segment_id": "notebooklm-001",
                        "slide_id": "sl_01_legacy_delivery",
                        "slide_number": 1,
                        "speaker": "Host A / Host B",
                        "narration": (
                            "Host A frames deployment-centric trustworthiness. "
                            "Host B connects the visual evidence to layered defenses."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    builder = _load_build_presenter_video_script()

    payload = builder._delivery_script_payload(
        delivery_contract=delivery_contract,
        style="notebooklm",
        voice_label="Host A / Host B",
    )

    assert payload["segments"] == [
        {"speaker": "Host A", "text": "frames deployment-centric trustworthiness."},
        {"speaker": "Host B", "text": "connects the visual evidence to layered defenses."},
    ]
    assert [
        slide["slide_number"] for slide in payload["metadata"]["slides"]
    ] == [1, 1]


def test_audio_silence_ceiling_admits_conversational_cadence_but_not_dead_air() -> None:
    """Calibration guard (ICML 2026 topic 02).

    silencedetect at -35dB/0.25s counts ordinary inter-clause pauses, so a
    healthy two-host narration measured 0.163 total silence over 1432s with a
    longest span of 1.15s. That must pass, while a genuinely gappy track and any
    long dead span must still fail.
    """
    conversational = validate_audio_silence_profile(
        duration_seconds=1432.0,
        silence_spans=[(index * 3.0, index * 3.0 + 0.5) for index in range(467)],
    )
    assert conversational == []

    gappy = validate_audio_silence_profile(
        duration_seconds=1000.0,
        silence_spans=[(index * 3.0, index * 3.0 + 1.2) for index in range(250)],
    )
    assert {finding.code for finding in gappy} == {AUDIO_TOTAL_SILENCE_EXCEEDED}

    # The dead-air guard is independent of the ratio and unchanged.
    dead_air = validate_audio_silence_profile(
        duration_seconds=1432.0,
        silence_spans=[(10.0, 22.0)],
    )
    assert {finding.code for finding in dead_air} == {AUDIO_SILENCE_SPAN_EXCEEDED}
