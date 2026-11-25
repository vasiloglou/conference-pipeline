"""reconcile_contract_hashes.py stamps render_spec timing from the narrated delivery.

The render spec is generated before the engaging narration rebuilds the delivery
contracts, so it carried the stub timing (12 s a slide, 300 s a deck) while the
delivery it names ran ~27 minutes; CVPR 2026 Gate A reviewers failed decks on it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "presentations" / "reconcile_contract_hashes.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_render_spec_timing_follows_the_delivery_contracts(tmp_path: Path) -> None:
    module = _load()
    _write(
        tmp_path / "delivery_single_speaker.json",
        {
            "total_duration_seconds": 121.5,
            "segments": [
                {"slide_id": "sl_01", "duration_seconds": "60.5"},
                {"slide_id": "sl_02", "duration_seconds": "61.0"},
            ],
        },
    )
    _write(tmp_path / "delivery_notebooklm.json", {"total_duration_seconds": 130.0, "segments": []})
    _write(
        tmp_path / "render_spec.json",
        {
            "slides": [
                {"slide_id": "sl_01", "timing_seconds": 12.0},
                {"slide_id": "sl_02", "timing_seconds": 12.0},
                {"slide_id": "s-events"},
            ],
            "delivery": {
                "single_speaker": {"total_duration_seconds": 300.0},
                "notebooklm": {"total_duration_seconds": 300.0},
            },
            "input_hashes": {},
        },
    )

    changes = module.reconcile(tmp_path)
    spec = json.loads((tmp_path / "render_spec.json").read_text(encoding="utf-8"))

    assert "render_spec:timing_from_delivery" in changes
    assert [slide.get("timing_seconds") for slide in spec["slides"]] == [60.5, 61.0, None]
    assert spec["delivery"]["single_speaker"]["total_duration_seconds"] == 121.5
    assert spec["delivery"]["notebooklm"]["total_duration_seconds"] == 130.0
    assert "render_spec:timing_from_delivery" not in module.reconcile(tmp_path)
