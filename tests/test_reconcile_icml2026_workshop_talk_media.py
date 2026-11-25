from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).parents[1] / "scripts/reconcile_icml2026_workshop_talk_media.py"
    spec = importlib.util.spec_from_file_location("icml_workshop_reconcile_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reconciliation_uses_official_event_ids_and_marks_cancelled_event() -> None:
    module = _load_module()
    listing = """
    <div class="event-card" data-event-type="Workshop" data-event-title="Recorded Workshop" data-event-id="54001">
      <h3 class="event-title"><a href="/virtual/2026/workshop/54001">Recorded Workshop</a></h3>
    </div>
    <div class="event-card" data-event-type="Workshop" data-event-title="[Cancelled] Workshop" data-event-id="54002">
      <h3 class="event-title"><a href="/virtual/2026/workshop/54002">[Cancelled] Workshop</a></h3>
    </div>
    """
    events = module.parse_workshop_events(
        listing, listing_url="https://icml.cc/virtual/2026/events/workshop"
    )
    inventory = module.build_reconciled_inventory(
        events=events,
        audio_by_event={
            "54001": {
                "local_path": "cache/acquisition-downloads/media_asset/workshop-54001/slideslive_audio/raw.mp3",
                "sha256": "a" * 64,
                "byte_count": 123,
                "media_url": "https://slideslive.com/39000001",
                "slideslive_id": "39000001",
                "content_type": "audio/mpeg",
            }
        },
        listing_url="https://icml.cc/virtual/2026/events/workshop",
        listing_sha256="b" * 64,
        run_id="test-run",
        workspace=Path("/tmp/test-workspace"),
    )

    assert inventory["status"] == "succeeded"
    assert inventory["media_record_count"] == 1
    assert inventory["cancelled_event_count"] == 1
    assert inventory["blocker_count"] == 0
    assert inventory["records"][0]["status"] == "succeeded"
    assert inventory["records"][1]["status"] == "cancelled"
