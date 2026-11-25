#!/usr/bin/env python3
"""Reproducible pipeline step: re-stamp cross-referenced contract hashes after the
deck/render-spec/delivery contracts are modified (e.g. by add_bookend_slides.py
and rebuild_delivery_engaging.py).

Keeps the contract chain internally consistent:
- delivery_*.json `presentation_deck_sha256` / `source_binding_report_sha256` /
  `visual_plan_sha256` are set from the current files,
- render_spec.json `input_hashes` are recomputed for every key that resolves to a
  file in the deck directory (deck, deliveries, source binding, visual plan,
  figure specs, etc.). Keys that point at upstream .planning artifacts are left
  untouched (those files are not modified here).

Run after add_bookend_slides.py + rebuild_delivery_engaging.py and before render
export / review.

Usage:
  python scripts/presentations/reconcile_contract_hashes.py \
      --deck-dir conferences/iclr2026/decks/04_safety_alignment_privacy
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DELIVERY_HASH_FIELDS = {
    "presentation_deck_sha256": "presentation_deck.json",
    "source_binding_report_sha256": "source_binding_report.json",
    "visual_plan_sha256": "visual_plan.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reconcile(deck_dir: Path) -> list[str]:
    changes: list[str] = []

    # 1. Delivery contracts first (render_spec input_hashes will read their new hashes).
    for delivery_name in ("delivery_notebooklm.json", "delivery_single_speaker.json"):
        path = deck_dir / delivery_name
        if not path.is_file():
            continue
        delivery = _load(path)
        dirty = False
        for field, target in DELIVERY_HASH_FIELDS.items():
            target_path = deck_dir / target
            if field in delivery and target_path.is_file():
                current = _sha256(target_path)
                if delivery.get(field) != current:
                    delivery[field] = current
                    dirty = True
                    changes.append(f"{delivery_name}:{field}")
        if dirty:
            _dump(path, delivery)

    # 2. render_spec timing follows the narrated delivery. It is stamped from the
    # delivery contracts that exist when the render spec is generated -- the
    # pre-narration stubs (12 s a slide, 300 s a deck) -- and the engaging
    # narration rebuilds them afterwards; left alone, the spec contradicts the
    # delivery it names (flagged by the CVPR 2026 Gate A reviewers).
    render_spec_path = deck_dir / "render_spec.json"
    if render_spec_path.is_file():
        render_spec = _load(render_spec_path)
        timing_dirty = False
        single = deck_dir / "delivery_single_speaker.json"
        if single.is_file():
            durations: dict[str, float] = {}
            for segment in _load(single).get("segments") or []:
                try:
                    seconds = float(segment.get("duration_seconds"))
                except (TypeError, ValueError):
                    continue
                slide_id = str(segment.get("slide_id") or "")
                if slide_id and seconds > 0:
                    durations[slide_id] = round(durations.get(slide_id, 0.0) + seconds, 2)
            for slide in render_spec.get("slides") or []:
                seconds = durations.get(str(slide.get("slide_id") or ""))
                if seconds is not None and slide.get("timing_seconds") != seconds:
                    slide["timing_seconds"] = seconds
                    timing_dirty = True
        for variant, name in (("single_speaker", "delivery_single_speaker.json"), ("notebooklm", "delivery_notebooklm.json")):
            delivery_path = deck_dir / name
            entry = (render_spec.get("delivery") or {}).get(variant)
            if delivery_path.is_file() and isinstance(entry, dict):
                total = _load(delivery_path).get("total_duration_seconds")
                if total is not None and entry.get("total_duration_seconds") != total:
                    entry["total_duration_seconds"] = total
                    timing_dirty = True
        if timing_dirty:
            _dump(render_spec_path, render_spec)
            changes.append("render_spec:timing_from_delivery")

    # 3. render_spec input_hashes: recompute every key that is a deck-dir file.
    if render_spec_path.is_file():
        render_spec = _load(render_spec_path)
        input_hashes = render_spec.get("input_hashes")
        if isinstance(input_hashes, dict):
            dirty = False
            for key in list(input_hashes.keys()):
                candidate = deck_dir / key
                if candidate.is_file():
                    current = _sha256(candidate)
                    if input_hashes.get(key) != current:
                        input_hashes[key] = current
                        dirty = True
                        changes.append(f"render_spec.input_hashes:{key}")
            if dirty:
                _dump(render_spec_path, render_spec)

    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    changes = reconcile(args.deck_dir.expanduser().resolve())
    if changes:
        print(f"re-stamped {len(changes)} hash reference(s):")
        for c in changes:
            print(f"  {c}")
    else:
        print("all contract hashes already consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
