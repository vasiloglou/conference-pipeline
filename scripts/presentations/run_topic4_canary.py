#!/usr/bin/env python3
"""Topic 4-only canary runner for 04_safety_alignment_privacy.

Structural guard: only runs for the Topic 4 deck. Cannot be bypassed
via flags. Checks Phase 03 gate artifacts before proceeding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOPIC4_DECK_ID = "04_safety_alignment_privacy"

STORYBOARD_REVIEW_FILE = "03-STORYBOARD-REVIEW.json"
CREATIVE_SYNTHESIS_FILE = "04-CREATIVE-STORYBOARD-CROSS-AI-SYNTHESIS.json"
REQUIRED_GATE_STATUS = "approved_for_render_planning"


def _resolve_inside_workspace(path: Path, workspace: Path) -> Path:
    resolved = path.resolve()
    workspace_resolved = workspace.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        pass
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True, help="Deck directory (must be 04_safety_alignment_privacy).")
    parser.add_argument("--phase-dir", type=Path, required=True, help="Phase 03 artifact directory containing 03-STORYBOARD-REVIEW.json.")
    parser.add_argument("--phase04-dir", type=Path, required=False, help="Phase 04 artifact directory containing 04-CREATIVE-STORYBOARD-CROSS-AI-SYNTHESIS.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print would-check JSON and exit 0 without running gates.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    workspace = args.deck_dir.parent.parent
    deck_dir = _resolve_inside_workspace(args.deck_dir, workspace)
    phase_dir = args.phase_dir.resolve()
    phase04_dir = args.phase04_dir.resolve() if args.phase04_dir else None

    if deck_dir.name != TOPIC4_DECK_ID:
        print(
            f"Canary runner is Topic 4 only: expected deck name '{TOPIC4_DECK_ID}', got '{deck_dir.name}'",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(json.dumps({
            "canary_deck": str(deck_dir),
            "phase03_dir": str(phase_dir),
            "phase04_dir": str(phase04_dir) if phase04_dir else None,
            "would_check": [STORYBOARD_REVIEW_FILE, CREATIVE_SYNTHESIS_FILE],
        }))
        return 0

    storyboard_review = phase_dir / STORYBOARD_REVIEW_FILE
    creative_synthesis = (phase04_dir / CREATIVE_SYNTHESIS_FILE) if phase04_dir else None

    found_any = False

    if storyboard_review.exists():
        found_any = True
        try:
            data = json.loads(storyboard_review.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to read {STORYBOARD_REVIEW_FILE}: {exc}", file=sys.stderr)
            return 1
        approval = data.get("approval", {}) if isinstance(data, dict) else {}
        status = approval.get("status") if isinstance(approval, dict) else data.get("status")
        if status != REQUIRED_GATE_STATUS:
            print(
                f"Blocker: {STORYBOARD_REVIEW_FILE} approval.status is '{status}', "
                f"expected '{REQUIRED_GATE_STATUS}'",
                file=sys.stderr,
            )
            return 1

    if creative_synthesis is not None and creative_synthesis.exists():
        found_any = True
        try:
            data = json.loads(creative_synthesis.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to read {CREATIVE_SYNTHESIS_FILE}: {exc}", file=sys.stderr)
            return 1
        status = data.get("gate_status") if isinstance(data, dict) else None
        if status != REQUIRED_GATE_STATUS:
            print(
                f"Blocker: {CREATIVE_SYNTHESIS_FILE} gate_status is '{status}', "
                f"expected '{REQUIRED_GATE_STATUS}'",
                file=sys.stderr,
            )
            return 1

    if not found_any:
        print(
            "Phase 03 gate artifacts not yet produced; proceeding (build-time only)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
