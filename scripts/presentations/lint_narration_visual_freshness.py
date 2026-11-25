#!/usr/bin/env python3
"""Lint narration freshness against the current visual deck artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_figure_detail_narration import bad_figure_detail_count  # noqa: E402


FORBIDDEN_TTS_TEXT = ("**CONTEXT**", "**REFS**", "not spoken", "[arXiv]", "Open with", "Close the deck")


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    deck_id: str | None = None
    value: Any = None
    threshold: Any = None


@dataclass
class DeckMetrics:
    deck_id: str
    presenter_stale_against_images: bool
    notebooklm_stale_against_images: bool
    presenter_bad_figure_blocks: int
    notebooklm_bad_figure_blocks: int
    presenter_script_json_forbidden_hits: list[str] = field(default_factory=list)


def _manifest_deck_ids(decks_dir: Path, manifest_path: Path | None) -> list[str]:
    path = manifest_path or decks_dir / "presentation_manifest.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("presentations", [])
        ids = [str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id")]
        if ids:
            return ids
    return [
        path.name
        for path in sorted(decks_dir.iterdir())
        if path.is_dir() and not path.name.startswith("_")
    ]


def _latest_mtime(paths: list[Path]) -> float:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else 0.0


def _forbidden_hits_in_presenter_json(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join(
        str(segment.get("text") or "")
        for segment in payload.get("segments", [])
        if isinstance(segment, dict)
    )
    lowered = text.lower()
    return [token for token in FORBIDDEN_TTS_TEXT if token.lower() in lowered]


def lint_deck(deck_dir: Path) -> DeckMetrics:
    images_mtime = _latest_mtime(
        [deck_dir / "5_draft_slides_with_images.md", deck_dir / "asset_inventory.json"]
    )
    presenter_path = deck_dir / "6_presenter_script.md"
    notebook_path = deck_dir / "6_presenter_script_notebooklm_2host.md"
    presenter_mtime = presenter_path.stat().st_mtime if presenter_path.exists() else 0.0
    notebook_mtime = notebook_path.stat().st_mtime if notebook_path.exists() else 0.0
    return DeckMetrics(
        deck_id=deck_dir.name,
        presenter_stale_against_images=bool(images_mtime and presenter_mtime and presenter_mtime < images_mtime),
        notebooklm_stale_against_images=bool(images_mtime and notebook_mtime and notebook_mtime < images_mtime),
        presenter_bad_figure_blocks=bad_figure_detail_count(presenter_path),
        notebooklm_bad_figure_blocks=bad_figure_detail_count(notebook_path),
        presenter_script_json_forbidden_hits=_forbidden_hits_in_presenter_json(
            deck_dir / "video_build" / "script.json"
        ),
    )


def _status(findings: list[Finding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "failed"
    if any(finding.severity == "warn" for finding in findings):
        return "warning"
    return "passed"


def lint_workspace(decks_dir: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    findings: list[Finding] = []
    decks: list[DeckMetrics] = []
    for deck_id in _manifest_deck_ids(decks_dir, manifest_path):
        deck_dir = decks_dir / deck_id
        if not deck_dir.exists():
            findings.append(
                Finding(
                    rule="manifest_deck_missing",
                    severity="fail",
                    message="Presentation manifest references a deck directory that is missing.",
                    deck_id=deck_id,
                )
            )
            continue
        metrics = lint_deck(deck_dir)
        decks.append(metrics)
        if metrics.presenter_stale_against_images:
            findings.append(
                Finding(
                    rule="presenter_narration_stale_against_images",
                    severity="fail",
                    message="Single-presenter script is older than the current image markdown or asset inventory.",
                    deck_id=deck_id,
                )
            )
        if metrics.notebooklm_stale_against_images:
            findings.append(
                Finding(
                    rule="notebooklm_narration_stale_against_images",
                    severity="fail",
                    message="NotebookLM script is older than the current image markdown or asset inventory.",
                    deck_id=deck_id,
                )
            )
        if metrics.presenter_bad_figure_blocks:
            findings.append(
                Finding(
                    rule="presenter_bad_figure_detail_language",
                    severity="fail",
                    message="Single-presenter figure-detail narration still contains generic or malformed visual language.",
                    deck_id=deck_id,
                    value=metrics.presenter_bad_figure_blocks,
                    threshold=0,
                )
            )
        if metrics.notebooklm_bad_figure_blocks:
            findings.append(
                Finding(
                    rule="notebooklm_bad_figure_detail_language",
                    severity="fail",
                    message="NotebookLM figure-detail narration still contains generic or malformed visual language.",
                    deck_id=deck_id,
                    value=metrics.notebooklm_bad_figure_blocks,
                    threshold=0,
                )
            )
        if metrics.presenter_script_json_forbidden_hits:
            findings.append(
                Finding(
                    rule="presenter_script_json_forbidden_text",
                    severity="fail",
                    message="Presenter TTS JSON contains non-spoken text.",
                    deck_id=deck_id,
                    value=metrics.presenter_script_json_forbidden_hits,
                    threshold=[],
                )
            )

    return {
        "status": _status(findings),
        "metrics": {
            "deck_count": len(decks),
            "presenter_stale_against_images": sum(d.presenter_stale_against_images for d in decks),
            "notebooklm_stale_against_images": sum(d.notebooklm_stale_against_images for d in decks),
            "presenter_bad_figure_blocks": sum(d.presenter_bad_figure_blocks for d in decks),
            "notebooklm_bad_figure_blocks": sum(d.notebooklm_bad_figure_blocks for d in decks),
            "presenter_script_json_forbidden_decks": sum(
                bool(d.presenter_script_json_forbidden_hits) for d in decks
            ),
        },
        "findings": [asdict(finding) for finding in findings],
        "decks": [asdict(deck) for deck in decks],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = lint_workspace(args.decks_dir, manifest_path=args.manifest)
    output = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    print(
        "narration visual freshness lint: "
        f"{payload['status']} ({len(payload['findings'])} findings)",
        file=sys.stderr,
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
