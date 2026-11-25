#!/usr/bin/env python3
"""Export a Reveal.js presentation HTML file to PDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.structured_video import (
    export_pdf_with_playwright,
    export_render_spec_artifacts,
)


def _workspace_path(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} must stay inside the workspace: {resolved}"
        ) from exc
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a workspace-local Reveal.js HTML presentation to PDF."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--render-spec", type=Path)
    parser.add_argument("--input-html", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--slide-mapping", type=Path)
    parser.add_argument("--frame-manifest", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    workspace = args.workspace.expanduser().resolve()
    args.workspace = workspace
    if args.render_spec is not None:
        args.render_spec = _workspace_path(
            args.render_spec,
            workspace,
            label="--render-spec",
        )
    args.input_html = _workspace_path(
        args.input_html,
        workspace,
        label="--input-html",
    )
    args.output_pdf = _workspace_path(
        args.output_pdf,
        workspace,
        label="--output-pdf",
    )
    deck_dir = args.input_html.parent
    defaults = {
        "slide_mapping": deck_dir / "presentation_slide_mapping.json",
        "frame_manifest": deck_dir / "slide_frame_manifest.json",
        "frames_dir": deck_dir / "rendered_frames",
        "contact_sheet": deck_dir / "render_contact_sheet.png",
        "receipt_output": deck_dir / "render_export_receipt.json",
    }
    for attr, default in defaults.items():
        raw_value = getattr(args, attr)
        if raw_value is None:
            setattr(args, attr, default)
            continue
        setattr(
            args,
            attr,
            _workspace_path(
                raw_value,
                workspace,
                label=f"--{attr.replace('_', '-')}",
            ),
        )
    return args


def export_pdf(input_html: Path, output_pdf: Path) -> None:
    export_pdf_with_playwright(input_html=input_html, output_pdf=output_pdf)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        payload = {
            "status": "planned",
            "presentation_id": args.presentation_id,
            "presentation_type": args.presentation_type,
            "topic": args.topic,
            "render_spec": str(args.render_spec) if args.render_spec else None,
            "input_html": str(args.input_html),
            "output_pdf": str(args.output_pdf),
            "slide_mapping": str(args.slide_mapping),
            "frame_manifest": str(args.frame_manifest),
            "frames_dir": str(args.frames_dir),
            "contact_sheet": str(args.contact_sheet),
            "receipt_output": str(args.receipt_output),
            "engine": "Playwright Chromium",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.render_spec is not None:
        receipt = export_render_spec_artifacts(
            render_spec=args.render_spec,
            output_html=args.input_html,
            output_pdf=args.output_pdf,
            slide_mapping=args.slide_mapping,
            frame_manifest=args.frame_manifest,
            frames_dir=args.frames_dir,
            contact_sheet=args.contact_sheet,
            receipt_output=args.receipt_output,
            workspace=args.workspace,
            command=[Path(__file__).as_posix(), *sys.argv[1:]],
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "pass" else 1
    export_pdf(args.input_html, args.output_pdf)
    print(f"PDF exported: {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
