"""Direct artifact-truth validation for generated presentation evidence.

This validator intentionally inspects rendered artifacts instead of trusting
success receipts. PDF content checks use pypdf; PDF raster checks use PyMuPDF
(`fitz`) plus Pillow. If the raster stack is unavailable, the check records a
blocker rather than falling back to page count evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence

from conference_pipeline.figures.frame import BADGED_FAMILIES
from conference_pipeline.figures.tokens import SCHEMATIC_BADGE_TEXT
from conference_pipeline.workflow_runs import write_json_atomic

HTML_PATH_TEXT_VISUAL = "HTML_PATH_TEXT_VISUAL"
HTML_NO_IMAGE_OR_SVG = "HTML_NO_IMAGE_OR_SVG"
SOURCE_GENERIC_FALLBACK = "SOURCE_GENERIC_FALLBACK"
SOURCE_UNRESOLVED_OR_TITLE_MISMATCHED = "SOURCE_UNRESOLVED_OR_TITLE_MISMATCHED"
SOURCE_MISSING_AUTHOR_OR_SPEAKER = "SOURCE_MISSING_AUTHOR_OR_SPEAKER"
SLUG_TITLE_VISIBLE = "SLUG_TITLE_VISIBLE"
FINAL_REPORT_LINK_MISSING = "FINAL_REPORT_LINK_MISSING"
TRUNCATED_VISIBLE_TEXT = "TRUNCATED_VISIBLE_TEXT"
PDF_CONTENT_STREAM_MISSING = "PDF_CONTENT_STREAM_MISSING"
PDF_RASTER_BLANK_OR_LOW_ENTROPY = "PDF_RASTER_BLANK_OR_LOW_ENTROPY"
PDF_RASTER_CHECK_DEPENDENCY_MISSING = "PDF_RASTER_CHECK_DEPENDENCY_MISSING"
HTML_PDF_SLIDE_COUNT_MISMATCH = "HTML_PDF_SLIDE_COUNT_MISMATCH"
SLIDE_FRAME_MISSING = "SLIDE_FRAME_MISSING"
SLIDE_FRAME_HASH_STALE = "SLIDE_FRAME_HASH_STALE"
HUMAN_REVIEW_NOT_APPROVED = "HUMAN_REVIEW_NOT_APPROVED"
RECEIPT_ONLY_EDIT_AGAINST_UNCHANGED_ARTIFACT = (
    "RECEIPT_ONLY_EDIT_AGAINST_UNCHANGED_ARTIFACT"
)
SCAFFOLD_OR_PROXY_EVIDENCE_USED = "SCAFFOLD_OR_PROXY_EVIDENCE_USED"
LEGACY_MARKDOWN_PROOF_USED = "LEGACY_MARKDOWN_PROOF_USED"
VIDEO_FRAME_PHASH_MISMATCH = "VIDEO_FRAME_PHASH_MISMATCH"
VIDEO_FRAME_PIXEL_MISMATCH = "VIDEO_FRAME_PIXEL_MISMATCH"
VIDEO_FLAT_FRAME_LOW_ENTROPY = "VIDEO_FLAT_FRAME_LOW_ENTROPY"
AUDIO_SILENCE_SPAN_EXCEEDED = "AUDIO_SILENCE_SPAN_EXCEEDED"
AUDIO_TOTAL_SILENCE_EXCEEDED = "AUDIO_TOTAL_SILENCE_EXCEEDED"
SYNTHETIC_FIXTURE_VIDEO_USED_AS_PRODUCTION = (
    "SYNTHETIC_FIXTURE_VIDEO_USED_AS_PRODUCTION"
)
FIGURE_NUMERIC_TICKS_WITHOUT_LEDGER_ROWS = "FIGURE_NUMERIC_TICKS_WITHOUT_LEDGER_ROWS"
FIGURE_SCHEMATIC_NOT_LABELLED_SCHEMATIC = "FIGURE_SCHEMATIC_NOT_LABELLED_SCHEMATIC"
FIGURE_SCHEMATIC_CARRIES_NUMERIC_TICKS = "FIGURE_SCHEMATIC_CARRIES_NUMERIC_TICKS"

DEFAULT_CHECKS = ("html", "pdf", "frames", "audio", "mp4", "source", "receipts", "figures")
FIGURE_MANIFEST_DIRS = ("assets/compiled", "assets/generated")
FIGURE_MANIFEST_GLOB = "*.manifest.json"
BLOCKER_SEVERITY = "blocker"
PDF_MIN_LUMINANCE_STDDEV = 2.0
PDF_MIN_NON_WHITE_COVERAGE = 0.02
LEGACY_MARKDOWN_FILENAMES = (
    "1_draft_slides.md",
    "2_draft_slides_enriched_with_papers.md",
    "3_draft_slides_reflowed.md",
    "3.5_draft_slides_consolidated.md",
    "4_draft_slides_sanity_check.md",
    "4.5_draft_slides_with_hyperlinks.md",
    "5_draft_slides_with_images.md",
)


@dataclass(frozen=True)
class ArtifactTruthFinding:
    code: str
    severity: str
    message: str
    path: str
    evidence: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ArtifactTruthResult:
    deck_dir: str
    checks: tuple[str, ...]
    findings: tuple[ArtifactTruthFinding, ...]
    generated_at: str

    @property
    def status(self) -> str:
        return "fail" if any(f.severity == BLOCKER_SEVERITY for f in self.findings) else "pass"

    def to_dict(self) -> dict[str, object]:
        blockers = [finding.to_dict() for finding in self.findings if finding.severity == BLOCKER_SEVERITY]
        return {
            "schema_version": 1,
            "status": self.status,
            "generated_at": self.generated_at,
            "deck_dir": self.deck_dir,
            "checks": list(self.checks),
            "findings": [finding.to_dict() for finding in self.findings],
            "blockers": blockers,
        }


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.section_count = 0
        self.image_count = 0
        self.inline_svg_count = 0
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        if normalized == "section":
            self.section_count += 1
        elif normalized == "img":
            self.image_count += 1
        elif normalized == "svg":
            self.inline_svg_count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        stripped = data.strip()
        if stripped:
            self.text_parts.append(stripped)

    @property
    def visible_text(self) -> str:
        return " ".join(self.text_parts)


def validate_artifact_truth(
    deck_dir: Path | str,
    *,
    checks: Iterable[str] | None = None,
    fixture_root: Path | str | None = None,
    current_artifacts: bool = False,
    production_mode: bool = True,
) -> ArtifactTruthResult:
    """Inspect rendered artifacts and return deterministic blocker findings."""

    root = Path(fixture_root or deck_dir).expanduser().resolve()
    requested = _normalize_checks(checks)
    findings: list[ArtifactTruthFinding] = []
    if production_mode:
        findings.extend(_check_fixture_metadata(root))
    if "html" in requested:
        findings.extend(_check_html(root))
    if "source" in requested:
        findings.extend(_check_source_binding(root))
    if "pdf" in requested:
        findings.extend(_check_pdf(root))
    if "frames" in requested:
        findings.extend(_check_slide_frames(root))
    if "receipts" in requested:
        direct_codes = {finding.code for finding in findings}
        findings.extend(
            _check_receipts(root, direct_blocker_codes=direct_codes, requested=requested)
        )
    if "figures" in requested:
        findings.extend(_check_figures(root))
    if "audio" in requested or "mp4" in requested:
        findings.extend(_check_media(root, requested=requested, current_artifacts=current_artifacts))
    return ArtifactTruthResult(
        deck_dir=str(root),
        checks=requested,
        findings=tuple(findings),
        generated_at=_isoformat_z(),
    )


def parse_checks(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip() or value.strip().lower() == "all":
        return DEFAULT_CHECKS
    parsed = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = sorted(set(parsed) - set(DEFAULT_CHECKS))
    if unknown:
        raise ValueError(f"unknown artifact truth checks: {', '.join(unknown)}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, help="Presentation deck directory to inspect.")
    parser.add_argument("--report-out", type=Path, help="Path for artifact_truth_report.json.")
    parser.add_argument("--checks", default="all", help="Comma list: html,pdf,frames,audio,mp4,source,receipts.")
    parser.add_argument("--expect-status", choices=("pass", "fail"))
    parser.add_argument("--fixture-root", type=Path, help="Validate a fixture directory instead of deck-dir.")
    parser.add_argument("--current-artifacts", action="store_true", help="Label report as current pre-repair artifact evidence.")
    parser.add_argument("--summary", action="store_true", help="Print a compact status summary.")
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target = args.fixture_root or args.deck_dir
    if target is None:
        raise SystemExit("--deck-dir or --fixture-root is required")
    try:
        checks = parse_checks(args.checks)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    result = validate_artifact_truth(
        args.deck_dir or target,
        checks=checks,
        fixture_root=args.fixture_root,
        current_artifacts=args.current_artifacts,
    )
    report_path = args.report_out or (Path(target) / "artifact_truth_report.json")
    write_json_atomic(report_path, result.to_dict())
    if args.summary:
        print(
            json.dumps(
                {
                    "status": result.status,
                    "finding_codes": sorted({finding.code for finding in result.findings}),
                    "report": str(report_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.expect_status:
        return 0 if result.status == args.expect_status else 2
    return 0 if result.status == "pass" else 2


def _normalize_checks(checks: Iterable[str] | None) -> tuple[str, ...]:
    if checks is None:
        return DEFAULT_CHECKS
    normalized = tuple(str(check).strip().lower() for check in checks if str(check).strip())
    if not normalized:
        return DEFAULT_CHECKS
    unknown = sorted(set(normalized) - set(DEFAULT_CHECKS))
    if unknown:
        raise ValueError(f"unknown artifact truth checks: {', '.join(unknown)}")
    return normalized


def _check_html(root: Path) -> list[ArtifactTruthFinding]:
    path = root / "presentation.html"
    if not path.is_file():
        return [_finding(HTML_NO_IMAGE_OR_SVG, "presentation.html is missing", path)]
    text = path.read_text(encoding="utf-8", errors="replace")
    parser = _VisibleTextParser()
    parser.feed(text)
    visible = parser.visible_text
    findings: list[ArtifactTruthFinding] = []
    if re.search(r"\b(?:decks/|assets/)[^\s<]+?\.(?:svg|png|jpe?g|webp)\b", visible):
        findings.append(
            _finding(
                HTML_PATH_TEXT_VISUAL,
                "HTML exposes asset file paths as visible visual evidence",
                path,
                {"sample": _sample(visible, r"(?:decks/|assets/)[^\s<]+?\.(?:svg|png|jpe?g|webp)")},
            )
        )
    if parser.image_count == 0 and parser.inline_svg_count == 0:
        findings.append(
            _finding(
                HTML_NO_IMAGE_OR_SVG,
                "HTML has no rendered img tags or inline SVG visuals",
                path,
                {"image_count": parser.image_count, "inline_svg_count": parser.inline_svg_count},
            )
        )
    if re.search(r"\b\d{2}_[a-z0-9]+(?:_[a-z0-9]+)+\b", visible):
        findings.append(
            _finding(
                SLUG_TITLE_VISIBLE,
                "Visible deck text contains a machine slug instead of a presentation title",
                path,
                {"sample": _sample(visible, r"\b\d{2}_[a-z0-9_]+\b")},
            )
        )
    if "drive.google.com" not in text and "Final report" not in visible:
        findings.append(
            _finding(
                FINAL_REPORT_LINK_MISSING,
                "Rendered HTML does not expose the canonical final report link",
                path,
            )
        )
    if re.search(r"\b(?:multi-t|sycophanc)(?:\s|$|\.)", visible) or re.search(
        r"\b[a-z]{4,}-\s*(?:Sources|$)", visible
    ):
        findings.append(
            _finding(
                TRUNCATED_VISIBLE_TEXT,
                "Rendered HTML contains visibly truncated text",
                path,
                {"sample": _sample(visible, r"(?:multi-t|sycophanc|[a-z]{4,}-)")},
            )
        )
    return findings


def _check_source_binding(root: Path) -> list[ArtifactTruthFinding]:
    path = root / "source_binding_report.json"
    payload = _load_json_object(path)
    if payload is None:
        return [_finding(SOURCE_GENERIC_FALLBACK, "source_binding_report.json is missing or invalid", path)]
    findings: list[ArtifactTruthFinding] = []
    for slide_index, slide in enumerate(_as_list(payload.get("slides"))):
        if not isinstance(slide, dict):
            continue
        unresolved = _as_list(slide.get("unresolved_sources"))
        if unresolved:
            findings.append(
                _finding(
                    SOURCE_UNRESOLVED_OR_TITLE_MISMATCHED,
                    "source binding includes unresolved sources",
                    path,
                    {"slide_index": slide_index, "unresolved_count": len(unresolved)},
                )
            )
        for ref_index, reference in enumerate(_as_list(slide.get("visible_references"))):
            if not isinstance(reference, dict):
                continue
            ref_path = f"slides[{slide_index}].visible_references[{ref_index}]"
            title = str(reference.get("title") or "").strip()
            url = str(reference.get("url") or "").strip()
            rationale = str(reference.get("relevance_rationale") or "").strip()
            status = str(reference.get("status") or reference.get("resolution_status") or "").strip().lower()
            people_fields = (
                _as_list(reference.get("authors"))
                + _as_list(reference.get("speakers"))
                + _as_list(reference.get("talks"))
                + _as_list(reference.get("resources"))
            )
            works = _as_list(reference.get("works"))
            has_named_work = bool(
                [item for item in works if str(item).strip()]
            ) or (bool(title) and title.lower() != "structured presentation source")
            if (
                title.lower() == "structured presentation source"
                or "fallback source binding" in rationale.lower()
                or url.rstrip("/") == "https://openreview.net"
            ):
                findings.append(
                    _finding(
                        SOURCE_GENERIC_FALLBACK,
                        "visible source is a generic fallback, not a resolved paper/talk/work",
                        path,
                        {"path": ref_path, "title": title, "url": url},
                    )
                )
            if status in {"unresolved", "ambiguous", "title_mismatch", "homepage_only", "packet_too_large"}:
                findings.append(
                    _finding(
                        SOURCE_UNRESOLVED_OR_TITLE_MISMATCHED,
                        "visible source carries an ineligible resolution status",
                        path,
                        {"path": ref_path, "status": status},
                    )
                )
            # A visible source names a real entity: a person (author/speaker/
            # talk/resource) OR a specific named work (paper/talk title).
            # Title-only works are citable when authors were not extracted
            # (ledger status missing_people) -- mirrors the same has_people-
            # or-has_named_work rule source_inventory.py already applies when
            # deciding binding eligibility in the first place, so this later
            # truth check does not re-block what upstream policy approved.
            if not any(str(item).strip() for item in people_fields) and not has_named_work:
                findings.append(
                    _finding(
                        SOURCE_MISSING_AUTHOR_OR_SPEAKER,
                        "visible source is missing author, speaker, talk, or resource attribution",
                        path,
                        {"path": ref_path, "title": title},
                    )
                )
    return findings


def _check_pdf(root: Path) -> list[ArtifactTruthFinding]:
    path = root / "presentation.pdf"
    if not path.is_file():
        return [_finding(PDF_CONTENT_STREAM_MISSING, "presentation.pdf is missing", path)]
    findings: list[ArtifactTruthFinding] = []
    page_count = 0
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        for index, page in enumerate(reader.pages, start=1):
            if page.get_contents() is None:
                findings.append(
                    _finding(
                        PDF_CONTENT_STREAM_MISSING,
                        "PDF page has no /Contents stream",
                        path,
                        {"page": index, "page_count": page_count, "library": "pypdf"},
                    )
                )
    except Exception as exc:
        findings.append(
            _finding(
                PDF_CONTENT_STREAM_MISSING,
                "PDF could not be parsed with pypdf",
                path,
                {"error": str(exc), "library": "pypdf"},
            )
        )
    findings.extend(_check_pdf_raster(path, page_count=page_count))
    html_sections = _html_section_count(root / "presentation.html")
    if html_sections is not None and page_count and html_sections != page_count:
        findings.append(
            _finding(
                HTML_PDF_SLIDE_COUNT_MISMATCH,
                "HTML section count does not match PDF page count",
                path,
                {"html_sections": html_sections, "pdf_pages": page_count},
            )
        )
    return findings


def _check_pdf_raster(path: Path, *, page_count: int) -> list[ArtifactTruthFinding]:
    try:
        import fitz
        from PIL import Image, ImageStat
    except ImportError as exc:  # pragma: no cover - environment dependent
        return [
            _finding(
                PDF_RASTER_CHECK_DEPENDENCY_MISSING,
                "PyMuPDF/Pillow raster stack is unavailable; page count cannot prove PDF truth",
                path,
                {"error": str(exc), "libraries": ["fitz", "Pillow"]},
            )
        ]
    findings: list[ArtifactTruthFinding] = []
    try:
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False)
                mode = "RGB" if pixmap.n < 4 else "RGBA"
                image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples).convert("L")
                stat = ImageStat.Stat(image)
                stddev = float(stat.stddev[0]) if stat.stddev else 0.0
                pixels = list(image.getdata())
                non_white = sum(1 for value in pixels if value < 245)
                coverage = non_white / len(pixels) if pixels else 0.0
                if stddev <= PDF_MIN_LUMINANCE_STDDEV or coverage <= PDF_MIN_NON_WHITE_COVERAGE:
                    findings.append(
                        _finding(
                            PDF_RASTER_BLANK_OR_LOW_ENTROPY,
                            "PDF page raster is blank or low entropy",
                            path,
                            {
                                "page": page_index,
                                "page_count": page_count,
                                "luminance_stddev": round(stddev, 4),
                                "non_white_coverage": round(coverage, 4),
                                "libraries": ["fitz", "Pillow"],
                            },
                        )
                    )
    except Exception as exc:
        findings.append(
            _finding(
                PDF_RASTER_BLANK_OR_LOW_ENTROPY,
                "PDF raster check failed",
                path,
                {"error": str(exc), "libraries": ["fitz", "Pillow"]},
            )
        )
    return findings


def _check_slide_frames(root: Path) -> list[ArtifactTruthFinding]:
    manifest_path = root / "slide_frame_manifest.json"
    if not manifest_path.is_file():
        return [_finding(SLIDE_FRAME_MISSING, "slide_frame_manifest.json is missing", manifest_path)]
    payload = _load_json_object(manifest_path)
    if payload is None:
        return [_finding(SLIDE_FRAME_MISSING, "slide frame manifest is invalid JSON", manifest_path)]
    findings: list[ArtifactTruthFinding] = []
    frame_entries = _as_list(payload.get("frames") or payload.get("slides"))
    if not frame_entries:
        findings.append(_finding(SLIDE_FRAME_MISSING, "slide frame manifest contains no frames", manifest_path))
    for index, entry in enumerate(frame_entries):
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path") or entry.get("frame_path") or entry.get("file")
        if not isinstance(raw_path, str) or not raw_path.strip():
            findings.append(
                _finding(
                    SLIDE_FRAME_MISSING,
                    "slide frame entry does not name a frame file",
                    manifest_path,
                    {"frame_index": index},
                )
            )
            continue
        frame_path = _resolve_deck_path(root, raw_path)
        if not frame_path.is_file():
            findings.append(
                _finding(
                    SLIDE_FRAME_MISSING,
                    "declared slide frame is missing",
                    frame_path,
                    {"frame_index": index, "declared_path": raw_path},
                )
            )
            continue
        expected_hash = str(entry.get("sha256") or "").strip()
        if expected_hash and _sha256_file(frame_path) != expected_hash:
            findings.append(
                _finding(
                    SLIDE_FRAME_HASH_STALE,
                    "declared slide frame hash does not match current file",
                    frame_path,
                    {"frame_index": index, "declared_path": raw_path},
                )
            )
    return findings


def _check_receipts(
    root: Path,
    *,
    direct_blocker_codes: set[str],
    requested: tuple[str, ...],
) -> list[ArtifactTruthFinding]:
    if "audio" not in requested and "mp4" not in requested:
        render_export_receipt = root / "render_export_receipt.json"
        if render_export_receipt.is_file():
            return _check_render_export_receipt(
                root,
                render_export_receipt,
                direct_blocker_codes=direct_blocker_codes,
            )

    findings: list[ArtifactTruthFinding] = []
    receipt_path = root / "human_review_receipt.json"
    receipt = _load_json_object(receipt_path)
    if receipt is None:
        findings.append(_finding(HUMAN_REVIEW_NOT_APPROVED, "human review receipt is missing", receipt_path))
    else:
        status = str(receipt.get("status") or "").strip()
        decision = str(receipt.get("decision") or "").strip()
        if status != "approved_for_bulk_rollout" or decision != "approved_for_bulk_rollout":
            findings.append(
                _finding(
                    HUMAN_REVIEW_NOT_APPROVED,
                    "human review does not approve bulk rollout",
                    receipt_path,
                    {"status": status, "decision": decision},
                )
            )
        if _receipt_claims_pass(receipt) and direct_blocker_codes:
            findings.append(
                _finding(
                    RECEIPT_ONLY_EDIT_AGAINST_UNCHANGED_ARTIFACT,
                    "receipt claims passing evidence while direct artifact checks fail",
                    receipt_path,
                    {"direct_blocker_codes": sorted(direct_blocker_codes)},
                )
            )
    for legacy in LEGACY_MARKDOWN_FILENAMES:
        legacy_path = root / legacy
        if legacy_path.exists():
            findings.append(
                _finding(
                    LEGACY_MARKDOWN_PROOF_USED,
                    "legacy numbered Markdown file is present and cannot serve as production proof",
                    legacy_path,
                )
            )
    return findings


def _check_render_export_receipt(
    root: Path,
    receipt_path: Path,
    *,
    direct_blocker_codes: set[str],
) -> list[ArtifactTruthFinding]:
    payload = _load_json_object(receipt_path)
    if payload is None:
        return [
            _finding(
                SCAFFOLD_OR_PROXY_EVIDENCE_USED,
                "render_export_receipt.json is missing or invalid",
                receipt_path,
            )
        ]
    findings: list[ArtifactTruthFinding] = []
    status = str(payload.get("status") or "").strip().lower()
    if status != "pass":
        findings.append(
            _finding(
                SCAFFOLD_OR_PROXY_EVIDENCE_USED,
                "render_export_receipt.json does not report pass",
                receipt_path,
                {"status": status},
            )
        )
    if payload.get("legacy_markdown_paths") not in ([], ()):
        findings.append(
            _finding(
                LEGACY_MARKDOWN_PROOF_USED,
                "render export receipt references old numbered Markdown proof",
                receipt_path,
                {"legacy_markdown_paths": payload.get("legacy_markdown_paths")},
            )
        )
    contact_sheet_path = payload.get("contact_sheet_path")
    if isinstance(contact_sheet_path, str) and contact_sheet_path.strip():
        if not _resolve_deck_path(root, contact_sheet_path).is_file() and not Path(contact_sheet_path).is_file():
            findings.append(
                _finding(
                    SCAFFOLD_OR_PROXY_EVIDENCE_USED,
                    "render export contact sheet is missing",
                    receipt_path,
                    {"contact_sheet_path": contact_sheet_path},
                )
            )
    else:
        findings.append(
            _finding(
                SCAFFOLD_OR_PROXY_EVIDENCE_USED,
                "render export receipt does not record a contact sheet",
                receipt_path,
            )
        )
    frame_manifest = payload.get("frame_manifest")
    if not isinstance(frame_manifest, dict) or not frame_manifest.get("path"):
        findings.append(
            _finding(
                SCAFFOLD_OR_PROXY_EVIDENCE_USED,
                "render export receipt does not record a frame_manifest",
                receipt_path,
            )
        )
    if status == "pass" and direct_blocker_codes:
        findings.append(
            _finding(
                RECEIPT_ONLY_EDIT_AGAINST_UNCHANGED_ARTIFACT,
                "render export receipt claims passing evidence while direct artifact checks fail",
                receipt_path,
                {"direct_blocker_codes": sorted(direct_blocker_codes)},
            )
        )
    return findings


def _check_figures(root: Path) -> list[ArtifactTruthFinding]:
    """Hold compiled figures to their grounding claim.

    Two invariants, read off the figure manifest rather than by scraping SVG
    text (the manifest is what the compiler promises it drew):

    * numeric ticks imply ledger rows -- a figure with a numeric axis must say
      which measured claims it plotted. This is what closes the "illustrative
      ticks" pattern that prose-level ``forbidden_patterns`` cannot see.
    * a schematic figure must carry the schematic badge and must NOT carry
      numeric ticks, so a reader can never mistake an illustration of a claim
      for a measurement of it.

    A deck with no figure manifests is silent here: the checks describe
    compiled figures, and a deck drawn by another route has nothing to read.
    """

    findings: list[ArtifactTruthFinding] = []
    for relative in FIGURE_MANIFEST_DIRS:
        directory = root / relative
        if not directory.is_dir():
            continue
        for manifest_path in sorted(directory.rglob(FIGURE_MANIFEST_GLOB)):
            payload = _load_json_object(manifest_path)
            if payload is None:
                continue
            grounding = payload.get("grounding")
            grounding = grounding if isinstance(grounding, dict) else {}
            kind = str(grounding.get("kind") or "").strip().lower()
            rows = [str(row) for row in _as_list(grounding.get("ledger_rows")) if str(row).strip()]
            ticks = [str(tick) for tick in _as_list(payload.get("numeric_ticks")) if str(tick).strip()]
            runs = [str(run) for run in _as_list(payload.get("text_runs"))]
            figure_id = str(payload.get("figure_id") or manifest_path.stem)
            path = str(manifest_path.relative_to(root))

            if ticks and not rows:
                findings.append(
                    _finding(
                        FIGURE_NUMERIC_TICKS_WITHOUT_LEDGER_ROWS,
                        (
                            f"Figure '{figure_id}' draws numeric ticks but cites no numeric-claims "
                            "ledger rows; a measured axis must name the measurements it plots."
                        ),
                        path,
                        {"figure_id": figure_id, "numeric_ticks": ticks[:8], "ledger_rows": rows},
                    )
                )
            if kind == "schematic":
                # Only figures that could pass for measured data carry the badge:
                # a curve or a chart needs it, a box-and-arrow mechanism does not.
                # The rule is imported rather than restated so the validator and
                # the compiler cannot drift apart on which figures need one.
                family = str(payload.get("family") or "").strip().lower()
                if family in BADGED_FAMILIES and not any(SCHEMATIC_BADGE_TEXT in run for run in runs):
                    findings.append(
                        _finding(
                            FIGURE_SCHEMATIC_NOT_LABELLED_SCHEMATIC,
                            (
                                f"Figure '{figure_id}' is grounded as schematic but does not render the "
                                "schematic badge, so it reads as measured data."
                            ),
                            path,
                            {"figure_id": figure_id, "expected_badge": SCHEMATIC_BADGE_TEXT},
                        )
                    )
                if ticks:
                    findings.append(
                        _finding(
                            FIGURE_SCHEMATIC_CARRIES_NUMERIC_TICKS,
                            (
                                f"Figure '{figure_id}' is grounded as schematic but draws numeric ticks; "
                                "a schematic illustrates the shape of a claim and carries no measurements."
                            ),
                            path,
                            {"figure_id": figure_id, "numeric_ticks": ticks[:8]},
                        )
                    )
    return findings


def _check_media(root: Path, *, requested: tuple[str, ...], current_artifacts: bool) -> list[ArtifactTruthFinding]:
    findings: list[ArtifactTruthFinding] = []
    try:
        from conference_pipeline.presentation.video_sync_validation import (
            validate_audio_silence_spans,
            validate_video_artifact_truth,
        )
    except ImportError:
        return findings
    if "mp4" in requested:
        for filename in ("presentation_video.mp4", "presentation_video_notebooklm.mp4"):
            video_path = root / filename
            if video_path.exists():
                findings.extend(
                    _video_findings_to_artifact_findings(validate_video_artifact_truth(video_path))
                )
            elif current_artifacts:
                findings.append(_finding(VIDEO_FLAT_FRAME_LOW_ENTROPY, "video artifact is missing", video_path))
    if "audio" in requested:
        for filename in (
            "video_build/narration_track.m4a",
            "video_build_notebooklm/narration_track.m4a",
        ):
            audio_path = root / filename
            if audio_path.exists():
                findings.extend(
                    _video_findings_to_artifact_findings(
                        validate_audio_silence_spans(audio_path=audio_path)
                    )
                )
    return findings


def _video_findings_to_artifact_findings(findings: Iterable[Any]) -> list[ArtifactTruthFinding]:
    converted: list[ArtifactTruthFinding] = []
    for finding in findings:
        code = str(getattr(finding, "code", ""))
        message = str(getattr(finding, "message", ""))
        path = str(getattr(finding, "path", ""))
        if code:
            converted.append(_finding(code, message or code, Path(path), getattr(finding, "evidence", {})))
    return converted


def _check_fixture_metadata(root: Path) -> list[ArtifactTruthFinding]:
    for filename in ("fixture_metadata.json", "metadata.json"):
        path = root / filename
        payload = _load_json_object(path)
        if payload is None:
            continue
        if payload.get("production_evidence") is False:
            return [
                _finding(
                    SYNTHETIC_FIXTURE_VIDEO_USED_AS_PRODUCTION,
                    "fixture metadata marks this media as non-production evidence",
                    path,
                    {"production_evidence": False},
                )
            ]
    return []


def _html_section_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.section_count


def _receipt_claims_pass(payload: dict[str, Any]) -> bool:
    if str(payload.get("status") or "").lower() in {"passed", "pass", "approved_for_bulk_rollout"}:
        return True
    if str(payload.get("decision") or "").lower() == "approved_for_bulk_rollout":
        return True
    evidence = payload.get("review_evidence")
    if isinstance(evidence, dict):
        return any(_receipt_claims_pass(value) for value in evidence.values() if isinstance(value, dict))
    status = str(payload.get("objective_evidence_status") or "").lower()
    return status in {"passed", "pass"}


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _resolve_deck_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finding(
    code: str,
    message: str,
    path: Path,
    evidence: dict[str, object] | None = None,
) -> ArtifactTruthFinding:
    return ArtifactTruthFinding(
        code=code,
        severity=BLOCKER_SEVERITY,
        message=message,
        path=str(path),
        evidence=evidence or {},
    )


def _sample(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0) if match else None


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
