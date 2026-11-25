from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pypdf import PdfWriter

from conference_pipeline.presentation.artifact_truth import (
    FINAL_REPORT_LINK_MISSING,
    HTML_NO_IMAGE_OR_SVG,
    HTML_PATH_TEXT_VISUAL,
    PDF_CONTENT_STREAM_MISSING,
    PDF_RASTER_BLANK_OR_LOW_ENTROPY,
    RECEIPT_ONLY_EDIT_AGAINST_UNCHANGED_ARTIFACT,
    SLUG_TITLE_VISIBLE,
    SOURCE_GENERIC_FALLBACK,
    SOURCE_MISSING_AUTHOR_OR_SPEAKER,
    SYNTHETIC_FIXTURE_VIDEO_USED_AS_PRODUCTION,
    VIDEO_FLAT_FRAME_LOW_ENTROPY,
    validate_artifact_truth,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_blank_pdf(path: Path, *, pages: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=1280, height=720)
    with path.open("wb") as handle:
        writer.write(handle)


def _write_good_html(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<!doctype html>
<html><body>
<section data-markdown-slide="1">
  <h1>Safety Alignment Privacy</h1>
  <img alt="paper-specific visual" src="assets/generated/slide-001.png">
  <a href="https://drive.google.com/file/d/final-report">Final report</a>
  <div class="reference-chips"><a>Benchmark paper by A. Researcher</a></div>
</section>
</body></html>
""",
        encoding="utf-8",
    )


def _write_valid_source_binding(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "slides": [
                {
                    "slide_id": "slide-001",
                    "unresolved_sources": [],
                    "visible_references": [
                        {
                            "label": "SRC-001",
                            "url": "https://openreview.net/forum?id=real-paper",
                            "title": "A real paper title",
                            "authors": ["A. Researcher"],
                            "source_type": "paper",
                            "relevance_rationale": "Grounds the slide claim.",
                        }
                    ],
                }
            ],
        },
    )


def _write_positive_receipt(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "status": "approved_for_bulk_rollout",
            "decision": "approved_for_bulk_rollout",
            "review_evidence": {
                "objective_evidence_report": {"status": "passed", "blockers": []}
            },
        },
    )


def _write_bad_html_and_source(deck_dir: Path) -> None:
    (deck_dir / "presentation.html").write_text(
        """<!doctype html>
<html><head><title>04_safety_alignment_privacy</title></head><body>
<section>
  <h1>04_safety_alignment_privacy</h1>
  <p class="visual-ref">decks/04_safety_alignment_privacy/assets/generated/sl_01_title-visual.svg</p>
  <div class="reference-chips"><a href="https://openreview.net/">Structured presentation source</a></div>
</section>
</body></html>
""",
        encoding="utf-8",
    )
    _write_json(
        deck_dir / "source_binding_report.json",
        {
            "schema_version": 1,
            "slides": [
                {
                    "slide_id": "slide-001",
                    "unresolved_sources": [],
                    "visible_references": [
                        {
                            "label": "SRC-001",
                            "url": "https://openreview.net/",
                            "title": "Structured presentation source",
                            "source_type": "report",
                            "relevance_rationale": "Fallback source binding generated from structured deck contract.",
                        }
                    ],
                }
            ],
        },
    )


def _codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def test_blank_pdf_with_nonzero_page_count_and_positive_receipt_still_fails(
    tmp_path: Path,
) -> None:
    deck_dir = tmp_path / "deck"
    _write_good_html(deck_dir / "presentation.html")
    _write_valid_source_binding(deck_dir / "source_binding_report.json")
    _write_blank_pdf(deck_dir / "presentation.pdf", pages=2)
    _write_positive_receipt(deck_dir / "human_review_receipt.json")

    result = validate_artifact_truth(deck_dir, checks={"pdf", "receipts"})

    assert result.status == "fail"
    assert PDF_CONTENT_STREAM_MISSING in _codes(result)
    assert PDF_RASTER_BLANK_OR_LOW_ENTROPY in _codes(result)
    assert RECEIPT_ONLY_EDIT_AGAINST_UNCHANGED_ARTIFACT in _codes(result)


def test_html_and_source_blockers_use_stable_codes(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    _write_bad_html_and_source(deck_dir)

    result = validate_artifact_truth(deck_dir, checks={"html", "source"})

    assert result.status == "fail"
    assert {
        HTML_PATH_TEXT_VISUAL,
        HTML_NO_IMAGE_OR_SVG,
        SLUG_TITLE_VISIBLE,
        FINAL_REPORT_LINK_MISSING,
        SOURCE_GENERIC_FALLBACK,
        SOURCE_MISSING_AUTHOR_OR_SPEAKER,
    }.issubset(_codes(result))


def test_css_asset_paths_are_not_visible_text_blockers(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    (deck_dir / "presentation.html").write_text(
        """<!doctype html>
<html><head>
<style>
  .title-logo { background-image: url("assets/brand/relationalai/logo_white.png"); }
  .chip { border-color: #fff; }
</style>
</head><body>
<section>
  <h1>Safety Alignment Privacy</h1>
  <img alt="brand logo" src="assets/brand/relationalai/logo_white.png">
  <a href="https://drive.google.com/file/d/final-report">Final report</a>
</section>
</body></html>
""",
        encoding="utf-8",
    )

    result = validate_artifact_truth(deck_dir, checks={"html"})

    assert HTML_PATH_TEXT_VISUAL not in _codes(result)


def test_cli_writes_fail_report_when_negative_evidence_is_expected(tmp_path: Path) -> None:
    deck_dir = tmp_path / "bad-deck"
    deck_dir.mkdir()
    _write_bad_html_and_source(deck_dir)
    report_path = tmp_path / "artifact_truth_report.json"
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "presentations" / "validate_artifact_truth.py"),
            "--deck-dir",
            str(deck_dir),
            "--checks",
            "html,source",
            "--expect-status",
            "fail",
            "--report-out",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert any(finding["code"] == HTML_PATH_TEXT_VISUAL for finding in report["findings"])


def test_bad_topic4_fixture_is_rejected_as_production_evidence() -> None:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "presentation_artifact_truth" / "bad_topic4"

    result = validate_artifact_truth(
        fixture_dir,
        checks={"html", "source", "frames", "receipts", "mp4"},
        fixture_root=fixture_dir,
        current_artifacts=True,
    )

    assert result.status == "fail"
    assert SYNTHETIC_FIXTURE_VIDEO_USED_AS_PRODUCTION in _codes(result)
    assert VIDEO_FLAT_FRAME_LOW_ENTROPY in _codes(result)


# -- compiled-figure grounding (figure-compiler design spec §5) ---------------
#
# Two checks the design spec asks the validator for: numeric ticks imply ledger
# rows, and a schematic figure carries the badge and no ticks. Together they
# close the "illustrative chart ticks" pattern, which a prose-level
# forbidden-patterns list cannot see because the lie is in the drawing.

from conference_pipeline.figures.compiler import compile_figure  # noqa: E402
from conference_pipeline.figures.examples import EXAMPLES  # noqa: E402
from conference_pipeline.presentation.artifact_truth import (  # noqa: E402
    FIGURE_NUMERIC_TICKS_WITHOUT_LEDGER_ROWS,
    FIGURE_SCHEMATIC_CARRIES_NUMERIC_TICKS,
    FIGURE_SCHEMATIC_NOT_LABELLED_SCHEMATIC,
)


def _compiled_deck(root: Path, archetypes: tuple[str, ...]) -> Path:
    out = root / "assets" / "compiled"
    out.mkdir(parents=True, exist_ok=True)
    for name in archetypes:
        figure = compile_figure(EXAMPLES[name])
        figure.write(out / f"{name}.svg", out / f"{name}.manifest.json")
    return out


def _figure_codes(root: Path) -> list[str]:
    result = validate_artifact_truth(root, checks={"figures"}, production_mode=False)
    return [finding.code for finding in result.findings]


def test_honestly_compiled_figures_pass_the_grounding_checks(tmp_path: Path) -> None:
    _compiled_deck(tmp_path, ("line_curve", "schematic_curve", "flow", "bar_compare"))
    assert _figure_codes(tmp_path) == []


def test_numeric_ticks_without_ledger_rows_are_blocked(tmp_path: Path) -> None:
    """A measured axis must name the measurements it plots."""

    out = _compiled_deck(tmp_path, ("line_curve",))
    manifest_path = out / "line_curve.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["numeric_ticks"], "a line chart draws numeric ticks"
    payload["grounding"]["ledger_rows"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert FIGURE_NUMERIC_TICKS_WITHOUT_LEDGER_ROWS in _figure_codes(tmp_path)


def test_a_schematic_figure_stripped_of_its_badge_is_blocked(tmp_path: Path) -> None:
    """Without the badge an illustration of a claim reads as a measurement."""

    out = _compiled_deck(tmp_path, ("schematic_curve",))
    manifest_path = out / "schematic_curve.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["text_runs"] = [run for run in payload["text_runs"] if "schematic" not in run]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert FIGURE_SCHEMATIC_NOT_LABELLED_SCHEMATIC in _figure_codes(tmp_path)


def test_a_schematic_figure_that_grew_numeric_ticks_is_blocked(tmp_path: Path) -> None:
    out = _compiled_deck(tmp_path, ("schematic_curve",))
    manifest_path = out / "schematic_curve.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["numeric_ticks"] = ["0", "50", "100"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert FIGURE_SCHEMATIC_CARRIES_NUMERIC_TICKS in _figure_codes(tmp_path)


def test_a_box_and_arrow_mechanism_is_not_asked_for_a_badge(tmp_path: Path) -> None:
    """Only figures that could pass for measured data carry the badge; carrying
    it on every mechanism diagram would be noise on every slide."""

    _compiled_deck(tmp_path, ("flow", "nested_architecture", "hierarchy_tree"))
    assert _figure_codes(tmp_path) == []


def test_a_deck_with_no_compiled_figures_is_silent(tmp_path: Path) -> None:
    """The checks describe compiled figures; another route has nothing to read."""

    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    assert _figure_codes(tmp_path) == []
