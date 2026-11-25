"""A second, advisory visual-QC pass: more recall, never a gate.

On the ICML 2026 canary decks Kimi K3 as a second vision reviewer found real text
defects the primary reviewer passed (a leaked ``\\texttt`` macro, a raw Markdown
link, labels cut mid-title) but failed 11 of 52 slides to the primary's 1, marking
style nits as blockers and misreading a datastore glyph as an overlap. That profile
suits a hint, not a verdict: its findings go to the Gate A reviewers, who check each
on the render, and into the deck record, and they never block a stage.

The pass is configured per conference (``presentation_review.advisory_visual_qc`` in
``config.yaml``, see ``review_settings``). With nothing configured nothing runs.
This module only builds the command and condenses the report; the stage runner
(the per-deck builder or the production chain) executes it, so each keeps its own
receipts and resume rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from conference_pipeline.workflow_runs import write_json_atomic

ADVISORY_REPORT = Path("assets/generated/visual_llm_qc_advisory_report.json")
ADVISORY_SUMMARY = Path("assets/generated/visual_llm_qc_advisory_summary.json")

HOW_TO_READ = (
    "ADVISORY second opinion from a different vision model, run for recall. It reports more "
    "than the primary reviewer and over-states severity; it has mistaken a datastore glyph "
    "for an overlap and a striped table for a split panel. Treat each finding as a hint: "
    "look at the slide, and report only what you can see, at the severity you judge."
)
_PROVIDER_FAILURE_CHECK = "vision_llm_visual_qc_parseable"


def advisory_command(
    *,
    workspace: Path,
    deck_id: str,
    presentation_type: str,
    screenshots_dir: Path,
    report_output: Path,
    provider: str,
    model: str,
) -> list[str]:
    """The ``inspect_rendered_slides_llm.py`` invocation for the advisory pass."""

    return [
        sys.executable,
        "scripts/presentations/inspect_rendered_slides_llm.py",
        "--workspace", str(workspace),
        "--presentation-id", deck_id,
        "--presentation-type", presentation_type,
        "--topic", deck_id,
        "--screenshots-dir", str(screenshots_dir),
        "--report-output", str(report_output),
        "--provider", provider,
        "--model", model,
    ]


def summarize_advisory_report(
    report: dict[str, Any] | None,
    *,
    primary: dict[str, Any] | None = None,
    max_description: int = 240,
    max_issues_per_slide: int = 3,
) -> dict[str, Any]:
    """Condense an advisory report to what a reviewer needs: the flagged slides and why."""

    base: dict[str, Any] = {"kind": "visual_qc_advisory_summary", "advisory": True, "how_to_read": HOW_TO_READ}
    if not isinstance(report, dict) or not isinstance(report.get("slides"), list):
        return {**base, "status": "unavailable", "flagged_slides": [], "unavailable_slides": []}
    # A pass that never judged a slide (no credential, no screenshots, a provider refusal) is
    # not a clean bill of health. It wrote a report with no slides and a blocker; calling that
    # "ok, 0 flagged" would tell Gate A the second opinion found nothing.
    if report.get("status") == "blocked" or not report["slides"]:
        errors = [e for e in report.get("errors") or [] if isinstance(e, dict)]
        reason = str(errors[0].get("description") or "") if errors else "the advisory pass judged no slides"
        return {
            **base,
            "status": "unavailable",
            "reason": reason[:300],
            "provider": report.get("provider"),
            "model": report.get("model"),
            "slides_reviewed": 0,
            "flagged_slides": [],
            "unavailable_slides": [],
        }
    primary_flagged = {
        int(slide["slide_number"])
        for slide in (primary or {}).get("slides", [])
        if isinstance(slide, dict) and slide.get("status") not in (None, "pass")
    }
    flagged: list[dict[str, Any]] = []
    unavailable: list[int] = []
    for slide in report["slides"]:
        if not isinstance(slide, dict):
            continue
        number = int(slide.get("slide_number") or 0)
        issues = [issue for issue in slide.get("issues") or [] if isinstance(issue, dict)]
        if any(issue.get("check_id") == _PROVIDER_FAILURE_CHECK for issue in issues):
            unavailable.append(number)
            continue
        if slide.get("status") == "pass":
            continue
        flagged.append(
            {
                "slide_number": number,
                "status": slide.get("status"),
                "also_flagged_by_primary": number in primary_flagged,
                "issues": [
                    {
                        "check_id": str(issue.get("check_id") or "other"),
                        "severity": str(issue.get("severity") or "warn"),
                        "description": str(issue.get("description") or "")[:max_description],
                    }
                    for issue in issues[:max_issues_per_slide]
                ],
            }
        )
    return {
        **base,
        "status": "ok" if not unavailable else "partial",
        "provider": report.get("provider"),
        "model": report.get("model"),
        "slides_reviewed": len(report["slides"]),
        "flagged_slide_count": len(flagged),
        "flagged_only_by_advisory": sum(1 for item in flagged if not item["also_flagged_by_primary"]),
        "flagged_slides": flagged,
        "unavailable_slides": unavailable,
    }


def write_advisory_summary(
    report_path: Path, summary_path: Path, *, primary_report_path: Path | None = None
) -> dict[str, Any]:
    """Read the advisory report (if it exists), write and return its summary."""

    def load(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    summary = summarize_advisory_report(load(report_path), primary=load(primary_report_path))
    write_json_atomic(summary_path, summary)
    return summary
