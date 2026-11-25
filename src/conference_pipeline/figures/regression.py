"""Vision-judge findings, recorded against the archetype that produced them.

The compiler route has no redraw loop: the same spec compiles to byte-identical
SVG, so paying a model to look at a figure and then recompiling it can only
reproduce the same pixels. The pixel judge still earns its keep, but as a
*regression signal* rather than a retry trigger -- if `swimlane_sequence` draws
findings on four decks running, the fix belongs in the layout code, not in one
slide.

This module does the join the runner never had: a vision-QC report indexes its
findings by slide NUMBER, the render spec turns a number into a slide id, and a
figure manifest turns a slide id into an archetype. Tallying the three together
gives a per-archetype health record that is comparable across decks and over
time.

Nothing here calls a provider or mutates a figure.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

REGRESSION_VERSION = "figure-archetype-regression/1"

# Findings that describe the deck contract rather than the figure. A slide-title
# defect is a blueprint problem; recording it against `flow` would blame the
# layout for someone else's text.
NON_FIGURE_FINDING_MARKERS = ("slide title",)
COUNTED_SEVERITIES = frozenset({"blocker", "warn"})


@dataclass
class ArchetypeHealth:
    """One archetype's standing against the vision judge."""

    archetype: str
    figures_reviewed: int = 0
    figures_with_findings: int = 0
    finding_counts: dict[str, int] = field(default_factory=dict)
    example_figures: list[str] = field(default_factory=list)

    @property
    def finding_rate(self) -> float:
        if not self.figures_reviewed:
            return 0.0
        return self.figures_with_findings / self.figures_reviewed

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype": self.archetype,
            "figures_reviewed": self.figures_reviewed,
            "figures_with_findings": self.figures_with_findings,
            "finding_rate": round(self.finding_rate, 4),
            "finding_counts": dict(sorted(self.finding_counts.items())),
            "example_figures": self.example_figures[:5],
        }


def _slide_ids_by_number(render_spec: Mapping[str, Any]) -> dict[int, str]:
    """Slide number (1-based, as the judge counts) to slide id."""

    slides = render_spec.get("slides")
    if not isinstance(slides, list):
        return {}
    mapping: dict[int, str] = {}
    for index, slide in enumerate(slides, start=1):
        if isinstance(slide, dict):
            slide_id = str(slide.get("slide_id") or "").strip()
            if slide_id:
                mapping[index] = slide_id
    return mapping


def _archetype_by_slide(manifests: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, str]]:
    """Slide id to (archetype, figure_id)."""

    mapping: dict[str, tuple[str, str]] = {}
    for manifest in manifests:
        slide_id = str(manifest.get("slide_id") or "").strip()
        archetype = str(manifest.get("archetype") or "").strip()
        if slide_id and archetype:
            mapping[slide_id] = (archetype, str(manifest.get("figure_id") or ""))
    return mapping


def _figure_findings(entry: Mapping[str, Any]) -> list[str]:
    """The judge's figure-relevant check ids for one slide."""

    found: list[str] = []
    for issue in entry.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        if str(issue.get("severity") or "").strip().lower() not in COUNTED_SEVERITIES:
            continue
        description = str(issue.get("description") or "").lower()
        if any(marker in description for marker in NON_FIGURE_FINDING_MARKERS):
            continue
        check_id = str(issue.get("check_id") or "unnamed_check").strip() or "unnamed_check"
        found.append(check_id)
    return found


def sample_figures(
    manifests: list[Mapping[str, Any]], *, size: int | None = None, seed: str = ""
) -> list[Mapping[str, Any]]:
    """A deterministic, archetype-stratified sample of a deck's figures.

    §7 asks for the judge on "a sample (or on demand)". Sampling one figure per
    archetype before taking a second of any is what makes a small sample
    informative: twelve looks spread over twelve archetypes says more about the
    compiler than twelve looks at `flow`.

    Deterministic by construction -- ordering comes from a hash of the figure id
    and ``seed``, never from a PRNG -- so the same deck samples the same figures
    until its figures change.
    """

    if size is None or size >= len(manifests):
        return list(manifests)
    if size <= 0:
        return []

    def rank(manifest: Mapping[str, Any]) -> str:
        key = f"{seed}:{manifest.get('figure_id') or manifest.get('slide_id') or ''}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for manifest in manifests:
        buckets.setdefault(str(manifest.get("archetype") or "unknown"), []).append(manifest)
    for archetype in buckets:
        buckets[archetype].sort(key=rank)

    chosen: list[Mapping[str, Any]] = []
    depth = 0
    while len(chosen) < size:
        added = False
        for archetype in sorted(buckets):
            if depth < len(buckets[archetype]):
                chosen.append(buckets[archetype][depth])
                added = True
                if len(chosen) == size:
                    break
        if not added:
            break
        depth += 1
    return chosen


def archetype_regression(
    *,
    visual_qc_report: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    manifests: list[Mapping[str, Any]],
    sample_size: int | None = None,
    seed: str = "",
) -> dict[str, Any]:
    """Tally vision-judge findings per archetype. Never triggers a retry."""

    reviewed = sample_figures(manifests, size=sample_size, seed=seed)
    slide_ids = _slide_ids_by_number(render_spec)
    by_slide = _archetype_by_slide(reviewed)

    health: dict[str, ArchetypeHealth] = {}
    for manifest in reviewed:
        archetype = str(manifest.get("archetype") or "").strip()
        if not archetype:
            continue
        record = health.setdefault(archetype, ArchetypeHealth(archetype=archetype))
        record.figures_reviewed += 1

    unmatched: list[int] = []
    for entry in visual_qc_report.get("slides") or []:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry.get("slide_number") or 0)
        except (TypeError, ValueError):
            continue
        slide_id = slide_ids.get(number, "")
        if not slide_id or slide_id not in by_slide:
            if entry.get("status") == "fail":
                unmatched.append(number)
            continue
        checks = _figure_findings(entry)
        if not checks:
            continue
        archetype, figure_id = by_slide[slide_id]
        record = health.setdefault(archetype, ArchetypeHealth(archetype=archetype))
        record.figures_with_findings += 1
        for check_id in checks:
            record.finding_counts[check_id] = record.finding_counts.get(check_id, 0) + 1
        if figure_id and figure_id not in record.example_figures:
            record.example_figures.append(figure_id)

    totals = Counter()
    for record in health.values():
        for check_id, count in record.finding_counts.items():
            totals[check_id] += count

    return {
        "regression_version": REGRESSION_VERSION,
        "figures_available": len(manifests),
        "figures_reviewed": len(reviewed),
        "sampled": sample_size is not None and sample_size < len(manifests),
        "figures_with_findings": sum(r.figures_with_findings for r in health.values()),
        "finding_total": sum(totals.values()),
        "finding_counts": dict(sorted(totals.items())),
        "archetypes": [health[name].to_dict() for name in sorted(health)],
        "unmatched_failing_slides": sorted(unmatched),
        "drives_retry": False,
    }


def load_manifests(directory: Path) -> list[dict[str, Any]]:
    """Every figure manifest under ``directory``, in a stable order."""

    manifests: list[dict[str, Any]] = []
    if not directory.is_dir():
        return manifests
    for path in sorted(directory.rglob("*.manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            manifests.append(payload)
    return manifests


__all__ = [
    "REGRESSION_VERSION",
    "ArchetypeHealth",
    "archetype_regression",
    "load_manifests",
    "sample_figures",
]
