"""The figure manifest.

Reviews read this instead of scraping SVG text: what was drawn, which entities
appear verbatim, which numbers appear and which ledger rows they came from,
which glyphs were used, and the spec hash that reproduces the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.figures.spec import FigureSpec, content_labels

MANIFEST_VERSION = "figure-manifest/2"


@dataclass
class FigureManifest:
    figure_id: str
    slide_id: str
    archetype: str
    family: str
    claim: str
    takeaway: str
    grounding_kind: str
    ledger_rows: list[str] = field(default_factory=list)
    entities_rendered: list[str] = field(default_factory=list)
    entities_missing: list[str] = field(default_factory=list)
    numbers_rendered: list[str] = field(default_factory=list)
    numeric_ticks: list[str] = field(default_factory=list)
    glyphs_used: list[str] = field(default_factory=list)
    text_runs: list[str] = field(default_factory=list)
    labels_drawn: list[str] = field(default_factory=list)
    labels_relocated: list[dict[str, str]] = field(default_factory=list)
    labels_dropped: list[dict[str, str]] = field(default_factory=list)
    labels_ignored: list[str] = field(default_factory=list)
    lint_findings: list[dict[str, str]] = field(default_factory=list)
    spec_hash: str = ""
    svg_sha256: str = ""
    compiler_version: str = ""
    layout_attempt: int = 0
    layout_options: dict[str, Any] = field(default_factory=dict)
    gate_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "figure_id": self.figure_id,
            "slide_id": self.slide_id,
            "archetype": self.archetype,
            "family": self.family,
            "claim": self.claim,
            "takeaway": self.takeaway,
            "grounding": {"kind": self.grounding_kind, "ledger_rows": self.ledger_rows},
            "entities_rendered": self.entities_rendered,
            "entities_missing": self.entities_missing,
            "numbers_rendered": self.numbers_rendered,
            "numeric_ticks": self.numeric_ticks,
            "glyphs_used": self.glyphs_used,
            "text_runs": self.text_runs,
            "lint_findings": self.lint_findings,
            "labels": {
                "drawn": self.labels_drawn,
                "relocated": self.labels_relocated,
                "dropped": self.labels_dropped,
                "ignored": self.labels_ignored,
            },
            "spec_hash": self.spec_hash,
            "svg_sha256": self.svg_sha256,
            "compiler_version": self.compiler_version,
            "layout": {"attempt": self.layout_attempt, "options": self.layout_options},
            "gate_findings": self.gate_findings,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def entity_present(entity: str, text_runs: list[str]) -> bool:
    """The semantic reviewer's presence rule, applied to the rendered runs.

    Exact phrase, or all distinctive tokens, or >= 60% of them for entities of
    three or more tokens -- the same rule ``review_programmatic_figures`` uses,
    so a compiled figure that passes here cannot fail there.
    """

    import re

    ignored = {"source", "slide", "figure", "visual", "show", "showing", "depict"}

    def tokens_of(value: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", value.lower())
            if token not in ignored
        }

    normalized = str(entity).strip().lower()
    if not normalized:
        return True
    haystack = " ".join(text_runs).lower()
    if normalized in haystack:
        return True
    wanted = tokens_of(normalized)
    have = tokens_of(haystack)
    if wanted and wanted.issubset(have):
        return True
    if len(wanted) >= 3:
        return sum(1 for token in wanted if token in have) / len(wanted) >= 0.6
    return False


def numbers_in_runs(text_runs: list[str]) -> list[str]:
    import re

    found: list[str] = []
    for run in text_runs:
        for match in re.findall(r"-?\d+(?:\.\d+)?\s*(?:%|x|×)?", run):
            value = match.strip()
            if value and value not in found:
                found.append(value)
    return found


def account_for_labels(
    spec: FigureSpec, outcomes: list[Any]
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Reconcile what the spec asked to draw against what the layout did.

    Returns (drawn, relocated, dropped, ignored). A label counts as drawn when
    it is CONTAINED in some rendered run, not equal to one: layouts routinely
    compose "label (unit)", "label - 55%", "caption (licence)", and an equality
    test would report those as missing.

    ``ignored`` is the interesting bucket: content the spec carries under a key
    this layout never reads. Nothing is broken on the canvas, but it is a
    disagreement between the spec's author and the layout, and it is exactly
    where a misused archetype shows up -- so the compiler refuses on it rather
    than record it.
    """

    expected = content_labels(spec.content)
    drawn_sources = [
        " ".join(str(outcome.text).split()).casefold()
        for outcome in outcomes
        if outcome.status in {"drawn", "relocated"}
    ]
    relocated = [outcome.to_dict() for outcome in outcomes if outcome.status == "relocated"]
    dropped = [outcome.to_dict() for outcome in outcomes if outcome.status == "dropped"]
    handled = {
        " ".join(str(outcome.text).split()).casefold()
        for outcome in outcomes
        if outcome.status in {"relocated", "dropped"}
    }

    drawn: list[str] = []
    ignored: list[str] = []
    for label in expected:
        needle = label.casefold()
        if any(needle in source for source in drawn_sources):
            drawn.append(label)
        elif needle in handled:
            continue
        else:
            ignored.append(label)
    return drawn, relocated, dropped, ignored


def build_manifest(
    spec: FigureSpec,
    *,
    text_runs: list[str],
    numeric_ticks: list[str] | None = None,
    glyphs_used: list[str],
    label_outcomes: list[Any] | None = None,
    lint_findings: list[dict[str, str]] | None = None,
    svg_sha256: str,
    compiler_version: str,
    layout_attempt: int,
    layout_options: dict[str, Any],
    gate_findings: list[str],
) -> FigureManifest:
    rendered = [entity for entity in spec.required_entities if entity_present(entity, text_runs)]
    missing = [entity for entity in spec.required_entities if entity not in rendered]
    drawn, relocated, dropped, ignored = account_for_labels(spec, list(label_outcomes or []))
    return FigureManifest(
        figure_id=spec.figure_id,
        slide_id=spec.slide_id,
        archetype=spec.archetype,
        family=spec.family,
        claim=spec.claim,
        takeaway=spec.takeaway,
        grounding_kind=spec.grounding.kind,
        ledger_rows=list(spec.grounding.ledger_rows),
        entities_rendered=rendered,
        entities_missing=missing,
        numbers_rendered=numbers_in_runs(text_runs),
        numeric_ticks=list(numeric_ticks or []),
        glyphs_used=list(glyphs_used),
        text_runs=list(text_runs),
        labels_drawn=drawn,
        labels_relocated=relocated,
        labels_dropped=dropped,
        labels_ignored=ignored,
        lint_findings=list(lint_findings or []),
        spec_hash=spec.spec_hash(),
        svg_sha256=svg_sha256,
        compiler_version=compiler_version,
        layout_attempt=layout_attempt,
        layout_options=layout_options,
        gate_findings=list(gate_findings),
    )
