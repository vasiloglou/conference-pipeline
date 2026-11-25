"""The compiler pipeline: spec in, SVG and manifest out.

    spec -> validate -> shape text -> layout -> style -> geometry gate -> emit

No model call, no network, no randomness. The same spec and the same compiler
version always produce byte-identical SVG, so a figure can be cached by spec
hash and a rebuild is free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.figures import tokens
from conference_pipeline.figures.errors import (
    FigureGeometryError,
    FigureLayoutError,
    FigureSpecError,
)
from conference_pipeline.figures.frame import draw_frame
from conference_pipeline.figures.gate import remedies, run_gate
from conference_pipeline.figures.layouts import LAYOUTS
from conference_pipeline.figures.layouts._common import LayoutOptions
from conference_pipeline.figures.lint import lint_spec
from conference_pipeline.figures.manifest import FigureManifest, account_for_labels, build_manifest
from conference_pipeline.figures.spec import (
    FigureSpec,
    budget_findings,
    load_spec,
    validate_spec,
)
from conference_pipeline.figures.svgdoc import SvgDoc

# Bumped whenever a layout's output for an unchanged spec changes, because the
# deck cache keys on this and the spec hash: 1.1.0 is the arrow-run floor, the
# scene router, glyph-named primitives and content-sized contrast panels.
COMPILER_VERSION = "figure-compiler/1.8.6"


@dataclass
class CompiledFigure:
    """One compiled figure: the SVG text and everything a review needs."""

    spec: FigureSpec
    svg: str
    manifest: FigureManifest
    attempt: int = 0
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def svg_sha256(self) -> str:
        return hashlib.sha256(self.svg.encode("utf-8")).hexdigest()

    def write(self, svg_path: Path, manifest_path: Path | None = None) -> tuple[Path, Path | None]:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(self.svg, encoding="utf-8")
        written_manifest = None
        if manifest_path is not None:
            written_manifest = self.manifest.write(manifest_path)
        return svg_path, written_manifest


def _render_once(spec: FigureSpec, options: LayoutOptions) -> SvgDoc:
    doc = SvgDoc(title=spec.title or spec.claim, description=spec.claim)
    frame = draw_frame(doc, spec)
    layout = LAYOUTS.get(spec.archetype)
    if layout is None:  # pragma: no cover - validation rejects unknown archetypes first
        raise FigureSpecError(f"no layout registered for archetype '{spec.archetype}'")
    layout(doc, spec, frame.plot, options)
    return doc


def compile_figure(spec_payload: dict[str, Any] | FigureSpec) -> CompiledFigure:
    """Compile one figure spec, applying the remedy ladder until the gate passes."""

    spec = (spec_payload if isinstance(spec_payload, FigureSpec)
            else load_spec(spec_payload, mode="compile"))
    errors = validate_spec(spec, mode="compile")
    if errors:
        raise FigureSpecError("; ".join(errors[:6]))
    # Advisory structure findings: judgements about what a reader will conclude,
    # recorded rather than refused. By compile time the only remedy is
    # re-authoring the slide, which is the operator's call.
    advisories = [
        finding.to_dict() for finding in lint_spec(spec) if finding.severity == "warn"
    ]
    # Over the published budget, and drawn anyway if it draws. The budget is a
    # floor measured on one worked shape per archetype, so a spec above it may
    # still fit; refusing one that demonstrably does buys nothing, and the
    # geometry gate still stops the ones that do not. Recorded either way, so
    # the manifest says the figure is outside the guaranteed envelope.
    advisories.extend(
        {"code": "over_budget", "severity": "warn", "message": message}
        for message in budget_findings(spec)
    )

    ladder = remedies(spec.archetype)
    last_findings: list[str] = []
    layout_error: FigureLayoutError | None = None
    # The FIRST attempt's outcome is the diagnosis. Reporting the last rung's
    # sent a six-layer chain's refusal as "needs 789px of height" -- the flipped
    # layout's complaint -- when the unflipped one had missed by 8px of width.
    first_failure: str = ""
    for attempt, patch in enumerate(ladder):
        options = LayoutOptions(**patch)  # type: ignore[arg-type]
        try:
            doc = _render_once(spec, options)
        except FigureLayoutError as error:
            layout_error = layout_error or error
            first_failure = first_failure or str(error)
            continue
        svg = doc.to_svg()
        findings = run_gate(doc, svg)
        if not findings:
            # Content the spec carries under a key this layout never read is
            # content nobody will see, and no ladder rung can change which keys
            # a layout reads. For a while this was a manifest line and not a
            # refusal; a spec declaring three fusion strategies shipped as one
            # box while its manifest named the three lost labels every build.
            _, _, _, ignored = account_for_labels(spec, list(doc.label_outcomes))
            if ignored:
                raise FigureGeometryError(
                    f"'{spec.figure_id}' ({spec.archetype}) carries content the layout never "
                    f"drew: {', '.join(repr(label) for label in ignored[:4])}"
                    f"{' ...' if len(ignored) > 4 else ''}; the spec and the archetype disagree, "
                    "so re-author under an archetype that reads those keys",
                    findings=[f"error:content_ignored: {label!r} is in the spec and not in the figure"
                              for label in ignored],
                )
            manifest = build_manifest(
                spec,
                text_runs=doc.rendered_text,
                numeric_ticks=doc.numeric_ticks,
                glyphs_used=doc.glyphs_used,
                label_outcomes=doc.label_outcomes,
                lint_findings=advisories,
                svg_sha256=hashlib.sha256(svg.encode("utf-8")).hexdigest(),
                compiler_version=COMPILER_VERSION,
                layout_attempt=attempt,
                layout_options=dict(patch),
                gate_findings=[],
            )
            return CompiledFigure(spec=spec, svg=svg, manifest=manifest, attempt=attempt, options=dict(patch))
        last_findings = [str(finding) for finding in findings]
        if not first_failure:
            first_failure = "; ".join(last_findings[:2])

    if layout_error is not None and not last_findings:
        raise layout_error
    raise FigureGeometryError(
        f"'{spec.figure_id}' ({spec.archetype}) still fails the geometry gate after "
        f"{len(ladder)} deterministic remedies; the first attempt reported: {first_failure}",
        findings=last_findings,
    )


def compile_spec_file(path: Path) -> CompiledFigure:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return compile_figure(payload)


__all__ = ["COMPILER_VERSION", "CompiledFigure", "compile_figure", "compile_spec_file"]
