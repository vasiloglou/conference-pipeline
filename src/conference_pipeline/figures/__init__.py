"""Deterministic figure compiler.

The model authors *what* a figure claims (a validated declarative spec); this
package decides *where* every mark goes. Layout is computed from measured text,
so a compiled figure cannot overflow a box, overlap a label, or leave a
connector dangling -- the defects that the LLM-authored absolute-coordinate
drawsvg path produced on 57 of the CVPR 2026 canary's findings and could not
converge away in two redraw rounds.

Design spec: ``docs/design/figure-compiler-design-spec.md``.
"""

from __future__ import annotations

from conference_pipeline.figures.compiler import CompiledFigure, compile_figure
from conference_pipeline.figures.errors import (
    FigureCompilerError,
    FigureGeometryError,
    FigureLayoutError,
    FigureSpecError,
)
from conference_pipeline.figures.spec import ARCHETYPES, FigureSpec, load_spec

__all__ = [
    "ARCHETYPES",
    "CompiledFigure",
    "FigureCompilerError",
    "FigureGeometryError",
    "FigureLayoutError",
    "FigureSpec",
    "compile_figure",
    "load_spec",
]
