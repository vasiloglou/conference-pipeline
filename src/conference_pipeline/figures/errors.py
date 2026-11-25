"""Failure modes of the figure compiler.

Every one of these is a *hard* failure: the compiler never emits a figure it
could not lay out, and never calls a model to get out of trouble.
"""

from __future__ import annotations


class FigureCompilerError(Exception):
    """Base class for every compiler failure."""


class FigureSpecError(FigureCompilerError):
    """The declarative spec is invalid (schema, budgets, or grounding)."""

    def __init__(self, message: str, *, code: str = "spec_invalid", field: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.field = field


class FigureLayoutError(FigureCompilerError):
    """The content cannot be laid out inside the canvas at the legibility floor."""

    def __init__(self, message: str, *, code: str = "layout_impossible") -> None:
        super().__init__(message)
        self.code = code


class FigureGeometryError(FigureCompilerError):
    """The emitted SVG failed the geometry gate after every remedy."""

    def __init__(self, message: str, *, findings: list[str] | None = None) -> None:
        super().__init__(message)
        self.findings = list(findings or [])
