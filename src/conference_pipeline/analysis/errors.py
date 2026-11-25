"""Errors raised by the analysis subpackage."""

from __future__ import annotations


class AnalysisContractError(Exception):
    """Base for fail-closed failures at analysis/podcast/deck boundaries.

    Deliberately NOT a ValueError - this keeps non-live mode fallbacks in
    `workflow_runs._commands_and_digest` (which swallow ValueError) from
    silently masking manifest validation failures. Fail-closed applies in
    live, dry-run, and mock modes equally (D-05).
    """


class ManifestValidationError(AnalysisContractError):
    """Raised when analysis_manifest.json is missing, malformed, or declares
    missing paths, or schema_version is incompatible."""


class StepError(AnalysisContractError):
    """Raised when a composable analysis step fails unrecoverably (e.g., LLM
    returns empty content, vector store write fails post-retry)."""
