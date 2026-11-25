"""Errors raised by the presentation contract package."""

from __future__ import annotations


class PresentationContractError(Exception):
    """Base for fail-closed presentation contract failures.

    Deliberately NOT a ValueError so non-live fallback surfaces cannot silently
    swallow contract-validation failures.
    """


class PresentationManifestValidationError(PresentationContractError):
    """Raised when presentation_manifest.json is missing, malformed, or invalid."""


class PresentationValidationError(PresentationContractError):
    """Raised when a structural validation check fails deterministically.

    Deliberately NOT a ValueError so non-live fallback surfaces cannot
    silently swallow validation failures. Per Phase 25 CONTEXT.md D-09/D-15.
    """
