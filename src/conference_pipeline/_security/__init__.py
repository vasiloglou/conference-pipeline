"""Internal security helpers for the conference pipeline.

Single-source-of-truth implementations for security-perimeter checks
that were previously duplicated across modules with diverging
strictness (v1.9 review HIGH-4).
"""

from .loopback import is_loopback_url

__all__ = ["is_loopback_url"]
