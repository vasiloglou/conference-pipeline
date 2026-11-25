"""Strict loopback URL validation — v1.9 review HIGH-4 consolidation.

Three modules previously carried divergent ``_is_loopback`` helpers
with different strictness levels:

* ``indexing/ollama_embeddings.py``: strict (``ipaddress``-based,
  rejects ``0.0.0.0`` / ``::`` / lone ``"0"``).
* ``extractions/classifier.py``: medium (string-prefix on ``127.``
  plus ``localhost`` and ``::1``; only ``http`` / ``https`` schemes).
* ``model_governance/mlx_client.py``: loose (only literal
  ``127.0.0.1`` and ``localhost``).

A user with ``http://127.0.0.2:8080`` would be rejected by mlx_client
but accepted by the others. ``http://[::1]:11434`` would be accepted
by classifier and ollama but rejected by mlx_client. Three checks
means three places where a future regression could relax the
security perimeter and only one of the call sites would notice.

This module provides one strict implementation; all three sites
import it.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

__all__ = ["is_loopback_url"]


def is_loopback_url(url: str) -> bool:
    """Return ``True`` iff ``url`` resolves to a loopback host.

    Strict implementation:

    * ``localhost`` (any case) is always loopback.
    * Wildcard binds (``0.0.0.0``, ``::``, lone ``"0"``) are explicitly
      rejected — they bind to every interface, not loopback.
    * Everything else is parsed via :mod:`ipaddress`, which correctly
      classifies the entire ``127.0.0.0/8`` block, ``::1``, and
      IPv4-mapped IPv6 loopback (``::ffff:127.0.0.1``).
    * Empty / unparseable hosts → ``False``.

    Scheme-agnostic: callers that need to additionally restrict to
    ``http``/``https`` should layer that check on top.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").casefold()
    if not host:
        return False
    if host == "localhost":
        return True
    if host in {"0", "0.0.0.0", "::"}:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
