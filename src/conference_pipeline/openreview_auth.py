"""Optional OpenReview authentication for challenge-gated endpoints.

Since roughly 2026-07, OpenReview serves an anti-bot "challenge" (HTTP 403
``ChallengeRequiredError``, "Verifying your browser") to unauthenticated
non-browser clients on the notes API (``api2.openreview.net/notes``) and on the
PDF endpoints (``openreview.net/pdf`` and ``api2.openreview.net/attachment``).
A Bearer token minted from account credentials bypasses the challenge.

Credentials are read only from the environment (``OPENREVIEW_USERNAME`` /
``OPENREVIEW_PASSWORD``). This module never logs, prints, or returns secret
values; receipts get a names-only ``auth_mode`` label plus the env-var names
that were used.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

API2_BASE = "https://api2.openreview.net"
LOGIN_URL = f"{API2_BASE}/login"
ATTACHMENT_URL_TEMPLATE = API2_BASE + "/attachment?id={note_id}&name=pdf"
USERNAME_ENV = "OPENREVIEW_USERNAME"
PASSWORD_ENV = "OPENREVIEW_PASSWORD"

# Accepted values for a CLI ``--openreview-auth`` flag.
AUTH_MODES = ("auto", "required", "off")

# A source-discovery run may need several authenticated, non-mutating
# OpenReview requests in one process (multiple source families plus the
# declared readiness probe).  Reusing the short-lived bearer session avoids
# repeatedly logging in and triggering OpenReview's login rate limit.  This is
# process-local only and is never serialised, logged, or exposed by receipts.
_CACHED_AUTH: "OpenReviewAuth | None" = None


class OpenReviewAuthError(RuntimeError):
    """Raised when authentication is required but cannot be established."""


class OpenReviewAuth:
    """Carries an optional Bearer token and produces request headers.

    Never stringifies or exposes the token; only ``mode`` (a label) and
    ``env_names`` (the env-var names, not values) are safe for receipts.
    """

    def __init__(self, token: str | None, mode: str, env_names: list[str]) -> None:
        self._token = token
        self.mode = mode  # "authenticated" | "unauthenticated"
        self.env_names = env_names

    @property
    def authenticated(self) -> bool:
        return self._token is not None

    def headers(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Return request headers, adding ``Authorization`` when authenticated."""
        merged = dict(base or {})
        if self._token is not None:
            merged["Authorization"] = f"Bearer {self._token}"
        return merged

    def receipt_fields(self) -> dict[str, Any]:
        """Return names-only auth provenance for stage receipts."""
        return {
            "openreview_auth_mode": self.mode,
            "openreview_auth_env_names": list(self.env_names),
            "secret_values_recorded": False,
        }

    def __repr__(self) -> str:  # never leak the token
        return f"OpenReviewAuth(mode={self.mode!r}, authenticated={self.authenticated})"


def load_openreview_auth(
    user_agent: str,
    *,
    mode: str = "auto",
    timeout: float = 30.0,
    force_refresh: bool = False,
) -> OpenReviewAuth:
    """Build an :class:`OpenReviewAuth` from environment credentials.

    ``mode``:
    - ``"auto"`` (default): authenticate if both env vars are set; otherwise
      return an unauthenticated instance.
    - ``"required"``: raise :class:`OpenReviewAuthError` if credentials are
      missing or login fails.
    - ``"off"``: never authenticate (returns an unauthenticated instance).
    """
    global _CACHED_AUTH

    if mode not in AUTH_MODES:
        raise ValueError(f"mode must be one of {AUTH_MODES}, got {mode!r}")
    if mode == "off":
        return OpenReviewAuth(None, "unauthenticated", [])

    if not force_refresh and _CACHED_AUTH is not None and _CACHED_AUTH.authenticated:
        return _CACHED_AUTH

    user = os.environ.get(USERNAME_ENV)
    password = os.environ.get(PASSWORD_ENV)
    if not (user and password):
        if mode == "required":
            raise OpenReviewAuthError(
                "OpenReview credentials required but missing: set "
                f"{USERNAME_ENV} and {PASSWORD_ENV} in the environment."
            )
        return OpenReviewAuth(None, "unauthenticated", [])

    payload = json.dumps({"id": user, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        LOGIN_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": user_agent},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            token = json.load(response).get("token")
    except Exception as exc:  # noqa: BLE001 - surface a clean, secret-free error
        raise OpenReviewAuthError(
            f"OpenReview login failed for env user {USERNAME_ENV} "
            f"({exc.__class__.__name__})."
        ) from exc
    if not token:
        raise OpenReviewAuthError("OpenReview login returned no token.")
    authenticated = OpenReviewAuth(token, "authenticated", sorted([USERNAME_ENV, PASSWORD_ENV]))
    _CACHED_AUTH = authenticated
    return authenticated
