"""Action-token guard for local state-changing control-plane routes."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request


TOKEN_HEADER = "x-control-plane-token"


def install_action_token(app) -> str:
    token = secrets.token_urlsafe(32)
    app.state.action_token = token
    return token


def require_action_token(
    request: Request,
    token_header: str | None = None,
    token_form: str | None = None,
) -> None:
    expected = getattr(request.app.state, "action_token", None)
    provided = token_header or token_form
    if not expected or not provided or not secrets.compare_digest(str(provided), str(expected)):
        raise HTTPException(status_code=403, detail="invalid action token")
