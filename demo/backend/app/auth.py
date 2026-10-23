"""Bearer tokens minted by the frontend's ``/api/auth/token`` for signed-in users.

A token is ``base64url(json).hex(hmac-sha256)``, signed with ``DEMO_TOKEN_SECRET``,
which the Vercel frontend and this app share. Its payload names the signed-in
Google account, the audience ``trustcall-demo-api`` and a short expiry.

Without ``DEMO_TOKEN_SECRET`` the API is open, which is what local development
wants; every deployment must set it.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Annotated, Any

from fastapi import Header, HTTPException

__all__ = ["AUDIENCE", "TOKEN_SECRET_ENV", "require_user", "verify_token"]

AUDIENCE = "trustcall-demo-api"
TOKEN_SECRET_ENV = "DEMO_TOKEN_SECRET"

logger = logging.getLogger(__name__)


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={"code": "unauthorized", "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_token(token: str, secret: str, now: float | None = None) -> str | None:
    """Return the email in a valid token, or None.

    Args:
        token: ``<base64url payload>.<hex signature>``.
        secret: The shared HMAC key.
        now: The current Unix time; defaults to ``time.time()``.

    Returns:
        The signed-in email, or None when the token is forged, malformed, expired
        or meant for another audience.
    """
    payload, dot, signature = token.rpartition(".")
    if not dot or not payload:
        return None
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padded = payload + "=" * (-len(payload) % 4)
        data: Any = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    email, exp = data.get("email"), data.get("exp")
    if data.get("aud") != AUDIENCE or not isinstance(email, str):
        return None
    if not isinstance(exp, int | float) or exp < (
        now if now is not None else time.time()
    ):
        return None
    return email


def require_user(
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    """FastAPI dependency: the signed-in email, or 401.

    Returns None, admitting everyone, when ``DEMO_TOKEN_SECRET`` is unset.
    """
    secret = os.environ.get(TOKEN_SECRET_ENV)
    if not secret:
        return None
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized("Sign in to the demo to use the API.")
    email = verify_token(token.strip(), secret)
    if email is None:
        raise _unauthorized("The sign-in token is invalid or expired.")
    return email


if not os.environ.get(TOKEN_SECRET_ENV):
    logger.warning(
        "%s is not set: the API accepts unauthenticated calls.", TOKEN_SECRET_ENV
    )
