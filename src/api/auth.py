"""API key authentication for the FrictionSim2D REST API.

Provides FastAPI dependencies for verifying API keys passed as
``X-API-Key`` headers. Read-only endpoints are public (no auth),
while write endpoints require a valid API key.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level database reference (set by server startup)
# ---------------------------------------------------------------------------

_db_instance = None


def set_db(db) -> None:
    """Set the shared :class:`~src.data.database.FrictionDB` instance."""
    global _db_instance  # noqa: PLW0603
    _db_instance = db


def get_db():
    """Return the shared database instance."""
    if _db_instance is None:
        raise RuntimeError("Database not initialised — call set_db() first")
    return _db_instance


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------


def _get_api_key_header(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Extract the API key from the request header."""
    return x_api_key


def require_api_key(
    api_key: Optional[str] = Depends(_get_api_key_header),
    db=Depends(get_db),
) -> str:
    """FastAPI dependency that enforces a valid API key.

    Returns the user name associated with the key.

    Raises:
        HTTPException 401: If the key is missing or invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    user_name = db.verify_api_key(api_key)
    if user_name is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )
    return user_name


def optional_api_key(
    api_key: Optional[str] = Depends(_get_api_key_header),
    db=Depends(get_db),
) -> Optional[str]:
    """FastAPI dependency that optionally verifies an API key.

    Returns the user name if a valid key is provided, ``None`` otherwise.
    Does **not** raise on missing/invalid keys (used for read endpoints
    where auth is optional).
    """
    if not api_key:
        return None
    return db.verify_api_key(api_key)
