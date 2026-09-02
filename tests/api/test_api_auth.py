"""Focused tests for API authentication helpers."""

from __future__ import annotations

# pylint: disable=wrong-import-position

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from src.api.auth import _get_api_key_header, require_api_key


def test_get_api_key_header_strips_whitespace() -> None:
    assert _get_api_key_header("  valid-key  ") == "valid-key"


def test_require_api_key_uses_normalized_header_value() -> None:
    db = MagicMock()
    db.verify_api_key.return_value = "alice"

    user = require_api_key(api_key=_get_api_key_header("  valid-key  "), db=db)

    assert user == "alice"
    db.verify_api_key.assert_called_once_with("valid-key")
