"""Focused tests for API client header handling."""

import sys
from types import SimpleNamespace

from src.api.client import FrictionHTTPClient


class _DummyHttpxClient:  # pylint: disable=too-few-public-methods
    def __init__(self, timeout: float):
        self.timeout = timeout

    def close(self) -> None:
        return None


def _auth_headers(client: FrictionHTTPClient) -> dict[str, str]:
    return client._headers(auth=True)  # pylint: disable=protected-access


def test_constructor_strips_api_key_whitespace(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=_DummyHttpxClient))
    client = FrictionHTTPClient(api_key="  valid-key\n")

    assert _auth_headers(client) == {
        "Accept": "application/json",
        "X-API-Key": "valid-key",
    }
    client.close()

def test_constructor_normalizes_blank_api_key(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=_DummyHttpxClient))
    client = FrictionHTTPClient(api_key="   ")

    assert _auth_headers(client) == {"Accept": "application/json"}
    client.close()
