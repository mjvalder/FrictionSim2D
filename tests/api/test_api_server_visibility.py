"""Focused visibility tests for the API server public read endpoints."""

from __future__ import annotations

# pyright: reportMissingImports=false
# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api.auth import set_db
from src.api.server import app


def _make_db(rows):
    db = MagicMock()
    db.query.return_value = pd.DataFrame(rows)
    db.verify_api_key.return_value = "alice"
    return db


@pytest.fixture(name="client_with_mixed_visibility")
def _client_with_mixed_visibility_fixture():
    rows = [
        {"id": 1, "material": "h-MoS2", "simulation_type": "afm", "status": "published", "mean_cof": 0.1},
        {"id": 2, "material": "h-WS2", "simulation_type": "afm", "status": "staged", "mean_cof": 0.2},
    ]
    db = _make_db(rows)
    set_db(db)
    return TestClient(app)


def test_public_results_hide_non_published_rows(client_with_mixed_visibility) -> None:
    response = client_with_mixed_visibility.get("/results")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert [row["id"] for row in body["results"]] == [1]


def test_public_result_lookup_hides_non_published_rows(client_with_mixed_visibility) -> None:
    response = client_with_mixed_visibility.get("/results/2")

    assert response.status_code == 404


def test_authenticated_results_can_see_non_published_rows(client_with_mixed_visibility) -> None:
    response = client_with_mixed_visibility.get("/results", headers={"X-API-Key": "valid-key"})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2


def test_curator_can_request_non_published_status(
    client_with_mixed_visibility,
) -> None:
    response = client_with_mixed_visibility.get(
        "/results",
        params={"status": "staged"},
        headers={"X-API-Key": "valid-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert [row["id"] for row in body["results"]] == [2]