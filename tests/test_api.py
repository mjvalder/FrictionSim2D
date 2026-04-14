"""Tests for the FrictionSim2D REST API and HTTP client.

Uses FastAPI's TestClient (backed by httpx) so no real server or database
is needed — all database interactions go through a mock ``FrictionDB``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fastapi.testclient import TestClient

from src.api.auth import set_db
from src.api.server import app
from src.data.models import ResultRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_ROW = {
    'id': 1,
    'uploaded_at': '2025-01-01T00:00:00+00:00',
    'uploader': 'alice',
    'material': 'h-MoS2',
    'simulation_type': 'afm',
    'layers': 1,
    'size_x': 20.0,
    'size_y': 20.0,
    'stack_type': None,
    'force_nN': 10.0,
    'pressure_gpa': None,
    'scan_angle': 0.0,
    'scan_speed': 1.0,
    'temperature': 300.0,
    'tip_material': 'Si',
    'tip_radius': 5.0,
    'substrate_material': None,
    'substrate_amorphous': False,
    'potential_type': 'sw',
    'mean_cof': 0.05,
    'std_cof': 0.01,
    'mean_lf': 0.5,
    'std_lf': 0.1,
    'mean_nf': 10.0,
    'std_nf': 0.5,
    'mean_lfx': 0.3,
    'std_lfx': 0.05,
    'mean_lfy': 0.4,
    'std_lfy': 0.06,
    'ntimesteps': 500000,
    'time_series_hash': 'abc123',
    'is_complete': True,
    'status': 'published',
    'notes': None,
    'metadata': None,
    'data_url': None,
}


def _make_mock_db(rows: Optional[List[Dict[str, Any]]] = None):
    """Create a mock FrictionDB with standard methods."""
    if rows is None:
        rows = [_SAMPLE_ROW]

    mock = MagicMock()
    mock.query.return_value = pd.DataFrame(rows)
    mock.get_statistics.return_value = {
        'total_rows': len(rows),
        'by_material': {'h-MoS2': len(rows)},
        'by_type': {'afm': len(rows)},
        'cof_global_mean': 0.05,
    }
    mock.upload_result.return_value = 42
    mock.verify_api_key.return_value = 'alice'
    mock.publish.return_value = True
    mock.reject.return_value = True

    # validate_staged returns a mock ValidationResult
    vr = MagicMock()
    vr.is_valid = True
    vr.errors = []
    vr.warnings = []
    mock.validate_staged.return_value = vr

    return mock


@pytest.fixture
def mock_db():
    """Provide a mock DB and register it with the auth module."""
    db = _make_mock_db()
    set_db(db)
    yield db


@pytest.fixture
def client(mock_db):
    """Provide a FastAPI test client with a mock DB."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /results
# ---------------------------------------------------------------------------


class TestQueryResults:

    def test_query_all(self, client, mock_db):
        resp = client.get("/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["results"][0]["material"] == "h-MoS2"
        mock_db.query.assert_called_once()

    def test_query_with_filters(self, client, mock_db):
        resp = client.get("/results", params={
            "material": "h-WS2",
            "layers": 2,
            "limit": 10,
        })
        assert resp.status_code == 200
        call_kwargs = mock_db.query.call_args
        assert call_kwargs.kwargs.get("material") == "h-WS2" or \
               (call_kwargs.args and call_kwargs.args[0] == "h-WS2") or \
               call_kwargs[1].get("material") == "h-WS2"

    def test_query_empty(self, client, mock_db):
        mock_db.query.return_value = pd.DataFrame()
        resp = client.get("/results")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0
        assert resp.json()["results"] == []

    def test_query_with_force_range(self, client, mock_db):
        resp = client.get("/results", params={
            "force_min": 5.0,
            "force_max": 15.0,
        })
        assert resp.status_code == 200

    def test_query_with_status_filter(self, client, mock_db):
        resp = client.get("/results", params={"status": "published"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /results/{id}
# ---------------------------------------------------------------------------


class TestGetResult:

    def test_get_existing(self, client, mock_db):
        resp = client.get("/results/1")
        assert resp.status_code == 200
        assert resp.json()["id"] == 1
        assert resp.json()["material"] == "h-MoS2"

    def test_get_nonexistent(self, client, mock_db):
        mock_db.query.return_value = pd.DataFrame([_SAMPLE_ROW])
        resp = client.get("/results/999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /statistics
# ---------------------------------------------------------------------------


class TestStatistics:

    def test_get_statistics(self, client, mock_db):
        resp = client.get("/statistics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_rows"] == 1
        assert body["by_material"]["h-MoS2"] == 1
        assert body["cof_global_mean"] == 0.05


# ---------------------------------------------------------------------------
# GET /materials
# ---------------------------------------------------------------------------


class TestMaterials:

    def test_list_materials(self, client, mock_db):
        resp = client.get("/materials")
        assert resp.status_code == 200
        assert "h-MoS2" in resp.json()["materials"]

    def test_list_materials_empty(self, client, mock_db):
        mock_db.query.return_value = pd.DataFrame()
        resp = client.get("/materials")
        assert resp.status_code == 200
        assert resp.json()["materials"] == []


# ---------------------------------------------------------------------------
# GET /conditions
# ---------------------------------------------------------------------------


class TestConditions:

    def test_get_conditions(self, client, mock_db):
        resp = client.get("/conditions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["force_nN"]["min"] == 10.0
        assert body["temperature"]["min"] == 300.0

    def test_get_conditions_empty(self, client, mock_db):
        mock_db.query.return_value = pd.DataFrame()
        resp = client.get("/conditions")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /results (auth required)
# ---------------------------------------------------------------------------


class TestStageResult:

    def test_stage_success(self, client, mock_db):
        resp = client.post(
            "/results",
            json={"material": "h-MoS2", "simulation_type": "afm", "mean_cof": 0.05},
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == 42
        assert body["status"] == "staged"

    def test_stage_no_auth(self, client, mock_db):
        resp = client.post(
            "/results",
            json={"material": "h-MoS2"},
        )
        assert resp.status_code == 401

    def test_stage_invalid_key(self, client, mock_db):
        mock_db.verify_api_key.return_value = None
        resp = client.post(
            "/results",
            json={"material": "h-MoS2"},
            headers={"X-API-Key": "bad-key"},
        )
        assert resp.status_code == 401

    def test_stage_missing_material(self, client, mock_db):
        resp = client.post(
            "/results",
            json={"simulation_type": "afm"},
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 422  # validation error


# ---------------------------------------------------------------------------
# POST /results/{id}/validate
# ---------------------------------------------------------------------------


class TestValidateResult:

    def test_validate_success(self, client, mock_db):
        resp = client.post(
            "/results/1/validate",
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_valid"] is True
        assert body["errors"] == []

    def test_validate_no_auth(self, client, mock_db):
        resp = client.post("/results/1/validate")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /results/{id}/publish
# ---------------------------------------------------------------------------


class TestPublishResult:

    def test_publish_success(self, client, mock_db):
        resp = client.post(
            "/results/1/publish",
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    def test_publish_not_found(self, client, mock_db):
        mock_db.publish.return_value = False
        resp = client.post(
            "/results/999/publish",
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /results/{id}/reject
# ---------------------------------------------------------------------------


class TestRejectResult:

    def test_reject_success(self, client, mock_db):
        resp = client.post(
            "/results/1/reject",
            json={"reason": "Duplicate data"},
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"
        mock_db.reject.assert_called_once_with(1, reason="Duplicate data")

    def test_reject_no_reason(self, client, mock_db):
        resp = client.post(
            "/results/1/reject",
            json={},
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 200

    def test_reject_not_found(self, client, mock_db):
        mock_db.reject.return_value = False
        resp = client.post(
            "/results/999/reject",
            json={},
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# HTTP Client tests (FrictionHTTPClient against TestClient-backed server)
# ---------------------------------------------------------------------------


class TestHTTPClient:
    """Test the HTTP client by patching httpx.Client with FastAPI's TestClient."""

    def _make_client(self):
        """Create a FrictionHTTPClient backed by the TestClient transport."""
        from src.api.client import FrictionHTTPClient

        # The TestClient IS an httpx.Client subclass — inject it directly
        test_client = TestClient(app)

        client = FrictionHTTPClient.__new__(FrictionHTTPClient)
        client._base = ""  # TestClient already has the base URL
        client._api_key = None
        client._client = test_client
        return client

    def _make_auth_client(self):
        """Create an authenticated FrictionHTTPClient."""
        client = self._make_client()
        client._api_key = "valid-key"
        return client

    def test_import(self):
        from src.api.client import FrictionHTTPClient
        assert FrictionHTTPClient is not None

    def test_query(self, mock_db):
        """Client.query() should return a DataFrame from the API."""
        client = self._make_client()
        df = client.query(material="h-MoS2")
        assert len(df) == 1
        assert df.iloc[0]["material"] == "h-MoS2"

    def test_upload(self, mock_db):
        """Client.upload_result() should POST and return row ID."""
        client = self._make_auth_client()
        row_id = client.upload_result(material="h-MoS2", mean_cof=0.05)
        assert row_id == 42

    def test_get_statistics(self, mock_db):
        """Client.get_statistics() returns the stats dict."""
        client = self._make_client()
        stats = client.get_statistics()
        assert stats["total_rows"] == 1

    def test_list_materials(self, mock_db):
        """Client.list_materials() returns a list of strings."""
        client = self._make_client()
        mats = client.list_materials()
        assert "h-MoS2" in mats

    def test_publish(self, mock_db):
        """Client.publish() should POST to the publish endpoint."""
        client = self._make_auth_client()
        result = client.publish(1)
        assert result is True

    def test_reject(self, mock_db):
        """Client.reject() should POST to the reject endpoint."""
        client = self._make_auth_client()
        result = client.reject(1, reason="Bad data")
        assert result is True
        mock_db.reject.assert_called_once_with(1, reason="Bad data")

    def test_get_conditions(self, mock_db):
        """Client.get_conditions() returns parameter ranges."""
        client = self._make_client()
        cond = client.get_conditions()
        assert "force_nN" in cond

    def test_upload_record(self, mock_db):
        """Client.upload_record() wraps a ResultRecord."""
        client = self._make_auth_client()
        record = ResultRecord(material="h-WS2", simulation_type="afm", mean_cof=0.03)
        row_id = client.upload_record(record)
        assert row_id == 42

    def test_upload_no_auth(self, mock_db):
        """Upload without API key should fail."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="401"):
            client.upload_result(material="h-MoS2")


# ---------------------------------------------------------------------------
# get_client factory
# ---------------------------------------------------------------------------


class TestGetClientFactory:

    def test_api_mode(self):
        from src.data import get_client
        client = get_client(mode='api', api_url='http://example.com', api_key='k')
        assert hasattr(client, '_base')
        assert client._base == 'http://example.com'
        client.close()

    def test_direct_mode_import_error(self):
        """Direct mode raises ImportError when psycopg2 unavailable."""
        from src.data import get_client
        with pytest.raises(ImportError):
            get_client(mode='direct')


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestDatabaseAPISettings:

    def test_defaults(self):
        from src.core.config import DatabaseSettings
        cfg = DatabaseSettings()
        assert cfg.api_url == 'http://localhost:8000'
        assert cfg.api_host == '0.0.0.0'
        assert cfg.api_port == 8000

    def test_custom(self):
        from src.core.config import DatabaseSettings
        cfg = DatabaseSettings(
            api_url='https://db.example.com',
            api_port=9000,
        )
        assert cfg.api_url == 'https://db.example.com'
        assert cfg.api_port == 9000
