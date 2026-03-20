"""Tests for src.data.database (FrictionDB).

All tests mock the psycopg2 dependency so they run without a real PostgreSQL
server or psycopg2 installation.  Tests that verify the ImportError path
temporarily remove the mock.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Build a minimal psycopg2 stub for tests that exercise FrictionDB internals
# ---------------------------------------------------------------------------

def _make_psycopg2_stub():
    """Return a minimal psycopg2 stub module."""
    stub = ModuleType("psycopg2")

    class OperationalError(Exception):
        pass

    stub.OperationalError = OperationalError  # type: ignore[attr-defined]
    stub.connect = MagicMock()
    return stub


_PSYCOPG2_STUB = _make_psycopg2_stub()


# ---------------------------------------------------------------------------
# Fake cursor / connection
# ---------------------------------------------------------------------------

class _FakeCursor:
    """Minimal cursor that satisfies FrictionDB's usage pattern."""

    def __init__(self, rows=None, returning_id=1):
        self._rows_queue: list = list(rows or [])
        self._returning_id = returning_id
        self.rowcount = 0
        self.executed: list = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "RETURNING id" in sql:
            self._rows_queue = [(self._returning_id,)]
        elif "COUNT(*)" in sql and "GROUP BY" not in sql:
            self._rows_queue = [(4,)]
        elif "AVG(" in sql:
            self._rows_queue = [(0.015,)]
        elif "GROUP BY material" in sql:
            self._rows_queue = [("h-MoS2", 3), ("h-WS2", 1)]
        elif "GROUP BY simulation_type" in sql:
            self._rows_queue = [("afm", 4)]

    def fetchone(self):
        return self._rows_queue[0] if self._rows_queue else None

    def fetchall(self):
        return list(self._rows_queue)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _inject_psycopg2_stub(monkeypatch):
    """Inject the psycopg2 stub into sys.modules for the duration of the test."""
    monkeypatch.setitem(sys.modules, "psycopg2", _PSYCOPG2_STUB)
    import importlib
    import src.data.database as db_module
    importlib.reload(db_module)
    yield
    importlib.reload(db_module)


@pytest.fixture()
def db():
    """FrictionDB with a mocked cursor (no real DB required)."""
    from src.data.database import FrictionDB

    with patch("src.data.database.FrictionDB._ensure_schema"):
        instance = FrictionDB(
            host="localhost", port=5432,
            dbname="test_db", user="test", password="test",
            auto_create=False,
        )

    fake_cursor = _FakeCursor()

    @contextmanager
    def _mock_cursor(commit=False):
        yield fake_cursor

    instance._cursor = _mock_cursor  # type: ignore[method-assign]
    return instance


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestFrictionDBConnectionParams:
    """Test _get_connection_params helper."""

    def test_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("FRICTION_DB_HOST", "myhost")
        monkeypatch.setenv("FRICTION_DB_PORT", "5433")
        monkeypatch.setenv("FRICTION_DB_NAME", "mydb")
        monkeypatch.setenv("FRICTION_DB_USER", "alice")
        monkeypatch.setenv("FRICTION_DB_PASSWORD", "s3cr3t")

        from src.data.database import _get_connection_params
        params = _get_connection_params()

        assert params["host"] == "myhost"
        assert params["port"] == 5433
        assert params["dbname"] == "mydb"
        assert params["user"] == "alice"
        assert params["password"] == "s3cr3t"

    def test_kwargs_override_env(self, monkeypatch):
        monkeypatch.setenv("FRICTION_DB_HOST", "envhost")
        from src.data.database import _get_connection_params
        params = _get_connection_params(host="arghost")
        assert params["host"] == "arghost"

    def test_defaults_when_no_env(self, monkeypatch):
        for var in ("FRICTION_DB_HOST", "FRICTION_DB_PORT",
                    "FRICTION_DB_NAME", "FRICTION_DB_USER", "FRICTION_DB_PASSWORD"):
            monkeypatch.delenv(var, raising=False)

        from src.data.database import _get_connection_params
        params = _get_connection_params()
        assert params["host"] == "localhost"
        assert params["port"] == 5432
        assert params["dbname"] == "frictionsim2d"


class TestFrictionDBUpload:
    """Test upload_result method."""

    def test_upload_returns_row_id(self, db):
        row_id = db.upload_result(
            material="h-MoS2",
            simulation_type="afm",
            layers=1,
            force_nN=10.0,
            scan_angle=0.0,
            temperature=300.0,
            mean_cof=0.012,
        )
        assert isinstance(row_id, int)

    def test_upload_all_fields(self, db):
        row_id = db.upload_result(
            material="h-WS2",
            simulation_type="sheetonsheet",
            layers=2,
            force_nN=5.0,
            pressure_gpa=None,
            scan_angle=30.0,
            scan_speed=1.0,
            temperature=300.0,
            tip_material="Si",
            tip_radius=20.0,
            mean_cof=0.015,
            std_cof=0.002,
            mean_lf=0.15,
            mean_nf=10.0,
            uploader="tester",
            notes="unit test",
            metadata={"source": "test"},
        )
        assert isinstance(row_id, int)

    def test_upload_minimal(self, db):
        """Only mandatory field 'material' is required."""
        row_id = db.upload_result(material="silicene")
        assert isinstance(row_id, int)


class TestFrictionDBQuery:
    """Test query method."""

    def _make_db_with_rows(self):
        from src.data.database import FrictionDB

        sample_rows = [
            (1, "2024-01-01", "alice", "h-MoS2", "afm", 1,
             10.0, None, 0.0, 1.0, 300.0, "Si", 20.0, 0.012, 0.001, 0.12, 10.0,
             None, None),
            (2, "2024-01-02", "bob", "h-WS2", "afm", 1,
             10.0, None, 0.0, 1.0, 300.0, "Si", 20.0, 0.020, 0.002, 0.20, 10.0,
             None, None),
        ]

        with patch("src.data.database.FrictionDB._ensure_schema"):
            instance = FrictionDB(auto_create=False)

        @contextmanager
        def _cursor_with_rows(commit=False):
            yield _FakeCursor(rows=list(sample_rows))

        instance._cursor = _cursor_with_rows  # type: ignore[method-assign]
        return instance

    def test_query_returns_dataframe(self):
        pytest.importorskip("pandas")
        db = self._make_db_with_rows()
        df = db.query(material="h-MoS2")
        assert hasattr(df, 'columns')

    def test_query_columns_present(self):
        pytest.importorskip("pandas")
        from src.data.database import _COLUMN_NAMES
        db = self._make_db_with_rows()
        df = db.query()
        assert list(df.columns) == _COLUMN_NAMES

    def test_empty_result(self):
        pytest.importorskip("pandas")
        from src.data.database import FrictionDB

        with patch("src.data.database.FrictionDB._ensure_schema"):
            instance = FrictionDB(auto_create=False)

        @contextmanager
        def _empty_cursor(commit=False):
            yield _FakeCursor(rows=[])

        instance._cursor = _empty_cursor  # type: ignore[method-assign]
        df = instance.query(material="nonexistent")
        assert df.empty


class TestFrictionDBStatistics:
    """Test get_statistics method."""

    def test_statistics_keys(self, db):
        stats = db.get_statistics()
        assert "total_rows" in stats
        assert "by_material" in stats
        assert "by_type" in stats
        assert "cof_global_mean" in stats

    def test_statistics_types(self, db):
        stats = db.get_statistics()
        assert isinstance(stats["total_rows"], int)
        assert isinstance(stats["by_material"], dict)
        assert isinstance(stats["by_type"], dict)


class TestFrictionDBDelete:
    """Test delete_own_results method."""

    def test_delete_returns_count(self, db):
        count = db.delete_own_results("alice")
        assert isinstance(count, int)


class TestFrictionDBImportError:
    """Test that ImportError is raised when psycopg2 is truly missing."""

    def test_import_error_when_no_psycopg2(self, monkeypatch):
        """FrictionDB raises ImportError when psycopg2 is absent."""
        import importlib
        import src.data.database as db_module

        # Remove stub so the real import fails
        monkeypatch.delitem(sys.modules, "psycopg2", raising=False)
        importlib.reload(db_module)

        with pytest.raises(ImportError, match="psycopg2"):
            db_module.FrictionDB()

        # Restore stub for subsequent tests (autouse fixture will also reload)
        monkeypatch.setitem(sys.modules, "psycopg2", _PSYCOPG2_STUB)
        importlib.reload(db_module)
