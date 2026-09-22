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

class _JsonMarker:
    """Stand-in for `psycopg2.extras.Json`: wraps a value, doesn't adapt it.

    A real assertion that a JSONB column went through `Json(...)` needs a type
    to check `isinstance` against -- a passthrough lambda returns the bare
    dict, which is indistinguishable from a column that was never wrapped at
    all.
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _JsonMarker) and other.value == self.value

    def __repr__(self):
        return f"_JsonMarker({self.value!r})"


def _make_psycopg2_stub():
    """Return a minimal psycopg2 stub module, plus its `extras` submodule.

    `upload_record` does `from psycopg2.extras import Json` -- without a
    registered `psycopg2.extras` module in `sys.modules` that import raises
    ImportError in every upload test, not just the ones carrying `extras`.
    """
    stub = ModuleType("psycopg2")

    class OperationalError(Exception):
        pass

    stub.OperationalError = OperationalError  # type: ignore[attr-defined]
    stub.connect = MagicMock()

    extras_stub = ModuleType("psycopg2.extras")
    extras_stub.Json = _JsonMarker  # type: ignore[attr-defined]
    stub.extras = extras_stub  # type: ignore[attr-defined]
    return stub, extras_stub


_PSYCOPG2_STUB, _PSYCOPG2_EXTRAS_STUB = _make_psycopg2_stub()


# ---------------------------------------------------------------------------
# Fake cursor / connection
# ---------------------------------------------------------------------------

class _FakeCursor:
    """Minimal cursor that satisfies FrictionDB's usage pattern."""

    def __init__(self, rows=None, returning_id=1, existing_columns=None,
                 duplicate=False):
        self._rows_queue: list = list(rows or [])
        self._returning_id = returning_id
        self.rowcount = 0
        self.executed: list = []
        #: `None` means `simulations` doesn't exist yet (the default -- a
        #: fresh install). A list simulates an existing table with exactly
        #: these columns, for `init_schema`'s pre-v3 detection.
        self._existing_columns = existing_columns
        #: Simulates `ON CONFLICT (time_series_hash) DO NOTHING` matching an
        #: existing row: the INSERT executes but returns no row.
        self._duplicate = duplicate

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "RETURNING id" in sql:
            self._rows_queue = [] if self._duplicate else [(self._returning_id,)]
        elif "COUNT(*)" in sql and "GROUP BY" not in sql:
            self._rows_queue = [(4,)]
        elif "AVG(" in sql:
            self._rows_queue = [(0.015,)]
        elif "GROUP BY material" in sql:
            self._rows_queue = [("h-MoS2", 3), ("h-WS2", 1)]
        elif "GROUP BY family" in sql:
            self._rows_queue = [("afm", 4)]
        elif "to_regclass" in sql:
            self._rows_queue = ([("simulations",)] if self._existing_columns
                               is not None else [(None,)])
        elif "information_schema.columns" in sql:
            self._rows_queue = [(c,) for c in (self._existing_columns or [])]
        elif "ON CONFLICT" in sql:
            pass  # schema_version upsert
        elif "time_series_hash" in sql and "SELECT" in sql:
            self._rows_queue = []  # no existing hashes by default

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
    monkeypatch.setitem(sys.modules, "psycopg2.extras", _PSYCOPG2_EXTRAS_STUB)
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
        assert params["dbname"] == "frictionsim2ddb"


class TestFrictionDBUpload:
    """Test upload_result method."""

    def test_upload_returns_row_id(self, db):
        row_id = db.upload_result(
            material="h-MoS2",
            family="afm",
            model="afm",
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
            family="sheetonsheet",
            model="sheetonsheet",
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
            extras={"source": "test"},
        )
        assert isinstance(row_id, int)

    def test_upload_minimal(self, db):
        """Only mandatory field 'material' is required."""
        row_id = db.upload_result(material="silicene")
        assert isinstance(row_id, int)

    def test_upload_uses_on_conflict_do_nothing(self, db):
        """The INSERT itself must be conflict-safe, not just its caller."""
        db.upload_result(material="h-MoS2", time_series_hash="abc123")
        with db._cursor() as cur:
            sql, _params = cur.executed[-1]
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    def test_duplicate_time_series_hash_returns_none(self):
        """A repeat upload of the same result is a clean `None`, not a crash.

        `time_series_hash` is `UNIQUE`; `ON CONFLICT ... DO NOTHING` makes the
        INSERT a no-op and `RETURNING id` yields no row. Real code (a raw
        `psycopg2.errors.IntegrityError` on a naive INSERT) never reaches
        `upload_record`'s caller at all.
        """
        from src.data.database import FrictionDB

        with patch("src.data.database.FrictionDB._ensure_schema"):
            instance = FrictionDB(auto_create=False)

        fake_cursor = _FakeCursor(duplicate=True)

        @contextmanager
        def _mock_cursor(commit=False):
            yield fake_cursor

        instance._cursor = _mock_cursor  # type: ignore[method-assign]

        row_id = instance.upload_result(
            material="h-MoS2", family="afm", model="afm",
            time_series_hash="dupe-hash",
        )

        assert row_id is None


class TestFrictionDBQuery:
    """Test query method."""

    def _make_db_with_rows(self):
        from src.data.database import FrictionDB, _COLUMN_NAMES

        # Build sample rows from _COLUMN_NAMES itself (the record IS the
        # schema, per Task 30) rather than hand-listing the schema-2 tuple.
        def _make_row(row_id, uploader, material, family, layers, force, cof):
            values = dict.fromkeys(_COLUMN_NAMES)
            values.update(
                id=row_id, uploaded_at="2024-01-01", uploader=uploader,
                model=family, family=family,
                materials=[material], material=material,
                stack_type="AA", layers=layers, size_x=100.0, size_y=100.0,
                temperature=300.0, tip_material="Si", tip_radius=20.0,
                potential_type="sw",
                force_nN=force, scan_angle=0.0, scan_speed=1.0,
                mean_cof=cof, std_cof=0.001, mean_lf=0.12, std_lf=0.01,
                mean_nf=10.0, std_nf=0.5, mean_lfx=0.10, std_lfx=0.01,
                mean_lfy=0.06, std_lfy=0.005,
                ntimesteps=500000, is_complete=True, status="published",
            )
            return tuple(values[name] for name in _COLUMN_NAMES)

        sample_rows = [
            _make_row(1, "alice", "h-MoS2", "afm", 1, 10.0, 0.012),
            _make_row(2, "bob", "h-WS2", "afm", 1, 10.0, 0.020),
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

        # Block the real psycopg2 import by setting the entry to None
        monkeypatch.setitem(sys.modules, "psycopg2", None)
        importlib.reload(db_module)

        with pytest.raises(ImportError, match="psycopg2"):
            db_module.FrictionDB()

        # Restore stub for subsequent tests (autouse fixture will also reload)
        monkeypatch.setitem(sys.modules, "psycopg2", _PSYCOPG2_STUB)
        importlib.reload(db_module)


# ===========================================================================
# Tests for src.data.models
# ===========================================================================

import numpy as np  # noqa: E402


class TestComputeFrictionStats:
    """Test the canonical compute_friction_stats function."""

    def test_basic_computation(self):
        from src.data.models import compute_friction_stats

        nf = np.full(100, 10.0)
        lfx = np.full(100, 0.5)
        lfy = np.full(100, 0.0)

        stats = compute_friction_stats(nf, lfx, lfy, skip_fraction=0.0)

        assert stats['mean_cof'] == pytest.approx(0.05, abs=1e-6)
        assert stats['mean_nf'] == pytest.approx(10.0, abs=1e-6)
        assert stats['mean_lf'] == pytest.approx(0.5, abs=1e-6)

    def test_transient_skip(self):
        from src.data.models import compute_friction_stats

        # First 20% has wildly different values
        n = 100
        nf = np.full(n, 10.0)
        lfx = np.full(n, 1.0)
        lfy = np.zeros(n)
        # Transient: first 20 points have huge lateral force
        lfx[:20] = 100.0

        stats_no_skip = compute_friction_stats(nf, lfx, lfy, skip_fraction=0.0)
        stats_with_skip = compute_friction_stats(nf, lfx, lfy, skip_fraction=0.2)

        # Without skip, COF is dominated by transient
        assert stats_no_skip['mean_cof'] > stats_with_skip['mean_cof']
        # With skip, COF should be close to 1.0/10.0 = 0.1
        assert stats_with_skip['mean_cof'] == pytest.approx(0.1, abs=1e-6)

    def test_ratio_of_means(self):
        """Verify we use ratio-of-means (not mean-of-ratios)."""
        from src.data.models import compute_friction_stats

        # Variable normal force: ratio-of-means != mean-of-ratios
        nf = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        lfx = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        lfy = np.zeros(5)

        stats = compute_friction_stats(nf, lfx, lfy, skip_fraction=0.0)

        # Ratio-of-means: mean(lf) / mean(nf) = 1.5 / 15.0 = 0.1
        assert stats['mean_cof'] == pytest.approx(0.1, abs=1e-6)

        # Mean-of-ratios would be: mean([0.1, 0.1, 0.1, 0.1, 0.1]) = 0.1
        # (same in this case since all ratios are equal)

    def test_empty_array_raises(self):
        from src.data.models import compute_friction_stats

        with pytest.raises(ValueError, match="must not be empty"):
            compute_friction_stats(np.array([]), np.array([]), np.array([]))

    def test_mismatched_shapes_raises(self):
        from src.data.models import compute_friction_stats

        with pytest.raises(ValueError, match="shape mismatch"):
            compute_friction_stats(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
            )

    def test_invalid_skip_fraction_raises(self):
        from src.data.models import compute_friction_stats

        nf = np.array([1.0, 2.0, 3.0])
        lfx = np.array([0.1, 0.1, 0.1])
        lfy = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="skip_fraction"):
            compute_friction_stats(nf, lfx, lfy, skip_fraction=1.0)

        with pytest.raises(ValueError, match="skip_fraction"):
            compute_friction_stats(nf, lfx, lfy, skip_fraction=-0.1)


class TestComputeDerivedColumns:
    """Test element-wise derived column computation."""

    def test_lateral_force_magnitude(self):
        from src.data.models import compute_derived_columns

        lfx = np.array([3.0, 0.0])
        lfy = np.array([4.0, 5.0])
        nf = np.array([10.0, 10.0])

        lf, cof = compute_derived_columns(lfx, lfy, nf)

        assert lf[0] == pytest.approx(5.0, abs=1e-6)
        assert lf[1] == pytest.approx(5.0, abs=1e-6)
        assert cof[0] == pytest.approx(0.5, abs=1e-6)

    def test_zero_normal_force(self):
        from src.data.models import compute_derived_columns

        lfx = np.array([1.0])
        lfy = np.array([1.0])
        nf = np.array([0.0])

        _, cof = compute_derived_columns(lfx, lfy, nf)
        assert cof[0] == 0.0  # Safe division


class TestTimeSeriesHash:
    """Test deduplication hash."""

    def test_deterministic(self):
        from src.data.models import compute_time_series_hash

        nf = np.array([1.0, 2.0, 3.0])
        lfx = np.array([0.1, 0.2, 0.3])
        lfy = np.array([0.0, 0.0, 0.0])

        h1 = compute_time_series_hash(nf, lfx, lfy)
        h2 = compute_time_series_hash(nf, lfx, lfy)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_data_different_hash(self):
        from src.data.models import compute_time_series_hash

        nf1 = np.array([1.0, 2.0, 3.0])
        nf2 = np.array([1.0, 2.0, 3.1])
        lfx = np.array([0.1, 0.2, 0.3])
        lfy = np.zeros(3)

        assert compute_time_series_hash(nf1, lfx, lfy) != compute_time_series_hash(nf2, lfx, lfy)


class TestResultRecord:
    """Test the canonical ResultRecord model."""

    def test_create_minimal(self):
        from src.data.models import ResultRecord

        r = ResultRecord(material="h-MoS2")
        assert r.material == "h-MoS2"
        assert r.family is None
        assert r.model is None
        assert r.status == "staged"

    def test_force_and_pressure_exclusive(self):
        from src.data.models import ResultRecord

        with pytest.raises(ValueError, match="force_nN.*OR.*pressure_gpa"):
            ResultRecord(material="h-MoS2", force_nN=10.0, pressure_gpa=1.0)

    def test_populate_stats(self):
        from src.data.models import ResultRecord

        r = ResultRecord(material="h-MoS2")
        nf = np.full(100, 10.0)
        lfx = np.full(100, 1.0)
        lfy = np.zeros(100)
        r.populate_stats(nf, lfx, lfy, skip_fraction=0.0)

        assert r.mean_cof == pytest.approx(0.1, abs=1e-6)
        assert r.ntimesteps == 100
        assert r.time_series_hash is not None

    def test_json_round_trip(self):
        from src.data.models import ResultRecord

        r = ResultRecord(
            material="h-MoS2", layers=1, force_nN=10.0,
            mean_cof=0.012, temperature=300.0,
        )
        json_str = r.to_json()
        r2 = ResultRecord.from_json(json_str)
        assert r2.material == r.material
        assert r2.mean_cof == r.mean_cof

    def test_to_upload_dict(self):
        from src.data.models import ResultRecord

        r = ResultRecord(material="h-MoS2", mean_cof=0.012)
        d = r.to_upload_dict()
        assert 'id' not in d
        assert 'uploaded_at' not in d
        assert d['material'] == 'h-MoS2'


# ===========================================================================
# Tests for src.data.validation
# ===========================================================================


class TestValidation:
    """Test the validation module."""

    def test_valid_record_passes(self):
        from src.data.models import ResultRecord
        from src.data.validation import validate_record

        r = ResultRecord(material="h-MoS2", family="afm", model="afm", mean_cof=0.012)
        vr = validate_record(r)
        assert vr.is_valid

    def test_missing_material_fails(self):
        from src.data.models import ResultRecord
        from src.data.validation import validate_record

        r = ResultRecord(material="", family="afm", model="afm", mean_cof=0.012)
        vr = validate_record(r)
        assert not vr.is_valid

    def test_out_of_range_cof_fails(self):
        from src.data.models import ResultRecord
        from src.data.validation import validate_record

        r = ResultRecord(material="h-MoS2", family="afm", model="afm", mean_cof=10.0)
        vr = validate_record(r)
        assert not vr.is_valid

    def test_duplicate_hash_fails(self):
        from src.data.models import ResultRecord
        from src.data.validation import validate_record

        r = ResultRecord(material="h-MoS2", family="afm", model="afm",
                         mean_cof=0.01, time_series_hash="abc123")
        vr = validate_record(r, existing_hashes=["abc123"])
        assert not vr.is_valid

    def test_cof_consistency_warning(self):
        from src.data.models import ResultRecord
        from src.data.validation import validate_record

        # mean_lf/mean_nf != mean_cof
        r = ResultRecord(
            material="h-MoS2", family="afm", model="afm", mean_cof=0.5,
            mean_lf=1.0, mean_nf=10.0,  # expected COF = 0.1
        )
        vr = validate_record(r)
        assert len(vr.warnings) > 0


# ===========================================================================
# Unit conversion constants
# ===========================================================================

class TestUnitConversionConstants:
    """Verify the physical constants are self-consistent."""

    def test_ev_a3_to_gpa_derived_from_ev_a_to_nn(self):
        from src.data.models import EV_A_TO_NN, EV_A3_TO_GPA
        assert EV_A3_TO_GPA == pytest.approx(EV_A_TO_NN * 100)

    def test_nm_to_ev_a2_derived_from_ev_a_to_nn(self):
        from src.data.models import EV_A_TO_NN, NM_TO_EV_A2
        assert NM_TO_EV_A2 == pytest.approx(EV_A_TO_NN * 10)

    def test_ev_a_to_nn_value(self):
        from src.data.models import EV_A_TO_NN
        assert EV_A_TO_NN == pytest.approx(1.602176565)


# ===========================================================================
# Column mapping consistency
# ===========================================================================

class TestColumnMappings:
    """Verify DataReader column definitions match template output order."""

    def test_afm_columns_no_phantoms(self):
        from src.postprocessing.read_data import DataReader
        # AFM template outputs 9 variables + timestep = 10 columns.
        # No phantom columns (lateral_force / cof) should be in the list.
        assert len(DataReader._AFM_FILE_COLUMNS) == 10
        assert 'lateral_force' not in DataReader._AFM_FILE_COLUMNS
        assert 'cof' not in DataReader._AFM_FILE_COLUMNS

    def test_afm_columns_have_required_fields(self):
        from src.postprocessing.read_data import DataReader
        for col in ('time', 'nf', 'lfx', 'lfy'):
            assert col in DataReader._AFM_FILE_COLUMNS

    def test_sheet_rename_maps_to_canonical_names(self):
        from src.core.observables import rename_map
        rename = rename_map('sheetonsheet', 'window')
        assert rename['v_xfrict'] == 'lfx'
        assert rename['v_fz'] == 'nf'


# ===========================================================================
# Phase 2: Database settings, profiles, migration runner
# ===========================================================================

class TestDatabaseSettings:
    """Verify DatabaseSettings Pydantic model from config.py -- central-only, flat."""

    def test_central_host_defaults_empty(self):
        from src.core.config import DatabaseSettings
        ds = DatabaseSettings()
        assert ds.host == ''
        assert ds.port == 5432
        assert ds.dbname == 'frictionsim2ddb'

    def test_global_settings_includes_database(self):
        from src.core.config import GlobalSettings
        gs = GlobalSettings()
        assert hasattr(gs, 'database')
        assert gs.database.dbname == 'frictionsim2ddb'

    def test_skip_fraction_default(self):
        from src.core.config import DatabaseSettings
        ds = DatabaseSettings()
        assert ds.skip_fraction == 0.2


class TestCentralDb:
    """`central_db` replaces `db_from_profile` -- one database, no profile."""

    def test_central_db_reads_the_flat_settings(self, tmp_path):
        from unittest.mock import patch

        from src.data.database import central_db

        settings_file = tmp_path / 'settings.yaml'
        settings_file.write_text(
            'database:\n  host: db.example.org\n  port: 6543\n'
            '  dbname: friction\n  user: alice\n  password: s3cr3t\n',
            encoding='utf-8')

        with patch("src.data.database.FrictionDB._ensure_schema"):
            db = central_db(settings_file=settings_file)

        assert db._conn_params['host'] == 'db.example.org'
        assert db._conn_params['port'] == 6543
        assert db._conn_params['user'] == 'alice'

    def test_db_from_profile_is_gone(self):
        import src.data.database as db_mod

        assert not hasattr(db_mod, 'db_from_profile')

    def test_get_client_defaults_to_the_api(self, monkeypatch):
        from src.data import get_client

        seen = {}
        monkeypatch.setattr('src.api.client.FrictionHTTPClient',
                            lambda **kw: seen.update(kw) or 'CLIENT')

        assert get_client() == 'CLIENT'
        assert 'api_url' in seen

    def test_cli_has_no_local_bootstrap(self):
        import src.cli as cli_mod

        assert not hasattr(cli_mod, '_bootstrap_local_db_once')


class TestSchemaVersion:
    """Basic checks for fixed pre-release schema versioning."""

    def test_schema_version_constant(self):
        from src.data.database import SCHEMA_VERSION
        assert SCHEMA_VERSION == 3


# ===========================================================================
# Task 30: schema-3 init/upload/query
# ===========================================================================

def _recording_db(existing_columns=None):
    """A `FrictionDB` whose fake cursor records every executed statement."""
    from src.data.database import FrictionDB

    with patch("src.data.database.FrictionDB._ensure_schema"):
        instance = FrictionDB(host="localhost", port=5432, dbname="test_db",
                              user="test", password="test", auto_create=False)
    cursor = _FakeCursor(existing_columns=existing_columns)

    @contextmanager
    def _mock_cursor(commit=False):
        yield cursor

    instance._cursor = _mock_cursor  # type: ignore[method-assign]
    return instance, cursor


def test_init_schema_drop_recreates():
    db, cursor = _recording_db()

    db.init_schema(drop=True)

    executed = ' '.join(sql for sql, _params in cursor.executed)
    assert 'DROP TABLE IF EXISTS simulations' in executed
    assert 'CREATE TABLE IF NOT EXISTS simulations' in executed


def test_init_schema_refuses_an_existing_schema2_table():
    """Without --drop, a pre-v3 table must be left alone, not silently
    stamped as schema_version=3 -- see the docstring on
    `FrictionDB.init_schema` for why that would be worse than doing nothing.
    """
    db, cursor = _recording_db(
        existing_columns=['id', 'uploaded_at', 'uploader', 'material',
                          'simulation_type', 'layers', 'metadata'])

    with pytest.raises(RuntimeError, match='db init --drop'):
        db.init_schema(drop=False)

    executed = [sql for sql, _params in cursor.executed]
    assert not any('INSERT INTO schema_version' in sql for sql in executed)
    assert not any(sql.strip().startswith('CREATE TABLE IF NOT EXISTS simulations')
                   for sql in executed)


def test_init_schema_proceeds_when_no_table_exists_yet():
    """A fresh install has no `simulations` table -- that is NOT pre-v3."""
    db, cursor = _recording_db(existing_columns=None)

    db.init_schema(drop=False)

    executed = ' '.join(sql for sql, _params in cursor.executed)
    assert 'CREATE TABLE IF NOT EXISTS simulations' in executed
    assert 'INSERT INTO schema_version' in executed


def test_extras_reach_the_cursor_wrapped_in_json():
    """`upload_record` must wrap `extras` in `psycopg2.extras.Json` before it
    reaches the cursor -- a bare dict makes the real driver raise
    `ProgrammingError: can't adapt type 'dict'`. Assert the actual TYPE, not
    just that a dict-shaped value made it through: a passthrough would look
    identical to a real dict at that point.
    """
    db, cursor = _recording_db()

    db.upload_result(material="h-MoS2", family="afm", model="afm",
                     extras={"foo": "bar"})

    sql, params = cursor.executed[-1]
    columns = sql.split('(', 1)[1].split(')', 1)[0].split(', ')
    value = params[columns.index('extras')]
    assert isinstance(value, _JsonMarker)
    assert value.value == {"foo": "bar"}


def test_query_filters_on_family_and_set():
    db, cursor = _recording_db()

    db.query(family='hetero', set_label='260910-afm',
             pressure_range=(0.5, 2.0), limit=10)

    sql, params = cursor.executed[-1]
    assert 'family = %s' in sql
    assert 'set_label = %s' in sql
    assert 'pressure_gpa BETWEEN %s AND %s' in sql
    assert 'LIMIT 10' in sql
    assert params == ['hetero', '260910-afm', 0.5, 2.0]
