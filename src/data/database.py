"""Central PostgreSQL database interface for FrictionSim2D.

This module implements the **central (remote) database** — one half of the
two-database architecture used by FrictionSim2D.

Two-database architecture
-------------------------
FrictionSim2D uses two distinct databases with different roles:

**Local database (AiiDA)**
    Managed by AiiDA on your own machine.  Stores full provenance and complete
    time-series data for every simulation you generate.  Used via the
    ``FrictionSim2D aiida`` command group.  Set up once with
    ``FrictionSim2D aiida setup``.

**Central database (this module)**
    A shared PostgreSQL instance hosted on a remote server.  Stores only
    **summary statistics** (~1 KB per simulation: mean COF, forces, conditions).
    Used to share results across users and machines.  Accessed via the
    ``FrictionSim2D db`` command group.  Full time-series data is *not*
    uploaded here — it stays in your local AiiDA database.

Typical workflow::

    # 1. Run simulations; AiiDA stores full provenance locally.
    FrictionSim2D run afm afm_config.ini --aiida

    # 2. Import completed results into local AiiDA DB (full timeseries).
    FrictionSim2D aiida import ./simulation_YYYYMMDD

    # 3. Upload summary statistics to the central shared DB.
    FrictionSim2D db upload ./simulation_YYYYMMDD --uploader alice --profile central

Central DB staging pipeline::

    staged → validated → published
           ↘ rejected

Usage
-----
Configure the central DB connection via environment variables, keyword
arguments, or ``settings.yaml`` profiles::

    export FRICTION_DB_HOST=db.example.com
    export FRICTION_DB_PORT=5432
    export FRICTION_DB_NAME=frictionsim2ddb
    export FRICTION_DB_USER=myuser
    export FRICTION_DB_PASSWORD=secret

    from src.data.database import FrictionDB
    from src.data.models import ResultRecord

    db = FrictionDB()

    record = ResultRecord(
        material="h-MoS2", family="afm", model="afm",
        layers=1, force_nN=10.0, temperature=300.0, mean_cof=0.012,
    )
    db.upload_record(record)

    df = db.query(material="h-MoS2", layers=1)
    print(df)

Connection
----------
:func:`central_db` creates a :class:`FrictionDB` from ``settings.database`` --
there is one database, so there is no profile to choose any more.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Union

from .models import ResultRecord

if TYPE_CHECKING:
    import pandas

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 3

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS simulations (
    id                  SERIAL PRIMARY KEY,
    uploaded_at         TIMESTAMPTZ DEFAULT now(),
    uploader            TEXT,
    contact             TEXT,
    run_uuid            TEXT,
    result_uuid         TEXT,
    set_label           TEXT,
    source_host         TEXT,
    model               TEXT,
    family              TEXT,
    materials           TEXT[],
    material            TEXT,
    stack_type          TEXT,
    layers              INTEGER,
    size_x              REAL,
    size_y              REAL,
    temperature         REAL,
    tip_material        TEXT,
    tip_radius          REAL,
    substrate_material  TEXT,
    substrate_amorphous BOOLEAN,
    potential_type      TEXT,
    potential_name      TEXT,
    pot_sha256          TEXT,
    git_commit          TEXT,
    force_nN            REAL,
    pressure_gpa        REAL,
    scan_angle          REAL,
    scan_speed          REAL,
    mean_cof            REAL,
    std_cof             REAL,
    mean_lf             REAL,
    std_lf              REAL,
    mean_nf             REAL,
    std_nf              REAL,
    mean_lfx            REAL,
    std_lfx             REAL,
    mean_lfy            REAL,
    std_lfy             REAL,
    ntimesteps          INTEGER,
    time_series_hash    TEXT UNIQUE,
    is_complete         BOOLEAN DEFAULT true,
    extras              JSONB,
    status              TEXT DEFAULT 'staged',
    notes               TEXT,
    data_url            TEXT
);
"""

_DROP_TABLE_SQL = "DROP TABLE IF EXISTS simulations CASCADE;"

_CREATE_API_KEYS_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          SERIAL PRIMARY KEY,
    key_hash    TEXT UNIQUE NOT NULL,
    user_name   TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    is_active   BOOLEAN DEFAULT true
);
"""

_CREATE_SCHEMA_VERSION_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT now()
);
"""

#: Derived, not repeated: the record IS the schema (spec §5).
_COLUMN_NAMES = list(ResultRecord.model_fields)

# Allowed tokens for query ORDER BY sanitisation
_ALLOWED_ORDER_COLS = set(_COLUMN_NAMES) | {'ASC', 'DESC', 'asc', 'desc'}

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _get_connection_params(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    host: Optional[str] = None,
    port: Optional[int] = None,
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    """Build psycopg2 connection parameter dict.

    Values provided as arguments take precedence over environment variables.

    Environment variables:
        ``FRICTION_DB_HOST``, ``FRICTION_DB_PORT``, ``FRICTION_DB_NAME``,
        ``FRICTION_DB_USER``, ``FRICTION_DB_PASSWORD``.

    Args:
        host: Database hostname or IP.
        port: Database port.
        dbname: Database name.
        user: Database username.
        password: Database password.

    Returns:
        Dict suitable for passing to :func:`psycopg2.connect`.
    """
    return {
        'host': host or os.environ.get('FRICTION_DB_HOST', 'localhost'),
        'port': port or int(os.environ.get('FRICTION_DB_PORT', '5432')),
        'dbname': dbname or os.environ.get('FRICTION_DB_NAME', 'frictionsim2ddb'),
        'user': user or os.environ.get('FRICTION_DB_USER', ''),
        'password': password or os.environ.get('FRICTION_DB_PASSWORD', ''),
    }


def central_db(settings_file: Optional[Union[str, Path]] = None) -> 'FrictionDB':
    """The central summary DB, from `settings.database`.

    One database, so one function: there is no profile to choose any more.
    """
    from ..core.config import load_settings  # pylint: disable=import-outside-toplevel

    cfg = load_settings(settings_file=settings_file).database
    return FrictionDB(host=cfg.host or None, port=cfg.port, dbname=cfg.dbname,
                      user=cfg.user or None, password=cfg.password or None)


# ---------------------------------------------------------------------------
# FrictionDB
# ---------------------------------------------------------------------------


class FrictionDB:
    """Client for the shared FrictionSim2D PostgreSQL database.

    Provides :meth:`upload_result` to contribute data and :meth:`query` to
    retrieve and filter the shared dataset.

    Args:
        host: PostgreSQL hostname (default: ``FRICTION_DB_HOST`` env var or
            ``'localhost'``).
        port: PostgreSQL port (default: ``FRICTION_DB_PORT`` env var or 5432).
        dbname: Database name (default: ``FRICTION_DB_NAME`` env var or
            ``'frictionsim2d'``).
        user: Username (default: ``FRICTION_DB_USER`` env var).
        password: Password (default: ``FRICTION_DB_PASSWORD`` env var).
        auto_create: If ``True``, create the ``simulations`` table on first
            connection (default: ``True``).

    Raises:
        ImportError: If ``psycopg2`` is not installed.
        psycopg2.OperationalError: If the database connection fails.

    Example::

        db = FrictionDB(host="db.example.com", user="alice", password="s3cr3t")
        db.upload_result(
            material="h-MoS2", layers=1, force_nN=10.0,
            mean_cof=0.012, temperature=300.0,
        )
        df = db.query(material="h-MoS2")
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        auto_create: bool = True,
    ):
        try:
            import psycopg2  # pyright: ignore[reportMissingModuleSource]  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for database access. "
                "Install with: conda install -c conda-forge psycopg2"
            ) from exc

        self._psycopg2 = psycopg2
        self._conn_params = _get_connection_params(host, port, dbname, user, password)

        if auto_create:
            self._ensure_schema()

    # -- Context manager -------------------------------------------------------

    @contextmanager
    def _cursor(self, commit: bool = False) -> Generator:
        """Yield an auto-closing cursor, optionally committing after the block.

        Args:
            commit: Whether to commit the transaction after the block.

        Yields:
            psycopg2 cursor object.
        """
        conn = self._psycopg2.connect(**self._conn_params)
        try:
            cur = conn.cursor()
            yield cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -- Schema ----------------------------------------------------------------

    @staticmethod
    def _existing_table_is_pre_v3(cur) -> bool:
        """True if `simulations` already exists and predates schema 3.

        A fresh install has no such table yet -- `to_regclass` returns NULL --
        and that is NOT pre-v3, it is "nothing to migrate". Only a table that
        exists AND lacks the `family` column (added in schema 3) is pre-v3;
        checking the column itself is robust even if a `schema_version` row
        was never written (an old table on a DB nobody ran `db init` against
        again since).
        """
        cur.execute("SELECT to_regclass('simulations');")
        row = cur.fetchone()
        if not row or row[0] is None:
            return False
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'simulations';"
        )
        columns = {r[0] for r in cur.fetchall()}
        return 'family' not in columns

    def init_schema(self, drop: bool = False) -> None:
        """Create the schema; `drop=True` recreates `simulations` from scratch.

        `db init --drop` is the supported upgrade path (spec U6): there is no
        migration from schema 2 and the central DB is not deployed yet.
        Without `--drop`, an existing schema-2 table is left untouched and
        this raises -- silently stamping `schema_version = 3` next to a table
        that is still schema 2 would make every later `_ensure_schema()` call
        (every `FrictionDB()` construction with the default `auto_create=True`)
        believe the schema is current when it is not.
        """
        with self._cursor(commit=True) as cur:
            cur.execute(_CREATE_SCHEMA_VERSION_SQL)
            if drop:
                cur.execute(_DROP_TABLE_SQL)
            elif self._existing_table_is_pre_v3(cur):
                raise RuntimeError(
                    "schema-2 'simulations' table found; there is no "
                    "in-place migration to schema 3 -- run "
                    "`FrictionSim2D db init --drop` (this drops the table)."
                )
            cur.execute(_CREATE_TABLE_SQL)
            cur.execute(_CREATE_API_KEYS_SQL)

            # Record current version
            cur.execute(
                "INSERT INTO schema_version (version) VALUES (%s) "
                "ON CONFLICT (version) DO NOTHING;",
                (SCHEMA_VERSION,),
            )
        logger.debug("Database schema verified (version %d)", SCHEMA_VERSION)

    def _ensure_schema(self) -> None:
        """Create/verify the current schema in place.

        Initialisation always enforces the current schema version.
        """
        self.init_schema(drop=False)

    # -- Upload ----------------------------------------------------------------

    #: Columns whose Python value is a dict and must be adapted before it
    #: reaches psycopg2. There is no `register_adapter(dict, Json)` anywhere in
    #: this repo, and `psycopg2.extensions.adapt({'a': 1})` raises
    #: `ProgrammingError: can't adapt type 'dict'`. `ResultRecord.extras`
    #: defaults to `{}` and `to_upload_dict` drops only `None`, so without this
    #: EVERY upload fails, not just the ones carrying extras. (The schema-2 code
    #: did `json.dumps(metadata) if metadata else None` at database.py:459.)
    _JSONB_COLUMNS = ('extras',)

    def upload_result(self, **fields) -> Optional[int]:
        """Insert one row, validated through :class:`ResultRecord` first."""
        return self.upload_record(ResultRecord(**fields))

    def upload_record(self, record: Any, uploader: Optional[str] = None) -> Optional[int]:
        """Upload a :class:`~src.data.models.ResultRecord` to the database.

        This is the **preferred upload method** — it guarantees all
        statistics have been computed canonically.

        Args:
            record: A :class:`ResultRecord` instance.
            uploader: Override the uploader field (if not set on the record).

        Returns:
            Database row ``id`` of the inserted record, or ``None`` if a row
            with the same ``time_series_hash`` already exists. The `simulations`
            table's ``time_series_hash`` column is ``UNIQUE``, so this is a
            genuine, expected outcome (the same result uploaded twice, e.g. a
            retried `db upload`) -- ``ON CONFLICT ... DO NOTHING`` turns it into
            a `None` return, a clear counted outcome for callers, rather than
            a raw driver ``IntegrityError`` reaching them as a crash.
        """
        from psycopg2.extras import Json  # pylint: disable=import-outside-toplevel

        data = record.to_upload_dict()
        if uploader:
            data['uploader'] = uploader
        columns = [c for c in _COLUMN_NAMES if c in data]
        values = [Json(data[c]) if c in self._JSONB_COLUMNS else data[c]
                  for c in columns]
        sql = (f"INSERT INTO simulations ({', '.join(columns)}) "
               f"VALUES ({', '.join(['%s'] * len(columns))}) "
               "ON CONFLICT (time_series_hash) DO NOTHING RETURNING id;")
        with self._cursor(commit=True) as cur:
            cur.execute(sql, values)
            row = cur.fetchone()

        if row is None:
            logger.info(
                "Skipped duplicate upload for %s (time_series_hash already present)",
                data.get('material'),
            )
            return None

        row_id = int(row[0])
        logger.info(
            "Uploaded result for %s (id=%d, COF=%s, status=%s)",
            data.get('material'), row_id, data.get('mean_cof'),
            data.get('status'),
        )
        return row_id

    # -- Query -----------------------------------------------------------------

    #: filter kwarg -> SQL fragment. Equality first, then the ranges; the order
    #: here is the order of the `params` list the test asserts on.
    _EQ_CLAUSES = {
        'material': 'material = %s',
        # ALL, not ANY: a row matches when its `materials` contains every
        # requested one -- the same meaning as `Friction2DDB.query` and the
        # CLI's repeatable `--material`.
        'materials': 'materials @> %s',
        'family': 'family = %s',
        'model': 'model = %s',
        'set_label': 'set_label = %s',
        'potential_type': 'potential_type = %s',
        'layers': 'layers = %s',
        'uploader': 'uploader = %s',
        'status': 'status = %s',
        'is_complete': 'is_complete = %s',
    }
    #: range kwarg -> column. A `(lo, hi)` span becomes `BETWEEN`; a span with
    #: one bound `None` is open-ended (`>=` or `<=`), never `BETWEEN NULL AND x`,
    #: which matches no row at all.
    _RANGE_COLUMNS = {
        'force_range': 'force_nN',
        'pressure_range': 'pressure_gpa',
        'temperature_range': 'temperature',
        'angle_range': 'scan_angle',
    }

    @staticmethod
    def _range_clause(column: str, span) -> Tuple[Optional[str], List[Any]]:
        lo, hi = span
        if lo is not None and hi is not None:
            return f'{column} BETWEEN %s AND %s', [lo, hi]
        if lo is not None:
            return f'{column} >= %s', [lo]
        if hi is not None:
            return f'{column} <= %s', [hi]
        return None, []

    def query(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, *, material: Optional[str] = None, materials: Optional[List[str]] = None,
        family: Optional[str] = None, model: Optional[str] = None,
        set_label: Optional[str] = None, potential_type: Optional[str] = None,
        layers: Optional[int] = None, uploader: Optional[str] = None,
        force_range: Optional[tuple] = None, pressure_range: Optional[tuple] = None,
        temperature_range: Optional[tuple] = None, angle_range: Optional[tuple] = None,
        status: Optional[str] = None, is_complete: Optional[bool] = None,
        limit: Optional[int] = None, order_by: str = 'uploaded_at DESC',
    ) -> 'pandas.DataFrame':
        """Query the central database. Filters are the schema-3 columns.

        `is_complete=True` is how a caller excludes rows whose COF could not be
        computed. Those rows carry `mean_cof IS NULL`, never 0.0, so
        `mean_cof BETWEEN ...` also drops them -- but the flag says WHY, and
        without it in the filter list `is_complete` would be stored and never
        consulted anywhere.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        equalities = {
            'material': material, 'materials': materials, 'family': family,
            'model': model, 'set_label': set_label,
            'potential_type': potential_type, 'layers': layers,
            'uploader': uploader, 'status': status, 'is_complete': is_complete,
        }
        ranges = {
            'force_range': force_range, 'pressure_range': pressure_range,
            'temperature_range': temperature_range, 'angle_range': angle_range,
        }

        conditions: List[str] = []
        params: List[Any] = []
        for name, value in equalities.items():
            if value is not None:
                conditions.append(self._EQ_CLAUSES[name])
                params.append(list(value) if name == 'materials' else value)
        for name, span in ranges.items():
            if span is None:
                continue
            clause, values = self._range_clause(self._RANGE_COLUMNS[name], span)
            if clause is not None:
                conditions.append(clause)
                params.extend(values)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""

        # Sanitise order_by: allow only column names + ASC/DESC
        for token in order_by.replace(',', ' ').split():
            if token not in _ALLOWED_ORDER_COLS:
                raise ValueError(f"Invalid order_by token: {token!r}")

        sql = f"SELECT * FROM simulations {where} ORDER BY {order_by} {limit_clause};"

        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=_COLUMN_NAMES)
        return df

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics about the shared dataset.

        Returns:
            Dict with ``total_rows``, ``by_material``, ``by_type``,
            ``cof_global_mean``.
        """
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM simulations;")
            total: int = cur.fetchone()[0]

            cur.execute(
                "SELECT material, COUNT(*) FROM simulations "
                "GROUP BY material ORDER BY COUNT(*) DESC;"
            )
            by_material = dict(cur.fetchall())

            cur.execute(
                "SELECT family, COUNT(*) FROM simulations GROUP BY family;"
            )
            by_type = dict(cur.fetchall())

            cur.execute("SELECT AVG(mean_cof) FROM simulations WHERE mean_cof IS NOT NULL;")
            row = cur.fetchone()
            cof_mean = float(row[0]) if row and row[0] is not None else None

        return {
            'total_rows': total,
            'by_material': by_material,
            'by_type': by_type,
            'cof_global_mean': cof_mean,
        }

    def delete_own_results(self, uploader: str) -> int:
        """Delete all rows belonging to a specific uploader.

        Args:
            uploader: The uploader identifier whose rows should be removed.

        Returns:
            Number of rows deleted.
        """
        sql = "DELETE FROM simulations WHERE uploader = %s;"
        with self._cursor(commit=True) as cur:
            cur.execute(sql, (uploader,))
            count: int = cur.rowcount

        logger.info("Deleted %d rows for uploader '%s'", count, uploader)
        return count

    # -- Staging pipeline ------------------------------------------------------

    def set_status(self, row_id: int, status: str, uploader: Optional[str] = None) -> bool:
        """Update the status of a result row.

        If *uploader* is given, only the row owner can change its status
        (except for curator actions ``publish`` / ``reject``).

        Args:
            row_id: Database row ID.
            status: New status string.
            uploader: The requesting user (for ownership check).

        Returns:
            ``True`` if a row was updated.
        """
        if uploader:
            sql = "UPDATE simulations SET status = %s WHERE id = %s AND uploader = %s;"
            params: tuple = (status, row_id, uploader)
        else:
            sql = "UPDATE simulations SET status = %s WHERE id = %s;"
            params = (status, row_id)

        with self._cursor(commit=True) as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def validate_staged(self, row_id: int) -> 'ValidationResult':  # type: ignore[name-defined]
        """Run automated validation on a staged result and update status.

        Args:
            row_id: Database row ID (must have ``status='staged'``).

        Returns:
            :class:`~src.data.validation.ValidationResult`.
        """
        from .validation import validate_record  # pylint: disable=import-outside-toplevel

        with self._cursor() as cur:
            cur.execute("SELECT * FROM simulations WHERE id = %s;", (row_id,))
            row = cur.fetchone()

        if row is None:
            from .validation import ValidationResult  # pylint: disable=import-outside-toplevel
            vr = ValidationResult()
            vr.add_error(f"Row {row_id} not found")
            return vr

        row_dict = dict(zip(_COLUMN_NAMES, row))
        record = ResultRecord.from_db_row(row_dict)

        # Gather existing hashes for duplicate check
        with self._cursor() as cur:
            cur.execute(
                "SELECT time_series_hash FROM simulations "
                "WHERE time_series_hash IS NOT NULL AND id != %s;",
                (row_id,),
            )
            existing_hashes = [r[0] for r in cur.fetchall()]

        vr = validate_record(record, existing_hashes)

        new_status = 'validated' if vr.is_valid else 'rejected'
        self.set_status(row_id, new_status)
        logger.info("Row %d validation: %s", row_id, new_status)
        return vr

    def publish(self, row_id: int) -> bool:
        """Promote a validated result to published (curator action)."""
        return self.set_status(row_id, 'published')

    def reject(self, row_id: int, reason: Optional[str] = None) -> bool:
        """Reject a staged/validated result (curator action)."""
        if reason:
            with self._cursor(commit=True) as cur:
                cur.execute(
                    "UPDATE simulations SET notes = COALESCE(notes, '') || %s "
                    "WHERE id = %s;",
                    (f"\n[REJECTED] {reason}", row_id),
                )
        return self.set_status(row_id, 'rejected')

    def get_existing_hashes(self) -> List[str]:
        """Return all non-null time_series_hash values in the database."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT time_series_hash FROM simulations "
                "WHERE time_series_hash IS NOT NULL;"
            )
            return [r[0] for r in cur.fetchall()]

    # -- API key management ----------------------------------------------------

    def create_api_key(self, user_name: str) -> str:
        """Generate a new API key for a user.

        Args:
            user_name: Human-readable identifier for the key holder.

        Returns:
            The raw API key string (show to user once, then discard).
        """
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO api_keys (key_hash, user_name) VALUES (%s, %s);",
                (key_hash, user_name),
            )

        logger.info("Created API key for user '%s'", user_name)
        return raw_key

    def verify_api_key(self, raw_key: str) -> Optional[str]:
        """Verify an API key and return the associated user name.

        Args:
            raw_key: The raw API key string.

        Returns:
            User name if valid, ``None`` otherwise.
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._cursor() as cur:
            cur.execute(
                "SELECT user_name FROM api_keys "
                "WHERE key_hash = %s AND is_active = true;",
                (key_hash,),
            )
            row = cur.fetchone()

        return row[0] if row else None

    def revoke_api_key(self, raw_key: str) -> bool:
        """Deactivate an API key."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._cursor(commit=True) as cur:
            cur.execute(
                "UPDATE api_keys SET is_active = false WHERE key_hash = %s;",
                (key_hash,),
            )
            return cur.rowcount > 0
