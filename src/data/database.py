"""Online PostgreSQL database interface for FrictionSim2D.

Provides :class:`FrictionDB`, a lightweight wrapper around ``psycopg2`` for
uploading simulation results to a shared PostgreSQL instance and querying
results contributed by all users.

The database uses a **federated catalog** model:

- **Central DB** stores queryable summary statistics (~1 KB per simulation).
- **Full time-series data** stays with the contributor (local AiiDA / files).
- Published results may link to an archive URL (Zenodo, Figshare, etc.)
  for on-demand download of full data.

Results go through a **staging pipeline**::

    local → staged → validated → published
                   ↘ rejected

Usage
-----
Configure the connection via environment variables, keyword arguments, or
``settings.yaml`` profiles::

    export FRICTION_DB_HOST=db.example.com
    export FRICTION_DB_PORT=5432
    export FRICTION_DB_NAME=frictionsim2d
    export FRICTION_DB_USER=myuser
    export FRICTION_DB_PASSWORD=secret

    from src.data.database import FrictionDB
    from src.data.models import ResultRecord

    db = FrictionDB()

    record = ResultRecord(
        material="h-MoS2", simulation_type="afm",
        layers=1, force_nN=10.0, temperature=300.0, mean_cof=0.012,
    )
    db.upload_record(record)

    df = db.query(material="h-MoS2", layers=1)
    print(df)

Connection profiles
-------------------
:func:`db_from_profile` creates a :class:`FrictionDB` from the active profile
in ``settings.yaml`` (``database.local`` or ``database.central``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS simulations (
    id                  SERIAL PRIMARY KEY,
    uploaded_at         TIMESTAMPTZ DEFAULT now(),
    uploader            TEXT,

    -- Material & geometry
    material            TEXT,
    simulation_type     TEXT,
    layers              INTEGER,
    size_x              REAL,
    size_y              REAL,
    stack_type          TEXT,

    -- Loading conditions
    force_nN            REAL,
    pressure_gpa        REAL,
    scan_angle          REAL,
    scan_speed          REAL,
    temperature         REAL,

    -- Tip / substrate (AFM-specific)
    tip_material        TEXT,
    tip_radius          REAL,
    substrate_material  TEXT,
    substrate_amorphous BOOLEAN,

    -- Potential
    potential_type      TEXT,

    -- Results (from compute_friction_stats)
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

    -- Data provenance
    ntimesteps          INTEGER,
    time_series_hash    TEXT UNIQUE,
    is_complete         BOOLEAN DEFAULT true,

    -- Staging pipeline
    status              TEXT DEFAULT 'staged',

    -- Free-form
    notes               TEXT,
    metadata            JSONB,

    -- Federated file storage
    data_url            TEXT
);
"""

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

_COLUMN_NAMES = [
    'id', 'uploaded_at', 'uploader',
    'material', 'simulation_type', 'layers', 'size_x', 'size_y', 'stack_type',
    'force_nN', 'pressure_gpa', 'scan_angle', 'scan_speed', 'temperature',
    'tip_material', 'tip_radius', 'substrate_material', 'substrate_amorphous',
    'potential_type',
    'mean_cof', 'std_cof', 'mean_lf', 'std_lf', 'mean_nf', 'std_nf',
    'mean_lfx', 'std_lfx', 'mean_lfy', 'std_lfy',
    'ntimesteps', 'time_series_hash', 'is_complete',
    'status', 'notes', 'metadata', 'data_url',
]

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _get_connection_params(
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


def db_from_profile(profile: Optional[str] = None) -> 'FrictionDB':
    """Create a :class:`FrictionDB` from a ``settings.yaml`` database profile.

    Args:
        profile: Profile name (``'local'`` or ``'central'``).  If ``None``,
            uses the ``database.active_profile`` value from settings.

    Returns:
        Configured :class:`FrictionDB` instance.
    """
    from src.core.config import load_settings  # noqa: PLC0415

    settings = load_settings()
    db_cfg = settings.database
    name = profile or db_cfg.active_profile

    if name == 'local':
        p = db_cfg.local
    elif name == 'central':
        p = db_cfg.central
    else:
        raise ValueError(f"Unknown database profile: {name!r}. Use 'local' or 'central'.")

    return FrictionDB(
        host=p.host or None,
        port=p.port,
        dbname=p.dbname,
        user=p.user or None,
        password=p.password or None,
    )


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

# Registry of migration modules, keyed by (from_version, to_version).
_MIGRATIONS = {
    (1, 2): 'src.data.migrations.v001_to_v002',
}


def get_current_schema_version(cursor) -> int:
    """Read the current schema version from the database.

    Args:
        cursor: An open database cursor.

    Returns:
        Current schema version number.  Returns 1 if the ``schema_version``
        table does not exist (legacy databases created before versioning).
    """
    try:
        cursor.execute(
            "SELECT MAX(version) FROM schema_version;"
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else 1
    except Exception:  # table doesn't exist yet → legacy v1
        # Must reset the transaction after the failed query
        cursor.execute("ROLLBACK;")
        return 1


def apply_migrations(cursor, *, target: int = SCHEMA_VERSION) -> List[int]:
    """Apply pending migrations up to *target* version.

    Args:
        cursor: An open database cursor (caller must commit afterwards).
        target: Version to migrate to (default: ``SCHEMA_VERSION``).

    Returns:
        List of version numbers that were applied.
    """
    import importlib  # noqa: PLC0415

    current = get_current_schema_version(cursor)
    applied: List[int] = []

    while current < target:
        key = (current, current + 1)
        module_path = _MIGRATIONS.get(key)
        if module_path is None:
            raise RuntimeError(
                f"No migration registered for {current} → {current + 1}"
            )

        mod = importlib.import_module(module_path)
        for stmt in mod.UP:
            cursor.execute(stmt)

        applied.append(current + 1)
        logger.info("Applied migration v%d → v%d", current, current + 1)
        current += 1

    return applied


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

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        auto_create: bool = True,
    ):
        try:
            import psycopg2  # pylint: disable=import-outside-toplevel
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

    def _ensure_schema(self) -> None:
        """Create tables and apply pending migrations if needed.

        For a fresh database this creates the full v2 schema.  For an
        existing v1 database it applies migrations incrementally.
        """
        with self._cursor(commit=True) as cur:
            cur.execute(_CREATE_SCHEMA_VERSION_SQL)
            cur.execute(_CREATE_TABLE_SQL)
            cur.execute(_CREATE_API_KEYS_SQL)

            # Apply any pending migrations
            applied = apply_migrations(cur)
            if applied:
                logger.info(
                    "Migrated database to v%d (applied: %s)",
                    applied[-1], applied,
                )

            # Record current version
            cur.execute(
                "INSERT INTO schema_version (version) VALUES (%s) "
                "ON CONFLICT (version) DO NOTHING;",
                (SCHEMA_VERSION,),
            )
        logger.debug("Database schema verified (version %d)", SCHEMA_VERSION)

    # -- Upload ----------------------------------------------------------------

    def upload_result(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        material: str,
        simulation_type: str = 'afm',
        layers: Optional[int] = None,
        size_x: Optional[float] = None,
        size_y: Optional[float] = None,
        stack_type: Optional[str] = None,
        force_nN: Optional[float] = None,
        pressure_gpa: Optional[float] = None,
        scan_angle: Optional[float] = None,
        scan_speed: Optional[float] = None,
        temperature: Optional[float] = None,
        tip_material: Optional[str] = None,
        tip_radius: Optional[float] = None,
        substrate_material: Optional[str] = None,
        substrate_amorphous: Optional[bool] = None,
        potential_type: Optional[str] = None,
        mean_cof: Optional[float] = None,
        std_cof: Optional[float] = None,
        mean_lf: Optional[float] = None,
        std_lf: Optional[float] = None,
        mean_nf: Optional[float] = None,
        std_nf: Optional[float] = None,
        mean_lfx: Optional[float] = None,
        std_lfx: Optional[float] = None,
        mean_lfy: Optional[float] = None,
        std_lfy: Optional[float] = None,
        ntimesteps: Optional[int] = None,
        time_series_hash: Optional[str] = None,
        is_complete: bool = True,
        status: str = 'staged',
        uploader: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        data_url: Optional[str] = None,
    ) -> int:
        """Upload a single simulation result to the database.

        Prefer :meth:`upload_record` which accepts a :class:`ResultRecord`
        and guarantees consistent field naming.

        Returns:
            Database row ``id`` of the inserted record.
        """
        sql = """
            INSERT INTO simulations (
                uploader, material, simulation_type, layers,
                size_x, size_y, stack_type,
                force_nN, pressure_gpa, scan_angle, scan_speed, temperature,
                tip_material, tip_radius, substrate_material, substrate_amorphous,
                potential_type,
                mean_cof, std_cof, mean_lf, std_lf, mean_nf, std_nf,
                mean_lfx, std_lfx, mean_lfy, std_lfy,
                ntimesteps, time_series_hash, is_complete,
                status, notes, metadata, data_url
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING id;
        """
        values = (
            uploader, material, simulation_type, layers,
            size_x, size_y, stack_type,
            force_nN, pressure_gpa, scan_angle, scan_speed, temperature,
            tip_material, tip_radius, substrate_material, substrate_amorphous,
            potential_type,
            mean_cof, std_cof, mean_lf, std_lf, mean_nf, std_nf,
            mean_lfx, std_lfx, mean_lfy, std_lfy,
            ntimesteps, time_series_hash, is_complete,
            status, notes, json.dumps(metadata) if metadata else None, data_url,
        )
        with self._cursor(commit=True) as cur:
            cur.execute(sql, values)
            row_id: int = cur.fetchone()[0]

        logger.info(
            "Uploaded result for %s (id=%d, COF=%.4f, status=%s)",
            material, row_id, mean_cof or 0.0, status,
        )
        return row_id

    def upload_record(self, record: Any, uploader: Optional[str] = None) -> int:
        """Upload a :class:`~src.data.models.ResultRecord` to the database.

        This is the **preferred upload method** — it guarantees all
        statistics have been computed canonically.

        Args:
            record: A :class:`ResultRecord` instance.
            uploader: Override the uploader field (if not set on the record).

        Returns:
            Database row ``id`` of the inserted record.
        """
        d = record.to_upload_dict()
        if uploader:
            d['uploader'] = uploader
        return self.upload_result(**d)

    def upload_from_aiida(self, result_node: Any, uploader: Optional[str] = None) -> int:
        """Upload a ``FrictionResultsData`` AiiDA node to the shared database.

        Uses :func:`~src.data.models.compute_friction_stats` for canonical
        COF calculation.

        Args:
            result_node: A stored ``FrictionResultsData`` AiiDA node.
            uploader: Optional contributor identifier.

        Returns:
            Database row ``id`` of the inserted record.
        """
        from src.data.models import compute_friction_stats  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        ts = result_node.time_series
        nf = np.asarray(ts.get('nf', []))
        lfx = np.asarray(ts.get('lfx', []))
        lfy = np.asarray(ts.get('lfy', []))

        stats = compute_friction_stats(nf, lfx, lfy) if nf.size > 0 else {}

        return self.upload_result(
            material=result_node.material,
            simulation_type=getattr(result_node, 'simulation_type', 'afm'),
            layers=getattr(result_node, 'layers', None),
            force_nN=getattr(result_node, 'force', None),
            scan_angle=getattr(result_node, 'angle', None),
            scan_speed=getattr(result_node, 'speed', None),
            temperature=getattr(result_node, 'temperature', None),
            uploader=uploader,
            metadata={'aiida_uuid': str(result_node.uuid)},
            **stats,
        )

    # -- Query -----------------------------------------------------------------

    def query(  # pylint: disable=too-many-arguments,too-many-branches,too-many-locals
        self,
        material: Optional[str] = None,
        simulation_type: Optional[str] = None,
        layers: Optional[int] = None,
        force_range: Optional[tuple] = None,
        temperature_range: Optional[tuple] = None,
        angle_range: Optional[tuple] = None,
        uploader: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: str = 'uploaded_at DESC',
    ) -> 'pandas.DataFrame':
        """Query the shared database with optional filters.

        Args:
            material: Filter by material name (exact match).
            simulation_type: Filter by type (``'afm'`` or ``'sheetonsheet'``).
            layers: Filter by number of layers.
            force_range: ``(min_nN, max_nN)`` tuple.
            temperature_range: ``(min_K, max_K)`` tuple.
            angle_range: ``(min_deg, max_deg)`` tuple.
            uploader: Filter by uploader identifier.
            limit: Maximum rows to return.
            order_by: SQL ``ORDER BY`` clause (default: newest first).

        Returns:
            pandas ``DataFrame`` with one row per matching simulation.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        conditions: List[str] = []
        params: List[Any] = []

        if material is not None:
            conditions.append("material = %s")
            params.append(material)
        if simulation_type is not None:
            conditions.append("simulation_type = %s")
            params.append(simulation_type)
        if layers is not None:
            conditions.append("layers = %s")
            params.append(layers)
        if uploader is not None:
            conditions.append("uploader = %s")
            params.append(uploader)
        if force_range is not None:
            conditions.append("force_nN BETWEEN %s AND %s")
            params.extend(force_range)
        if temperature_range is not None:
            conditions.append("temperature BETWEEN %s AND %s")
            params.extend(temperature_range)
        if angle_range is not None:
            conditions.append("scan_angle BETWEEN %s AND %s")
            params.extend(angle_range)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""

        # Sanitise order_by: allow only column names + ASC/DESC
        _ALLOWED_ORDER_COLS = set(_COLUMN_NAMES) | {'ASC', 'DESC', 'asc', 'desc'}
        order_tokens = order_by.replace(',', ' ').split()
        for token in order_tokens:
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
                "SELECT material, COUNT(*) FROM simulations GROUP BY material ORDER BY COUNT(*) DESC;"
            )
            by_material = dict(cur.fetchall())

            cur.execute(
                "SELECT simulation_type, COUNT(*) FROM simulations GROUP BY simulation_type;"
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
        import pandas as pd  # pylint: disable=import-outside-toplevel
        from src.data.models import ResultRecord  # noqa: PLC0415
        from src.data.validation import validate_record  # noqa: PLC0415

        with self._cursor() as cur:
            cur.execute("SELECT * FROM simulations WHERE id = %s;", (row_id,))
            row = cur.fetchone()

        if row is None:
            from src.data.validation import ValidationResult  # noqa: PLC0415
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
