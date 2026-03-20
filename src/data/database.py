"""Online PostgreSQL database interface for FrictionSim2D.

Provides :class:`FrictionDB`, a lightweight wrapper around ``psycopg2`` for
uploading simulation results to a shared PostgreSQL instance and querying
results contributed by all users.

The schema is intentionally simple and schema-version-aware:

.. code-block:: sql

    CREATE TABLE simulations (
        id              SERIAL PRIMARY KEY,
        uploaded_at     TIMESTAMPTZ DEFAULT now(),
        uploader        TEXT,
        material        TEXT,
        simulation_type TEXT,
        layers          INTEGER,
        force_nN        REAL,
        pressure_gpa    REAL,
        scan_angle      REAL,
        scan_speed      REAL,
        temperature     REAL,
        tip_material    TEXT,
        tip_radius      REAL,
        mean_cof        REAL,
        std_cof         REAL,
        mean_lf         REAL,
        mean_nf         REAL,
        notes           TEXT,
        metadata        JSONB
    );

Usage
-----
Configure the connection via environment variables or keyword arguments::

    export FRICTION_DB_HOST=db.example.com
    export FRICTION_DB_PORT=5432
    export FRICTION_DB_NAME=frictionsim2d
    export FRICTION_DB_USER=myuser
    export FRICTION_DB_PASSWORD=secret

    from src.data.database import FrictionDB

    db = FrictionDB()
    db.upload_result(
        material="h-MoS2",
        simulation_type="afm",
        layers=1,
        force_nN=10.0,
        scan_angle=0,
        temperature=300.0,
        mean_cof=0.012,
    )

    df = db.query(material="h-MoS2", layers=1)
    print(df)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS simulations (
    id              SERIAL PRIMARY KEY,
    uploaded_at     TIMESTAMPTZ DEFAULT now(),
    uploader        TEXT,
    material        TEXT,
    simulation_type TEXT,
    layers          INTEGER,
    force_nN        REAL,
    pressure_gpa    REAL,
    scan_angle      REAL,
    scan_speed      REAL,
    temperature     REAL,
    tip_material    TEXT,
    tip_radius      REAL,
    mean_cof        REAL,
    std_cof         REAL,
    mean_lf         REAL,
    mean_nf         REAL,
    notes           TEXT,
    metadata        JSONB
);
"""

_COLUMN_NAMES = [
    'id', 'uploaded_at', 'uploader', 'material', 'simulation_type',
    'layers', 'force_nN', 'pressure_gpa', 'scan_angle', 'scan_speed',
    'temperature', 'tip_material', 'tip_radius', 'mean_cof', 'std_cof',
    'mean_lf', 'mean_nf', 'notes', 'metadata',
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
        'dbname': dbname or os.environ.get('FRICTION_DB_NAME', 'frictionsim2d'),
        'user': user or os.environ.get('FRICTION_DB_USER', ''),
        'password': password or os.environ.get('FRICTION_DB_PASSWORD', ''),
    }


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
        """Create the ``simulations`` table if it does not already exist."""
        with self._cursor(commit=True) as cur:
            cur.execute(_CREATE_TABLE_SQL)
        logger.debug("Database schema verified")

    # -- Upload ----------------------------------------------------------------

    def upload_result(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        material: str,
        simulation_type: str = 'afm',
        layers: Optional[int] = None,
        force_nN: Optional[float] = None,
        pressure_gpa: Optional[float] = None,
        scan_angle: Optional[float] = None,
        scan_speed: Optional[float] = None,
        temperature: Optional[float] = None,
        tip_material: Optional[str] = None,
        tip_radius: Optional[float] = None,
        mean_cof: Optional[float] = None,
        std_cof: Optional[float] = None,
        mean_lf: Optional[float] = None,
        mean_nf: Optional[float] = None,
        uploader: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Upload a single simulation result to the shared database.

        Args:
            material: 2D material name (e.g. ``'h-MoS2'``).
            simulation_type: ``'afm'`` or ``'sheetonsheet'``.
            layers: Number of 2D material layers.
            force_nN: Applied normal force (nN).
            pressure_gpa: Applied pressure (GPa). Alternative to *force_nN*.
            scan_angle: Scan angle (degrees).
            scan_speed: Scan speed (m/s).
            temperature: Simulation temperature (K).
            tip_material: Tip material (AFM only).
            tip_radius: Tip radius in Ångströms (AFM only).
            mean_cof: Mean coefficient of friction.
            std_cof: Standard deviation of COF.
            mean_lf: Mean lateral force (nN).
            mean_nf: Mean normal force (nN).
            uploader: Optional identifier for the contributor.
            notes: Free-text notes.
            metadata: Arbitrary extra JSON metadata.

        Returns:
            Database row ``id`` of the inserted record.
        """
        sql = """
            INSERT INTO simulations (
                uploader, material, simulation_type, layers,
                force_nN, pressure_gpa, scan_angle, scan_speed, temperature,
                tip_material, tip_radius,
                mean_cof, std_cof, mean_lf, mean_nf,
                notes, metadata
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
            RETURNING id;
        """
        values = (
            uploader, material, simulation_type, layers,
            force_nN, pressure_gpa, scan_angle, scan_speed, temperature,
            tip_material, tip_radius,
            mean_cof, std_cof, mean_lf, mean_nf,
            notes, json.dumps(metadata) if metadata else None,
        )
        with self._cursor(commit=True) as cur:
            cur.execute(sql, values)
            row_id: int = cur.fetchone()[0]

        logger.info(
            "Uploaded result for %s (id=%d, COF=%.4f)",
            material, row_id, mean_cof or 0.0,
        )
        return row_id

    def upload_from_aiida(self, result_node: Any, uploader: Optional[str] = None) -> int:
        """Upload a ``FrictionResultsData`` AiiDA node to the shared database.

        Convenience wrapper that extracts all relevant fields from an
        AiiDA ``FrictionResultsData`` node and passes them to
        :meth:`upload_result`.

        Args:
            result_node: A stored ``FrictionResultsData`` AiiDA node.
            uploader: Optional contributor identifier.

        Returns:
            Database row ``id`` of the inserted record.
        """
        stats = result_node.get_summary_statistics()

        return self.upload_result(
            material=result_node.material,
            simulation_type=getattr(result_node, 'simulation_type', 'afm'),
            layers=getattr(result_node, 'layers', None),
            force_nN=getattr(result_node, 'force', None),
            scan_angle=getattr(result_node, 'angle', None),
            scan_speed=getattr(result_node, 'speed', None),
            temperature=getattr(result_node, 'temperature', None),
            mean_cof=result_node.mean_cof,
            std_cof=stats.get('cof', {}).get('std'),
            mean_lf=stats.get('lfx', {}).get('mean'),
            mean_nf=stats.get('nf', {}).get('mean'),
            uploader=uploader,
            metadata={'aiida_uuid': str(result_node.uuid)},
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
